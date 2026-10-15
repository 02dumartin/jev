"""⑤ Jev (TypeSafe API) on track summaries + the small VLM's description (PLAN.md v7 §4.2). Text only.

One Noul question per track over a shared state (the tracks of one video, in chunks):
`<uid>|ok`  "Is this track a real <majority class>?" -> p_ok.
Raw requests and responses are kept; a response is reused only for an identical request body.
"""

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from jevgate.judges.base import Budget

RETRY_STATUS = {408, 429, 499, 529}  # plus every 5xx
NOT_ANIMAL = "not_an_animal"


class JevError(RuntimeError):
    """Jev returned an error or an answer that fails validation; nothing is guessed."""


def validate_noul(answer) -> float:
    """The Noul probability, or JevError if it is missing or outside [0, 1]."""
    try:
        value = answer["noul"]
    except (KeyError, TypeError):
        raise JevError(f"missing noul in answer: {answer!r}") from None
    if type(value) not in (int, float) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise JevError(f"noul outside [0, 1]: {value!r}")
    return float(value)


def validate_choice(answer, options: list[str]) -> dict[str, float]:
    """Choice probabilities over `options` (renormalised), or JevError if keys/values/sum are wrong."""
    try:
        probs = answer["probabilities"]
    except (KeyError, TypeError):
        raise JevError(f"missing probabilities in choice answer: {answer!r}") from None
    if set(probs) != set(options):
        raise JevError(f"choice keys {sorted(probs)} != {sorted(options)}")
    vals = list(probs.values())
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in vals):
        raise JevError(f"choice probability outside [0, 1]: {probs}")
    total = sum(vals)
    if abs(total - 1) > 0.02:
        raise JevError(f"choice probabilities sum to {total}")
    return {c: float(probs[c]) / total for c in options}


def detector_reliability(cfg: dict, repo: Path) -> dict | None:
    """YOLO's per-class precision/recall on the dev (val) videos, when enabled in the config."""
    if not cfg["jev"].get("include_detector_reliability"):
        return None
    evals = json.loads((repo / cfg["jev"]["detector_reliability_file"]).read_text())
    val = next(e for e in evals if e["split"] == "val")
    return {c: {"precision": round(v["precision"], 2), "recall": round(v["recall"], 2)}
            for c, v in val["per_class"].items()}


def question_key(uid: str) -> str:
    return f"{uid}|ok"


def build_request(cfg: dict, tracks: list[dict], descriptions: dict[str, str] | None = None,
                  reliability: dict | None = None) -> dict:
    """Request body for tracks of one video. Track ids and video names are not sent (T1, T2, ... only)."""
    p = cfg["prompts"]
    names = {t["track_uid"]: f"T{k + 1}" for k, t in enumerate(tracks)}
    view = {}
    for t in tracks:
        seen = {"visual_description": descriptions[t["track_uid"]]} if descriptions is not None else {}
        view[names[t["track_uid"]]] = seen | t["evidence"]
    state = {"context": p["context"],
             "class_definitions": {**cfg["classes"], NOT_ANIMAL: cfg["not_an_animal"]},
             "tracks": view}
    if descriptions is not None:
        state["visual_description_source"] = ("a small vision-language model looked at the track's best frame and "
                                              "described the box without being told the detector's class")
    if reliability is not None:
        state["detector_reliability_on_other_videos"] = reliability
    questions = {}
    for t in tracks:
        cls = t["majority_class"]
        questions[question_key(t["track_uid"])] = {
            "type": "noul",
            "instructions": {"question": p["question"].format(cls=cls),
                             "target": f"Judge only track `{names[t['track_uid']]}` from the state."},
            "criteria": {"true": p["true_means"].format(cls=cls), "false": p["false_means"]}}
    return {"model": cfg["jev"]["model"], "state": state, "questions": questions}


def body_key(body: dict) -> str:
    """Canonical request text: a stored response is reused only for an identical request."""
    return json.dumps(body, sort_keys=True, ensure_ascii=False)


class JevClient:
    """HTTP client with retry on rate limits/overload/5xx. The API key is never logged."""

    def __init__(self, cfg: dict, budget: Budget):
        key = os.environ.get(cfg["jev"]["api_key_env"])
        if not key:
            raise JevError(f"{cfg['jev']['api_key_env']} is not set (.env)")
        self.endpoint, self.max_retries, self.budget = cfg["jev"]["endpoint"], cfg["jev"]["max_retries"], budget
        self.client = httpx.Client(http2=True, timeout=cfg["jev"]["timeout_s"],
                                   headers={"Authorization": f"Bearer {key}"})

    def post(self, body: dict) -> tuple[dict, float]:
        """Send one request; return (response json, latency ms of the successful attempt)."""
        self.budget.reserve_jev_request()
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                response = self.client.post(self.endpoint, json=body)
            except httpx.HTTPError as exc:
                if attempt + 1 < self.max_retries:
                    time.sleep(0.5 * 2**attempt)
                    continue
                raise JevError(f"connection failed: {type(exc).__name__}") from None
            latency = (time.perf_counter() - started) * 1000
            retryable = response.status_code in RETRY_STATUS or response.status_code >= 500
            if retryable and attempt + 1 < self.max_retries:
                time.sleep(min(30.0, 1.0 * 2**attempt))
                continue
            if response.is_error:
                raise JevError(f"HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
            usage = data.get("usage", {})
            self.budget.add_jev_tokens(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
            return data, latency
        raise JevError("retries exhausted")


class JevJudge:
    """Asks Jev about every track; writes requests/responses as they happen."""

    def __init__(self, cfg: dict, repo: Path, budget: Budget, requests_file: Path, responses_file: Path,
                 resume: list[Path], descriptions: dict[str, str] | None = None, log=None):
        self.cfg, self.log, self.descriptions = cfg, log, descriptions
        self.reliability = detector_reliability(cfg, repo)
        self.client = JevClient(cfg, budget)
        self.requests_file, self.responses_file = requests_file, responses_file
        self.lock = threading.Lock()
        self.stored: dict[str, dict] = {}
        for path in resume:
            for line in Path(path).open():
                if line.strip():
                    rec = json.loads(line)
                    self.stored[body_key(rec["request"])] = rec

    def chunks(self, tracks: list[dict]) -> list[list[dict]]:
        """Tracks of the same video, max_tracks_per_request at a time."""
        n, by_video = self.cfg["jev"]["max_tracks_per_request"], {}
        for t in tracks:
            by_video.setdefault(t["video"], []).append(t)
        return [ts[i:i + n] for ts in by_video.values() for i in range(0, len(ts), n)]

    def _run_chunk(self, chunk: list[dict]) -> list[dict]:
        body = build_request(self.cfg, chunk, self.descriptions, self.reliability)
        stored = self.stored.get(body_key(body))
        data, latency = (stored["response"], stored["latency_ms"]) if stored else self.client.post(body)
        answers, usage, n = data.get("answers", {}), data.get("usage", {}), len(chunk)
        rows = []
        for t in chunk:
            rows.append({"track_uid": t["track_uid"], "p_ok": validate_noul(answers.get(question_key(t["track_uid"]))),
                         "latency_ms": round(latency / n, 3), "request_latency_ms": round(latency, 1),
                         "request_size": n, "tokens_in": usage.get("input_tokens", 0) / n,
                         "tokens_out": usage.get("output_tokens", 0) / n, "jev_model": data.get("model"),
                         "reused": stored is not None})
        uids = [t["track_uid"] for t in chunk]
        with self.lock:
            with self.requests_file.open("a") as f:
                f.write(json.dumps({"track_uids": uids, "request": body}, ensure_ascii=False) + "\n")
            with self.responses_file.open("a") as f:
                f.write(json.dumps({"track_uids": uids, "request": body, "response": data,
                                    "latency_ms": round(latency, 1), "reused": stored is not None},
                                   ensure_ascii=False) + "\n")
        return rows

    def judge(self, tracks: list[dict]) -> list[dict]:
        """One decision row per track, in track order."""
        chunks = self.chunks(tracks)
        by_uid = {}
        with ThreadPoolExecutor(self.cfg["jev"]["max_concurrency"]) as pool:
            for k, rows in enumerate(pool.map(self._run_chunk, chunks), 1):
                by_uid.update({r["track_uid"]: r for r in rows})
                if self.log and k % 50 == 0:
                    self.log(f"  Jev {k}/{len(chunks)} requests")
        return [by_uid[t["track_uid"]] for t in tracks]
