"""③ Jev (TypeSafe API): a Noul question per candidate box over text evidence.

Jev accepts text only, so each box is described by its detector score, geometry and causal track
features (evidence.py). With batching='per_frame' one request carries every box of a frame as
separate Noul questions over a shared state.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from .base import Budget, Judge, decision

RETRY_STATUS = {429, 503, 529}


class JevError(RuntimeError):
    """Jev returned an error or an answer that fails validation; nothing is guessed."""


def box_view(ev: dict) -> dict:
    """The part of an evidence row that Jev sees (no ids, no frame indices)."""
    return {"detector": ev["detector"], "geometry": ev["geometry"], "track": ev["track"]}


def validate_noul(answer: dict) -> float:
    """Return the Noul probability or raise JevError if it is missing or outside [0, 1]."""
    try:
        value = answer["noul"]
    except (KeyError, TypeError):
        raise JevError(f"missing noul in answer: {answer!r}") from None
    if type(value) not in (int, float) or not 0.0 <= value <= 1.0:
        raise JevError(f"noul outside [0, 1]: {value!r}")
    return float(value)


def build_request(cfg: dict, boxes: list[tuple[str, dict]]) -> dict:
    """Request body for a group of (cand_id, evidence) pairs that share one state."""
    question = cfg["prompts"]["question"]
    criteria = {"true": "a real animal is inside the box",
                "false": "no animal: turbidity, particle, bubble, seabed texture or background"}
    if len(boxes) == 1:
        cand_id, ev = boxes[0]
        state = {"context": cfg["prompts"]["context"], "candidate_box": box_view(ev)}
        questions = {cand_id: {"type": "noul", "instructions": question, "criteria": criteria}}
    else:
        names = {cand_id: f"B{k + 1}" for k, (cand_id, _) in enumerate(boxes)}
        state = {
            "context": cfg["prompts"]["context"],
            "candidates_in_frame": len(boxes),
            "candidate_boxes": {names[cid]: box_view(ev) for cid, ev in boxes},
        }
        questions = {
            cid: {"type": "noul",
                  "instructions": {"question": question,
                                   "target": f"Judge only candidate box `{names[cid]}` from the state."},
                  "criteria": criteria}
            for cid, _ in boxes
        }
    return {"model": cfg["jev"]["model"], "state": state, "questions": questions}


class JevClient:
    """Thin HTTP client with retry on rate limits/overload. The API key is never logged."""

    def __init__(self, cfg: dict, budget: Budget):
        key = os.environ.get(cfg["jev"]["api_key_env"])
        if not key:
            raise JevError(f"{cfg['jev']['api_key_env']} is not set (.env)")
        self.endpoint = cfg["jev"]["endpoint"]
        self.max_retries = cfg["jev"]["max_retries"]
        self.budget = budget
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
            if response.status_code in RETRY_STATUS and attempt + 1 < self.max_retries:
                time.sleep(0.5 * 2**attempt)
                continue
            if response.is_error:
                raise JevError(f"HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
            usage = data.get("usage", {})
            self.budget.add_jev_tokens(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))
            return data, latency
        raise JevError("retries exhausted")


class JevJudge(Judge):
    name = "jev"

    def __init__(self, cfg: dict, budget: Budget):
        self.cfg = cfg
        self.client = JevClient(cfg, budget)

    def _groups(self, candidates, evidence) -> list[list[tuple[str, dict]]]:
        ev = {e["cand_id"]: e for e in evidence}
        if self.cfg["jev"]["batching"] == "per_box":
            return [[(c["cand_id"], ev[c["cand_id"]])] for c in candidates]
        groups: dict[tuple, list] = {}
        for c in candidates:
            groups.setdefault((c["clip"], c["frame_idx"]), []).append((c["cand_id"], ev[c["cand_id"]]))
        return list(groups.values())

    def _run_group(self, group) -> list[dict]:
        body = build_request(self.cfg, group)
        data, latency = self.client.post(body)
        usage = data.get("usage", {})
        n = len(group)
        rows = []
        for cand_id, _ in group:
            p = validate_noul(data.get("answers", {}).get(cand_id))
            rows.append(decision(
                cand_id, p >= 0.5, p, latency / n,
                tokens_in=usage.get("input_tokens", 0) / n, tokens_out=usage.get("output_tokens", 0) / n,
                request_latency_ms=round(latency, 1), request_size=n, jev_model=data.get("model"),
            ))
        return rows

    def judge(self, frames, candidates, evidence):
        groups = self._groups(candidates, evidence)
        with ThreadPoolExecutor(self.cfg["jev"]["max_concurrency"]) as pool:
            results = list(pool.map(self._run_group, groups))
        by_id = {r["cand_id"]: r for rows in results for r in rows}
        return [by_id[c["cand_id"]] for c in candidates]
