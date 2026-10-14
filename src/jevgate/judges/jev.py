"""③ Jev (TypeSafe API): two questions per candidate box over text evidence.

- `class`: choice over the judgment classes -> class probabilities.
- `need_vlm`: 5-level score "how much would looking at the image be needed" -> level 1-5.

Jev accepts text only, so each box is described by its detector label/score, geometry and causal
track features (evidence.py). With batching='per_frame' one request carries every box of a frame.
"""

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from ..config import class_names
from .base import Budget, Judge, decision

RETRY_STATUS = {429, 500, 502, 503, 529}


class JevError(RuntimeError):
    """Jev returned an error or an answer that fails validation; nothing is guessed."""


def box_view(ev: dict) -> dict:
    """The part of an evidence row that Jev sees (no ids, no frame indices)."""
    return {"detector": ev["detector"], "geometry": ev["geometry"], "track": ev["track"]}


def validate_choice(answer: dict, classes: list[str]) -> dict[str, float]:
    """Class probabilities from a choice answer, or JevError if keys/values/sum are wrong."""
    try:
        probs = answer["probabilities"]
    except (KeyError, TypeError):
        raise JevError(f"missing probabilities in choice answer: {answer!r}") from None
    if set(probs) != set(classes):
        raise JevError(f"choice keys {sorted(probs)} != classes")
    vals = list(probs.values())
    if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in vals):
        raise JevError(f"choice probability outside [0, 1]: {probs}")
    total = sum(vals)
    if abs(total - 1) > 0.02:
        raise JevError(f"choice probabilities sum to {total}")
    return {c: float(probs[c]) / total for c in classes}


def validate_score(answer: dict, n_levels: int) -> dict:
    """need_vlm level on a 1..n scale (Jev reports levels 0..n-1) plus its level probabilities."""
    try:
        score, probs = answer["score"], answer["probabilities"]
    except (KeyError, TypeError):
        raise JevError(f"missing score in answer: {answer!r}") from None
    if type(score) not in (int, float) or not 0 <= score <= n_levels - 1:
        raise JevError(f"score outside [0, {n_levels - 1}]: {score!r}")
    return {"need_vlm": round(float(score) + 1, 4),
            "need_vlm_probs": {str(int(k) + 1): float(v) for k, v in probs.items()},
            "need_vlm_confidence": answer.get("confidence")}


def question_keys(cand_id: str) -> tuple[str, str]:
    return f"{cand_id}|class", f"{cand_id}|need_vlm"


def build_request(cfg: dict, boxes: list[tuple[str, dict]]) -> dict:
    """Request body for a group of (cand_id, evidence) pairs that share one state.

    Class definitions and the need_vlm scale are written once in the state; each box gets two
    questions that refer to them (fewer tokens than repeating them in every question).
    """
    p = cfg["prompts"]
    names = {cid: f"B{k + 1}" for k, (cid, _) in enumerate(boxes)}
    levels = list(p["need_vlm_levels"])
    state = {"context": p["context"], "class_definitions": dict(cfg["classes"]), "need_vlm_scale": levels,
             "candidates_in_frame": len(boxes), "candidate_boxes": {names[cid]: box_view(ev) for cid, ev in boxes}}
    questions = {}
    for cid, _ in boxes:
        target = f"Judge only candidate box `{names[cid]}` from the state."
        q_class, q_need = question_keys(cid)
        questions[q_class] = {"type": "choice",
                              "instructions": {"question": p["question"], "target": target,
                                               "options": "defined in `class_definitions`"},
                              "criteria": dict.fromkeys(cfg["classes"])}
        questions[q_need] = {"type": "score",
                             "instructions": {"question": p["need_vlm_question"], "target": target,
                                              "levels": "defined in `need_vlm_scale`"},
                             "criteria": [str(k + 1) for k in range(len(levels))]}
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

    def __init__(self, cfg: dict, budget: Budget, log=None):
        self.cfg = cfg
        self.classes = class_names(cfg)
        self.n_levels = len(cfg["prompts"]["need_vlm_levels"])
        self.client = JevClient(cfg, budget)
        self.log = log
        self.raw: list[dict] = []

    def _groups(self, candidates, evidence) -> list[list[tuple[str, dict]]]:
        ev = {e["cand_id"]: e for e in evidence}
        if self.cfg["jev"]["batching"] == "per_box":
            return [[(c["cand_id"], ev[c["cand_id"]])] for c in candidates]
        groups: dict[tuple, list] = {}
        for c in candidates:
            groups.setdefault((c["clip"], c["frame_idx"]), []).append((c["cand_id"], ev[c["cand_id"]]))
        return list(groups.values())

    def _run_group(self, group) -> tuple[list[dict], dict]:
        body = build_request(self.cfg, group)
        data, latency = self.client.post(body)
        usage = data.get("usage", {})
        answers = data.get("answers", {})
        n = len(group)
        rows = []
        for cid, _ in group:
            q_class, q_need = question_keys(cid)
            probs = validate_choice(answers.get(q_class), self.classes)
            need = validate_score(answers.get(q_need), self.n_levels)
            rows.append(decision(
                cid, probs, latency / n,
                tokens_in=usage.get("input_tokens", 0) / n, tokens_out=usage.get("output_tokens", 0) / n,
                request_latency_ms=round(latency, 1), request_size=n, jev_model=data.get("model"),
                jev_choice_confidence=answers[q_class].get("confidence"), **need,
            ))
        raw = {"cand_ids": [cid for cid, _ in group], "request": body, "response": data,
               "latency_ms": round(latency, 1)}
        return rows, raw

    def judge(self, frames, candidates, evidence):
        groups = self._groups(candidates, evidence)
        by_id, done = {}, 0
        with ThreadPoolExecutor(self.cfg["jev"]["max_concurrency"]) as pool:
            for rows, raw in pool.map(self._run_group, groups):
                by_id.update({r["cand_id"]: r for r in rows})
                self.raw.append(raw)
                done += 1
                if self.log and done % 1000 == 0:
                    self.log(f"  Jev {done}/{len(groups)} requests")
        return [by_id[c["cand_id"]] for c in candidates]
