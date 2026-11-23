"""Jev step: one STATE, many typed questions in one request; plus the same questions one request each."""

import time
from concurrent.futures import ThreadPoolExecutor

from jevgate.models.jev import JevClient, JevError, validate_noul


def question_body(q: dict) -> dict:
    """The API form of one configured question."""
    body = {"type": q["type"], "instructions": q["instructions"]}
    if q.get("criteria") is not None:
        body["criteria"] = q["criteria"]
    return body


def request(cfg: dict, state: dict, keys: list[str]) -> dict:
    return {
        "model": cfg["jev"]["model"],
        "state": {"camera": "convenience store ceiling camera", "observation": state},
        "questions": {k: question_body(cfg["questions"][k]) for k in keys},
    }


def card(key: str, q: dict, ans: dict, sure_at: float) -> dict:
    """One answer as a screen card: value, probabilities, top probability, sure or not."""
    if q["type"] == "noul":
        p = validate_noul(ans)
        probs = {"yes": p, "no": 1 - p}
        value = "YES" if p >= 0.5 else "NO"
    elif q["type"] == "choice":
        probs = {k: float(v) for k, v in ans["probabilities"].items()}
        value = ans["choice"]
    elif q["type"] == "score":
        legend = ans["legend"]
        probs = {legend[k]: float(v) for k, v in sorted(ans["probabilities"].items(), key=lambda kv: int(kv[0]))}
        value = legend[str(int(round(ans["score"])))]
    else:
        raise JevError(f"unknown question type {q['type']}")
    top = max(probs.values())
    return {
        "key": key,
        "title": q["title"],
        "type": q["type"],
        "value": value,
        "probs": probs,
        "top": round(top, 4),
        "sure": top >= sure_at,
        "p": probs.get("yes"),
    }


class Decider:
    def __init__(self, cfg: dict, budget):
        self.cfg, self.client = cfg, JevClient(cfg, budget)
        self.keys = list(cfg["questions"])

    def multi(self, state: dict) -> dict:
        """All questions in one request."""
        body = request(self.cfg, state, self.keys)
        data, ms = self.client.post(body)
        cards = [card(k, self.cfg["questions"][k], data["answers"][k], self.cfg["jev"]["confident"]) for k in self.keys]
        return {
            "request": body,
            "response": data,
            "latency_ms": round(ms, 1),
            "cards": cards,
            "usage": data.get("usage", {}),
            "model": data.get("model"),
        }

    def single(self, state: dict) -> dict:
        """Each question in its own request, sent concurrently; wall time and per-request latencies."""

        def one(k):
            data, ms = self.client.post(request(self.cfg, state, [k]))
            return k, data, ms

        started = time.perf_counter()
        with ThreadPoolExecutor(len(self.keys)) as pool:
            res = list(pool.map(one, self.keys))
        wall = (time.perf_counter() - started) * 1000
        return {
            "wall_ms": round(wall, 1),
            "latencies_ms": {k: round(ms, 1) for k, _, ms in res},
            "tokens": sum(
                d.get("usage", {}).get("input_tokens", 0) + d.get("usage", {}).get("output_tokens", 0)
                for _, d, _ in res
            ),
            "answers": {k: d["answers"][k] for k, d, _ in res},
        }


def check(rule_value, card: dict) -> bool:
    """'>=0.75' against a Boolean card's yes-probability (a card without a probability never matches)."""
    op, thr = rule_value[:2], float(rule_value[2:])
    p = card["p"] if card["p"] is not None else card["top"]
    if p is None:
        return False
    return p >= thr if op == ">=" else p <= thr


def action(cfg: dict, cards: list[dict]) -> dict:
    """First configured rule whose conditions all hold; a `from_choice` rule maps a Choice answer to a badge."""
    by_key = {c["key"]: c for c in cards}
    for rule in cfg["actions"]:
        if "from_choice" in rule:
            c = by_key[rule["from_choice"]]
            b = rule["badges"].get(c["value"], {"badge": "UNKNOWN", "color": "#9A9A9A"})
            return {
                "badge": b["badge"],
                "color": b["color"],
                "rule": {"from_choice": c["key"]},
                "sure": c["sure"],
                "p": c["top"],
            }
        ok = True
        for k, v in rule["when"].items():
            if k == "any_unsure":
                ok &= any(not c["sure"] for c in cards) == v
            elif k == "unsure_in":
                ok &= any(not by_key[q]["sure"] for q in v)
            else:
                ok &= check(v, by_key[k])
        if ok:
            why = [by_key[q]["title"] for q in rule["when"].get("unsure_in", []) if not by_key[q]["sure"]]
            return {
                "badge": rule["badge"],
                "color": rule["color"],
                "rule": rule["when"],
                "sure": True,
                "reason": f"unsure: {', '.join(why)}" if why else None,
            }
    return {"badge": "NO ACTION", "color": "#4ED36B", "rule": {}, "sure": True}


def option_values(q: dict) -> list[str]:
    """Allowed answers for a question in the VLM-only path."""
    if q["type"] == "noul":
        return ["yes", "no"]
    return list(q["criteria"])


def vlm_cards(cfg: dict, answers: dict) -> list[dict]:
    """VLM-only answers as cards: a plain value, no probabilities (sure by construction, top = None)."""
    cards = []
    for key, q in cfg["questions"].items():
        raw = str(answers.get(key, "")).strip().lower().replace(" ", "_")
        opts = option_values(q)
        match = next((o for o in opts if o.lower().replace(" ", "_") == raw), None)
        value = ("YES" if match == "yes" else "NO") if q["type"] == "noul" and match else (match or "?")
        p = (1.0 if match == "yes" else 0.0) if q["type"] == "noul" and match else None
        cards.append(
            {
                "key": key,
                "title": q["title"],
                "type": q["type"],
                "value": value,
                "probs": {},
                "top": None,
                "sure": True,
                "p": p,
                "valid": match is not None,
            }
        )
    return cards
