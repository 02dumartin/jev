"""Common judge interface: every system turns candidates into one decision row per candidate."""

from abc import ABC, abstractmethod


class BudgetExceeded(RuntimeError):
    """A paid or slow resource hit its configured cap; the run stops instead of overspending."""


class Budget:
    """Counts requests/tokens/calls and refuses work past the caps in `budget:`."""

    def __init__(self, cfg: dict):
        self.caps = cfg["budget"]
        self.jev_requests = 0
        self.jev_tokens = 0
        self.vlm_calls = 0

    def reserve_jev_request(self) -> None:
        if self.jev_requests >= self.caps["max_jev_requests"]:
            raise BudgetExceeded(f"max_jev_requests={self.caps['max_jev_requests']} reached")
        if self.jev_tokens >= self.caps["max_jev_tokens"]:
            raise BudgetExceeded(f"max_jev_tokens={self.caps['max_jev_tokens']} reached")
        self.jev_requests += 1

    def add_jev_tokens(self, n: int) -> None:
        self.jev_tokens += n

    def reserve_vlm_calls(self, n: int) -> None:
        if self.vlm_calls + n > self.caps["max_vlm_calls"]:
            raise BudgetExceeded(f"max_vlm_calls={self.caps['max_vlm_calls']} would be exceeded")
        self.vlm_calls += n

    def summary(self) -> dict:
        return {"jev_requests": self.jev_requests, "jev_tokens": self.jev_tokens, "vlm_calls": self.vlm_calls}


def decision(cand_id: str, probs: dict[str, float], latency_ms: float, **extra) -> dict:
    """One S5 row: pred = argmax class, confidence = its probability. Contract fields come first."""
    pred = max(probs, key=probs.get)
    row = {"cand_id": cand_id, "pred": pred, "probs": {k: round(float(v), 5) for k, v in probs.items()},
           "confidence": round(float(probs[pred]), 5), "latency_ms": round(float(latency_ms), 3),
           "tokens_in": 0, "tokens_out": 0}
    row.update(extra)
    return row


class Judge(ABC):
    """A system that assigns one of the judgment classes to every candidate box."""

    name: str

    @abstractmethod
    def judge(self, frames: list[dict], candidates: list[dict], evidence: list[dict]) -> list[dict]:
        """Return decision rows in the same order as `candidates`."""
