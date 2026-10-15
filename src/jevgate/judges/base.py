"""Budget shared by the paid/slow judges: the run stops instead of overspending."""

import threading


class BudgetExceeded(RuntimeError):
    """A paid or slow resource hit its configured cap."""


class Budget:
    """Counts Jev requests/tokens and VLM calls; refuses work past the caps in `budget:`. Thread-safe."""

    def __init__(self, cfg: dict):
        self.caps = cfg["budget"]
        self.jev_requests = 0
        self.jev_tokens = 0
        self.vlm_calls = 0
        self._lock = threading.Lock()

    def reserve_jev_request(self) -> None:
        with self._lock:
            if self.jev_requests >= self.caps["max_jev_requests"]:
                raise BudgetExceeded(f"max_jev_requests={self.caps['max_jev_requests']} reached")
            if self.jev_tokens >= self.caps["max_jev_tokens"]:
                raise BudgetExceeded(f"max_jev_tokens={self.caps['max_jev_tokens']} reached")
            self.jev_requests += 1

    def add_jev_tokens(self, n: int) -> None:
        with self._lock:
            self.jev_tokens += n

    def reserve_vlm_calls(self, n: int) -> None:
        with self._lock:
            if self.vlm_calls + n > self.caps["max_vlm_calls"]:
                raise BudgetExceeded(f"max_vlm_calls={self.caps['max_vlm_calls']} would be exceeded")
            self.vlm_calls += n

    def summary(self) -> dict:
        return {"jev_requests": self.jev_requests, "jev_tokens": self.jev_tokens, "vlm_calls": self.vlm_calls}
