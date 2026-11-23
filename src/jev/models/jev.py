"""TypeSafe HTTP client, retries, and typed probability validation."""

import math
import os
import time

import httpx

from jev.models.budget import Budget

RETRY_STATUS = {408, 429, 499, 529}


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


class JevClient:
    """HTTP client with retry on rate limits/overload/5xx. The API key is never logged."""

    def __init__(self, cfg: dict, budget: Budget):
        key = os.environ.get(cfg["jev"]["api_key_env"])
        if not key:
            raise JevError(f"{cfg['jev']['api_key_env']} is not set (.env)")
        self.endpoint, self.max_retries, self.budget = cfg["jev"]["endpoint"], cfg["jev"]["max_retries"], budget
        self.client = httpx.Client(
            http2=True, timeout=cfg["jev"]["timeout_s"], headers={"Authorization": f"Bearer {key}"}
        )

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
