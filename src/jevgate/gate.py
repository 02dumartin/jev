"""④ Jev gate and threshold fitting (t for the detector baseline, tau for the gate). Dev only."""

import numpy as np


def jev_confidence(p_real: float) -> float:
    """Confidence of Jev's own keep/reject call: max(p, 1 - p)."""
    return max(p_real, 1.0 - p_real)


def combine(jev_rows: list[dict], vlm_rows: list[dict], tau: float) -> list[dict]:
    """Gate decisions: keep Jev's call when its confidence >= tau, otherwise use the VLM's call.

    Latency of a routed box = Jev latency + VLM latency (the Jev call always happens first).
    """
    vlm = {r["cand_id"]: r for r in vlm_rows}
    out = []
    for j in jev_rows:
        auto = jev_confidence(j["p_real"]) >= tau
        v = vlm[j["cand_id"]]
        out.append({
            "cand_id": j["cand_id"],
            "keep": j["keep"] if auto else v["keep"],
            "p_real": j["p_real"],
            "routed_to": "jev" if auto else "vlm",
            "latency_ms": j["latency_ms"] + (0.0 if auto else v["latency_ms"]),
            "tokens_in": j.get("tokens_in", 0),
            "tokens_out": j.get("tokens_out", 0),
            "tau": tau,
        })
    return out


def fit_detector_threshold(y, scores) -> float:
    """t that maximises accuracy of `score >= t` on dev (ties -> the lowest such t)."""
    y, scores = np.asarray(y, dtype=bool), np.asarray(scores, dtype=float)
    candidates = np.unique(np.concatenate([scores, [np.inf]]))
    accs = [((scores >= t) == y).mean() for t in candidates]
    return float(candidates[int(np.argmax(accs))])


def tau_curve(y, p, grid) -> list[dict]:
    """For each tau: share of boxes Jev handles alone and the error rate on those boxes."""
    y, p = np.asarray(y, dtype=bool), np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p)
    rows = []
    for tau in grid:
        auto = conf >= tau
        err = float(((p[auto] >= 0.5) != y[auto]).mean()) if auto.any() else None
        rows.append({"tau": float(tau), "auto_rate": float(auto.mean()) if len(y) else None,
                     "auto_error": err, "n_auto": int(auto.sum())})
    return rows


def fit_tau(y, p, max_auto_error: float, step: float) -> float:
    """Smallest tau in [0.5, 1] whose auto-handled error on dev is <= max_auto_error.

    Returns 1.0 + step (route everything to the VLM) when no tau qualifies.
    """
    grid = np.round(np.arange(0.5, 1.0 + 1e-9, step), 6)
    for row in tau_curve(y, p, grid):
        if row["n_auto"] > 0 and row["auto_error"] <= max_auto_error:
            return row["tau"]
    return float(1.0 + step)
