"""Turning judge outputs into box predictions, the two gates, and the dev-only thresholds (PLAN_brackish.md v7 §3)."""

import numpy as np

from jevgate.core.config import BACKGROUND


def jev_confidence(p: float) -> float:
    """Confidence of Jev's own accept/reject call: max(p, 1 - p)."""
    return max(p, 1.0 - p)


def gate_decisions(jev_rows: list[dict], vlm_accept: dict[str, bool], vlm_ms: dict[str, float],
                   tau: float, pre_ms: dict[str, float] | None = None) -> list[dict]:
    """⑥: Jev decides when max(p, 1-p) >= tau, otherwise the track goes to the large VLM.

    Latency of a track = describer + its Jev share (+ the large VLM's time when routed).
    """
    out = []
    for j in jev_rows:
        auto = jev_confidence(j["p_ok"]) >= tau
        uid = j["track_uid"]
        base = (pre_ms or {}).get(uid, 0.0) + j["latency_ms"]
        out.append({"track_uid": uid, "accept": j["p_ok"] >= 0.5 if auto else vlm_accept[uid],
                    "routed_to": "jev" if auto else "vlm", "p_ok": j["p_ok"],
                    "latency_ms": base + (0.0 if auto else vlm_ms[uid])})
    return out


def band_gate_decisions(small_rows: list[dict], vlm_accept: dict[str, bool], vlm_ms: dict[str, float],
                        lo: float, hi: float) -> list[dict]:
    """⑦: the small VLM decides alone when p_small >= hi (accept) or <= lo (reject), else the large VLM."""
    out = []
    for r in small_rows:
        uid, p = r["track_uid"], r["p_yes"]
        auto = p >= hi or p <= lo
        out.append({"track_uid": uid, "accept": p >= hi if auto else vlm_accept[uid],
                    "routed_to": "small_vlm" if auto else "vlm", "p_small": p,
                    "latency_ms": r["latency_ms"] + (0.0 if auto else vlm_ms[uid])})
    return out


def box_predictions(detections: list[dict], majority: dict[str, str], accept: dict[str, bool] | None,
                    conf_t: float, per_frame: bool = False) -> dict[str, str]:
    """box_id -> predicted class for one system.

    per_frame=True           -> ① yolo: every box is YOLO's class when conf >= t, else background
    accept=None              -> ② track_vote: every tracked box gets its track's majority class
    accept=dict              -> ③-⑦: tracked boxes of accepted tracks get the majority class, rejected -> background
    Untracked boxes (the tracker never confirmed them, so no judge saw them) follow ①'s rule in every system.
    """
    out = {}
    for d in detections:
        uid = d["track_uid"]
        if per_frame or uid is None:
            out[d["box_id"]] = d["cls"] if d["conf"] >= conf_t else BACKGROUND
        else:
            out[d["box_id"]] = majority[uid] if accept is None or accept[uid] else BACKGROUND
    return out


def fit_conf_threshold(y: list[str], cls: list[str], conf: list[float]) -> float:
    """t maximising box accuracy of `cls if conf >= t else background` (ties -> the lowest t)."""
    y, cls, conf = np.asarray(y), np.asarray(cls), np.asarray(conf, dtype=float)
    best_t, best_acc = float("inf"), -1.0
    for t in np.unique(np.r_[conf, np.inf]):
        acc = float((np.where(conf >= t, cls, BACKGROUND) == y).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def fit_accept_threshold(y_track: list[int], score: list[float]) -> float:
    """v maximising track accuracy of `accept if score >= v` (ties -> the lowest v)."""
    y, s = np.asarray(y_track, dtype=bool), np.asarray(score, dtype=float)
    best_v, best_acc = float("inf"), -1.0
    for v in np.unique(np.r_[s, np.inf]):
        acc = float(((s >= v) == y).mean())
        if acc > best_acc:
            best_v, best_acc = float(v), acc
    return best_v


def tau_curve(y_track: list[int], p: list[float], grid) -> list[dict]:
    """Per tau: share of tracks Jev handles alone and the error rate on them."""
    y, p = np.asarray(y_track, dtype=bool), np.asarray(p, dtype=float)
    conf = np.maximum(p, 1 - p)
    rows = []
    for tau in grid:
        auto = conf >= tau
        err = float(((p[auto] >= 0.5) != y[auto]).mean()) if auto.any() else None
        rows.append({"tau": float(tau), "auto_rate": float(auto.mean()) if len(y) else None,
                     "auto_error": err, "n_auto": int(auto.sum())})
    return rows


def fit_tau(y_track: list[int], p: list[float], max_auto_error: float, step: float) -> float:
    """The tau with the most Jev-handled tracks whose dev error <= max_auto_error (1 + step if none)."""
    grid = np.round(np.arange(0.5, 1.0 + 1e-9, step), 6)
    ok = [r for r in tau_curve(y_track, p, grid) if r["n_auto"] > 0 and r["auto_error"] <= max_auto_error]
    return min(r["tau"] for r in ok) if ok else float(1.0 + step)


def fit_band(y_track: list[int], p: list[float], max_auto_error: float) -> tuple[float, float]:
    """(lo, hi) for ⑦: accept when p >= hi, reject when p <= lo, both sides auto-handled.

    Chooses the band with the most auto-handled tracks whose dev error <= max_auto_error.
    (lo, hi) = (-1, 2) routes every track to the large VLM.
    """
    y, p = np.asarray(y_track, dtype=bool), np.asarray(p, dtype=float)
    cuts = np.unique(np.r_[p, -1.0, 2.0])
    best, best_n = (-1.0, 2.0), 0
    for lo in cuts:
        for hi in cuts[cuts > lo]:
            acc_side, rej_side = p >= hi, p <= lo
            n = int(acc_side.sum() + rej_side.sum())
            if n <= best_n:
                continue
            errors = int((~y[acc_side]).sum() + y[rej_side].sum())
            if errors / n <= max_auto_error:
                best, best_n = (float(lo), float(hi)), n
    return best
