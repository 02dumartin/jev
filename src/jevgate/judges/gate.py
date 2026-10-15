"""Turning judge outputs into box predictions, and the dev-only thresholds t, v, tau (PLAN.md v6 §3)."""

import numpy as np

from jevgate.core.config import BACKGROUND


def jev_confidence(p: float) -> float:
    """Confidence of Jev's own accept/reject call: max(p, 1 - p)."""
    return max(p, 1.0 - p)


def gate_decisions(jev_rows: list[dict], vlm_accept: dict[str, bool], vlm_ms: dict[str, float],
                   tau: float) -> list[dict]:
    """⑤: Jev decides when max(p, 1-p) >= tau, otherwise the track goes to the VLM.

    Latency of a routed track = its Jev share + its VLM time (Jev is always asked first).
    """
    out = []
    for j in jev_rows:
        auto = jev_confidence(j["p_ok"]) >= tau
        uid = j["track_uid"]
        out.append({"track_uid": uid, "accept": j["p_ok"] >= 0.5 if auto else vlm_accept[uid],
                    "routed_to": "jev" if auto else "vlm", "p_ok": j["p_ok"],
                    "latency_ms": j["latency_ms"] + (0.0 if auto else vlm_ms[uid])})
    return out


def box_predictions(detections: list[dict], majority: dict[str, str], accept: dict[str, bool] | None,
                    conf_t: float | None = None) -> dict[str, str]:
    """box_id -> predicted class for one system.

    accept=None, conf_t set  -> ① yolo: YOLO class when conf >= t, else background (no tracking)
    accept=None, conf_t None -> ② track_vote: every tracked box gets its track's majority class
    accept=dict              -> ③④⑤: tracked boxes of accepted tracks get the majority class
    Untracked boxes are background in ②-⑤.
    """
    out = {}
    for d in detections:
        if conf_t is not None:
            out[d["box_id"]] = d["cls"] if d["conf"] >= conf_t else BACKGROUND
            continue
        uid = d["track_uid"]
        keep = uid is not None and (accept is None or accept[uid])
        out[d["box_id"]] = majority[uid] if keep else BACKGROUND
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
