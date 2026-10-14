"""④ Jev gates and threshold fitting (dev only).

A gate keeps Jev's class for the boxes it handles alone ("auto") and sends the rest to the VLM.
- gate        (primary): send to the VLM when Jev's need_vlm level >= tau_e   (Jev asked directly)
- gate_conf   : send when Jev's top-class probability < tau_c                  (confidence only)
- gate_either : send when either rule fires
Each tau is fitted on dev as the setting with the most auto-handled boxes whose error <= max_auto_error.
"""

import numpy as np

from jevgate.core.config import BACKGROUND

GATES = ("gate", "gate_conf", "gate_either")


def routes_to_vlm(gate: str, jev_row: dict, th: dict) -> bool:
    """True when this gate sends the box to the VLM."""
    need = jev_row["need_vlm"] >= th["tau_need_vlm"]
    unsure = jev_row["confidence"] < th["tau_conf"]
    return {"gate": need, "gate_conf": unsure, "gate_either": need or unsure}[gate]


def combine(gate: str, jev_rows: list[dict], vlm_rows: list[dict], th: dict) -> list[dict]:
    """Gate decisions. Latency of a routed box = Jev latency + VLM latency (Jev always runs first)."""
    vlm = {r["cand_id"]: r for r in vlm_rows}
    out = []
    for j in jev_rows:
        to_vlm = routes_to_vlm(gate, j, th)
        src = vlm[j["cand_id"]] if to_vlm else j
        out.append({
            "cand_id": j["cand_id"], "pred": src["pred"], "probs": src["probs"], "confidence": src["confidence"],
            "routed_to": "vlm" if to_vlm else "jev",
            "latency_ms": j["latency_ms"] + (vlm[j["cand_id"]]["latency_ms"] if to_vlm else 0.0),
            "tokens_in": j.get("tokens_in", 0), "tokens_out": j.get("tokens_out", 0),
            "need_vlm": j["need_vlm"], "jev_confidence": j["confidence"],
        })
    return out


def detector_pred(dino_label: str, score: float, t: float) -> str:
    return dino_label if score >= t else BACKGROUND


def fit_detector_threshold(y: list[str], dino_labels: list[str], scores: list[float]) -> float:
    """t that maximises 7-way accuracy of the detector baseline on dev (ties -> the lowest t)."""
    y, labels, scores = np.asarray(y), np.asarray(dino_labels), np.asarray(scores, dtype=float)
    best_t, best_acc = float("inf"), -1.0
    for t in np.unique(np.concatenate([scores, [np.inf]])):
        acc = float((np.where(scores >= t, labels, BACKGROUND) == y).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    return best_t


def auto_curve(correct, value, grid, auto_when: str) -> list[dict]:
    """For each threshold: share of boxes Jev handles alone and the error on those boxes.

    auto_when='below' -> auto if value < thr (need_vlm); 'at_least' -> auto if value >= thr (confidence).
    """
    correct, value = np.asarray(correct, dtype=bool), np.asarray(value, dtype=float)
    rows = []
    for thr in grid:
        auto = value < thr if auto_when == "below" else value >= thr
        rows.append({"tau": float(thr), "auto_rate": float(auto.mean()) if len(value) else None,
                     "auto_error": float((~correct[auto]).mean()) if auto.any() else None,
                     "n_auto": int(auto.sum())})
    return rows


def need_vlm_grid(levels: int, step: float) -> np.ndarray:
    return np.round(np.arange(1.0, levels + step + 1e-9, step), 6)


def conf_grid(step: float) -> np.ndarray:
    return np.round(np.arange(0.0, 1.0 + step + 1e-9, step), 6)


def fit_tau(curve: list[dict], max_auto_error: float, no_auto_value: float) -> float:
    """The qualifying threshold with the most auto-handled boxes; no_auto_value when none qualifies."""
    ok = [r for r in curve if r["n_auto"] > 0 and r["auto_error"] <= max_auto_error]
    return max(ok, key=lambda r: r["n_auto"])["tau"] if ok else no_auto_value  # first max = smallest tau in grid order


def fit_gate_thresholds(y: list[str], jev_rows: list[dict], cfg: dict) -> dict:
    """tau_need_vlm and tau_conf from dev Jev decisions on scored boxes."""
    correct = [r["pred"] == yy for r, yy in zip(jev_rows, y, strict=True)]
    levels = len(cfg["prompts"]["need_vlm_levels"])
    eps = cfg["gate"]["max_auto_error"]
    need_curve = auto_curve(correct, [r["need_vlm"] for r in jev_rows],
                            need_vlm_grid(levels, cfg["gate"]["need_vlm_grid_step"]), "below")
    conf_curve = auto_curve(correct, [r["confidence"] for r in jev_rows],
                            conf_grid(cfg["gate"]["conf_grid_step"]), "at_least")
    return {
        "tau_need_vlm": fit_tau(need_curve, eps, 1.0),          # 1.0 -> every box goes to the VLM
        "tau_conf": fit_tau(conf_curve, eps, 1.0 + cfg["gate"]["conf_grid_step"]),
        "max_auto_error": eps,
    }
