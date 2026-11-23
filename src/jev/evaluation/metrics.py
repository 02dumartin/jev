"""Binary calibration against an explicit target; no fitted calibration transform.

ECE uses equal-width bins over P(yes). Confidence ECE instead bins max(p, 1-p)
against correctness. Empty bins have null means and zero weight; p=1 is in the last bin.
"""

import numpy as np


def binary_metrics(labels, probabilities, bins: int = 10) -> dict:
    y, p = np.asarray(labels, dtype=float), np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError("labels/probabilities must be nonempty aligned 1D arrays")
    if not np.all(np.isin(y, [0, 1])) or not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("binary labels and finite probabilities in [0,1] required")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    pred = p >= 0.5
    correct = pred == y

    def reliability(values, targets):
        indices = np.minimum((values * bins).astype(int), bins - 1)
        rows = []
        for i in range(bins):
            mask = indices == i
            n = int(mask.sum())
            rows.append(
                {
                    "lower": i / bins,
                    "upper": (i + 1) / bins,
                    "n": n,
                    "mean_probability": float(values[mask].mean()) if n else None,
                    "observed_rate": float(targets[mask].mean()) if n else None,
                }
            )
        ece = sum(r["n"] / len(y) * abs(r["mean_probability"] - r["observed_rate"]) for r in rows if r["n"])
        mce = max(abs(r["mean_probability"] - r["observed_rate"]) for r in rows if r["n"])
        return rows, float(ece), float(mce)

    rel, ece, mce = reliability(p, y)
    conf = np.maximum(p, 1 - p)
    conf_rel, conf_ece, _ = reliability(conf, correct)
    safe = np.clip(p, 1e-15, 1 - 1e-15)
    tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
    fn, tn = int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    selective = []
    for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]:
        keep = conf >= threshold
        selective.append(
            {
                "threshold": threshold,
                "n": int(keep.sum()),
                "coverage": float(keep.mean()),
                "risk": float(1 - correct[keep].mean()) if keep.any() else None,
            }
        )
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "accuracy": float(correct.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "brier": float(np.mean((p - y) ** 2)),
        "nll": float(-np.mean(y * np.log(safe) + (1 - y) * np.log(1 - safe))),
        "ece": ece,
        "mce": mce,
        "confidence_ece": conf_ece,
        "reliability": rel,
        "confidence_reliability": conf_rel,
        "selective": selective,
    }


def took_product(events: list[dict]) -> bool:
    """Proxy only: count(take) > count(put); no events is negative, not unknown.

    Counts do not establish object identity, possession, or a purchase.
    """
    return sum(e["label"] == "take" for e in events) > sum(e["label"] == "put" for e in events)
