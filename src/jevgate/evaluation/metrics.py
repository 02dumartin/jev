"""Scoring: 7-way box accuracy and F1, ROC / AUC, calibration, and cluster-bootstrap CIs (PLAN.md v6 §6)."""

from collections import defaultdict

import numpy as np


def multiclass_metrics(y, pred, classes: list[str]) -> dict:
    """Accuracy, macro F1, per-class precision/recall/F1/support and the confusion matrix (rows = y)."""
    y, pred = np.asarray(y), np.asarray(pred)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(y, pred, strict=True):
        cm[idx[a], idx[b]] += 1
    per_class = {}
    for c, i in idx.items():
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        p = tp / (tp + fp) if tp + fp else None
        r = tp / (tp + fn) if tp + fn else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
        per_class[c] = {"precision": p, "recall": r, "f1": f1, "support": int(cm[i, :].sum())}
    f1s = [v["f1"] for v in per_class.values() if v["support"] > 0 and v["f1"] is not None]
    return {
        "accuracy": float((y == pred).mean()) if len(y) else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "n": int(len(y)),
        "per_class": per_class,
        "confusion": {"classes": classes, "matrix": cm.tolist()},
    }


def ovr_counts(y, pred, cls: str) -> dict:
    """One-vs-rest 2x2 for one class: tp, fn, fp, tn, sensitivity, specificity."""
    y, pred = np.asarray(y) == cls, np.asarray(pred) == cls
    tp, fn = int((y & pred).sum()), int((y & ~pred).sum())
    fp, tn = int((~y & pred).sum()), int((~y & ~pred).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "sens": tp / (tp + fn) if tp + fn else None, "spec": tn / (tn + fp) if tn + fp else None}


def auroc(y_bin, score) -> float | None:
    """Rank-based AUC (ties count half). None when one side is empty."""
    y_bin, score = np.asarray(y_bin, dtype=bool), np.asarray(score, dtype=float)
    n_pos, n_neg = int(y_bin.sum()), int((~y_bin).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    # average ranks for ties
    _, first, counts = np.unique(sorted_s, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2 + 1
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[y_bin].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_points(y_bin, score, max_points: int = 400) -> dict:
    """ROC curve (fpr, tpr) thinned to at most max_points for plotting."""
    y_bin, score = np.asarray(y_bin, dtype=bool), np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    s, yb = score[order], y_bin[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps, fps = np.cumsum(yb)[distinct], np.cumsum(~yb)[distinct]
    tpr = np.r_[0, tps / max(yb.sum(), 1)]
    fpr = np.r_[0, fps / max((~yb).sum(), 1)]
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).round().astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}


def ece_top(correct, confidence, n_bins: int) -> float | None:
    """Top-label ECE: per confidence bin |mean confidence - accuracy|, weighted by bin share."""
    correct, conf = np.asarray(correct, dtype=float), np.asarray(confidence, dtype=float)
    if len(conf) == 0:
        return None
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((bins == b).mean() * abs(conf[bins == b].mean() - correct[bins == b].mean())
                     for b in range(n_bins) if (bins == b).any()))


def reliability_bins(correct, confidence, n_bins: int) -> list[dict]:
    """Per-bin mean confidence, accuracy and count (reliability diagram)."""
    correct, conf = np.asarray(correct, dtype=float), np.asarray(confidence, dtype=float)
    bins = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    return [{"lo": b / n_bins, "hi": (b + 1) / n_bins, "count": int((bins == b).sum()),
             "mean_conf": float(conf[bins == b].mean()) if (bins == b).any() else None,
             "accuracy": float(correct[bins == b].mean()) if (bins == b).any() else None}
            for b in range(n_bins)]


def brier_multi(y, probs: list[dict], classes: list[str]) -> float | None:
    """Mean over boxes of sum_k (p_k - 1[y = k])^2."""
    if not len(y):
        return None
    p = np.asarray([[pr[c] for c in classes] for pr in probs], dtype=float)
    onehot = np.asarray([[float(yy == c) for c in classes] for yy in y])
    return float(((p - onehot) ** 2).sum(1).mean())


class ClusterBootstrap:
    """Resample whole clusters (a track and all its boxes); statistics see the items of the drawn clusters."""

    def __init__(self, cluster_of_item: list, n: int, seed: int):
        members = defaultdict(list)
        for i, key in enumerate(cluster_of_item):
            members[key].append(i)
        self.members = [np.asarray(v, dtype=int) for v in members.values()]
        rng = np.random.default_rng(seed)
        k = len(self.members)
        self.draws = [rng.integers(0, k, k) for _ in range(n)] if k else []

    def ci(self, stat) -> tuple[float | None, float | None]:
        """2.5 / 97.5 percentiles of stat(item indices) over the resamples (None values skipped)."""
        vals = [v for draw in self.draws
                if (v := stat(np.concatenate([self.members[c] for c in draw]))) is not None]
        if not vals:
            return None, None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)


def brier_binary(y, p) -> float | None:
    """Mean squared error of P(y = 1)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(((p - y) ** 2).mean()) if len(y) else None


def evaluate_endpoints(delta_ci_low: float | None, call_ratio: float | None, steps: list[dict]) -> list[dict]:
    """Pre-registered sequential criteria: check in order, stop at the first failure."""
    out, still_testing = [], True
    for step in steps:
        if not still_testing or delta_ci_low is None or call_ratio is None:
            out.append({**step, "result": "not_tested"})
            continue
        ok = delta_ci_low >= step["min_delta_acc_ci_low"] and call_ratio <= step["max_vlm_call_ratio"]
        out.append({**step, "result": "pass" if ok else "fail"})
        still_testing = ok
    return out
