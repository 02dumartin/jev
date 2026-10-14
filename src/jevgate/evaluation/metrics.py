"""Scoring for the 7-way class judgment: accuracy, F1, one-vs-rest 2x2 / ROC / AUC, calibration,
and frame-block bootstrap CIs (PLAN.md §6)."""

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


def frame_blocks(frame_keys, block: int) -> list[list[tuple]]:
    """Consecutive frames (within each clip, sorted) grouped into blocks for the bootstrap."""
    by_clip: dict[str, list] = defaultdict(list)
    for key in sorted(set(frame_keys)):
        by_clip[key[0]].append(key)
    return [keys[s:s + block] for keys in by_clip.values() for s in range(0, len(keys), block)]


class BlockBootstrap:
    """Resample whole frame blocks; statistics are computed on the boxes of the sampled blocks."""

    def __init__(self, box_frames: list[tuple], block: int, n: int, seed: int):
        blocks = frame_blocks(box_frames, block)
        block_of = {k: b for b, keys in enumerate(blocks) for k in keys}
        members = defaultdict(list)
        for i, key in enumerate(box_frames):
            members[block_of[key]].append(i)
        self.members = [np.asarray(members[b], dtype=int) for b in range(len(blocks))]
        rng = np.random.default_rng(seed)
        self.block_draws = [rng.integers(0, len(blocks), len(blocks)) for _ in range(n)] if blocks else []

    def ci(self, stat) -> tuple[float | None, float | None]:
        """2.5 / 97.5 percentiles of stat(indices) over the resamples (None values skipped)."""
        vals = []
        for draw in self.block_draws:
            v = stat(np.concatenate([self.members[b] for b in draw]))
            if v is not None:
                vals.append(v)
        if not vals:
            return None, None
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)


def frame_visible_accuracy(frame_keys, has_animal: dict, predicted_animal: dict) -> float | None:
    """Frame-level "is an animal shown?": predicted = any box judged an animal, truth = any GT box."""
    keys = list(frame_keys)
    if not keys:
        return None
    return float(np.mean([bool(has_animal[k]) == bool(predicted_animal.get(k, False)) for k in keys]))


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
