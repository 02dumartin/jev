"""Scoring: box-level accuracy/precision/recall, calibration (ECE, Brier), block-bootstrap CI."""

from collections import defaultdict

import numpy as np


def binary_metrics(y, keep) -> dict:
    """Accuracy, precision, recall of keep (=predicted real) against y (1 = real)."""
    y, keep = np.asarray(y, dtype=bool), np.asarray(keep, dtype=bool)
    n = len(y)
    tp = int((y & keep).sum())
    fp = int((~y & keep).sum())
    fn = int((y & ~keep).sum())
    return {
        "accuracy": float((y == keep).mean()) if n else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "n": n, "tp": tp, "fp": fp, "fn": fn,
    }


def ece(y, p, n_bins: int) -> float | None:
    """Expected calibration error of P(real) with equal-width bins over [0, 1].

    Each bin contributes |mean(p) - mean(y)| weighted by its share of samples. This is the
    binary (P(y=1)) form, so it measures whether "p = 0.8" means "80% of these are real".
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return None
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability_bins(y, p, n_bins: int) -> list[dict]:
    """Per-bin mean confidence, observed frequency and count (for the reliability diagram)."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append({
            "lo": b / n_bins, "hi": (b + 1) / n_bins, "count": int(mask.sum()),
            "mean_p": float(p[mask].mean()) if mask.any() else None,
            "frac_real": float(y[mask].mean()) if mask.any() else None,
        })
    return rows


def brier(y, p) -> float | None:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(((p - y) ** 2).mean()) if len(y) else None


def frame_visible_accuracy(frame_keys, y_any_real: dict, kept_any: dict) -> float | None:
    """Frame-level "is an animal shown?": predicted = any kept box, truth = any real box (or GT box)."""
    keys = list(frame_keys)
    if not keys:
        return None
    return float(np.mean([bool(y_any_real[k]) == bool(kept_any.get(k, False)) for k in keys]))


def block_bootstrap_delta(frame_keys, correct_a: dict, correct_b: dict, n: int, block: int, seed: int) -> dict:
    """95% CI of acc(a) - acc(b) by resampling blocks of consecutive frames within each clip.

    correct_x maps frame key -> list of 0/1 correctness for the scored boxes in that frame
    (same boxes for a and b). Consecutive frames are correlated, so whole blocks are resampled.
    """
    by_clip: dict[str, list] = defaultdict(list)
    for key in sorted(frame_keys):
        by_clip[key[0]].append(key)
    blocks = []
    for keys in by_clip.values():
        for s in range(0, len(keys), block):
            chunk = keys[s:s + block]
            a = sum(sum(correct_a[k]) for k in chunk)
            b = sum(sum(correct_b[k]) for k in chunk)
            m = sum(len(correct_a[k]) for k in chunk)
            if m:
                blocks.append((a, b, m))
    if not blocks:
        return {"delta": None, "ci_low": None, "ci_high": None, "n_blocks": 0}
    arr = np.asarray(blocks, dtype=float)
    point = (arr[:, 0].sum() - arr[:, 1].sum()) / arr[:, 2].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    sample = arr[idx]
    deltas = (sample[..., 0].sum(1) - sample[..., 1].sum(1)) / sample[..., 2].sum(1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi), "n_blocks": len(arr)}


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
