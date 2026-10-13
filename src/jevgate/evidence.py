"""S4: causal, text-ready evidence for each candidate (PLAN.md §4).

Only the current frame and the previous K frames' *candidates* are used. GT and future frames are
never read, so Jev sees what an online system would see.
"""

import numpy as np

from .boxes import area, iou_matrix


def geometry(box, wh) -> dict:
    """Size, shape and position features of one box, normalised by the frame size."""
    w, h = wh
    bw, bh = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h
    margin = 2.0
    return {
        "area_frac": round(area(box) / (w * h), 5),
        "width_px": round(bw, 1),
        "height_px": round(bh, 1),
        "aspect_wh": round(bw / bh, 3) if bh > 0 else None,
        "center_x": round(cx, 3),
        "center_y": round(cy, 3),
        "touches_border": bool(box[0] <= margin or box[1] <= margin or box[2] >= w - margin or box[3] >= h - margin),
    }


def track_features(box, history: list[list[dict]], match_iou: float) -> dict:
    """Temporal consistency of `box` against previous frames' candidates (most recent first)."""
    k = len(history)
    seen, best_prev_iou, mean_scores = 0, 0.0, []
    displacement = area_ratio = None
    diag = float(np.hypot(box[2] - box[0], box[3] - box[1])) or 1.0
    for step, prev in enumerate(history):
        if not prev:
            continue
        ious = iou_matrix([box], [c["box_xyxy"] for c in prev])[0]
        j = int(ious.argmax())
        if step == 0:
            best_prev_iou = float(ious[j])
        if ious[j] >= match_iou:
            seen += 1
            mean_scores.append(prev[j]["dino_score"])
            if step == 0:
                pb = prev[j]["box_xyxy"]
                dx = (box[0] + box[2] - pb[0] - pb[2]) / 2
                dy = (box[1] + box[3] - pb[1] - pb[3]) / 2
                displacement = round(float(np.hypot(dx, dy)) / diag, 3)
                area_ratio = round(area(box) / area(pb), 3) if area(pb) > 0 else None
    return {
        "history_frames_available": k,
        "seen_in_prev_frames": seen,
        "iou_with_prev_frame": round(best_prev_iou, 3),
        "motion_vs_prev_frame_per_diag": displacement,
        "area_ratio_vs_prev_frame": area_ratio,
        "track_mean_score": round(float(np.mean(mean_scores)), 3) if mean_scores else None,
    }


def build_evidence(candidates: list[dict], cfg: dict) -> list[dict]:
    """One evidence row per candidate, computed clip by clip in frame order."""
    wh = cfg["data"]["frame_wh"]
    k, match_iou = cfg["evidence"]["history_frames"], cfg["evidence"]["match_iou"]
    by_clip: dict[str, dict[int, list[dict]]] = {}
    for c in candidates:
        by_clip.setdefault(c["clip"], {}).setdefault(c["frame_idx"], []).append(c)

    out = {}
    for frames in by_clip.values():
        idxs = sorted(frames)
        for pos, idx in enumerate(idxs):
            # Previous K frame indices that exist in the clip, most recent first (strictly < idx).
            prev_idxs = [i for i in idxs[max(0, pos - k):pos]][::-1]
            history = [frames[i] for i in prev_idxs]
            current = frames[idx]
            for c in current:
                out[c["cand_id"]] = {
                    "cand_id": c["cand_id"],
                    "history_frame_idxs": prev_idxs,
                    "detector": {"label": c["label"], "score": round(c["dino_score"], 3)},
                    "geometry": geometry(c["box_xyxy"], wh),
                    "track": track_features(c["box_xyxy"], history, match_iou),
                    "scene": {"candidates_in_frame": len(current)},
                }
    return [out[c["cand_id"]] for c in candidates]
