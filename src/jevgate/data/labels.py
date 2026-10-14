"""S3: box-level class ground truth for each DINO candidate (PLAN.md §2.2)."""

from collections import Counter

from jevgate.data.boxes import iou_matrix
from jevgate.core.config import BACKGROUND
from jevgate.data.brackish import BrackishGT

AMBIGUOUS = "ambiguous"


def iou_to_label(max_iou: float, gt_label: str | None, match_iou: float, background_iou: float) -> str:
    """>= match_iou -> the matched GT class, < background_iou -> background, otherwise ambiguous."""
    if max_iou >= match_iou and gt_label is not None:
        return gt_label
    if max_iou < background_iou:
        return BACKGROUND
    return AMBIGUOUS


def label_candidates(candidates: list[dict], frames: list[dict], gt: BrackishGT, cfg: dict) -> list[dict]:
    """One label row per candidate: y is a class, 'background' or 'ambiguous' (not scored)."""
    match_iou, background_iou = cfg["label"]["match_iou"], cfg["label"]["background_iou"]
    file_of = {(f["clip"], f["frame_idx"]): f["file"] for f in frames}
    by_frame: dict[tuple, list[dict]] = {}
    for c in candidates:
        by_frame.setdefault((c["clip"], c["frame_idx"]), []).append(c)

    rows = {}
    for key, cands in by_frame.items():
        gt_boxes = gt.frame_boxes(file_of[key])
        ious = iou_matrix([c["box_xyxy"] for c in cands], [g.box_xyxy for g in gt_boxes])
        for i, c in enumerate(cands):
            j = int(ious[i].argmax()) if ious.shape[1] else None
            max_iou = float(ious[i, j]) if j is not None else 0.0
            matched = gt_boxes[j] if j is not None and max_iou > 0 else None
            y = iou_to_label(max_iou, matched.label if matched else None, match_iou, background_iou)
            rows[c["cand_id"]] = {
                "cand_id": c["cand_id"],
                "max_iou": round(max_iou, 4),
                "y": y,
                "matched_object_id": matched.object_id if matched else None,
                "matched_gt_label": matched.label if matched else None,
                "scored": y != AMBIGUOUS,
            }
    return [rows[c["cand_id"]] for c in candidates]


def detector_coverage(candidates: list[dict], frames: list[dict], gt: BrackishGT, match_iou: float,
                      classes: list[str]) -> dict:
    """Per GT class: objects, objects covered by a candidate with IoU >= match_iou (recall upper bound)."""
    by_frame: dict[tuple, list] = {}
    for c in candidates:
        by_frame.setdefault((c["clip"], c["frame_idx"]), []).append(c["box_xyxy"])
    n_gt, n_hit = Counter(), Counter()
    for f in frames:
        gt_boxes = gt.frame_boxes(f["file"])
        if not gt_boxes:
            continue
        ious = iou_matrix([g.box_xyxy for g in gt_boxes], by_frame.get((f["clip"], f["frame_idx"]), []))
        for g, row in zip(gt_boxes, ious, strict=True):
            n_gt[g.label] += 1
            n_hit[g.label] += int(len(row) > 0 and row.max() >= match_iou)
    per_class = {c: {"gt": n_gt[c], "covered": n_hit[c], "missed": n_gt[c] - n_hit[c],
                     "recall_upper_bound": n_hit[c] / n_gt[c] if n_gt[c] else None}
                 for c in classes if c != BACKGROUND}
    total_gt, total_hit = sum(n_gt.values()), sum(n_hit.values())
    return {"gt": total_gt, "covered": total_hit, "missed": total_gt - total_hit,
            "recall_upper_bound": total_hit / total_gt if total_gt else None, "per_class": per_class}


def label_summary(candidates: list[dict], labels: list[dict], coverage: dict, classes: list[str]) -> dict:
    """Count flow: GT objects -> covered/missed; candidates -> y per class, ambiguous, scored."""
    y = Counter(r["y"] for r in labels)
    dino_vs_y = Counter((c.get("dino_label"), r["y"]) for c, r in zip(candidates, labels, strict=True)
                        if c.get("dino_label") is not None)
    return {
        "gt_objects": coverage,
        "candidates": len(candidates),
        "scored": sum(r["scored"] for r in labels),
        "ambiguous": y[AMBIGUOUS],
        "y_per_class": {c: y[c] for c in classes},
        "dino_label_counts": dict(Counter(c["dino_label"] for c in candidates if c.get("dino_label") is not None)),
        "dino_label_vs_y": {f"{a}->{b}": n for (a, b), n in sorted(dino_vs_y.items())},
    }
