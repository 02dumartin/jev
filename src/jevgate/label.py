"""S3: box-level ground truth for each DINO candidate (class-agnostic IoU with AAU boxes)."""

from .boxes import iou_matrix
from .data import BrackishGT


def iou_to_label(max_iou: float, real_iou: float, fake_iou: float) -> int | None:
    """>= real_iou -> 1 (real), < fake_iou -> 0 (fake), otherwise None (ambiguous, not scored)."""
    if max_iou >= real_iou:
        return 1
    if max_iou < fake_iou:
        return 0
    return None


def label_candidates(candidates: list[dict], frames: list[dict], gt: BrackishGT, cfg: dict) -> list[dict]:
    """One label row per candidate. Frames with unknown GT get y = None and scored = False."""
    real_iou, fake_iou = cfg["label"]["real_iou"], cfg["label"]["fake_iou"]
    status = {(f["clip"], f["frame_idx"]): (f["file"], f["gt_status"]) for f in frames}
    by_frame: dict[tuple, list[dict]] = {}
    for c in candidates:
        by_frame.setdefault((c["clip"], c["frame_idx"]), []).append(c)

    rows = []
    for key, cands in by_frame.items():
        file, gt_status = status[key]
        gt_boxes = gt.frame_boxes(file)
        ious = iou_matrix([c["box_xyxy"] for c in cands], [g.box_xyxy for g in gt_boxes])
        for i, c in enumerate(cands):
            if gt_status == "unknown":
                rows.append({"cand_id": c["cand_id"], "max_iou": None, "y": None,
                             "matched_object_id": None, "matched_label": None, "scored": False})
                continue
            j = int(ious[i].argmax()) if ious.shape[1] else None
            max_iou = float(ious[i, j]) if j is not None else 0.0
            y = iou_to_label(max_iou, real_iou, fake_iou)
            rows.append({
                "cand_id": c["cand_id"],
                "max_iou": round(max_iou, 4),
                "y": y,
                "matched_object_id": gt_boxes[j].object_id if j is not None and max_iou > 0 else None,
                "matched_label": gt_boxes[j].label if j is not None and max_iou > 0 else None,
                "scored": y is not None,
            })
    order = {c["cand_id"]: k for k, c in enumerate(candidates)}
    return sorted(rows, key=lambda r: order[r["cand_id"]])


def detector_recall_upper_bound(candidates: list[dict], frames: list[dict], gt: BrackishGT,
                                real_iou: float) -> dict:
    """Share of GT objects covered by at least one candidate with IoU >= real_iou."""
    by_frame: dict[tuple, list] = {}
    for c in candidates:
        by_frame.setdefault((c["clip"], c["frame_idx"]), []).append(c["box_xyxy"])
    n_gt = n_hit = 0
    for f in frames:
        if f["gt_status"] != "annotated":
            continue
        gt_boxes = [g.box_xyxy for g in gt.frame_boxes(f["file"])]
        cands = by_frame.get((f["clip"], f["frame_idx"]), [])
        ious = iou_matrix(gt_boxes, cands)
        n_gt += len(gt_boxes)
        n_hit += int((ious.max(axis=1) >= real_iou).sum()) if ious.shape[1] else 0
    return {"n_gt_objects": n_gt, "n_covered": n_hit, "recall_upper_bound": n_hit / n_gt if n_gt else None}
