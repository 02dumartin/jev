"""Ground truth for detections (y_box) and tracks (y_track), and the GT the detector missed (PLAN_brackish.md §2.2-2.3)."""

from collections import Counter, defaultdict

from jevgate.core.config import BACKGROUND
from jevgate.data.boxes import iou_matrix
from jevgate.data.brackish import BrackishGT


def iou_to_y(max_iou: float, gt_label: str | None, match_iou: float, background_iou: float) -> str | None:
    """>= match_iou -> the GT class, < background_iou -> background, in between -> None (ambiguous)."""
    if max_iou >= match_iou:
        return gt_label
    if max_iou < background_iou:
        return BACKGROUND
    return None


def label_boxes(detections: list[dict], gt: BrackishGT, cfg: dict) -> list[dict]:
    """One label per detection: class-agnostic max IoU with the frame's GT boxes.

    Frames with unknown GT (outside every split list) are not scored.
    """
    match_iou, bg_iou = cfg["label"]["match_iou"], cfg["label"]["background_iou"]
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for d in detections:
        by_frame[d["file"]].append(d)
    labels = {}
    for file, dets in by_frame.items():
        status = gt.status(file)
        gts = gt.frame_boxes(file)
        ious = iou_matrix([d["box_xyxy"] for d in dets], [g.box_xyxy for g in gts])
        for i, d in enumerate(dets):
            j = int(ious[i].argmax()) if ious.shape[1] else None
            max_iou = float(ious[i, j]) if j is not None else 0.0
            hit = gts[j] if j is not None and max_iou > 0 else None
            y = None if status == "unknown" else iou_to_y(max_iou, hit.label if hit else None, match_iou, bg_iou)
            labels[d["box_id"]] = {"box_id": d["box_id"], "gt_status": status, "max_iou": round(max_iou, 4),
                                   "matched_object_id": hit.object_id if hit else None,
                                   "matched_label": hit.label if hit else None, "y": y, "scored": y is not None}
    return [labels[d["box_id"]] for d in detections]


def label_tracks(tracks: list[dict], box_labels: dict[str, dict], ok_share: float) -> list[dict]:
    """y_track = 1 when >= ok_share of the track's scored boxes have y = its majority class.

    type: correct | misclassified (mostly another animal) | false (mostly background) | unscored.
    """
    rows = []
    for t in tracks:
        ys = [box_labels[b]["y"] for b in t["box_ids"] if box_labels[b]["scored"]]
        if not ys:
            rows.append({"track_uid": t["track_uid"], "y_track": None, "type": "unscored", "n_scored": 0,
                         "y_votes": {}})
            continue
        share = sum(y == t["majority_class"] for y in ys) / len(ys)
        ok = share >= ok_share
        others = Counter(y for y in ys if y != t["majority_class"])
        kind = "correct" if ok else ("false" if others.most_common(1)[0][0] == BACKGROUND else "misclassified")
        rows.append({"track_uid": t["track_uid"], "y_track": int(ok), "type": kind, "n_scored": len(ys),
                     "majority_share_correct": round(share, 3), "y_votes": dict(Counter(ys).most_common())})
    return rows


def missed_objects(detections: list[dict], frames: list[dict], gt: BrackishGT, match_iou: float) -> dict:
    """GT objects with no detection at IoU >= match_iou (no judge can recover them), per class."""
    by_frame: dict[str, list] = defaultdict(list)
    for d in detections:
        by_frame[d["file"]].append(d["box_xyxy"])
    total, covered = Counter(), Counter()
    for f in frames:
        gts = gt.frame_boxes(f["file"])
        if not gts:
            continue
        ious = iou_matrix([g.box_xyxy for g in gts], by_frame.get(f["file"], []))
        for i, g in enumerate(gts):
            total[g.label] += 1
            covered[g.label] += int(ious.shape[1] > 0 and ious[i].max() >= match_iou)
    n, c = sum(total.values()), sum(covered.values())
    return {"n_gt_objects": n, "n_covered": c, "recall_upper_bound": c / n if n else None,
            "per_class": {k: {"gt": total[k], "covered": covered[k],
                              "recall_upper_bound": covered[k] / total[k]} for k in sorted(total)}}
