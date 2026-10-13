"""Box geometry in pixel xyxy: IoU, clipping, class-agnostic NMS."""

import numpy as np


def clip_box(box, wh) -> list[float]:
    """Clip an xyxy box to the frame [0, w] x [0, h]."""
    w, h = wh
    x1, y1, x2, y2 = box
    return [min(max(x1, 0.0), w), min(max(y1, 0.0), h), min(max(x2, 0.0), w), min(max(y2, 0.0), h)]


def area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between N boxes `a` and M boxes `b` (xyxy). Returns an N x M array."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes, scores, iou_thr: float) -> list[int]:
    """Greedy class-agnostic NMS. Returns kept indices, highest score first."""
    order = list(np.argsort(-np.asarray(scores, dtype=float), kind="stable"))
    ious = iou_matrix(boxes, boxes)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        order = [j for j in order if ious[i, j] < iou_thr]
    return keep
