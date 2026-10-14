"""S8: per-frame overlays (GT, DINO candidates, each system's class calls) and a 2x3 comparison sheet.

Box colour rule for a system's call (BGR for OpenCV):
  correct animal = green, animal judged as another animal = yellow, background judged as an
  animal (FP) = red, animal judged as background (FN) = orange (thin), background kept as
  background = grey (thin), ambiguous y = white (thin).
"""

from pathlib import Path

import cv2
import numpy as np

from .config import BACKGROUND
from .data import image_path
from .label import AMBIGUOUS
from .runio import RunDir

GREEN, YELLOW, RED, ORANGE, GREY, WHITE = (80, 190, 80), (0, 200, 235), (60, 60, 230), (0, 140, 255), \
    (150, 150, 150), (255, 255, 255)
VIEWS = ["gt", "dino", "threshold", "vlm", "jev", "gate"]


def call_colour(y: str, pred: str) -> tuple[tuple, int]:
    """(colour, thickness) for a system's call on a box with label y."""
    if y == AMBIGUOUS:
        return WHITE, 1
    if y == BACKGROUND:
        return (GREY, 1) if pred == BACKGROUND else (RED, 2)
    if pred == BACKGROUND:
        return ORANGE, 1
    return (GREEN, 2) if pred == y else (YELLOW, 2)


def _text(img, text, x, y, colour):
    cv2.putText(img, text, (x, max(10, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, max(10, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)


def _caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_gt(img, gt_boxes):
    for g in gt_boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in g.box_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        _text(img, g.label, x1, y1 - 3, GREEN)
    return img


def draw_dino(img, cands, labels):
    for c in cands:
        x1, y1, x2, y2 = (int(round(v)) for v in c["box_xyxy"])
        y = labels[c["cand_id"]]["y"]
        colour = WHITE if y == AMBIGUOUS else GREY if y == BACKGROUND else GREEN
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 1 if y in (AMBIGUOUS, BACKGROUND) else 2)
        _text(img, f"{c['dino_label']} {c['dino_score']:.2f}", x1, y1 - 3, colour)
    return img


def draw_calls(img, cands, labels, decisions):
    for c in cands:
        x1, y1, x2, y2 = (int(round(v)) for v in c["box_xyxy"])
        d = decisions[c["cand_id"]]
        colour, width = call_colour(labels[c["cand_id"]]["y"], d["pred"])
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, width)
        if d["pred"] != BACKGROUND or labels[c["cand_id"]]["y"] not in (BACKGROUND, AMBIGUOUS):
            tag = f"{d['pred']} {d['confidence']:.2f}" + (" >VLM" if d.get("routed_to") == "vlm" else "")
            _text(img, tag, x1, y1 - 3, colour)
    return img


def render_samples(run: RunDir, gt, n: int, seed: int) -> Path:
    """n random frames: one image per view under viz/frames/<view>/ plus a 2x3 compare sheet."""
    cfg = run.config()
    frames = run.read("frames")
    labels = {r["cand_id"]: r for r in run.read("labels")}
    by_frame: dict[tuple, list] = {}
    for c in run.read("candidates"):
        by_frame.setdefault((c["clip"], c["frame_idx"]), []).append(c)
    primary = cfg["gate"]["primary"]
    systems = {s: {r["cand_id"]: r for r in run.read_decisions(s)} for s in run.systems_done()}
    out = run.path / "viz" / "frames"
    rng = np.random.default_rng(seed)
    for k in sorted(rng.choice(len(frames), size=min(n, len(frames)), replace=False)):
        f = frames[k]
        base = cv2.imread(str(image_path(cfg, f["file"])))
        cands = by_frame.get((f["clip"], f["frame_idx"]), [])
        stem = f"{f['clip'][-25:]}-{f['frame_idx']:04d}.jpg"
        panels = {
            "gt": _caption(draw_gt(base.copy(), gt.frame_boxes(f["file"])), f"GT  {f['file']}"),
            "dino": _caption(draw_dino(base.copy(), cands, labels), f"DINO candidates ({len(cands)})"),
        }
        for view, s in (("threshold", "threshold"), ("vlm", "vlm"), ("jev", "jev"), ("gate", primary)):
            if s in systems:
                panels[view] = _caption(draw_calls(base.copy(), cands, labels, systems[s]), f"{s}")
        for view, img in panels.items():
            (out / view).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / view / stem), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles = [panels.get(v, np.zeros_like(base)) for v in VIEWS]
        sheet = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6])])
        sheet = cv2.resize(sheet, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
        (out / "compare").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "compare" / stem), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out
