"""S8: overlays of sampled frames (viz/): one folder per view (GT, DINO candidates, each system) + compare/.

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
        if c.get("dino_label") is not None:
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
    """n random frames -> viz/<view>/<clip>-<frame>.jpg per view, and viz/compare/ with all views in one grid."""
    cfg = run.config()
    frames = run.read("frames")
    boxes = run.read("boxes")
    labels = {b["cand_id"]: b for b in boxes}
    by_frame: dict[tuple, list] = {}
    for b in boxes:
        by_frame.setdefault((b["clip"], b["frame_idx"]), []).append(b)
    systems = {s: {r["cand_id"]: r for r in rows} for s, rows in run.system_rows().items()}
    views = [s for s in ("threshold", "vlm", "jev", cfg["gate"]["primary"]) if s in systems]
    out = run.path / "viz"
    rng = np.random.default_rng(seed)
    for k in sorted(rng.choice(len(frames), size=min(n, len(frames)), replace=False)):
        f = frames[k]
        base = cv2.imread(str(image_path(cfg, f["file"])))
        cands = by_frame.get((f["clip"], f["frame_idx"]), [])
        stem = f"{f['clip'][-25:]}-{f['frame_idx']:04d}.jpg"
        panels = {"gt": _caption(draw_gt(base.copy(), gt.frame_boxes(f["file"])), f"GT  {f['file']}")}
        if cfg["candidates"]["source"] == "dino":
            panels["dino"] = _caption(draw_dino(base.copy(), cands, labels), f"DINO candidates ({len(cands)})")
        panels |= {s: _caption(draw_calls(base.copy(), cands, labels, systems[s]), s) for s in views}
        for view, img in panels.items():
            (out / view).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / view / stem), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        tiles = list(panels.values())
        cols = 3 if len(tiles) > 4 else 2
        tiles += [np.zeros_like(base)] * (-len(tiles) % cols)
        sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
        sheet = cv2.resize(sheet, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
        (out / "compare").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "compare" / stem), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out
