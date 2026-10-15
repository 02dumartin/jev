"""evaluation/viz/ (sample frames) and evaluation/videos/ (one mp4 per video: GT | Jev gate -> VLM).

Colour = label (`viz.class_colors`, the same in every image, video and figure).
Style = correctness against GT:
  solid thick      kept with the right class
  solid + "x"      kept with the wrong class (tag names the GT class)
  dashed           kept although there is no animal
  dotted thin      an animal the system dropped (drawn in the GT class colour)
Gate boxes also say who decided their track: [Jev], [VLM] or [untracked] (never reached a judge).
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from jevgate.core.config import BACKGROUND
from jevgate.data.brackish import image_path
from jevgate.outputs.colors import class_colors_bgr

PANELS = [("yolo", "1 YOLO per frame"), ("track_vote", "2 track majority"),
          ("vlm_all", "3 VLM on every track"), ("gate", "5 Jev gate -> VLM")]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def dashed_rect(img, p1, p2, colour, thickness: int, dash: int, gap: int) -> None:
    """Rectangle drawn as dashes (OpenCV has no dashed line)."""
    (x1, y1), (x2, y2) = p1, p2
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, max(length, 1), dash + gap):
            e = min(s + dash, length)
            t0, t1 = s / max(length, 1), e / max(length, 1)
            q0 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            q1 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, q0, q1, colour, thickness, cv2.LINE_AA)


def label_text(img, text: str, x: int, y: int, colour) -> None:
    """Tag on a dark backing so every label colour stays readable on the murky frames."""
    (w, h), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    y = max(h + 4, y)
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 2), (20, 20, 20), -1)
    cv2.putText(img, text, (x + 2, y - 2), FONT, 0.42, colour, 1, cv2.LINE_AA)


def draw_gt(img: np.ndarray, gt_boxes, colours: dict) -> np.ndarray:
    """GT boxes in their class colour."""
    for g in gt_boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in g.box_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), colours[g.label], 2)
        label_text(img, g.label, x1, y1 - 2, colours[g.label])
    return img


def draw(img: np.ndarray, boxes: list[dict], preds: dict[str, str], ys: dict[str, str | None], colours: dict,
         route: dict[str, str] | None = None) -> np.ndarray:
    """System boxes: colour = predicted label (GT label for dropped animals), style = correctness."""
    for b in boxes:
        pred, y = preds[b["box_id"]], ys[b["box_id"]]
        p1 = tuple(int(round(v)) for v in b["box_xyxy"][:2])
        p2 = tuple(int(round(v)) for v in b["box_xyxy"][2:])
        who = ""
        if route is not None:
            who = f" [{'untracked' if b['track_uid'] is None else 'Jev' if route[b['track_uid']] == 'jev' else 'VLM'}]"
        track = f" #{b['track_uid'].split('#')[1]}" if b["track_uid"] else ""
        if pred == BACKGROUND:
            if y in (BACKGROUND, None):
                continue  # correctly dropped (or unscored): nothing to show
            dashed_rect(img, p1, p2, colours[y], 1, 3, 4)
            label_text(img, f"missed {y}{who}", p1[0], p1[1] - 2, colours[y])
            continue
        colour = colours[pred]
        if y == BACKGROUND:
            dashed_rect(img, p1, p2, colour, 2, 8, 5)
            tag = f"x {pred} (no animal){track}{who}"
        elif y is not None and y != pred:
            cv2.rectangle(img, p1, p2, colour, 2)
            tag = f"x {pred} (GT {y}){track}{who}"
        else:
            cv2.rectangle(img, p1, p2, colour, 3 if y == pred else 1)
            tag = f"{pred} {b['conf']:.2f}{track}{who}"
        label_text(img, tag, p1[0], p1[1] - 2, colour)
    return img


def caption(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def legend(width: int, colours: dict) -> np.ndarray:
    """Bottom strip: one swatch per label plus the style key."""
    strip = np.zeros((28, width, 3), dtype=np.uint8)
    x = 8
    for name, colour in colours.items():
        cv2.rectangle(strip, (x, 8), (x + 14, 20), colour, -1)
        cv2.putText(strip, name, (x + 18, 19), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
        x += 26 + cv2.getTextSize(name, FONT, 0.45, 1)[0][0]
    key = "solid = right   x = wrong class   dashed = no animal   dotted = missed animal"
    cv2.putText(strip, key, (max(x + 20, width - 640), 19), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


class Renderer:
    """Indexes boxes per frame once; draws sample sheets and videos."""

    def __init__(self, cfg: dict, frames: list[dict], detections: list[dict], judgments: list[dict], gt,
                 gate_route: dict[str, str] | None = None):
        self.cfg, self.frames, self.gt, self.gate_route = cfg, frames, gt, gate_route
        self.colours = class_colors_bgr(cfg)
        self.by_frame = defaultdict(list)
        for d in detections:
            self.by_frame[d["file"]].append(d)
        self.preds = {s: {j["box_id"]: j[s] for j in judgments} for s, _ in PANELS if s in judgments[0]} \
            if judgments else {}
        self.ys = {j["box_id"]: j["y"] for j in judgments}

    def panel(self, f: dict, system: str, title: str) -> np.ndarray:
        """One frame drawn for `system`; system 'gt' shows the AAU boxes only."""
        img = cv2.imread(str(image_path(self.cfg, f["file"])))
        gt_boxes = self.gt.frame_boxes(f["file"])
        if system == "gt":
            return caption(draw_gt(img, gt_boxes, self.colours), f"GT (AAU): {len(gt_boxes)} animals   {f['file']}")
        route = self.gate_route if system == "gate" else None
        img = draw(img, self.by_frame[f["file"]], self.preds[system], self.ys, self.colours, route)
        shown = sum(self.preds[system][b["box_id"]] != BACKGROUND for b in self.by_frame[f["file"]])
        return caption(img, f"{title}: {shown} kept   {f['file']}")

    def samples(self, out: Path, n: int, seed: int) -> None:
        picks = np.random.default_rng(seed).choice(len(self.frames), size=min(n, len(self.frames)), replace=False)
        for k in sorted(picks):
            f = self.frames[k]
            panels = [self.panel(f, s, t) for s, t in PANELS if s in self.preds]
            while len(panels) % 2:
                panels.append(np.zeros_like(panels[0]))
            sheet = np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)])
            sheet = np.vstack([sheet, legend(sheet.shape[1], self.colours)])
            sheet = cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{Path(f['file']).stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def video(self, out: Path, video: str, left: str, right: str, fps: int) -> Path:
        """Left: system `left`, right: system `right`, frame by frame (H.264 mp4: plays in VS Code and PPT)."""
        titles = dict(PANELS) | {"gt": "GT"}
        path, writer = out / f"{video}.mp4", None
        for f in (f for f in self.frames if f["video"] == video):
            frame = np.hstack([self.panel(f, left, titles[left]), self.panel(f, right, titles[right])])
            frame = np.vstack([frame, legend(frame.shape[1], self.colours)])
            if writer is None:
                writer = h264_writer(path, frame.shape[1], frame.shape[0], fps)
            writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
        if writer is not None:
            writer.close()
        return path


def h264_writer(path: Path, width: int, height: int, fps: int):
    """Generator that takes RGB frames and encodes H.264 / yuv420p with the ffmpeg bundled in imageio-ffmpeg."""
    import imageio_ffmpeg

    writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                         pix_fmt_out="yuv420p", quality=7, macro_block_size=2,
                                         output_params=["-movflags", "+faststart"])
    writer.send(None)  # start the generator
    return writer


def transcode_h264(src: Path, dst: Path) -> None:
    """Re-encode any video OpenCV can read (e.g. mp4v) to H.264."""
    cap = cv2.VideoCapture(str(src))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 15
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            writer = h264_writer(dst, frame.shape[1], frame.shape[0], fps)
        writer.send(np.ascontiguousarray(frame[:, :, ::-1]))
    cap.release()
    if writer is not None:
        writer.close()
