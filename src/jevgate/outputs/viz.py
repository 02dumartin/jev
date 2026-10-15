"""evaluation/viz/ (sample frames) and evaluation/videos/ (one mp4 per video).

Colours (BGR): green = right class, yellow = animal with the wrong class, red = kept but no animal,
orange dashed = an animal the system dropped, grey thin = dropped and indeed no animal.
"""

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from jevgate.core.config import BACKGROUND
from jevgate.data.brackish import image_path

GREEN, YELLOW, RED, ORANGE, GREY = (80, 200, 80), (0, 210, 240), (60, 60, 230), (0, 140, 255), (140, 140, 140)
PANELS = [("yolo", "1 YOLO per frame"), ("track_vote", "2 track majority"),
          ("vlm_all", "3 VLM on every track"), ("gate", "5 Jev gate -> VLM")]


def _colour(pred: str, y: str | None):
    if pred != BACKGROUND:
        return GREEN if pred == y else (RED if y == BACKGROUND else YELLOW if y else GREY)
    return ORANGE if y not in (BACKGROUND, None) else None


def draw(img: np.ndarray, boxes: list[dict], preds: dict[str, str], ys: dict[str, str | None], gt=None) -> np.ndarray:
    for g in gt or []:
        x1, y1, x2, y2 = (int(round(v)) for v in g.box_xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for b in boxes:
        pred, y = preds[b["box_id"]], ys[b["box_id"]]
        x1, y1, x2, y2 = (int(round(v)) for v in b["box_xyxy"])
        colour = _colour(pred, y)
        if colour is None:
            cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2 if pred != BACKGROUND else 1)
        if pred != BACKGROUND:
            tag = f"{pred} {b['conf']:.2f}" + (f" #{b['track_uid'].split('#')[1]}" if b["track_uid"] else "")
            cv2.putText(img, tag, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)
    return img


def caption(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class Renderer:
    """Indexes boxes per frame once; draws sample sheets and videos."""

    def __init__(self, cfg: dict, frames: list[dict], detections: list[dict], judgments: list[dict], gt):
        self.cfg, self.frames, self.gt = cfg, frames, gt
        self.by_frame = defaultdict(list)
        for d in detections:
            self.by_frame[d["file"]].append(d)
        self.preds = {s: {j["box_id"]: j[s] for j in judgments} for s, _ in PANELS if s in judgments[0]} \
            if judgments else {}
        self.ys = {j["box_id"]: j["y"] for j in judgments}

    def panel(self, f: dict, system: str, title: str) -> np.ndarray:
        img = cv2.imread(str(image_path(self.cfg, f["file"])))
        img = draw(img, self.by_frame[f["file"]], self.preds[system], self.ys, self.gt.frame_boxes(f["file"]))
        return caption(img, f"{title}   {f['file']}")

    def samples(self, out: Path, n: int, seed: int) -> None:
        picks = np.random.default_rng(seed).choice(len(self.frames), size=min(n, len(self.frames)), replace=False)
        for k in sorted(picks):
            f = self.frames[k]
            panels = [self.panel(f, s, t) for s, t in PANELS if s in self.preds]
            while len(panels) % 2:
                panels.append(np.zeros_like(panels[0]))
            sheet = np.vstack([np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)])
            sheet = cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"{Path(f['file']).stem}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])

    def video(self, out: Path, video: str, left: str, right: str, fps: int) -> Path:
        """Left: system `left`, right: system `right`, frame by frame (H.264 mp4: plays in VS Code and PPT)."""
        titles = dict(PANELS)
        path, writer = out / f"{video}.mp4", None
        for f in (f for f in self.frames if f["video"] == video):
            frame = np.hstack([self.panel(f, left, titles[left]), self.panel(f, right, titles[right])])
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
