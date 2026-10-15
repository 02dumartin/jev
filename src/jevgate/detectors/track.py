"""YOLO on whole held-out videos: per-frame detection vs ByteTrack tracking, side by side.

The detector is the image model trained on the video split; tracking only links its boxes across
consecutive frames (track ids, and spurious one-frame boxes are not confirmed as tracks).
"""

import json
from pathlib import Path

import cv2
import numpy as np

from jevgate.data.brackish import image_path, parse_frame_name
from jevgate.data.splits import load_split


def clip_frames(cfg: dict, clip: str) -> list[str]:
    """Frame files of one video, in frame order."""
    folder = Path(cfg["data"]["root"]) / cfg["data"]["images_dir"]
    files = [p.name for p in folder.glob(f"{clip}-*.png")]
    if not files:
        raise FileNotFoundError(f"no frames for {clip} under {folder}")
    return sorted(files, key=lambda f: parse_frame_name(f)[1])


def split_videos(cfg: dict, repo: Path) -> list[str]:
    """Videos assigned to `track.split` in the frozen video split."""
    return load_split(repo / cfg["video_split"]["file"])[cfg["track"]["split"]]


def count_flicker(counts: list[int]) -> float:
    """Mean absolute change in the number of shown boxes between consecutive frames (lower = steadier)."""
    return float(np.mean(np.abs(np.diff(counts)))) if len(counts) > 1 else 0.0


def track_stats(rows: list[dict]) -> dict:
    """Number of distinct tracks and their length in frames (median, max)."""
    lengths: dict[int, int] = {}
    for r in rows:
        lengths[r["track_id"]] = lengths.get(r["track_id"], 0) + 1
    vals = sorted(lengths.values())
    return {"n_tracks": len(vals), "track_len_median": float(np.median(vals)) if vals else 0.0,
            "track_len_max": vals[-1] if vals else 0}


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class ClipTracker:
    """Runs one detect-only model and one tracking model over the frames of a video."""

    def __init__(self, cfg: dict, weights: Path):
        from ultralytics import YOLO

        self.cfg = cfg
        self.weights = weights
        self.detector = YOLO(str(weights))
        self.yolo_args = {"conf": cfg["track"]["conf"], "iou": cfg["track"]["iou"],
                          "imgsz": cfg["yolo"]["imgsz"], "device": cfg["yolo"]["device"], "verbose": False}

    def run(self, clip: str, out_dir: Path) -> dict:
        """Write <clip>.mp4 (left: per-frame detection, right: tracking) and <clip>.jsonl; return a summary."""
        from ultralytics import YOLO

        tracker = YOLO(str(self.weights))  # fresh model = fresh tracker state for every video
        out_dir.mkdir(parents=True, exist_ok=True)
        writer, rows = None, []
        det_counts, trk_counts = [], []
        for file in clip_frames(self.cfg, clip):
            frame = cv2.imread(str(image_path(self.cfg, file)))
            det = self.detector.predict(frame, **self.yolo_args)[0]
            trk = tracker.track(frame, persist=True, tracker=self.cfg["track"]["tracker"], **self.yolo_args)[0]
            idx = parse_frame_name(file)[1]
            ids = trk.boxes.id.int().tolist() if trk.boxes.id is not None else []
            for tid, cls, conf, box in zip(ids, trk.boxes.cls.int().tolist(), trk.boxes.conf.tolist(),
                                           trk.boxes.xyxy.tolist(), strict=False):
                rows.append({"frame_idx": idx, "track_id": tid, "class": trk.names[cls], "conf": round(conf, 4),
                             "box_xyxy": [round(v, 1) for v in box]})
            det_counts.append(len(det.boxes))
            trk_counts.append(len(ids))
            panel = np.hstack([_caption(det.plot(line_width=2), f"per-frame detection ({len(det.boxes)})"),
                               _caption(trk.plot(line_width=2), f"tracking ({len(ids)})")])
            if writer is None:
                writer = cv2.VideoWriter(str(out_dir / f"{clip}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         self.cfg["track"]["fps"], (panel.shape[1], panel.shape[0]))
            writer.write(panel)
        if writer is not None:
            writer.release()
        (out_dir / f"{clip}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return {"clip": clip, "frames": len(det_counts),
                "flicker_detect": count_flicker(det_counts), "flicker_track": count_flicker(trk_counts),
                **track_stats(rows)}
