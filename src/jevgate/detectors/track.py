"""YOLO + ByteTrack on whole videos, and per-track summaries (PLAN_brackish.md v6 §1.2, §4.1).

The detector is the image model trained on the video split; the tracker only links its boxes across
consecutive frames. Every detection (conf >= track.conf) is kept as a row; boxes the tracker did not
confirm have track_uid = None. Each tracked box keeps YOLO's own per-frame class, so a track's
majority class is a vote over its frames.
"""

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from jevgate.data.boxes import iou_matrix
from jevgate.data.brackish import image_path, parse_frame_name


def video_short(video: str) -> str:
    """'2019-03-19_17-01-06to2019-03-19_17-01-19_1' -> '0319-170106' (unique per Brackish video)."""
    return f"{video[5:7]}{video[8:10]}-{video[11:19].replace('-', '')}"


def associate(tracked, detected, min_iou: float = 0.5) -> list[int | None]:
    """For each tracked box, the index of the detection it came from (greedy by IoU), or None."""
    ious = iou_matrix(tracked, detected)
    out: list[int | None] = [None] * len(ious)
    used: set[int] = set()
    for flat in np.argsort(-ious, axis=None):
        i, j = divmod(int(flat), ious.shape[1])
        if ious[i, j] < min_iou:
            break
        if out[i] is None and j not in used:
            out[i], used = j, used | {j}
    return out


class VideoTracker:
    """Runs YOLO (all detections) and YOLO + ByteTrack (track ids) over the frames of each video."""

    def __init__(self, cfg: dict, repo: Path):
        from ultralytics import YOLO

        self.cfg = cfg
        self.weights = repo / cfg["track"]["weights"]
        self.detector = YOLO(str(self.weights))
        t = cfg["track"]
        self.args = {"conf": t["conf"], "iou": t["iou"], "imgsz": t["imgsz"], "device": t["device"], "verbose": False}

    def run(self, video: str, files: list[str]) -> tuple[list[dict], list[float]]:
        """Detection rows for one video and the per-frame time (ms) of detection + tracking."""
        from ultralytics import YOLO

        tracker = YOLO(str(self.weights))  # fresh model = fresh tracker state for every video
        short, rows, frame_ms = video_short(video), [], []
        for file in files:
            frame = cv2.imread(str(image_path(self.cfg, file)))
            started = time.perf_counter()
            det = self.detector.predict(frame, **self.args)[0]
            trk = tracker.track(frame, persist=True, tracker=self.cfg["track"]["tracker"], **self.args)[0]
            frame_ms.append((time.perf_counter() - started) * 1000)
            boxes = det.boxes.xyxy.tolist()
            track_uid: list[str | None] = [None] * len(boxes)
            if trk.boxes.id is not None:
                for tid, j in zip(trk.boxes.id.int().tolist(), associate(trk.boxes.xyxy.tolist(), boxes), strict=True):
                    if j is not None:
                        track_uid[j] = f"{short}#{tid}"
            idx = parse_frame_name(file)[1]
            for k, (box, cls, conf) in enumerate(zip(boxes, det.boxes.cls.int().tolist(), det.boxes.conf.tolist(),
                                                     strict=True)):
                rows.append({"box_id": f"{short}-{idx:04d}-{k:02d}", "video": video, "frame_idx": idx, "file": file,
                             "box_xyxy": [round(v, 1) for v in box], "cls": det.names[cls], "conf": round(conf, 4),
                             "track_uid": track_uid[k]})
        return rows, frame_ms


def majority_class(classes: list[str], confs: list[float]) -> tuple[str, float]:
    """Most frequent class (ties -> larger conf sum) and its share of the votes."""
    votes = Counter(classes)
    conf_sum = Counter()
    for c, p in zip(classes, confs, strict=True):
        conf_sum[c] += p
    best = max(votes, key=lambda c: (votes[c], conf_sum[c]))
    return best, votes[best] / len(classes)


def conf_trend(confs: list[float], tol: float = 0.05) -> str:
    """'rising' / 'falling' / 'flat' from the mean conf of the first vs second half of the track."""
    if len(confs) < 2:
        return "flat"
    half = len(confs) // 2
    diff = float(np.mean(confs[half:]) - np.mean(confs[:half]))
    return "rising" if diff > tol else "falling" if diff < -tol else "flat"


def _summary(uid: str, boxes: list[dict], video_range: tuple[int, int], wh) -> dict:
    """Evidence about one track that needs no pixels: length, votes, conf, size, position, motion."""
    w, h = wh
    frames = [b["frame_idx"] for b in boxes]
    xyxy = np.asarray([b["box_xyxy"] for b in boxes], dtype=float)
    bw, bh = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)
    diag = float(np.median(np.hypot(bw, bh))) or 1.0
    steps = np.diff(frames)
    speeds = np.hypot(*np.diff(centers, axis=0).T) / np.maximum(steps, 1) if len(boxes) > 1 else np.zeros(0)
    margin = 2.0
    touch = (xyxy[:, 0] <= margin) | (xyxy[:, 1] <= margin) | (xyxy[:, 2] >= w - margin) | (xyxy[:, 3] >= h - margin)
    classes, confs = [b["cls"] for b in boxes], [b["conf"] for b in boxes]
    major, share = majority_class(classes, confs)
    span = frames[-1] - frames[0] + 1
    return {
        "track_uid": uid, "video": boxes[0]["video"], "box_ids": [b["box_id"] for b in boxes],
        "first_frame": frames[0], "last_frame": frames[-1],
        "rep_box_id": boxes[int(np.argmax(confs))]["box_id"],
        "majority_class": major,
        "evidence": {
            "length": {"frames_detected": len(boxes), "span_frames": span, "gap_frames": span - len(boxes),
                       "starts_at_video_start": frames[0] == video_range[0],
                       "ends_at_video_end": frames[-1] == video_range[1]},
            "classes": {"votes": dict(Counter(classes).most_common()), "majority_class": major,
                        "majority_share": round(share, 3)},
            "detector_conf": {"mean": round(float(np.mean(confs)), 3), "min": round(min(confs), 3),
                              "max": round(max(confs), 3), "trend": conf_trend(confs)},
            "box": {"median_width_px": round(float(np.median(bw)), 1),
                    "median_height_px": round(float(np.median(bh)), 1),
                    "median_center_x": round(float(np.median(centers[:, 0])) / w, 3),
                    "median_center_y": round(float(np.median(centers[:, 1])) / h, 3),
                    "border_touch_share": round(float(touch.mean()), 3)},
            "motion": {"total_displacement_per_box_diag": round(float(np.hypot(*(centers[-1] - centers[0]))) / diag, 3),
                       "mean_speed_px_per_frame": round(float(speeds.mean()), 2) if len(speeds) else 0.0},
        },
    }


def summarize_tracks(detections: list[dict], frames: list[dict], wh) -> list[dict]:
    """One summary per track (ordered by video, first frame), with concurrent tracks per class."""
    video_range: dict[str, tuple[int, int]] = {}
    for f in frames:
        lo, hi = video_range.get(f["video"], (f["frame_idx"], f["frame_idx"]))
        video_range[f["video"]] = (min(lo, f["frame_idx"]), max(hi, f["frame_idx"]))
    groups: dict[str, list[dict]] = {}
    for d in detections:
        if d["track_uid"] is not None:
            groups.setdefault(d["track_uid"], []).append(d)
    tracks = [_summary(uid, sorted(bx, key=lambda b: b["frame_idx"]), video_range[bx[0]["video"]], wh)
              for uid, bx in groups.items()]
    tracks.sort(key=lambda t: (t["video"], t["first_frame"], t["track_uid"]))
    for t in tracks:
        others = Counter(o["majority_class"] for o in tracks if o["video"] == t["video"] and o is not t
                         and o["first_frame"] <= t["last_frame"] and o["last_frame"] >= t["first_frame"])
        t["evidence"]["scene"] = {"other_tracks_at_same_time": dict(others.most_common())}
    return tracks


def count_flicker(counts: list[int]) -> float:
    """Mean absolute change in the number of shown boxes between consecutive frames (lower = steadier)."""
    return float(np.mean(np.abs(np.diff(counts)))) if len(counts) > 1 else 0.0
