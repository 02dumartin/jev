"""Brackish data access: split frame lists and AAU ground truth (PLAN_brackish.md §2).

GT rules:
- AAU CSV is the ground-truth source (covers train/valid/test; COCO json lacks valid).
- A frame in the original split lists but absent from AAU has no objects -> gt_status 'empty_by_absence'.
- A frame outside every list has unknown GT -> gt_status 'unknown' (not scored).
"""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jevgate.data.boxes import clip_box

FRAME_RE = re.compile(r"^(?P<clip>.+)-(?P<idx>\d{4})\.png$")


@dataclass
class GTBox:
    object_id: int
    label: str
    box_xyxy: list[float]


@dataclass
class BrackishGT:
    """Ground truth per frame file name, plus the frames listed in the original split files."""

    boxes: dict[str, list[GTBox]] = field(default_factory=lambda: defaultdict(list))
    listed: set[str] = field(default_factory=set)

    def status(self, file: str) -> str:
        if file in self.boxes:
            return "annotated"
        return "empty_by_absence" if file in self.listed else "unknown"

    def frame_boxes(self, file: str) -> list[GTBox]:
        return self.boxes.get(file, [])


def parse_frame_name(file: str) -> tuple[str, int]:
    """'<clip>-0042.png' -> ('<clip>', 42)."""
    m = FRAME_RE.match(file)
    if not m:
        raise ValueError(f"unexpected frame file name: {file}")
    return m["clip"], int(m["idx"])


def frame_file(clip: str, idx: int) -> str:
    return f"{clip}-{idx:04d}.png"


def load_gt(cfg: dict) -> BrackishGT:
    """Read AAU CSVs into a BrackishGT (boxes clipped to the frame)."""
    root = Path(cfg["data"]["root"])
    wh = cfg["data"]["frame_wh"]
    gt = BrackishGT()
    for rel in cfg["data"]["aau_csvs"]:
        with (root / rel).open() as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # header
            for row in reader:
                if not row:
                    continue
                file, obj, tag, x1, y1, x2, y2 = row[:7]
                box = clip_box([float(x1), float(y1), float(x2), float(y2)], wh)
                gt.boxes[Path(file).name].append(GTBox(object_id=int(obj), label=tag, box_xyxy=box))
    gt.boxes = dict(gt.boxes)
    gt.listed = set(listed_files(cfg))
    return gt


def listed_files(cfg: dict) -> list[str]:
    """Frames named in the original split lists (train/valid/test.txt), sorted by (video, index)."""
    root = Path(cfg["data"]["root"])
    files = set()
    for rel in cfg["data"]["split_lists"]:
        files |= {line.strip() for line in (root / rel).open() if line.strip()}
    return sorted(files, key=parse_frame_name)


def video_frames(cfg: dict, video: str) -> list[str]:
    """Every frame image of one video in the images folder, in frame order."""
    folder = Path(cfg["data"]["root"]) / cfg["data"]["images_dir"]
    files = [p.name for p in folder.glob(f"{video}-*.png")]
    if not files:
        raise FileNotFoundError(f"no frames for {video} under {folder}")
    return sorted(files, key=lambda f: parse_frame_name(f)[1])


def evenly_spaced(items: list, n: int | None) -> list:
    """`n` items spread evenly over the list (all items when n is None or >= len)."""
    if not n or n >= len(items):
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def build_frames(cfg: dict, gt: BrackishGT, videos: list[str]) -> list[dict]:
    """Frame records of the run's videos, ordered by (video, frame). Smoke runs keep the first N frames."""
    limit = cfg["run"].get("max_frames_per_video")
    rows = []
    for video in videos:
        files = video_frames(cfg, video)
        for file in files[:limit] if limit else files:
            rows.append({"video": video, "frame_idx": parse_frame_name(file)[1], "file": file,
                         "gt_status": gt.status(file)})
    return rows


def image_path(cfg: dict, file: str) -> Path:
    return Path(cfg["data"]["root"]) / cfg["data"]["images_dir"] / file
