"""Brackish data access: clip frame lists and AAU ground truth (PLAN.md §2).

GT rules:
- AAU CSV is the ground-truth source (covers train/valid/test; COCO json lacks valid).
- A frame listed in a split file but absent from AAU has no objects -> gt_status 'empty_by_absence'.
- A frame outside every split file has unknown GT -> gt_status 'unknown' (excluded from scoring).
"""

import csv
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .boxes import clip_box

FRAME_RE = re.compile(r"^(?P<clip>.+)-(?P<idx>\d{4})\.png$")


@dataclass
class GTBox:
    object_id: int
    label: str
    box_xyxy: list[float]


@dataclass
class BrackishGT:
    """Ground truth per frame file name, plus the set of frames that belong to any split."""

    boxes: dict[str, list[GTBox]] = field(default_factory=lambda: defaultdict(list))
    in_split: set[str] = field(default_factory=set)

    def status(self, file: str) -> str:
        if file in self.boxes:
            return "annotated"
        if file in self.in_split:
            return "empty_by_absence"
        return "unknown"

    def frame_boxes(self, file: str) -> list[GTBox]:
        return self.boxes.get(file, [])


def parse_frame_name(file: str) -> tuple[str, int]:
    """'<clip>-0042.png' -> ('<clip>', 42)."""
    m = FRAME_RE.match(file)
    if not m:
        raise ValueError(f"unexpected frame file name: {file}")
    return m["clip"], int(m["idx"])


def load_gt(cfg: dict) -> BrackishGT:
    """Read AAU CSVs and split lists into a BrackishGT (boxes clipped to the frame)."""
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
                file = Path(file).name
                box = clip_box([float(x1), float(y1), float(x2), float(y2)], wh)
                gt.boxes[file].append(GTBox(object_id=int(obj), label=tag, box_xyxy=box))
    for rel in cfg["data"]["split_lists"]:
        gt.in_split |= {line.strip() for line in (root / rel).open() if line.strip()}
    gt.boxes = dict(gt.boxes)
    return gt


def list_clip_frames(cfg: dict, clip: str) -> list[str]:
    """Sorted frame file names of one clip in the images folder."""
    images = Path(cfg["data"]["root"]) / cfg["data"]["images_dir"]
    files = sorted(p.name for p in images.glob(f"{clip}-*.png"))
    if not files:
        raise FileNotFoundError(f"no frames for clip {clip} under {images}")
    return files


def build_frames(cfg: dict, gt: BrackishGT) -> list[dict]:
    """S1: frame records for the configured split (optionally truncated for smoke runs)."""
    split = cfg["run"]["split"]
    limit = cfg["run"].get("max_frames_per_clip")
    rows = []
    for clip in cfg["clips"][split]:
        files = list_clip_frames(cfg, clip)
        for file in files[:limit] if limit else files:
            _, idx = parse_frame_name(file)
            rows.append({"clip": clip, "split": split, "frame_idx": idx, "file": file,
                         "gt_status": gt.status(file)})
    return rows


def image_path(cfg: dict, file: str) -> Path:
    return Path(cfg["data"]["root"]) / cfg["data"]["images_dir"] / file
