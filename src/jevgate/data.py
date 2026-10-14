"""Brackish data access: split frame lists and AAU ground truth (PLAN.md §2).

GT rules:
- AAU CSV is the ground-truth source (covers train/valid/test; COCO json lacks valid).
- A frame listed in a split file but absent from AAU has no objects -> gt_status 'empty_by_absence'.
- dev = original train + valid frames, test = original test frames (frame-level random split).
"""

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boxes import clip_box

FRAME_RE = re.compile(r"^(?P<clip>.+)-(?P<idx>\d{4})\.png$")


@dataclass
class GTBox:
    object_id: int
    label: str
    box_xyxy: list[float]


@dataclass
class BrackishGT:
    """Ground truth per frame file name."""

    boxes: dict[str, list[GTBox]] = field(default_factory=lambda: defaultdict(list))

    def status(self, file: str) -> str:
        return "annotated" if file in self.boxes else "empty_by_absence"

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
    return gt


def split_files(cfg: dict, split: str) -> list[str]:
    """Sorted frame file names listed in the split's list files."""
    root = Path(cfg["data"]["root"])
    files = set()
    for rel in cfg["splits"][split]:
        files |= {line.strip() for line in (root / rel).open() if line.strip()}
    return sorted(files, key=parse_frame_name)


def evenly_spaced(items: list, n: int | None) -> list:
    """`n` items spread evenly over the list (all items when n is None or >= len)."""
    if not n or n >= len(items):
        return items
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def build_frames(cfg: dict, gt: BrackishGT) -> list[dict]:
    """S1: frame records for the configured split, ordered by (clip, frame_idx)."""
    split = cfg["run"]["split"]
    rows = []
    for file in evenly_spaced(split_files(cfg, split), cfg["run"].get("max_frames")):
        clip, idx = parse_frame_name(file)
        rows.append({"clip": clip, "split": split, "frame_idx": idx, "file": file, "gt_status": gt.status(file)})
    return rows


def available_frames(cfg: dict) -> dict[str, set[int]]:
    """Every frame index present in the images folder, per clip."""
    out: dict[str, set[int]] = defaultdict(set)
    for p in (Path(cfg["data"]["root"]) / cfg["data"]["images_dir"]).glob("*.png"):
        clip, idx = parse_frame_name(p.name)
        out[clip].add(idx)
    return dict(out)


def context_frames(frames: list[dict], history: int, available: dict[str, set[int]]) -> list[tuple[str, int]]:
    """Target frames plus the previous `history` frame numbers of the same video (any split).

    The detector runs on these, so causal evidence can look back at earlier frames even when they
    belong to the other split. Only detector outputs are used from them, never GT.
    """
    keys = set()
    for f in frames:
        for idx in range(f["frame_idx"] - history, f["frame_idx"] + 1):
            if idx in available.get(f["clip"], ()):
                keys.add((f["clip"], idx))
    return sorted(keys)


def split_summary(cfg: dict, gt: BrackishGT, split: str) -> dict:
    """n table for one split: videos, frames, empty frames, GT boxes and frames per class."""
    files = split_files(cfg, split)
    classes = [c for c in cfg["classes"] if c != "background"]
    box_n, frame_n = Counter(), Counter()
    for f in files:
        labels = [b.label for b in gt.frame_boxes(f)]
        box_n.update(labels)
        frame_n.update(set(labels))
    return {
        "videos": len({parse_frame_name(f)[0] for f in files}),
        "frames": len(files),
        "annotated_frames": sum(gt.status(f) == "annotated" for f in files),
        "empty_frames": sum(gt.status(f) == "empty_by_absence" for f in files),
        "gt_boxes": sum(box_n.values()),
        "gt_boxes_per_class": {c: box_n[c] for c in classes},
        "frames_per_class": {c: frame_n[c] for c in classes},
    }


def image_path(cfg: dict, file: str) -> Path:
    return Path(cfg["data"]["root"]) / cfg["data"]["images_dir"] / file
