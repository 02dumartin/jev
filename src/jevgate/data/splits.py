"""Video-level train/val/test split (no video appears in two splits).

The original Brackish lists split *frames* at random, so every video sits in every split and a
trained detector would see near-duplicates of test frames. Here whole videos are assigned, stratified
by the dataset's video category, and the assignment whose per-class box shares are closest to the
target fractions is kept (fixed seed, every class present in val and test).
"""

import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from jevgate.data.brackish import BrackishGT, parse_frame_name, split_files

SPLITS = ("train", "val", "test")


def video_categories(cfg: dict) -> dict[str, str]:
    """video name -> category folder under dataset/videos (crab, fish-big, ...)."""
    root = Path(cfg["data"]["root"]) / "dataset" / "videos"
    return {p.stem: cat.name for cat in sorted(root.iterdir()) if cat.is_dir() for p in sorted(cat.iterdir())}


def listed_frames(cfg: dict) -> list[str]:
    """Every frame in the original split lists (frames outside them have unknown GT and are dropped)."""
    return sorted({f for s in ("dev", "test") for f in split_files(cfg, s)}, key=parse_frame_name)


def video_stats(cfg: dict, gt: BrackishGT) -> dict[str, dict]:
    """Per video: listed frames and GT boxes per class."""
    stats: dict[str, dict] = defaultdict(lambda: {"frames": 0, "boxes": Counter()})
    for f in listed_frames(cfg):
        video, _ = parse_frame_name(f)
        stats[video]["frames"] += 1
        stats[video]["boxes"].update(b.label for b in gt.frame_boxes(f))
    return dict(stats)


def _score(assign: dict[str, str], stats: dict, classes: list[str], fractions: dict, min_boxes: int) -> float:
    """Distance of every split's per-class and frame share from its target fraction (lower is better)."""
    totals = {c: sum(s["boxes"][c] for s in stats.values()) for c in classes}
    frames_total = sum(s["frames"] for s in stats.values())
    score = 0.0
    for split in ("val", "test"):
        vids = [v for v, s in assign.items() if s == split]
        for c in classes:
            n = sum(stats[v]["boxes"][c] for v in vids)
            score += abs(n / totals[c] - fractions[split]) if totals[c] else 0.0
            score += 10.0 if n < min_boxes else 0.0
        score += abs(sum(stats[v]["frames"] for v in vids) / frames_total - fractions[split])
    return score


def make_video_split(stats: dict, categories: dict[str, str], classes: list[str], fractions: dict,
                     min_boxes: int, trials: int, seed: int) -> dict[str, str]:
    """Best of `trials` category-stratified random assignments (deterministic for a seed)."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for v in sorted(stats):
        by_cat[categories[v]].append(v)
    rng = random.Random(seed)
    best, best_score = None, float("inf")
    for _ in range(trials):
        assign = {}
        for vids in by_cat.values():
            vids = vids[:]
            rng.shuffle(vids)
            n_val = max(1, round(len(vids) * fractions["val"]))
            n_test = max(1, round(len(vids) * fractions["test"]))
            for k, v in enumerate(vids):
                assign[v] = "val" if k < n_val else "test" if k < n_val + n_test else "train"
        score = _score(assign, stats, classes, fractions, min_boxes)
        if score < best_score:
            best, best_score = assign, score
    return best


def split_summary(assign: dict[str, str], stats: dict, classes: list[str]) -> dict:
    out = {}
    for split in SPLITS:
        vids = sorted(v for v, s in assign.items() if s == split)
        out[split] = {"videos": len(vids), "frames": sum(stats[v]["frames"] for v in vids),
                      "boxes": {c: sum(stats[v]["boxes"][c] for v in vids) for c in classes}}
    return out


def write_split(path: Path, assign: dict[str, str], stats: dict, classes: list[str], params: dict) -> None:
    doc = {"params": params, "summary": split_summary(assign, stats, classes),
           "videos": {split: sorted(v for v, s in assign.items() if s == split) for split in SPLITS}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))


def load_split(path: Path) -> dict[str, list[str]]:
    """{split: [video, ...]} from a committed split file."""
    return yaml.safe_load(Path(path).read_text())["videos"]
