"""YOLO12 detector: dataset conversion (AAU -> YOLO format on the video split), training, progress, test eval."""

import csv
import os
from pathlib import Path

import yaml

from .data import BrackishGT, image_path, parse_frame_name
from .splits import SPLITS, listed_frames, load_split


def detector_classes(cfg: dict) -> list[str]:
    """The six animal classes, in config order (YOLO class ids 0-5)."""
    return [c for c in cfg["classes"] if c != "background"]


def yolo_lines(boxes, classes: list[str], wh) -> list[str]:
    """AAU boxes -> 'cls cx cy w h' (normalised); degenerate boxes (<= 1 px) are dropped."""
    w, h = wh
    out = []
    for b in boxes:
        x1, y1, x2, y2 = b.box_xyxy
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue
        out.append(f"{classes.index(b.label)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                   f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def build_dataset(cfg: dict, gt: BrackishGT, repo: Path) -> dict:
    """images/<split>/ (symlinks) + labels/<split>/*.txt + data.yaml. Returns per-split counts."""
    root = repo / cfg["yolo"]["dataset_dir"]
    classes = detector_classes(cfg)
    videos = {v: s for s, vids in load_split(repo / cfg["video_split"]["file"]).items() for v in vids}
    counts = {s: {"frames": 0, "empty_frames": 0, "boxes": 0} for s in SPLITS}
    for f in listed_frames(cfg):
        split = videos[parse_frame_name(f)[0]]
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        link = img_dir / f
        if not link.exists():
            os.symlink(image_path(cfg, f).resolve(), link)
        lines = yolo_lines(gt.frame_boxes(f), classes, cfg["data"]["frame_wh"])
        (lbl_dir / f"{Path(f).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split]["frames"] += 1
        counts[split]["empty_frames"] += not lines
        counts[split]["boxes"] += len(lines)
    data_yaml = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
                 "names": dict(enumerate(classes))}
    (root / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return counts


def train(cfg: dict, repo: Path):
    """Ultralytics training with the config's settings (everything else stays at Ultralytics defaults)."""
    from ultralytics import YOLO

    y = cfg["yolo"]
    model = YOLO(y["model"])
    return model.train(data=str(repo / y["dataset_dir"] / "data.yaml"), epochs=y["epochs"], imgsz=y["imgsz"],
                       batch=y["batch"], device=y["device"], seed=y["seed"], workers=y["workers"],
                       project=str(repo / y["project"]), name=y["name"], exist_ok=False)


def read_results(run_dir: Path) -> list[dict]:
    """Ultralytics results.csv as a list of {column: float} (column names stripped)."""
    with (run_dir / "results.csv").open() as f:
        return [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def convergence(rows: list[dict], window: int = 10) -> dict:
    """Is training still improving? Best epoch, recent slope of val mAP50-95, and loss trends."""
    m = [r["metrics/mAP50-95(B)"] for r in rows]
    best = max(range(len(m)), key=m.__getitem__)
    recent = m[-window:]
    slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)

    def trend(key):
        vals = [r[key] for r in rows[-window:]]
        return "down" if vals[-1] < vals[0] else "up" if vals[-1] > vals[0] else "flat"

    val_loss = [r["val/box_loss"] + r["val/cls_loss"] + r["val/dfl_loss"] for r in rows]
    return {
        "epochs_done": len(rows), "best_epoch": best + 1, "best_map50_95": m[best],
        "last_map50": rows[-1]["metrics/mAP50(B)"], "last_map50_95": m[-1],
        "map50_95_slope_per_epoch_recent": slope,
        "train_loss_trend": {k: trend(k) for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss")},
        "val_loss_trend": "down" if val_loss[-1] < val_loss[-min(window, len(val_loss))] else "up",
        "val_loss_min_epoch": min(range(len(val_loss)), key=val_loss.__getitem__) + 1,
    }
