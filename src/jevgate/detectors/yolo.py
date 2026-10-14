"""YOLO12 detector: dataset conversion (AAU -> YOLO format on the video split), training, progress, test eval."""

import csv
import os
from pathlib import Path

import yaml

from jevgate.data.brackish import BrackishGT, image_path, parse_frame_name
from jevgate.data.splits import SPLITS, listed_frames, load_split


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


def frame_assignment(cfg: dict, repo: Path) -> dict[str, str]:
    """frame file -> train/val/test, from `yolo.split_source`.

    video:    whole videos assigned by splits/video_split.yaml (no video in two splits).
    original: the dataset's own frame-level random lists (yolo.original_lists), as in Pedersen et al. 2019.
    """
    source = cfg["yolo"]["split_source"]
    if source == "video":
        videos = {v: s for s, vids in load_split(repo / cfg["video_split"]["file"]).items() for v in vids}
        return {f: videos[parse_frame_name(f)[0]] for f in listed_frames(cfg)}
    if source == "original":
        root = Path(cfg["data"]["root"])
        return {line.strip(): split for split, rel in cfg["yolo"]["original_lists"].items()
                for line in (root / rel).open() if line.strip()}
    raise ValueError(f"unknown yolo.split_source: {source}")


def build_dataset(cfg: dict, gt: BrackishGT, repo: Path) -> dict:
    """images/<split>/ (symlinks) + labels/<split>/*.txt + data.yaml. Returns per-split counts."""
    root = repo / cfg["yolo"]["dataset_dir"]
    classes = detector_classes(cfg)
    counts = {s: {"frames": 0, "empty_frames": 0, "boxes": 0} for s in SPLITS}
    for f, split in sorted(frame_assignment(cfg, repo).items()):
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
                       patience=y["patience"], freeze=y["freeze"],
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


def evaluate_split(cfg: dict, repo: Path, split: str, weights: Path) -> dict:
    """Ultralytics validation of `weights` on one split; plots go to train/<name>/<split>_eval/."""
    from ultralytics import YOLO

    y = cfg["yolo"]
    run_dir = repo / y["project"] / y["name"]
    res = YOLO(str(weights)).val(data=str(repo / y["dataset_dir"] / "data.yaml"), split=split, imgsz=y["imgsz"],
                                 batch=y["batch"], device=y["device"], project=str(run_dir), name=f"{split}_eval",
                                 exist_ok=True, plots=True, verbose=False)
    names = res.names
    per_class = {}
    for i, c in enumerate(res.ap_class_index):
        p, r, ap50, ap = res.box.class_result(i)
        per_class[names[int(c)]] = {"precision": float(p), "recall": float(r), "ap50": float(ap50),
                                    "ap50_95": float(ap)}
    return {"split": split, "weights": str(weights.relative_to(repo)),
            "precision": float(res.box.mp), "recall": float(res.box.mr), "map50": float(res.box.map50),
            "map50_95": float(res.box.map), "per_class": per_class,
            "speed_ms_per_image": {k: float(v) for k, v in res.speed.items()}}


def training_report(cfg: dict, run_dir: Path, conv: dict, evals: list[dict]) -> str:
    """train/<name>/README.md: convergence, per-split detector metrics, and a guide to the files."""
    def f(v):
        return f"{v:.3f}"

    train_trend = ", ".join(f"{k.split('/')[1]} {v}" for k, v in conv["train_loss_trend"].items())
    lines = [f"# {run_dir.name}", "",
             f"YOLO12s, COCO-pretrained, split `{cfg['yolo']['split_source']}`, imgsz {cfg['yolo']['imgsz']}, "
             f"freeze {cfg['yolo']['freeze']}, epochs {cfg['yolo']['epochs']} (patience {cfg['yolo']['patience']}). "
             f"{cfg['yolo'].get('note', '')}", "",
             "## Convergence", "",
             f"- epochs {conv['epochs_done']}, best epoch **{conv['best_epoch']}** "
             f"(val mAP50-95 {f(conv['best_map50_95'])}), "
             f"last epoch mAP50 {f(conv['last_map50'])} / mAP50-95 {f(conv['last_map50_95'])}",
             f"- train losses: {train_trend}; val loss lowest at epoch {conv['val_loss_min_epoch']}, "
             f"trend over the last 10 epochs: {conv['val_loss_trend']}",
             "- ![progress](progress.png)", ""]
    for e in evals:
        lines += [f"## {e['split']} (`{e['weights']}`)", "",
                  f"P {f(e['precision'])} · R {f(e['recall'])} · mAP50 **{f(e['map50'])}** · "
                  f"mAP50-95 **{f(e['map50_95'])}** · "
                  f"inference {e['speed_ms_per_image'].get('inference', 0):.1f} ms/image", "",
                  "| class | precision | recall | AP50 | AP50-95 |", "|---|---|---|---|---|"]
        lines += [f"| {c} | {f(v['precision'])} | {f(v['recall'])} | {f(v['ap50'])} | {f(v['ap50_95'])} |"
                  for c, v in e["per_class"].items()]
        lines += ["", f"![confusion]({e['split']}_eval/confusion_matrix_normalized.png)", ""]
    lines += ["## Files", "",
              "| file | content |", "|---|---|",
              "| `weights/best.pt` | weights of the best val epoch (used downstream) |",
              "| `results.csv`, `results.png` | Ultralytics per-epoch losses and val metrics |",
              "| `progress.json`, `progress.png` | convergence summary (best epoch, recent slope, loss trends) |",
              "| `<split>_eval/` | confusion matrix and PR curves on that split |",
              "| `args.yaml` | full training arguments |"]
    return "\n".join(lines) + "\n"
