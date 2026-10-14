"""Why does a trained YOLO miss one class? Per-video recall of that class, what the missed boxes became,
and a few overlay sheets (GT in green, predictions in yellow) for the worst videos.

uv run python scripts/diagnose_yolo_class.py --config configs/yolo.yaml --cls crab --split val
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from jevgate.core.config import load_config
from jevgate.data.boxes import iou_matrix
from jevgate.data.brackish import image_path, load_gt, parse_frame_name
from jevgate.detectors.yolo import detector_classes

REPO = Path(__file__).resolve().parents[1]


def predict(weights: Path, files: list[Path], cfg: dict, conf: float) -> dict[str, list]:
    """file name -> [(class, score, xyxy)] for every prediction above `conf`."""
    from ultralytics import YOLO

    model, y = YOLO(str(weights)), cfg["yolo"]
    out = {}
    for s in range(0, len(files), 64):
        for f, r in zip(files[s:s + 64], model.predict([str(p) for p in files[s:s + 64]], imgsz=y["imgsz"],
                                                         conf=conf, device=y["device"], verbose=False), strict=True):
            out[f.name] = [(r.names[int(c)], float(p), b) for c, p, b in
                           zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist(), strict=True)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--cls", default="crab")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25, help="prediction score cut for counting a detection")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--sheets", type=int, default=3, help="overlay frames per worst video")
    args = ap.parse_args()

    cfg = load_config(REPO / args.config)
    gt = load_gt(cfg)
    run_dir = REPO / cfg["yolo"]["project"] / cfg["yolo"]["name"]
    out_dir = run_dir / f"diagnose_{args.cls}_{args.split}"
    out_dir.mkdir(exist_ok=True)
    videos = set(yaml.safe_load((REPO / cfg["video_split"]["file"]).read_text())["videos"][args.split])
    files = sorted(f for f in gt.boxes if parse_frame_name(f)[0] in videos)
    preds = predict(run_dir / "weights" / "best.pt", [image_path(cfg, f) for f in files], cfg, args.conf)

    per_video = defaultdict(lambda: {"gt": 0, "hit": 0, "as_other": Counter(), "missed": 0})
    worst_frames = defaultdict(list)
    for f in files:
        video = parse_frame_name(f)[0]
        g = [b for b in gt.frame_boxes(f) if b.label == args.cls]
        if not g:
            continue
        p = preds.get(f, [])
        ious = iou_matrix([b.box_xyxy for b in g], [b for _, _, b in p])
        for i in range(len(g)):
            row = per_video[video]
            row["gt"] += 1
            same = [j for j in range(len(p)) if p[j][0] == args.cls and ious[i, j] >= args.iou]
            other = [j for j in range(len(p)) if p[j][0] != args.cls and ious[i, j] >= args.iou]
            if same:
                row["hit"] += 1
            elif other:
                row["as_other"][p[max(other, key=lambda j: ious[i, j])][0]] += 1
            else:
                row["missed"] += 1
        worst_frames[video].append(f)

    summary = {v: {"recall": r["hit"] / r["gt"], **{k: r[k] for k in ("gt", "hit", "missed")},
                   "as_other": dict(r["as_other"])} for v, r in per_video.items()}
    summary = dict(sorted(summary.items(), key=lambda kv: kv[1]["recall"]))
    (out_dir / "per_video.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.cls} on {args.split} (conf >= {args.conf}, IoU >= {args.iou})")
    for v, r in summary.items():
        print(f"  recall {r['recall']:.2f}  gt {r['gt']:5d}  missed {r['missed']:5d}  as other {r['as_other']}  {v}")
    total = {k: sum(r[k] for r in summary.values()) for k in ("gt", "hit", "missed")}
    print(f"  overall recall {total['hit'] / total['gt']:.3f} ({total})")

    classes = detector_classes(cfg)
    for v in list(summary)[:3]:
        frames = worst_frames[v]
        for f in [frames[k] for k in np.linspace(0, len(frames) - 1, args.sheets).astype(int)]:
            img = cv2.imread(str(image_path(cfg, f)))
            for b in gt.frame_boxes(f):
                x1, y1, x2, y2 = map(int, b.box_xyxy)
                colour = (80, 220, 80) if b.label == args.cls else (160, 160, 160)
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(img, b.label, (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
            for c, s, b in preds.get(f, []):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 1)
                cv2.putText(img, f"{c} {s:.2f}", (x1, min(img.shape[0] - 3, y2 + 11)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
            cv2.imwrite(str(out_dir / f"{f[:-4]}.jpg"), img)
    print(f"overlays and per_video.json: {out_dir}  (classes: {classes})")


if __name__ == "__main__":
    main()
