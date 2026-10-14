"""Create the video-level train/val/test split once and freeze it in splits/video_split.yaml."""

import argparse
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.data.brackish import load_gt
from jevgate.data.splits import make_video_split, video_categories, video_stats, write_split

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite an existing split (only before any result)")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config)
    vs = cfg["video_split"]
    out = REPO / vs["file"]
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; the split is frozen (use --force only if no result has used it)")
    classes = [c for c in cfg["classes"] if c != "background"]
    stats = video_stats(cfg, load_gt(cfg))
    assign = make_video_split(stats, video_categories(cfg), classes, vs["fractions"], vs["min_boxes_per_class"],
                              vs["trials"], vs["seed"])
    write_split(out, assign, stats, classes, {k: vs[k] for k in ("fractions", "min_boxes_per_class", "trials", "seed")})
    print(out.read_text().split("videos:")[0])


if __name__ == "__main__":
    main()
