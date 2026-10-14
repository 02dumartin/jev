"""Build the YOLO dataset (symlinked images, AAU labels converted to YOLO format) on the frozen video split."""

import argparse
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.data.brackish import load_gt
from jevgate.detectors.yolo import build_dataset

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config)
    for split, c in build_dataset(cfg, load_gt(cfg), REPO).items():
        print(f"{split:5s} frames {c['frames']:6d}  empty {c['empty_frames']:5d}  boxes {c['boxes']:6d}")


if __name__ == "__main__":
    main()
