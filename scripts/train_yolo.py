"""Train YOLO12 on the video split (outputs in train/<name>/)."""

import argparse
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.detectors.yolo import train

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    args = ap.parse_args()
    train(load_config(REPO / args.config), REPO)


if __name__ == "__main__":
    main()
