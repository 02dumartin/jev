"""Write data/retail-demo/retailaction/meta.jsonl: upright rotation (from pose) and events per selected clip."""

import argparse
from pathlib import Path

import cv2

from jevgate.core.config import load_plain_config
from jevgate.core.runio import write_jsonl
from jevgate.demo.meta import build_meta

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/retail_demo.yaml")
    args = ap.parse_args()
    cfg = load_plain_config(REPO / args.config)
    root = REPO / cfg["data"]["root"]

    def n_frames(sid):
        return int(cv2.VideoCapture(str(root / cfg["data"]["clips_dir"] / f"{sid}.mp4")).get(cv2.CAP_PROP_FRAME_COUNT))

    rows = build_meta(cfg | {"data": cfg["data"] | {"root": str(root)}}, n_frames)
    write_jsonl(root / cfg["data"]["meta"], rows)
    rot = {d: sum(r["rotation_cw"] == d for r in rows) for d in (0, 90, 180, 270)}
    print(f"{len(rows)} clips, rotation counts {rot}, without pose axis {sum(r['pose_axis'] is None for r in rows)}")


if __name__ == "__main__":
    main()
