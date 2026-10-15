"""Render held-out videos with the trained YOLO: per-frame detection vs tracking (train/<name>/tracks/)."""

import argparse
import json
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.detectors.track import ClipTracker, split_videos

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--clips", nargs="*", help="default: every video of track.split")
    args = ap.parse_args()
    cfg = load_config(REPO / args.config)
    run_dir = REPO / cfg["yolo"]["project"] / cfg["yolo"]["name"]
    out_dir = run_dir / cfg["track"]["out_dir"]
    tracker = ClipTracker(cfg, run_dir / "weights" / cfg["track"]["weights"])
    summary = []
    for clip in args.clips or split_videos(cfg, REPO):
        s = tracker.run(clip, out_dir)
        summary.append(s)
        print(f"{clip}: {s['frames']} frames, {s['n_tracks']} tracks (median {s['track_len_median']:.0f} frames), "
              f"flicker detect {s['flicker_detect']:.2f} -> track {s['flicker_track']:.2f}")
    (out_dir / "summary.json").write_text(json.dumps({"config": cfg["track"], "clips": summary}, indent=2))
    print(f"videos: {out_dir}")


if __name__ == "__main__":
    main()
