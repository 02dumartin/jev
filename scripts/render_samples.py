"""⑤ Material for the human check: n random frame overlays and a Before -> After video per clip."""

import argparse
from pathlib import Path

from jevgate.runio import RunDir
from jevgate.viz import render_samples, render_video


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-system", default="gate", help="system whose kept boxes form the After panel")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    print(f"overlays: {render_samples(run, args.n, args.seed)}")
    if args.video_system in run.systems_done():
        for clip in sorted({f["clip"] for f in run.read("frames")}):
            print(f"video: {render_video(run, clip, args.video_system, args.fps)}")


if __name__ == "__main__":
    main()
