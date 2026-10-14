"""⑤ Material for the human check: n random frames rendered per view (GT, DINO, each system) + compare sheet."""

import argparse
from pathlib import Path

from jevgate.core.runio import RunDir
from jevgate.data.brackish import load_gt
from jevgate.outputs.viz import render_samples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    print(f"overlays: {render_samples(run, load_gt(run.config()), args.n, args.seed)}")


if __name__ == "__main__":
    main()
