"""Redraw every S7 figure of a finished run from its metrics.json and plots_data.json."""

import argparse
from pathlib import Path

from jevgate.plots import plot_run
from jevgate.runio import RunDir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    for path in plot_run(RunDir(args.run_dir)):
        print(path)


if __name__ == "__main__":
    main()
