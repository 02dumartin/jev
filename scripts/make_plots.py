"""Redraw the S8 outputs of a finished run from its files: figures/, viz/ and README.md (no model calls)."""

import argparse
import shutil
from pathlib import Path

from jevgate import plots, report, viz
from jevgate.data import load_gt
from jevgate.runio import RunDir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    cfg = run.config()
    for old in ("figures", "viz", "samples"):
        shutil.rmtree(run.path / old, ignore_errors=True)
    figures = plots.plot_run(run)
    viz.render_samples(run, load_gt(cfg), cfg["viz"]["sample_frames"], cfg["viz"]["seed"])
    run.file("report").write_text(report.render(run, run.read_json("metrics"), figures))
    print(f"redrew {len(figures)} figures, viz/ and README.md in {run.path}")


if __name__ == "__main__":
    main()
