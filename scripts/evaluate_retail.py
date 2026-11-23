"""Offline Retail metrics, calibration, and report from a saved run (no model calls)."""

import argparse
from pathlib import Path

from jev.evaluation.report import evaluate_run

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    run_dir = REPO / args.run_dir
    output = REPO / args.output if args.output else run_dir / "evaluation/calibration"
    result = evaluate_run(run_dir, output, args.bins)
    print(f"{result['product_taken']['n']} clips evaluated: {output / 'report.md'}")


if __name__ == "__main__":
    main()
