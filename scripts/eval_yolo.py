"""Evaluate the trained YOLO (best.pt) on val and test, and write train/<name>/README.md + eval.json."""

import argparse
import json
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.detectors.yolo import convergence, evaluate_split, read_results, training_report

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    cfg = load_config(REPO / args.config)
    run_dir = REPO / cfg["yolo"]["project"] / cfg["yolo"]["name"]
    evals = [evaluate_split(cfg, REPO, s, run_dir / "weights" / "best.pt") for s in args.splits]
    (run_dir / "eval.json").write_text(json.dumps(evals, indent=2))
    (run_dir / "README.md").write_text(training_report(cfg, run_dir, convergence(read_results(run_dir)), evals))
    for e in evals:
        print(f"{e['split']}: mAP50 {e['map50']:.3f} mAP50-95 {e['map50_95']:.3f}")


if __name__ == "__main__":
    main()
