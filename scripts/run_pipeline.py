"""Run one experiment (PLAN.md v6): uv run python scripts/run_pipeline.py --config configs/smoke.yaml"""

import argparse
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.pipeline import Pipeline

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()
    run = Pipeline(load_config(REPO / args.config, args.set), REPO).run()
    print(f"\nrun_id: {run.run_id}\nfolder: {run.path}")


if __name__ == "__main__":
    main()
