"""Run the retail demo: uv run python scripts/run_retail.py [--set run.max_clips=5]"""

import argparse
from pathlib import Path

from jevgate.core.config import load_config
from jevgate.pipeline import RetailRun

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/retail.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    print(RetailRun(load_config(REPO / args.config, args.set), REPO).run())


if __name__ == "__main__":
    main()
