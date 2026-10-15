"""Run the retail demo: uv run python scripts/run_retail_demo.py [--set run.max_clips=5]"""

import argparse
from pathlib import Path

from jevgate.core.config import load_plain_config
from jevgate.demo.pipeline import DemoRun

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/retail_demo.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    print(DemoRun(load_plain_config(REPO / args.config, args.set), REPO).run())


if __name__ == "__main__":
    main()
