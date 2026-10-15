"""Redraw the dashboard video of a finished demo run (no model calls): uses the current configs/retail_demo.yaml."""

import argparse
from pathlib import Path

from jevgate.core.config import load_plain_config
from jevgate.demo.pipeline import rerender

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--config", default="configs/retail_demo.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    print(rerender(REPO / args.run_dir, load_plain_config(REPO / args.config, args.set), REPO))


if __name__ == "__main__":
    main()
