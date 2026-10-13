"""Run S1-S8 for one config: uv run python scripts/run_pipeline.py --config configs/smoke.yaml"""

import argparse
from pathlib import Path

from jevgate.config import load_config
from jevgate.pipeline import Pipeline
from jevgate.runio import RunDir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["dev", "test"], help="shortcut for --set run.split=...")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--reuse-from", type=Path, help="earlier run folder to copy outputs from")
    ap.add_argument("--reuse", default="", help="comma list of outputs to copy: candidates,vlm,jev")
    args = ap.parse_args()

    overrides = list(args.set) + ([f"run.split={args.split}"] if args.split else [])
    cfg = load_config(REPO / args.config, overrides)
    reuse = {}
    if args.reuse:
        if not args.reuse_from:
            ap.error("--reuse needs --reuse-from")
        source = RunDir(args.reuse_from)
        reuse = {what.strip(): source for what in args.reuse.split(",") if what.strip()}
    run = Pipeline(cfg, REPO, reuse).run()
    print(f"\nrun_id: {run.run_id}\nfolder: {run.path}")


if __name__ == "__main__":
    main()
