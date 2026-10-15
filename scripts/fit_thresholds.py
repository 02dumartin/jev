"""Freeze t, v and tau from a full dev run into thresholds.yaml (read by the test run, PLAN.md §8)."""

import argparse
from pathlib import Path

import yaml

from jevgate.core.runio import RunDir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="a full dev run with every system")
    ap.add_argument("--out", type=Path, default=REPO / "thresholds.yaml")
    args = ap.parse_args()
    th = yaml.safe_load(RunDir(args.run_dir).file("thresholds").read_text())
    if th.get("fit_split") != "dev":
        raise SystemExit("refusing: thresholds were not fitted on dev")
    if th.get("max_videos") or th.get("max_frames_per_video"):
        raise SystemExit("refusing: thresholds come from a truncated (smoke) run")
    missing = [k for k in ("t", "v", "tau") if k not in th]
    if missing:
        raise SystemExit(f"refusing: thresholds missing {missing}")
    args.out.write_text(yaml.safe_dump(th, sort_keys=False))
    print(f"wrote {args.out}: t={th['t']}, v={th['v']}, tau={th['tau']} (from {th['fit_run']})")


if __name__ == "__main__":
    main()
