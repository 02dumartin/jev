"""Freeze t and tau from a full dev run into thresholds.yaml (read by test runs, PLAN.md §7)."""

import argparse
from pathlib import Path

import yaml

from jevgate.runio import RunDir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="a dev run with all systems")
    ap.add_argument("--out", type=Path, default=REPO / "thresholds.yaml")
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    th = run.read_json("thresholds.json")
    if th.get("fit_split") != "dev":
        raise SystemExit("refusing: thresholds were not fitted on dev")
    if th.get("max_frames_per_clip"):
        raise SystemExit("refusing: thresholds come from a truncated (smoke) run")
    if "tau" not in th:
        raise SystemExit("refusing: run has no Jev decisions, tau missing")
    args.out.write_text(yaml.safe_dump(th, sort_keys=False))
    print(f"wrote {args.out}: t={th['t']}, tau={th['tau']} (from {run.run_id})")


if __name__ == "__main__":
    main()
