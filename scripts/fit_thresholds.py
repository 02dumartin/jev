"""Freeze t, tau_need_vlm and tau_conf from a full dev run into thresholds.yaml (read by test runs, PLAN.md §7)."""

import argparse
from pathlib import Path

import yaml

from jevgate.core.runio import RunDir

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="a dev run with all systems")
    ap.add_argument("--out", type=Path, default=REPO / "thresholds.yaml")
    args = ap.parse_args()
    run = RunDir(args.run_dir)
    th = run.meta()["thresholds"]
    if th.get("fit_split") != "dev":
        raise SystemExit("refusing: thresholds were not fitted on dev")
    if th.get("max_frames"):
        raise SystemExit("refusing: thresholds come from a truncated (smoke) run")
    if "tau_need_vlm" not in th:
        raise SystemExit("refusing: run has no Jev decisions, gate thresholds missing")
    args.out.write_text(yaml.safe_dump(th, sort_keys=False))
    fitted = ", ".join(f"{k}={v}" for k, v in th.items() if k == "t" or k.startswith("tau"))
    print(f"wrote {args.out}: {fitted} (from {run.run_id})")


if __name__ == "__main__":
    main()
