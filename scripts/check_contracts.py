"""② Data contracts for one run folder (PLAN.md §3). Exit code 1 if any contract is violated."""

import argparse
import sys
from pathlib import Path

from jevgate.contracts import check_run
from jevgate.runio import RunDir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    report = check_run(RunDir(args.run_dir))
    failed = False
    for stage, errors in report.items():
        print(f"{stage:11s} {'OK' if not errors else 'FAIL'}")
        for e in errors[:20]:
            print(f"    - {e}")
        failed |= bool(errors)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
