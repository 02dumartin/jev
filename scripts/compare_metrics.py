"""④ Regression check of a dev run against the last row of metrics_history.jsonl (PLAN.md §8).

Exit code 1 when accuracy/macro F1/precision/recall dropped, or ECE rose, by more than metrics.regression_tol.
With --append the run is added to the history (do it only after a pass or an explained drop).
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from jevgate.core.config import config_hash
from jevgate.core.runio import RunDir

REPO = Path(__file__).resolve().parents[1]
HIGHER_IS_BETTER = ["accuracy", "macro_f1", "macro_precision", "macro_recall"]
LOWER_IS_BETTER = ["ece"]
COLUMNS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "ece", "vlm_calls", "latency_ms_median",
           "jev_cost_usd"]


def system_columns(row: dict, ece) -> dict:
    vals = {k: row.get(k) for k in COLUMNS}
    for key in ("precision", "recall"):
        per = [v[key] for v in row["per_class"].values() if v["support"] and v[key] is not None]
        vals[f"macro_{key}"] = sum(per) / len(per) if per else None
    vals["ece"] = ece
    return vals


def history_row(run: RunDir) -> dict:
    m = run.read_json("metrics")
    cfg = run.config()
    ece = m["track_level"].get("jev_ece")
    return {
        "date": dt.date.today().isoformat(), "run_id": run.run_id, "git_commit": run.git().get("commit"),
        "git_dirty": run.git().get("dirty"), "config_hash": config_hash(cfg), "split": m["split"],
        "videos": run.file("videos").read_text().split(), "max_videos": cfg["run"].get("max_videos"),
        "max_frames_per_video": cfg["run"].get("max_frames_per_video"),
        "n_box": m["n"]["boxes_scored"], "n_track": m["n"]["tracks"], "thresholds": m["thresholds"],
        "systems": {s: system_columns(row, ece if s == "jev" else None) for s, row in m["systems"].items()},
    }


def regressions(new: dict, old: dict, tol: float) -> list[str]:
    out = []
    for system, row in new["systems"].items():
        prev = old["systems"].get(system)
        if not prev:
            continue
        for k in HIGHER_IS_BETTER:
            if None not in (row[k], prev[k]) and row[k] < prev[k] - tol:
                out.append(f"{system}.{k}: {prev[k]:.4f} -> {row[k]:.4f}")
        for k in LOWER_IS_BETTER:
            if None not in (row[k], prev[k]) and row[k] > prev[k] + tol:
                out.append(f"{system}.{k}: {prev[k]:.4f} -> {row[k]:.4f}")
    return out


def fmt(v):
    return "–" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--history", type=Path, default=REPO / "metrics_history.jsonl")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    new = history_row(run)
    if new["split"] != "dev" or new["max_videos"] or new["max_frames_per_video"]:
        sys.exit("regression history only takes full dev runs")
    tol = run.config()["metrics"]["regression_tol"]
    rows = [json.loads(line) for line in args.history.open()] if args.history.exists() else []
    comparable = [r for r in rows if r.get("videos") == new["videos"]]
    old = comparable[-1] if comparable else None

    print(f"{'system':10s} " + " ".join(f"{c:>18s}" for c in COLUMNS))
    for system, row in new["systems"].items():
        prev = old["systems"].get(system, {}) if old else {}
        cells = [f"{fmt(prev.get(c))}->{fmt(row[c])}" if old else fmt(row[c]) for c in COLUMNS]
        print(f"{system:10s} " + " ".join(f"{c:>18s}" for c in cells))

    found = regressions(new, old, tol) if old else []
    print(f"\nbaseline: {old['run_id'] if old else 'none (first entry)'}")
    print("regressions: " + ("; ".join(found) if found else "none"))
    if args.append:
        with args.history.open("a") as f:
            f.write(json.dumps(new, ensure_ascii=False) + "\n")
        print(f"appended {run.run_id} to {args.history.name}")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
