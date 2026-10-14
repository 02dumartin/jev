"""Summarise a YOLO training run: convergence numbers and a loss/mAP-per-epoch figure (train/<name>/progress.*)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jevgate.detectors.yolo import convergence, read_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = read_results(args.run_dir)
    conv = convergence(rows)
    (args.run_dir / "progress.json").write_text(json.dumps(conv, indent=2))
    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for k, c in (("box", "#2a78d6"), ("cls", "#eb6834"), ("dfl", "#1baf7a")):
        axes[0].plot(ep, [r[f"train/{k}_loss"] for r in rows], color=c, lw=2, label=f"train {k}")
        axes[1].plot(ep, [r[f"val/{k}_loss"] for r in rows], color=c, lw=2, label=f"val {k}")
    axes[2].plot(ep, [r["metrics/mAP50(B)"] for r in rows], color="#2a78d6", lw=2, label="val mAP50")
    axes[2].plot(ep, [r["metrics/mAP50-95(B)"] for r in rows], color="#eb6834", lw=2, label="val mAP50-95")
    axes[2].axvline(conv["best_epoch"], color="#52514e", ls="--", lw=1, label=f"best epoch {conv['best_epoch']}")
    for ax, t in zip(axes, ("train loss", "val loss", "val mAP"), strict=True):
        ax.set(title=t, xlabel="epoch")
        ax.grid(color="#e4e3df", lw=0.6)
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.run_dir / "progress.png", dpi=150, facecolor="white")
    print(json.dumps(conv, indent=2))


if __name__ == "__main__":
    main()
