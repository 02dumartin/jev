"""Figures under evaluation/ (PLAN.md v6 §7).

confusion/<system>.png  7x7 (six animals + background), rows = truth, row-normalised % with counts
plots/roc.png           track level: which score separates correct from wrong tracks? AUROC [95% CI]
plots/reliability.png   Jev p_ok vs the observed share of correct tracks (ECE)
plots/tau_curve.png     Jev-handled share, error on them and VLM call share against tau
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TITLES = {"yolo": "① YOLO (per frame)", "track_vote": "② track majority", "vlm_all": "③ VLM on every track",
          "jev": "④ Jev", "gate": "⑤ Jev gate → VLM"}
SCORES = {"jev_p_ok": ("Jev p_ok", "#4C78A8", "-"), "vlm_p_yes": ("VLM p_yes", "#E45756", "--"),
          "mean_conf": ("YOLO mean conf", "#888888", ":")}


def confusion(path: Path, system: str, conf: dict) -> None:
    classes, cm = conf["classes"], np.asarray(conf["matrix"], dtype=float)
    rows = cm.sum(1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.0%}\n{int(cm[i, j])}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.6 else "#222")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), [f"{c} (n={int(r)})" for c, r in zip(classes, rows[:, 0], strict=True)],
                  fontsize=8)
    ax.set(xlabel="predicted", ylabel="truth (box)", title=TITLES.get(system, system))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def roc(path: Path, curves: dict, track_level: dict, operating: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color="#bbb", lw=1, ls="--")
    for name, pts in curves.items():
        label, color, ls = SCORES[name]
        a = track_level.get(f"auroc_{name}", {})
        ci = f" [{a['ci_low']:.2f}, {a['ci_high']:.2f}]" if a.get("ci_low") is not None else ""
        auc = f"AUC {a['auc']:.3f}{ci}" if a.get("auc") is not None else "AUC n/a (one class only)"
        ax.plot(pts["fpr"], pts["tpr"], color=color, ls=ls, lw=2, label=f"{label}  {auc}")
        if name in operating:
            fpr, tpr = operating[name]
            ax.plot([fpr], [tpr], "o", color=color, ms=6, mec="white")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="false positive rate (wrong tracks accepted)",
           ylabel="true positive rate (correct tracks accepted)",
           title=f"Track level ROC (n={track_level['n_tracks_scored']})")
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def reliability(path: Path, bins: list[dict], ece: float | None) -> None:
    used = [b for b in bins if b["count"]]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect calibration")
    ax.bar([(b["lo"] + b["hi"]) / 2 for b in used], [b["accuracy"] for b in used], width=0.9 / len(bins),
           color="#4C78A8", alpha=0.85, label="observed share of correct tracks")
    for b in used:
        ax.annotate(str(b["count"]), ((b["lo"] + b["hi"]) / 2, b["accuracy"]), ha="center", va="bottom", fontsize=6)
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Jev p_ok", ylabel="share of tracks that are correct",
           title=f"Jev calibration (ECE {ece:.3f})" if ece is not None else "Jev calibration")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tau(path: Path, rows: list[dict], tau_value: float | None, max_err: float) -> None:
    rows = [r for r in rows if r["n_auto"]]
    t = [r["tau"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(t, [r["auto_rate"] for r in rows], color="#4C78A8", label="Jev-handled share")
    ax.plot(t, [r["auto_error"] for r in rows], color="#E45756", label="error on Jev-handled")
    ax.plot(t, [1 - r["auto_rate"] for r in rows], color="#72B7B2", ls=":", label="VLM call share")
    ax.axhline(max_err, color="#E45756", lw=0.8, ls="--", alpha=0.6)
    if tau_value is not None and tau_value <= 1:
        ax.axvline(tau_value, color="#333", lw=1, ls="--", label=f"tau = {tau_value:.2f} (dev)")
    ax.set(xlabel="tau", ylabel="share", ylim=(0, 1.02), title="Gate operating curve")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def operating_points(metrics: dict, curves: dict, run_data) -> dict:
    """(fpr, tpr) of the actual decisions: Jev at 0.5, VLM at v."""
    out = {}
    ys = {u: r["y_track"] for u, r in run_data.track_label.items() if r["y_track"] is not None}
    for name, src, thr, key in (("jev_p_ok", run_data.jev, 0.5, "p_ok"), ("vlm_p_yes", run_data.vlm,
                                                                          metrics["thresholds"].get("v"), "p_yes")):
        if not src or thr is None:
            continue
        acc = {u: src[u][key] >= thr for u in ys}
        pos = [u for u, y in ys.items() if y]
        neg = [u for u, y in ys.items() if not y]
        if pos and neg:
            out[name] = (np.mean([acc[u] for u in neg]), np.mean([acc[u] for u in pos]))
    return out


def all_figures(run, metrics: dict, curves: dict, run_data) -> None:
    for system, conf in curves.get("confusion", {}).items():
        confusion(run.dir("confusion") / f"{system}.png", system, conf)
    tl = metrics["track_level"]
    plots = run.dir("plots")
    if curves.get("roc"):
        roc(plots / "roc.png", curves["roc"], tl, operating_points(metrics, curves, run_data))
    if curves.get("reliability_jev"):
        reliability(plots / "reliability.png", curves["reliability_jev"], tl.get("jev_ece"))
    if curves.get("tau_curve"):
        tau(plots / "tau_curve.png", curves["tau_curve"], metrics["thresholds"].get("tau"),
            run.config()["gate"]["max_auto_error"])
