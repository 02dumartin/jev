"""S7 figures (PLAN.md §3.3): per-class ROC with AUC [CI], one-vs-rest 2x2 grids, confusion matrices,
reliability diagrams, gate curves, and a paper-style main figure (A/B ROC, C/D 2x2).

Class colours follow a fixed categorical order (validated palette); background is neutral grey.
Every class also has its own marker and a legend label, so identity never relies on colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .evaluate import SYSTEM_TITLE  # noqa: E402
from .runio import RunDir  # noqa: E402

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BACKGROUND_COLOR = "#8a8985"
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("cells", ["#f4f7fb", "#c5d6ea", "#5f87b4", "#1f3a5f"])

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2,
                     "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False})


def class_style(classes: list[str]) -> dict[str, dict]:
    """Fixed colour + marker per class in config order; background last and grey."""
    out, k = {}, 0
    for i, c in enumerate(classes):
        if c == "background":
            out[c] = {"color": BACKGROUND_COLOR, "marker": MARKERS[i % len(MARKERS)], "ls": "--"}
        else:
            out[c] = {"color": PALETTE[k % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)], "ls": "-"}
            k += 1
    return out


def _fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def draw_roc(ax, roc: dict, auc: dict, points: dict, classes: list[str], min_n: int, title: str):
    """One ROC panel: a curve per class, operating point marker, legend 'class  AUC [lo, hi]'."""
    style = class_style(classes)
    ax.plot([0, 1], [0, 1], color="#b8b7b1", lw=1, ls=(0, (3, 3)))
    for c in classes:
        a = auc[c]
        if a["auc"] is None:
            continue
        st = style[c]
        alpha = 1.0 if a["n_pos"] >= min_n else 0.35
        label = f"{c:<11s} {_fmt(a['auc'])} [{_fmt(a['ci_low'])}, {_fmt(a['ci_high'])}]  n={a['n_pos']}"
        ax.plot(roc[c]["fpr"], roc[c]["tpr"], color=st["color"], lw=2, ls=st["ls"], alpha=alpha, label=label,
                marker=st["marker"], markevery=[], ms=6)
        p = points.get(c, {})
        if p.get("fpr") is not None and p.get("tpr") is not None:
            ax.plot(p["fpr"], p["tpr"], st["marker"], color=st["color"], ms=8, mec="white", mew=1.5, alpha=alpha)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="false positive rate", ylabel="true positive rate")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    leg = ax.legend(loc="lower right", fontsize=7, frameon=False, prop={"family": "monospace", "size": 7},
                    handlelength=2.2)
    for text in leg.get_texts():
        text.set_color(INK)


def draw_2x2(ax, counts: dict, cls: str, color: str, greyed: bool):
    """One-vs-rest 2x2 cell block: counts + row %, sens/spec below, class title in its colour."""
    m = np.array([[counts["tp"], counts["fn"]], [counts["fp"], counts["tn"]]], dtype=float)
    row = m / np.maximum(m.sum(1, keepdims=True), 1)
    ax.imshow(row, cmap=BLUES, vmin=0, vmax=1, alpha=0.35 if greyed else 1.0)
    for i in range(2):
        for j in range(2):
            dark = row[i, j] > 0.55 and not greyed
            ax.text(j, i - 0.08, f"{int(m[i, j])}", ha="center", va="center", fontsize=10,
                    color="white" if dark else INK, fontweight="bold")
            ax.text(j, i + 0.25, f"{row[i, j] * 100:.0f}%", ha="center", va="center", fontsize=7,
                    color="white" if dark else INK_2)
    ax.set_xticks([0, 1], ["pred +", "pred −"], fontsize=7)
    ax.set_yticks([0, 1], ["GT +", "GT −"], fontsize=7)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(cls + ("  (n<min)" if greyed else ""), color=color, fontsize=10, pad=18)
    sens, spec = counts["sens"], counts["spec"]
    ax.set_xlabel(f"sens {_fmt(sens * 100 if sens is not None else None, 0)}%   "
                  f"spec {_fmt(spec * 100 if spec is not None else None, 0)}%", fontsize=7, color=INK_2)


def draw_2x2_grid(fig_or_sub, ovr: dict, support: dict, classes: list[str], min_n: int, title: str):
    style = class_style(classes)
    cols = 4
    axes = fig_or_sub.subplots(2, cols)
    for k, ax in enumerate(axes.flat):
        if k >= len(classes):
            ax.axis("off")
            continue
        c = classes[k]
        draw_2x2(ax, ovr[c], c, style[c]["color"], support[c] < min_n)
    fig_or_sub.suptitle(title, x=0.02, ha="left", fontsize=10, color=INK)


def draw_confusion(ax, conf: dict, title: str):
    cm = np.asarray(conf["matrix"], dtype=float)
    row = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(row, cmap=BLUES, vmin=0, vmax=1)
    n = len(conf["classes"])
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, f"{int(cm[i, j])}\n{row[i, j] * 100:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if row[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), conf["classes"], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(n), conf["classes"], fontsize=7)
    ax.set(xlabel="predicted", ylabel="ground truth (y)")
    ax.set_title(title, loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_reliability(ax, bins: list[dict], ece, title: str):
    used = [b for b in bins if b["count"]]
    width = (bins[0]["hi"] - bins[0]["lo"]) * 0.85
    ax.plot([0, 1], [0, 1], color="#b8b7b1", lw=1, ls=(0, (3, 3)), label="perfect calibration")
    ax.bar([(b["lo"] + b["hi"]) / 2 for b in used], [b["accuracy"] for b in used], width=width,
           color=PALETTE[0], alpha=0.85, label="observed accuracy")
    for b in used:
        ax.annotate(str(b["count"]), ((b["lo"] + b["hi"]) / 2, b["accuracy"]), fontsize=6, ha="center",
                    va="bottom", color=INK_2)
    ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="confidence (top-class probability)", ylabel="accuracy")
    ax.set_title(f"{title}   ECE {_fmt(ece, 3)}", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")


def draw_auto_curve(ax, curve: list[dict], tau, xlabel: str, title: str):
    rows = [r for r in curve if r["n_auto"]]
    x = [r["tau"] for r in rows]
    ax.plot(x, [r["auto_rate"] for r in rows], color=PALETTE[0], lw=2, label="auto-handled share")
    ax.plot(x, [r["auto_error"] for r in rows], color=PALETTE[1], lw=2, label="error on auto-handled")
    if tau is not None:
        ax.axvline(tau, color=INK_2, lw=1, ls="--", label=f"τ = {tau:.2f} (dev)")
    ax.set(ylim=(0, 1.02), xlabel=xlabel, ylabel="share")
    ax.grid(color=GRID, lw=0.6)
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)


def plot_run(run: RunDir) -> list[Path]:
    """Write every figure for a run into plots/. Returns the written paths."""
    m = run.read_json("metrics.json")
    data = run.read_json("plots_data.json")
    cfg = run.config()
    classes, min_n = m["classes"], cfg["metrics"]["min_class_n"]
    out = run.path / "plots"
    out.mkdir(exist_ok=True)
    written = []

    def save(fig, name):
        fig.savefig(out / name, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(out / name)

    n_tag = f"n = {m['n_box']}"
    for s, row in m["systems"].items():
        support = {c: row["per_class"][c]["support"] for c in classes}
        if s in data["roc"]:
            fig, ax = plt.subplots(figsize=(5.2, 5))
            draw_roc(ax, data["roc"][s], row["auc"], data["operating_points"][s], classes, min_n,
                     f"{SYSTEM_TITLE[s]}   {n_tag}")
            save(fig, f"roc_{s}.png")
        fig = plt.figure(figsize=(9, 5.2))
        draw_2x2_grid(fig, row["ovr"], support, classes, min_n, f"{SYSTEM_TITLE[s]}   {n_tag}")
        fig.subplots_adjust(hspace=0.7, wspace=0.45)
        save(fig, f"ovr_2x2_{s}.png")
        fig, ax = plt.subplots(figsize=(5.6, 5))
        draw_confusion(ax, row["confusion"], f"{SYSTEM_TITLE[s]}   acc {_fmt(row['accuracy'], 3)}")
        save(fig, f"confusion_{s}.png")
        if s in data["reliability"]:
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            draw_reliability(ax, data["reliability"][s], row["ece"], SYSTEM_TITLE[s])
            save(fig, f"reliability_{s}.png")

    th = m.get("thresholds") or {}
    if data["auto_curves"]:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
        draw_auto_curve(axes[0], data["auto_curves"]["need_vlm"], th.get("tau_need_vlm"),
                        "τ_e: send to VLM when need_vlm level ≥ τ_e", "Gate on Jev need_vlm (primary)")
        draw_auto_curve(axes[1], data["auto_curves"]["confidence"], th.get("tau_conf"),
                        "τ_c: send to VLM when Jev confidence < τ_c", "Gate on Jev confidence")
        fig.tight_layout()
        save(fig, "gate_curves.png")

    primary = cfg["gate"]["primary"]
    if {"jev", "vlm"} <= set(data["roc"]) and primary in m["systems"]:
        save(main_figure(m, data, classes, min_n, primary), "figure_main.png")
    return written


def main_figure(m: dict, data: dict, classes: list[str], min_n: int, primary: str):
    """A: ③ Jev ROC, B: ② VLM ROC, C: ③ Jev 2x2 per class, D: ④ primary gate 2x2 per class."""
    fig = plt.figure(figsize=(14, 12.5))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    a_ax, b_ax = top.subplots(1, 2)
    n_tag = f"n = {m['n_box']}"
    for ax, s, letter in ((a_ax, "jev", "A"), (b_ax, "vlm", "B")):
        draw_roc(ax, data["roc"][s], m["systems"][s]["auc"], data["operating_points"][s], classes, min_n,
                 f"{letter}   {SYSTEM_TITLE[s]}   {n_tag}")
    c_sub, d_sub = bottom.subfigures(1, 2)
    for sub, s, letter in ((c_sub, "jev", "C"), (d_sub, primary, "D")):
        row = m["systems"][s]
        draw_2x2_grid(sub, row["ovr"], {c: row["per_class"][c]["support"] for c in classes}, classes, min_n,
                      f"{letter}   {SYSTEM_TITLE[s]}   {n_tag}")
        sub.subplots_adjust(hspace=0.75, wspace=0.5)
    return fig
