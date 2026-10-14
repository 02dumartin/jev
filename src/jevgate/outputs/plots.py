"""S8 figures, drawn from metrics.json only. Each main figure answers one question:

  fig1_tradeoff     Does the gate keep accuracy while cutting VLM calls? (primary endpoint)
  fig2_per_class    Where do the systems differ by class? (A/B ROC + AUC [CI], C/D one-vs-rest 2x2)
  fig3_jev_knows    Does Jev know when it is wrong, and how was tau chosen?
  fig4_calibration  Can Jev's and the VLM's probabilities be trusted? (reliability diagrams)

figures/appendix/ (plots.appendix: true) adds per-system ROC, 2x2 grids and confusion matrices.

Class colours follow a fixed categorical order (validated palette); background is neutral grey.
Every class also has its own marker and a legend label, so identity never relies on colour alone.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from jevgate.core.runio import RunDir  # noqa: E402
from jevgate.evaluation.evaluate import SYSTEM_TITLE  # noqa: E402

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
    cols = (len(classes) + 1) // 2
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


SYSTEM_COLOR = {"threshold": PALETTE[3], "vlm": PALETTE[1], "jev": PALETTE[0], "gate": PALETTE[2],
                "gate_conf": PALETTE[4], "gate_either": PALETTE[5]}


def draw_tradeoff(ax, m: dict, step: dict):
    """Accuracy (95% CI) vs share of boxes sent to the VLM; shaded = primary success region."""
    n = m["n_candidates"]
    vlm_acc = m["systems"].get("vlm", {}).get("accuracy")
    if vlm_acc is not None:
        floor = vlm_acc + step["min_delta_acc_ci_low"]
        ax.axhspan(floor, 1.0, xmin=0, xmax=step["max_vlm_call_ratio"] / 1.05, color="#1baf7a", alpha=0.08, lw=0)
        ax.axhline(floor, color="#1baf7a", lw=1, ls=(0, (3, 3)))
        ax.axvline(step["max_vlm_call_ratio"], color="#1baf7a", lw=1, ls=(0, (3, 3)))
        ax.text(0.01, floor + 0.005, f"success region: calls ≤ {step['max_vlm_call_ratio']:.0%}, "
                f"accuracy ≥ ② {step['min_delta_acc_ci_low']:+.2f}", fontsize=7, color=INK_2, va="bottom")
    for s, r in m["systems"].items():
        x, y = r["vlm_calls"] / n, r["accuracy"]
        lo, hi = r.get("accuracy_ci", [None, None])
        if lo is not None:
            ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="none", ecolor=SYSTEM_COLOR[s], elinewidth=2, capsize=0)
        ax.plot(x, y, "o", color=SYSTEM_COLOR[s], ms=9, mec="white", mew=1.5)
        ax.annotate(f"{SYSTEM_TITLE[s]}\n{y:.3f}", (x, y), xytext=(8, -4), textcoords="offset points", fontsize=7.5,
                    color=INK)
    ax.set(xlim=(-0.03, 1.05), xlabel="share of boxes sent to the VLM", ylabel="accuracy (95% CI)")
    ax.grid(color=GRID, lw=0.6)
    p = m["primary"]
    verdict = "; ".join(f"{e['name']}: {e['result']}" for e in m["endpoints"])
    d = p.get("delta_acc") or {}
    ax.set_title(f"Gate vs VLM: Δacc {_fmt(d.get('delta'), 3)} "
                 f"[{_fmt(d.get('ci_low'), 3)}, {_fmt(d.get('ci_high'), 3)}], "
                 f"calls {_fmt(p.get('call_ratio'))}   ({verdict})", loc="left", fontsize=9)


def draw_need_vlm_levels(ax, by_level: dict):
    """Jev error rate per need_vlm level, with box counts."""
    levels = [lv for lv, v in by_level.items() if v["n"]]
    errs = [by_level[lv]["jev_error"] for lv in levels]
    ax.bar(levels, errs, color=PALETTE[0], width=0.6)
    for lv, e in zip(levels, errs, strict=True):
        ax.annotate(f"{e:.0%}\nn={by_level[lv]['n']}", (lv, e), ha="center", va="bottom", fontsize=7, color=INK_2,
                    xytext=(0, 2), textcoords="offset points")
    ax.set(ylim=(0, 1.1), xlabel="Jev need_vlm level (1 = no image check needed, 5 = must look)",
           ylabel="Jev error rate")
    ax.set_title("Jev's own need_vlm vs its actual error", loc="left", fontsize=10)


def plot_run(run: RunDir) -> list[Path]:
    """Write fig1-fig4 (and the appendix when enabled) into figures/. Returns the written paths."""
    m = run.read_json("metrics")
    curves = m["curves"]
    cfg = run.config()
    classes, min_n = m["classes"], cfg["metrics"]["min_class_n"]
    out = run.path / "figures"
    out.mkdir(exist_ok=True)
    written = []

    def save(fig, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written.append(path)

    th = m.get("thresholds") or {}
    primary = cfg["gate"]["primary"]
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_tradeoff(ax, m, cfg["endpoints"]["sequential"][0])
    save(fig, out / "fig1_tradeoff.png")

    if {"jev", "vlm"} <= set(curves["roc"]) and primary in m["systems"]:
        save(main_figure(m, curves, classes, min_n, primary), out / "fig2_per_class.png")

    if m.get("jev_self_knowledge") and "need_vlm" in curves["auto_curves"]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        draw_need_vlm_levels(axes[0], m["jev_self_knowledge"]["jev_error_by_need_vlm_level"])
        draw_auto_curve(axes[1], curves["auto_curves"]["need_vlm"], th.get("tau_need_vlm"),
                        "τ_e: send to the VLM when need_vlm ≥ τ_e", "Gate operating curve (τ_e fitted on dev)")
        fig.tight_layout()
        save(fig, out / "fig3_jev_knows.png")

    rel = [s for s in ("jev", "vlm") if s in curves["reliability"]]
    if rel:
        fig, axes = plt.subplots(1, len(rel), figsize=(4.8 * len(rel), 4.3), squeeze=False)
        for ax, s in zip(axes[0], rel, strict=True):
            draw_reliability(ax, curves["reliability"][s], m["systems"][s]["ece"], SYSTEM_TITLE[s])
        fig.tight_layout()
        save(fig, out / "fig4_calibration.png")

    if cfg["plots"]["appendix"]:
        _appendix(m, curves, classes, min_n, out / "appendix", save)
    return written


def _appendix(m, curves, classes, min_n, out: Path, save) -> None:
    """Per-system ROC, one-vs-rest 2x2 grids and confusion matrices."""
    n_tag = f"n = {m['n_box']}"
    for s, row in m["systems"].items():
        support = {c: row["per_class"][c]["support"] for c in classes}
        if s in curves["roc"]:
            fig, ax = plt.subplots(figsize=(5.2, 5))
            draw_roc(ax, curves["roc"][s], row["auc"], curves["operating_points"][s], classes, min_n,
                     f"{SYSTEM_TITLE[s]}   {n_tag}")
            save(fig, out / f"roc_{s}.png")
        fig = plt.figure(figsize=(9, 4.6))
        draw_2x2_grid(fig, row["ovr"], support, classes, min_n, f"{SYSTEM_TITLE[s]}   {n_tag}")
        fig.subplots_adjust(top=0.82, bottom=0.08, hspace=0.62, wspace=0.45)
        save(fig, out / f"ovr_2x2_{s}.png")
        fig, ax = plt.subplots(figsize=(5.6, 5))
        draw_confusion(ax, row["confusion"], f"{SYSTEM_TITLE[s]}   acc {_fmt(row['accuracy'], 3)}")
        save(fig, out / f"confusion_{s}.png")


def main_figure(m: dict, data: dict, classes: list[str], min_n: int, primary: str):
    """A: ③ Jev ROC, B: ② VLM ROC, C: ③ Jev 2x2 per class, D: ④ primary gate 2x2 per class."""
    fig = plt.figure(figsize=(14, 11))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 0.85])
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
        sub.subplots_adjust(top=0.82, bottom=0.06, hspace=0.62, wspace=0.5)
    return fig
