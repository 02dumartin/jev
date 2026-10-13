"""S8: reliability diagram, tau curve, per-frame overlays and the Before -> After video."""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import image_path  # noqa: E402
from .runio import RunDir  # noqa: E402

# BGR colours for OpenCV.
REAL, FAKE, AMBIG = (80, 200, 80), (60, 60, 230), (0, 200, 230)
KEPT, REJECTED = (255, 255, 255), (120, 120, 120)


def plot_run(run: RunDir) -> None:
    """Reliability diagram (③ Jev) and tau curve from plots_data.json."""
    data = run.read_json("plots_data.json")
    out = run.path / "plots"
    out.mkdir(exist_ok=True)
    if "reliability_jev" in data:
        bins = [b for b in data["reliability_jev"] if b["count"]]
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect calibration")
        ax.bar([(b["lo"] + b["hi"]) / 2 for b in bins], [b["frac_real"] for b in bins],
               width=1 / len(data["reliability_jev"]) * 0.9, color="#4C78A8", alpha=0.85, label="observed real rate")
        ax.plot([b["mean_p"] for b in bins], [b["frac_real"] for b in bins], "o", color="#1f3b5c", ms=4)
        for b in bins:
            ax.annotate(str(b["count"]), ((b["lo"] + b["hi"]) / 2, b["frac_real"]), fontsize=6,
                        ha="center", va="bottom", color="#555")
        ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="Jev p(real)", ylabel="fraction actually real",
               title="Reliability: DINO + Jev")
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(out / "reliability_jev.png", dpi=160)
        plt.close(fig)
    if "tau_curve" in data:
        rows = [r for r in data["tau_curve"] if r["n_auto"]]
        th = run.read_json("thresholds.json") if (run.path / "thresholds.json").exists() else {}
        fig, ax = plt.subplots(figsize=(5, 3.4))
        tau = [r["tau"] for r in rows]
        ax.plot(tau, [r["auto_rate"] for r in rows], color="#4C78A8", label="auto-handled share")
        ax.plot(tau, [r["auto_error"] for r in rows], color="#E45756", label="error on auto-handled")
        ax.plot(tau, [1 - r["auto_rate"] for r in rows], color="#72B7B2", ls=":", label="VLM call share")
        if "tau" in th:
            ax.axvline(th["tau"], color="#333", lw=1, ls="--", label=f"tau = {th['tau']:.2f} (dev)")
        ax.set(xlabel="tau", ylabel="share", ylim=(0, 1.02), title="Gate operating curve")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "tau_curve.png", dpi=160)
        plt.close(fig)


def _frame_index(run: RunDir):
    frames = run.read("frames")
    cands = run.read("candidates")
    labels = {r["cand_id"]: r for r in run.read("labels")}
    decisions = {s: {r["cand_id"]: r for r in run.read_decisions(s)} for s in run.systems_done()}
    by_frame: dict[tuple, list] = {}
    for c in cands:
        by_frame.setdefault((c["clip"], c["frame_idx"]), []).append(c)
    return frames, by_frame, labels, decisions


def draw_boxes(img, cands, labels, keep_of=None, show_label_colour=True):
    """Draw candidates: colour = GT label (green real / red fake / yellow ambiguous); thin grey = rejected."""
    for c in cands:
        x1, y1, x2, y2 = (int(round(v)) for v in c["box_xyxy"])
        y = labels[c["cand_id"]]["y"]
        colour = (REAL if y == 1 else FAKE if y == 0 else AMBIG) if show_label_colour else KEPT
        if keep_of is not None and not keep_of[c["cand_id"]]["keep"]:
            cv2.rectangle(img, (x1, y1), (x2, y2), REJECTED, 1)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        if keep_of is not None and keep_of[c["cand_id"]].get("p_real") is not None:
            cv2.putText(img, f"{keep_of[c['cand_id']]['p_real']:.2f}", (x1, max(10, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    return img


def _caption(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_samples(run: RunDir, n: int, seed: int) -> Path:
    """n random frames as 2x2 panels: all candidates (GT colours) and each system's kept boxes."""
    cfg = run.config()
    frames, by_frame, labels, decisions = _frame_index(run)
    out = run.path / "overlay"
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(frames), size=min(n, len(frames)), replace=False)
    systems = [s for s in ("threshold", "vlm", "jev", "gate") if s in decisions]
    for k in sorted(picks):
        f = frames[k]
        base = cv2.imread(str(image_path(cfg, f["file"])))
        cands = by_frame.get((f["clip"], f["frame_idx"]), [])
        panels = [_caption(draw_boxes(base.copy(), cands, labels), f"candidates + GT  ({f['file']})")]
        for s in systems:
            panels.append(_caption(draw_boxes(base.copy(), cands, labels, decisions[s]), f"{s}: kept boxes"))
        while len(panels) % 2:
            panels.append(np.zeros_like(base))
        rows = [np.hstack(panels[i:i + 2]) for i in range(0, len(panels), 2)]
        sheet = cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f"{f['clip'][-25:]}-{f['frame_idx']:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def render_video(run: RunDir, clip: str, system: str, fps: int) -> Path:
    """Before (all DINO candidates) | After (boxes `system` keeps), side by side, one clip."""
    cfg = run.config()
    frames, by_frame, labels, decisions = _frame_index(run)
    out = run.path / "video"
    out.mkdir(exist_ok=True)
    path = out / f"{clip[-25:]}_{system}.mp4"
    writer = None
    for f in (f for f in frames if f["clip"] == clip):
        base = cv2.imread(str(image_path(cfg, f["file"])))
        cands = by_frame.get((f["clip"], f["frame_idx"]), [])
        before = _caption(draw_boxes(base.copy(), cands, labels, None, show_label_colour=False),
                          f"Before: detector candidates ({len(cands)})")
        kept = sum(decisions[system][c["cand_id"]]["keep"] for c in cands)
        after = _caption(draw_boxes(base.copy(), cands, labels, decisions[system], show_label_colour=False),
                         f"After: {system} keeps {kept}")
        panel = np.hstack([before, after])
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel.shape[1], panel.shape[0]))
        writer.write(panel)
    if writer is not None:
        writer.release()
    return path
