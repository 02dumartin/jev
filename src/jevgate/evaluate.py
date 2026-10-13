"""S7: the results table, endpoints and plot data for one run (PLAN.md §5, §6)."""

from collections import defaultdict

import numpy as np

from . import metrics as M
from .gate import tau_curve
from .runio import RunDir

SYSTEM_ORDER = ["threshold", "vlm", "jev", "gate"]


def _system_row(system, rows, labels, timing, cfg, vlm_latency: dict) -> dict:
    """One table row. vlm_latency maps cand_id -> VLM ms, used to cost the boxes the gate routes."""
    scored = [(r, labels[r["cand_id"]]["y"]) for r in rows if labels[r["cand_id"]]["scored"]]
    y = [yy for _, yy in scored]
    row = M.binary_metrics(y, [r["keep"] for r, _ in scored])
    p_rows = [(r["p_real"], yy) for r, yy in scored if r["p_real"] is not None
              and (system != "gate" or r["routed_to"] == "jev")]
    n_bins = cfg["metrics"]["ece_bins"]
    if system in ("jev", "gate") and p_rows:
        ps, ys = zip(*p_rows, strict=True)
        row["ece"], row["brier"], row["n_calibrated"] = M.ece(ys, ps, n_bins), M.brier(ys, ps), len(ps)
    else:
        row["ece"] = row["brier"] = None
    # Diagnostic: ranking quality of the score each system thresholds (not an endpoint).
    score_key = {"threshold": "score", "vlm": "p_yes", "jev": "p_real"}.get(system)
    row["auroc"] = M.auroc(y, [r[score_key] for r, _ in scored]) if score_key else None
    row["vlm_calls"] = {"threshold": 0, "jev": 0, "vlm": len(rows)}.get(
        system, sum(r.get("routed_to") == "vlm" for r in rows))
    lat = [r["latency_ms"] for r in rows]
    row["latency_ms_median"] = float(np.median(lat)) if lat else None
    if system == "vlm":
        row["vlm_gpu_s"] = sum(vlm_latency.values()) / 1000
    elif system == "gate":
        row["vlm_gpu_s"] = sum(vlm_latency[r["cand_id"]] for r in rows if r["routed_to"] == "vlm") / 1000
    else:
        row["vlm_gpu_s"] = 0.0
    # The gate is not run separately: its wall time = Jev wall time + VLM time of the routed boxes.
    row["total_time_s"] = (timing.get("jev", 0.0) + row["vlm_gpu_s"]) if system == "gate" else timing.get(system)
    tokens = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rows)
    price = cfg["cost"].get("jev_usd_per_1m_tokens")
    row["jev_tokens"] = round(tokens)
    row["jev_cost_usd"] = tokens / 1e6 * price if price is not None else None
    if system == "gate":
        row["auto_rate"] = 1 - row["vlm_calls"] / len(rows) if rows else None
        auto = [(r["keep"], yy) for r, yy in scored if r["routed_to"] == "jev"]
        row["auto_error"] = float(np.mean([k != bool(yy) for k, yy in auto])) if auto else None
    return row


def _frame_keys(frames):
    return [(f["clip"], f["frame_idx"]) for f in frames if f["gt_status"] != "unknown"]


def _primary(run, cands, labels, decisions, frames, cfg) -> dict:
    if not {"gate", "vlm"} <= set(decisions):
        return {"delta_acc": None, "call_ratio": None}
    frame_of = {c["cand_id"]: (c["clip"], c["frame_idx"]) for c in cands}
    correct = {s: defaultdict(list) for s in ("gate", "vlm")}
    for s in correct:
        for r in decisions[s]:
            lab = labels[r["cand_id"]]
            if lab["scored"]:
                correct[s][frame_of[r["cand_id"]]].append(int(r["keep"] == bool(lab["y"])))
    keys = [k for k in _frame_keys(frames) if k in correct["vlm"]]
    b = cfg["metrics"]["bootstrap"]
    ci = M.block_bootstrap_delta(keys, correct["gate"], correct["vlm"], b["n"], b["block_frames"], b["seed"])
    n_vlm = len(decisions["vlm"])
    gate_calls = sum(r["routed_to"] == "vlm" for r in decisions["gate"])
    return {"delta_acc": ci, "call_ratio": gate_calls / n_vlm if n_vlm else None}


def _frame_visibility(frames, cands, decisions, gt_count) -> dict:
    keys = _frame_keys(frames)
    frame_of = {c["cand_id"]: (c["clip"], c["frame_idx"]) for c in cands}
    out = {}
    for s, rows in decisions.items():
        kept = defaultdict(bool)
        for r in rows:
            kept[frame_of[r["cand_id"]]] |= r["keep"]
        out[s] = M.frame_visible_accuracy(keys, {k: gt_count.get(k, 0) > 0 for k in keys}, kept)
    return out


def _sensitivity(labels, decisions, frames, cands) -> dict:
    """Accuracy when ambiguous boxes count as real / as fake, and when empty frames are excluded."""
    empty = {(f["clip"], f["frame_idx"]) for f in frames if f["gt_status"] == "empty_by_absence"}
    frame_of = {c["cand_id"]: (c["clip"], c["frame_idx"]) for c in cands}
    out = {}
    for variant in ("ambiguous_as_real", "ambiguous_as_fake", "exclude_empty_frames"):
        out[variant] = {}
        for s, rows in decisions.items():
            ys, ks = [], []
            for r in rows:
                lab = labels[r["cand_id"]]
                if lab["max_iou"] is None:
                    continue
                y = lab["y"]
                if y is None and variant != "exclude_empty_frames":
                    y = 1 if variant == "ambiguous_as_real" else 0
                if y is None or (variant == "exclude_empty_frames" and frame_of[r["cand_id"]] in empty):
                    continue
                ys.append(y)
                ks.append(r["keep"])
            out[variant][s] = M.binary_metrics(ys, ks)["accuracy"]
    return out


def evaluate_run(run: RunDir, gt_count: dict, recall_ub: dict) -> dict:
    """Build metrics.json for a run whose S1-S6 outputs exist."""
    cfg = run.config()
    frames, cands = run.read("frames"), run.read("candidates")
    labels = {r["cand_id"]: r for r in run.read("labels")}
    timing = run.read_json("timing.json") if (run.path / "timing.json").exists() else {}
    decisions = {s: run.read_decisions(s) for s in SYSTEM_ORDER if s in run.systems_done()}

    vlm_latency = {r["cand_id"]: r["latency_ms"] for r in decisions.get("vlm", [])}
    systems = {s: _system_row(s, rows, labels, timing, cfg, vlm_latency) for s, rows in decisions.items()}
    primary = _primary(run, cands, labels, decisions, frames, cfg)
    ci_low = primary["delta_acc"]["ci_low"] if primary["delta_acc"] else None
    endpoints = M.evaluate_endpoints(ci_low, primary["call_ratio"], cfg["endpoints"]["sequential"])

    plots = {}
    if "jev" in decisions:
        scored = [(r["p_real"], labels[r["cand_id"]]["y"]) for r in decisions["jev"] if labels[r["cand_id"]]["scored"]]
        if scored:
            ps, ys = zip(*scored, strict=True)
            plots["reliability_jev"] = M.reliability_bins(ys, ps, cfg["metrics"]["ece_bins"])
            grid = np.round(np.arange(0.5, 1.0 + 1e-9, cfg["gate"]["tau_grid_step"]), 6)
            plots["tau_curve"] = tau_curve(ys, ps, grid)
    run.write_json("plots_data.json", plots)

    n_scored = sum(r["scored"] for r in labels.values())
    return {
        "run_id": run.run_id,
        "split": cfg["run"]["split"],
        "git_commit": run.git().get("commit"),
        "git_dirty": run.git().get("dirty"),
        "n_clip": len({f["clip"] for f in frames}),
        "n_frame": len(frames),
        "n_box": n_scored,
        "n_candidates": len(cands),
        "n_ambiguous": sum(r["y"] is None and r["max_iou"] is not None for r in labels.values()),
        "n_real": sum(r["y"] == 1 for r in labels.values()),
        "n_fake": sum(r["y"] == 0 for r in labels.values()),
        "recall_upper_bound": recall_ub,
        "thresholds": run.read_json("thresholds.json") if (run.path / "thresholds.json").exists() else None,
        "systems": systems,
        "primary": primary,
        "endpoints": endpoints,
        "frame_visible_accuracy": _frame_visibility(frames, cands, decisions, gt_count),
        "sensitivity": _sensitivity(labels, decisions, frames, cands),
        "jev_models": sorted({r.get("jev_model") for r in decisions.get("jev", []) if r.get("jev_model")}),
    }
