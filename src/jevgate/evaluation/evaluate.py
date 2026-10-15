"""S5: every number of a run (PLAN.md v7 §5) from the files of the earlier stages.

Box level: 7-way accuracy / macro F1 / per-class P,R, detection recall, flicker.
Track level: Jev calibration (ECE, Brier, reliability), ROC / AUROC [CI] of Jev p, p_big, p_small, mean conf.
Primary: Δacc(gate − vlm_all) with a cluster bootstrap over tracks, and the large-VLM call ratio.
"""

from collections import defaultdict

import numpy as np

from jevgate.core.config import BACKGROUND, output_classes
from jevgate.detectors.track import count_flicker
from jevgate.evaluation import metrics as M
from jevgate.judges.gate import band_gate_decisions, box_predictions, gate_decisions, tau_curve

SYSTEM_ORDER = ["yolo", "track_vote", "vlm_all", "small_vlm", "jev", "gate", "gate_small"]
GATES = ("gate", "gate_small")


class RunData:
    """Everything evaluation needs, indexed once. `small` = describer rows (p_yes, description)."""

    def __init__(self, cfg, frames, detections, tracks, box_labels, track_labels, jev_rows, vlm_rows, th, timing,
                 small_rows=None):
        self.cfg, self.frames, self.detections, self.tracks, self.th, self.timing = (
            cfg, frames, detections, tracks, th, timing)
        self.box_label = {r["box_id"]: r for r in box_labels}
        self.track_label = {r["track_uid"]: r for r in track_labels}
        self.majority = {t["track_uid"]: t["majority_class"] for t in tracks}
        self.mean_conf = {t["track_uid"]: t["evidence"]["detector_conf"]["mean"] for t in tracks}
        self.jev = {r["track_uid"]: r for r in jev_rows or []}
        self.vlm = {r["track_uid"]: r for r in vlm_rows or []}
        self.small = {r["track_uid"]: r for r in small_rows or []}
        self.gates: dict[str, list[dict]] = {}

    @property
    def gate_rows(self) -> list[dict]:
        return self.gates.get("gate", [])

    def accepts(self) -> dict[str, dict | None]:
        """Per track-judging system: track_uid -> accepted?; also fills self.gates with routing rows."""
        out: dict = {"track_vote": None}
        vlm_ms = {u: r["latency_ms"] for u, r in self.vlm.items()}
        if self.vlm:
            out["vlm_all"] = {u: r["p_yes"] >= self.th["v"] for u, r in self.vlm.items()}
        if self.small:
            out["small_vlm"] = {u: r["p_yes"] >= self.th["v_small"] for u, r in self.small.items()}
        if self.jev:
            out["jev"] = {u: r["p_ok"] >= 0.5 for u, r in self.jev.items()}
        if self.vlm and self.jev:
            pre = {u: r["latency_ms"] for u, r in self.small.items()}
            self.gates["gate"] = gate_decisions(list(self.jev.values()), out["vlm_all"], vlm_ms, self.th["tau"], pre)
        if self.vlm and self.small:
            self.gates["gate_small"] = band_gate_decisions(list(self.small.values()), out["vlm_all"], vlm_ms,
                                                           self.th["small_lo"], self.th["small_hi"])
        for g, rows in self.gates.items():
            out[g] = {r["track_uid"]: r["accept"] for r in rows}
        return out


def predictions(data: RunData) -> dict[str, dict[str, str]]:
    """system -> box_id -> predicted class (six animals or background)."""
    t = data.th["t"]
    preds = {"yolo": box_predictions(data.detections, data.majority, None, t, per_frame=True)}
    for system, accept in data.accepts().items():
        preds[system] = box_predictions(data.detections, data.majority, accept, t)
    return {s: preds[s] for s in SYSTEM_ORDER if s in preds and s in data.cfg["run"]["systems"]}


def detection_recall(data: RunData, pred: dict[str, str], n_gt: int) -> float | None:
    """GT objects that end up covered by a box with the right class (missed objects stay in the denominator)."""
    hit = set()
    for d in data.detections:
        lab = data.box_label[d["box_id"]]
        if lab["scored"] and lab["y"] not in (BACKGROUND, None) and pred[d["box_id"]] == lab["y"]:
            hit.add((d["file"], lab["matched_object_id"]))
    return len(hit) / n_gt if n_gt else None


def flicker(data: RunData, pred: dict[str, str]) -> float:
    """Mean over videos of the frame-to-frame change in the number of boxes shown as an animal."""
    shown = defaultdict(int)
    for d in data.detections:
        shown[(d["video"], d["frame_idx"])] += pred[d["box_id"]] != BACKGROUND
    per_video = defaultdict(list)
    for f in data.frames:
        per_video[f["video"]].append(shown[(f["video"], f["frame_idx"])])
    return float(np.mean([count_flicker(v) for v in per_video.values()])) if per_video else 0.0


def judge_time(data: RunData, system: str) -> dict:
    """Judge cost per system: large-VLM calls, per-track latency median, judge seconds, Jev tokens.

    Small-VLM time counts for every system that uses it (small_vlm, gate, gate_small).
    """
    jev_tokens = round(sum(r["tokens_in"] + r["tokens_out"] for r in data.jev.values()))
    small_s = data.timing.get("describer_s") or 0.0
    if system == "vlm_all":
        lat = [r["latency_ms"] for r in data.vlm.values()]
        return {"vlm_calls": len(lat), "latency_ms_median": _median(lat), "judge_time_s": sum(lat) / 1000,
                "jev_tokens": 0}
    if system == "small_vlm":
        return {"vlm_calls": 0, "latency_ms_median": _median([r["latency_ms"] for r in data.small.values()]),
                "judge_time_s": small_s, "jev_tokens": 0}
    if system == "jev":
        lat = [r["latency_ms"] + data.small.get(u, {}).get("latency_ms", 0.0) for u, r in data.jev.items()]
        return {"vlm_calls": 0, "latency_ms_median": _median(lat),
                "judge_time_s": small_s + (data.timing.get("jev_s") or 0.0), "jev_tokens": jev_tokens}
    if system in GATES:
        rows = data.gates[system]
        routed = [r for r in rows if r["routed_to"] == "vlm"]
        vlm_s = sum(data.vlm[r["track_uid"]]["latency_ms"] for r in routed) / 1000
        jev_s = (data.timing.get("jev_s") or 0.0) if system == "gate" else 0.0
        return {"vlm_calls": len(routed), "latency_ms_median": _median([r["latency_ms"] for r in rows]),
                "judge_time_s": small_s + jev_s + vlm_s, "jev_tokens": jev_tokens if system == "gate" else 0}
    return {"vlm_calls": 0, "latency_ms_median": None, "judge_time_s": 0.0, "jev_tokens": 0}


def _median(values) -> float | None:
    return float(np.median(values)) if len(values) else None


def system_row(data: RunData, system: str, pred: dict[str, str], n_gt: int) -> dict:
    """One row of the results table."""
    scored = [d["box_id"] for d in data.detections if data.box_label[d["box_id"]]["scored"]]
    y = [data.box_label[b]["y"] for b in scored]
    row = M.multiclass_metrics(y, [pred[b] for b in scored], output_classes(data.cfg))
    row |= judge_time(data, system)
    row["total_time_s"] = (data.timing.get("yolo_track_s") or 0.0) + (row["judge_time_s"] or 0.0)
    price = data.cfg["cost"].get("jev_usd_per_1m_tokens")
    row["jev_cost_usd"] = row["jev_tokens"] / 1e6 * price if price is not None else None
    row["detection_recall"] = detection_recall(data, pred, n_gt)
    row["flicker"] = flicker(data, pred)
    if system in GATES:
        rows = data.gates[system]
        auto = [r for r in rows if r["routed_to"] != "vlm"]
        scored_auto = [r for r in auto if data.track_label[r["track_uid"]]["y_track"] is not None]
        row["auto_rate"] = len(auto) / len(rows) if rows else None
        row["auto_error"] = (float(np.mean([r["accept"] != bool(data.track_label[r["track_uid"]]["y_track"])
                                            for r in scored_auto])) if scored_auto else None)
    return row


def primary(data: RunData, preds: dict, cfg: dict) -> dict:
    """Δacc(a − b) on scored boxes with a cluster bootstrap (cluster = track, or untracked boxes of a video)."""
    a, b = cfg["endpoints"]["primary_pair"]
    if a not in preds or b not in preds:
        return {"pair": [a, b], "delta_acc": None, "call_ratio": None}
    scored = [d for d in data.detections if data.box_label[d["box_id"]]["scored"]]
    y = np.asarray([data.box_label[d["box_id"]]["y"] for d in scored])
    ca = np.asarray([preds[a][d["box_id"]] for d in scored]) == y
    cb = np.asarray([preds[b][d["box_id"]] for d in scored]) == y
    clusters = [d["track_uid"] or f"untracked:{d['video']}" for d in scored]
    boot = M.ClusterBootstrap(clusters, cfg["metrics"]["bootstrap"]["n"], cfg["metrics"]["bootstrap"]["seed"])
    lo, hi = boot.ci(lambda idx: float(ca[idx].mean() - cb[idx].mean()) if len(idx) else None)
    n_tracks = len(data.gates[a])
    calls = sum(r["routed_to"] == "vlm" for r in data.gates[a])
    return {"pair": [a, b], "delta_acc": {"delta": float(ca.mean() - cb.mean()), "ci_low": lo, "ci_high": hi,
                                          "n_clusters": len(boot.members)},
            "call_ratio": calls / n_tracks if n_tracks else None}


def track_level(data: RunData, cfg: dict) -> tuple[dict, dict]:
    """Jev calibration, AUROC [CI] of each track score, and the curve data for the figures."""
    uids = [u for u, r in data.track_label.items() if r["y_track"] is not None]
    y = np.asarray([data.track_label[u]["y_track"] for u in uids])
    scores = {"jev_p_ok": (data.jev, "p_ok"), "vlm_p_yes": (data.vlm, "p_yes"),
              "small_p_yes": (data.small, "p_yes"), "mean_conf": (data.mean_conf, None)}
    boot = M.ClusterBootstrap(uids, cfg["metrics"]["bootstrap"]["n"], cfg["metrics"]["bootstrap"]["seed"])
    out, curves = {"n_tracks_scored": len(uids), "n_ok": int(y.sum()) if len(y) else 0}, {"roc": {}}
    for name, (src, key) in scores.items():
        if not src or not uids:
            continue
        s = np.asarray([src[u][key] if key else src[u] for u in uids], dtype=float)
        lo, hi = boot.ci(lambda idx, s=s: M.auroc(y[idx], s[idx]))
        out[f"auroc_{name}"] = {"auc": M.auroc(y, s), "ci_low": lo, "ci_high": hi}
        curves["roc"][name] = M.roc_points(y, s)
    if data.jev and uids:
        p = np.asarray([data.jev[u]["p_ok"] for u in uids])
        out["jev_ece"] = M.ece_top(y, p, cfg["metrics"]["ece_bins"])
        out["jev_brier"] = M.brier_binary(y, p)
        curves["reliability_jev"] = M.reliability_bins(y, p, cfg["metrics"]["ece_bins"])
        grid = np.round(np.arange(0.5, 1.0 + 1e-9, cfg["gate"]["tau_grid_step"]), 6)
        curves["tau_curve"] = tau_curve(y.tolist(), p.tolist(), grid)
    if data.vlm and uids:
        accept = [data.vlm[u]["p_yes"] >= data.th["v"] for u in uids]
        out["vlm_track_accuracy_at_v"] = float(np.mean(np.asarray(accept) == y.astype(bool)))
    return out, curves


def secondary_gate(data: RunData, preds: dict, cfg: dict) -> dict | None:
    """S1: the small-VLM gate on the same footing as the primary (Δacc vs vlm_all, call ratio)."""
    if "gate_small" not in preds or "vlm_all" not in preds:
        return None
    pair = {"endpoints": {"primary_pair": ["gate_small", "vlm_all"]}, "metrics": cfg["metrics"]}
    return primary(data, preds, pair)


def sensitivity(data: RunData, preds: dict) -> dict:
    """Accuracy when ambiguous boxes (0.3 <= IoU < 0.5) are scored as background."""
    rows = [d for d in data.detections if data.box_label[d["box_id"]]["gt_status"] != "unknown"]
    y = [data.box_label[d["box_id"]]["y"] or BACKGROUND for d in rows]
    return {"ambiguous_as_background": {s: float(np.mean([p[d["box_id"]] == yy for d, yy in zip(rows, y, strict=True)]))
                                        for s, p in preds.items()}}


def evaluate(data: RunData, missed: dict, budget: dict, cfg: dict) -> tuple[dict, list[dict], dict]:
    """(metrics.json, judgments rows, curve data)."""
    preds = predictions(data)
    systems = {s: system_row(data, s, p, missed["n_gt_objects"]) for s, p in preds.items()}
    prim = primary(data, preds, cfg)
    ci_low = prim["delta_acc"]["ci_low"] if prim.get("delta_acc") else None
    endpoints = M.evaluate_endpoints(ci_low, prim.get("call_ratio"), cfg["endpoints"]["sequential"])
    tl, curves = track_level(data, cfg)
    curves["confusion"] = {s: row["confusion"] for s, row in systems.items()}
    labels = list(data.box_label.values())
    untracked = [d for d in data.detections if d["track_uid"] is None]
    metrics = {
        "split": cfg["run"]["split"], "thresholds": data.th,
        "n": {"videos": len({f["video"] for f in data.frames}), "frames": len(data.frames),
              "detections": len(data.detections), "boxes_scored": sum(r["scored"] for r in labels),
              "boxes_ambiguous": sum(r["y"] is None and r["gt_status"] != "unknown" for r in labels),
              "boxes_unknown_gt": sum(r["gt_status"] == "unknown" for r in labels),
              "tracks": len(data.tracks), "tracks_scored": tl["n_tracks_scored"],
              "track_types": dict(sorted(_count(r["type"] for r in data.track_label.values()).items())),
              "untracked_boxes": len(untracked),
              "untracked_real_animals": sum(data.box_label[d["box_id"]]["y"] not in (BACKGROUND, None)
                                            for d in untracked)},
        "gate_secondary": secondary_gate(data, preds, cfg),
        "missed_objects": missed,
        "systems": systems, "primary": prim, "endpoints": endpoints, "track_level": tl,
        "sensitivity": sensitivity(data, preds), "timing": data.timing, "budget_used": budget,
        "jev_models": sorted({r["jev_model"] for r in data.jev.values() if r.get("jev_model")}),
    }
    judgments = [{"box_id": d["box_id"], "track_uid": d["track_uid"], "y": data.box_label[d["box_id"]]["y"],
                  **{s: p[d["box_id"]] for s, p in preds.items()}} for d in data.detections]
    return metrics, judgments, curves


def _count(items) -> dict:
    out = defaultdict(int)
    for i in items:
        out[i] += 1
    return dict(out)
