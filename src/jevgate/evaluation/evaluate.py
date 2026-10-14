"""S7: every number of a run (and the curve data the figures draw) -> metrics.json (PLAN.md §5, §6)."""

import csv
from collections import defaultdict

import numpy as np

from jevgate.core.config import BACKGROUND, class_names
from jevgate.core.runio import RunDir
from jevgate.evaluation import metrics as M
from jevgate.judges.gate import GATES, auto_curve, conf_grid, need_vlm_grid

SYSTEM_ORDER = ["threshold", "vlm", "jev", "gate", "gate_conf", "gate_either"]
SYSTEM_TITLE = {"threshold": "① detector (DINO score ≥ t)", "vlm": "② VLM", "jev": "③ Jev",
                "gate": "④ Jev need_vlm gate", "gate_conf": "④a Jev confidence gate", "gate_either": "④c either gate"}
PROB_SYSTEMS = ("threshold", "vlm", "jev")  # systems with a full class-score vector for ROC


class RunData:
    """Everything S7 needs from a run folder, indexed by cand_id."""

    def __init__(self, run: RunDir):
        self.run = run
        self.cfg = run.config()
        self.classes = class_names(self.cfg)
        self.frames = run.read("frames")
        self.cands = run.read("boxes")
        self.labels = {b["cand_id"]: b for b in self.cands}
        rows = run.system_rows()
        self.decisions = {s: rows[s] for s in SYSTEM_ORDER if s in rows}
        self.frame_of = {c["cand_id"]: (c["clip"], c["frame_idx"]) for c in self.cands}
        self.scored_ids = [c["cand_id"] for c in self.cands if c["scored"]]
        self.y = [self.labels[i]["y"] for i in self.scored_ids]
        self.timing = run.meta().get("timing_s", {})
        self.detect_ms = {(f["clip"], f["frame_idx"]): f["detect_ms"] for f in self.frames if f.get("detect_ms")}

    def scored_rows(self, system: str) -> list[dict]:
        by_id = {r["cand_id"]: r for r in self.decisions[system]}
        return [by_id[i] for i in self.scored_ids]


def _frame_times(data: RunData, system: str) -> dict[tuple, float]:
    """End-to-end ms per frame if the system ran online: DINO + judging that frame's boxes."""
    rows = data.decisions[system]
    per_frame = defaultdict(float)
    if system in ("jev",) or system in GATES:
        # one Jev request per frame (per_frame batching): count its latency once per frame
        seen = set()
        jev = {r["cand_id"]: r for r in data.decisions["jev"]}
        for r in rows:
            key = data.frame_of[r["cand_id"]]
            j = jev[r["cand_id"]]
            if key not in seen:
                per_frame[key] += j.get("request_latency_ms", j["latency_ms"] * j.get("request_size", 1))
                seen.add(key)
            if system in GATES and r["routed_to"] == "vlm":
                per_frame[key] += r["latency_ms"] - j["latency_ms"]
    elif system == "vlm":
        for r in rows:
            per_frame[data.frame_of[r["cand_id"]]] += r["latency_ms"]
    return {k: data.detect_ms.get(k, 0.0) + per_frame.get(k, 0.0) for k in (
        (f["clip"], f["frame_idx"]) for f in data.frames)}


def _timing(data: RunData, system: str) -> dict:
    rows = data.decisions[system]
    lat = [r["latency_ms"] for r in rows]
    frame_ms = list(_frame_times(data, system).values())
    vlm_ms = {r["cand_id"]: r["latency_ms"] for r in data.decisions.get("vlm", [])}
    if system == "vlm":
        judge_s = sum(vlm_ms.values()) / 1000
    elif system in GATES:
        routed_ms = sum(vlm_ms[r["cand_id"]] for r in rows if r["routed_to"] == "vlm")
        judge_s = data.timing.get("jev", 0.0) + routed_ms / 1000
    else:
        judge_s = data.timing.get(system, 0.0)
    mean_frame = float(np.mean(frame_ms)) if frame_ms else None
    return {
        "latency_ms_median": float(np.median(lat)) if lat else None,
        "judge_time_s": round(judge_s, 2),
        "detect_time_s": round(sum(data.detect_ms.values()) / 1000, 2),
        "frame_ms_median": float(np.median(frame_ms)) if frame_ms else None,
        "frame_ms_mean": mean_frame,
        "fps": 1000 / mean_frame if mean_frame else None,
    }


def _system_row(data: RunData, system: str, boot: M.BlockBootstrap) -> dict:
    cfg, classes = data.cfg, data.classes
    rows = data.scored_rows(system)
    preds = [r["pred"] for r in rows]
    row = M.multiclass_metrics(data.y, preds, classes)
    correct = np.asarray([p == yy for p, yy in zip(preds, data.y, strict=True)], dtype=float)
    row["accuracy_ci"] = list(boot.ci(lambda idx: float(correct[idx].mean()) if len(idx) else None))
    row["macro_precision"] = _mean([v["precision"] for v in row["per_class"].values() if v["support"]])
    row["macro_recall"] = _mean([v["recall"] for v in row["per_class"].values() if v["support"]])
    row["ovr"] = {c: M.ovr_counts(data.y, preds, c) for c in classes}

    calib = [(r, yy) for r, yy in zip(rows, data.y, strict=True)
             if system in ("vlm", "jev") or (system in GATES and r["routed_to"] == "jev")]
    if calib:
        correct = [r["pred"] == yy for r, yy in calib]
        conf = [r["confidence"] for r, _ in calib]
        row["ece"] = M.ece_top(correct, conf, cfg["metrics"]["ece_bins"])
        row["brier"] = M.brier_multi([yy for _, yy in calib], [r["probs"] for r, _ in calib], classes)
        row["n_calibrated"] = len(calib)
    else:
        row["ece"] = row["brier"] = None

    if system in PROB_SYSTEMS:
        y_arr = np.asarray(data.y)
        row["auc"] = {}
        for c in classes:
            yb = y_arr == c
            score = np.asarray([r["probs"][c] for r in rows])
            point = M.auroc(yb, score)
            lo, hi = boot.ci(lambda idx, yb=yb, score=score: M.auroc(yb[idx], score[idx])) if point else (None, None)
            row["auc"][c] = {"auc": point, "ci_low": lo, "ci_high": hi, "n_pos": int(yb.sum())}

    all_rows = data.decisions[system]
    row["vlm_calls"] = {"vlm": len(all_rows)}.get(system, sum(r.get("routed_to") == "vlm" for r in all_rows))
    row.update(_timing(data, system))
    tokens = sum(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in all_rows)
    price = cfg["cost"].get("jev_usd_per_1m_tokens")
    row["jev_tokens"] = round(tokens)
    row["jev_cost_usd"] = tokens / 1e6 * price if price is not None else None
    if system in GATES:
        auto = [r["pred"] == yy for r, yy in zip(rows, data.y, strict=True) if r["routed_to"] == "jev"]
        row["auto_rate"] = 1 - row["vlm_calls"] / len(all_rows) if all_rows else None
        row["auto_error"] = 1 - float(np.mean(auto)) if auto else None
    return row


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _delta_acc(data: RunData, a: str, b: str, boot: M.BlockBootstrap) -> dict:
    ca = np.asarray([r["pred"] == yy for r, yy in zip(data.scored_rows(a), data.y, strict=True)], dtype=float)
    cb = np.asarray([r["pred"] == yy for r, yy in zip(data.scored_rows(b), data.y, strict=True)], dtype=float)
    lo, hi = boot.ci(lambda idx: float(ca[idx].mean() - cb[idx].mean()) if len(idx) else None)
    return {"delta": float(ca.mean() - cb.mean()) if len(ca) else None, "ci_low": lo, "ci_high": hi}


def _jev_selfknowledge(data: RunData) -> dict:
    """Does Jev know when it is wrong? AUROC of need_vlm (and of 1 - confidence) for predicting Jev errors."""
    rows = data.scored_rows("jev")
    wrong = np.asarray([r["pred"] != yy for r, yy in zip(rows, data.y, strict=True)])
    need = np.asarray([r["need_vlm"] for r in rows])
    unsure = np.asarray([1 - r["confidence"] for r in rows])
    by_level = {}
    for lv in range(1, len(data.cfg["prompts"]["need_vlm_levels"]) + 1):
        m = np.round(need) == lv
        by_level[str(lv)] = {"n": int(m.sum()), "jev_error": float(wrong[m].mean()) if m.any() else None}
    return {"auroc_need_vlm_for_error": M.auroc(wrong, need), "auroc_unconfidence_for_error": M.auroc(wrong, unsure),
            "jev_error_by_need_vlm_level": by_level}


def _sensitivity(data: RunData) -> dict:
    """Accuracy under alternative labelling choices (exploratory)."""
    empty = {(f["clip"], f["frame_idx"]) for f in data.frames if f["gt_status"] == "empty_by_absence"}
    merge = {"small_fish": "fish"}
    out = {"ambiguous_as_background": {}, "exclude_empty_frames": {}, "merge_fish_small_fish": {}}
    for s, rows in data.decisions.items():
        by_id = {r["cand_id"]: r for r in rows}
        ys, ps = zip(*[(data.labels[c["cand_id"]]["y"], by_id[c["cand_id"]]["pred"]) for c in data.cands],
                     strict=True)
        ids = [c["cand_id"] for c in data.cands]
        amb = [(BACKGROUND if y == "ambiguous" else y, p) for y, p in zip(ys, ps, strict=True)]
        out["ambiguous_as_background"][s] = float(np.mean([y == p for y, p in amb]))
        keep = [(y, p) for i, y, p in zip(ids, ys, ps, strict=True)
                if y != "ambiguous" and data.frame_of[i] not in empty]
        out["exclude_empty_frames"][s] = float(np.mean([y == p for y, p in keep])) if keep else None
        mer = [(merge.get(y, y), merge.get(p, p)) for y, p in zip(ys, ps, strict=True) if y != "ambiguous"]
        out["merge_fish_small_fish"][s] = float(np.mean([y == p for y, p in mer])) if mer else None
    return out


def _frame_visibility(data: RunData) -> dict:
    keys = [(f["clip"], f["frame_idx"]) for f in data.frames]
    has_animal = {(f["clip"], f["frame_idx"]): f["gt_status"] == "annotated" for f in data.frames}
    out = {}
    for s, rows in data.decisions.items():
        animal = defaultdict(bool)
        for r in rows:
            animal[data.frame_of[r["cand_id"]]] |= r["pred"] != BACKGROUND
        out[s] = M.frame_visible_accuracy(keys, has_animal, animal)
    return out


def _plot_data(data: RunData, systems: dict) -> dict:
    cfg, classes = data.cfg, data.classes
    y_arr = np.asarray(data.y)
    plots = {"roc": {}, "operating_points": {}, "reliability": {}, "auto_curves": {}}
    for s in data.decisions:
        rows = data.scored_rows(s)
        if s in PROB_SYSTEMS:
            plots["roc"][s] = {c: M.roc_points(y_arr == c, [r["probs"][c] for r in rows]) for c in classes}
        plots["operating_points"][s] = {c: {"fpr": 1 - v["spec"] if v["spec"] is not None else None, "tpr": v["sens"]}
                                        for c, v in systems[s]["ovr"].items()}
        if s in ("vlm", "jev"):
            plots["reliability"][s] = M.reliability_bins([r["pred"] == yy for r, yy in zip(rows, data.y, strict=True)],
                                                         [r["confidence"] for r in rows], cfg["metrics"]["ece_bins"])
    if "jev" in data.decisions:
        rows = data.scored_rows("jev")
        correct = [r["pred"] == yy for r, yy in zip(rows, data.y, strict=True)]
        levels = len(cfg["prompts"]["need_vlm_levels"])
        plots["auto_curves"]["need_vlm"] = auto_curve(correct, [r["need_vlm"] for r in rows],
                                                      need_vlm_grid(levels, cfg["gate"]["need_vlm_grid_step"]), "below")
        plots["auto_curves"]["confidence"] = auto_curve(correct, [r["confidence"] for r in rows],
                                                        conf_grid(cfg["gate"]["conf_grid_step"]), "at_least")
    return plots


def write_errors(data: RunData, path) -> int:
    """Boxes that any of ② ③ or the primary gate got wrong, with every system's call."""
    systems = [s for s in SYSTEM_ORDER if s in data.decisions]
    by_sys = {s: {r["cand_id"]: r for r in data.decisions[s]} for s in systems}
    file_of = {(f["clip"], f["frame_idx"]): f["file"] for f in data.frames}
    primary = data.cfg["gate"]["primary"]
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cand_id", "file", "box_xyxy", "y", "max_iou", "dino_label", "dino_score"]
                   + [f"{s}_pred" for s in systems] + [f"{s}_conf" for s in systems] + ["need_vlm", "routed_to"])
        cand = {c["cand_id"]: c for c in data.cands}
        for i, yy in zip(data.scored_ids, data.y, strict=True):
            wrong = [s for s in ("vlm", "jev", primary) if s in by_sys and by_sys[s][i]["pred"] != yy]
            if not wrong:
                continue
            c = cand[i]
            w.writerow([i, file_of[data.frame_of[i]], c["box_xyxy"], yy, data.labels[i]["max_iou"],
                        c.get("dino_label"), c.get("dino_score")] + [by_sys[s][i]["pred"] for s in systems]
                       + [by_sys[s][i]["confidence"] for s in systems]
                       + [by_sys["jev"][i].get("need_vlm") if "jev" in by_sys else None,
                          by_sys[primary][i].get("routed_to") if primary in by_sys else None])
            n += 1
    return n


def evaluate_run(run: RunDir, n_flow: dict) -> dict:
    """metrics.json content (curve data included) + errors.csv for a run whose boxes and judgments exist."""
    data = RunData(run)
    cfg = data.cfg
    b = cfg["metrics"]["bootstrap"]
    boot = M.BlockBootstrap([data.frame_of[i] for i in data.scored_ids], b["block_frames"], b["n"], b["seed"])
    systems = {s: _system_row(data, s, boot) for s in data.decisions}

    primary_gate = cfg["gate"]["primary"]
    deltas = {g: _delta_acc(data, g, "vlm", boot) for g in GATES if g in data.decisions and "vlm" in data.decisions}
    call_ratio = {g: systems[g]["vlm_calls"] / systems["vlm"]["vlm_calls"] for g in deltas}
    primary = {"gate": primary_gate, "delta_acc": deltas.get(primary_gate), "call_ratio": call_ratio.get(primary_gate)}
    ci_low = primary["delta_acc"]["ci_low"] if primary["delta_acc"] else None
    endpoints = M.evaluate_endpoints(ci_low, primary["call_ratio"], cfg["endpoints"]["sequential"])

    n_errors = write_errors(data, run.file("errors"))
    return {
        "run_id": run.run_id,
        "split": cfg["run"]["split"],
        "git_commit": run.git().get("commit"),
        "git_dirty": run.git().get("dirty"),
        "classes": data.classes,
        "n_clip": len({f["clip"] for f in data.frames}),
        "n_frame": len(data.frames),
        "n_box": len(data.scored_ids),
        "n_candidates": len(data.cands),
        "candidate_source": cfg["candidates"]["source"],
        "n_flow": n_flow,
        "thresholds": run.meta().get("thresholds"),
        "systems": systems,
        "primary": primary,
        "secondary_gate_deltas": {g: {"delta_acc": deltas[g], "call_ratio": call_ratio[g]} for g in deltas},
        "endpoints": endpoints,
        "jev_self_knowledge": _jev_selfknowledge(data) if "jev" in data.decisions else None,
        "frame_visible_accuracy": _frame_visibility(data),
        "sensitivity": _sensitivity(data),
        "timing_stages_s": data.timing,
        "jev_models": sorted({r.get("jev_model") for r in data.decisions.get("jev", []) if r.get("jev_model")}),
        "n_error_rows": n_errors,
        "curves": _plot_data(data, systems),
    }
