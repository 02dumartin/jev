"""Data contracts of a run folder (PLAN.md §3). Each check returns a list of violation strings."""

from collections import Counter

from .config import class_names
from .data import evenly_spaced, split_files
from .gate import GATES
from .label import AMBIGUOUS
from .runio import RunDir

REQUIRED = {
    "frames": ["clip", "split", "frame_idx", "file", "gt_status", "detect_ms"],
    "boxes": ["cand_id", "clip", "frame_idx", "file", "box_xyxy", "y", "max_iou", "matched_object_id", "scored",
              "evidence"],
    "boxes_dino": ["dino_label", "dino_score"],
    "evidence": ["cand_id", "history_frame_idxs", "geometry", "track"],
    "decision": ["pred", "probs", "confidence", "latency_ms"],
}
METRIC_KEYS = ["systems", "n_box", "n_frame", "n_clip", "git_commit", "n_flow", "endpoints", "curves"]
SYSTEM_COLUMNS = ["accuracy", "accuracy_ci", "macro_f1", "macro_precision", "macro_recall", "ece", "vlm_calls",
                  "latency_ms_median", "frame_ms_mean", "fps", "per_class", "ovr", "confusion"]


def missing_fields(rows: list[dict], fields: list[str], name: str) -> list[str]:
    errors = []
    for k, row in enumerate(rows):
        absent = [f for f in fields if f not in row]
        if absent:
            errors.append(f"{name} row {k}: missing {absent}")
            if len(errors) >= 5:
                errors.append(f"{name}: further rows not listed")
                break
    return errors


def check_frames(frames, cfg) -> list[str]:
    """Frame rows = the split list (or its evenly spaced subset for smoke runs)."""
    errors = missing_fields(frames, REQUIRED["frames"], "frames")
    expected = evenly_spaced(split_files(cfg, cfg["run"]["split"]), cfg["run"].get("max_frames"))
    if [f["file"] for f in frames] != expected:
        errors.append(f"frames: {len(frames)} rows do not match the {len(expected)} split frames")
    return errors


def check_evidence(evidence, cands, cfg) -> list[str]:
    """Causality rule: history frames must be strictly earlier than the box's frame and at most K."""
    errors = missing_fields(evidence, REQUIRED["evidence"], "evidence")
    frame_of = {c["cand_id"]: c["frame_idx"] for c in cands}
    k = cfg["evidence"]["history_frames"]
    for e in evidence:
        idx, hist = frame_of.get(e["cand_id"]), e["history_frame_idxs"]
        if any(h >= idx for h in hist):
            errors.append(f"evidence: future/current frame leak in {e['cand_id']}: {hist} vs {idx}")
        if len(hist) > k:
            errors.append(f"evidence: history longer than K={k} in {e['cand_id']}")
    return errors


def check_boxes(boxes, frames, cfg) -> list[str]:
    """Schema, unique ids, boxes inside their (listed) frame, valid truth, causal evidence."""
    dino = cfg["candidates"]["source"] == "dino"
    errors = missing_fields(boxes, REQUIRED["boxes"] + (REQUIRED["boxes_dino"] if dino else []), "boxes")
    w, h = cfg["data"]["frame_wh"]
    dup = [k for k, v in Counter(b["cand_id"] for b in boxes).items() if v > 1]
    if dup:
        errors.append(f"boxes: duplicate cand_id {dup[:3]}")
    listed = {(f["clip"], f["frame_idx"]) for f in frames}
    allowed = set(class_names(cfg)) | {AMBIGUOUS}
    for b in boxes:
        x1, y1, x2, y2 = b["box_xyxy"]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            errors.append(f"boxes: outside frame or degenerate {b['cand_id']} {b['box_xyxy']}")
        if (b["clip"], b["frame_idx"]) not in listed:
            errors.append(f"boxes: frame not in frames.jsonl {b['cand_id']}")
        if b["y"] not in allowed or b["scored"] != (b["y"] != AMBIGUOUS):
            errors.append(f"boxes: bad y/scored {b['cand_id']}={b['y']}")
        if dino and (not 0 <= b["dino_score"] <= 1 or b["dino_label"] not in cfg["classes"]):
            errors.append(f"boxes: bad detector fields {b['cand_id']}")
        if len(errors) > 50:
            return errors
    errors += check_evidence([{"cand_id": b["cand_id"], **b["evidence"]} for b in boxes], boxes, cfg)
    return errors


def check_judgments(rows, boxes, cfg) -> list[str]:
    """Same boxes in the same order, every system on every box, valid class distributions, gates consistent."""
    errors = []
    if [r["cand_id"] for r in rows] != [b["cand_id"] for b in boxes]:
        return [f"judgments: cand_ids differ from boxes.jsonl ({len(rows)} vs {len(boxes)} rows)"]
    classes = set(class_names(cfg))
    expected = set(cfg["run"]["systems"])
    for r in rows:
        if set(r["systems"]) != expected:
            errors.append(f"judgments: systems {sorted(r['systems'])} != config for {r['cand_id']}")
        for s, d in r["systems"].items():
            errors += [f"judgments: {s} missing {f} for {r['cand_id']}" for f in REQUIRED["decision"] if f not in d]
            if d.get("pred") not in classes or set(d.get("probs", {})) != classes:
                errors.append(f"judgments: {s} bad pred/probs keys for {r['cand_id']}")
            elif s != "threshold" and abs(sum(d["probs"].values()) - 1) > 0.02:
                errors.append(f"judgments: {s} probs do not sum to 1 for {r['cand_id']}")
            if s in GATES and "vlm" in r["systems"]:
                if d.get("routed_to") not in ("jev", "vlm"):
                    errors.append(f"judgments: {s} bad routed_to for {r['cand_id']}")
                elif d["routed_to"] == "vlm" and d["pred"] != r["systems"]["vlm"]["pred"]:
                    errors.append(f"judgments: {s} routed to vlm but pred differs for {r['cand_id']}")
        if len(errors) > 50:
            break
    return errors


def check_metrics(run: RunDir) -> list[str]:
    if not run.file("metrics").exists():
        return ["metrics.json missing"]
    m = run.read_json("metrics")
    errors = [f"metrics: missing {k}" for k in METRIC_KEYS if k not in m]
    for system, row in m.get("systems", {}).items():
        errors += [f"metrics.systems.{system}: missing {k}" for k in SYSTEM_COLUMNS if k not in row]
    return errors


def check_run(run: RunDir) -> dict[str, list[str]]:
    """All contracts for one run folder, keyed by file."""
    cfg = run.config()
    report: dict[str, list[str]] = {}
    if not run.file("frames").exists():
        return {"frames": ["frames.jsonl missing"]}
    frames = run.read("frames")
    report["frames"] = check_frames(frames, cfg)
    if not run.file("boxes").exists():
        return report | {"boxes": ["boxes.jsonl missing"]}
    boxes = run.read("boxes")
    report["boxes"] = check_boxes(boxes, frames, cfg)
    report["judgments"] = (check_judgments(run.read("judgments"), boxes, cfg) if run.file("judgments").exists()
                           else ["judgments.jsonl missing"])
    report["metrics"] = check_metrics(run)
    return report
