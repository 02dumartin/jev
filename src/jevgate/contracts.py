"""Data contracts between stages (PLAN.md §3). Each check returns a list of violation strings."""

from collections import Counter

from .data import list_clip_frames
from .runio import RunDir

REQUIRED = {
    "frames": ["clip", "split", "frame_idx", "file", "gt_status"],
    "candidates": ["cand_id", "clip", "frame_idx", "box_xyxy", "label", "dino_score"],
    "labels": ["cand_id", "max_iou", "y", "matched_object_id"],
    "evidence": ["cand_id", "history_frame_idxs", "detector", "geometry", "track"],
    "decision": ["cand_id", "keep", "p_real", "latency_ms", "tokens_in", "tokens_out"],
}
METRIC_KEYS = ["systems", "n_box", "n_frame", "n_clip", "git_commit", "recall_upper_bound", "endpoints"]
SYSTEM_COLUMNS = ["accuracy", "precision", "recall", "ece", "vlm_calls", "latency_ms_median", "total_time_s"]


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
    errors = missing_fields(frames, REQUIRED["frames"], "frames")
    limit = cfg["run"].get("max_frames_per_clip")
    counts = Counter(f["clip"] for f in frames)
    for clip in cfg["clips"][cfg["run"]["split"]]:
        expected = len(list_clip_frames(cfg, clip))
        expected = min(expected, limit) if limit else expected
        if counts.get(clip, 0) != expected:
            errors.append(f"frames: clip {clip} has {counts.get(clip, 0)} rows, expected {expected}")
    if any(f["gt_status"] == "unknown" for f in frames):
        errors.append("frames: some frames have unknown GT (outside every split list)")
    return errors


def check_candidates(cands, cfg) -> list[str]:
    errors = missing_fields(cands, REQUIRED["candidates"], "candidates")
    w, h = cfg["data"]["frame_wh"]
    dup = [k for k, v in Counter(c["cand_id"] for c in cands).items() if v > 1]
    if dup:
        errors.append(f"candidates: duplicate cand_id {dup[:3]}")
    for c in cands:
        x1, y1, x2, y2 = c["box_xyxy"]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            errors.append(f"candidates: box outside frame or degenerate {c['cand_id']} {c['box_xyxy']}")
        if not 0 <= c["dino_score"] <= 1:
            errors.append(f"candidates: score outside [0,1] {c['cand_id']}")
    return errors


def check_one_to_one(ids_a: list[str], ids_b: list[str], name: str) -> list[str]:
    if ids_a != ids_b:
        missing, extra = set(ids_a) - set(ids_b), set(ids_b) - set(ids_a)
        return [f"{name}: cand_id mismatch (missing {len(missing)}, extra {len(extra)}, order differs="
                f"{not missing and not extra})"]
    return []


def check_labels(labels, cands) -> list[str]:
    errors = missing_fields(labels, REQUIRED["labels"], "labels")
    errors += check_one_to_one([c["cand_id"] for c in cands], [r["cand_id"] for r in labels], "labels")
    errors += [f"labels: bad y {r['cand_id']}={r['y']}" for r in labels if r["y"] not in (0, 1, None)]
    return errors


def check_evidence(evidence, cands, cfg) -> list[str]:
    """Schema, 1:1 ids and the causality rule: history frames must be strictly earlier and within K."""
    errors = missing_fields(evidence, REQUIRED["evidence"], "evidence")
    errors += check_one_to_one([c["cand_id"] for c in cands], [e["cand_id"] for e in evidence], "evidence")
    frame_of = {c["cand_id"]: c["frame_idx"] for c in cands}
    k = cfg["evidence"]["history_frames"]
    for e in evidence:
        idx = frame_of.get(e["cand_id"])
        hist = e["history_frame_idxs"]
        if any(h >= idx for h in hist):
            errors.append(f"evidence: future/current frame leak in {e['cand_id']}: {hist} vs {idx}")
        if len(hist) > k:
            errors.append(f"evidence: history longer than K={k} in {e['cand_id']}")
    return errors


def check_decisions(run: RunDir, cands) -> list[str]:
    errors = []
    ids = [c["cand_id"] for c in cands]
    systems = run.systems_done()
    for system in systems:
        rows = run.read_decisions(system)
        errors += missing_fields(rows, REQUIRED["decision"], f"decisions/{system}")
        errors += check_one_to_one(ids, [r["cand_id"] for r in rows], f"decisions/{system}")
        for r in rows:
            if r["p_real"] is not None and not 0 <= r["p_real"] <= 1:
                errors.append(f"decisions/{system}: p_real outside [0,1] {r['cand_id']}")
    if "gate" in systems and "vlm" in systems:
        vlm = {r["cand_id"]: r["keep"] for r in run.read_decisions("vlm")}
        for r in run.read_decisions("gate"):
            if r.get("routed_to") not in ("jev", "vlm"):
                errors.append(f"decisions/gate: bad routed_to {r['cand_id']}")
            elif r["routed_to"] == "vlm" and r["keep"] != vlm[r["cand_id"]]:
                errors.append(f"decisions/gate: routed to vlm but keep differs {r['cand_id']}")
    return errors


def check_metrics(run: RunDir) -> list[str]:
    if not run.file("metrics").exists():
        return ["metrics.json missing"]
    m = run.read_json("metrics.json")
    errors = [f"metrics: missing {k}" for k in METRIC_KEYS if k not in m]
    for system, row in m.get("systems", {}).items():
        errors += [f"metrics.systems.{system}: missing {k}" for k in SYSTEM_COLUMNS if k not in row]
    return errors


def check_run(run: RunDir) -> dict[str, list[str]]:
    """All contracts for one run folder, keyed by stage. Stages without output are reported as such."""
    cfg = run.config()
    report: dict[str, list[str]] = {}
    if not run.file("frames").exists():
        return {"frames": ["frames.jsonl missing"]}
    report["frames"] = check_frames(run.read("frames"), cfg)
    cands = run.read("candidates") if run.file("candidates").exists() else None
    report["candidates"] = check_candidates(cands, cfg) if cands is not None else ["candidates.jsonl missing"]
    if cands is not None:
        report["labels"] = (check_labels(run.read("labels"), cands) if run.file("labels").exists()
                            else ["labels.jsonl missing"])
        report["evidence"] = (check_evidence(run.read("evidence"), cands, cfg) if run.file("evidence").exists()
                              else ["evidence.jsonl missing"])
        report["decisions"] = check_decisions(run, cands)
    report["metrics"] = check_metrics(run)
    return report
