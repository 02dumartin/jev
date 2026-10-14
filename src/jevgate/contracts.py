"""Data contracts between stages (PLAN.md §3). Each check returns a list of violation strings."""

from collections import Counter

from .config import class_names
from .data import evenly_spaced, split_files
from .label import AMBIGUOUS
from .runio import RunDir

REQUIRED = {
    "frames": ["clip", "split", "frame_idx", "file", "gt_status"],
    "candidates": ["cand_id", "clip", "frame_idx", "box_xyxy", "dino_label", "dino_score"],
    "labels": ["cand_id", "max_iou", "y", "matched_object_id", "matched_gt_label", "scored"],
    "evidence": ["cand_id", "history_frame_idxs", "detector", "geometry", "track"],
    "decision": ["cand_id", "pred", "probs", "confidence", "latency_ms", "tokens_in", "tokens_out"],
}
METRIC_KEYS = ["systems", "n_box", "n_frame", "n_clip", "git_commit", "recall_upper_bound", "endpoints"]
SYSTEM_COLUMNS = ["accuracy", "macro_f1", "macro_precision", "macro_recall", "ece", "vlm_calls",
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
        if c["dino_label"] not in cfg["classes"]:
            errors.append(f"candidates: dino_label not a class {c['cand_id']}={c['dino_label']}")
    return errors


def check_one_to_one(ids_a: list[str], ids_b: list[str], name: str) -> list[str]:
    if ids_a != ids_b:
        missing, extra = set(ids_a) - set(ids_b), set(ids_b) - set(ids_a)
        return [f"{name}: cand_id mismatch (missing {len(missing)}, extra {len(extra)}, order differs="
                f"{not missing and not extra})"]
    return []


def check_labels(labels, cands, cfg) -> list[str]:
    errors = missing_fields(labels, REQUIRED["labels"], "labels")
    errors += check_one_to_one([c["cand_id"] for c in cands], [r["cand_id"] for r in labels], "labels")
    allowed = set(cfg["classes"]) | {AMBIGUOUS}
    errors += [f"labels: bad y {r['cand_id']}={r['y']}" for r in labels if r["y"] not in allowed]
    errors += [f"labels: scored flag wrong {r['cand_id']}" for r in labels if r["scored"] != (r["y"] != AMBIGUOUS)]
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


def check_decisions(run: RunDir, cands, cfg) -> list[str]:
    """Same candidates for every system, valid class distributions, gate rows consistent with the VLM."""
    errors = []
    classes = set(class_names(cfg))
    ids = [c["cand_id"] for c in cands]
    systems = run.systems_done()
    for system in systems:
        rows = run.read_decisions(system)
        errors += missing_fields(rows, REQUIRED["decision"], f"decisions/{system}")
        errors += check_one_to_one(ids, [r["cand_id"] for r in rows], f"decisions/{system}")
        for r in rows:
            if r["pred"] not in classes or set(r["probs"]) != classes:
                errors.append(f"decisions/{system}: bad pred/probs keys {r['cand_id']}")
            elif system != "threshold" and abs(sum(r["probs"].values()) - 1) > 0.02:
                errors.append(f"decisions/{system}: probs do not sum to 1 {r['cand_id']}")
            if len(errors) > 50:
                return errors
    if "vlm" in systems:
        vlm = {r["cand_id"]: r["pred"] for r in run.read_decisions("vlm")}
        for g in (s for s in systems if s.startswith("gate")):
            for r in run.read_decisions(g):
                if r.get("routed_to") not in ("jev", "vlm"):
                    errors.append(f"decisions/{g}: bad routed_to {r['cand_id']}")
                elif r["routed_to"] == "vlm" and r["pred"] != vlm[r["cand_id"]]:
                    errors.append(f"decisions/{g}: routed to vlm but pred differs {r['cand_id']}")
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
        report["labels"] = (check_labels(run.read("labels"), cands, cfg) if run.file("labels").exists()
                            else ["labels.jsonl missing"])
        report["evidence"] = (check_evidence(run.read("evidence"), cands, cfg) if run.file("evidence").exists()
                              else ["evidence.jsonl missing"])
        report["decisions"] = check_decisions(run, cands, cfg)
    report["metrics"] = check_metrics(run)
    return report
