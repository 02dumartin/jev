"""Data contracts between stages (PLAN.md v6 §7). Each check returns a list of violation strings."""

from collections import Counter

from jevgate.core.config import class_names, output_classes
from jevgate.core.runio import RunDir

REQUIRED = {
    "detections": ["box_id", "video", "frame_idx", "file", "box_xyxy", "cls", "conf", "track_uid"],
    "tracks": ["track_uid", "video", "box_ids", "rep_box_id", "majority_class", "evidence"],
    "box_labels": ["box_id", "gt_status", "max_iou", "y", "scored"],
    "track_labels": ["track_uid", "y_track", "type"],
    "jev": ["track_uid", "p_ok", "fix_probs", "latency_ms", "tokens_in", "tokens_out"],
    "vlm": ["track_uid", "p_yes", "latency_ms", "rep_box_id"],
}
METRIC_KEYS = ["n", "systems", "primary", "endpoints", "track_level", "missed_objects", "timing", "thresholds"]


def missing_fields(rows: list[dict], fields: list[str], name: str) -> list[str]:
    errors = []
    for k, row in enumerate(rows):
        absent = [f for f in fields if f not in row]
        if absent:
            errors.append(f"{name} row {k}: missing {absent}")
            if len(errors) >= 5:
                return errors + [f"{name}: further rows not listed"]
    return errors


def same_ids(expected: list[str], got: list[str], name: str) -> list[str]:
    if expected == got:
        return []
    missing, extra = set(expected) - set(got), set(got) - set(expected)
    return [f"{name}: id mismatch (missing {len(missing)}, extra {len(extra)}, "
            f"order differs={not missing and not extra})"]


def check_detections(dets: list[dict], cfg: dict) -> list[str]:
    errors = missing_fields(dets, REQUIRED["detections"], "detections")
    w, h = cfg["data"]["frame_wh"]
    dup = [k for k, v in Counter(d["box_id"] for d in dets).items() if v > 1]
    if dup:
        errors.append(f"detections: duplicate box_id {dup[:3]}")
    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        if not (-1 <= x1 < x2 <= w + 1 and -1 <= y1 < y2 <= h + 1):
            errors.append(f"detections: box outside frame {d['box_id']} {d['box_xyxy']}")
        if not cfg["track"]["conf"] - 1e-6 <= d["conf"] <= 1:
            errors.append(f"detections: conf outside [track.conf, 1] {d['box_id']}")
        if d["cls"] not in class_names(cfg):
            errors.append(f"detections: unknown class {d['cls']}")
    return errors[:20]


def check_tracks(tracks: list[dict], dets: list[dict]) -> list[str]:
    """Every tracked detection belongs to exactly its track; majority class is one of its votes."""
    errors = missing_fields(tracks, REQUIRED["tracks"], "tracks")
    by_track = {}
    for d in dets:
        if d["track_uid"] is not None:
            by_track.setdefault(d["track_uid"], set()).add(d["box_id"])
    if set(by_track) != {t["track_uid"] for t in tracks}:
        errors.append("tracks: track_uids differ from the detections' track_uids")
    for t in tracks:
        if set(t["box_ids"]) != by_track.get(t["track_uid"], set()):
            errors.append(f"tracks: box_ids of {t['track_uid']} differ from detections")
        if t["rep_box_id"] not in t["box_ids"]:
            errors.append(f"tracks: rep_box_id outside {t['track_uid']}")
        if t["majority_class"] not in t["evidence"]["classes"]["votes"]:
            errors.append(f"tracks: majority class not voted in {t['track_uid']}")
    return errors[:20]


def check_labels(box_labels, track_labels, dets, tracks, cfg) -> list[str]:
    errors = missing_fields(box_labels, REQUIRED["box_labels"], "box_labels")
    errors += missing_fields(track_labels, REQUIRED["track_labels"], "track_labels")
    errors += same_ids([d["box_id"] for d in dets], [r["box_id"] for r in box_labels], "box_labels")
    errors += same_ids([t["track_uid"] for t in tracks], [r["track_uid"] for r in track_labels], "track_labels")
    valid = set(output_classes(cfg)) | {None}
    errors += [f"box_labels: bad y {r['box_id']}={r['y']}" for r in box_labels if r["y"] not in valid][:5]
    errors += [f"track_labels: bad y_track {r['track_uid']}" for r in track_labels if r["y_track"] not in (0, 1, None)]
    return errors


def check_judges(run: RunDir, tracks: list[dict]) -> list[str]:
    errors, uids = [], [t["track_uid"] for t in tracks]
    for name, key in (("jev", "p_ok"), ("vlm", "p_yes")):
        if not run.exists(name):
            continue
        rows = run.read(name)
        errors += missing_fields(rows, REQUIRED[name], name)
        errors += same_ids(uids, [r["track_uid"] for r in rows], name)
        errors += [f"{name}: {key} outside [0,1] {r['track_uid']}" for r in rows if not 0 <= r[key] <= 1][:5]
    return errors


def check_judgments(run: RunDir, dets: list[dict]) -> list[str]:
    if not run.exists("judgments"):
        return ["judgments.jsonl missing"]
    rows = run.read("judgments")
    errors = same_ids([d["box_id"] for d in dets], [r["box_id"] for r in rows], "judgments")
    systems = [s for s in run.config()["run"]["systems"]]
    valid = set(output_classes(run.config()))
    for r in rows:
        bad = [s for s in systems if r.get(s) not in valid]
        if bad:
            errors.append(f"judgments: {r['box_id']} has no valid prediction for {bad}")
            break
        if r["track_uid"] is None and any(r[s] != "background" for s in systems if s != "yolo"):
            errors.append(f"judgments: untracked box {r['box_id']} kept by a track system")
            break
    return errors


def check_metrics(run: RunDir) -> list[str]:
    if not run.exists("metrics"):
        return ["metrics.json missing"]
    m = run.read_json("metrics")
    return [f"metrics: missing {k}" for k in METRIC_KEYS if k not in m]


def check_run(run: RunDir) -> dict[str, list[str]]:
    """All contracts for one run folder, keyed by stage."""
    cfg = run.config()
    if not run.exists("detections"):
        return {"yolo": ["detections.jsonl missing"]}
    dets = run.read("detections")
    tracks = run.read("tracks") if run.exists("tracks") else []
    report = {"detections": check_detections(dets, cfg), "tracks": check_tracks(tracks, dets)}
    if run.exists("box_labels") and run.exists("track_labels"):
        report["labels"] = check_labels(run.read("box_labels"), run.read("track_labels"), dets, tracks, cfg)
    else:
        report["labels"] = ["label files missing"]
    report["judges"] = check_judges(run, tracks)
    report["judgments"] = check_judgments(run, dets)
    report["metrics"] = check_metrics(run)
    return report
