"""Clip metadata for the demo: dataset action events, interaction points, and the upright rotation.

Rotation: the subject's body axis (hip -> head) from the dataset's pose keypoints, averaged over frames, picks the
multiple of 90 degrees (clockwise) that makes it point up. frames.json is ~2 GB, so it is streamed with ijson.
"""

import json
import math
from pathlib import Path

import numpy as np

ROTATIONS = (0, 90, 180, 270)  # clockwise degrees


def rotate_vector(dx: float, dy: float, deg: int) -> tuple[float, float]:
    """Image-space vector (y down) after rotating the image clockwise by deg."""
    return {0: (dx, dy), 90: (-dy, dx), 180: (-dx, -dy), 270: (dy, -dx)}[deg]


def rotate_point(x: float, y: float, deg: int) -> tuple[float, float]:
    """Normalised point after rotating a square image clockwise by deg."""
    return {0: (x, y), 90: (1 - y, x), 180: (1 - x, 1 - y), 270: (y, 1 - x)}[deg]


def upright_rotation(axis: tuple[float, float] | None) -> int:
    """Clockwise rotation that turns the hip->head axis closest to straight up (0 when unknown)."""
    if axis is None:
        return 0
    return max(ROTATIONS, key=lambda d: -rotate_vector(*axis, d)[1])


def _num(v) -> float:
    return float(v) if isinstance(v, int | float) else math.nan


def body_axis(frames: list[dict], head: list[int], hip: list[int], min_frames: int) -> tuple[float, float] | None:
    """Mean hip->head vector over frames where both ends are detected, or None with too few frames."""
    vecs = []
    for fr in frames:
        kps = (fr.get("pose_keypoints") or {}).get("keypoints") or []
        if not kps:
            continue
        pts = np.asarray([[_num(a), _num(b)] for a, b in kps[0]["points"]], dtype=float)
        h, p = np.nanmean(pts[head], axis=0), np.nanmean(pts[hip], axis=0)
        if np.all(np.isfinite(h)) and np.all(np.isfinite(p)):
            vecs.append(h - p)
    if len(vecs) < min_frames:
        return None
    v = np.mean(vecs, axis=0)
    return float(v[0]), float(v[1])


def stream_frames(frames_json: Path, sample_ids: set[str]) -> dict[str, list[dict]]:
    """Per-frame records of the wanted samples (by FiftyOne sample _id), streamed from frames.json."""
    import ijson

    out: dict[str, list[dict]] = {}
    with frames_json.open("rb") as f:
        for fr in ijson.items(f, "frames.item", use_float=True):
            sid = fr["_sample_id"]["$oid"]
            if sid in sample_ids:
                out.setdefault(sid, []).append(fr)
    return out


def events_of(sample: dict) -> list[dict]:
    """Annotated actions with their 1-based frame span and the interaction point (same order in the dataset)."""
    acts = sample["actions"]["detections"]
    pts = sample.get("interaction_points", {}).get("detections", [])
    out = []
    for k, a in enumerate(acts):
        p = (
            pts[k]
            if k < len(pts) and pts[k]["label"] == a["label"]
            else next((q for q in pts if q["label"] == a["label"]), None)
        )
        out.append(
            {
                "label": a["label"],
                "frames": a["support"],
                "point": [p["interaction_x"], p["interaction_y"]] if p else None,
            }
        )
    return sorted(out, key=lambda e: e["frames"][0])


def build_meta(cfg: dict, n_frames_of) -> list[dict]:
    """One record per selected clip: rotation, events (points rotated into the upright view), timing."""
    root = Path(cfg["data"]["root"])
    selection = json.loads((root / cfg["data"]["selection"]).read_text())
    by_path = {s["filepath"]: s for s in json.loads((root / cfg["data"]["samples"]).read_text())["samples"]}
    samples = {r["sample_id"]: by_path[r["filepath"]] for r in selection}
    frames = stream_frames(root / cfg["data"]["frames"], {s["_id"]["$oid"] for s in samples.values()})
    rc = cfg["rotation"]
    rows = []
    for r in selection:
        s = samples[r["sample_id"]]
        axis = body_axis(frames.get(s["_id"]["$oid"], []), rc["head_joints"], rc["hip_joints"], rc["min_frames"])
        deg = upright_rotation(axis)
        events = events_of(s)
        for e in events:
            e["point"] = list(rotate_point(*e["point"], deg)) if e["point"] else None
        rows.append(
            {
                "sample_id": r["sample_id"],
                "rotation_cw": deg,
                "pose_axis": axis,
                "n_frames": n_frames_of(r["sample_id"]),
                "duration_s": round(s["segment_duration"], 2),
                "events": events,
            }
        )
    return rows
