"""Reading a clip as upright RGB frames, and marking known events on them."""

from pathlib import Path

import cv2
import numpy as np

CV_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def read_frames(path: Path, rotation_cw: int) -> list[np.ndarray]:
    """All frames of a clip, rotated clockwise by rotation_cw, as RGB arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if rotation_cw:
            im = cv2.rotate(im, CV_ROT[rotation_cw])
        frames.append(im[:, :, ::-1].copy())
    cap.release()
    return frames


def active_events(events: list[dict], frame_no: int) -> list[dict]:
    """Events whose 1-based frame span contains frame_no (1-based)."""
    return [e for e in events if e["frames"][0] <= frame_no <= e["frames"][1]]


def mark_events(img: np.ndarray, events: list[dict], radius_frac: float, color, label: bool = True) -> np.ndarray:
    """Circle (and label) at each event's interaction point; img is RGB, points are normalised."""
    out = img.copy()
    h, w = out.shape[:2]
    for e in events:
        if not e["point"]:
            continue
        c = (int(e["point"][0] * w), int(e["point"][1] * h))
        r = max(6, int(radius_frac * min(w, h)))
        cv2.circle(out, c, r, tuple(int(v) for v in color), 3, cv2.LINE_AA)
        if label:
            rgb = tuple(int(v) for v in color)
            cv2.putText(out, e["label"], (c[0] + r + 4, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 2, cv2.LINE_AA)
    return out


def sample_indices(n: int, k: int) -> list[int]:
    """k evenly spaced 0-based frame indices out of n."""
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, min(k, n))})
