from pathlib import Path

import pytest

from jevgate.core.config import load_config

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    return load_config(REPO / "configs" / "base.yaml")


def det(box_id, frame, box, cls="crab", conf=0.5, track=None, video="v"):
    """One detection row as written by VideoTracker."""
    return {"box_id": box_id, "video": video, "frame_idx": frame, "file": f"{video}-{frame:04d}.png",
            "box_xyxy": box, "cls": cls, "conf": conf, "track_uid": track}
