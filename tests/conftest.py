from pathlib import Path

import pytest

from jevgate.config import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(REPO / "configs" / "base.yaml")


def cand(cid, frame, box, score=0.5, clip="c", label="fish"):
    return {"cand_id": cid, "clip": clip, "frame_idx": frame, "box_xyxy": box, "dino_label": label,
            "dino_score": score}
