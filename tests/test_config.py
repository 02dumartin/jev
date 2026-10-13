import pytest
from conftest import REPO

from jevgate.config import ConfigError, deep_merge, load_config, parse_overrides


def test_smoke_inherits_base():
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    assert cfg["run"]["name"] == "smoke" and cfg["run"]["max_frames_per_clip"] == 5
    assert cfg["detector"]["model"] == "IDEA-Research/grounding-dino-base"


def test_overrides_and_merge():
    assert parse_overrides(["a.b=0.3", "c=[1, 2]"]) == {"a": {"b": 0.3}, "c": [1, 2]}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_test_split_requires_dev_thresholds():
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "base.yaml", ["run.split=test"])
    assert load_config(REPO / "configs" / "test.yaml")["thresholds_file"] == "thresholds.yaml"


def test_rejects_overlapping_clips_and_bad_bands():
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "base.yaml",
                    ["clips.test=[2019-03-19_17-02-30to2019-03-19_17-02-42_1]"])
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "base.yaml", ["label.fake_iou=0.7"])
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "base.yaml", ["run.systems=[threshold, gate]"])
