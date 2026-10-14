import pytest
from conftest import REPO

from jevgate.config import ConfigError, deep_merge, load_config, parse_overrides


def test_smoke_inherits_base():
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    assert cfg["run"]["name"] == "smoke" and cfg["run"]["max_frames"] == 12
    assert list(cfg["classes"])[-1] == "background" and len(cfg["classes"]) == 7


def test_overrides_and_merge():
    assert parse_overrides(["a.b=0.3", "c=[1, 2]"]) == {"a": {"b": 0.3}, "c": [1, 2]}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_test_split_requires_dev_thresholds():
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "base.yaml", ["run.split=test"])
    assert load_config(REPO / "configs" / "test.yaml")["thresholds_file"] == "thresholds.yaml"


def test_rejects_bad_settings():
    base = REPO / "configs" / "base.yaml"
    for bad in (["splits.test=[valid.txt]"], ["label.background_iou=0.7"], ["run.systems=[threshold, gate]"],
                ["gate.primary=jev"], ["detector.phrase_to_class={fish: eel}"]):
        with pytest.raises(ConfigError):
            load_config(base, bad)
