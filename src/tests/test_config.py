import pytest
from conftest import REPO

from jevgate.core.config import ConfigError, deep_merge, load_config, output_classes, parse_overrides


def test_smoke_inherits_base():
    cfg = load_config(REPO / "configs" / "smoke.yaml")
    assert cfg["run"]["name"] == "smoke" and cfg["run"]["max_videos"] == 2
    assert cfg["track"]["weights"] == "train/yolo12s_video/weights/best.pt"


def test_yolo_training_configs_still_load():
    for name in ("yolo.yaml", "yolo_orig.yaml", "yolo_video_paper416.yaml", "yolo_orig_paper416.yaml"):
        assert load_config(REPO / "configs" / name)["yolo"]["name"].startswith("yolo12s")


def test_overrides_and_merge():
    assert parse_overrides(["a.b=0.3", "c=[1, 2]"]) == {"a": {"b": 0.3}, "c": [1, 2]}
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_test_split_requires_dev_thresholds():
    with pytest.raises(ConfigError):
        load_config(REPO / "configs" / "base.yaml", ["run.split=test"])
    assert load_config(REPO / "configs" / "test.yaml")["thresholds_file"] == "thresholds.yaml"


def test_rejects_bad_settings():
    base = REPO / "configs" / "base.yaml"
    for bad in (["label.background_iou=0.7"], ["run.systems=[yolo, gate]"], ["prompts.question=Is it real?"]):
        with pytest.raises(ConfigError):
            load_config(base, bad)


def test_output_classes_end_with_background(cfg):
    assert output_classes(cfg)[-1] == "background" and len(output_classes(cfg)) == 7


def test_every_label_has_a_colour(cfg):
    from jevgate.outputs.colors import class_colors_bgr

    colours = class_colors_bgr(cfg)
    assert list(colours) == output_classes(cfg) and len(set(colours.values())) == len(colours)
