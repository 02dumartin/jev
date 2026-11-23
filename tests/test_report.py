"""Saved-run integration: evaluate without the model/API and reject incomplete joins."""

import json
from pathlib import Path

import pytest
import yaml

from jevgate.core.config import load_config
from jevgate.core.runio import write_jsonl
from jevgate.decision.policy import vlm_cards
from jevgate.evaluation.report import evaluate_run

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path):
    cfg = load_config(REPO / "configs/retail.yaml")
    cfg["jev"]["single_question_compare"] = False
    cfg["vlm_only"]["enabled"] = False
    (tmp_path / "run.yaml").write_text(yaml.safe_dump({"config": cfg}))
    write_jsonl(
        tmp_path / "data/meta.jsonl",
        [{"sample_id": "a", "events": [{"label": "take"}], "rotation_cw": 0, "pose_axis": None}],
    )
    write_jsonl(tmp_path / "vlm/output/states.jsonl", [{"sample_id": "a", "state": {}, "vlm_s": 2}])
    cards = vlm_cards(cfg, {})
    for c in cards:
        c.update(p=0.8 if c["key"] == "product_taken" else 0.1, top=0.8)
    write_jsonl(tmp_path / "jev/output/decisions.jsonl", [{"sample_id": "a", "cards": cards}])
    write_jsonl(
        tmp_path / "jev/output/responses.jsonl",
        [{"sample_id": "a", "latency_ms": 200, "response": {"model": "fixture", "usage": {}}}],
    )
    return tmp_path


def test_report_without_optional_comparisons(run):
    result = evaluate_run(run, run / "report")
    assert result["product_taken"]["brier"] == pytest.approx(0.04)
    assert result["summary"]["end_to_end_s_median"] == 2.2
    assert result["baselines"]["vlm_valid"] == 0
    assert (run / "report/calibration.png").exists()
    assert json.loads((run / "report/metrics.json").read_text())["source_sha256"]


def test_report_rejects_mismatched_ids(run):
    write_jsonl(run / "jev/output/decisions.jsonl", [{"sample_id": "wrong", "cards": []}])
    with pytest.raises(ValueError, match="mismatch"):
        evaluate_run(run, run / "report")


def test_report_rejects_duplicate_ids(run):
    write_jsonl(run / "jev/output/decisions.jsonl", [{"sample_id": "a"}, {"sample_id": "a"}])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_run(run, run / "report")
