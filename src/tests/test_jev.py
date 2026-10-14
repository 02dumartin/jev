import pytest

from jevgate.core.config import class_names
from jevgate.judges.base import Budget, BudgetExceeded
from jevgate.judges.jev import JevError, build_request, question_keys, validate_choice, validate_score

CLASSES = ["fish", "crab", "background"]


def test_validate_choice():
    p = validate_choice({"probabilities": {"fish": 0.2, "crab": 0.79, "background": 0.0}}, CLASSES)
    assert p["crab"] == pytest.approx(0.79 / 0.99)
    for bad in ({"probabilities": {"fish": 1.0}}, {"probabilities": {"fish": 0.5, "crab": 0.1, "background": 0.1}},
                {"probabilities": {"fish": 1.2, "crab": 0, "background": 0}}, None, {}):
        with pytest.raises(JevError):
            validate_choice(bad, CLASSES)


def test_validate_score_shifts_to_one_based():
    out = validate_score({"score": 1.96, "probabilities": {"0": 0.01, "1": 0.43, "2": 0.19, "3": 0.34, "4": 0.03},
                          "confidence": 0.19}, 5)
    assert out["need_vlm"] == pytest.approx(2.96) and out["need_vlm_probs"]["2"] == 0.43
    for bad in ({"score": 5.0, "probabilities": {}}, {"score": "2", "probabilities": {}}, {}):
        with pytest.raises(JevError):
            validate_score(bad, 5)


def _ev():
    return {"detector": {"label": "fish", "score": 0.3}, "geometry": {"area_frac": 0.01}, "track": {}}


def test_request_has_class_and_need_vlm_per_box_and_hides_ids(cfg):
    body = build_request(cfg, [("id-a", _ev()), ("id-b", _ev())])
    assert set(body["questions"]) == {*question_keys("id-a"), *question_keys("id-b")}
    assert set(body["state"]["candidate_boxes"]) == {"B1", "B2"}
    assert "id-a" not in str(body["state"])
    q_class, q_need = question_keys("id-b")
    assert body["questions"][q_class]["type"] == "choice"
    assert set(body["questions"][q_class]["criteria"]) == set(class_names(cfg))
    assert body["questions"][q_need]["type"] == "score" and len(body["questions"][q_need]["criteria"]) == 5


def test_budget_stops_requests(cfg):
    cfg["budget"]["max_jev_requests"] = 1
    b = Budget(cfg)
    b.reserve_jev_request()
    with pytest.raises(BudgetExceeded):
        b.reserve_jev_request()


def test_resume_reuses_identical_request_and_checkpoints(cfg, tmp_path, monkeypatch):
    import json

    from jevgate.judges.jev import JevJudge

    monkeypatch.setenv(cfg["jev"]["api_key_env"], "dummy")
    group = [("id-a", _ev())]
    body = build_request(cfg, group)
    q_class, q_need = question_keys("id-a")
    probs = dict.fromkeys(class_names(cfg), 0.0) | {"crab": 1.0}
    response = {"model": "jev-x", "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {q_class: {"probabilities": probs, "confidence": 1.0},
                            q_need: {"score": 0.0, "probabilities": {"0": 1.0}}}}
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"cand_ids": ["id-a"], "request": body, "response": response, "latency_ms": 5}) + "\n")
    judge = JevJudge(cfg, Budget(cfg), checkpoint=tmp_path / "new.jsonl", resume=[old])
    monkeypatch.setattr(judge.client, "post", lambda b: (_ for _ in ()).throw(AssertionError("no HTTP call")))
    row = judge._run_group(group)[0][0]
    assert row["pred"] == "crab" and row["need_vlm"] == 1.0
    assert json.loads((tmp_path / "new.jsonl").read_text())["reused"] is True
