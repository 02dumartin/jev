import pytest

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
    assert set(body["questions"][q_class]["criteria"]) == set(cfg["classes"])
    assert body["questions"][q_need]["type"] == "score" and len(body["questions"][q_need]["criteria"]) == 5


def test_budget_stops_requests(cfg):
    cfg["budget"]["max_jev_requests"] = 1
    b = Budget(cfg)
    b.reserve_jev_request()
    with pytest.raises(BudgetExceeded):
        b.reserve_jev_request()
