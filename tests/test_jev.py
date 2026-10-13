import pytest

from jevgate.judges.base import Budget, BudgetExceeded
from jevgate.judges.jev import JevError, build_request, validate_noul


def test_validate_noul():
    assert validate_noul({"type": "noul", "noul": 0.23}) == 0.23
    assert validate_noul({"noul": 1}) == 1.0
    for bad in ({"noul": 1.2}, {"noul": "0.5"}, {}, None, {"noul": True}):
        with pytest.raises(JevError):
            validate_noul(bad)


def _ev():
    return {"detector": {"label": "fish", "score": 0.3}, "geometry": {"area_frac": 0.01}, "track": {}}


def test_per_frame_request_has_one_question_per_box_and_hides_ids(cfg):
    body = build_request(cfg, [("id-a", _ev()), ("id-b", _ev())])
    assert set(body["questions"]) == {"id-a", "id-b"}
    assert set(body["state"]["candidate_boxes"]) == {"B1", "B2"}
    assert "id-a" not in str(body["state"])
    assert body["questions"]["id-b"]["instructions"]["question"] == cfg["prompts"]["question"]


def test_budget_stops_requests(cfg):
    cfg["budget"]["max_jev_requests"] = 1
    b = Budget(cfg)
    b.reserve_jev_request()
    with pytest.raises(BudgetExceeded):
        b.reserve_jev_request()
