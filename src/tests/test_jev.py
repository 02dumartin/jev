import pytest

from jevgate.judges.base import Budget, BudgetExceeded
from jevgate.judges.jev import JevError, build_request, question_keys, validate_choice, validate_noul


def test_validate_noul():
    assert validate_noul({"type": "noul", "noul": 0.23}) == 0.23
    for bad in ({"noul": 1.2}, {"noul": "0.5"}, {}, None, {"noul": float("nan")}):
        with pytest.raises(JevError):
            validate_noul(bad)


def test_validate_choice():
    assert validate_choice({"probabilities": {"a": 0.5, "b": 0.5}}, ["a", "b"]) == {"a": 0.5, "b": 0.5}
    with pytest.raises(JevError):
        validate_choice({"probabilities": {"a": 1.0}}, ["a", "b"])


def _track(uid, cls="crab"):
    return {"track_uid": uid, "video": "secret-video", "majority_class": cls,
            "evidence": {"classes": {"majority_class": cls}}}


def test_request_shares_state_and_hides_ids(cfg):
    body = build_request(cfg, [_track("0319-1#4"), _track("0319-1#9", "fish")])
    assert set(body["state"]["tracks"]) == {"T1", "T2"}
    assert "secret-video" not in str(body["state"]) and "#4" not in str(body["state"])
    q_ok, q_fix = question_keys("0319-1#9")
    assert body["questions"][q_ok]["type"] == "noul"
    assert body["questions"][q_ok]["instructions"]["question"] == "Is this track a real fish?"
    assert "not_an_animal" in body["questions"][q_fix]["criteria"]


def test_budget_stops_requests(cfg):
    cfg["budget"]["max_jev_requests"] = 1
    b = Budget(cfg)
    b.reserve_jev_request()
    with pytest.raises(BudgetExceeded):
        b.reserve_jev_request()
