from pathlib import Path

import pytest

from jev.core.config import ConfigError, load_config
from jev.decision.policy import action, card, request, vlm_cards
from jev.models.budget import Budget, BudgetExceeded
from jev.models.jev import JevError, validate_noul
from jev.pipeline import pick_for_video
from jev.preprocess.metadata import rotate_point, upright_rotation

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(REPO / "configs/retail.yaml")


def test_config_overrides_and_validation(cfg):
    loaded = load_config(REPO / "configs/retail.yaml", ["run.max_clips=5", "render.include=[]"])
    assert loaded["run"]["max_clips"] == 5
    assert loaded["render"]["include"] == []
    with pytest.raises(ConfigError):
        load_config(REPO / "configs/retail.yaml", ["jev.confident=1.1"])


def test_rotation():
    assert upright_rotation((1, 0)) == 270
    assert upright_rotation(None) == 0
    assert rotate_point(0.2, 0.4, 90) == (0.6, 0.2)


def test_request_and_unsure_action(cfg):
    body = request(cfg, {"people_count": 1}, list(cfg["questions"]))
    assert len(body["questions"]) == 6
    assert body["state"]["observation"] == {"people_count": 1}
    cards = vlm_cards(
        cfg,
        {
            "needs_help": "no",
            "product_taken": "no",
            "fridge_open": "no",
            "crowding": "one shopper",
            "shopper_activity": "browsing",
            "staff_action": "none",
        },
    )
    assert action(cfg, cards)["badge"] == "NO ACTION"
    cards[1] = card("product_taken", cfg["questions"]["product_taken"], {"noul": 0.55}, 0.75)
    assert action(cfg, cards)["badge"] == "MANAGER CHECK"
    cards[2] = card("fridge_open", cfg["questions"]["fridge_open"], {"noul": 0.8}, 0.75)
    assert action(cfg, cards)["badge"] == "CLOSE FRIDGE"


def test_budget_and_invalid_probability(cfg):
    cfg["budget"]["max_jev_requests"] = 1
    budget = Budget(cfg)
    budget.reserve_jev_request()
    with pytest.raises(BudgetExceeded):
        budget.reserve_jev_request()
    with pytest.raises(JevError):
        validate_noul({"noul": float("nan")})


def test_video_selection():
    rows = [
        {"sample_id": "a", "action": {"badge": "NO ACTION"}},
        {"sample_id": "b", "action": {"badge": "MANAGER CHECK"}},
    ]
    assert pick_for_video(rows, 1, 0, include=["b"]) == [rows[1]]
    with pytest.raises(ValueError):
        pick_for_video(rows, 1, 0, include=["missing"])
