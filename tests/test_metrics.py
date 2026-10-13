import pytest

from jevgate import metrics as M


def test_binary_metrics():
    m = M.binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (m["accuracy"], m["precision"], m["recall"]) == (0.5, 0.5, 0.5)
    assert M.binary_metrics([0, 0], [0, 0])["precision"] is None


def test_ece_perfect_and_overconfident():
    assert M.ece([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5], 10) == pytest.approx(0.0)
    assert M.ece([1, 0], [1.0, 1.0], 10) == pytest.approx(0.5)  # says 100% real, half are real


def test_ece_p_equal_one_falls_in_last_bin():
    rows = M.reliability_bins([1], [1.0], 15)
    assert rows[-1]["count"] == 1


def test_brier():
    assert M.brier([1, 0], [1.0, 0.0]) == 0.0
    assert M.brier([1, 0], [0.0, 1.0]) == 1.0


def test_block_bootstrap_identical_systems_has_zero_delta():
    keys = [("c", i) for i in range(30)]
    correct = {k: [1, 0] for k in keys}
    ci = M.block_bootstrap_delta(keys, correct, correct, 200, 5, 0)
    assert ci["delta"] == 0 and ci["ci_low"] == 0 and ci["ci_high"] == 0 and ci["n_blocks"] == 6


def test_block_bootstrap_blocks_stay_within_clip():
    keys = [("a", 0), ("a", 1), ("b", 0)]
    corr = {k: [1] for k in keys}
    assert M.block_bootstrap_delta(keys, corr, corr, 10, 10, 0)["n_blocks"] == 2


def test_endpoints_stop_at_first_failure():
    steps = [{"min_delta_acc_ci_low": -0.02, "max_vlm_call_ratio": 0.5},
             {"min_delta_acc_ci_low": -0.01, "max_vlm_call_ratio": 0.5},
             {"min_delta_acc_ci_low": -0.01, "max_vlm_call_ratio": 0.3}]
    res = [r["result"] for r in M.evaluate_endpoints(-0.015, 0.4, steps)]
    assert res == ["pass", "fail", "not_tested"]
    assert [r["result"] for r in M.evaluate_endpoints(None, 0.4, steps)] == ["not_tested"] * 3
