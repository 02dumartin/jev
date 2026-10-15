from conftest import det

from jevgate.judges.gate import (
    band_gate_decisions,
    box_predictions,
    fit_accept_threshold,
    fit_band,
    fit_conf_threshold,
    fit_tau,
    gate_decisions,
    tau_curve,
)


def test_gate_routes_uncertain_tracks():
    jev = [{"track_uid": u, "p_ok": p, "latency_ms": 10.0} for u, p in (("a", 0.95), ("b", 0.6), ("c", 0.02))]
    out = gate_decisions(jev, {"a": False, "b": False, "c": True}, {"a": 100.0, "b": 100.0, "c": 100.0}, 0.9)
    assert [r["routed_to"] for r in out] == ["jev", "vlm", "jev"]
    assert [r["accept"] for r in out] == [True, False, False]
    assert out[1]["latency_ms"] == 110.0


def test_box_predictions_per_system():
    dets = [det("1", 1, [0, 0, 1, 1], "shrimp", 0.6, "t"), det("2", 2, [0, 0, 1, 1], "crab", 0.2, "t"),
            det("3", 3, [0, 0, 1, 1], "fish", 0.9, None)]
    maj = {"t": "crab"}
    assert box_predictions(dets, maj, None, 0.5, per_frame=True) == {"1": "shrimp", "2": "background", "3": "fish"}
    # untracked box 3 follows the per-frame rule in every track system
    assert box_predictions(dets, maj, None, 0.5) == {"1": "crab", "2": "crab", "3": "fish"}
    assert box_predictions(dets, maj, {"t": False}, 0.95) == {"1": "background", "2": "background", "3": "background"}


def test_gate_latency_includes_describer():
    out = gate_decisions([{"track_uid": "a", "p_ok": 0.6, "latency_ms": 10.0}], {"a": True}, {"a": 100.0}, 0.9,
                         {"a": 50.0})
    assert out[0]["routed_to"] == "vlm" and out[0]["latency_ms"] == 160.0


def test_band_gate_and_fit():
    rows = [{"track_uid": u, "p_yes": p, "latency_ms": 5.0} for u, p in (("a", 0.9), ("b", 0.01), ("c", 0.3))]
    out = band_gate_decisions(rows, {"a": False, "b": True, "c": True}, {"a": 1.0, "b": 1.0, "c": 1.0}, 0.05, 0.8)
    assert [(r["routed_to"], r["accept"]) for r in out] == [("small_vlm", True), ("small_vlm", False), ("vlm", True)]
    lo, hi = fit_band([1, 1, 0, 0, 1], [0.9, 0.8, 0.01, 0.02, 0.3], 0.0)
    assert 0.02 <= lo < hi <= 0.3  # every track handled alone without an error
    assert fit_band([1, 0], [0.5, 0.5], 0.0) == (-1.0, 2.0)


def test_fit_thresholds():
    assert fit_conf_threshold(["crab", "background"], ["crab", "crab"], [0.6, 0.2]) == 0.6
    assert fit_accept_threshold([0, 1, 1], [0.0001, 0.0003, 0.2]) == 0.0003


def test_fit_tau_most_auto_meeting_target():
    y, p = [1, 0, 0, 0], [0.99, 0.7, 0.4, 0.01]
    assert fit_tau(y, p, 0.0, 0.01) == 0.71
    assert {r["tau"]: r["n_auto"] for r in tau_curve(y, p, [0.5, 0.71])} == {0.5: 4, 0.71: 2}
    assert fit_tau([0, 1], [0.99, 0.01], 0.0, 0.01) > 1.0
