from conftest import det

from jevgate.judges.gate import (
    box_predictions,
    fit_accept_threshold,
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
    assert box_predictions(dets, maj, None, conf_t=0.5) == {"1": "shrimp", "2": "background", "3": "fish"}
    assert box_predictions(dets, maj, None) == {"1": "crab", "2": "crab", "3": "background"}
    assert box_predictions(dets, maj, {"t": False}) == {"1": "background", "2": "background", "3": "background"}


def test_fit_thresholds():
    assert fit_conf_threshold(["crab", "background"], ["crab", "crab"], [0.6, 0.2]) == 0.6
    assert fit_accept_threshold([0, 1, 1], [0.0001, 0.0003, 0.2]) == 0.0003


def test_fit_tau_most_auto_meeting_target():
    y, p = [1, 0, 0, 0], [0.99, 0.7, 0.4, 0.01]
    assert fit_tau(y, p, 0.0, 0.01) == 0.71
    assert {r["tau"]: r["n_auto"] for r in tau_curve(y, p, [0.5, 0.71])} == {0.5: 4, 0.71: 2}
    assert fit_tau([0, 1], [0.99, 0.01], 0.0, 0.01) > 1.0
