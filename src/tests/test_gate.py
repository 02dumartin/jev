from jevgate.judges.gate import auto_curve, combine, fit_detector_threshold, fit_gate_thresholds, fit_tau


def _jev(preds, conf, need):
    return [{"cand_id": str(i), "pred": p, "probs": {p: c}, "confidence": c, "need_vlm": n, "latency_ms": 10.0}
            for i, (p, c, n) in enumerate(zip(preds, conf, need, strict=True))]


def _vlm(preds):
    return [{"cand_id": str(i), "pred": p, "probs": {p: 0.9}, "confidence": 0.9, "latency_ms": 100.0}
            for i, p in enumerate(preds)]


TH = {"tau_need_vlm": 3.0, "tau_conf": 0.8}


def test_primary_gate_routes_on_need_vlm_only():
    out = combine("gate", _jev(["crab", "fish", "background"], [0.95, 0.99, 0.5], [1.2, 4.1, 2.0]),
                  _vlm(["crab", "small_fish", "shrimp"]), TH)
    assert [r["routed_to"] for r in out] == ["jev", "vlm", "jev"]
    assert [r["pred"] for r in out] == ["crab", "small_fish", "background"]
    assert out[1]["latency_ms"] == 110.0 and out[0]["latency_ms"] == 10.0


def test_conf_and_either_gates():
    jev = _jev(["crab", "fish"], [0.5, 0.95], [1.0, 4.0])
    vlm = _vlm(["shrimp", "small_fish"])
    assert [r["routed_to"] for r in combine("gate_conf", jev, vlm, TH)] == ["vlm", "jev"]
    assert [r["routed_to"] for r in combine("gate_either", jev, vlm, TH)] == ["vlm", "vlm"]


def test_fit_detector_threshold_maximises_7way_accuracy():
    y = ["background", "background", "crab", "crab"]
    assert fit_detector_threshold(y, ["crab"] * 4, [0.1, 0.2, 0.3, 0.4]) == 0.3
    # a wrong DINO class cannot be fixed by t: accuracy ties at 0.5, the lowest t wins
    assert fit_detector_threshold(["fish", "background"], ["crab", "crab"], [0.9, 0.1]) == 0.9


def test_auto_curve_directions():
    below = {r["tau"]: r for r in auto_curve([True, False], [1.0, 4.0], [2.0, 5.0], "below")}
    assert below[2.0]["n_auto"] == 1 and below[2.0]["auto_error"] == 0.0
    assert below[5.0]["n_auto"] == 2 and below[5.0]["auto_error"] == 0.5
    at_least = {r["tau"]: r for r in auto_curve([True, False], [0.9, 0.4], [0.5], "at_least")}
    assert at_least[0.5]["n_auto"] == 1


def test_fit_tau_most_auto_within_error_and_fallback():
    curve = auto_curve([True, True, False], [1.0, 2.0, 4.5], [1.5, 2.5, 5.0], "below")
    assert fit_tau(curve, 0.0, 1.0) == 2.5
    assert fit_tau(auto_curve([False], [1.0], [2.0], "below"), 0.0, 1.0) == 1.0


def test_fit_gate_thresholds(cfg):
    jev = _jev(["crab", "crab", "fish"], [0.99, 0.9, 0.6], [1.0, 1.5, 4.5])
    th = fit_gate_thresholds(["crab", "crab", "small_fish"], jev, cfg)
    assert 1.5 < th["tau_need_vlm"] <= 4.5 and 0.6 < th["tau_conf"] <= 0.9
