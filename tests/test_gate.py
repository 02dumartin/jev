from jevgate.gate import combine, fit_detector_threshold, fit_tau, tau_curve


def _rows(ps):
    return [{"cand_id": str(i), "keep": p >= 0.5, "p_real": p, "latency_ms": 10.0} for i, p in enumerate(ps)]


def test_combine_routes_uncertain_boxes_to_vlm():
    jev = _rows([0.95, 0.6, 0.02])
    vlm = [{"cand_id": str(i), "keep": k, "latency_ms": 100.0} for i, k in enumerate([False, False, True])]
    out = combine(jev, vlm, tau=0.9)
    assert [r["routed_to"] for r in out] == ["jev", "vlm", "jev"]
    assert [r["keep"] for r in out] == [True, False, False]
    assert out[1]["latency_ms"] == 110.0 and out[0]["latency_ms"] == 10.0


def test_tau_one_point_something_routes_everything():
    out = combine(_rows([1.0, 0.0]), [{"cand_id": "0", "keep": False, "latency_ms": 1},
                                      {"cand_id": "1", "keep": True, "latency_ms": 1}], tau=1.01)
    assert all(r["routed_to"] == "vlm" for r in out)


def test_fit_detector_threshold_maximises_accuracy():
    assert fit_detector_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 0.3


def test_fit_tau_smallest_meeting_error_target():
    y = [1, 1, 0, 0]
    p = [0.99, 0.7, 0.4, 0.01]  # the 0.4 box is right (reject); make 0.7 wrong by flipping its label
    y = [1, 0, 0, 0]
    assert fit_tau(y, p, max_auto_error=0.0, step=0.01) == 0.71
    curve = {r["tau"]: r for r in tau_curve(y, p, [0.5, 0.71])}
    assert curve[0.5]["auto_error"] == 0.25 and curve[0.71]["n_auto"] == 2


def test_fit_tau_none_qualifies_routes_all():
    assert fit_tau([0, 1], [0.99, 0.01], 0.0, 0.01) > 1.0
