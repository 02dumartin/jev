import numpy as np
import pytest

from jevgate.evaluation import metrics as M

C = ["fish", "crab", "background"]


def test_multiclass_metrics_and_confusion():
    m = M.multiclass_metrics(["fish", "crab", "background", "crab"], ["fish", "fish", "background", "crab"], C)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["crab"]["recall"] == 0.5 and m["per_class"]["fish"]["precision"] == 0.5
    assert m["confusion"]["matrix"][1] == [1, 1, 0]  # row = y crab


def test_ovr_counts():
    o = M.ovr_counts(["fish", "crab", "fish"], ["fish", "fish", "crab"], "fish")
    assert (o["tp"], o["fn"], o["fp"], o["tn"]) == (1, 1, 1, 0) and o["sens"] == 0.5 and o["spec"] == 0.0


def test_auroc_and_roc():
    assert M.auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert M.auroc([1, 0], [0.5, 0.5]) == 0.5
    assert M.auroc([1, 1], [0.1, 0.2]) is None
    r = M.roc_points([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert r["fpr"][0] == 0 and r["tpr"][-1] == 1 and r["fpr"][-1] == 1


def test_ece_top_and_brier():
    assert M.ece_top([1, 0], [0.5, 0.5], 10) == pytest.approx(0.0)
    assert M.ece_top([1, 0], [1.0, 1.0], 10) == pytest.approx(0.5)
    assert M.reliability_bins([1], [1.0], 15)[-1]["count"] == 1
    assert M.brier_multi(["fish"], [{"fish": 1.0, "crab": 0.0, "background": 0.0}], C) == 0.0
    assert M.brier_multi(["fish"], [{"fish": 0.0, "crab": 1.0, "background": 0.0}], C) == 2.0


def test_cluster_bootstrap_resamples_whole_clusters():
    boot = M.ClusterBootstrap(["a", "a", "b", "c"], 200, 0)
    assert len(boot.members) == 3
    lo, hi = boot.ci(lambda idx: 0.0)
    assert lo == hi == 0.0
    # a statistic that is constant within clusters never sees a split cluster
    vals = np.asarray([1, 1, 0, 0])
    lo, hi = boot.ci(lambda idx: float(vals[idx].mean()))
    assert 0.0 <= lo <= hi <= 1.0


def test_brier_binary():
    assert M.brier_binary([1, 0], [1.0, 0.0]) == 0.0
    assert M.brier_binary([1, 0], [0.0, 1.0]) == 1.0


def test_endpoints_stop_at_first_failure():
    steps = [{"min_delta_acc_ci_low": -0.02, "max_vlm_call_ratio": 0.5},
             {"min_delta_acc_ci_low": -0.01, "max_vlm_call_ratio": 0.5},
             {"min_delta_acc_ci_low": -0.01, "max_vlm_call_ratio": 0.3}]
    assert [r["result"] for r in M.evaluate_endpoints(-0.015, 0.4, steps)] == ["pass", "fail", "not_tested"]
    assert [r["result"] for r in M.evaluate_endpoints(None, 0.4, steps)] == ["not_tested"] * 3
