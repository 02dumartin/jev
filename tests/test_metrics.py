import numpy as np
import pytest

from jev.evaluation.metrics import binary_metrics, took_product


def test_known_calibration_values():
    m = binary_metrics([0, 1], [0.2, 0.8], bins=2)
    assert m["brier"] == pytest.approx(0.04)
    assert m["nll"] == pytest.approx(-np.log(0.8))
    assert m["ece"] == pytest.approx(0.2)
    assert m["confidence_ece"] == pytest.approx(0.2)
    assert (m["tn"], m["tp"], m["fp"], m["fn"]) == (1, 1, 0, 0)


def test_endpoints_empty_bins_and_zero_coverage():
    m = binary_metrics([0, 1], [0, 1])
    assert m["ece"] == 0
    assert m["reliability"][0]["n"] == 1
    assert m["reliability"][-1]["n"] == 1
    assert m["reliability"][1]["observed_rate"] is None
    m = binary_metrics([0, 1], [0.5, 0.5])
    assert m["selective"][-1]["risk"] is None
    assert m["selective"][0]["coverage"] == 1
    assert m["accuracy"] == 0.5


@pytest.mark.parametrize(
    "y,p", [([], []), ([0], [0, 1]), ([2], [0.5]), ([1], [float("nan")]), ([0], [-0.1]), ([1], [1.1])]
)
def test_invalid_inputs(y, p):
    with pytest.raises(ValueError):
        binary_metrics(y, p)


def test_proxy_is_net_count_not_any_take():
    assert took_product([{"label": "take"}])
    assert not took_product([{"label": "take"}, {"label": "put"}])
    assert not took_product([])
