import numpy as np
import pytest

from jevgate.boxes import clip_box, iou_matrix, nms


def test_iou_identical_disjoint_and_half():
    m = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == 0.0
    assert m[0, 2] == pytest.approx(50 / 150)


def test_iou_empty_inputs_have_right_shape():
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def test_degenerate_box_gives_zero_not_nan():
    m = iou_matrix([[5, 5, 5, 5]], [[5, 5, 5, 5]])
    assert np.isfinite(m).all() and m[0, 0] == 0.0


def test_clip_box_to_frame():
    assert clip_box([-5, 10, 970, 600], (960, 540)) == [0.0, 10, 960, 540]


def test_nms_keeps_highest_and_drops_overlaps():
    boxes = [[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]]
    assert nms(boxes, [0.6, 0.9, 0.5], 0.5) == [1, 2]
