from conftest import cand

from jevgate.data import BrackishGT, GTBox
from jevgate.label import detector_recall_upper_bound, iou_to_label, label_candidates


def test_iou_bands():
    assert iou_to_label(0.5, 0.5, 0.3) == 1
    assert iou_to_label(0.49, 0.5, 0.3) is None
    assert iou_to_label(0.3, 0.5, 0.3) is None
    assert iou_to_label(0.29, 0.5, 0.3) == 0


def _gt():
    gt = BrackishGT(boxes={"c-0001.png": [GTBox(7, "fish", [0, 0, 10, 10])]},
                    in_split={"c-0001.png", "c-0002.png"})
    frames = [{"clip": "c", "frame_idx": 1, "file": "c-0001.png", "gt_status": gt.status("c-0001.png")},
              {"clip": "c", "frame_idx": 2, "file": "c-0002.png", "gt_status": gt.status("c-0002.png")},
              {"clip": "c", "frame_idx": 3, "file": "c-0003.png", "gt_status": gt.status("c-0003.png")}]
    return gt, frames


def test_labels_real_fake_empty_and_unknown(cfg):
    gt, frames = _gt()
    cands = [cand("a", 1, [0, 0, 10, 10]), cand("b", 1, [40, 40, 50, 50]),
             cand("c", 2, [0, 0, 10, 10]), cand("d", 3, [0, 0, 10, 10])]
    rows = {r["cand_id"]: r for r in label_candidates(cands, frames, gt, cfg)}
    assert rows["a"]["y"] == 1 and rows["a"]["matched_object_id"] == 7
    assert rows["b"]["y"] == 0 and rows["b"]["matched_object_id"] is None
    assert rows["c"]["y"] == 0 and rows["c"]["scored"]          # empty_by_absence frame: every box is fake
    assert rows["d"]["y"] is None and not rows["d"]["scored"]   # unknown GT: excluded
    assert [r["cand_id"] for r in label_candidates(cands, frames, gt, cfg)] == ["a", "b", "c", "d"]


def test_recall_upper_bound():
    gt, frames = _gt()
    ub = detector_recall_upper_bound([cand("a", 1, [0, 0, 10, 9])], frames, gt, 0.5)
    assert ub == {"n_gt_objects": 1, "n_covered": 1, "recall_upper_bound": 1.0}
    assert detector_recall_upper_bound([], frames, gt, 0.5)["recall_upper_bound"] == 0.0
