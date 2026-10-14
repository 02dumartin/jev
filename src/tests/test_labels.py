from conftest import cand

from jevgate.data.brackish import BrackishGT, GTBox
from jevgate.data.labels import AMBIGUOUS, detector_coverage, iou_to_label, label_candidates


def test_iou_bands():
    assert iou_to_label(0.5, "crab", 0.5, 0.3) == "crab"
    assert iou_to_label(0.49, "crab", 0.5, 0.3) == AMBIGUOUS
    assert iou_to_label(0.3, "crab", 0.5, 0.3) == AMBIGUOUS
    assert iou_to_label(0.29, "crab", 0.5, 0.3) == "background"
    assert iou_to_label(0.0, None, 0.5, 0.3) == "background"


def _gt():
    gt = BrackishGT(boxes={"c-0001.png": [GTBox(7, "crab", [0, 0, 10, 10])]})
    frames = [{"clip": "c", "frame_idx": i, "file": f"c-000{i}.png", "gt_status": gt.status(f"c-000{i}.png")}
              for i in (1, 2)]
    return gt, frames


def test_labels_use_gt_class_and_background(cfg):
    gt, frames = _gt()
    cands = [cand("a", 1, [0, 0, 10, 10]), cand("b", 1, [40, 40, 50, 50]), cand("c", 2, [0, 0, 10, 10]),
             cand("d", 1, [0, 0, 10, 16])]
    rows = {r["cand_id"]: r for r in label_candidates(cands, frames, gt, cfg)}
    assert rows["a"]["y"] == "crab" and rows["a"]["matched_object_id"] == 7  # DINO said fish; y follows GT
    assert rows["b"]["y"] == "background" and rows["b"]["matched_object_id"] is None
    assert rows["c"]["y"] == "background" and rows["c"]["scored"]   # empty frame: every box is background
    assert rows["d"]["y"] == "crab"                                  # IoU 0.625
    assert [r["cand_id"] for r in label_candidates(cands, frames, gt, cfg)] == ["a", "b", "c", "d"]


def test_coverage_per_class(cfg):
    gt, frames = _gt()
    cov = detector_coverage([cand("a", 1, [0, 0, 10, 9])], frames, gt, 0.5, list(cfg["classes"]))
    assert cov["recall_upper_bound"] == 1.0 and cov["per_class"]["crab"]["covered"] == 1
    assert detector_coverage([], frames, gt, 0.5, list(cfg["classes"]))["per_class"]["crab"]["missed"] == 1
