from conftest import det

from jevgate.data.brackish import BrackishGT, GTBox
from jevgate.data.labels import iou_to_y, label_boxes, label_tracks, missed_objects


def test_iou_bands():
    assert iou_to_y(0.5, "crab", 0.5, 0.3) == "crab"
    assert iou_to_y(0.4, "crab", 0.5, 0.3) is None
    assert iou_to_y(0.29, "crab", 0.5, 0.3) == "background"


def _gt():
    return BrackishGT(boxes={"v-0001.png": [GTBox(7, "crab", [0, 0, 10, 10])]},
                      listed={"v-0001.png", "v-0002.png"})


def test_label_boxes_statuses(cfg):
    dets = [det("a", 1, [0, 0, 10, 10]), det("b", 1, [50, 50, 60, 60]), det("c", 2, [0, 0, 10, 10]),
            det("d", 3, [0, 0, 10, 10])]
    rows = {r["box_id"]: r for r in label_boxes(dets, _gt(), cfg)}
    assert rows["a"]["y"] == "crab" and rows["a"]["matched_object_id"] == 7
    assert rows["b"]["y"] == "background" and rows["c"]["y"] == "background"   # empty_by_absence frame
    assert rows["d"]["gt_status"] == "unknown" and not rows["d"]["scored"]


def test_label_tracks_types():
    tracks = [{"track_uid": "ok", "majority_class": "crab", "box_ids": ["1", "2"]},
              {"track_uid": "mis", "majority_class": "crab", "box_ids": ["3", "4"]},
              {"track_uid": "fp", "majority_class": "crab", "box_ids": ["5"]},
              {"track_uid": "none", "majority_class": "crab", "box_ids": ["6"]}]
    labels = {"1": {"y": "crab", "scored": True}, "2": {"y": "background", "scored": True},
              "3": {"y": "shrimp", "scored": True}, "4": {"y": "shrimp", "scored": True},
              "5": {"y": "background", "scored": True}, "6": {"y": None, "scored": False}}
    rows = {r["track_uid"]: r for r in label_tracks(tracks, labels, 0.5)}
    assert (rows["ok"]["y_track"], rows["ok"]["type"]) == (1, "correct")
    assert (rows["mis"]["y_track"], rows["mis"]["type"]) == (0, "misclassified")
    assert (rows["fp"]["y_track"], rows["fp"]["type"]) == (0, "false")
    assert rows["none"]["y_track"] is None


def test_missed_objects():
    frames = [{"file": "v-0001.png"}, {"file": "v-0002.png"}]
    assert missed_objects([det("a", 1, [0, 0, 10, 9])], frames, _gt(), 0.5)["recall_upper_bound"] == 1.0
    assert missed_objects([], frames, _gt(), 0.5)["n_covered"] == 0
