from conftest import det

from jevgate.detectors.track import associate, conf_trend, count_flicker, majority_class, summarize_tracks, video_short


def test_count_flicker():
    assert count_flicker([2, 2, 2]) == 0.0
    assert count_flicker([1, 3, 1]) == 2.0
    assert count_flicker([5]) == 0.0


def test_video_short():
    assert video_short("2019-03-19_17-01-06to2019-03-19_17-01-19_1") == "0319-170106"


def test_associate_matches_by_iou_one_to_one():
    tracked = [[0, 0, 10, 10], [100, 100, 110, 110], [500, 500, 510, 510]]
    detected = [[101, 100, 111, 110], [0, 0, 10, 11]]
    assert associate(tracked, detected) == [1, 0, None]


def test_majority_vote_ties_go_to_conf_sum():
    assert majority_class(["crab"] * 9 + ["shrimp"] * 6, [0.5] * 15) == ("crab", 0.6)
    assert majority_class(["crab", "shrimp"], [0.3, 0.8])[0] == "shrimp"


def test_conf_trend():
    assert conf_trend([0.2, 0.2, 0.6, 0.6]) == "rising"
    assert conf_trend([0.5]) == "flat"


def test_summarize_tracks_evidence_and_scene():
    frames = [{"video": "v", "frame_idx": i} for i in range(1, 6)]
    dets = [det(f"a{i}", i, [10 * i, 0, 10 * i + 20, 20], "crab", 0.4 + 0.1 * i, "v#1") for i in range(1, 4)]
    dets += [det("b1", 2, [500, 500, 540, 520], "starfish", 0.9, "v#2"), det("u", 5, [0, 0, 5, 5], "fish", 0.1)]
    tracks = {t["track_uid"]: t for t in summarize_tracks(dets, frames, (960, 540))}
    a = tracks["v#1"]
    assert a["box_ids"] == ["a1", "a2", "a3"] and a["rep_box_id"] == "a3" and a["majority_class"] == "crab"
    ev = a["evidence"]
    assert ev["length"]["frames_detected"] == 3 and ev["length"]["starts_at_video_start"]
    assert not ev["length"]["ends_at_video_end"] and ev["detector_conf"]["trend"] == "rising"
    assert ev["scene"]["other_tracks_at_same_time"] == {"starfish": 1}
    assert set(tracks) == {"v#1", "v#2"}
