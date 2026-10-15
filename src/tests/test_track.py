from jevgate.detectors.track import count_flicker, track_stats


def test_count_flicker():
    assert count_flicker([2, 2, 2]) == 0.0
    assert count_flicker([1, 3, 1]) == 2.0
    assert count_flicker([5]) == 0.0


def test_track_stats():
    rows = [{"track_id": 1}, {"track_id": 1}, {"track_id": 1}, {"track_id": 2}]
    assert track_stats(rows) == {"n_tracks": 2, "track_len_median": 2.0, "track_len_max": 3}
    assert track_stats([])["n_tracks"] == 0
