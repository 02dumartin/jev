from conftest import cand

from jevgate.contracts import check_evidence
from jevgate.evidence import build_evidence


def test_track_uses_only_previous_frames(cfg):
    cands = [cand("f1", 1, [0, 0, 10, 10]), cand("f2", 2, [1, 0, 11, 10]), cand("f3", 3, [100, 100, 110, 110])]
    ev = {e["cand_id"]: e for e in build_evidence(cands, cfg)}
    assert ev["f1"]["history_frame_idxs"] == [] and ev["f1"]["track"]["seen_in_prev_frames"] == 0
    assert ev["f2"]["history_frame_idxs"] == [1] and ev["f2"]["track"]["seen_in_prev_frames"] == 1
    assert ev["f3"]["track"]["seen_in_prev_frames"] == 0


def test_future_frames_do_not_change_evidence(cfg):
    base = [cand("f1", 1, [0, 0, 10, 10]), cand("f2", 2, [0, 0, 10, 10])]
    with_future = base + [cand("f3", 3, [0, 0, 10, 10])]
    a = {e["cand_id"]: e for e in build_evidence(base, cfg)}
    b = {e["cand_id"]: e for e in build_evidence(with_future, cfg)}
    assert a["f2"] == b["f2"]


def test_history_window_is_k(cfg):
    k = cfg["evidence"]["history_frames"]
    cands = [cand(f"f{i}", i, [0, 0, 10, 10]) for i in range(k + 3)]
    last = build_evidence(cands, cfg)[-1]
    assert len(last["history_frame_idxs"]) == k and last["track"]["seen_in_prev_frames"] == k


def test_contract_flags_leak(cfg):
    cands = [cand("f2", 2, [0, 0, 10, 10])]
    leaked = [{"cand_id": "f2", "history_frame_idxs": [3], "detector": {}, "geometry": {}, "track": {}}]
    assert any("leak" in e for e in check_evidence(leaked, cands, cfg))
