from conftest import cand

from jevgate.contracts import check_evidence
from jevgate.data import context_frames
from jevgate.evidence import build_evidence


def _ev(targets, context, cfg):
    keys = sorted({(c["clip"], c["frame_idx"]) for c in context + targets})
    return {e["cand_id"]: e for e in build_evidence(targets, context, keys, cfg)}


def test_track_uses_only_previous_frames(cfg):
    cands = [cand("f1", 1, [0, 0, 10, 10]), cand("f2", 2, [1, 0, 11, 10]), cand("f3", 3, [100, 100, 110, 110])]
    ev = _ev(cands, cands, cfg)
    assert ev["f1"]["history_frame_idxs"] == [] and ev["f1"]["track"]["seen_in_prev_frames"] == 0
    assert ev["f2"]["history_frame_idxs"] == [1] and ev["f2"]["track"]["seen_in_prev_frames"] == 1
    assert ev["f3"]["track"]["seen_in_prev_frames"] == 0


def test_future_frames_do_not_change_evidence(cfg):
    base = [cand("f1", 1, [0, 0, 10, 10]), cand("f2", 2, [0, 0, 10, 10])]
    future = base + [cand("f3", 3, [0, 0, 10, 10])]
    assert _ev(base[1:], base, cfg)["f2"] == _ev(base[1:], future, cfg)["f2"]


def test_history_window_is_k_and_counts_empty_frames(cfg):
    k = cfg["evidence"]["history_frames"]
    cands = [cand(f"f{i}", i, [0, 0, 10, 10]) for i in range(1, k + 4)]
    keys = [("c", i) for i in range(1, k + 4)]
    last = build_evidence(cands[-1:], cands[:-2], keys, cfg)[0]  # frame k+2 processed but empty
    assert len(last["history_frame_idxs"]) == k and last["track"]["seen_in_prev_frames"] == k - 1


def test_context_frames_include_previous_k_of_any_split():
    frames = [{"clip": "c", "frame_idx": 10}]
    assert context_frames(frames, 3, {"c": {7, 8, 9, 10, 11}}) == [("c", 7), ("c", 8), ("c", 9), ("c", 10)]
    assert context_frames(frames, 3, {"c": {9, 10}}) == [("c", 9), ("c", 10)]


def test_contract_flags_leak(cfg):
    cands = [cand("f2", 2, [0, 0, 10, 10])]
    leaked = [{"cand_id": "f2", "history_frame_idxs": [3], "detector": {}, "geometry": {}, "track": {}}]
    assert any("leak" in e for e in check_evidence(leaked, cands, cfg))
