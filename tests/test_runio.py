import datetime as dt

from jevgate.runio import RunDir, next_run_dir


def test_run_dir_naming(tmp_path):
    day = dt.date(2026, 10, 13)
    first = next_run_dir(tmp_path, "smoke", day)
    assert first.name == "261013_smoke"
    first.mkdir()
    assert next_run_dir(tmp_path, "smoke", day).name == "261013_smoke_2"


def test_meta_merges_sections_and_system_rows(tmp_path):
    run = RunDir(tmp_path)
    run.update_meta({"timing_s": {"jev": 1.0}, "config": {"a": 1}})
    run.update_meta({"timing_s": {"vlm": 2.0}})
    assert run.meta()["timing_s"] == {"jev": 1.0, "vlm": 2.0} and run.config() == {"a": 1}
    run.write("judgments", [{"cand_id": "b1", "systems": {"vlm": {"pred": "crab"}, "jev": {"pred": "fish"}}}])
    assert run.system_rows()["jev"] == [{"cand_id": "b1", "pred": "fish"}]
