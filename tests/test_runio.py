import datetime as dt

from jevgate.runio import next_run_dir


def test_run_dir_naming(tmp_path):
    day = dt.date(2026, 10, 13)
    first = next_run_dir(tmp_path, "smoke", day)
    assert first.name == "261013_smoke"
    first.mkdir()
    assert next_run_dir(tmp_path, "smoke", day).name == "261013_smoke_2"
