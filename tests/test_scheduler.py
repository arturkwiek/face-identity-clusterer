from datetime import datetime, time as dtime

import pytest

from face_clusterer.config import AppConfig, ScheduleConfig
from face_clusterer.scheduler import (
    DayNightScheduler, analysis_due, in_capture_window, last_window_end,
    next_transition, parse_hhmm,
)

DAY = dtime(7, 0)          # capture starts
NIGHT = dtime(15, 0)       # capture ends, analysis begins


def at(hour, minute=0, day=15):
    return datetime(2026, 7, day, hour, minute)


def test_parse_hhmm():
    assert parse_hhmm("07:00") == dtime(7, 0)
    assert parse_hhmm("15:30") == dtime(15, 30)
    for bad in ("7", "25:00", "abc", "", None):
        with pytest.raises(ValueError):
            parse_hhmm(bad)


@pytest.mark.parametrize(
    "moment, capturing",
    [
        (at(6, 59), False),        # just before the window opens
        (at(7, 0), True),          # boundary: start is inclusive
        (at(11, 30), True),        # midday
        (at(14, 59), True),
        (at(15, 0), False),        # boundary: end is exclusive
        (at(23, 30), False),       # night
        (at(3, 0), False),         # small hours
    ],
)
def test_capture_window(moment, capturing):
    assert in_capture_window(moment, DAY, NIGHT) is capturing


def test_window_crossing_midnight():
    """A night-shift window (22:00-06:00) must also work."""
    start, end = dtime(22, 0), dtime(6, 0)
    assert in_capture_window(at(23, 0), start, end)
    assert in_capture_window(at(2, 0), start, end)
    assert not in_capture_window(at(12, 0), start, end)


def test_zero_length_window_never_captures():
    assert not in_capture_window(at(7, 0), dtime(7, 0), dtime(7, 0))


def test_next_transition():
    # during capture -> the window's end, same day
    assert next_transition(at(9, 0), DAY, NIGHT) == at(15, 0)
    # during analysis, before midnight -> next morning
    assert next_transition(at(20, 0), DAY, NIGHT) == at(7, 0, day=16)
    # during analysis, after midnight -> this morning
    assert next_transition(at(3, 0), DAY, NIGHT) == at(7, 0)


def test_last_window_end():
    assert last_window_end(at(20, 0), NIGHT) == at(15, 0)
    assert last_window_end(at(3, 0), NIGHT) == at(15, 0, day=14)
    assert last_window_end(at(14, 0), NIGHT) == at(15, 0, day=14)


def test_analysis_due_when_never_run():
    assert analysis_due(at(16, 0), NIGHT, None)


def test_analysis_due_once_per_night():
    evening = at(16, 0)
    # nothing ran since the window closed at 15:00 -> due
    assert analysis_due(evening, NIGHT, at(15, 0, day=14).timestamp())
    # it ran at 15:30 tonight -> not due again
    assert not analysis_due(evening, NIGHT, at(15, 30).timestamp())
    # still not due at 04:00, the same night
    assert not analysis_due(at(4, 0, day=16), NIGHT, at(15, 30).timestamp())
    # due again the following night
    assert analysis_due(at(16, 0, day=16), NIGHT, at(15, 30).timestamp())


def test_interrupted_night_is_retried():
    """A run that never completed leaves an old timestamp -> retried."""
    assert analysis_due(at(2, 0, day=16), NIGHT, at(15, 30, day=14).timestamp())


def _scheduler(tmp_path, **schedule):
    cfg = AppConfig(schedule=ScheduleConfig(**schedule))
    cfg.acquisition.dir = str(tmp_path / "dataset")
    return DayNightScheduler(cfg)


def test_state_roundtrip(tmp_path):
    sched = _scheduler(tmp_path)
    assert sched._load_state() == {}
    sched._save_state({"last_analysis": 123.0})
    assert sched._load_state()["last_analysis"] == 123.0


def test_unreadable_state_does_not_crash(tmp_path):
    sched = _scheduler(tmp_path)
    sched.state_path.parent.mkdir(parents=True, exist_ok=True)
    sched.state_path.write_text("{broken", encoding="utf-8")
    assert sched._load_state() == {}


def test_camera_source_parsing(tmp_path):
    assert _scheduler(tmp_path, camera="0")._camera_source() == 0
    url = "rtsp://cam.local/stream"
    assert _scheduler(tmp_path, camera=url)._camera_source() == url


def test_invalid_window_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError):
        _scheduler(tmp_path, capture_start="7am")


def test_capture_failure_does_not_stop_the_schedule(tmp_path, monkeypatch):
    """An unplugged camera must be retried, not fatal."""
    sched = _scheduler(tmp_path, retry_seconds=0.0)
    monkeypatch.setattr(
        sched, "capture_slice",
        lambda seconds: (_ for _ in ()).throw(IOError("camera busy")),
    )
    monkeypatch.setattr(
        "face_clusterer.scheduler.in_capture_window", lambda *a: True
    )
    monkeypatch.setattr(DayNightScheduler, "_sleep", staticmethod(lambda s: None))

    results = sched.run(max_cycles=3)
    assert results["errors"] == 3 and results["captures"] == 0


def test_analysis_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, retry_seconds=0.0)
    monkeypatch.setattr(
        "face_clusterer.scheduler.in_capture_window", lambda *a: False
    )
    monkeypatch.setattr(
        sched, "analyze",
        lambda: (_ for _ in ()).throw(RuntimeError("no images")),
    )
    monkeypatch.setattr(DayNightScheduler, "_sleep", staticmethod(lambda s: None))

    results = sched.run(max_cycles=2)
    assert results["errors"] == 2 and results["analyses"] == 0
    assert "no images" in sched._load_state()["last_analysis_error"]


def test_successful_analysis_runs_once_then_idles(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path)
    monkeypatch.setattr(
        "face_clusterer.scheduler.in_capture_window", lambda *a: False
    )
    monkeypatch.setattr(sched, "analyze", lambda: {"identities": 3})
    monkeypatch.setattr(DayNightScheduler, "_sleep", staticmethod(lambda s: None))

    results = sched.run(max_cycles=3)
    # first cycle analyzes; the rest find it already done and idle
    assert results["analyses"] == 1
    assert sched._load_state()["last_analysis_summary"] == {"identities": 3}
