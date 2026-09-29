"""Unit tests for the demo-video step timeline.

CI-safe: a fake clock stands in for ``time.time()``; no browser, no ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docs.screenshots import video_timeline
from docs.screenshots.video_timeline import Step, Timeline, step


class FakeClock:
    """Returns the queued times in order; each call advances by one tick."""

    def __init__(self, start: float = 1000.0, tick: float = 1.0) -> None:
        self.now = start
        self.tick = tick

    def __call__(self) -> float:
        value = self.now
        self.now += self.tick
        return value


def _timeline(clock: FakeClock | None = None, theme: str = "dark") -> Timeline:
    return Timeline(theme=theme, clock=clock or FakeClock(), version="1.2.3")


def test_step_stamps_entry_and_exit_with_clock() -> None:
    clock = FakeClock(start=100.0, tick=0.0)
    tl = _timeline(clock)
    with step(tl, "focus", caption="Click the terminal"):
        clock.now = 104.5
    assert tl.steps == [
        Step(name="focus", caption="Click the terminal", fast_forward=False, start=100.0, end=104.5)
    ]


def test_step_default_clock_is_epoch_time(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([50.0, 51.0, 52.0, 53.0])
    monkeypatch.setattr(video_timeline.time, "time", lambda: next(times))
    tl = Timeline(theme="light", version="9")
    with step(tl, "a"):
        pass
    assert (tl.steps[0].start, tl.steps[0].end) == (51.0, 52.0)


def test_step_records_fast_forward_flag_and_totals() -> None:
    clock = FakeClock(start=0.0, tick=0.0)
    tl = _timeline(clock)
    with step(tl, "wait", fast_forward=True):
        clock.now = 30.0
    with step(tl, "hold"):
        clock.now = 32.5
    assert tl.steps[0].fast_forward is True
    assert tl.steps[1].fast_forward is False
    assert tl.ff_total == pytest.approx(30.0)
    assert tl.realtime_total == pytest.approx(2.5)


def test_step_calls_callbacks_around_body_in_order() -> None:
    events: list[str] = []
    tl = _timeline()

    def on_enter(s: Step) -> None:
        events.append(f"enter:{s.name}:{s.caption}:{s.fast_forward}")

    def on_exit(s: Step) -> None:
        events.append(f"exit:{s.name}:{s.end is not None}")

    with step(tl, "plot", "Waiting for the plot", True, on_enter=on_enter, on_exit=on_exit):
        events.append("body")
    assert events == ["enter:plot:Waiting for the plot:True", "body", "exit:plot:True"]


def test_step_start_is_stamped_after_on_enter_and_end_before_on_exit() -> None:
    clock = FakeClock(start=0.0, tick=0.0)
    tl = _timeline(clock)

    def on_enter(_s: Step) -> None:
        clock.now = 10.0  # caption toggle takes time; the span starts after it

    def on_exit(_s: Step) -> None:
        clock.now = 99.0  # frame flush after the span ends

    with step(tl, "a", on_enter=on_enter, on_exit=on_exit):
        clock.now = 12.0
    assert (tl.steps[0].start, tl.steps[0].end) == (10.0, 12.0)


def test_step_records_even_when_body_raises_and_reraises() -> None:
    clock = FakeClock(start=5.0, tick=0.0)
    exited: list[str] = []
    tl = _timeline(clock)
    with pytest.raises(RuntimeError, match="boom"):
        with step(tl, "post", on_exit=lambda s: exited.append(s.name)):
            clock.now = 7.0
            raise RuntimeError("boom")
    assert exited == ["post"]
    assert len(tl.steps) == 1
    recorded = tl.steps[0]
    assert (recorded.name, recorded.start, recorded.end) == ("post", 5.0, 7.0)
    assert recorded.error == "RuntimeError: boom"


def test_step_success_has_no_error() -> None:
    tl = _timeline()
    with step(tl, "ok"):
        pass
    assert tl.steps[0].error is None


def test_step_yields_the_step() -> None:
    tl = _timeline()
    with step(tl, "rotate", caption="Rotate") as s:
        assert s.name == "rotate"
        assert s.start is not None and s.end is None
    assert s is tl.steps[0]


def test_timeline_records_version_and_recording_time() -> None:
    tl = Timeline(theme="dark", clock=FakeClock(start=1_700_000_000.0), version="2.0")
    assert tl.osprey_version == "2.0"
    assert tl.recorded_at == 1_700_000_000.0


def test_timeline_default_version_comes_from_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    from docs.screenshots import capture

    monkeypatch.setattr(capture, "osprey_version", lambda: "7.7.7")
    tl = Timeline(theme="dark", clock=FakeClock())
    assert tl.osprey_version == "7.7.7"


def test_timeline_save_and_load_round_trip(tmp_path: Path) -> None:
    clock = FakeClock(start=10.0, tick=0.5)
    tl = _timeline(clock, theme="light")
    with step(tl, "type", caption="Prompt 1"):
        pass
    with pytest.raises(ValueError):
        with step(tl, "wait", fast_forward=True):
            raise ValueError("late")
    path = tl.save(tmp_path)
    assert path == tmp_path / "timeline-light.json"
    loaded = Timeline.load(path)
    assert loaded.theme == "light"
    assert loaded.osprey_version == tl.osprey_version
    assert loaded.recorded_at == tl.recorded_at
    assert loaded.steps == tl.steps


def test_timeline_json_shape(tmp_path: Path) -> None:
    tl = _timeline(FakeClock(start=1.0, tick=1.0))
    with step(tl, "focus"):
        pass
    data = json.loads(tl.save(tmp_path).read_text())
    assert data["theme"] == "dark"
    assert data["osprey_version"] == "1.2.3"
    assert data["recorded_at"] == 1.0
    assert "T" in data["recorded_at_iso"]
    assert data["steps"] == [
        {
            "name": "focus",
            "caption": None,
            "fast_forward": False,
            "start": 2.0,
            "end": 3.0,
            "error": None,
        }
    ]


def test_timeline_save_creates_directory(tmp_path: Path) -> None:
    out = tmp_path / "docs" / "demo-video"
    path = _timeline().save(out)
    assert path.exists()


def test_timeline_path_for_theme(tmp_path: Path) -> None:
    assert Timeline.path_for(tmp_path, "dark") == tmp_path / "timeline-dark.json"


def test_loaded_timeline_uses_epoch_clock_for_new_steps(tmp_path: Path) -> None:
    path = _timeline().save(tmp_path)
    loaded = Timeline.load(path)
    import time

    before = time.time()
    assert before <= loaded.clock() <= time.time()


def test_module_does_not_call_time_sleep() -> None:
    source = Path(video_timeline.__file__).read_text()
    assert "time.sleep" not in source
