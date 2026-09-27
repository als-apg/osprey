"""Unit tests for the cron trigger source."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from osprey.dispatch.clock_schedule import parse_clock_schedule
from osprey.dispatch.pool import QueueFullError
from osprey.dispatch.sources.cron import CronSource
from osprey.dispatch.trigger_config import TriggerConfig


def _make_trigger(name: str, interval_sec=None) -> TriggerConfig:
    source_config: dict = {}
    if interval_sec is not None:
        source_config["interval_sec"] = interval_sec
    return TriggerConfig(
        name=name,
        source="cron",
        action={"prompt": "tick", "allowed_tools": []},
        source_config=source_config,
    )


class _RecordingCallback:
    def __init__(self) -> None:
        self.calls: list[tuple[TriggerConfig, dict]] = []

    async def __call__(self, trigger: TriggerConfig, payload: dict) -> str | None:
        self.calls.append((trigger, payload))
        return "d-1"


@pytest.mark.asyncio
async def test_loop_fires_at_interval_then_stops(monkeypatch):
    """The loop fires the trigger each interval and stops cleanly when cancelled.

    Deterministic and pollution-proof: the interval wait is replaced with an
    immediate yield (loop body runs without real time), the test waits on an
    ``Event`` for the first fire — never racing the spawned task — then
    ``stop()`` cancels the loop. Earlier this test stopped the loop by counting
    ``asyncio.sleep`` calls and raising ``CancelledError`` on the second; because
    the patch lands on the *global* ``asyncio.sleep`` and the counter is shared,
    any other coroutine's ``sleep`` in the same loop could push the count so the
    loop's *first* sleep raised before the callback ever fired — an intermittent
    ``0 == 1`` under full-suite load. Termination now keys off the callback, not
    the sleep count, so a stray ``sleep`` can no longer skew it.
    """
    source = CronSource()
    trigger = _make_trigger("nightly", interval_sec=300)

    calls: list[tuple[TriggerConfig, dict]] = []
    fired = asyncio.Event()

    async def callback(trig: TriggerConfig, payload: dict) -> str | None:
        calls.append((trig, payload))
        fired.set()
        return "d-1"

    # Capture the genuine sleep before patching so the no-op interval still
    # yields control to the event loop (without any real delay).
    real_sleep = asyncio.sleep

    async def instant_interval(_seconds):
        await real_sleep(0)

    monkeypatch.setattr("osprey.dispatch.sources.cron.asyncio.sleep", instant_interval)

    await source.start([trigger], callback)
    await asyncio.wait_for(fired.wait(), timeout=5)
    await source.stop()

    assert source._tasks == []  # stop() cancelled the loop
    assert len(calls) >= 1  # fired at least once at the interval
    fired_trigger, payload = calls[0]
    assert fired_trigger is trigger
    assert payload["source"] == "cron"
    assert payload["trigger"] == "nightly"
    assert "timestamp" in payload


@pytest.mark.asyncio
async def test_the_first_fire_waits_one_full_interval(monkeypatch):
    """An interval trigger waits one whole interval before its first fire.

    This is the behaviour the event-dispatch how-to documents: nothing fires at
    start, so ``interval_sec: 86400`` means once a day counted from start. The
    wait is recorded, not slept, and only the order of the first two events is
    asserted, so another coroutine's ``sleep`` in the same loop cannot skew it.
    """
    source = CronSource()
    order: list[tuple[str, object]] = []
    fired = asyncio.Event()

    async def callback(trig: TriggerConfig, payload: dict) -> str | None:  # noqa: ARG001 - fire-callback signature; the order is what is asserted
        order.append(("fire", trig.name))
        fired.set()
        return "d-1"

    real_sleep = asyncio.sleep

    async def recorded_wait(seconds):
        order.append(("wait", seconds))
        await real_sleep(0)

    monkeypatch.setattr("osprey.dispatch.sources.cron.asyncio.sleep", recorded_wait)

    await source.start([_make_trigger("daily", interval_sec=86400)], callback)
    await asyncio.wait_for(fired.wait(), timeout=5)
    await source.stop()

    assert order[:2] == [("wait", 86400.0), ("fire", "daily")]


@pytest.mark.asyncio
async def test_invalid_interval_spawns_no_task():
    callback = _RecordingCallback()
    source = CronSource()
    triggers = [
        _make_trigger("missing"),  # no interval_sec
        _make_trigger("zero", interval_sec=0),
        _make_trigger("negative", interval_sec=-5),
        _make_trigger("not_a_number", interval_sec="soon"),
        _make_trigger("boolean", interval_sec=True),
    ]
    await source.start(triggers, callback)
    assert source._tasks == []


@pytest.mark.asyncio
async def test_valid_interval_spawns_task():
    callback = _RecordingCallback()
    source = CronSource()
    trigger = _make_trigger("hourly", interval_sec=3600)
    await source.start([trigger], callback)
    try:
        assert len(source._tasks) == 1
    finally:
        await source.stop()


@pytest.mark.asyncio
async def test_stop_cancels_running_tasks():
    """stop() cancels parked tasks deterministically — no reliance on real-clock timing.

    A long interval keeps each task parked in ``asyncio.sleep`` (so it never fires
    during the test); the only behavior exercised is cancellation, which is
    deterministic and non-flaky.
    """
    callback = _RecordingCallback()
    source = CronSource()
    trigger = _make_trigger("slow", interval_sec=3600)
    await source.start([trigger], callback)
    assert len(source._tasks) == 1
    tasks = list(source._tasks)

    # Yield once so the spawned task reaches its first `await asyncio.sleep(...)`.
    await asyncio.sleep(0)
    await source.stop()

    assert source._tasks == []
    assert all(t.cancelled() or t.done() for t in tasks)
    assert callback.calls == []  # parked in sleep the whole time, never fired


@pytest.mark.asyncio
async def test_stop_with_no_tasks_is_noop():
    source = CronSource()
    await source.stop()  # should not raise
    assert source._tasks == []


def test_register_routes_is_noop():
    """Cron has no HTTP routes; register_routes() returns None and does not raise."""
    assert CronSource().register_routes(object()) is None


# ---------------------------------------------------------------------------
# Clock-time triggers
# ---------------------------------------------------------------------------

_LA = ZoneInfo("America/Los_Angeles")
_BERLIN = ZoneInfo("Europe/Berlin")


def _clock_trigger(name: str, source_config: dict) -> TriggerConfig:
    schedule = parse_clock_schedule(name, source_config)
    return TriggerConfig(
        name=name,
        source="cron",
        action={"prompt": "tick", "allowed_tools": []},
        source_config=source_config,
        schedule=schedule,
    )


class _FakeClock:
    """A wall clock that ``asyncio.sleep`` advances by the requested seconds.

    ``extra`` maps a step index to seconds added on that step only, standing in
    for a host that was suspended while the loop slept.
    """

    def __init__(self, start: datetime, extra: dict[int, float] | None = None) -> None:
        self.current = start.astimezone(UTC)
        self.steps = 0
        self.extra = extra or {}

    def now(self) -> datetime:
        return self.current

    def install(self, monkeypatch) -> None:
        real_sleep = asyncio.sleep

        async def fake_sleep(seconds):
            self.current += timedelta(seconds=seconds + self.extra.get(self.steps, 0.0))
            self.steps += 1
            await real_sleep(0)

        monkeypatch.setattr("osprey.dispatch.sources.cron.asyncio.sleep", fake_sleep)


async def _collect_fires(
    monkeypatch,
    clock: _FakeClock,
    trigger: TriggerConfig,
    zone,
    count: int,
    *,
    fail_first_with: Exception | None = None,
) -> list[datetime]:
    fires: list[datetime] = []
    done = asyncio.Event()

    async def callback(trig: TriggerConfig, payload: dict) -> str | None:  # noqa: ARG001 - fire-callback signature; the fire instant is what is asserted
        fires.append(clock.current)
        if len(fires) >= count:
            done.set()
        if fail_first_with is not None and len(fires) == 1:
            raise fail_first_with
        return "d-1"

    clock.install(monkeypatch)
    source = CronSource(now=clock.now, zone=zone)
    await source.start([trigger], callback)
    try:
        await asyncio.wait_for(done.wait(), timeout=30)
    finally:
        await source.stop()
    return fires


@pytest.mark.asyncio
async def test_a_clock_trigger_fires_on_the_listed_weekdays(monkeypatch):
    trigger = _clock_trigger(
        "weekday-report", {"at": ["07:45"], "days": ["mon", "tue", "wed", "thu", "fri"]}
    )
    clock = _FakeClock(datetime(2026, 9, 25, 15, 0, tzinfo=_LA))

    fires = await _collect_fires(monkeypatch, clock, trigger, _LA, 2)

    assert fires[:2] == [
        datetime(2026, 9, 28, 7, 45, tzinfo=_LA),
        datetime(2026, 9, 29, 7, 45, tzinfo=_LA),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("zone", "start", "first", "second"),
    [
        pytest.param(
            _LA,
            datetime(2027, 3, 13, 12, 0, tzinfo=_LA),
            datetime(2027, 3, 14, 10, 0, tzinfo=UTC),
            datetime(2027, 3, 15, 9, 30, tzinfo=UTC),
            id="America/Los_Angeles",
        ),
        pytest.param(
            _BERLIN,
            datetime(2027, 3, 27, 12, 0, tzinfo=_BERLIN),
            datetime(2027, 3, 28, 1, 0, tzinfo=UTC),
            datetime(2027, 3, 29, 0, 30, tzinfo=UTC),
            id="Europe/Berlin",
        ),
    ],
)
async def test_a_time_skipped_by_spring_forward_fires_at_the_first_valid_minute(
    monkeypatch, zone, start, first, second
):
    trigger = _clock_trigger("night", {"at": ["02:30"]})

    fires = await _collect_fires(monkeypatch, _FakeClock(start), trigger, zone, 2)

    assert fires[:2] == [first, second]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("zone", "at", "start", "first", "second", "repeat"),
    [
        pytest.param(
            _LA,
            "01:30",
            datetime(2026, 10, 31, 12, 0, tzinfo=_LA),
            datetime(2026, 11, 1, 8, 30, tzinfo=UTC),
            datetime(2026, 11, 2, 9, 30, tzinfo=UTC),
            datetime(2026, 11, 1, 9, 30, tzinfo=UTC),
            id="America/Los_Angeles",
        ),
        pytest.param(
            _BERLIN,
            "02:30",
            datetime(2026, 10, 24, 12, 0, tzinfo=_BERLIN),
            datetime(2026, 10, 25, 0, 30, tzinfo=UTC),
            datetime(2026, 10, 26, 1, 30, tzinfo=UTC),
            datetime(2026, 10, 25, 1, 30, tzinfo=UTC),
            id="Europe/Berlin",
        ),
    ],
)
async def test_a_time_repeated_by_fall_back_fires_once_at_its_first_occurrence(
    monkeypatch, zone, at, start, first, second, repeat
):
    trigger = _clock_trigger("night", {"at": [at]})

    fires = await _collect_fires(monkeypatch, _FakeClock(start), trigger, zone, 2)

    assert fires[:2] == [first, second]
    assert repeat not in fires


@pytest.mark.asyncio
async def test_a_slot_the_dispatcher_wakes_late_for_is_skipped_not_made_up(monkeypatch, caplog):
    trigger = _clock_trigger("morning", {"at": ["07:45"]})
    # Start at 07:40 local: the fifth 60 s step is where the slot falls, and the
    # host sleeps an extra hour on the first step.
    clock = _FakeClock(datetime(2026, 9, 28, 7, 40, tzinfo=_LA), extra={0: 3600.0})

    with caplog.at_level(logging.WARNING, logger="osprey.dispatch.sources.cron"):
        fires = await _collect_fires(monkeypatch, clock, trigger, _LA, 1)

    assert fires == [datetime(2026, 9, 29, 7, 45, tzinfo=_LA)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("morning" in m and "2026-09-28T07:45:00-07:00" in m for m in warnings)


@pytest.mark.asyncio
async def test_start_logs_the_zone_the_slot_before_start_and_the_next_fire(caplog):
    trigger = _clock_trigger("morning", {"at": ["07:45"]})
    clock = _FakeClock(datetime(2026, 9, 28, 12, 0, tzinfo=_LA))
    source = CronSource(now=clock.now, zone=_LA)

    with caplog.at_level(logging.INFO, logger="osprey.dispatch.sources.cron"):
        await source.start([trigger], _RecordingCallback())
    await source.stop()

    lines = [r.getMessage() for r in caplog.records if "morning" in r.getMessage()]
    assert any(
        "America/Los_Angeles" in m
        and "2026-09-28T07:45:00-07:00" in m
        and "2026-09-29T07:45:00-07:00" in m
        for m in lines
    )


@pytest.mark.asyncio
async def test_a_clock_tick_on_a_full_queue_is_dropped_and_the_next_slot_still_fires(
    monkeypatch, caplog
):
    trigger = _clock_trigger("morning", {"at": ["07:45"]})
    clock = _FakeClock(datetime(2026, 9, 28, 7, 0, tzinfo=_LA))

    with caplog.at_level(logging.WARNING, logger="osprey.dispatch.sources.cron"):
        fires = await _collect_fires(
            monkeypatch, clock, trigger, _LA, 2, fail_first_with=QueueFullError("full")
        )

    assert fires[:2] == [
        datetime(2026, 9, 28, 7, 45, tzinfo=_LA),
        datetime(2026, 9, 29, 7, 45, tzinfo=_LA),
    ]
    assert any(
        r.levelno == logging.WARNING and "queue full" in r.getMessage() for r in caplog.records
    )


@pytest.mark.asyncio
async def test_the_facility_zone_comes_from_system_timezone(monkeypatch, tmp_path, caplog):
    import osprey.utils.config as _cfg

    (tmp_path / "config.yml").write_text("system:\n  timezone: Europe/Berlin\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    monkeypatch.setattr(_cfg, "_config_cache", {})

    trigger = _clock_trigger("morning", {"at": ["07:45"]})
    source = CronSource()
    with caplog.at_level(logging.INFO, logger="osprey.dispatch.sources.cron"):
        await source.start([trigger], _RecordingCallback())
    await source.stop()

    assert any(
        "morning" in r.getMessage() and "Europe/Berlin" in r.getMessage() for r in caplog.records
    )
