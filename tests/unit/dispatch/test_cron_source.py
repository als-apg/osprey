"""Unit tests for the cron trigger source."""

from __future__ import annotations

import asyncio

import pytest

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
    """Deterministic: fake sleep fires once, then cancels the loop."""
    callback = _RecordingCallback()
    source = CronSource()
    trigger = _make_trigger("nightly", interval_sec=300)

    sleep_calls = {"n": 0}

    async def fake_sleep(_secs):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            raise asyncio.CancelledError
        return None

    monkeypatch.setattr("osprey.dispatch.sources.cron.asyncio.sleep", fake_sleep)

    await source.start([trigger], callback)
    # Let the spawned task run until it cancels itself on the 2nd sleep.
    await asyncio.gather(*source._tasks, return_exceptions=True)

    assert len(callback.calls) == 1
    fired_trigger, payload = callback.calls[0]
    assert fired_trigger is trigger
    assert payload["source"] == "cron"
    assert payload["trigger"] == "nightly"
    assert "timestamp" in payload


@pytest.mark.asyncio
async def test_invalid_interval_spawns_no_task(monkeypatch):
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
