"""Tests for `HealthCheckEngine`, the `/checks` cache + refresh scheduler (task 2.3).

Behaviors under test, per the proposal's route-test criteria:

- **Cold / warming:** the first `/checks` returns a ``warming`` envelope and kicks
  exactly one background run; concurrent cold callers share it.
- **Steady state + refresh-ahead:** the cache is served in constant time; a
  request past ``interval_s − suite_timeout_s`` kicks a refresh (fake clock);
  ``stale`` flips only past ``interval_s``.
- **Envelope + unfiltered serving:** ``CheckReport.to_dict()`` plus
  ``stale/warming/interval_s/title``; the suite always runs ``full=False``,
  ``categories=None``; plugin-diagnostic and restart-notice rows are appended.
- **Circuit breaker:** a wedged sync phase (real ``offload`` timeout leaving a
  live daemon thread) trips suppression; a completing phase does not; suppression
  resumes on a disk-signature change or after the backoff.

Timing is driven by an injected fake clock and the breaker's disk probe by an
injected signature, so no test sleeps or touches the wall clock.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest

from osprey.health.config import HealthSettings
from osprey.health.models import CheckResult, Status
from osprey.interfaces.health.engine import HealthCheckEngine
from osprey.interfaces.health.lifecycle import RESTART_NOTICE_MESSAGE, HealthRuntimeLifecycle
from osprey.interfaces.health.loader import LoadedHealthConfig


def _settings(
    *, suite_timeout_s: float = 30.0, interval_s: float = 60.0, title: str = "T"
) -> HealthSettings:
    return HealthSettings(
        suite_timeout_s=suite_timeout_s,
        interval_s=interval_s,
        on_demand_timeout_s=None,
        title=title,
    )


def _loaded(
    settings: HealthSettings,
    *,
    expanded: dict[str, Any] | None = None,
    extra_rows: list[Any] | None = None,
    config_ok: bool = True,
) -> LoadedHealthConfig:
    exp = {} if expanded is None else expanded
    return LoadedHealthConfig(
        records=[],  # empty suite → run_health_suite returns an empty report fast
        extra_rows=extra_rows or [],
        settings=settings,
        expanded=exp,
        control_system=(exp or {}).get("control_system", {}) or {},
        config_ok=config_ok,
    )


class _FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _FakeLoader:
    """Loader returning a fixed (or swappable) `LoadedHealthConfig`, counting calls."""

    def __init__(self, result: LoadedHealthConfig) -> None:
        self.result = result
        self.calls = 0

    def load(self) -> LoadedHealthConfig:
        self.calls += 1
        return self.result


class _HangingLoader:
    """Loader whose `load` blocks until released — trips the breaker via a live thread."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.calls = 0

    def load(self) -> LoadedHealthConfig:
        self.calls += 1
        self.release.wait(timeout=10)
        return _loaded(_settings())


def _engine(loader: Any, **kwargs: Any) -> tuple[HealthCheckEngine, HealthRuntimeLifecycle]:
    lifecycle = HealthRuntimeLifecycle()
    engine = HealthCheckEngine(loader=loader, lifecycle=lifecycle, **kwargs)
    return engine, lifecycle


async def _drain(engine: HealthCheckEngine) -> None:
    """Await the in-flight refresh, if any, so its cache write lands."""
    task = engine.current_refresh_task()
    if task is not None:
        await task


# -- cold / warming ------------------------------------------------------------


async def test_cold_checks_returns_warming_and_kicks_one_run() -> None:
    loader = _FakeLoader(_loaded(_settings(interval_s=90.0, title="Zero-Config")))
    engine, _ = _engine(loader, settings=_settings(interval_s=90.0, title="Zero-Config"))

    env = engine.get_checks()
    assert env["warming"] is True
    assert env["stale"] is True
    assert env["results"] == []
    assert env["interval_s"] == 90.0
    assert env["title"] == "Zero-Config"
    assert engine.current_refresh_task() is not None  # first run kicked

    await _drain(engine)
    assert loader.calls == 1

    env2 = engine.get_checks()
    assert env2["warming"] is False


async def test_concurrent_cold_callers_share_one_run() -> None:
    loader = _FakeLoader(_loaded(_settings()))
    engine, _ = _engine(loader)

    first = engine.get_checks()
    second = engine.get_checks()  # while the first run is still in flight
    assert first["warming"] is True
    assert second["warming"] is True

    await _drain(engine)
    assert loader.calls == 1  # exactly one suite run, not one per caller


# -- envelope + unfiltered serving ---------------------------------------------


async def test_envelope_shape_and_appended_rows() -> None:
    extra = CheckResult("plugin.load", "plugins", Status.ERROR, "boom")
    loader = _FakeLoader(_loaded(_settings(title="System Health"), extra_rows=[extra]))
    engine, _ = _engine(loader, settings=_settings(title="System Health"))

    engine.get_checks()
    await _drain(engine)
    env = engine.get_checks()

    # Wire contract: report keys + envelope keys.
    for key in (
        "summary",
        "ok",
        "errors",
        "skips",
        "total",
        "elapsed_ms",
        "deadline_hit",
        "results",
    ):
        assert key in env
    assert env["stale"] is False
    assert env["warming"] is False
    assert env["interval_s"] == 60.0
    assert env["title"] == "System Health"
    # Plugin-diagnostic row is appended, unfiltered.
    assert any(r["name"] == "plugin.load" for r in env["results"])


async def test_suite_run_is_unfiltered_and_never_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_suite(records: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        from osprey.health.models import CheckReport

        return CheckReport()

    monkeypatch.setattr("osprey.interfaces.health.engine.run_health_suite", fake_suite)

    loader = _FakeLoader(_loaded(_settings(), expanded={"api": {"providers": {}}}))
    engine, _ = _engine(loader)
    engine.get_checks()
    await _drain(engine)

    assert captured["full"] is False
    assert captured["categories"] is None
    assert captured["config"] == {"api": {"providers": {}}}  # config conduit forwarded


async def test_restart_notice_row_is_appended(monkeypatch: pytest.MonkeyPatch) -> None:
    notice = CheckResult("control_system", "configuration", Status.WARNING, RESTART_NOTICE_MESSAGE)

    class _NoticeLifecycle(HealthRuntimeLifecycle):
        def reconcile(self, expanded: dict[str, Any] | None) -> list[CheckResult]:
            super().reconcile(expanded)  # preserve the runtime-construction contract
            return [notice]

    lifecycle = _NoticeLifecycle()
    loader = _FakeLoader(_loaded(_settings()))
    engine = HealthCheckEngine(loader=loader, lifecycle=lifecycle)

    engine.get_checks()
    await _drain(engine)
    env = engine.get_checks()

    assert any(r["message"] == RESTART_NOTICE_MESSAGE for r in env["results"])


# -- refresh-ahead + staleness (fake clock) ------------------------------------


async def test_refresh_ahead_and_stale_thresholds() -> None:
    clock = _FakeClock()
    settings = _settings(suite_timeout_s=30.0, interval_s=60.0)
    loader = _FakeLoader(_loaded(settings))
    engine, _ = _engine(loader, settings=settings, clock=clock)

    # First (cold) run, cached at t.
    engine.get_checks()
    await _drain(engine)
    assert loader.calls == 1

    # Age below the refresh-ahead threshold (interval − suite_timeout = 30): no kick.
    clock.advance(25.0)
    env = engine.get_checks()
    await _drain(engine)
    assert loader.calls == 1
    assert env["stale"] is False

    # Age past the refresh-ahead threshold but under interval: kick, not yet stale.
    clock.advance(10.0)  # age now 35
    env = engine.get_checks()
    assert env["stale"] is False
    await _drain(engine)
    assert loader.calls == 2

    # Fresh cache again (re-cached at t=1035). Push age past interval → stale.
    clock.advance(65.0)
    env = engine.get_checks()
    assert env["stale"] is True
    await _drain(engine)


async def test_get_checks_is_single_flight_under_repeated_calls() -> None:
    clock = _FakeClock()
    settings = _settings(suite_timeout_s=30.0, interval_s=60.0)
    loader = _FakeLoader(_loaded(settings))
    engine, _ = _engine(loader, settings=settings, clock=clock)

    engine.get_checks()
    await _drain(engine)

    # Past the refresh-ahead threshold; three rapid polls must not stack runs.
    clock.advance(40.0)
    engine.get_checks()
    engine.get_checks()
    engine.get_checks()
    await _drain(engine)
    assert loader.calls == 2  # one cold + one refresh-ahead, never three


# -- circuit breaker -----------------------------------------------------------


async def test_breaker_trips_on_wedged_sync_phase_and_suppresses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    loader = _HangingLoader()
    # Tiny suite_timeout so offload.run_sync abandons the hung load quickly.
    settings = _settings(suite_timeout_s=0.05, interval_s=0.1)
    engine, _ = _engine(loader, settings=settings, clock=clock)

    try:
        with caplog.at_level(logging.WARNING, logger="osprey.interfaces.health.engine"):
            engine.get_checks()  # kicks the wedged refresh
            await _drain(engine)  # times out → thread abandoned, still alive → trip

        assert loader.calls == 1
        # Suppressed: a subsequent poll must not kick a second run.
        env = engine.get_checks()
        assert env["warming"] is True  # still no report
        await _drain(engine)
        assert loader.calls == 1
        assert any("suppressing refreshes" in r.message for r in caplog.records)
    finally:
        loader.release.set()  # let the daemon thread finish so it prunes


async def test_breaker_does_not_trip_on_completing_phase() -> None:
    clock = _FakeClock()
    loader = _FakeLoader(_loaded(_settings(suite_timeout_s=30.0, interval_s=60.0)))
    engine, _ = _engine(
        loader, settings=_settings(suite_timeout_s=30.0, interval_s=60.0), clock=clock
    )

    engine.get_checks()
    await _drain(engine)
    # A completing sync phase abandons nothing; a later refresh-ahead kick fires.
    clock.advance(40.0)
    engine.get_checks()
    await _drain(engine)
    assert loader.calls == 2


async def test_breaker_resumes_on_disk_signature_change(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = _FakeClock()
    loader = _HangingLoader()
    sig = {"v": 0}
    settings = _settings(suite_timeout_s=0.05, interval_s=0.1)
    engine, _ = _engine(
        loader,
        settings=settings,
        clock=clock,
        disk_signature=lambda: sig["v"],
        backoff_factor=1000.0,  # backoff far away → only a disk change can resume
    )

    try:
        engine.get_checks()
        await _drain(engine)
        assert loader.calls == 1

        # Still suppressed (backoff not elapsed, sig unchanged).
        engine.get_checks()
        await _drain(engine)
        assert loader.calls == 1

        # A config/.env edit changes the signature → next poll resumes.
        sig["v"] = 1
        engine.get_checks()
        await _drain(engine)
        assert loader.calls == 2
    finally:
        loader.release.set()


async def test_breaker_resumes_after_backoff_elapses() -> None:
    clock = _FakeClock()
    loader = _HangingLoader()
    settings = _settings(suite_timeout_s=0.05, interval_s=0.1)
    engine, _ = _engine(
        loader,
        settings=settings,
        clock=clock,
        disk_signature=lambda: 0,  # constant → only the backoff can resume
        backoff_factor=10.0,
    )

    try:
        engine.get_checks()
        await _drain(engine)
        assert loader.calls == 1

        # Within the backoff window: still suppressed.
        clock.advance(0.5)
        engine.get_checks()
        await _drain(engine)
        assert loader.calls == 1

        # Past ~backoff_factor × interval_s (10 × 0.1 = 1.0s): resume.
        clock.advance(5.0)
        engine.get_checks()
        await _drain(engine)
        assert loader.calls == 2
    finally:
        loader.release.set()


# -- teardown wiring -----------------------------------------------------------


async def test_engine_wires_inflight_task_provider_to_lifecycle() -> None:
    loader = _FakeLoader(_loaded(_settings()))
    engine, lifecycle = _engine(loader)

    # Idle: no in-flight task exposed.
    assert lifecycle._inflight_task_provider is not None
    assert lifecycle._inflight_task_provider() is None

    engine.get_checks()  # kick a refresh
    # The lifecycle now sees the engine's in-flight refresh task for teardown.
    assert lifecycle._inflight_task_provider() is engine.current_refresh_task()
    await _drain(engine)
