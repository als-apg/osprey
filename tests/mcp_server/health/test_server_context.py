"""Unit tests for :mod:`osprey.mcp_server.health.server_context`.

The context composes a config loader, a runtime lifecycle, and the health suite
runner behind a validity check, a wedge breaker, and a single-flight lock. These
tests replace all three collaborators with lightweight stubs (monkeypatched onto
the module) and drive time and the disk signature through injected callables, so
nothing sleeps and no real config, connector, or worker thread is touched.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey.health.config import parse_health_config
from osprey.health.models import CheckReport, CheckResult, Status
from osprey.mcp_server.health import server_context as sc
from osprey.mcp_server.health.server_context import (
    HealthRefreshSuppressedError,
    HealthServerContext,
    get_server_context,
    initialize_server_context,
    reset_server_context,
)

# --- stubs ------------------------------------------------------------------


class _StubLoaded:
    """Stand-in for :class:`LoadedHealthConfig` — only the read fields matter."""

    def __init__(self, *, tag: str = "cfg") -> None:
        self.records: list[Any] = []
        self.extra_rows: list[CheckResult] = []
        self.settings = parse_health_config(None)  # interval_s=60, suite_timeout_s=30
        self.expanded: dict[str, Any] = {"control_system": {"type": "mock", "tag": tag}}
        self.control_system: dict[str, Any] = self.expanded["control_system"]
        self.config_ok = True


class _StubLoader:
    """Records ``load()`` calls and returns a fresh ``_StubLoaded`` each time."""

    def __init__(self, config_path: Any = None) -> None:
        self.config_path = config_path
        self.load_calls = 0

    def load(self) -> _StubLoaded:
        self.load_calls += 1
        return _StubLoaded(tag=f"load{self.load_calls}")


class _StubRuntime:
    """Opaque runtime sentinel passed through to the suite runner."""


class _StubLifecycle:
    """Counts reconcile/shutdown and returns caller-configured notice rows."""

    def __init__(self) -> None:
        self.reconcile_calls = 0
        self.shutdown_calls = 0
        self.notice_rows: list[CheckResult] = []
        self.runtime: Any = _StubRuntime()

    def reconcile(self, expanded: Any) -> list[CheckResult]:
        self.reconcile_calls += 1
        return list(self.notice_rows)

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def env(monkeypatch):
    """Install stubs on the module and return the mutable control state.

    ``state`` drives the injected clock (``t``), the injected disk signature
    (``sig``), and the abandoned-thread count (``alive``). ``suite_calls`` counts
    real suite runs. ``loaders`` / ``lifecycles`` capture the instances the
    context constructed so tests can assert on them.
    """
    state: dict[str, Any] = {"t": 0.0, "sig": ("sigA",), "alive": 0, "suite_calls": 0}
    loaders: list[_StubLoader] = []
    lifecycles: list[_StubLifecycle] = []

    def _make_loader(config_path: Any = None) -> _StubLoader:
        loader = _StubLoader(config_path)
        loaders.append(loader)
        return loader

    def _make_lifecycle(*args: Any, **kwargs: Any) -> _StubLifecycle:
        lifecycle = _StubLifecycle()
        lifecycles.append(lifecycle)
        return lifecycle

    async def _run_sync(fn, *args, timeout_s):
        # Yield once so single-flight contention is genuinely exercised.
        await asyncio.sleep(0)
        return fn(*args)

    async def _run_health_suite(records, **kwargs):
        state["suite_calls"] += 1
        return CheckReport(results=[CheckResult("suite", "suite", Status.OK, "ok")])

    monkeypatch.setattr(sc, "HealthConfigLoader", _make_loader)
    monkeypatch.setattr(sc, "HealthRuntimeLifecycle", _make_lifecycle)
    monkeypatch.setattr(sc, "run_health_suite", _run_health_suite)
    monkeypatch.setattr(sc.offload, "run_sync", _run_sync)
    monkeypatch.setattr(sc.offload, "abandoned_alive_count", lambda: state["alive"])

    state["loaders"] = loaders
    state["lifecycles"] = lifecycles
    return state


def _make_ctx(env) -> HealthServerContext:
    return HealthServerContext(
        clock=lambda: env["t"],
        disk_signature=lambda: env["sig"],
    )


@pytest.fixture(autouse=True)
def _reset_health_singleton():
    reset_server_context()
    yield
    reset_server_context()


# --- caching / validity -----------------------------------------------------


async def test_cold_call_runs_suite_once_then_serves_cache(env):
    ctx = _make_ctx(env)

    first = await ctx.get_poll_report()
    assert first.cached is False
    assert first.refresh_suppressed is False
    assert first.age_s == 0.0
    assert env["suite_calls"] == 1
    assert env["loaders"][0].load_calls == 1
    # The suite row plus nothing else (no extra rows / notices configured).
    assert [r.name for r in first.report.results] == ["suite"]

    # Second call within interval_s, unchanged signature: pure cache serve.
    env["t"] = 10.0
    second = await ctx.get_poll_report()
    assert second.cached is True
    assert second.refresh_suppressed is False
    assert second.age_s == pytest.approx(10.0)
    assert env["suite_calls"] == 1  # no new suite run
    assert env["loaders"][0].load_calls == 1  # no new load


async def test_age_past_interval_triggers_refresh(env):
    ctx = _make_ctx(env)
    await ctx.get_poll_report()
    assert env["suite_calls"] == 1

    env["t"] = 61.0  # interval_s default is 60
    result = await ctx.get_poll_report()
    assert result.cached is False
    assert env["suite_calls"] == 2


async def test_changed_signature_refreshes_regardless_of_age(env):
    ctx = _make_ctx(env)
    await ctx.get_poll_report()
    assert env["suite_calls"] == 1

    env["t"] = 1.0  # well within interval
    env["sig"] = ("sigB",)  # but the config changed on disk
    result = await ctx.get_poll_report()
    assert result.cached is False
    assert env["suite_calls"] == 2


# --- wedge breaker ----------------------------------------------------------


async def test_wedge_serves_cache_flagged_and_does_not_refresh(env):
    ctx = _make_ctx(env)
    # Cold call runs the suite cleanly (not yet wedged) and caches a snapshot.
    first = await ctx.get_poll_report()
    assert first.cached is False
    assert env["suite_calls"] == 1

    # A worker thread is now wedged; snapshot goes stale; signature unchanged.
    env["alive"] = 1
    env["t"] = 100.0
    suppressed = await ctx.get_poll_report()
    assert suppressed.cached is True
    assert suppressed.refresh_suppressed is True
    assert env["suite_calls"] == 1  # no refresh
    assert env["loaders"][0].load_calls == 1


async def test_wedge_disk_change_permits_exactly_one_refresh(env):
    ctx = _make_ctx(env)
    await ctx.get_poll_report()  # clean cold refresh caches a snapshot
    assert env["suite_calls"] == 1

    # Wedge appears; a stale, unchanged-signature call anchors suppression at sigA.
    env["alive"] = 1
    env["t"] = 100.0
    anchored = await ctx.get_poll_report()
    assert anchored.refresh_suppressed is True
    assert env["suite_calls"] == 1

    # Signature change escapes suppression for exactly one attempt.
    env["sig"] = ("sigB",)
    escaped = await ctx.get_poll_report()
    assert escaped.cached is False
    assert escaped.refresh_suppressed is False
    assert env["suite_calls"] == 2

    # Still wedged, still sigB, snapshot stale again -> suppressed once more.
    env["t"] = 300.0
    again = await ctx.get_poll_report()
    assert again.refresh_suppressed is True
    assert env["suite_calls"] == 2  # no second escape for the same signature


async def test_cold_and_suppressed_raises(env):
    ctx = _make_ctx(env)
    env["alive"] = 2  # wedged before any snapshot exists
    with pytest.raises(HealthRefreshSuppressedError) as excinfo:
        await ctx.get_poll_report()
    assert excinfo.value.wedged_count == 2
    assert env["suite_calls"] == 0


# --- single-flight ----------------------------------------------------------


async def test_concurrent_cold_calls_run_one_suite(env):
    ctx = _make_ctx(env)

    results = await asyncio.gather(
        ctx.get_poll_report(),
        ctx.get_poll_report(),
    )

    assert env["suite_calls"] == 1
    assert env["loaders"][0].load_calls == 1
    # Both callers observe the same cached report object.
    assert results[0].report is results[1].report
    # Exactly one caller ran the suite; the other served the fresh cache.
    assert {r.cached for r in results} == {False, True}


# --- reconcile pass-through -------------------------------------------------


async def test_reconcile_runs_each_refresh_and_notice_rows_passed_through(env):
    ctx = _make_ctx(env)
    lifecycle = None

    await ctx.get_poll_report()
    lifecycle = env["lifecycles"][0]
    assert lifecycle.reconcile_calls == 1

    # Configure a restart-notice row the lifecycle emits on the next refresh.
    notice = CheckResult("control_system", "configuration", Status.WARNING, "restart notice")
    lifecycle.notice_rows = [notice]

    env["t"] = 61.0
    result = await ctx.get_poll_report()
    assert lifecycle.reconcile_calls == 2
    assert notice in result.report.results


# --- shutdown ---------------------------------------------------------------


async def test_shutdown_delegates_once_even_when_called_twice(env):
    ctx = _make_ctx(env)
    await ctx.get_poll_report()
    lifecycle = env["lifecycles"][0]

    await ctx.shutdown()
    await ctx.shutdown()
    assert lifecycle.shutdown_calls == 1


# --- singleton --------------------------------------------------------------


async def test_singleton_lifecycle(env):
    with pytest.raises(RuntimeError):
        get_server_context()

    ctx = initialize_server_context()
    assert get_server_context() is ctx

    reset_server_context()
    with pytest.raises(RuntimeError):
        get_server_context()
