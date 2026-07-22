"""Unit tests for the ``health_check_full`` MCP tool.

The tool is the approval-gated escalation path: it always runs a fresh full-tier
suite, constructs a private per-call :class:`~osprey.health.runtime.HealthRuntime`
(shut down by its context manager even on error), and never reads or writes the
poll path's cached snapshot. These tests replace the config loader, the runtime,
and the suite runner with lightweight stubs installed on the tool module, so no
real config, connector, or worker thread is touched.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey.health.config import parse_health_config
from osprey.health.models import CheckReport, CheckResult, Status
from osprey.mcp_server.health.tools import health_check_full as hcf
from tests.mcp_server.conftest import (
    assert_raises_error,
    extract_response_dict,
    get_tool_fn,
)

# --- stubs ------------------------------------------------------------------


class _StubRecord:
    """A merged category record — only ``name`` matters for validation."""

    def __init__(self, name: str) -> None:
        self.name = name


class _StubLoaded:
    """Stand-in for :class:`LoadedHealthConfig` — the read fields the tool uses."""

    def __init__(
        self,
        *,
        records: list[_StubRecord] | None = None,
        extra_rows: list[CheckResult] | None = None,
    ) -> None:
        self.records = (
            records
            if records is not None
            else [_StubRecord("file_system"), _StubRecord("providers")]
        )
        self.extra_rows = extra_rows if extra_rows is not None else []
        self.settings = parse_health_config(None)  # suite_timeout_s=30, on_demand_timeout_s=None
        self.expanded: dict[str, Any] = {"control_system": {"type": "mock", "tag": "full"}}
        self.control_system: dict[str, Any] = self.expanded["control_system"]
        self.config_ok = True


class _StubLoader:
    """Returns the caller-configured ``_StubLoaded`` and counts ``load()`` calls."""

    def __init__(self, loaded: _StubLoaded) -> None:
        self._loaded = loaded
        self.load_calls = 0

    def load(self) -> _StubLoaded:
        self.load_calls += 1
        return self._loaded


class _StubContext:
    """Fake server context exposing only ``loader`` plus a poll-snapshot sentinel.

    ``_snapshot`` stands in for the poll path's cached snapshot; the tool must
    never read or mutate it. Tests assert its identity is unchanged after a run.
    """

    def __init__(self, loaded: _StubLoaded, *, snapshot: Any) -> None:
        self.loader = _StubLoader(loaded)
        self._snapshot = snapshot


class _StubRuntime:
    """Async-CM runtime sentinel that records enter/shutdown for teardown checks."""

    def __init__(self, control_system_config: dict[str, Any]) -> None:
        self.control_system_config = control_system_config
        self.entered = False
        self.shutdown_calls = 0

    async def __aenter__(self) -> _StubRuntime:
        self.entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.shutdown_calls += 1
        return False  # never suppress the suite's exception


@pytest.fixture
def env(monkeypatch):
    """Install tool-module stubs and return the mutable control state.

    ``state`` lets a test configure the loaded config, the report the suite
    returns, and (optionally) an exception the suite raises. ``runtimes`` and
    ``suite_calls`` capture what the tool constructed / passed so tests can
    assert on the per-call runtime and the suite kwargs.
    """
    loaded = _StubLoaded()
    snapshot_sentinel = object()
    context = _StubContext(loaded, snapshot=snapshot_sentinel)

    state: dict[str, Any] = {
        "loaded": loaded,
        "context": context,
        "snapshot_sentinel": snapshot_sentinel,
        "runtimes": [],
        "suite_calls": [],
        "suite_records": [],
        "suite_raises": None,
        "report_factory": lambda: CheckReport(
            results=[CheckResult("suite", "file_system", Status.OK, "ok")],
            elapsed_ms=12.0,
        ),
    }

    def _make_runtime(control_system_config: dict[str, Any]) -> _StubRuntime:
        rt = _StubRuntime(control_system_config)
        state["runtimes"].append(rt)
        return rt

    async def _run_sync(fn, *args, timeout_s):
        # Yield once so concurrency is genuinely exercised; then run the load.
        await asyncio.sleep(0)
        return fn(*args)

    async def _run_health_suite(records, **kwargs):
        state["suite_calls"].append(kwargs)
        state["suite_records"].append(list(records))
        if state["suite_raises"] is not None:
            raise state["suite_raises"]
        await asyncio.sleep(0)
        return state["report_factory"]()

    monkeypatch.setattr(hcf, "get_server_context", lambda: context)
    monkeypatch.setattr(hcf, "HealthRuntime", _make_runtime)
    monkeypatch.setattr(hcf, "run_health_suite", _run_health_suite)
    monkeypatch.setattr(hcf.offload, "run_sync", _run_sync)
    # The full tool must be indifferent to the wedge breaker; default a live
    # abandoned thread so any accidental suppression check would trip.
    monkeypatch.setattr(hcf.offload, "abandoned_alive_count", lambda: 3)
    return state


def _call():
    return get_tool_fn(hcf.health_check_full)


# --- envelope ---------------------------------------------------------------


async def test_envelope_wire_shape_and_always_fresh(env):
    result = extract_response_dict(await _call()(None))

    # Locked report wire keys.
    for key in (
        "summary",
        "ok",
        "warnings",
        "errors",
        "skips",
        "total",
        "elapsed_ms",
        "deadline_hit",
        "results",
    ):
        assert key in result, f"missing report key {key!r}"

    # Full tier is always fresh and never suppressed.
    assert result["cached"] is False
    assert result["age_s"] == 0
    assert result["refresh_suppressed"] is False


# --- pass-through / private per-call runtime --------------------------------


async def test_full_true_and_categories_and_config_passed_through_unfiltered(env):
    await _call()(None)

    assert len(env["suite_calls"]) == 1
    kwargs = env["suite_calls"][0]
    assert kwargs["full"] is True
    assert kwargs["categories"] is None
    assert kwargs["config"] is env["loaded"].expanded
    assert kwargs["suite_timeout_s"] == env["loaded"].settings.suite_timeout_s
    assert kwargs["on_demand_timeout_s"] == env["loaded"].settings.on_demand_timeout_s

    # Exactly one private runtime, built from the loaded control-system section,
    # shut down exactly once by its context manager.
    assert len(env["runtimes"]) == 1
    runtime = env["runtimes"][0]
    assert runtime.control_system_config is env["loaded"].control_system
    assert runtime.entered is True
    assert runtime.shutdown_calls == 1


async def test_categories_deduped_and_passed_through_when_filtered(env):
    await _call()(["file_system", "file_system"])

    kwargs = env["suite_calls"][0]
    assert kwargs["full"] is True
    assert kwargs["categories"] == ["file_system"]


async def test_runtime_shut_down_once_when_suite_raises(env):
    env["suite_raises"] = RuntimeError("suite boom")

    with pytest.raises(RuntimeError, match="suite boom"):
        await _call()(None)

    # The context manager tore the private runtime down even on the error path.
    assert len(env["runtimes"]) == 1
    assert env["runtimes"][0].shutdown_calls == 1


# --- validation -------------------------------------------------------------


async def test_unknown_category_errors_and_lists_valid_names(env):
    with assert_raises_error(error_type="unknown_category") as ctx:
        await _call()(["file_system", "bogus"])

    envelope = ctx["envelope"]
    assert "bogus" in envelope["error_message"]
    # Valid names surfaced for the operator/agent (same shape as health_check).
    assert "file_system" in envelope["error_message"] and "providers" in envelope["error_message"]
    assert envelope["details"]["unknown"] == ["bogus"]
    assert "file_system" in envelope["details"]["valid"]
    # No suite ran once validation failed.
    assert env["suite_calls"] == []


# --- plugin diagnostic rows -------------------------------------------------


async def test_extra_rows_present_once_when_unfiltered(env):
    diag = CheckResult("plugin.load", "plugins", Status.WARNING, "plugin diag")
    env["loaded"].extra_rows = [diag]

    result = extract_response_dict(await _call()(None))

    names = [r["name"] for r in result["results"]]
    assert names.count("plugin.load") == 1


async def test_extra_rows_absent_when_filtered(env):
    diag = CheckResult("plugin.load", "plugins", Status.WARNING, "plugin diag")
    env["loaded"].extra_rows = [diag]

    result = extract_response_dict(await _call()(["file_system"]))

    names = [r["name"] for r in result["results"]]
    assert "plugin.load" not in names


# --- poll-snapshot isolation ------------------------------------------------


async def test_pre_existing_poll_snapshot_untouched(env):
    ctx = env["context"]
    assert ctx._snapshot is env["snapshot_sentinel"]

    await _call()(None)

    # The full run never reads or writes the poll snapshot.
    assert ctx._snapshot is env["snapshot_sentinel"]


async def test_runs_with_no_poll_snapshot(env):
    ctx = env["context"]
    ctx._snapshot = None

    result = extract_response_dict(await _call()(None))

    assert result["cached"] is False
    assert ctx._snapshot is None  # still untouched


# --- wedge indifference -----------------------------------------------------


async def test_wedge_breaker_does_not_block_full_run(env, monkeypatch):
    # A persistently-wedged suite (abandoned threads alive) must NOT suppress the
    # approval-gated full run — operator approval is the rate limiter.
    monkeypatch.setattr(hcf.offload, "abandoned_alive_count", lambda: 9)

    result = extract_response_dict(await _call()(None))

    assert result["cached"] is False
    assert result["refresh_suppressed"] is False
    assert len(env["suite_calls"]) == 1
    assert len(env["runtimes"]) == 1


# --- concurrency ------------------------------------------------------------


async def test_full_runs_independently_of_poll_refresh_in_flight(env):
    # Simulate a poll-side refresh holding its own runtime mid-flight. The full
    # path must construct ITS OWN private runtime and complete without touching
    # or double-constructing the poll runtime.
    poll_started = asyncio.Event()
    poll_release = asyncio.Event()
    poll_runtime_constructions = {"n": 0}

    async def _poll_refresh():
        poll_runtime_constructions["n"] += 1  # poll builds its own runtime once
        poll_started.set()
        await poll_release.wait()

    poll_task = asyncio.create_task(_poll_refresh())
    await poll_started.wait()  # poll is now mid-flight

    result = extract_response_dict(await _call()(None))

    # Full completed while the poll refresh is still blocked.
    assert result["cached"] is False
    assert not poll_release.is_set()
    # Full constructed exactly one private runtime and tore it down.
    assert len(env["runtimes"]) == 1
    assert env["runtimes"][0].shutdown_calls == 1
    # The full path did not construct or duplicate the poll runtime.
    assert poll_runtime_constructions["n"] == 1

    poll_release.set()
    await poll_task
