"""Failure-class stamping at the dispatch_api terminal error sites.

Covers every site that produces a terminal error record for a run — the outer
worker timeout, the generic orchestration except, the stale-run sweep, and the
user-cancel branch (including the sweep-won-the-race merge) — and asserts the
core invariant: each run ends with exactly one terminal record and exactly one
per-class counter increment, carrying the honest tool-call count from the shared
stats map.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from osprey.mcp_server.dispatch_worker import dispatch_api, failure_class, run_stats, sdk_runner
from osprey.mcp_server.dispatch_worker.dispatch_api import DispatchRequest


@pytest.fixture
def counter_calls(monkeypatch):
    """Observe every _stamp counter increment via the register_counter_hook seam."""
    calls: list[str] = []
    failure_class.register_counter_hook(calls.append)
    yield calls
    failure_class.register_counter_hook(None)


@pytest.fixture
def persist_calls(monkeypatch):
    """Record persisted run_ids and neutralise the on-disk write during tests."""
    calls: list[str] = []

    def _fake_persist(run_id, run):
        calls.append(run_id)

    monkeypatch.setattr(dispatch_api, "_persist_run", _fake_persist)
    return calls


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Isolate the module-level run/stats maps between tests."""
    dispatch_api._runs.clear()
    dispatch_api._queues.clear()
    dispatch_api._tasks.clear()
    run_stats._run_stats.clear()
    yield
    dispatch_api._runs.clear()
    dispatch_api._queues.clear()
    dispatch_api._tasks.clear()
    run_stats._run_stats.clear()


def _req() -> DispatchRequest:
    return DispatchRequest(prompt="hello", allowed_tools=[])


# ---------------------------------------------------------------------------
# Per-site drivers: each drives one terminal error site to completion and
# leaves the terminal record in dispatch_api._runs[run_id].
# ---------------------------------------------------------------------------


async def _drive_timeout(monkeypatch, run_id: str) -> None:
    monkeypatch.setattr(dispatch_api, "DISPATCH_TIMEOUT_SEC", 0.05)

    async def _slow(**_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _slow)
    await dispatch_api._run_dispatch_task(run_id, _req())


async def _drive_generic(monkeypatch, run_id: str) -> None:
    async def _boom(**_kwargs):
        raise ValueError("orchestration blew up")

    monkeypatch.setattr(sdk_runner, "run_dispatch", _boom)
    await dispatch_api._run_dispatch_task(run_id, _req())


async def _drive_cancel(monkeypatch, run_id: str) -> None:
    async def _slow(**_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _slow)
    dispatch_api._runs[run_id] = {"status": "pending", "created_at": time.time()}
    task = asyncio.create_task(dispatch_api._run_dispatch_task(run_id, _req()))
    await asyncio.sleep(0.05)  # let the coroutine enter run_dispatch
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _drive_sweep(monkeypatch, run_id: str) -> None:
    # A stale *pending* run, created long enough ago to exceed the sweep cutoff.
    stale_age = dispatch_api.DISPATCH_TIMEOUT_SEC + 100
    dispatch_api._runs[run_id] = {"status": "pending", "created_at": time.time() - stale_age}
    dispatch_api._sweep_stale_runs()


_DRIVERS = {
    "timeout": (_drive_timeout, failure_class.FAILURE_INFRASTRUCTURE),
    "generic": (_drive_generic, failure_class.FAILURE_INFRASTRUCTURE),
    "cancel": (_drive_cancel, failure_class.FAILURE_RUN),
    "sweep": (_drive_sweep, failure_class.FAILURE_INFRASTRUCTURE),
}


# ---------------------------------------------------------------------------
# Core invariant, parametrized across every site
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site", sorted(_DRIVERS))
async def test_site_stamps_expected_class_exactly_once(
    site, monkeypatch, counter_calls, persist_calls
):
    driver, expected_class = _DRIVERS[site]
    run_id = f"run-{site}"

    await driver(monkeypatch, run_id)

    record = dispatch_api._runs[run_id]
    # Exactly one terminal record, stamped with the expected class...
    assert record["status"] == "error"
    assert record["failure_class"] == expected_class
    # ...and exactly one counter increment for this run.
    assert counter_calls == [expected_class]
    # Every terminal record is persisted.
    assert persist_calls.count(run_id) == 1


# ---------------------------------------------------------------------------
# Honest tool-call count sourced from the stats map (not hardcoded [])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("site", sorted(_DRIVERS))
async def test_site_stamps_honest_tool_call_count(site, monkeypatch, counter_calls, persist_calls):
    driver, _expected_class = _DRIVERS[site]
    run_id = f"count-{site}"
    for _ in range(3):
        run_stats.increment_tool_calls(run_id)

    await driver(monkeypatch, run_id)

    record = dispatch_api._runs[run_id]
    assert record["num_tool_calls"] == 3


async def test_zero_tool_calls_defaults_to_zero(monkeypatch, counter_calls, persist_calls):
    run_id = "count-none"
    await _drive_generic(monkeypatch, run_id)
    assert dispatch_api._runs[run_id]["num_tool_calls"] == 0


# ---------------------------------------------------------------------------
# Site-specific record shape
# ---------------------------------------------------------------------------


async def test_timeout_record_carries_timeout_message_and_duration(
    monkeypatch, counter_calls, persist_calls
):
    await _drive_timeout(monkeypatch, "r-timeout")
    record = dispatch_api._runs["r-timeout"]
    assert record["error"] == "Timed out after 0.05s"
    assert record["duration_sec"] == 0.05


async def test_generic_record_carries_exception_message(monkeypatch, counter_calls, persist_calls):
    await _drive_generic(monkeypatch, "r-generic")
    record = dispatch_api._runs["r-generic"]
    assert record["error"] == "orchestration blew up"


async def test_genuine_cancel_marks_cancelled_flag(monkeypatch, counter_calls, persist_calls):
    await _drive_cancel(monkeypatch, "r-cancel")
    record = dispatch_api._runs["r-cancel"]
    assert record["error"] == "cancelled by user"
    assert record["cancelled"] is True
    assert record["failure_class"] == failure_class.FAILURE_RUN


# ---------------------------------------------------------------------------
# CancelledError merge when the stale sweep won the race
# ---------------------------------------------------------------------------


async def test_cancel_merges_preserving_swept_record(monkeypatch, counter_calls, persist_calls):
    """Sweep wins the race: the cancel branch must not re-stamp or re-count."""
    run_id = "r-merge"

    async def _slow(**_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _slow)
    dispatch_api._runs[run_id] = {"status": "pending", "created_at": time.time()}
    task = asyncio.create_task(dispatch_api._run_dispatch_task(run_id, _req()))
    await asyncio.sleep(0.05)  # coroutine is now suspended inside run_dispatch

    # Simulate the sweep having already written + stamped a terminal record and
    # incremented the counter once. Clear counter_calls so we observe only what
    # the cancel branch adds after this point.
    swept = {
        "status": "error",
        "error": "Timed out after 330s",
        "failure_class": failure_class.FAILURE_INFRASTRUCTURE,
        "num_tool_calls": 0,
        "completed_at": time.time(),
    }
    dispatch_api._runs[run_id] = swept
    counter_calls.clear()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    record = dispatch_api._runs[run_id]
    # Swept record preserved verbatim; NOT overwritten with "cancelled by user".
    assert record["error"] == "Timed out after 330s"
    assert record["failure_class"] == failure_class.FAILURE_INFRASTRUCTURE
    # The cancel branch stamped nothing and counted nothing on top of the sweep.
    assert counter_calls == []


async def test_sweep_then_cancel_yields_single_record_and_increment(
    monkeypatch, counter_calls, persist_calls
):
    """End-to-end race: real sweep marks the run terminal, cancels its orphaned
    coroutine, and the coroutine's cancel branch merges — one record, one count."""
    run_id = "r-e2e"

    async def _slow(**_kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _slow)
    run_stats.increment_tool_calls(run_id)  # honest count = 1

    stale_age = dispatch_api.DISPATCH_TIMEOUT_SEC + 100
    dispatch_api._runs[run_id] = {"status": "pending", "created_at": time.time() - stale_age}
    task = asyncio.create_task(dispatch_api._run_dispatch_task(run_id, _req()))
    dispatch_api._tasks[run_id] = task
    await asyncio.sleep(0.05)  # coroutine enters run_dispatch

    dispatch_api._sweep_stale_runs()  # stamps infrastructure + cancels the task
    with pytest.raises(asyncio.CancelledError):
        await task

    record = dispatch_api._runs[run_id]
    assert record["status"] == "error"
    assert record["failure_class"] == failure_class.FAILURE_INFRASTRUCTURE
    assert record["num_tool_calls"] == 1
    # Exactly one increment total across sweep + cancel-merge.
    assert counter_calls == [failure_class.FAILURE_INFRASTRUCTURE]
    # The sweep persisted the terminal record; the merge branch did not re-persist.
    assert persist_calls.count(run_id) == 1
