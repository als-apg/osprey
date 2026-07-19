"""Unit tests for the dispatch-worker per-run stats map.

Covers the ``run_stats`` module in isolation (increment/get/pop semantics and
defaults), the runner's increment-per-ToolUseBlock wiring, and the dispatch-API
contract that the stats entry is popped in the same ``finally`` that pops
``_tasks`` (no leak).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    ToolUseBlock,
)

from osprey.mcp_server.dispatch_worker import run_stats, sdk_runner


@pytest.fixture(autouse=True)
def _clean_stats():
    """Isolate the module-level stats map between tests."""
    run_stats._run_stats.clear()
    yield
    run_stats._run_stats.clear()


# ---------------------------------------------------------------------------
# run_stats module in isolation
# ---------------------------------------------------------------------------


def test_increment_creates_entry_lazily():
    run_stats.increment_tool_calls("r1")
    assert run_stats.get_run_stats("r1") == {"num_tool_calls": 1}


def test_increment_accumulates():
    for _ in range(3):
        run_stats.increment_tool_calls("r1")
    assert run_stats.get_run_stats("r1")["num_tool_calls"] == 3


def test_runs_are_isolated_by_id():
    run_stats.increment_tool_calls("r1")
    run_stats.increment_tool_calls("r1")
    run_stats.increment_tool_calls("r2")
    assert run_stats.get_run_stats("r1")["num_tool_calls"] == 2
    assert run_stats.get_run_stats("r2")["num_tool_calls"] == 1


def test_get_returns_zeroed_default_when_absent():
    assert run_stats.get_run_stats("missing") == {"num_tool_calls": 0}
    # A default read must not create an entry (no accidental leak/count).
    assert "missing" not in run_stats._run_stats


def test_get_returns_live_entry():
    run_stats.increment_tool_calls("r1")
    live = run_stats.get_run_stats("r1")
    run_stats.increment_tool_calls("r1")
    # The returned dict is the live entry, so a later increment is visible.
    assert live["num_tool_calls"] == 2


def test_pop_removes_and_returns():
    run_stats.increment_tool_calls("r1")
    run_stats.increment_tool_calls("r1")
    popped = run_stats.pop_run_stats("r1")
    assert popped == {"num_tool_calls": 2}
    assert "r1" not in run_stats._run_stats


def test_pop_absent_returns_default():
    assert run_stats.pop_run_stats("missing") == {"num_tool_calls": 0}


def test_pop_is_idempotent():
    run_stats.increment_tool_calls("r1")
    run_stats.pop_run_stats("r1")
    # A second pop (e.g. finally after an already-cleaned entry) must not raise.
    assert run_stats.pop_run_stats("r1") == {"num_tool_calls": 0}


# ---------------------------------------------------------------------------
# sdk_runner increments the map per ToolUseBlock
# ---------------------------------------------------------------------------


@pytest.fixture
def _stub_osprey_helpers(monkeypatch):
    """Stub the deferred OSPREY helpers so run_dispatch runs without a project."""
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.operator_session.build_clean_env",
        lambda **kw: {},
    )
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.sdk_context.build_system_prompt",
        lambda *a, **k: "system",
    )
    monkeypatch.setattr(
        "osprey.utils.config.get_facility_timezone",
        lambda *a, **k: "UTC",
    )


def _result_message(cost_usd: float, num_turns: int) -> ResultMessage:
    rm = MagicMock(spec=ResultMessage)
    rm.cost_usd = cost_usd
    rm.num_turns = num_turns
    return rm


@pytest.mark.asyncio
async def test_run_dispatch_increments_per_tool_use(monkeypatch, _stub_osprey_helpers):
    async def fake_query(prompt, options):
        yield AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name="Read", input={}),
                ToolUseBlock(id="t2", name="Read", input={}),
            ],
            model="m",
        )
        yield AssistantMessage(content=[ToolUseBlock(id="t3", name="Grep", input={})], model="m")
        yield _result_message(cost_usd=0.1, num_turns=2)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch(
        "go", ["Read", "Grep"], event_queue=asyncio.Queue(), run_id="run-xyz"
    )

    assert result["status"] == "completed"
    # Three ToolUseBlocks processed -> truthful count of 3. The runner does not
    # pop the entry (dispatch_api owns cleanup), so it is still readable here.
    assert run_stats.get_run_stats("run-xyz")["num_tool_calls"] == 3


@pytest.mark.asyncio
async def test_run_dispatch_counts_beyond_retained_cap(monkeypatch, _stub_osprey_helpers):
    """num_tool_calls stays truthful past the retained tool_calls cap."""
    monkeypatch.setattr(sdk_runner, "_MAX_TOOL_CALLS", 2)

    async def fake_query(prompt, options):
        for i in range(5):
            yield AssistantMessage(
                content=[ToolUseBlock(id=f"t{i}", name="Read", input={})], model="m"
            )
        yield _result_message(cost_usd=0.1, num_turns=5)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch(
        "go", ["Read"], event_queue=asyncio.Queue(), run_id="run-cap"
    )

    # Only 2 retained in memory, but all 5 counted.
    assert len(result["tool_calls"]) == 2
    assert run_stats.get_run_stats("run-cap")["num_tool_calls"] == 5


@pytest.mark.asyncio
async def test_run_dispatch_without_run_id_creates_no_entry(monkeypatch, _stub_osprey_helpers):
    async def fake_query(prompt, options):
        yield AssistantMessage(content=[ToolUseBlock(id="t1", name="Read", input={})], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue())

    # No run_id -> nothing keyed (guards against a leaked ``None`` entry).
    assert run_stats._run_stats == {}


# ---------------------------------------------------------------------------
# dispatch_api pops the entry in its finally (no leak)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_task_pops_stats_on_completion(monkeypatch):
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda run_id, run: None)
    monkeypatch.setattr(dispatch_api, "describe_run_artifacts", lambda run_id: [])

    async def _fake_run_dispatch(*, run_id, event_queue, **kw):
        run_stats.increment_tool_calls(run_id)
        run_stats.increment_tool_calls(run_id)
        return {
            "status": "completed",
            "text_output": "ok",
            "tool_calls": [],
            "error": None,
            "duration_sec": 0.0,
            "cost_usd": 0.0,
            "num_turns": 1,
        }

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _fake_run_dispatch)

    request = dispatch_api.DispatchRequest(prompt="go", allowed_tools=["Read"])
    await dispatch_api._run_dispatch_task("run-done", request)

    # finally popped both the task handle and the stats entry.
    assert "run-done" not in dispatch_api._tasks
    assert "run-done" not in run_stats._run_stats


@pytest.mark.asyncio
async def test_dispatch_task_pops_stats_on_error(monkeypatch):
    from osprey.mcp_server.dispatch_worker import dispatch_api

    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_persist_run", lambda run_id, run: None)

    async def _boom(*, run_id, event_queue, **kw):
        run_stats.increment_tool_calls(run_id)
        raise RuntimeError("kaboom")

    monkeypatch.setattr(dispatch_api.sdk_runner, "run_dispatch", _boom)

    request = dispatch_api.DispatchRequest(prompt="go", allowed_tools=["Read"])
    await dispatch_api._run_dispatch_task("run-err", request)

    # Even on the error branch the finally must clear the stats entry.
    assert "run-err" not in run_stats._run_stats
