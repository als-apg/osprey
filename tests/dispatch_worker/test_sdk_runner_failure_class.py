"""Failure-class stamping and error-flip behavior of ``sdk_runner.run_dispatch``.

The runner translates the Claude Agent SDK message stream into a result dict.
These tests exercise the four error exits — a terminal error ``ResultMessage``
(the "completed" branch flipping to status ``error``), the inactivity watchdog,
the generic ``except``, and the SDK-missing guard — asserting each stamps the
right ``failure_class`` and a truthful ``num_tool_calls`` (from the run-stats
map), and that a successful run is left untouched.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from osprey.mcp_server.dispatch_worker import failure_class, run_stats, sdk_runner


@pytest.fixture(autouse=True)
def _isolation():
    """Reset the stats map and detach any counter hook between tests."""
    run_stats._run_stats.clear()
    failure_class.register_counter_hook(None)
    yield
    run_stats._run_stats.clear()
    failure_class.register_counter_hook(None)


@pytest.fixture(autouse=True)
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


def _result_message(
    *,
    is_error: bool = False,
    subtype: str = "success",
    result: str | None = None,
    api_error_status: object | None = None,
    cost_usd: float = 0.1,
    num_turns: int = 1,
) -> ResultMessage:
    """A ResultMessage stand-in with all fields the runner reads set explicitly.

    ``MagicMock(spec=ResultMessage)`` returns truthy mocks for unset attributes,
    so every field the runner inspects must be assigned (``is_error`` in
    particular, else a success case would read as an error).
    """
    rm = MagicMock(spec=ResultMessage)
    rm.is_error = is_error
    rm.subtype = subtype
    rm.result = result
    rm.api_error_status = api_error_status
    rm.cost_usd = cost_usd
    rm.num_turns = num_turns
    return rm


async def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(await queue.get())
    return events


# ---------------------------------------------------------------------------
# Successful run — unchanged behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_is_unchanged(monkeypatch):
    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text="hi")], model="m")
        yield _result_message(is_error=False, subtype="success")

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=queue, run_id="ok")

    assert result["status"] == "completed"
    assert result["error"] is None
    # Success results carry no failure taxonomy fields.
    assert "failure_class" not in result
    types = [e["type"] for e in await _drain(queue)]
    assert "done" in types
    assert "error" not in types


# ---------------------------------------------------------------------------
# Terminal error ResultMessage — completed branch flips to error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_cap_subtype_flips_to_run(monkeypatch):
    async def fake_query(prompt, options):
        yield AssistantMessage(content=[ToolUseBlock(id="t1", name="Read", input={})], model="m")
        yield _result_message(
            is_error=True, subtype="error_max_budget_usd", result="Budget exceeded"
        )

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=queue, run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_RUN
    assert result["error"] == "Budget exceeded"
    assert result["num_tool_calls"] == 1
    # SSE gets an error, not a done, so stream consumers see the failure.
    types = [e["type"] for e in await _drain(queue)]
    assert "error" in types
    assert "done" not in types


@pytest.mark.asyncio
async def test_max_turns_subtype_flips_to_run(monkeypatch):
    async def fake_query(prompt, options):
        yield _result_message(is_error=True, subtype="error_max_turns", result="Max turns")

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_RUN


@pytest.mark.asyncio
async def test_error_result_with_provider_text_is_provider(monkeypatch):
    """A non-budget error whose text reads as a provider fault stays retryable."""

    async def fake_query(prompt, options):
        yield _result_message(
            is_error=True,
            subtype="error_during_execution",
            result="upstream 429 rate limit exceeded",
        )

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_PROVIDER


@pytest.mark.asyncio
async def test_error_result_api_status_folds_into_classification(monkeypatch):
    """api_error_status is appended to the error text and drives classification."""

    async def fake_query(prompt, options):
        yield _result_message(
            is_error=True,
            subtype="error_during_execution",
            result="request failed",
            api_error_status=429,
        )

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["failure_class"] == failure_class.FAILURE_PROVIDER
    assert "429" in result["error"]


@pytest.mark.asyncio
async def test_error_result_generic_is_run(monkeypatch):
    async def fake_query(prompt, options):
        yield _result_message(
            is_error=True, subtype="error_during_execution", result="a tool crashed mid-run"
        )

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["failure_class"] == failure_class.FAILURE_RUN


@pytest.mark.asyncio
async def test_error_result_without_text_gets_synthesized_message(monkeypatch):
    async def fake_query(prompt, options):
        yield _result_message(is_error=True, subtype="error_during_execution", result=None)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert "error_during_execution" in result["error"]


# ---------------------------------------------------------------------------
# Inactivity watchdog — provider fault
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactivity_timeout_is_provider(monkeypatch):
    monkeypatch.setattr(sdk_runner, "_INACTIVITY_TIMEOUT_SEC", 0.05)

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[ToolUseBlock(id="t1", name="Read", input={})], model="m")
        await asyncio.sleep(10)  # provider goes silent -> watchdog trips
        yield _result_message()

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_PROVIDER
    # One tool call was processed before the stall — the count is truthful.
    assert result["num_tool_calls"] == 1


# ---------------------------------------------------------------------------
# Generic exception — classified from the exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_exception_run(monkeypatch):
    async def fake_query(prompt, options):
        yield AssistantMessage(content=[ToolUseBlock(id="t1", name="Read", input={})], model="m")
        raise RuntimeError("something broke")

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_RUN
    assert result["num_tool_calls"] == 1


@pytest.mark.asyncio
async def test_generic_exception_provider_message(monkeypatch):
    async def fake_query(prompt, options):
        raise RuntimeError("401 unauthorized: invalid api key")
        yield  # unreachable — makes this an async generator, like the real query

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["failure_class"] == failure_class.FAILURE_PROVIDER


# ---------------------------------------------------------------------------
# SDK missing — infrastructure fault
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sdk_missing_is_infrastructure(monkeypatch):
    monkeypatch.setattr(sdk_runner, "HAS_SDK", False)

    result = await sdk_runner.run_dispatch("go", ["Read"], run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_INFRASTRUCTURE
    assert result["num_tool_calls"] == 0
    assert result["error"] == "claude_agent_sdk is not installed"


# ---------------------------------------------------------------------------
# Counter hook integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counter_hook_bumped_on_error(monkeypatch):
    seen: list[str] = []
    failure_class.register_counter_hook(seen.append)

    async def fake_query(prompt, options):
        yield _result_message(is_error=True, subtype="error_during_execution", result="crash")

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert seen == [failure_class.FAILURE_RUN]
