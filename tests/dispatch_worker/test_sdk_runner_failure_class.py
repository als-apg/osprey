"""Failure-class stamping and error-flip behavior of ``sdk_runner.run_dispatch``.

The runner translates the agent runner's event records into a result dict.
These tests exercise the four error exits — a terminal error result record
(the "completed" branch flipping to status ``error``), the inactivity watchdog,
the generic ``except``, and the SDK-missing guard — asserting each stamps the
right ``failure_class`` and a truthful ``num_tool_calls`` (from the run-stats
map), and that a successful run is left untouched.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from osprey.agent_runner import AgentRunError, ResultEvent, TextEvent, ToolUseEvent
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
        "osprey.agent_runner.clean_env.build_clean_env",
        lambda **kw: {},
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt",
        lambda *a, **k: "system",
    )
    monkeypatch.setattr(
        "osprey.utils.config.get_facility_timezone",
        lambda *a, **k: "UTC",
    )


def _result(**overrides: Any) -> ResultEvent:
    """A successful result record; any field can be overridden."""
    fields: dict[str, Any] = {
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "duration_ms": 0,
        "session_id": "s",
        "total_cost_usd": 0.1,
        "usage": None,
        "result": None,
        "api_error_status": None,
    }
    fields.update(overrides)
    return ResultEvent(**fields)


def _tool_use(tool_use_id: str) -> ToolUseEvent:
    return ToolUseEvent(tool_use_id=tool_use_id, name="Read", input={}, parent_tool_use_id=None)


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
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield TextEvent(text="hi", parent_tool_use_id=None)
        yield _result(is_error=False, subtype="success")

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

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
# Terminal error result record — completed branch flips to error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_cap_subtype_flips_to_run(monkeypatch):
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("t1")
        yield _result(is_error=True, subtype="error_max_budget_usd", result="Budget exceeded")

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

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
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _result(is_error=True, subtype="error_max_turns", result="Max turns")

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_RUN


@pytest.mark.asyncio
async def test_error_result_with_provider_text_is_provider(monkeypatch):
    """A non-budget error whose text reads as a provider fault stays retryable."""

    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _result(
            is_error=True,
            subtype="error_during_execution",
            result="upstream 429 rate limit exceeded",
        )

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_PROVIDER


@pytest.mark.asyncio
async def test_error_result_api_status_folds_into_classification(monkeypatch):
    """api_error_status is appended to the error text and drives classification."""

    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _result(
            is_error=True,
            subtype="error_during_execution",
            result="request failed",
            api_error_status=429,
        )

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["failure_class"] == failure_class.FAILURE_PROVIDER
    assert "429" in result["error"]


@pytest.mark.asyncio
async def test_error_result_generic_is_run(monkeypatch):
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _result(
            is_error=True, subtype="error_during_execution", result="a tool crashed mid-run"
        )

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["failure_class"] == failure_class.FAILURE_RUN


@pytest.mark.asyncio
async def test_error_result_without_text_gets_synthesized_message(monkeypatch):
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _result(is_error=True, subtype="error_during_execution", result=None)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert "error_during_execution" in result["error"]


# ---------------------------------------------------------------------------
# Inactivity watchdog — provider fault
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactivity_timeout_is_provider(monkeypatch):
    monkeypatch.setattr(sdk_runner, "_INACTIVITY_TIMEOUT_SEC", 0.05)

    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("t1")
        await asyncio.sleep(10)  # provider goes silent -> watchdog trips
        yield _result()

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

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
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("t1")
        raise RuntimeError("something broke")

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["status"] == "error"
    assert result["failure_class"] == failure_class.FAILURE_RUN
    assert result["num_tool_calls"] == 1


@pytest.mark.asyncio
async def test_generic_exception_provider_message(monkeypatch):
    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        raise RuntimeError("401 unauthorized: invalid api key")
        yield  # unreachable — makes this an async generator, like the real stream

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

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

    async def fake_stream(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        yield _result(is_error=True, subtype="error_during_execution", result="crash")

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert seen == [failure_class.FAILURE_RUN]


@pytest.mark.asyncio
async def test_an_agent_run_error_is_classified_like_the_error_it_wraps(monkeypatch):
    """The runner wraps an agent SDK error; its text and cause chain classify it."""

    async def provider_failure(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        raise AgentRunError("401 unauthorized", error_type="ProcessError") from RuntimeError(
            "401 unauthorized"
        )
        yield  # unreachable — makes this an async generator, like the real stream

    monkeypatch.setattr(sdk_runner, "stream_query", provider_failure)
    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r")

    assert result["failure_class"] == failure_class.FAILURE_PROVIDER
    assert result["error"] == "401 unauthorized"

    async def run_failure(project_dir, prompt, **_kw):  # noqa: ARG001 - matches the stream_query signature
        raise AgentRunError("exit 1", error_type="ProcessError")
        yield  # unreachable — makes this an async generator, like the real stream

    monkeypatch.setattr(sdk_runner, "stream_query", run_failure)
    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue(), run_id="r2")

    assert result["failure_class"] == failure_class.FAILURE_RUN
