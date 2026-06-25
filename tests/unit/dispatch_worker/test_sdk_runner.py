"""Unit tests for the dispatch-worker SDK runner.

``run_dispatch`` does *deferred* imports of OSPREY helpers and iterates the
Claude Agent SDK ``query()`` async generator, translating SDK messages into a
result dict and onto an event queue. These tests:

  * monkeypatch the deferred OSPREY helpers on their *source* modules so the
    in-function imports pick up the stubs,
  * monkeypatch ``query`` on the sdk_runner module with a fake async generator,
  * yield real ``AssistantMessage``/``TextBlock`` instances (so the runner's
    ``isinstance`` checks hold) and a ``MagicMock(spec=ResultMessage)`` for the
    result message — ``MagicMock(spec=X)`` passes ``isinstance(..., X)`` and lets
    us set the ``cost_usd``/``num_turns`` attributes the runner reads via getattr.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from osprey.mcp_server.dispatch_worker import sdk_runner


@pytest.fixture(autouse=True)
def _stub_osprey_helpers(monkeypatch):
    """Stub the deferred OSPREY helper imports on their source modules."""
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.operator_session.build_clean_env",
        lambda **kw: {},
    )
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.sdk_context.build_system_prompt",
        lambda *a, **k: "system",
    )
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.sdk_context.make_tool_allowlist",
        lambda tools, denied=(): lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "osprey.utils.config.get_facility_timezone",
        lambda *a, **k: "UTC",
    )


def _result_message(cost_usd: float, num_turns: int) -> ResultMessage:
    """A ResultMessage stand-in that passes isinstance and exposes cost_usd."""
    rm = MagicMock(spec=ResultMessage)
    rm.cost_usd = cost_usd
    rm.num_turns = num_turns
    return rm


async def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(await queue.get())
    return events


@pytest.mark.asyncio
async def test_happy_path(monkeypatch):
    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text="hello world")], model="m")
        yield _result_message(cost_usd=0.5, num_turns=4)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)

    assert result["status"] == "completed"
    assert result["text_output"] == "hello world"
    assert result["error"] is None
    assert result["cost_usd"] == 0.5
    assert result["num_turns"] == 4

    events = await _drain(queue)
    types = [e["type"] for e in events]
    assert "text" in types
    assert "done" in types
    text_event = next(e for e in events if e["type"] == "text")
    assert text_event["content"] == "hello world"


@pytest.mark.asyncio
async def test_config_file_env_points_at_project(monkeypatch):
    """The dispatched agent's env sets CONFIG_FILE to the project's config.yml.

    The worker process CWD is the image WORKDIR (not the project dir), and
    OSPREY config resolution falls back to CWD/config.yml when CONFIG_FILE is
    unset — so without this, every dispatched run errors with "No config.yml
    found in current directory". Regression guard for that container bug.
    """
    monkeypatch.setenv("OSPREY_PROJECT_DIR", "/srv/myproj")
    captured: dict = {}

    async def fake_query(prompt, options):
        captured["options"] = options
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert captured["options"].env["CONFIG_FILE"] == "/srv/myproj/config.yml"


@pytest.mark.asyncio
async def test_sdk_missing(monkeypatch):
    monkeypatch.setattr(sdk_runner, "HAS_SDK", False)

    # query must NOT be invoked when the SDK is unavailable.
    def _boom(*a, **k):
        raise AssertionError("query should not be called when HAS_SDK is False")

    monkeypatch.setattr(sdk_runner, "query", _boom)

    result = await sdk_runner.run_dispatch("do it", ["Read"])
    assert result["status"] == "error"
    assert "not installed" in result["error"]


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text="partial")], model="m")
        raise asyncio.CancelledError

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    with pytest.raises(asyncio.CancelledError):
        await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())


@pytest.mark.asyncio
async def test_error_path_does_not_raise(monkeypatch):
    async def fake_query(prompt, options):
        raise Exception("boom")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)

    assert result["status"] == "error"
    assert result["error"] == "boom"
    assert result["text_output"] == ""

    events = await _drain(queue)
    assert any(e["type"] == "error" and e["message"] == "boom" for e in events)


# ---------------------------------------------------------------------------
# Inactivity watchdog (fast-fail on a silently hung provider, e.g. bad cred)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactivity_timeout_aborts_with_clear_error(monkeypatch):
    """A provider that never responds (bad/expired credential, unreachable base
    URL) is aborted at the inactivity window with a clear message, instead of
    stalling silently to the outer dispatch timeout."""
    monkeypatch.setattr(sdk_runner, "_INACTIVITY_TIMEOUT_SEC", 0.2, raising=False)

    async def fake_query(prompt, options):
        await asyncio.sleep(30)  # hang — never yields a message
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    queue: asyncio.Queue = asyncio.Queue()
    # Outer guard so a regression (no watchdog) fails fast instead of hanging
    # the whole suite for 30s.
    result = await asyncio.wait_for(
        sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue),
        timeout=5,
    )

    assert result["status"] == "error"
    assert "No response from the model provider" in result["error"]
    assert "credential" in result["error"].lower()
    # Aborted at the inactivity window, not the full 30s hang.
    assert result["duration_sec"] < 5

    events = await _drain(queue)
    assert any(e["type"] == "error" for e in events)


@pytest.mark.asyncio
async def test_inactivity_timeout_after_partial_progress(monkeypatch):
    """The watchdog resets on each streamed message: a run that emits output and
    then stalls is still aborted, with the partial text preserved."""
    monkeypatch.setattr(sdk_runner, "_INACTIVITY_TIMEOUT_SEC", 0.2, raising=False)

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text="working...")], model="m")
        await asyncio.sleep(30)  # then hang
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    result = await asyncio.wait_for(
        sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue()),
        timeout=5,
    )

    assert result["status"] == "error"
    assert result["text_output"] == "working..."  # partial output retained
    assert "No response from the model provider" in result["error"]


# ---------------------------------------------------------------------------
# Per-run memory caps + secret scrubbing (lifecycle robustness)
# ---------------------------------------------------------------------------


def test_cap_text_truncates_with_marker():
    big = "x" * (sdk_runner._MAX_TEXT_OUTPUT + 5000)
    capped = sdk_runner._cap_text(big)
    assert len(capped) < len(big)
    assert "[truncated" in capped


def test_scrub_replaces_secret_values():
    secrets = ["supersecret-token-123456"]
    out = sdk_runner._scrub("auth=supersecret-token-123456 done", secrets)
    assert "supersecret-token-123456" not in out
    assert "***" in out


@pytest.mark.asyncio
async def test_oversized_text_output_is_truncated(monkeypatch):
    huge = "y" * (sdk_runner._MAX_TEXT_OUTPUT + 10000)

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text=huge)], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert len(result["text_output"]) <= sdk_runner._MAX_TEXT_OUTPUT + 100
    assert "[truncated" in result["text_output"]


@pytest.mark.asyncio
async def test_secret_scrubbed_from_text_output(monkeypatch):
    secret = "tok-abcdef-1234567890"  # len >= 12
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", secret)

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text=f"leaked {secret} here")], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert secret not in result["text_output"]
    assert "***" in result["text_output"]


@pytest.mark.asyncio
async def test_tool_use_and_result_are_captured(monkeypatch):
    """A ToolUseBlock + matching ToolResultBlock land in tool_calls with the result."""
    from claude_agent_sdk import ToolResultBlock, ToolUseBlock

    async def fake_query(prompt, options):
        yield AssistantMessage(
            content=[ToolUseBlock(id="tu1", name="Read", input={"path": "f"})], model="m"
        )
        yield AssistantMessage(
            content=[ToolResultBlock(tool_use_id="tu1", content="file contents")], model="m"
        )
        yield _result_message(cost_usd=0.2, num_turns=2)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)

    assert result["status"] == "completed"
    assert len(result["tool_calls"]) == 1
    call = result["tool_calls"][0]
    assert call["name"] == "Read"
    assert call["input"] == {"path": "f"}
    assert call["result"] == "file contents"

    events = await _drain(queue)
    types = [e["type"] for e in events]
    assert "tool_start" in types
    assert "tool_result" in types


@pytest.mark.asyncio
async def test_oversized_tool_result_is_truncated(monkeypatch):
    from claude_agent_sdk import ToolResultBlock, ToolUseBlock

    huge = "z" * (sdk_runner._MAX_TOOL_RESULT + 5000)

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[ToolUseBlock(id="tu1", name="Read", input={})], model="m")
        yield AssistantMessage(
            content=[ToolResultBlock(tool_use_id="tu1", content=huge)], model="m"
        )
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    body = result["tool_calls"][0]["result"]
    assert len(body) <= sdk_runner._MAX_TOOL_RESULT + 100
    assert "[truncated" in body
