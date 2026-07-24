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
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from osprey.mcp_server.dispatch_worker import sdk_runner


@pytest.fixture(autouse=True)
def _stub_osprey_helpers(monkeypatch):
    """Stub the deferred OSPREY helper imports on their source modules."""
    monkeypatch.setattr(
        "osprey.agent_runner.clean_env.build_clean_env",
        lambda **kw: {},
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt",
        lambda *a, **k: "system",
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.make_tool_allowlist",
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
async def test_tool_result_in_user_message_is_captured(monkeypatch):
    """Tool results arrive as ToolResultBlock inside UserMessage — the runner
    must pair them with the originating ToolUseBlock (permission-denial
    messages surface this way; the parity e2e depends on seeing them)."""

    async def fake_query(prompt, options):
        yield AssistantMessage(
            content=[ToolUseBlock(id="tu1", name="mcp__x__y", input={})], model="m"
        )
        yield UserMessage(
            content=[
                ToolResultBlock(
                    tool_use_id="tu1",
                    content="Tool 'mcp__x__y' is not in this trigger's allowed_tools list",
                )
            ]
        )
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue())

    assert result["tool_calls"] == [
        {
            "name": "mcp__x__y",
            "input": {},
            "result": "Tool 'mcp__x__y' is not in this trigger's allowed_tools list",
        }
    ]


@pytest.mark.asyncio
async def test_tool_policy_wiring(monkeypatch, tmp_path):
    """run_dispatch wires the dispatch tool policy into ClaudeAgentOptions.

    The PreToolUse hook is the single authority (fires even for
    settings-allowed calls); allowed_tools stays trigger-only (no subagent
    union — that would widen the main thread); exact denied tools plus
    server-level rules for prefix entries land in disallowed_tools; the
    context-aware backstop replaces the flat allowlist callback; and
    OSPREY_DISPATCH_RUN=1 marks the session for the approval hook's guard.
    """
    from types import SimpleNamespace

    # Arrange — a project with one declared subagent
    agents_dir = tmp_path / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "channel-finder.md").write_text(
        "---\nname: channel-finder\ntools: mcp__channel-finder__search\n---\n"
    )
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    captured: dict = {}

    async def fake_query(prompt, options):
        captured["options"] = options
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)

    # Act
    await sdk_runner.run_dispatch(
        "do it",
        ["mcp__controls__channel_read"],
        event_queue=asyncio.Queue(),
        denied_tools=["Bash", "WebFetch", "mcp__plugin_playwright_playwright__*"],
    )
    options = captured["options"]

    # Assert — trigger-only allowed_tools, unchanged setting sources
    assert options.allowed_tools == ["mcp__controls__channel_read"]
    assert options.setting_sources == ["project"]
    assert options.env["OSPREY_DISPATCH_RUN"] == "1"

    # Assert — exact denies + server rule for the prefix entry, model-context strip
    assert options.disallowed_tools == [
        "Bash",
        "WebFetch",
        "mcp__plugin_playwright_playwright",
    ]

    # Assert — hook registered as catch-all PreToolUse matcher and enforcing
    matchers = options.hooks["PreToolUse"]
    assert len(matchers) == 1
    assert matchers[0].matcher is None
    (hook,) = matchers[0].hooks
    denied = await hook({"tool_name": "mcp__osprey_workspace__data_list"}, "t", None)
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    allowed = await hook({"tool_name": "mcp__controls__channel_read"}, "t", None)
    assert allowed == {}
    sub_ok = await hook(
        {
            "tool_name": "mcp__channel-finder__search",
            "agent_id": "a1",
            "agent_type": "channel-finder",
        },
        "t",
        None,
    )
    assert sub_ok == {}

    # Assert — backstop is context-aware, not the flat allowlist
    backstop = options.can_use_tool
    main_deny = await backstop("mcp__channel-finder__search", {}, SimpleNamespace(agent_id=None))
    assert type(main_deny).__name__ == "PermissionResultDeny"
    sub_allow = await backstop("mcp__channel-finder__search", {}, SimpleNamespace(agent_id="a1"))
    assert type(sub_allow).__name__ == "PermissionResultAllow"


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
async def test_surface_prompt_forwarded_to_build_system_prompt(monkeypatch):
    """A provided ``surface_prompt`` reaches ``build_system_prompt`` as ``extra``.

    Asserting on the call args to ``build_system_prompt`` (rather than the
    rendered prompt text) sidesteps the ``datetime.now(tz)`` timestamp baked
    into the real implementation.
    """
    calls: list[tuple[tuple, dict]] = []

    def _spy_build_system_prompt(*args, **kwargs):
        calls.append((args, kwargs))
        return "system"

    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt",
        _spy_build_system_prompt,
    )

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    await sdk_runner.run_dispatch(
        "do it", ["Read"], event_queue=asyncio.Queue(), surface_prompt="triggered from Slack"
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs.get("extra") == "triggered from Slack"


@pytest.mark.asyncio
async def test_surface_prompt_omitted_leaves_system_prompt_unchanged(monkeypatch):
    """When ``surface_prompt`` is not passed, ``build_system_prompt`` gets no
    ``extra`` (or ``extra=None``) — identical to pre-Task-2.3 behavior."""
    calls: list[tuple[tuple, dict]] = []

    def _spy_build_system_prompt(*args, **kwargs):
        calls.append((args, kwargs))
        return "system"

    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt",
        _spy_build_system_prompt,
    )

    async def fake_query(prompt, options):
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield _result_message(cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "query", fake_query)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs.get("extra") is None


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
