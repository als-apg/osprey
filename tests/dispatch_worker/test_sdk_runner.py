"""Unit tests for the dispatch-worker run loop.

``run_dispatch`` does *deferred* imports of OSPREY helpers and iterates the
agent runner's ``stream_query`` event records, translating them into a result
dict and onto an event queue. These tests:

  * monkeypatch the deferred OSPREY helpers on their *source* modules so the
    in-function imports pick up the stubs,
  * monkeypatch ``stream_query`` on the sdk_runner module with a fake async
    generator that yields event records and captures the keywords it was
    called with.
"""

from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_runner import (
    ApiErrorEvent,
    McpNotReadyError,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
    mcp_snapshot_summary,
)
from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.mcp_server.dispatch_worker import sdk_runner
from osprey_connectors.posture_store import CONTROL_OWNER_ENV_VAR, NO_OWNER


@pytest.fixture(autouse=True)
def _stub_osprey_helpers(monkeypatch, tmp_path):
    """Stub the deferred OSPREY helper imports on their source modules."""
    # The runner creates the agent's Claude state directory under the agent-data
    # root; keep that inside the test's own directory.
    monkeypatch.setattr(
        "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
        lambda: tmp_path / "stub-agent-data",
    )
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


def _text(text: str) -> TextEvent:
    return TextEvent(text=text, parent_tool_use_id=None)


def _tool_use(tool_use_id: str, name: str, tool_input: dict[str, Any]) -> ToolUseEvent:
    return ToolUseEvent(
        tool_use_id=tool_use_id, name=name, input=tool_input, parent_tool_use_id=None
    )


def _tool_result(tool_use_id: str, content: Any) -> ToolResultEvent:
    return ToolResultEvent(
        tool_use_id=tool_use_id, content=content, is_error=False, parent_tool_use_id=None
    )


async def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(await queue.get())
    return events


@pytest.mark.asyncio
async def test_happy_path(monkeypatch):
    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text("hello world")
        yield _result(total_cost_usd=0.5, num_turns=4)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

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
async def test_the_record_carries_the_cost_the_agent_reports(monkeypatch):
    """The cost is read from the field the agent SDK's result message defines."""

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text("ok")
        yield _result(total_cost_usd=0.25, num_turns=2)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)

    assert result["cost_usd"] == 0.25
    events = await _drain(queue)
    assert {"type": "result", "cost_usd": 0.25, "num_turns": 2} in events


@pytest.mark.asyncio
async def test_tool_result_in_user_message_is_captured(monkeypatch):
    """A tool result is paired with the call it answers (permission-denial
    messages surface this way; the parity e2e depends on seeing them)."""

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("tu1", "mcp__x__y", {})
        yield _tool_result("tu1", "Tool 'mcp__x__y' is not in this trigger's allowed_tools list")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=asyncio.Queue())

    assert result["tool_calls"] == [
        {
            "name": "mcp__x__y",
            "input": {},
            "result": "Tool 'mcp__x__y' is not in this trigger's allowed_tools list",
        }
    ]


@pytest.mark.asyncio
async def test_a_tool_result_with_no_content_is_recorded_as_empty(monkeypatch):
    """A call that returned nothing is recorded as returned, with an empty result."""

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("tu1", "mcp__x__y", {})
        yield _tool_result("tu1", None)
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("go", ["Read"], event_queue=queue)

    assert result["tool_calls"][0]["result"] == ""
    events = await _drain(queue)
    assert {"type": "tool_result", "name": "mcp__x__y", "result": ""} in events


@pytest.mark.asyncio
async def test_tool_policy_wiring(monkeypatch, tmp_path):
    """run_dispatch wires the dispatch tool policy into the agent runner call.

    The PreToolUse hook is the single authority (fires even for
    settings-allowed calls); allowed_tools stays trigger-only (no subagent
    union — that would widen the main thread); exact denied tools plus
    server-level rules for prefix entries land in disallowed_tools; the
    context-aware backstop replaces the flat allowlist callback; and
    OSPREY_DISPATCH_RUN=1 marks the session for the approval hook's guard.
    """
    from types import SimpleNamespace

    # Arrange — a repo whose RENDER declares one subagent. ``.claude/`` is build
    # output, so it sits in the render beside the rendered config, never at the
    # repo root.
    agents_dir = tmp_path / "build" / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "channel-finder.md").write_text(
        "---\nname: channel-finder\ntools: mcp__channel-finder__search\n---\n"
    )
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "build" / "config.yml"))
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["kw"] = kw
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    # Act
    await sdk_runner.run_dispatch(
        "do it",
        ["mcp__controls__channel_read"],
        event_queue=asyncio.Queue(),
        denied_tools=["Bash", "WebFetch", "mcp__plugin_playwright_playwright__*"],
    )
    kw = captured["kw"]

    # Assert — trigger-only allowed_tools, unchanged setting sources
    assert kw["allowed_tools"] == ["mcp__controls__channel_read"]
    assert kw["setting_sources"] == ["project"]
    assert kw["env"]["OSPREY_DISPATCH_RUN"] == "1"

    # Assert — exact denies + server rule for the prefix entry, model-context strip
    assert kw["disallowed_tools"] == [
        "Bash",
        "WebFetch",
        "mcp__plugin_playwright_playwright",
    ]

    # Assert — one PreToolUse hook (the runner registers it as the catch-all
    # matcher) and it enforces
    (hook,) = kw["pre_tool_use_hooks"]
    denied = await hook({"tool_name": "mcp__osprey_workspace__artifact_list"}, "t", None)
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
    backstop = kw["can_use_tool"]
    main_deny = await backstop("mcp__channel-finder__search", {}, SimpleNamespace(agent_id=None))
    assert type(main_deny).__name__ == "PermissionResultDeny"
    sub_allow = await backstop("mcp__channel-finder__search", {}, SimpleNamespace(agent_id="a1"))
    assert type(sub_allow).__name__ == "PermissionResultAllow"


@pytest.mark.asyncio
async def test_config_file_env_points_at_the_render(monkeypatch):
    """The dispatched agent's env sets CONFIG_FILE to the RENDER's config.yml.

    The worker process CWD is the image WORKDIR (not the project dir), and
    OSPREY config resolution falls back to CWD/config.yml when CONFIG_FILE is
    unset — so without this, every dispatched run errors with "No config.yml
    found in current directory". With no CONFIG_FILE in the worker's own
    environment the fallback is the repo's render zone, never a flat
    ``<repo>/config.yml`` — a path no three-zone deployment has.
    """
    monkeypatch.setenv("OSPREY_PROJECT_DIR", "/srv/myproj")
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["kw"] = kw
        captured["project_dir"] = project_dir
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert captured["kw"]["env"]["CONFIG_FILE"] == "/srv/myproj/build/config.yml"


@pytest.mark.asyncio
async def test_worker_trusts_the_config_file_its_service_sets(monkeypatch):
    """CONFIG_FILE from the environment is carried through, never re-derived.

    The dispatch-worker compose service sets it to the staged render config it
    bind-mounts; a runner that recomputed the path from OSPREY_PROJECT_DIR would
    silently point the agent at a different file than the worker itself reads.
    """
    monkeypatch.setenv("OSPREY_PROJECT_DIR", "/srv/myproj")
    monkeypatch.setenv("CONFIG_FILE", "/srv/staged/config.yml")
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["kw"] = kw
        captured["project_dir"] = project_dir
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert captured["kw"]["env"]["CONFIG_FILE"] == "/srv/staged/config.yml"
    # ...and the agent runs beside that config, so it discovers the .claude/
    # tree and .mcp.json that were rendered with it.
    assert captured["project_dir"] == Path("/srv/staged")


@pytest.mark.asyncio
async def test_subagent_surfaces_are_discovered_from_the_render(tmp_path, monkeypatch):
    """Declared subagents are read from the render's ``.claude/agents/``.

    SAFETY surface. The allowlist is enforced per-context: a subagent is held to
    its own declared tools, and delegation to an agent the parser never saw is
    denied outright. Reading the repo root instead of the render finds no agent
    files at all — which stays fail-closed (every delegation denied) but costs
    dispatch its subagents silently, so pin the discovery path itself.
    """
    agents_dir = tmp_path / "build" / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "channel-finder.md").write_text(
        "---\nname: channel-finder\ntools: mcp__channel-finder__search\n---\n"
    )
    # A decoy at the repo root: the parser must NOT pick this up.
    root_agents = tmp_path / ".claude" / "agents"
    root_agents.mkdir(parents=True)
    (root_agents / "impostor.md").write_text("---\nname: impostor\ntools: Bash\n---\n")

    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "build" / "config.yml"))
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["kw"] = kw
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    (hook,) = captured["kw"]["pre_tool_use_hooks"]
    delegate = {"tool_name": "Task", "tool_input": {"subagent_type": "channel-finder"}}
    assert await hook(delegate, "t", None) == {}
    # The repo-root file is not the agent surface — delegating to it is denied.
    impostor = {"tool_name": "Task", "tool_input": {"subagent_type": "impostor"}}
    assert (await hook(impostor, "t", None))["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_no_discoverable_agents_denies_every_delegation(tmp_path, monkeypatch):
    """A worker that finds NO agent files stays fail-closed.

    An empty surface map must never mean "anything goes": both delegation and
    any tool call arriving in a subagent context are denied.
    """
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "build" / "config.yml"))
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["kw"] = kw
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    (hook,) = captured["kw"]["pre_tool_use_hooks"]
    delegate = {"tool_name": "Task", "tool_input": {"subagent_type": "channel-finder"}}
    assert (await hook(delegate, "t", None))["hookSpecificOutput"]["permissionDecision"] == "deny"
    in_subagent = {"tool_name": "Read", "agent_id": "a1", "agent_type": "channel-finder"}
    assert (await hook(in_subagent, "t", None))["hookSpecificOutput"][
        "permissionDecision"
    ] == "deny"


@pytest.mark.asyncio
async def test_subagent_delegation_runs_in_the_foreground(monkeypatch):
    """The dispatched agent's env disables background tasks.

    Since Claude Code CLI 2.1.x the Agent tool auto-backgrounds delegated
    subagents: the turn ends immediately and the subagent's results arrive on a
    *later* turn as a task notification. ``_drain_response`` stops at the first
    result, so the worker would answer "the agent is searching, I'll
    notify you when it completes" and never deliver the delegated work.

    ``build_clean_env()`` strips every ``CLAUDE_CODE_*`` key, so this guard
    cannot be inherited — the worker has to set it explicitly. Regression guard
    for that silent under-run.
    """
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["kw"] = kw
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert captured["kw"]["env"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"


@pytest.mark.asyncio
async def test_sdk_missing(monkeypatch):
    monkeypatch.setattr(sdk_runner, "HAS_SDK", False)

    # query must NOT be invoked when the SDK is unavailable.
    def _boom(*a, **k):
        raise AssertionError("query should not be called when HAS_SDK is False")

    monkeypatch.setattr(sdk_runner, "stream_query", _boom)

    result = await sdk_runner.run_dispatch("do it", ["Read"])
    assert result["status"] == "error"
    assert "not installed" in result["error"]


@pytest.mark.asyncio
async def test_cancellation_propagates(monkeypatch):
    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text("partial")
        raise asyncio.CancelledError

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    with pytest.raises(asyncio.CancelledError):
        await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())


@pytest.mark.asyncio
async def test_error_path_does_not_raise(monkeypatch):
    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        raise Exception("boom")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)

    assert result["status"] == "error"
    assert result["error"] == "boom"
    assert result["text_output"] == ""

    events = await _drain(queue)
    assert any(e["type"] == "error" and e["message"] == "boom" for e in events)


_UNTRUSTED_NOTICE = (
    "Ignoring 4 {kind} entries from .claude/settings.json: this workspace has not been "
    "trusted. Run Claude Code interactively here once and accept the trust dialog, or set "
    'projects["/app/build"].hasTrustDialogAccepted: true in '
    "/var/osprey/agent_data/claude-config/.claude.json."
)


async def _stderr_of_failed_run(monkeypatch, *lines: str) -> str | None:
    """Fail a run after the agent CLI wrote ``lines`` to stderr; return the record's stderr."""

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        for line in lines:
            kw["stderr"](line)
        raise Exception("boom")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())
    return result["stderr"]


@pytest.mark.asyncio
async def test_the_untrusted_allow_rules_notice_stays_out_of_the_run_record(monkeypatch):
    """The CLI's notice about the off allow rules never sits above a failure's cause."""
    notice = _UNTRUSTED_NOTICE.format(kind="permissions.allow")

    stderr = await _stderr_of_failed_run(monkeypatch, notice, "real failure detail")

    assert stderr == "real failure detail"


@pytest.mark.asyncio
async def test_another_untrusted_workspace_notice_is_kept(monkeypatch):
    """Only the allow-rules notice is dropped; any other kind is new information."""
    notice = _UNTRUSTED_NOTICE.format(kind="permissions.additionalDirectories")

    stderr = await _stderr_of_failed_run(monkeypatch, notice)

    assert stderr == notice


# ---------------------------------------------------------------------------
# Inactivity watchdog (fast-fail on a silently hung provider, e.g. bad cred)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inactivity_timeout_aborts_with_clear_error(monkeypatch):
    """A provider that never responds (bad/expired credential, unreachable base
    URL) is aborted at the inactivity window with a clear message, instead of
    stalling silently to the outer dispatch timeout."""
    monkeypatch.setattr(sdk_runner, "_INACTIVITY_TIMEOUT_SEC", 0.2, raising=False)

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        await asyncio.sleep(30)  # hang — never yields a message
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

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

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text("working...")
        await asyncio.sleep(30)  # then hang
        yield  # pragma: no cover - never reached

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
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

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text(huge)
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert len(result["text_output"]) <= sdk_runner._MAX_TEXT_OUTPUT + 100
    assert "[truncated" in result["text_output"]


@pytest.mark.asyncio
async def test_secret_scrubbed_from_text_output(monkeypatch):
    secret = "tok-abcdef-1234567890"  # len >= 12
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", secret)

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text(f"leaked {secret} here")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert secret not in result["text_output"]
    assert "***" in result["text_output"]


@pytest.mark.asyncio
async def test_tool_use_and_result_are_captured(monkeypatch):
    """A tool call and its matching result land in tool_calls with the result."""

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("tu1", "Read", {"path": "f"})
        yield _tool_result("tu1", "file contents")
        yield _result(total_cost_usd=0.2, num_turns=2)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
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

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
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

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs.get("extra") is None


@pytest.mark.asyncio
async def test_oversized_tool_result_is_truncated(monkeypatch):
    huge = "z" * (sdk_runner._MAX_TOOL_RESULT + 5000)

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_use("tu1", "Read", {})
        yield _tool_result("tu1", huge)
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    body = result["tool_calls"][0]["result"]
    assert len(body) <= sdk_runner._MAX_TOOL_RESULT + 100
    assert "[truncated" in body


# ---------------------------------------------------------------------------
# MCP readiness: a required server that is not up is an infrastructure error,
# and every run record carries the readiness snapshot.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_not_ready_is_an_infrastructure_error(monkeypatch):
    """The worker's own machinery (the CLI's MCP servers) was not ready before
    the run: stamped infrastructure (retryable once the host catches up), never
    a "completed" run that quietly lacked the tool it was dispatched to use."""
    raw = [
        {"name": "controls", "status": "connected", "tools": [{"name": "t"}] * 6},
        {"name": "osprey_workspace", "status": "pending", "tools": []},
    ]

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        kw["on_mcp_status"](raw)
        raise McpNotReadyError(
            "MCP server(s) this run requires were not connected: osprey_workspace (pending)",
            servers=raw,
            missing=["osprey_workspace"],
        )
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)

    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch(
        "do it", ["Read", "mcp__osprey_workspace__artifact_register"], event_queue=queue
    )

    assert result["status"] == "error"
    assert result["failure_class"] == "infrastructure"
    assert "osprey_workspace" in result["error"]
    assert result["mcp_servers"] == mcp_snapshot_summary(raw)
    events = await _drain(queue)
    assert any(e["type"] == "error" and "osprey_workspace" in e["message"] for e in events)


@pytest.mark.asyncio
async def test_run_record_carries_the_mcp_snapshot(monkeypatch):
    """Persisted with the run so a missing tool can be read off the record as
    INFRA (server not connected) or MODEL (tool registered, agent ignored it)."""
    raw = [{"name": "controls", "status": "connected", "tools": [{"name": "t"}] * 6}]

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        kw["on_mcp_status"](raw)
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())

    assert result["status"] == "completed"
    assert result["mcp_servers"] == mcp_snapshot_summary(raw)


@pytest.mark.asyncio
async def test_required_servers_are_the_allow_listed_ones(monkeypatch):
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured.update(kw)
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch(
        "do it",
        ["Glob", "mcp__osprey_workspace__artifact_register", "mcp__controls__channel_read"],
        event_queue=asyncio.Queue(),
    )

    assert captured["require_mcp_servers"] == {"osprey_workspace", "controls"}
    # The declared set is the one awaited: the runner's default.
    assert "await_mcp_servers" not in captured


@pytest.mark.asyncio
async def test_cli_mcp_startup_limit_matches_the_barrier(monkeypatch):
    """The CLI marks a stdio server failed after its own ``MCP_TIMEOUT`` (30s by
    default) — shorter than the barrier, which would then wait for a server the
    CLI has already given up on. One figure drives both; an operator's explicit
    ``MCP_TIMEOUT`` wins."""
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["env"] = kw["env"]
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())
    assert captured["env"]["MCP_TIMEOUT"] == str(int(sdk_runner.MCP_READY_TIMEOUT_S * 1000))

    monkeypatch.setattr(
        "osprey.agent_runner.clean_env.build_clean_env", lambda **kw: {"MCP_TIMEOUT": "5000"}
    )
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue())
    assert captured["env"]["MCP_TIMEOUT"] == "5000"


# ---------------------------------------------------------------------------
# Agent-data root and owner export
# ---------------------------------------------------------------------------


async def _env_of_run(monkeypatch, **kwargs) -> dict[str, str]:
    """Run a dispatch through a stub stream and return the agent's environment."""
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured["env"] = kw["env"]
        yield _text("ok")
        yield _result(total_cost_usd=0.1, num_turns=1)

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], event_queue=asyncio.Queue(), **kwargs)
    return captured["env"]


@pytest.mark.asyncio
async def test_the_agent_data_root_is_stamped_for_the_agent(monkeypatch, tmp_path):
    """The dispatched agent is handed the directory its control state lives in.

    Everything below the spawn otherwise re-derives that directory for itself —
    the controls server from config, the hooks from a repo root they resolve
    with the standard library — and a worker reads a config staged on another
    machine, which is one derivation too many for readers that must agree on a
    single file. The stamp is the answer all of them prefer.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setattr(
        "osprey_connectors.workspace.resolve_shared_data_root",
        lambda: root,
    )

    env = await _env_of_run(monkeypatch)

    assert env[OSPREY_AGENT_DATA_ROOT] == str(root)


@pytest.mark.asyncio
async def test_the_agent_config_dir_is_on_the_agent_data_volume(monkeypatch, tmp_path):
    """The dispatched agent's transcripts outlive a recreate of the worker.

    The container's own layer is discarded at every recreate; the agent-data
    root is the worker's volume, so the Claude state directory goes there.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setattr(
        "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
        lambda: root,
    )

    env = await _env_of_run(monkeypatch)

    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "var/agent_data/claude-config")
    assert (root / "claude-config").is_dir()


@pytest.mark.asyncio
async def test_the_run_marks_nothing_trusted_in_its_config_dir(monkeypatch, tmp_path):
    """The runner writes no Claude state into the agent's config dir.

    Unattended dispatch runs keep the project's allow rules off on purpose:
    project settings must not widen what a trigger may do, so nothing marks
    the render trusted where the run's CLI looks for it.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setattr(
        "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
        lambda: root,
    )

    await _env_of_run(monkeypatch)

    assert not (root / "claude-config" / ".claude.json").exists()


@pytest.mark.asyncio
async def test_an_unresolvable_agent_data_root_falls_back_to_home(monkeypatch, tmp_path):
    """A root that cannot be resolved costs the durable location, never the run."""

    def _raise():
        raise RuntimeError("no config here")

    monkeypatch.setattr("osprey.agent_runner.artifact_resolve.deployed_agent_data_root", _raise)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    env = await _env_of_run(monkeypatch)

    assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "home" / ".claude")


@pytest.mark.asyncio
async def test_an_unresolvable_root_leaves_the_variable_absent(monkeypatch):
    """A config that cannot be read costs the stamp, never the run.

    Absent, the readers fall back to deriving the directory themselves, which is
    what they do for every process nobody stamped. An empty value would be a
    path of no name for them to resolve against.
    """

    def _raise():
        raise RuntimeError("no config here")

    monkeypatch.setattr("osprey_connectors.workspace.resolve_shared_data_root", _raise)

    env = await _env_of_run(monkeypatch)

    assert OSPREY_AGENT_DATA_ROOT not in env


@pytest.mark.asyncio
async def test_owner_is_exported_to_the_agent_environment(monkeypatch):
    """A dispatch that carried an owner stamps OSPREY_CONTROL_OWNER for the run.

    The stamp is rung 3 of the connector's owner ladder, so every control-system
    write the agent makes is judged against that person's narrowing instead of
    the deployment ceiling.
    """
    env = await _env_of_run(monkeypatch, owner="alice")

    assert env[CONTROL_OWNER_ENV_VAR] == "alice"


@pytest.mark.asyncio
async def test_ownerless_dispatch_leaves_the_variable_unset(monkeypatch):
    """No owner means the key is absent, not empty.

    An empty stamp is a value the ladder would have to interpret; an absent one
    falls through to the rung below it, which is what an owner-less run (a cron
    fire) is entitled to.
    """
    env = await _env_of_run(monkeypatch)

    assert CONTROL_OWNER_ENV_VAR not in env


@pytest.mark.asyncio
async def test_no_owner_sentinel_is_never_stamped(monkeypatch):
    """NO_OWNER is refused by identity, so its printable form never reaches the env.

    The sentinel exists to be printed in a warning line, and it prints as
    ``<no owner>``; a truthiness guard would stamp the run with an account of
    that name and look up a narrowing for it.
    """
    env = await _env_of_run(monkeypatch, owner=NO_OWNER)

    assert CONTROL_OWNER_ENV_VAR not in env
    assert str(NO_OWNER) not in env.values()


# ---------------------------------------------------------------------------
# Earlier answers the run may read back
# ---------------------------------------------------------------------------

_PRIOR_RUN = "3f2b6c1e-8a4d-4f0e-9b1a-2c3d4e5f6a7b"


@pytest.mark.asyncio
async def test_prior_answer_runs_are_exported_to_the_agent_environment(monkeypatch):
    env = await _env_of_run(monkeypatch, prior_answer_runs=[_PRIOR_RUN, "../x"])

    assert env["OSPREY_DISPATCH_PRIOR_ANSWER_RUNS"] == _PRIOR_RUN


@pytest.mark.asyncio
async def test_no_prior_answer_runs_leaves_the_variable_absent(monkeypatch):
    env = await _env_of_run(monkeypatch)

    assert "OSPREY_DISPATCH_PRIOR_ANSWER_RUNS" not in env


@pytest.mark.asyncio
async def test_a_stray_worker_value_never_reaches_a_run_that_names_no_prior_answers(monkeypatch):
    monkeypatch.setenv("OSPREY_DISPATCH_PRIOR_ANSWER_RUNS", _PRIOR_RUN)

    env = await _env_of_run(monkeypatch)

    assert "OSPREY_DISPATCH_PRIOR_ANSWER_RUNS" not in env


# ---------------------------------------------------------------------------
# The agent runner call and the records outside the run record
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_run_goes_to_the_agent_runner_with_no_permission_mode_and_no_budget(
    monkeypatch,
):
    """The runner's defaults are a bypass permission mode and a spend ceiling;
    dispatch overrides both, so the backstop stays consulted and no run stops
    on spend."""
    captured: dict = {}

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        captured.update(kw)
        yield _text("ok")
        yield _result()

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    await sdk_runner.run_dispatch("do it", ["Read"], max_turns=7, event_queue=asyncio.Queue())

    assert captured["permission_mode"] is None
    assert captured["max_budget_usd"] is None
    assert captured["max_turns"] == 7
    assert captured["setting_sources"] == ["project"]
    for key in ("model", "provider", "mcp_servers", "resume"):
        assert key not in captured
    assert captured["session_id"] == captured["env"]["OSPREY_TELEMETRY_SESSION_ID"]


@pytest.mark.asyncio
async def test_events_outside_the_record_leave_it_unchanged(monkeypatch):
    """Thinking, API-error and system records add nothing to the record or the
    live events."""

    async def run_with(events: list) -> tuple[dict, list[dict]]:
        async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
            for event in events:
                yield event

        monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
        queue: asyncio.Queue = asyncio.Queue()
        result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)
        result.pop("duration_sec")
        result.pop("session_id")
        return result, await _drain(queue)

    alone = await run_with([_text("hello"), _result()])
    surrounded = await run_with(
        [
            SystemEvent(subtype="init", data={"subtype": "init"}),
            ThinkingEvent(text="hm"),
            _text("hello"),
            ApiErrorEvent(error="rate_limit"),
            _result(),
        ]
    )

    assert surrounded == alone


@pytest.mark.asyncio
async def test_a_result_for_an_unknown_call_streams_without_a_name(monkeypatch):
    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        yield _tool_result("never-issued", "orphan")
        yield _result()

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    queue: asyncio.Queue = asyncio.Queue()
    result = await sdk_runner.run_dispatch("do it", ["Read"], event_queue=queue)

    assert result["tool_calls"] == []
    assert {"type": "tool_result", "name": None, "result": None} in await _drain(queue)


def test_the_worker_module_imports_without_the_agent_sdk():
    """The worker module loads, and reports the SDK absent, when the agent SDK
    cannot be imported; its source names no agent SDK import."""
    code = (
        "import sys\n"
        "sys.modules['claude_agent_sdk'] = None\n"
        "from osprey.mcp_server.dispatch_worker import sdk_runner\n"
        "print(sdk_runner.HAS_SDK)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "False"

    tree = ast.parse(Path(sdk_runner.__file__).read_text())
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not [name for name in imported if name.split(".")[0] == "claude_agent_sdk"]
