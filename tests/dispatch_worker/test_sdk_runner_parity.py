"""A fixed trigger gets the agent options and the run record dispatch always produced.

``run_dispatch`` goes through the real ``stream_query`` → ``build_agent_options``
→ event-record path here. The only double is a recording client in place of the
agent SDK's client: it keeps the options it was built with, records what its
``query`` received, answers the MCP status poll and replays scripted agent SDK
messages. Every expected literal below is what dispatch built and recorded for
this trigger when it assembled the options and parsed the messages itself.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    HookMatcher,
    ProcessError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from osprey.mcp_server.dispatch_worker import dispatch_api, run_stats, sdk_runner

_TOOL = "mcp__controls__channel_read"

_ENV_KEYS = [
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS",
    "CLAUDE_CONFIG_DIR",
    "CONFIG_FILE",
    "MCP_TIMEOUT",
    "OSPREY_AGENT_DATA_ROOT",
    "OSPREY_CONTROL_OWNER",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_DISPATCH_RUN_ID",
    "OSPREY_TELEMETRY_SESSION_ID",
    "OSPREY_TELEMETRY_SESSION_START",
]


@pytest.fixture(autouse=True)
def render_dir(monkeypatch, tmp_path) -> Path:
    """A render with one subagent and one declared MCP server; OSPREY helpers stubbed."""
    monkeypatch.setattr(
        "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
        lambda: tmp_path / "stub-agent-data",
    )
    monkeypatch.setattr("osprey.agent_runner.clean_env.build_clean_env", lambda **kw: {})
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt", lambda *a, **k: "system"
    )
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.make_tool_allowlist",
        lambda tools, denied=(): lambda *a, **k: None,
    )
    monkeypatch.setattr("osprey.utils.config.get_facility_timezone", lambda *a, **k: "UTC")
    monkeypatch.setattr(
        "osprey_connectors.workspace.resolve_shared_data_root",
        lambda: tmp_path / "agent-data",
    )
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {})
    run_stats._run_stats.clear()

    render = tmp_path / "build"
    agents = render / ".claude" / "agents"
    agents.mkdir(parents=True)
    (agents / "channel-finder.md").write_text(
        "---\nname: channel-finder\ntools: mcp__channel-finder__search\n---\n"
    )
    (render / ".mcp.json").write_text('{"mcpServers": {"controls": {"command": "controls"}}}')
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIG_FILE", str(render / "config.yml"))
    yield render
    run_stats._run_stats.clear()


class _RecordingClient:
    """Stands in for the agent SDK client: records its options and its query,
    reports ``controls`` connected, and replays a scripted response."""

    def __init__(self, options: ClaudeAgentOptions, script: Any) -> None:
        self.options = options
        self.queries: list[Any] = []
        self.exited = False
        self._script = script

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self.exited = True
        return False

    async def get_mcp_status(self) -> dict:
        return {
            "mcpServers": [
                {"name": "controls", "status": "connected", "tools": [{"name": "channel_read"}]}
            ]
        }

    async def query(self, prompt: Any) -> None:
        if isinstance(prompt, str):
            self.queries.append(prompt)
        else:
            self.queries.append([message async for message in prompt])

    async def receive_response(self):
        async for message in self._script(self):
            yield message


def _install(monkeypatch, script) -> list[_RecordingClient]:
    clients: list[_RecordingClient] = []

    def _make(options: ClaudeAgentOptions) -> _RecordingClient:
        client = _RecordingClient(options, script)
        clients.append(client)
        return client

    monkeypatch.setattr("osprey.agent_runner.runner.ClaudeSDKClient", _make)
    return clients


def _result_message() -> ResultMessage:
    return ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=2,
        session_id="s",
        total_cost_usd=0.25,
    )


async def _fixed_stream(_client: _RecordingClient):
    yield SystemMessage(subtype="init", data={"subtype": "init"})
    yield AssistantMessage(
        content=[
            TextBlock(text="Reading."),
            ToolUseBlock(id="tu1", name=_TOOL, input={"channels": ["X"]}),
        ],
        model="m",
    )
    yield UserMessage(
        content=[ToolResultBlock(tool_use_id="tu1", content=[{"type": "text", "text": "X=1"}])]
    )
    yield AssistantMessage(
        content=[ThinkingBlock(thinking="hm", signature="s"), TextBlock(text="Done.")],
        model="m",
    )
    yield _result_message()


async def _run(queue: asyncio.Queue | None = None) -> dict:
    return await sdk_runner.run_dispatch(
        "go",
        [_TOOL],
        max_turns=7,
        event_queue=queue,
        denied_tools=["Bash", "mcp__plugin_x__*"],
        run_id="run-p",
        owner="alice",
    )


async def _drain(queue: asyncio.Queue) -> list[dict]:
    events = []
    while not queue.empty():
        events.append(await queue.get())
    return events


@pytest.mark.asyncio
async def test_the_agent_options_are_the_ones_dispatch_always_built(monkeypatch, render_dir):
    clients = _install(monkeypatch, _fixed_stream)
    await _run()
    (client,) = clients
    options = client.options
    sid = options.env["OSPREY_TELEMETRY_SESSION_ID"]

    # Every field outside the callables and the env, at its dispatch value or
    # the agent SDK's default: no permission mode, model or budget is set.
    assert dataclasses.replace(
        options, can_use_tool=None, hooks=None, stderr=None, env={}
    ) == ClaudeAgentOptions(
        allowed_tools=[_TOOL],
        system_prompt="system",
        disallowed_tools=["Bash", "mcp__plugin_x"],
        cwd=str(render_dir),
        max_turns=7,
        setting_sources=["project"],
        session_id=sid,
        mcp_servers=str(render_dir / ".mcp.json"),
        strict_mcp_config=True,
    )
    assert sorted(options.env) == _ENV_KEYS

    assert options.hooks is not None
    assert list(options.hooks) == ["PreToolUse"]
    (matcher,) = options.hooks["PreToolUse"]
    (hook,) = matcher.hooks
    assert matcher == HookMatcher(matcher=None, hooks=[hook])
    denied = await hook({"tool_name": "mcp__osprey_workspace__artifact_list"}, "t", None)
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert await hook({"tool_name": _TOOL}, "t", None) == {}

    backstop = options.can_use_tool
    assert backstop is not None
    main = await backstop("mcp__channel-finder__search", {}, SimpleNamespace(agent_id=None))
    assert type(main).__name__ == "PermissionResultDeny"
    sub = await backstop("mcp__channel-finder__search", {}, SimpleNamespace(agent_id="a1"))
    assert type(sub).__name__ == "PermissionResultAllow"

    assert callable(options.stderr)


@pytest.mark.asyncio
async def test_the_recorded_run_and_its_live_events_are_unchanged_for_a_fixed_stream(monkeypatch):
    clients = _install(monkeypatch, _fixed_stream)
    queue: asyncio.Queue = asyncio.Queue()

    result = await _run(queue)

    (client,) = clients
    sid = client.options.env["OSPREY_TELEMETRY_SESSION_ID"]
    duration = result.pop("duration_sec")
    assert isinstance(duration, float) and duration >= 0
    assert result == {
        "status": "completed",
        "text_output": "Reading.Done.",
        "tool_calls": [{"name": _TOOL, "input": {"channels": ["X"]}, "result": "X=1"}],
        "error": None,
        "cost_usd": 0.25,
        "num_turns": 2,
        "session_id": sid,
        "mcp_servers": [{"name": "controls", "status": "connected", "tools": 1, "error": None}],
    }
    assert await _drain(queue) == [
        {"type": "text", "content": "Reading."},
        {"type": "tool_start", "name": _TOOL, "input": {"channels": ["X"]}},
        {"type": "tool_result", "name": _TOOL, "result": "X=1"},
        {"type": "text", "content": "Done."},
        {"type": "result", "cost_usd": 0.25, "num_turns": 2},
        {"type": "done"},
    ]
    assert client.queries == ["go"]


@pytest.mark.asyncio
async def test_an_inlined_image_goes_out_as_one_user_message_of_content_blocks(monkeypatch):
    image_b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"body" * 8).decode("ascii")
    monkeypatch.setattr(
        dispatch_api,
        "_run_input_seam",
        {
            "run-p": [
                {
                    "filename": "p.png",
                    "mime": "image/png",
                    "entry_id": None,
                    "content_b64": image_b64,
                }
            ]
        },
    )
    clients = _install(monkeypatch, _fixed_stream)

    await _run()

    ((message,),) = clients[0].queries
    image_block = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
    }
    text_block = {
        "type": "text",
        "text": "go\n\nFiles provided with this request:\n"
        "- p.png (image/png) — image shown inline [shown_inline]",
    }
    assert {key: message[key] for key in ("type", "message")} == {
        "type": "user",
        "message": {"role": "user", "content": [image_block, text_block]},
    }


@pytest.mark.asyncio
async def test_the_agent_stderr_reaches_a_failed_run_record(monkeypatch):
    async def _failing(client: _RecordingClient):
        client.options.stderr("cli: boom")
        raise ProcessError("exit 1")
        yield  # pragma: no cover - makes this an async generator

    _install(monkeypatch, _failing)

    result = await _run()

    assert result["status"] == "error"
    assert result["stderr"] == "cli: boom"
    assert result["error"] == str(ProcessError("exit 1"))


@pytest.mark.asyncio
async def test_a_cancelled_run_closes_the_agent_client(monkeypatch):
    started = asyncio.Event()

    async def _blocking(_client: _RecordingClient):
        started.set()
        await asyncio.Event().wait()
        yield  # pragma: no cover - never reached

    clients = _install(monkeypatch, _blocking)
    task = asyncio.create_task(_run())
    await asyncio.wait_for(started.wait(), timeout=10)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (client,) = clients
    assert client.exited
