"""Unit tests for the MCP readiness barrier a dispatch run goes through.

``run_dispatch`` hands its run to the agent runner, whose barrier holds the
first turn until the render's declared MCP servers report ``connected``, and
refuses the run before the prompt when a server the trigger's allow-list names
is not connected. Without it a dispatch whose turn fires during MCP
registration sees no OSPREY tools and answers "I don't have that tool" — a
cold start scored as a model give-up.

The ordering assertion is the load-bearing one: polling that happens *after*
the prompt is sent would satisfy a naive "barrier ran" check while leaving the
race wide open. No subprocess — a fake client scripts the status snapshots and
answers with real agent SDK messages, so the path is proven through the real
``stream_query`` and its event records.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from osprey.agent_runner import mcp_snapshot_summary
from osprey.mcp_server.dispatch_worker import sdk_runner


@pytest.fixture(autouse=True)
def _stub_osprey_helpers(monkeypatch, tmp_path):
    """Stub the deferred OSPREY helpers and point the render at ``tmp_path``."""
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
    (tmp_path / "build").mkdir()
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "build" / "config.yml"))


class _FakeClient:
    """Reports ``connecting`` for the first ``connect_after`` polls, then
    ``connected``. Records the order of status polls and the query call."""

    def __init__(self, events: list[str], connect_after: int = 1) -> None:
        self._events = events
        self._connect_after = connect_after
        self.polls = 0

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        self._events.append("disconnect")
        return False

    async def get_mcp_status(self) -> dict:
        self.polls += 1
        self._events.append("poll")
        status = "connected" if self.polls > self._connect_after else "connecting"
        return {"mcpServers": [{"name": "controls", "status": status}]}

    async def query(self, _prompt: object) -> None:
        self._events.append("query")

    async def receive_response(self):
        self._events.append("receive")
        yield AssistantMessage(content=[TextBlock(text="ok")], model="m")
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s",
            total_cost_usd=0.1,
        )


def _patch_client(monkeypatch, events: list[str]) -> None:
    monkeypatch.setattr(
        "osprey.agent_runner.runner.ClaudeSDKClient", lambda options: _FakeClient(events)
    )


def _patch_expected(monkeypatch, names: set[str]) -> None:
    monkeypatch.setattr("osprey.agent_runner.primitives.expected_mcp_servers", lambda _p: names)


def _patch_ready(monkeypatch, servers: list[dict]) -> None:
    monkeypatch.setattr(
        "osprey.agent_runner.primitives.await_mcp_ready", lambda _c, _e: _ready(servers)
    )


async def _dispatch(allowed_tools: list[str]) -> dict:
    return await sdk_runner.run_dispatch("go", allowed_tools, event_queue=asyncio.Queue())


@pytest.mark.asyncio
async def test_mcp_is_polled_to_connected_before_the_prompt_is_sent(monkeypatch):
    """The barrier is only worth anything if it precedes the first turn."""
    events: list[str] = []
    _patch_client(monkeypatch, events)
    _patch_expected(monkeypatch, {"controls"})

    result = await _dispatch(["Read"])

    assert result["status"] == "completed"
    assert result["text_output"] == "ok"
    # Every poll precedes the query — and polling continued until connected.
    assert events.index("query") > max(i for i, e in enumerate(events) if e == "poll")
    assert events.count("poll") == 2
    assert events[-1] == "disconnect"


@pytest.mark.asyncio
async def test_run_proceeds_and_warns_when_a_server_never_connects(monkeypatch, caplog):
    """A server that never registers must not fail the dispatch outright — it
    is logged so a missing tool is diagnosable as infra, not model behaviour."""
    events: list[str] = []
    _patch_client(monkeypatch, events)
    _patch_expected(monkeypatch, {"controls", "archiver"})
    # Barrier timed out: only one of the two expected servers came up.
    _patch_ready(monkeypatch, [{"name": "controls", "status": "connected"}])

    with caplog.at_level(logging.WARNING):
        result = await _dispatch(["Read"])

    assert result["status"] == "completed"  # the run still happens
    assert "archiver" in caplog.text
    assert "not connected" in caplog.text


@pytest.mark.asyncio
async def test_no_declared_servers_is_skipped_rather_than_polled_to_the_deadline(monkeypatch):
    """An empty expectation is never satisfiable, so the underlying barrier
    would poll it to its full multi-second deadline. A project that declares no
    MCP servers (or whose .mcp.json cannot be read) must not pay that on every
    run — the barrier is skipped outright, not merely waited out."""
    events: list[str] = []
    _patch_client(monkeypatch, events)
    _patch_expected(monkeypatch, set())

    started = time.monotonic()
    result = await _dispatch(["Read"])
    elapsed = time.monotonic() - started

    assert result["status"] == "completed"
    assert "query" in events
    assert "poll" not in events, "barrier ran despite there being nothing to wait for"
    assert elapsed < 1.0, f"run stalled {elapsed:.1f}s on an empty MCP expectation"


@pytest.mark.asyncio
async def test_required_server_not_connected_refuses_to_send_the_prompt(monkeypatch):
    """The CLI fixes a run's MCP toolset at the first turn, so a server the
    trigger's own allow-list names that is not connected by then is lost for
    the whole run: the agent would run without the tool it was dispatched to
    use and report "completed". Refuse before the prompt goes out, naming the
    server and the status it was left in."""
    events: list[str] = []
    _patch_client(monkeypatch, events)
    _patch_expected(monkeypatch, {"controls", "osprey_workspace"})
    servers = [
        {"name": "controls", "status": "connected", "tools": [{"name": "channel_read"}]},
        {"name": "osprey_workspace", "status": "pending"},
    ]
    _patch_ready(monkeypatch, servers)

    result = await _dispatch(["mcp__osprey_workspace__artifact_register"])

    assert result["status"] == "error"
    assert result["failure_class"] == "infrastructure"
    assert "osprey_workspace (pending)" in result["error"]
    # The snapshot is handed back even on refusal, so the run record carries it.
    assert result["mcp_servers"] == mcp_snapshot_summary(servers)
    assert "query" not in events, "prompt was sent despite a required server being absent"
    assert events[-1] == "disconnect"


@pytest.mark.asyncio
async def test_failed_required_server_is_named_with_its_error(monkeypatch):
    events: list[str] = []
    _patch_client(monkeypatch, events)
    _patch_expected(monkeypatch, {"osprey_workspace"})
    _patch_ready(
        monkeypatch, [{"name": "osprey_workspace", "status": "failed", "error": "spawn: ENOENT"}]
    )

    result = await _dispatch(["mcp__osprey_workspace__artifact_register"])

    assert result["status"] == "error"
    assert "failed" in result["error"] and "spawn: ENOENT" in result["error"]
    assert "query" not in events


@pytest.mark.asyncio
async def test_optional_server_not_connected_still_runs(monkeypatch, caplog):
    """A declared server the trigger does not allow-list cannot be called by
    the main thread anyway; its absence is logged, not fatal."""
    events: list[str] = []
    _patch_client(monkeypatch, events)
    _patch_expected(monkeypatch, {"controls", "graph"})
    _patch_ready(
        monkeypatch,
        [
            {"name": "controls", "status": "connected", "tools": [{"name": "channel_read"}]},
            {"name": "graph", "status": "pending"},
        ],
    )

    with caplog.at_level(logging.WARNING):
        result = await _dispatch(["mcp__controls__channel_read"])

    assert result["status"] == "completed"
    assert "graph" in caplog.text and "not connected" in caplog.text
    assert [s["tools"] for s in result["mcp_servers"]] == [1, 0]


def test_required_servers_are_read_off_the_allow_list():
    """Every ``mcp__<server>__<tool>`` (or server-level ``mcp__<server>``) entry
    names a server the run cannot do without; built-in tools name none."""
    assert sdk_runner.required_mcp_servers(
        ["Glob", "Read", "mcp__osprey_workspace__artifact_register", "mcp__controls", "Task"]
    ) == {"osprey_workspace", "controls"}
    assert sdk_runner.required_mcp_servers([]) == set()
    assert sdk_runner.required_mcp_servers(["mcp__channel-finder__search"]) == {"channel-finder"}


@pytest.mark.asyncio
async def test_the_barrier_waits_for_the_servers_the_render_declares(monkeypatch, tmp_path):
    """With no readiness set of its own, the run waits for the servers the
    render's ``.mcp.json`` declares."""
    (tmp_path / "build" / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"controls": {"command": "controls-server"}}})
    )
    events: list[str] = []
    _patch_client(monkeypatch, events)

    result = await _dispatch(["Read"])

    assert result["status"] == "completed"
    assert "poll" in events
    assert events.index("poll") < events.index("query")


async def _ready(servers: list[dict]) -> list[dict]:
    return servers
