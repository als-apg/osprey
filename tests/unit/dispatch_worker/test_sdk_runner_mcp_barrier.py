"""Unit tests for the dispatch worker's MCP readiness barrier.

``_stream_with_ready_mcp`` holds the run's first turn until the project's
declared MCP servers report ``connected``. Without it a dispatch whose turn
fires during MCP registration sees no OSPREY tools and answers "I don't have
that tool" — a cold start scored as a model give-up.

The ordering assertion is the load-bearing one: polling that happens *after*
the prompt is sent would satisfy a naive "barrier ran" check while leaving the
race wide open. No SDK, no subprocess — a fake client scripts the snapshots.
"""

from __future__ import annotations

import logging
import time

import pytest

from osprey.mcp_server.dispatch_worker import sdk_runner

pytestmark = pytest.mark.unit


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

    async def query(self, prompt: object) -> None:
        self._events.append("query")

    async def receive_response(self):
        self._events.append("receive")
        yield "message-1"
        yield "message-2"


async def _drain(agen) -> list[str]:
    return [m async for m in agen]


@pytest.mark.asyncio
async def test_mcp_is_polled_to_connected_before_the_prompt_is_sent(monkeypatch):
    """The barrier is only worth anything if it precedes the first turn."""
    events: list[str] = []
    monkeypatch.setattr(sdk_runner, "ClaudeSDKClient", lambda options: _FakeClient(events))
    monkeypatch.setattr(sdk_runner, "expected_mcp_servers", lambda _p: {"controls"})

    messages = await _drain(sdk_runner._stream_with_ready_mcp(object(), "/proj", object()))

    assert messages == ["message-1", "message-2"]
    # Every poll precedes the query — and polling continued until connected.
    assert events.index("query") > max(i for i, e in enumerate(events) if e == "poll")
    assert events.count("poll") == 2
    assert events[-1] == "disconnect"


@pytest.mark.asyncio
async def test_run_proceeds_and_warns_when_a_server_never_connects(monkeypatch, caplog):
    """A server that never registers must not fail the dispatch outright — it
    is logged so a missing tool is diagnosable as infra, not model behaviour."""
    events: list[str] = []
    monkeypatch.setattr(sdk_runner, "ClaudeSDKClient", lambda options: _FakeClient(events))
    monkeypatch.setattr(sdk_runner, "expected_mcp_servers", lambda _p: {"controls", "archiver"})
    # Barrier timed out: only one of the two expected servers came up.
    monkeypatch.setattr(
        sdk_runner,
        "await_mcp_ready",
        lambda _c, _e: _ready([{"name": "controls", "status": "connected"}]),
    )

    with caplog.at_level(logging.WARNING, logger=sdk_runner.logger.name):
        messages = await _drain(sdk_runner._stream_with_ready_mcp(object(), "/proj", object()))

    assert messages == ["message-1", "message-2"]  # the run still happens
    assert "archiver" in caplog.text
    assert "not connected" in caplog.text


@pytest.mark.asyncio
async def test_no_declared_servers_is_skipped_rather_than_polled_to_the_deadline(monkeypatch):
    """An empty expectation is never satisfiable, so the underlying barrier
    would poll it to its full multi-second deadline. A project that declares no
    MCP servers (or whose .mcp.json cannot be read) must not pay that on every
    run — the barrier is skipped outright, not merely waited out."""
    events: list[str] = []
    monkeypatch.setattr(sdk_runner, "ClaudeSDKClient", lambda options: _FakeClient(events))
    monkeypatch.setattr(sdk_runner, "expected_mcp_servers", lambda _p: set())

    started = time.monotonic()
    messages = await _drain(sdk_runner._stream_with_ready_mcp(object(), "/proj", object()))
    elapsed = time.monotonic() - started

    assert messages == ["message-1", "message-2"]
    assert "query" in events
    assert "poll" not in events, "barrier ran despite there being nothing to wait for"
    assert elapsed < 1.0, f"run stalled {elapsed:.1f}s on an empty MCP expectation"


async def _ready(servers: list[dict]) -> list[dict]:
    return servers
