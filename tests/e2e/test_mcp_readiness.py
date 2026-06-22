"""Unit tests for the MCP readiness barrier + instrumentation in sdk_helpers.

These exercise the cold-start-race fix without any API calls or a real CLI:
``_await_mcp_ready`` is driven by a fake client, and the ``SDKWorkflowResult``
accessors are pure data transforms. The integration behaviour (the barrier
actually eliminating the controls cold-start flake) is covered by the e2e suite.

Regression target: the OSPREY controls MCP server registers ~1.5s after launch,
so an agent whose first turn fires earlier sees no controls tools and may give up.
The barrier polls ``get_mcp_status()`` until the declared servers are connected.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.e2e.sdk_helpers import (
    SDKWorkflowResult,
    ToolTrace,
    _await_mcp_ready,
    _expected_mcp_servers,
)

pytestmark = pytest.mark.unit


class _FakeClient:
    """Returns a scripted sequence of get_mcp_status() responses, then repeats
    the last one. Optionally raises on the first N calls to mimic the stream not
    being live yet during early startup."""

    def __init__(self, snapshots: list[list[dict]], raise_first: int = 0) -> None:
        self._snapshots = snapshots
        self._i = 0
        self._raise_first = raise_first
        self.calls = 0

    async def get_mcp_status(self) -> dict:
        self.calls += 1
        if self.calls <= self._raise_first:
            raise RuntimeError("stream not ready")
        snap = self._snapshots[min(self._i, len(self._snapshots) - 1)]
        self._i += 1
        return {"mcpServers": snap}


def _srv(name: str, status: str, tools: list[str] | None = None) -> dict:
    return {"name": name, "status": status, "tools": [{"name": t} for t in (tools or [])]}


def test_await_returns_once_all_expected_connected() -> None:
    """Polls past 'pending' snapshots and returns as soon as every expected
    server reaches 'connected' (the controls cold-start trajectory)."""
    client = _FakeClient(
        [
            [_srv("controls", "pending"), _srv("python", "pending")],
            [_srv("controls", "pending"), _srv("python", "connected", ["execute"])],
            [
                _srv("controls", "connected", ["channel_write"]),
                _srv("python", "connected", ["execute"]),
            ],
        ]
    )
    servers = asyncio.run(_await_mcp_ready(client, {"controls", "python"}, timeout_s=5, poll_s=0))
    status = {s["name"]: s["status"] for s in servers}
    assert status == {"controls": "connected", "python": "connected"}
    assert client.calls == 3  # did not over-poll once ready


def test_await_tolerates_early_get_status_errors() -> None:
    """get_mcp_status() raising during early startup must not crash the barrier."""
    client = _FakeClient(
        [[_srv("controls", "connected", ["channel_write"])]],
        raise_first=2,
    )
    servers = asyncio.run(_await_mcp_ready(client, {"controls"}, timeout_s=5, poll_s=0))
    assert {s["name"]: s["status"] for s in servers} == {"controls": "connected"}


def test_await_returns_last_snapshot_on_timeout_without_raising() -> None:
    """If a server never connects, return the last snapshot (don't raise) so the
    caller records a genuine registration failure rather than masking it."""
    client = _FakeClient([[_srv("controls", "pending"), _srv("python", "connected")]])
    servers = asyncio.run(_await_mcp_ready(client, {"controls"}, timeout_s=0.05, poll_s=0.01))
    assert {s["name"]: s["status"] for s in servers} == {
        "controls": "pending",
        "python": "connected",
    }


def test_result_accessors_expose_authoritative_registration() -> None:
    """The snapshot drives the infra-vs-model discriminator on the result."""
    wf = SDKWorkflowResult()
    wf.mcp_servers = [
        _srv("controls", "connected", ["channel_read", "channel_write"]),
        _srv("python", "connected", ["execute"]),
    ]
    assert wf.mcp_server_status == {"controls": "connected", "python": "connected"}
    assert "mcp__controls__channel_write" in wf.registered_tools
    assert wf.tool_was_registered("mcp__controls__channel_write") is True
    assert wf.tool_was_registered("mcp__controls__nonexistent") is False


def test_tool_was_registered_is_none_without_snapshot() -> None:
    """No snapshot => cannot tell (None), distinct from 'not registered' (False)."""
    assert SDKWorkflowResult().tool_was_registered("mcp__controls__channel_write") is None


def test_expected_mcp_servers_reads_mcp_json(tmp_path) -> None:
    (tmp_path / ".mcp.json").write_text(
        '{"mcpServers": {"controls": {}, "python": {}, "osprey_workspace": {}}}',
        encoding="utf-8",
    )
    assert _expected_mcp_servers(tmp_path) == {"controls", "python", "osprey_workspace"}


def test_expected_mcp_servers_empty_when_missing(tmp_path) -> None:
    assert _expected_mcp_servers(tmp_path) == set()


def test_redelegation_loop_detected_on_repeated_identical_agent_spawns() -> None:
    """3+ identical Agent spawns = non-convergence loop (a MODEL timeout)."""
    wf = SDKWorkflowResult()
    same = {"description": "Find RF cavity C1 temperature channels"}
    wf.tool_traces = [
        ToolTrace(name="Agent", input=same),
        ToolTrace(name="mcp__channel-finder__get_options", input={"level": 1}),
        ToolTrace(name="Agent", input=same),
        ToolTrace(name="Agent", input=same),
    ]
    assert wf.has_redelegation_loop is True
    assert wf.repeated_tool_calls  # the duplicate Agent call is recorded


def test_no_redelegation_loop_for_distinct_or_few_delegations() -> None:
    """Distinct subtasks (or a single retry) are not a loop."""
    wf = SDKWorkflowResult()
    wf.tool_traces = [
        ToolTrace(name="Agent", input={"description": "find temperature channels"}),
        ToolTrace(name="Agent", input={"description": "find reflected-power channels"}),
        ToolTrace(name="Agent", input={"description": "find temperature channels"}),
    ]
    assert wf.has_redelegation_loop is False  # only 2 identical, below threshold


def test_repeated_non_delegation_tool_is_not_a_redelegation_loop() -> None:
    """Hammering a normal tool is recorded but is not flagged as a delegation loop."""
    wf = SDKWorkflowResult()
    wf.tool_traces = [ToolTrace(name="mcp__controls__channel_read", input={"c": "x"})] * 4
    assert wf.has_redelegation_loop is False
    assert wf.repeated_tool_calls  # still surfaced in the digest
