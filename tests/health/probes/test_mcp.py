"""Tests for the ``mcp`` health probe.

The ``mcp`` SDK client is mocked by patching ``streamablehttp_client`` and
``ClientSession`` in the probe module, so no real MCP server is spawned. Covers
a successful handshake, the ``expect_tools`` diff, a zero-tools server, a
handshake timeout, and a connection failure.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from osprey.health.models import Status
from osprey.health.probes import mcp as probe


class _FakeStreamCM:
    """Async CM standing in for ``streamablehttp_client(url)``."""

    def __init__(self, *, connect_error: Exception | None = None):
        self._connect_error = connect_error

    async def __aenter__(self):
        if self._connect_error is not None:
            raise self._connect_error
        return ("read", "write", lambda: "session-id")

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Async CM standing in for ``ClientSession(read, write)``."""

    def __init__(self, tool_names, *, hang: bool = False):
        self._tool_names = tool_names
        self._hang = hang

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return SimpleNamespace(instructions="")

    async def list_tools(self):
        if self._hang:
            await asyncio.sleep(10)
        tools = [SimpleNamespace(name=n, description="") for n in self._tool_names]
        return SimpleNamespace(tools=tools)


def _patch_client(monkeypatch, *, tool_names=None, connect_error=None, hang=False):
    """Wire the probe's SDK symbols to the fakes above."""
    tool_names = tool_names or []

    def _fake_streamable(url, *args, **kwargs):
        return _FakeStreamCM(connect_error=connect_error)

    def _fake_session(read, write, *args, **kwargs):
        return _FakeSession(tool_names, hang=hang)

    monkeypatch.setattr(probe, "streamablehttp_client", _fake_streamable)
    monkeypatch.setattr(probe, "ClientSession", _fake_session)


_CTX = object()  # ProbeContext is unused by the mcp probe.


class TestSuccessfulHandshake:
    async def test_no_expect_tools_reports_count(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=["a", "b", "c"])
        result = await probe.run({"url": "http://x/mcp"}, _CTX)
        assert result.status == Status.OK
        assert result.value == "3 tools"
        assert result.latency_ms >= 0
        assert result.name == "mcp"
        assert result.category == "mcp"

    async def test_name_and_category_overrides(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=["a"])
        spec = {"url": "http://x/mcp", "name": "controls", "category": "servers"}
        result = await probe.run(spec, _CTX)
        assert result.name == "controls"
        assert result.category == "servers"

    async def test_zero_tools_without_expect_is_error(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=[])
        result = await probe.run({"url": "http://x/mcp"}, _CTX)
        assert result.status == Status.ERROR
        assert result.value == "0 tools"
        assert "no tools" in result.message


class TestExpectTools:
    async def test_all_expected_present_is_ok(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=["read", "write", "extra"])
        spec = {"url": "http://x/mcp", "expect_tools": ["read", "write"]}
        result = await probe.run(spec, _CTX)
        assert result.status == Status.OK
        assert result.value == "3/2 tools"

    async def test_missing_expected_is_error_with_value(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=["read"])
        spec = {"url": "http://x/mcp", "expect_tools": ["read", "write", "delete"]}
        result = await probe.run(spec, _CTX)
        assert result.status == Status.ERROR
        assert result.value == "1/3 tools"
        assert "write" in result.message
        assert "delete" in result.message
        assert "read" in result.details

    async def test_expect_tools_against_empty_server_is_error(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=[])
        spec = {"url": "http://x/mcp", "expect_tools": ["read"]}
        result = await probe.run(spec, _CTX)
        assert result.status == Status.ERROR
        assert result.value == "0/1 tools"
        assert "(none)" in result.details


class TestFailureModes:
    async def test_handshake_timeout_uses_timeout_status(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=["a"], hang=True)
        spec = {"url": "http://x/mcp", "timeout_s": 0.01}
        result = await probe.run(spec, _CTX)
        assert result.status == Status.ERROR
        assert "timed out" in result.message
        assert result.latency_ms >= 0

    async def test_timeout_status_warning_opt_in(self, monkeypatch):
        _patch_client(monkeypatch, tool_names=["a"], hang=True)
        spec = {"url": "http://x/mcp", "timeout_s": 0.01, "timeout_status": "warning"}
        result = await probe.run(spec, _CTX)
        assert result.status == Status.WARNING

    async def test_connection_failure_is_error(self, monkeypatch):
        _patch_client(monkeypatch, connect_error=ConnectionRefusedError("refused"))
        result = await probe.run({"url": "http://x/mcp"}, _CTX)
        assert result.status == Status.ERROR
        assert "failed" in result.message
        assert "refused" in result.details
