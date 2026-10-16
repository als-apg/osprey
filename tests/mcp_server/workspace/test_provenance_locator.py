"""Tests for the provenance_locator MCP tool.

Covers the return paths that matter for a filed issue's telemetry pointer:
the OSPREY-forced id, the harness fallback id, their precedence, and the
honest-degradation paths (telemetry disabled, or no id resolvable) that must
return ``session_id: null`` with a note rather than a dangling pointer. The
tool must never raise.
"""

import json
from datetime import datetime

import pytest

from tests.mcp_server.conftest import get_tool_fn

# Env vars the tool reads — cleared before each test so ambient values in the
# runner environment cannot leak into an assertion.
_RELEVANT_ENV = (
    "OSPREY_TELEMETRY_SESSION_ID",
    "OSPREY_TELEMETRY_SESSION_START",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_ENABLE_TELEMETRY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_RESOURCE_ATTRIBUTES",
)


def _fn():
    from osprey.mcp_server.workspace.tools.provenance_locator import provenance_locator

    return get_tool_fn(provenance_locator)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _RELEVANT_ENV:
        monkeypatch.delenv(name, raising=False)


def _enable_telemetry(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")


@pytest.mark.asyncio
async def test_env_hit_returns_forced_id(monkeypatch):
    """The OSPREY-forced id + telemetry on → full coordinates for that id."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_START", "2026-07-15T00:00:00+00:00")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=claude-code,host=x")

    result = json.loads(await _fn()())

    assert result["session_id"] == "sess-abc-123"
    assert result["service_name"] == "claude-code"
    assert result["since"] == "2026-07-15T00:00:00+00:00"
    assert "note" not in result
    # emitted_at is stamped server-side and is a parseable ISO-8601 timestamp.
    datetime.fromisoformat(result["emitted_at"])


@pytest.mark.asyncio
async def test_fallback_to_harness_id(monkeypatch):
    """No OSPREY id but the harness exports one → the harness id is used."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "harness-xyz")

    result = json.loads(await _fn()())

    assert result["session_id"] == "harness-xyz"
    assert result["service_name"] == "claude-code"  # default, no resource attrs


@pytest.mark.asyncio
async def test_osprey_id_takes_precedence(monkeypatch):
    """When both are present the OSPREY-forced id wins (it is authoritative)."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "osprey-wins")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "harness-loses")

    result = json.loads(await _fn()())

    assert result["session_id"] == "osprey-wins"


@pytest.mark.asyncio
async def test_telemetry_disabled_degrades_honestly(monkeypatch):
    """An id is present but telemetry is off → null + note, not a dangling id."""
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")
    # CLAUDE_CODE_ENABLE_TELEMETRY intentionally unset.

    result = json.loads(await _fn()())

    assert result["session_id"] is None
    assert "note" in result and "unavailable" in result["note"]
    datetime.fromisoformat(result["emitted_at"])


@pytest.mark.asyncio
async def test_no_id_resolvable_degrades_honestly(monkeypatch):
    """Telemetry on but no id anywhere → null + note."""
    _enable_telemetry(monkeypatch)

    result = json.loads(await _fn()())

    assert result["session_id"] is None
    assert "note" in result


@pytest.mark.asyncio
async def test_since_optional(monkeypatch):
    """since is null when no session-start was injected, and the call succeeds."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-1")

    result = json.loads(await _fn()())

    assert result["session_id"] == "sess-1"
    assert result["since"] is None


@pytest.mark.asyncio
async def test_the_locator_carries_exactly_the_session_coordinates(monkeypatch):
    """The store is found by session id, service name and time alone.

    No store-specific addressing (an organization, a stream) is handed out: a
    key that names nothing in the store would still template into a filed issue.
    """
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")

    result = json.loads(await _fn()())

    assert set(result) == {"session_id", "service_name", "since", "emitted_at"}
