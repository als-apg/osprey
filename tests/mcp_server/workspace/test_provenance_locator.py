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


def _enable_telemetry(monkeypatch, org="default", backend="openobserve"):
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", f"http://localhost:5080/api/{org}")
    telemetry: dict = {"enabled": True}
    if backend is not None:
        telemetry["backend"] = backend
    if org is not None:
        telemetry[backend or "openobserve"] = {"org": org}
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {"claude_code": {"telemetry": telemetry}},
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_env_hit_returns_forced_id(monkeypatch):
    """The OSPREY-forced id + telemetry on → full coordinates for that id."""
    _enable_telemetry(monkeypatch, org="example")
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_START", "2026-07-15T00:00:00+00:00")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=claude-code,host=x")

    result = json.loads(await _fn()())

    assert result["session_id"] == "sess-abc-123"
    assert result["service_name"] == "claude-code"
    assert result["org"] == "example"
    assert result["stream"] == "default"
    assert result["since"] == "2026-07-15T00:00:00+00:00"
    assert "note" not in result
    # emitted_at is stamped server-side and is a parseable ISO-8601 timestamp.
    datetime.fromisoformat(result["emitted_at"])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_to_harness_id(monkeypatch):
    """No OSPREY id but the harness exports one → the harness id is used."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "harness-xyz")

    result = json.loads(await _fn()())

    assert result["session_id"] == "harness-xyz"
    assert result["service_name"] == "claude-code"  # default, no resource attrs
    assert result["org"] == "default"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_osprey_id_takes_precedence(monkeypatch):
    """When both are present the OSPREY-forced id wins (it is authoritative)."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "osprey-wins")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "harness-loses")

    result = json.loads(await _fn()())

    assert result["session_id"] == "osprey-wins"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_telemetry_disabled_degrades_honestly(monkeypatch):
    """An id is present but telemetry is off → null + note, not a dangling id."""
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")
    # CLAUDE_CODE_ENABLE_TELEMETRY intentionally unset.

    result = json.loads(await _fn()())

    assert result["session_id"] is None
    assert "note" in result and "unavailable" in result["note"]
    datetime.fromisoformat(result["emitted_at"])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_no_id_resolvable_degrades_honestly(monkeypatch):
    """Telemetry on but no id anywhere → null + note."""
    _enable_telemetry(monkeypatch)

    result = json.loads(await _fn()())

    assert result["session_id"] is None
    assert "note" in result


@pytest.mark.asyncio
@pytest.mark.unit
async def test_since_optional(monkeypatch):
    """since is null when no session-start was injected, and the call succeeds."""
    _enable_telemetry(monkeypatch)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-1")

    result = json.loads(await _fn()())

    assert result["session_id"] == "sess-1"
    assert result["since"] is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_generic_backend_carries_no_org_or_stream(monkeypatch):
    """Org and stream address OpenObserve records and nothing else.

    They were emitted unconditionally, the org re-parsed out of the OTLP
    endpoint URL and the stream hard-coded, so a deployment on any other
    backend handed a consumer two coordinates that name nothing in its store.
    Omitted rather than nulled: a null still templates into a filed issue.
    """
    _enable_telemetry(monkeypatch, org="example", backend="generic")
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")

    result = json.loads(await _fn()())

    assert result["session_id"] == "sess-abc-123"
    assert "org" not in result
    assert "stream" not in result


@pytest.mark.asyncio
@pytest.mark.unit
async def test_the_org_comes_from_config_not_from_the_endpoint_url(monkeypatch):
    """One producer: the exporter's URL is built FROM the org, not the reverse."""
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:5080/api/stale")
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {
            "claude_code": {
                "telemetry": {"backend": "openobserve", "openobserve": {"org": "example"}}
            }
        },
    )
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")

    result = json.loads(await _fn()())

    assert result["org"] == "example"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_an_explicitly_empty_org_is_reported_as_written(monkeypatch):
    """The default belongs to an absent key; an explicit ``org: ""`` is a value.

    The provisioner resolves it that way because the exporter does, and the
    locator has to agree with both: rewriting an empty org to ``default`` here
    would point a maintainer at an organization the deployment never writes to.
    """
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:5080/api/")
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {
            "claude_code": {"telemetry": {"backend": "openobserve", "openobserve": {"org": ""}}}
        },
    )
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")

    result = json.loads(await _fn()())

    assert result["org"] == ""


@pytest.mark.asyncio
@pytest.mark.unit
async def test_an_absent_org_falls_back_to_the_default_organization(monkeypatch):
    """No ``org`` key at all is the one case the default is for."""
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:5080/api/default")
    monkeypatch.setattr(
        "osprey.utils.workspace.load_osprey_config",
        lambda: {"claude_code": {"telemetry": {"backend": "openobserve"}}},
    )
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")

    result = json.loads(await _fn()())

    assert result["org"] == "default"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_an_unreadable_config_omits_the_coordinates(monkeypatch):
    """Degrade by saying less, never by guessing a coordinate."""
    monkeypatch.setenv("CLAUDE_CODE_ENABLE_TELEMETRY", "1")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:5080/api/example")
    monkeypatch.setattr("osprey.utils.workspace.load_osprey_config", lambda: {})
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "sess-abc-123")

    result = json.loads(await _fn()())

    assert result["session_id"] == "sess-abc-123"
    assert "org" not in result
