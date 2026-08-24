"""Shared fixtures for MCP server tool tests.

Provides mock connectors, temporary workspace directories, and config factories
for testing MCP tools in isolation from the real control system.

IMPORTANT: FastMCP's @mcp.tool() decorator wraps functions into FunctionTool
objects. To call the original async function in tests, use the `.fn` attribute:
    tool.fn(channels=["SR:CURRENT:RB"])
"""

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastmcp.exceptions import ToolError
from mcp.types import CallToolResult

from osprey.mcp_server.control_system.server_context import (
    initialize_server_context,
    reset_server_context,
)
from osprey.mcp_server.errors import extract_error_envelope
from osprey.mcp_server.workspace.tools.screen_capture_backends import reset_backend
from osprey.stores.artifact_store import reset_artifact_store
from osprey.utils.workspace import reset_config_cache


def get_tool_fn(tool_or_fn):
    """Extract the raw async function from a FastMCP FunctionTool.

    If already a plain function/coroutine, returns it unchanged.
    """
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn


def assert_error(result, *, error_type: str | None = None) -> dict:
    """Assert tool returned an error and return its structured envelope dict.

    Accepts either a ``CallToolResult(isError=True)`` (the post-migration
    contract) or a legacy JSON string with ``{"error": true, ...}`` so callers
    can be migrated incrementally without forcing every test to land in one
    commit. Returns the parsed envelope so callers can make further
    assertions on ``error_message``, ``details``, etc.
    """
    if isinstance(result, str):
        envelope = json.loads(result)
        assert isinstance(envelope, dict) and envelope.get("error") is True, (
            f"Expected error envelope, got: {envelope!r}"
        )
    elif isinstance(result, CallToolResult):
        envelope = extract_error_envelope(result)
        assert envelope is not None, f"Expected isError=True with error envelope, got: {result!r}"
    else:
        raise AssertionError(f"Unexpected tool result type: {type(result).__name__}: {result!r}")
    if error_type is not None:
        assert envelope.get("error_type") == error_type, (
            f"Expected error_type={error_type!r}, got {envelope.get('error_type')!r}"
        )
    return envelope


@contextmanager
def assert_raises_error(*, error_type: str | None = None):
    """Assert the wrapped tool call raises ToolError carrying the standard envelope.

    Yields a dict with ``envelope`` populated after the block exits, so callers can
    make further assertions on ``error_message``, ``details``, etc.::

        with assert_raises_error(error_type="validation_error") as ctx:
            tool.fn(...)
        assert "expected token" in ctx["envelope"]["error_message"]
    """
    captured: dict = {}
    with pytest.raises(ToolError) as exc_info:
        yield captured
    envelope = json.loads(str(exc_info.value))
    assert isinstance(envelope, dict) and envelope.get("error") is True, (
        f"Expected error envelope in ToolError message, got: {envelope!r}"
    )
    if error_type is not None:
        assert envelope.get("error_type") == error_type, (
            f"Expected error_type={error_type!r}, got {envelope.get('error_type')!r}"
        )
    captured["envelope"] = envelope


def extract_response_dict(result) -> dict:
    """Pull the structured response dict from a tool result.

    Handles both legacy JSON-string returns and the new
    ``CallToolResult(content=[TextContent(...)])`` shape used by the python
    executor's response builder. Useful for tests that assert on shape keys
    like ``has_errors``, ``status``, or ``summary``.
    """
    if isinstance(result, str):
        return json.loads(result)
    if isinstance(result, CallToolResult):
        for block in result.content or []:
            text = getattr(block, "text", None)
            if not text:
                continue
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
        raise AssertionError(f"CallToolResult had no JSON-decodable text content: {result!r}")
    raise AssertionError(f"Unexpected tool result type: {type(result).__name__}: {result!r}")


def registered_tool_names(mcp) -> list[str]:
    """Tool names registered on a FastMCP server, across FastMCP versions.

    ``get_tools()`` is a coroutine on newer FastMCP releases and a plain call on
    older ones, and its result is a dict on some and a list of tools on others.
    """
    tools = mcp.get_tools() if hasattr(mcp, "get_tools") else mcp.list_tools()
    if asyncio.iscoroutine(tools):
        tools = asyncio.run(tools)
    if isinstance(tools, dict):
        return list(tools)
    return [t.name for t in tools]


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset the MCP registry, ArtifactStore, screen-capture backend and config caches.

    Leak guarded: the server context, artifact store, screen-capture backend and
    the ``osprey.utils.config`` caches are all process-wide singletons. Every one
    of them is reset both before and after the test, so a directory that ran
    earlier in the same worker cannot hand its state to the first test here, and
    this directory cannot hand its state to whatever runs next.
    """
    import osprey.utils.config as _cfg

    def _reset_all():
        reset_server_context()
        reset_artifact_store()
        reset_backend()
        reset_config_cache()
        _cfg._config_cache.clear()

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    saved_cache = _cfg._config_cache.copy()
    _reset_all()

    yield

    _reset_all()
    _cfg._config_cache.update(saved_cache)


@pytest.fixture(autouse=True)
def _unregister_artifact_activity():
    """Disarm the artifact-activity listener around every test in this directory.

    ``initialize_workspace_singletons()`` subscribes the listener to the
    ArtifactStore *class*, so any test that calls it leaves every later test in
    the same worker emitting real ``/api/agent-activity`` POSTs at whatever is
    listening on the web-terminal port. Unregistering on both sides keeps that
    process-global arming inside the test that asked for it.
    """
    from osprey.mcp_server.artifact_activity import unregister_artifact_activity_listeners

    unregister_artifact_activity_listeners()
    yield
    unregister_artifact_activity_listeners()


@pytest.fixture(autouse=True)
def _block_web_terminal_posts(request, monkeypatch):
    """Keep the notify_* helpers' HTTP POSTs inside the test process.

    Every ``notify_*`` helper in :mod:`osprey.mcp_server.http` opens a real
    socket to the resolved web-terminal port. On CI nothing is listening and the
    connection is merely refused, but on a developer box a live web terminal is
    — and then the unit suite drives the operator's actual UI, glowing tiles and
    filling the activity strip. Both posters are stubbed here to the outcome
    they already produce when the terminal is down: ``post_json`` swallows and
    returns ``None``; ``_post_json_with_response`` raises ``URLError`` so
    ``notify_panel_register`` / ``notify_panel_arrange`` take their existing
    unreachable branch. Patching only ``post_json`` would miss those two.

    Patched in the ``http`` module's own namespace, which is where the helpers
    resolve them from, so it holds however a tool module imported the helper.
    Tests that assert on emits patch ``notify_agent_activity`` at their own call
    site — above this seam — and are unaffected. ``test_http.py`` exercises the
    posters themselves and opts out with the ``real_http_posters`` marker.
    """
    if request.node.get_closest_marker("real_http_posters"):
        return

    import urllib.error

    from osprey.mcp_server import http as _http

    def _unreachable(url, payload, *, timeout=3):
        raise urllib.error.URLError("web terminal POSTs are blocked in unit tests")

    monkeypatch.setattr(_http, "post_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(_http, "_post_json_with_response", _unreachable)


@pytest.fixture
def init_registry(tmp_path, monkeypatch):
    """Initialize the MCP registry after chdir and config setup.

    Call this fixture AFTER writing config.yml and chdir-ing to tmp_path.
    Returns the registry instance for direct use in tests.
    """

    def _init():
        return initialize_server_context()

    return _init


@pytest.fixture
def workspace_dir(tmp_path):
    """Create temporary workspace structure matching _agent_data layout."""
    ws = tmp_path / "_agent_data"
    for subdir in [
        "channel_results",
        "archiver_data",
        "python_outputs",
        "search_results",
        "screenshots",
    ]:
        (ws / subdir).mkdir(parents=True)
    return ws


@pytest.fixture
def mock_config(tmp_path):
    """Create a minimal config.yml for MCP tool tests."""
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.dump(
            {
                "control_system": {
                    "type": "mock",
                    "writes_enabled": True,
                    "write_verification": {
                        "default_level": "callback",
                        "default_tolerance": 0.1,
                    },
                    "limits_checking": {"enabled": False},
                },
                "archiver": {"type": "mock"},
            }
        )
    )
    return config


@pytest.fixture
def mock_config_writes_disabled(tmp_path):
    """Config with writes_enabled: false."""
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.dump(
            {
                "control_system": {
                    "type": "mock",
                    "writes_enabled": False,
                    "limits_checking": {"enabled": False},
                },
            }
        )
    )
    return config


@pytest.fixture
def mock_config_with_limits(tmp_path):
    """Config with limits_checking enabled and a channel_limits.json database."""
    limits_db = tmp_path / "channel_limits.json"
    limits_db.write_text(
        json.dumps(
            {
                "TEST:PV:SETPOINT": {
                    "min_value": 0.0,
                    "max_value": 100.0,
                    "writable": True,
                },
                "TEST:PV:READONLY": {
                    "writable": False,
                },
            }
        )
    )
    config = tmp_path / "config.yml"
    config.write_text(
        yaml.dump(
            {
                "control_system": {
                    "type": "mock",
                    "writes_enabled": True,
                    "limits_checking": {
                        "enabled": True,
                        "database_path": str(limits_db),
                        "allow_unlisted_channels": False,
                        "on_violation": "error",
                    },
                },
                "archiver": {"type": "mock"},
                "execution": {
                    "execution_method": "subprocess",
                },
                "python_executor": {
                    "execution_timeout_seconds": 60,
                },
            }
        )
    )
    return config


@pytest.fixture
def mock_connector():
    """Mock control system connector with async methods."""
    connector = AsyncMock()
    connector.disconnect = AsyncMock()
    return connector


@pytest.fixture
def mock_channel_value():
    """Factory for creating mock ChannelValue objects."""

    def _make(value=500.2, units="mA", alarm_status="NO_ALARM", timestamp="2024-01-15T10:30:00"):
        cv = MagicMock()
        cv.value = value
        cv.timestamp = timestamp
        cv.metadata = MagicMock()
        cv.metadata.units = units
        cv.metadata.alarm_status = alarm_status
        cv.metadata.precision = 3
        cv.metadata.description = "Test channel"
        cv.metadata.min_value = 0.0
        cv.metadata.max_value = 1000.0
        # Not an enum channel. Spelled out because a bare MagicMock attribute is
        # truthy, and the read tool ships these keys only when they are not None.
        cv.metadata.enum_label = None
        cv.metadata.enum_labels = None
        cv.metadata.raw_metadata = {}
        return cv

    return _make


@pytest.fixture
def mock_write_result():
    """Factory for creating mock ChannelWriteResult objects."""

    def _make(channel="TEST:PV", value=1.0, success=True, error_message=None):
        result = MagicMock()
        result.channel_address = channel
        result.value_written = value
        result.success = success
        result.error_message = error_message
        result.verification = MagicMock()
        result.verification.level = "callback"
        result.verification.verified = True
        result.verification.readback_value = value
        result.verification.tolerance_used = 0.1
        result.verification.notes = ""
        return result

    return _make
