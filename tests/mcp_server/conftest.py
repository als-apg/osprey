"""Shared fixtures for MCP server tool tests.

Provides mock connectors, temporary workspace directories, and config factories
for testing MCP tools in isolation from the real control system.

IMPORTANT: FastMCP's @mcp.tool() decorator wraps functions into FunctionTool
objects. To call the original async function in tests, use the `.fn` attribute:
    tool.fn(channels=["SR:CURRENT:RB"])
"""

import asyncio
import json
import sys
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
import yaml
from fastmcp.exceptions import ToolError
from mcp.types import CallToolResult

from osprey.mcp_server.control_system.server_context import (
    reset_server_context,
)
from osprey.mcp_server.workspace.tools.screen_capture_backends import reset_backend
from osprey.stores.artifact_store import reset_artifact_store
from osprey_connectors.workspace import reset_config_cache
from tests._config_lookup import reset_config_caches  # noqa: F401 - fixture, used by name

# The two-target child harness, re-exported so pytest resolves its fixtures for
# every module in this directory. A fixture a sibling module needs has to be
# where pytest looks for it, or each consumer carries a rebind of its own and a
# dependency added to one of them resolves in no module but the one it was
# written in. None of them is autouse, so a module that neither requests nor
# defines them is unaffected, and a module with a fixture of the same name keeps
# shadowing these with its own.
from tests.mcp_server._switch_harness import (  # noqa: F401
    child_environment,
    fixture_dir,
    live_type,
    make_manager,
    posture_root,
    state_root,
)


def hook_error_class_map() -> dict[str, str]:
    """Read ``ERROR_CLASS_MAP`` out of the shipped error-guidance hook.

    Parsed from source rather than imported: the hook is a standalone script
    that inserts its own directory onto ``sys.path`` at import time, and a test
    that pins its taxonomy should not have that side effect.

    Returns:
        The hook's ``error_type`` to error-class mapping.
    """
    import ast
    import pathlib

    import osprey

    hook = (
        pathlib.Path(osprey.__file__).parent
        / "templates/claude_code/claude/hooks/osprey_error_guidance.py"
    )
    tree = ast.parse(hook.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ERROR_CLASS_MAP"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"ERROR_CLASS_MAP not found in {hook}")


def extract_error_envelope(result: CallToolResult) -> dict | None:
    """Pull the structured envelope out of a wire-form error CallToolResult.

    Returns ``None`` when the result is not an error or its content cannot be
    parsed as the envelope dict. Used by :func:`assert_error` to inspect the
    structured payload behind ``isError=True``.
    """
    if not isinstance(result, CallToolResult):
        return None
    if result.isError is not True:
        return None
    for block in result.content or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("error") is True:
            return parsed
    return None


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

    Handles both JSON-string returns and the
    ``CallToolResult(content=[TextContent(...)])`` shape. Useful for tests that
    assert on shape keys like ``has_errors``, ``status``, or ``summary``.
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
    """Tool names registered on a FastMCP server.

    ``list_tools()`` is FastMCP's public, supported listing API — a coroutine
    returning the tools a client would actually be offered, with disabled,
    backend-only and auth-gated ones already filtered out.
    """
    tools = asyncio.run(mcp.list_tools())
    return [t.name for t in tools]


def _reset_facility_knowledge_bundle() -> None:
    """Clear the facility-knowledge bundle singletons, if that server was imported.

    Looked up through ``sys.modules`` rather than imported outright: almost no
    test in this directory touches the facility-knowledge server, and importing
    it for all of them to clear state only a handful of them create is a cost
    with no payer. A module nothing imported holds no state to clear.
    """
    module = sys.modules.get("osprey.mcp_server.facility_knowledge.server")
    if module is not None:
        module.reset_bundle()


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """Reset the MCP registry, ArtifactStore, screen-capture backend and config caches.

    Leak guarded: the server context, artifact store, screen-capture backend,
    the facility-knowledge bundle and the ``osprey_connectors.config`` caches are all
    process-wide singletons. Every one of them is reset both before and after
    the test, so a directory that ran earlier in the same worker cannot hand its
    state to the first test here, and this directory cannot hand its state to
    whatever runs next.
    """
    import osprey_connectors.config as _cfg

    def _reset_all():
        reset_server_context()
        reset_artifact_store()
        reset_backend()
        _reset_facility_knowledge_bundle()
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
    ``notify_panel_register`` and ``panel_op`` take their existing
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

    def _unreachable(_url, _payload, *, timeout=3):  # noqa: ARG001 - _post_json_with_response fixes this keyword-only parameter
        raise urllib.error.URLError("web terminal POSTs are blocked in unit tests")

    monkeypatch.setattr(_http, "post_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(_http, "_post_json_with_response", _unreachable)


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
                    "type": "virtual_accelerator",
                    "connector": {"virtual_accelerator": {"serving": "in_process"}},
                    "writes": "allowed",
                    "limits_checking": {"enabled": False},
                },
                "archiver": {"type": "mock"},
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


# ---- session scope ----------------------------------------------------------

#: The variables :func:`osprey.stores.artifact_scope.current_scope` reads. Both
#: session fixtures clear all of them first, so a test run from an ``osprey chat``
#: shell or a web terminal never resolves the developer's own session.
_SCOPE_ENV = (
    "OSPREY_SESSION_ID",
    "OSPREY_DISPATCH_RUN_ID",
    "OSPREY_DISPATCH_RUN",
    "OSPREY_TERMINAL_SESSION",
    "OSPREY_CLI_LAUNCH",
    "OSPREY_TELEMETRY_SESSION_ID",
    "CLAUDE_CODE_SESSION_ID",
)


class WebSession:
    """A web terminal session: a key with one ledger row naming its transcript."""

    key = "11111111-1111-4111-8111-111111111111"

    def __init__(self, state_dir):
        self.state_dir = state_dir
        self.transcript = self.key  # a key's first transcript id equals the key

    def clear(self, transcript: str) -> None:
        """Move the key to *transcript*, as the turn-state hook does on a ``/clear``."""
        from osprey.stores import session_ledger
        from osprey_connectors import posture_store

        assert session_ledger.append(self.state_dir, self.key, transcript)
        posture_store.invalidate_cache()
        self.transcript = transcript


class CliSession:
    """An ``osprey chat`` launch whose CLI pointer names its conversation."""

    launch = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    terminal_session = "chat-ffffffff-ffff-4fff-8fff-ffffffffffff"

    def __init__(self, root):
        self.root = root
        self.transcript = ""

    @property
    def pointer(self):
        return self.root / "cli-transcripts" / self.launch

    def clear(self, transcript: str) -> None:
        """Rewrite the pointer to *transcript*, as the turn-state hook does on a ``/clear``."""
        self.pointer.parent.mkdir(parents=True, exist_ok=True)
        self.pointer.write_text(transcript + "\n", encoding="utf-8")
        self.transcript = transcript


@pytest.fixture
def web_session(tmp_path, monkeypatch) -> WebSession:
    """This process in a web terminal session, with a temp ledger bound.

    ``OSPREY_SESSION_ID`` is the session's key and ``OSPREY_CONTROL_CONTEXT_DIR``
    a temp state directory whose ledger maps the key to ``transcript``; call
    ``clear(new_id)`` to move it.
    """
    from osprey.stores import session_ledger
    from osprey_connectors import posture_store

    for name in _SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
    session = WebSession(state_dir)
    assert session_ledger.append(state_dir, session.key, session.transcript, session_ledger.EPOCH)
    monkeypatch.setenv("OSPREY_SESSION_ID", session.key)
    posture_store.invalidate_cache()
    yield session
    posture_store.invalidate_cache()


@pytest.fixture
def cli_session(tmp_path, monkeypatch) -> CliSession:
    """This process in an ``osprey chat`` launch, its pointer under ``tmp_path / "_agent_data"``.

    The root is :func:`workspace_dir`'s, so a store built on that fixture tags
    and lists from this pointer. ``CLAUDE_CODE_SESSION_ID`` keeps the launch's
    first id, as the harness's MCP children do; ``clear(new_id)`` moves only the
    pointer. An empty temp ledger is bound so nothing reads a real one.
    """
    from osprey_connectors import posture_store

    for name in _SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(state_dir))
    session = CliSession(tmp_path / "_agent_data")
    session.clear("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
    monkeypatch.setenv("OSPREY_TERMINAL_SESSION", session.terminal_session)
    monkeypatch.setenv("OSPREY_CLI_LAUNCH", session.launch)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", session.transcript)
    posture_store.invalidate_cache()
    yield session
    posture_store.invalidate_cache()
