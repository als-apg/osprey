"""Unit tests for the graph MCP server skeleton and its ``capabilities`` tool.

``create_server()`` is exercised with the config priming and the server-context
initialisation replaced by stubs, so no config file is read and no graph store is
dialled. The ``capabilities`` tool is a static manifest, so its payload is
asserted directly through the unwrapped function.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from tests.mcp_server.conftest import get_tool_fn, registered_tool_names

pytestmark = pytest.mark.unit

_SERVER_CONTEXT_MODULE = "osprey.mcp_server.graph.server_context"

#: The full tool vocabulary the graph server advertises. Kept as a literal so a
#: renamed or dropped tool fails here rather than silently shrinking the manifest.
_EXPECTED_TOOLS = {"read_cypher", "get_schema", "example_queries", "capabilities"}

#: Independent marker list (do not import the production constant) mirroring the
#: registry-wide destructive-name scan in tests/agent_runner/test_write_tools.py.
_DESTRUCTIVE_MARKERS = ("delete", "remove", "clear", "wipe", "purge", "destroy")


#: Probe run in a fresh interpreter: stub the server context (so nothing dials a
#: store), call ``create_server()``, and print the registered tool names.
_REGISTRATION_PROBE = """
import asyncio, json

import osprey.mcp_server.graph.server_context as graph_ctx

# create_server() resolves this attribute at call time, so neither config nor a
# store is touched. Nothing is dialed either way — the driver is lazy — but the
# probe is about tool registration, not about context initialisation.
graph_ctx.initialize_server_context = lambda: None

from osprey.mcp_server.graph.server import create_server

mcp = create_server()
tools = mcp.get_tools() if hasattr(mcp, "get_tools") else mcp.list_tools()
if asyncio.iscoroutine(tools):
    tools = asyncio.run(tools)
names = list(tools) if isinstance(tools, dict) else [t.name for t in tools]
print("TOOLS " + json.dumps(sorted(names)))
"""


def _registered_tools_in_subprocess() -> list[str]:
    """Tool names ``create_server()`` registers in a pristine interpreter."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k not in {"OSPREY_CONFIG", "CONFIG_FILE"}}
    proc = subprocess.run(
        [sys.executable, "-c", _REGISTRATION_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    marker = [line for line in proc.stdout.splitlines() if line.startswith("TOOLS ")]
    assert marker, f"probe printed no tool list:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(marker[-1].removeprefix("TOOLS "))


@pytest.fixture(autouse=True)
def _reset_graph_context():
    """Reset the graph server-context singleton when that module exists."""
    yield
    try:
        module = importlib.import_module(_SERVER_CONTEXT_MODULE)
    except ImportError:
        return
    reset = getattr(module, "reset_server_context", None)
    if reset is not None:
        reset()


@pytest.fixture()
def stubbed_startup(monkeypatch):
    """Neutralise config priming and server-context init; record the calls.

    ``server_context`` is written by a sibling task. When the module is absent a
    stub is installed under its import path; when it is present its real
    ``initialize_server_context`` is replaced. Either way ``create_server()``
    touches neither config nor a graph store.
    """
    calls: dict[str, int] = {"prime": 0, "context": 0}

    import osprey.mcp_server.startup as startup

    monkeypatch.setattr(
        startup, "prime_config_builder", lambda: calls.__setitem__("prime", calls["prime"] + 1)
    )

    try:
        module = importlib.import_module(_SERVER_CONTEXT_MODULE)
    except ImportError:
        module = types.ModuleType(_SERVER_CONTEXT_MODULE)
        module.reset_server_context = lambda: None
        monkeypatch.setitem(sys.modules, _SERVER_CONTEXT_MODULE, module)

    monkeypatch.setattr(
        module,
        "initialize_server_context",
        lambda: calls.__setitem__("context", calls["context"] + 1),
        raising=False,
    )
    return calls


class TestCapabilitiesServerCreation:
    """``create_server()`` wiring — the skeleton every graph tool hangs off."""

    def test_capabilities_server_creation_returns_module_singleton(self, stubbed_startup):
        from osprey.mcp_server.graph import server as graph_server

        assert graph_server.create_server() is graph_server.mcp
        assert stubbed_startup == {"prime": 1, "context": 1}

    def test_capabilities_registered_after_server_creation(self, stubbed_startup):
        from osprey.mcp_server.graph import server as graph_server

        names = registered_tool_names(graph_server.create_server())

        assert "capabilities" in names, f"capabilities not registered; got {sorted(names)}"

    def test_create_server_registers_all_tools(self):
        """Every advertised tool is imported by ``create_server()``.

        A tool module that is never imported never runs its ``@mcp.tool()``
        decorator, so the tool is simply absent at runtime while the manifest and
        the registry allowlist both still name it — nothing errors, the agent
        just cannot call it.

        Run in a fresh interpreter: ``mcp`` is a module singleton, so a sibling
        test that imported a tool module directly would leave that tool
        registered and make the assertion vacuous in-process.
        """
        names = set(_registered_tools_in_subprocess())

        missing = sorted(_EXPECTED_TOOLS - names)
        assert missing == [], (
            f"create_server() does not register {missing}; add the tool module to the "
            f"tool_imports block in create_server(). Registered: {sorted(names)}"
        )

    def test_capabilities_server_has_instructions(self):
        from osprey.mcp_server.graph.server import mcp

        assert mcp.name == "graph"
        assert "read-only" in (mcp.instructions or "").lower()


class TestCapabilitiesPayload:
    """The static manifest the agent reads before planning a graph query."""

    @pytest.fixture()
    def payload(self) -> dict:
        from osprey.mcp_server.graph.tools.capabilities import capabilities

        return json.loads(get_tool_fn(capabilities)())

    def test_capabilities_payload_has_required_keys(self, payload):
        assert set(payload) == {"server", "description", "tools", "notes"}
        assert payload["server"] == "graph"
        assert payload["description"]
        assert payload["notes"]

    def test_capabilities_payload_lists_every_tool(self, payload):
        assert {tool["name"] for tool in payload["tools"]} == _EXPECTED_TOOLS
        for tool in payload["tools"]:
            assert tool["purpose"].strip(), f"{tool['name']} has no purpose line"

    def test_capabilities_payload_names_the_bounding_config_keys(self, payload):
        notes = " ".join(payload["notes"])

        assert "services.graphdb.query_max_rows" in notes
        assert "services.graphdb.query_timeout_s" in notes

    def test_capabilities_payload_names_the_seed_command(self, payload):
        from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND

        assert GRAPHDB_SEED_COMMAND in " ".join(payload["notes"])

    def test_capabilities_tool_names_are_not_destructive(self, payload):
        for tool in payload["tools"]:
            offenders = [m for m in _DESTRUCTIVE_MARKERS if m in tool["name"].lower()]
            assert offenders == [], (
                f"{tool['name']} reads as destructive ({offenders}); a read-only "
                "server's tools must never be auto-allowed under such a name"
            )
