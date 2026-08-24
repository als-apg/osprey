"""Guards on the Graph channel-finder server's wiring and startup sequence.

Two things about this server are load-bearing and neither is visible from the
tools themselves:

* **The wire name is ``channel-finder``.** Every consumer selects the paradigm
  by config key and then reaches the server by that name — the rendered
  ``.mcp.json`` key, the agent's ``mcp__channel-finder__*`` allowlist, the
  benchmark harness's substring filter. A rename here silently unhooks all
  three: nothing raises, the agent just has no channel tools.
* **Startup is the lean graph sequence.** It primes the config builder and
  initialises the graph server context, and stops. Calling
  ``initialize_workspace_singletons()`` would arm the artifact-activity
  listeners in a process that saves no artifacts, so the test below makes that
  call an outright failure rather than a review comment.
"""

import asyncio
from pathlib import Path

import pytest

from osprey.mcp_server.channel_finder_graph import server as cf_graph_server

TOOL_MODULES = ("capabilities", "example_queries", "get_schema", "read_cypher")


def _tool_names(mcp) -> set[str]:
    """Tool names registered on a FastMCP server, across FastMCP versions."""
    tools = mcp.get_tools() if hasattr(mcp, "get_tools") else mcp.list_tools()
    if asyncio.iscoroutine(tools):
        tools = asyncio.run(tools)
    if isinstance(tools, dict):
        return set(tools)
    return {t.name for t in tools}


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A minimal project with no ``services.graphdb`` block.

    ``create_server()`` primes the real ConfigBuilder, which reads a config.yml
    from the working directory, so the startup tests need one to exist.
    """
    config = tmp_path / "config.yml"
    config.write_text("facility:\n  name: Test Facility\n")
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    return tmp_path


@pytest.fixture
def sequence(monkeypatch):
    """Record the startup sequence and make the forbidden step fail loudly.

    ``initialize_workspace_singletons`` is patched to raise: nothing in
    ``create_server()`` calls it today, and this is what keeps that true.
    """
    import osprey.mcp_server.graph.server_context as graph_context
    import osprey.mcp_server.startup as startup

    calls: list[str] = []

    real_initialize_context = graph_context.initialize_server_context

    def _record_context():
        calls.append("server_context")
        return real_initialize_context()

    def _record_tools():
        calls.append("tool_imports")

    def _forbidden():
        raise AssertionError(
            "create_server() called initialize_workspace_singletons(); the graph "
            "channel-finder server saves no artifacts and must not arm the listeners"
        )

    monkeypatch.setattr(graph_context, "initialize_server_context", _record_context)
    monkeypatch.setattr(cf_graph_server, "_import_tools", _record_tools)
    monkeypatch.setattr(startup, "initialize_workspace_singletons", _forbidden)
    return calls


@pytest.mark.unit
def test_wire_name_is_channel_finder():
    """The paradigm ships under the shared channel-finder name, not a graph-specific one."""
    assert cf_graph_server.mcp.name == "channel-finder"


@pytest.mark.unit
def test_instructions_route_the_agent_through_the_read_path():
    """The agent is told to learn the shapes first and that results are bounded."""
    instructions = cf_graph_server.mcp.instructions
    assert instructions is not None
    for tool in ("example_queries", "get_schema", "read_cypher"):
        assert tool in instructions, f"instructions never name {tool}"
    assert "bounded" in instructions


@pytest.mark.unit
def test_create_server_returns_the_module_instance(sequence, project):
    """Tool modules register on the module-level ``mcp``; the factory must return that one."""
    assert cf_graph_server.create_server() is cf_graph_server.mcp


@pytest.mark.unit
def test_startup_initialises_the_context_before_importing_tools(sequence, project):
    """Context first, tools second — the tools read the store through the context."""

    cf_graph_server.create_server()

    assert sequence == ["server_context", "tool_imports"]


@pytest.mark.unit
def test_startup_never_initialises_workspace_singletons(sequence, project):
    """The forbidden step is armed to raise, and ``create_server()`` never trips it."""
    import osprey.mcp_server.startup as startup

    with pytest.raises(AssertionError):
        startup.initialize_workspace_singletons()  # the guard is live

    cf_graph_server.create_server()  # ... and the startup sequence does not reach it

    assert sequence == ["server_context", "tool_imports"]


@pytest.mark.unit
def test_startup_leaves_a_usable_graph_context(sequence, project):
    """A project with no ``services.graphdb`` block still starts — unconfigured, not broken."""
    from osprey.mcp_server.graph.server_context import get_server_context

    cf_graph_server.create_server()

    assert get_server_context().configured is False


@pytest.mark.unit
def test_entry_point_delegates_to_run_cf_main(monkeypatch):
    """``python -m …channel_finder_graph`` goes through the shared channel-finder entry."""
    from osprey.mcp_server.channel_finder_graph import __main__ as entry

    seen: list[str] = []
    monkeypatch.setattr(entry, "run_cf_main", seen.append)

    entry.main()

    assert seen == ["osprey.mcp_server.channel_finder_graph.server"]


_TOOLS_DIR = Path(cf_graph_server.__file__).parent / "tools"
_MISSING = [name for name in TOOL_MODULES if not (_TOOLS_DIR / f"{name}.py").exists()]


@pytest.mark.unit
@pytest.mark.skipif(bool(_MISSING), reason=f"tool modules not written yet: {_MISSING}")
def test_create_server_registers_every_tool(project):
    """The real startup registers all four tools — and pulls in ``graph.server`` doing it.

    ``read_cypher`` and ``get_schema`` are re-registrations of the main-agent
    graph server's callables, so importing them imports that module too. This
    asserts the transitive import is harmless: our four tools land here, and
    the graph server's own registrations do not leak onto this instance.
    """
    import sys

    server = cf_graph_server.create_server()

    assert _tool_names(server) == set(TOOL_MODULES)
    assert "osprey.mcp_server.graph.server" in sys.modules
