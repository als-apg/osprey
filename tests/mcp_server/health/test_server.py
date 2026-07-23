"""Tests for the health MCP server scaffold.

Covers the server anatomy: ``create_server()`` wires the two tiered tools and
returns a FastMCP server exposing exactly ``health_check`` and
``health_check_full``, and ``__main__`` follows the shared ``run_mcp_server``
idiom.

The sibling ``server_context`` module is built concurrently; these tests stub
it in ``sys.modules`` so they pass regardless of the sibling's timing while
remaining correct once the real module lands.
"""

import importlib
import inspect
import sys
import types

import pytest


@pytest.fixture
def stub_server_context(monkeypatch):
    """Inject a stub ``health.server_context`` module and track init calls.

    ``create_server()`` imports ``initialize_server_context`` lazily, so a stub
    installed here is picked up whether or not the real module exists yet. The
    stub records that initialization was invoked.
    """
    state = {"initialized": 0}

    module = types.ModuleType("osprey.mcp_server.health.server_context")

    def initialize_server_context():
        state["initialized"] += 1
        return object()

    def get_server_context():
        return object()

    def reset_server_context():
        return None

    module.initialize_server_context = initialize_server_context
    module.get_server_context = get_server_context
    module.reset_server_context = reset_server_context

    monkeypatch.setitem(sys.modules, "osprey.mcp_server.health.server_context", module)
    return state


async def _tool_names(server) -> set[str]:
    """Return the set of registered tool names on a FastMCP server."""
    tools = await server._list_tools()
    return {getattr(t, "name", t) for t in tools}


@pytest.mark.unit
async def test_create_server_exposes_exactly_two_tools(stub_server_context):
    """create_server() returns a server exposing exactly the two tiered tools."""
    from osprey.mcp_server.health.server import create_server

    server = create_server()
    assert server is not None

    names = await _tool_names(server)
    assert names == {"health_check", "health_check_full"}


@pytest.mark.unit
async def test_create_server_initializes_server_context(stub_server_context):
    """create_server() drives initialize_server_context via the lazy import."""
    from osprey.mcp_server.health.server import create_server

    create_server()
    assert stub_server_context["initialized"] >= 1


@pytest.mark.unit
async def test_create_server_does_not_blow_up_when_context_stubbed(stub_server_context):
    """create_server() completes cleanly when server_context is stubbed."""
    from osprey.mcp_server.health.server import create_server

    # Should not raise even though the real server_context may not exist yet.
    server = create_server()
    assert server is not None


@pytest.mark.unit
async def test_tool_signatures_are_categories_only(stub_server_context):
    """Both tools carry the final ``(categories: list[str] | None = None)`` signature."""
    from osprey.mcp_server.health.tools.health_check import health_check
    from osprey.mcp_server.health.tools.health_check_full import health_check_full

    for tool in (health_check, health_check_full):
        fn = getattr(tool, "fn", tool)
        params = inspect.signature(fn).parameters
        assert list(params) == ["categories"]
        assert params["categories"].default is None


@pytest.mark.unit
def test_main_uses_run_mcp_server_idiom():
    """__main__ wires run_mcp_server against the health server module string."""
    main_mod = importlib.import_module("osprey.mcp_server.health.__main__")
    source = inspect.getsource(main_mod.main)
    assert 'run_mcp_server("osprey.mcp_server.health.server")' in source
