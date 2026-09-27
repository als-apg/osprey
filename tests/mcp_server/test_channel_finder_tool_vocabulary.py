"""Guards on the channel-finder MCP servers' agent-facing vocabulary.

The pipeline variants are interchangeable: the registry selects one by
config key, so a shared tool name across them *asserts* substitutability. It
used to lie — both the in-context and middle-layer variants registered
``query_channels``, one taking a natural-language question and the other an
executable SQL statement, so passing either argument to the other failed. The
names now state their contracts (``ask_channels`` / ``run_sql``).

These tests also pin **allowlist parity**, the failure mode a rename hits that
nothing reports: a tool renamed on the server but left under its old name in
``registry.mcp`` does not raise. The rendered permission list simply names a
tool that no longer exists, and the agent quietly loses the capability.
"""

import asyncio
import importlib
import pkgutil
from typing import Any

import pytest

_VARIANTS = ("in_context", "middle_layer", "hierarchical", "graph")


def _registered_tools(package: str) -> dict[str, Any]:
    """Registered tools of a FastMCP package, keyed by tool name.

    Serves any FastMCP package laid out as ``<package>.server`` +
    ``<package>.tools``. Each tool module registers itself via ``@mcp.tool()``
    at import time, so importing the modules is enough. ``create_server()`` is
    deliberately not used: it also initialises a live context.
    """
    tools_pkg = importlib.import_module(f"{package}.tools")
    for module in pkgutil.iter_modules(tools_pkg.__path__):
        importlib.import_module(f"{package}.tools.{module.name}")
    server = importlib.import_module(f"{package}.server")
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


@pytest.fixture(scope="module")
def variant_tools() -> dict[str, dict[str, Any]]:
    """Registered tools per pipeline variant, keyed by variant and then by tool name."""
    registered: dict[str, dict[str, Any]] = {}
    for variant in _VARIANTS:
        tools = _registered_tools(f"osprey.mcp_server.channel_finder_{variant}")
        assert tools, f"no tools registered for {variant}; the fixture is not registering"
        registered[variant] = tools
    return registered


def test_no_variant_still_registers_query_channels(variant_tools):
    """The one name that carried two incompatible contracts is gone."""
    offenders = {v: sorted(t) for v, t in variant_tools.items() if "query_channels" in t}
    assert offenders == {}, f"query_channels still registered: {offenders}"


def test_the_two_contracts_have_distinct_names(variant_tools):
    """A natural-language question and a SQL statement are not one tool."""
    assert "ask_channels" in variant_tools["in_context"], variant_tools["in_context"]
    assert "run_sql" in variant_tools["middle_layer"], variant_tools["middle_layer"]


@pytest.mark.parametrize("variant", _VARIANTS)
def test_registry_allowlist_names_only_tools_the_variant_registers(variant, variant_tools):
    """Allowlist parity, per pipeline variant.

    A stale name here renders an allow rule for a tool that does not exist, so
    the real tool falls through to a prompt or a denial. Nothing errors.
    """
    from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE

    listed = set(CHANNEL_FINDER_TOOLS_BY_PIPELINE[variant])
    stale = sorted(listed - set(variant_tools[variant]))
    assert stale == [], f"registry.mcp lists {variant} tools that are not registered: {stale}"
