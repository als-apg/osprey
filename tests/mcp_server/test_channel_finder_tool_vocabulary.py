"""Guards on the agent-facing vocabulary of the channel-finder and ARIEL MCP servers.

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

The tool descriptions and input-schema descriptions these servers offer an
agent, and the ARIEL search descriptors, are written for any control system.
They name no protocol (``PV``, ``EPICS``) and no demo address prefix (``SR:``).
The phrases that still do are listed in ``PENDING_REWORDING`` until their
wording changes.
"""

import asyncio
import importlib
import pkgutil
import re
from collections.abc import Iterator
from typing import Any

import pytest

_VARIANTS = ("in_context", "middle_layer", "hierarchical", "graph")

# Case-sensitive on purpose: lowercase ``epics`` names a connector type in config
# text, and the bare demo ring name ``SR`` without a colon is not a match.
PROTOCOL_WORDS = re.compile(r"\bPVs?\b|\bEPICS\b|SR:")

# Phrases that still name a protocol word; an entry leaves when its wording changes.
PENDING_REWORDING: frozenset[tuple[str, str, str]] = frozenset(
    {
        (
            "channel_finder_in_context",
            "ask_channels",
            "inputSchema/properties/question/description",
        ),
        ("channel_finder_graph", "read_cypher", "description"),
        ("channel_finder_graph", "search_channels", "description"),
        ("ariel", "keyword_search", "description"),
        ("ariel_search", "keyword_search", "description"),
    }
)

SERVERS = tuple(f"channel_finder_{v}" for v in _VARIANTS) + ("ariel", "ariel_search")


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


@pytest.fixture(scope="module")
def ariel_tools() -> dict[str, Any]:
    """Registered tools of the ARIEL MCP server, keyed by tool name."""
    tools = _registered_tools("osprey.mcp_server.ariel")
    assert tools, "no tools registered for ariel; the fixture is not registering"
    return tools


@pytest.fixture(scope="module")
def ariel_search_descriptors() -> dict[str, Any]:
    """ARIEL search tool descriptors, keyed by descriptor name."""
    search_pkg = importlib.import_module("osprey.services.ariel_search.search")
    descriptors: dict[str, Any] = {}
    for module_info in pkgutil.iter_modules(search_pkg.__path__):
        module = importlib.import_module(f"{search_pkg.__name__}.{module_info.name}")
        get_descriptor = getattr(module, "get_tool_descriptor", None)
        if callable(get_descriptor):
            descriptor = get_descriptor()
            descriptors[descriptor.name] = descriptor
    assert descriptors, (
        "no ARIEL search module exports get_tool_descriptor(); the walk is not finding them"
    )
    return descriptors


def _schema_descriptions(node: Any, pointer: str) -> Iterator[tuple[str, str]]:
    """Every ``description`` string in a JSON schema, with its pointer.

    A key named ``description`` whose value is not a string (a parameter called
    ``description``) is recursed into, not read.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                yield f"{pointer}/description", value
            else:
                yield from _schema_descriptions(value, f"{pointer}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _schema_descriptions(value, f"{pointer}/{index}")


@pytest.fixture(scope="module")
def agent_facing_texts(
    variant_tools, ariel_tools, ariel_search_descriptors
) -> dict[tuple[str, str, str], str]:
    """Every text an agent reads before calling a tool, keyed (server, tool, field)."""
    servers = {f"channel_finder_{v}": tools for v, tools in variant_tools.items()}
    servers["ariel"] = ariel_tools
    texts: dict[tuple[str, str, str], str] = {}
    for server, tools in servers.items():
        for name, tool in tools.items():
            texts[(server, name, "description")] = tool.description or ""
            for pointer, text in _schema_descriptions(tool.parameters, "inputSchema"):
                texts[(server, name, pointer)] = text
    for name, descriptor in ariel_search_descriptors.items():
        texts[("ariel_search", name, "description")] = descriptor.description
        schema = descriptor.args_schema.model_json_schema()
        for pointer, text in _schema_descriptions(schema, "args_schema"):
            texts[("ariel_search", name, pointer)] = text
    return texts


@pytest.mark.parametrize("server", SERVERS)
def test_agent_facing_text_names_no_protocol_word(server, agent_facing_texts):
    """The text an agent reads before calling a tool names no protocol word or demo address."""
    offenders = {
        key: words
        for key, text in agent_facing_texts.items()
        if key[0] == server
        and key not in PENDING_REWORDING
        and (words := PROTOCOL_WORDS.findall(text))
    }
    assert offenders == {}, (
        f"{offenders} name a protocol word or demo address; "
        "the text an agent reads should call it a channel or channel address"
    )


def test_pending_rewording_entries_still_name_a_protocol_word(agent_facing_texts):
    """Every baseline entry still excuses a real hit."""
    stale = sorted(
        key
        for key in PENDING_REWORDING
        if not PROTOCOL_WORDS.search(agent_facing_texts.get(key, ""))
    )
    assert stale == [], (
        f"{stale} no longer name a protocol word; delete them from PENDING_REWORDING"
    )


@pytest.mark.parametrize("sample", ["the PV for X", "PVs", "EPICS gateway", "SR:C01-MG{PS:QF}"])
def test_protocol_words_pattern_matches(sample):
    assert PROTOCOL_WORDS.search(sample)


@pytest.mark.parametrize("sample", ["SR", "Storage Ring", "PVT", "UPV", "epics", "channel address"])
def test_protocol_words_pattern_ignores(sample):
    assert not PROTOCOL_WORDS.search(sample)
