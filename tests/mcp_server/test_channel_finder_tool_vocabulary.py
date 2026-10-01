"""Guards on the agent-facing vocabulary of the MCP servers.

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

The tool descriptions and input-schema descriptions every MCP server offers an
agent, and the ARIEL search descriptors, are written for any control system.
They name no protocol (``PV``, ``EPICS``) and no demo address prefix (``SR:``),
and they lose ``RATCHET_WORD``. The words and the texts that still carry them
live in :mod:`tests._vocabulary`: ``PENDING_REWORDING`` until the wording
changes, ``RATCHET_PENDING`` under the file ratchet's stage tags.
"""

import asyncio
import importlib
import importlib.util
import pkgutil
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests._vocabulary import (
    PENDING_REWORDING,
    PROTOCOL_WORDS,
    RATCHET_PENDING,
    RATCHET_WORD,
    TextKey,
)
from tests.facility.test_word_ratchet import violations

_VARIANTS = ("in_context", "middle_layer", "hierarchical", "graph")

#: Every package under ``osprey.mcp_server`` that defines a FastMCP server.
MCP_SERVERS: tuple[str, ...] = (
    "ariel",
    "bluesky",
    "channel_finder_graph",
    "channel_finder_hierarchical",
    "channel_finder_in_context",
    "channel_finder_middle_layer",
    "control_system",
    "facility_knowledge",
    "graph",
    "health",
    "phoebus",
    "python_executor",
    "workspace",
)

SOURCES = MCP_SERVERS + ("ariel_search",)


def _registered_tools(package: str) -> dict[str, Any]:
    """Registered tools of a FastMCP package, keyed by tool name.

    Serves any FastMCP package laid out as ``<package>.server``, with its tools
    in the server module or under ``<package>.tools``. Each tool module
    registers itself via ``@mcp.tool()`` at import time, so importing the
    modules is enough. ``create_server()`` is deliberately not used: it also
    initialises a live context.
    """
    server = importlib.import_module(f"{package}.server")
    if importlib.util.find_spec(f"{package}.tools") is not None:
        tools_pkg = importlib.import_module(f"{package}.tools")
        for module in pkgutil.walk_packages(tools_pkg.__path__, f"{package}.tools."):
            importlib.import_module(module.name)
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


@pytest.fixture(scope="module")
def server_tools() -> dict[str, dict[str, Any]]:
    """Registered tools per MCP server, keyed by server and then by tool name."""
    registered: dict[str, dict[str, Any]] = {}
    for server in MCP_SERVERS:
        tools = _registered_tools(f"osprey.mcp_server.{server}")
        assert tools, f"no tools registered for {server}; the fixture is not registering"
        registered[server] = tools
    return registered


@pytest.fixture(scope="module")
def variant_tools(server_tools) -> dict[str, dict[str, Any]]:
    """Registered tools per pipeline variant, keyed by variant and then by tool name."""
    return {variant: server_tools[f"channel_finder_{variant}"] for variant in _VARIANTS}


def test_every_mcp_server_is_guarded():
    """A server package the guard does not list would offer unread text."""
    import osprey.mcp_server

    root = Path(osprey.mcp_server.__path__[0])
    assert sorted(path.parent.name for path in root.glob("*/server.py")) == list(MCP_SERVERS)


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
def agent_facing_texts(server_tools, ariel_search_descriptors) -> dict[TextKey, str]:
    """Every text an agent reads before calling a tool, keyed (source, tool, field)."""
    texts: dict[TextKey, str] = {}
    for server, tools in server_tools.items():
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


def protocol_offenders(
    texts: Mapping[TextKey, str], pending: frozenset[TextKey] = PENDING_REWORDING
) -> dict[TextKey, list[str]]:
    """The protocol words each text names, for the texts no entry excuses."""
    return {
        key: words
        for key, text in texts.items()
        if key not in pending and (words := PROTOCOL_WORDS.findall(text))
    }


def stale_rewording_entries(
    texts: Mapping[TextKey, str], pending: frozenset[TextKey] = PENDING_REWORDING
) -> list[TextKey]:
    """The entries whose text no longer names a protocol word."""
    return sorted(key for key in pending if not PROTOCOL_WORDS.search(texts.get(key, "")))


def ratchet_violations(
    texts: Mapping[TextKey, str], pending: Mapping[TextKey, str] = RATCHET_PENDING
) -> list[str]:
    """What the file ratchet's rule refuses, over texts instead of files."""
    hits = {" ".join(key) for key, text in texts.items() if re.search(RATCHET_WORD, text)}
    return violations(hits, {" ".join(key): tag for key, tag in pending.items()})


@pytest.mark.parametrize("source", SOURCES)
def test_agent_facing_text_names_no_protocol_word(source, agent_facing_texts):
    """The text an agent reads before calling a tool names no protocol word or demo address."""
    offenders = protocol_offenders(
        {key: text for key, text in agent_facing_texts.items() if key[0] == source}
    )
    assert offenders == {}, (
        f"{offenders} name a protocol word or demo address; "
        "the text an agent reads should call it a channel or channel address"
    )


def test_pending_rewording_entries_still_name_a_protocol_word(agent_facing_texts):
    """Every baseline entry still excuses a real hit."""
    stale = stale_rewording_entries(agent_facing_texts)
    assert stale == [], (
        f"{stale} no longer name a protocol word; delete them from PENDING_REWORDING"
    )


def test_agent_facing_text_carries_the_ratchet_word_only_where_listed(agent_facing_texts):
    """A tool text carrying the word has an entry; an entry's text still carries it."""
    assert ratchet_violations(agent_facing_texts) == []


def test_ratchet_word_in_an_unlisted_text_is_refused():
    texts = {("server", "tool", "description"): "the storage ring", ("server", "b", "x"): "strings"}
    assert ratchet_violations(texts, {}) == [
        "server tool description: carries the word and has no allowlist entry"
    ]


def test_protocol_word_in_an_unlisted_text_is_refused():
    key = ("server", "tool", "description")
    assert protocol_offenders({key: "read the PV"}, frozenset()) == {key: ["PV"]}
    assert protocol_offenders({key: "read the PV"}, frozenset({key})) == {}
    assert stale_rewording_entries({key: "read the channel"}, frozenset({key})) == [key]


@pytest.mark.parametrize("sample", ["the PV for X", "PVs", "EPICS gateway", "SR:C01-MG{PS:QF}"])
def test_protocol_words_pattern_matches(sample):
    assert PROTOCOL_WORDS.search(sample)


@pytest.mark.parametrize("sample", ["SR", "Storage Ring", "PVT", "UPV", "epics", "channel address"])
def test_protocol_words_pattern_ignores(sample):
    assert not PROTOCOL_WORDS.search(sample)
