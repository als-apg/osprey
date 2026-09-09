"""Every framework MCP tool must be classified by the registry.

The registry is where a tool is told apart from a write: ``permissions_allow``
and ``permissions_ask`` decide what the rendered ``settings.json`` auto-approves
and what it prompts for, and ``hooks_pre`` decides which gate runs first.
``agent_runner.write_tools`` derives the headless read-only block list from those
same lists, so a tool in none of them is a tool the headless path will happily
call — which is exactly how ``entry_publish`` shipped able to write through to a
facility logbook with no gate at all.

This test walks the *live* servers rather than any list of names, so a tool
added to a ``tools/`` module and forgotten in the registry fails here.

There are no known gaps left: the baseline is empty and the assertion is
equality, so a tool that stops being classified fails here and a tool that
starts being classified cannot be quietly left in a baseline.
"""

import asyncio
import importlib
import pkgutil

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS

pytestmark = pytest.mark.unit


#: Tools registered by a framework server but named nowhere in its registry
#: entry. Empty: the nine workspace tools this baseline used to carry are now
#: classified — the rail verbs and ``register_panel`` ask, ``artifact_pin`` and
#: the five lattice tools are allowed. Keep it empty; a tool belongs in a
#: permission list, not in a list nobody reads.
_UNCLASSIFIED_BASELINE: dict[str, set[str]] = {}


def _registered_tool_names(module: str) -> set[str]:
    """Tool names the live FastMCP singleton of *module* registers.

    Importing the server module and every module under its ``tools`` package is
    what runs the ``@mcp.tool()`` decorators; ``create_server()`` is deliberately
    not called, since it does heavy config and workspace startup. A server whose
    tools live in ``server.py`` itself has no ``tools`` package, which is not an
    error.
    """
    server_mod = importlib.import_module(f"{module}.server")
    try:
        tools_pkg = importlib.import_module(f"{module}.tools")
    except ModuleNotFoundError:
        tools_pkg = None
    if tools_pkg is not None:
        for found in pkgutil.iter_modules(tools_pkg.__path__):
            importlib.import_module(f"{module}.tools.{found.name}")
    # ``list_tools`` is FastMCP's public listing; the private
    # ``_list_tools`` an upgrade may rename would turn this safety guard
    # into a collection error instead of a verdict.
    tools = asyncio.run(server_mod.mcp.list_tools())
    return {getattr(tool, "name", tool) for tool in tools}


def _classified_tool_names(server) -> set[str]:
    """Every tool name this server's registry entry accounts for.

    A tool counts as classified when a permission list names it *or* a
    ``hooks_pre`` matcher does: ``channel_read`` and ``archiver_read`` sit in no
    permission list on purpose — they are governed by the approval hook's own
    per-tool policy — and that is a decision, not a gap.
    """
    names = {
        *server.permissions_allow,
        *server.permissions_ask,
        *server.fixed_allow,
        *server.fixed_ask,
    }
    prefix = f"mcp__{server.name}__"
    for rule in server.hooks_pre:
        if rule.matcher.startswith(prefix):
            names.add(rule.matcher[len(prefix) :])
    return names


_WALKABLE = sorted(name for name, s in FRAMEWORK_SERVERS.items() if "{" not in s.module)


def test_every_framework_server_is_walkable_or_templated():
    """The walk must not go quiet because a module path stopped resolving."""
    assert set(_WALKABLE) | {"channel-finder"} == set(FRAMEWORK_SERVERS)


@pytest.mark.parametrize("server_name", _WALKABLE)
def test_registered_tools_are_classified_by_the_registry(server_name):
    server = FRAMEWORK_SERVERS[server_name]
    registered = _registered_tool_names(server.module)
    assert registered, f"{server_name} registered no tools — the walk found nothing to check"

    unclassified = registered - _classified_tool_names(server)
    expected = _UNCLASSIFIED_BASELINE.get(server_name, set())

    assert unclassified == expected, (
        f"{server_name}: tools the registry does not classify changed.\n"
        f"  newly unclassified: {sorted(unclassified - expected)}\n"
        f"  newly classified (drop them from the baseline): {sorted(expected - unclassified)}"
    )


def test_ariel_gates_both_halves_of_a_logbook_write():
    """entry_create drafts the entry; entry_publish is what reaches the logbook.

    Gating only the draft prompts for the half that never leaves the deployment.
    Pinned by name as well as by the walk above: parity alone would stay green
    if both tools lost their gate together.
    """
    ariel = FRAMEWORK_SERVERS["ariel"]

    assert "entry_create" in ariel.permissions_ask
    assert "entry_publish" in ariel.permissions_ask

    gated = {rule.matcher for rule in ariel.hooks_pre}
    assert "mcp__ariel__entry_create" in gated
    assert "mcp__ariel__entry_publish" in gated
