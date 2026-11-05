"""The per-server tool inventory on ``architecture/mcp-servers`` matches the code.

``architecture/mcp-servers.rst`` lists, server by server, the tools each
in-tree MCP server registers. That list is kept by hand, so it drifts without a
sound: a tool ships and the page keeps describing the old surface, or a tool is
renamed and the page keeps the dead name. :mod:`tests.docs.test_panel_tool_docs`
and :mod:`tests.docs.test_phoebus_tool_docs` hold two slices of the page; this
guard holds all of it, in both directions:

* every server package under ``src/osprey/mcp_server/`` with a ``server.py``
  has a section on the page, introduced by its ``Package:`` line;
* every tool the server can register has a bullet in its section;
* every tool a section's bullets name is one the server can register.

"Can register" is wider than what one process lists. The live FastMCP
singleton is read after its tool modules are imported, with
``list_tools(run_middleware=False)`` so a tool a middleware hides from
``tools/list`` (``phoebus_drive`` outside ``read_write``) still counts. A tool a
server registers only on some renders is added by a ``register(server)``
function in its tools package, called by ``create_server()`` when the render
qualifies (the pyAML tools of ``python_executor``); every such function is
called here on a throwaway FastMCP, and the registry's
``ServerDefinition.optional_tools`` must all be found that way.
``create_server()`` itself is not called: it does config and workspace startup.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

from osprey.build.modes import VALID_CHANNEL_FINDER_MODES
from osprey.registry.mcp import FRAMEWORK_SERVERS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _REPO_ROOT / "docs" / "source" / "architecture" / "mcp-servers.rst"
_SERVER_ROOT = _REPO_ROOT / "src" / "osprey" / "mcp_server"

#: ``Package: ``osprey.mcp_server.x``` opens a server's section on the page.
_PACKAGE_LINE = re.compile(r"^Package: ``(osprey\.mcp_server\.[a-z_]+)``\s*$", re.M)

#: A tool bullet: ``- ``name`` -- description``.
_TOOL_BULLET = re.compile(r"^- ``([a-z_]+)`` --", re.M)


def _server_modules() -> list[str]:
    return sorted(
        f"osprey.mcp_server.{path.parent.name}" for path in _SERVER_ROOT.glob("*/server.py")
    )


def _names(server) -> set[str]:
    tools = asyncio.run(server.list_tools(run_middleware=False))
    return {tool.name for tool in tools}


@functools.cache
def _registrable_tools(module: str) -> frozenset[str]:
    """Every tool *module*'s server registers on any render."""
    from fastmcp import FastMCP

    server_mod = importlib.import_module(f"{module}.server")
    registrars = []
    try:
        tools_pkg = importlib.import_module(f"{module}.tools")
    except ModuleNotFoundError:
        tools_pkg = None
    if tools_pkg is not None:
        for found in pkgutil.iter_modules(tools_pkg.__path__):
            tool_mod = importlib.import_module(f"{module}.tools.{found.name}")
            register = getattr(tool_mod, "register", None)
            if callable(register) and getattr(register, "__module__", None) == tool_mod.__name__:
                registrars.append(register)

    names = _names(server_mod.mcp)
    if registrars:
        probe = FastMCP(f"{module}-inventory-probe")
        for register in registrars:
            register(probe)
        names |= _names(probe)
    return frozenset(names)


def _sections() -> dict[str, str]:
    """Each ``Package:`` module on the page, mapped to its section's text."""
    text = _PAGE.read_text(encoding="utf-8")
    starts = list(_PACKAGE_LINE.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        sections[match.group(1)] = text[match.end() : end]
    return sections


def test_the_walk_sees_hidden_and_render_gated_tools():
    """Negative control: the two ways a tool escapes a plain ``tools/list``."""
    assert "phoebus_drive" in _registrable_tools("osprey.mcp_server.phoebus")
    assert {"pyaml_measure", "model_response_matrix"} <= _registrable_tools(
        "osprey.mcp_server.python_executor"
    )


def test_every_registry_server_module_is_a_server_package():
    """The registry's stdio servers are the packages this guard walks."""
    modules = set(_server_modules())
    for name, server in FRAMEWORK_SERVERS.items():
        if not server.module:
            continue
        if "{" in server.module:
            expected = {
                server.module.format(channel_finder_pipeline=mode)
                for mode in VALID_CHANNEL_FINDER_MODES
            }
        else:
            expected = {server.module}
        assert expected <= modules, f"{name}: {sorted(expected - modules)} has no server.py"


@pytest.mark.parametrize(
    "name", sorted(n for n, s in FRAMEWORK_SERVERS.items() if s.optional_tools)
)
def test_every_optional_tool_is_found_by_the_walk(name):
    server = FRAMEWORK_SERVERS[name]
    missing = set(server.optional_tools) - _registrable_tools(server.module)
    assert not missing, f"{name}: optional tools the walk does not reach: {sorted(missing)}"


def test_every_server_package_has_a_section():
    missing = set(_server_modules()) - set(_sections())
    assert not missing, f"{_PAGE.name} has no 'Package:' section for {sorted(missing)}"


def test_every_section_names_a_server_package():
    unknown = set(_sections()) - set(_server_modules())
    assert not unknown, f"{_PAGE.name} has sections for no server package: {sorted(unknown)}"


def _unlisted(module: str, section: str) -> set[str]:
    """Tools *module* can register that *section* gives no bullet of their own."""
    return set(_registrable_tools(module)) - set(_TOOL_BULLET.findall(section))


def test_a_tool_named_only_in_prose_is_reported():
    """Negative control: a deleted bullet fails even where the intro names the tool."""
    module = "osprey.mcp_server.bluesky"
    section = _sections()[module]
    pruned = re.sub(r"^- ``queue_add`` --.*$", "", section, flags=re.M)
    assert "``queue_add``" in pruned, "the control needs a prose mention to stay meaningful"
    assert _unlisted(module, pruned) == {"queue_add"}


@pytest.mark.parametrize("module", _server_modules())
def test_every_registered_tool_has_a_bullet_in_its_section(module):
    missing = _unlisted(module, _sections().get(module, ""))
    assert not missing, f"{_PAGE.name}, section {module}, has no bullet for {sorted(missing)}"


@pytest.mark.parametrize("module", _server_modules())
def test_every_listed_tool_is_registered(module):
    section = _sections().get(module, "")
    listed = set(_TOOL_BULLET.findall(section))
    assert listed, f"{_PAGE.name}, section {module}, lists no tool bullets"
    stale = listed - _registrable_tools(module)
    assert not stale, f"{_PAGE.name}, section {module}, lists unregistered tools {sorted(stale)}"
