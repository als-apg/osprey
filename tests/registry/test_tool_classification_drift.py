"""Every framework MCP tool must be classified by the registry.

The registry is where a tool is told apart from a write: ``permissions_allow``
and ``permissions_ask`` decide what the rendered ``settings.json`` auto-approves
and what it prompts for, and ``hooks_pre`` decides which gate runs first.
``agent_runner.write_tools`` derives the headless read-only block list from those
same lists, so a tool in none of them is a tool the headless path will happily
call — which is exactly how ``entry_publish`` shipped able to write through to a
facility logbook with no gate at all.

This test walks the *live* servers rather than any list of names, so a tool
added to a ``tools/`` module and forgotten in the registry fails here. The walk
reaches a server this process can import, which a URL server is not: its tools
belong to a service the deployment runs, so a URL server is checked through the
function that registers them instead (see ``_URL_SERVER_REGISTRARS`` below).
Between the two, every framework server is covered, and the coverage is held by
a map rather than by prose: a server named in neither fails the completeness
test.

There are no known gaps left: the baseline is empty and the assertion is
equality, so a tool that stops being classified fails here and a tool that
starts being classified cannot be quietly left in a baseline.
"""

import asyncio
import importlib
import inspect
import pkgutil

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS

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


#: Servers this process can walk: the ones launched as ``python -m <module>``
#: with a module path that is fixed rather than templated per pipeline. A URL
#: server names no module at all — its tools are registered inside a service
#: the deployment runs and reached over HTTP — so there is nothing here to
#: import.
_WALKABLE = sorted(
    name for name, s in FRAMEWORK_SERVERS.items() if s.module and "{" not in s.module
)

#: The registration function behind each URL server, as ``(module, attribute)``.
#: Importing a URL server's own module registers nothing — the ``@mcp.tool()``
#: decorators run inside the service's factory — so the comparison is made
#: against the function that factory calls, on a throwaway FastMCP. A URL
#: server missing from here fails the completeness test below, so its tools
#: cannot go unchecked for want of an entry.
_URL_SERVER_REGISTRARS: dict[str, tuple[str, str]] = {
    "event_dispatcher": ("osprey.dispatch.mcp_tools", "register_tools"),
}


def _registrar_tool_names(module: str, attribute: str) -> set[str]:
    """Tool names *attribute* of *module* registers on a throwaway FastMCP.

    The tool closures never run, so the collaborators they capture only need to
    exist: every parameter past the FastMCP itself is passed as ``None``, since
    registration reads the tool signatures alone.
    """
    from fastmcp import FastMCP

    register = getattr(importlib.import_module(module), attribute)
    probe = FastMCP(f"{attribute}-classification-probe")
    collaborators = list(inspect.signature(register).parameters)[1:]
    register(probe, **dict.fromkeys(collaborators))

    tools = asyncio.run(probe.list_tools())
    return {getattr(tool, "name", tool) for tool in tools}


def test_every_framework_server_is_walkable_or_templated():
    """Every server is covered by the walk or by a registrar, with no third way."""
    assert set(_WALKABLE) | set(_URL_SERVER_REGISTRARS) | {"channel-finder"} == set(
        FRAMEWORK_SERVERS
    )

    # …and each exception is out of the walk for its stated reason, not because
    # a module path went missing: a stdio server that lost its module would
    # otherwise drop out silently and take its tools' gates with it.
    assert "{" in FRAMEWORK_SERVERS["channel-finder"].module
    for server_name in _URL_SERVER_REGISTRARS:
        assert FRAMEWORK_SERVERS[server_name].url


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


@pytest.mark.parametrize("server_name", sorted(_URL_SERVER_REGISTRARS))
def test_url_server_tools_are_classified_by_the_registry(server_name):
    """A URL server's registered tools are exactly the ones its entry classifies.

    Equality in both directions: a tool added to the service and forgotten in
    the registry reaches a session with no permission entry and no gate, and a
    permission entry for a tool the service no longer registers is a gate over
    nothing.
    """
    server = FRAMEWORK_SERVERS[server_name]
    registered = _registrar_tool_names(*_URL_SERVER_REGISTRARS[server_name])
    assert registered, f"{server_name}: the registrar registered no tools"

    assert registered == {*server.permissions_allow, *server.permissions_ask}


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


# ---------------------------------------------------------------------------
# A facility server's declared writes, through a real render
# ---------------------------------------------------------------------------

#: A facility server that declares two of its four tools as writes. The other
#: two are plain reads: allowed, not in ``permissions.ask``, no destructive
#: marker in the name, so nothing but the declaration decides their class.
_DECLARED_WRITES = ("set_value", "ramp_value")
_DECLARING_READS = ("get_value", "list_values")
_DECLARING_SERVER = {
    "plc": {
        "command": "node",
        "args": ["plc.js"],
        "writes": list(_DECLARED_WRITES),
        "permissions": {"allow": [*_DECLARING_READS, *_DECLARED_WRITES]},
    },
}


@pytest.fixture(scope="module")
def declaring_render(tmp_path_factory):
    """A control-assistant project regenerated with :data:`_DECLARING_SERVER`.

    The regeneration is the render that ships, so its ``hook_config.json`` is
    what the writes-check hook and the audit middleware actually read.
    """
    import yaml

    from osprey.cli.build_profile import resolve_build_profile
    from osprey.cli.templates.manager import TemplateManager
    from osprey.utils.config_writer import config_update_fields
    from tests._preset_data import bundle_data_root

    tmp_path = tmp_path_factory.mktemp("declaring-render")
    manager = TemplateManager()
    profile, _ = resolve_build_profile(None, preset="control-assistant")
    project = manager.create_project(
        project_name="declaring-render",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        data_root=bundle_data_root(),
        context={"channel_finder_mode": "hierarchical"},
        force=True,
    )
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project,
        "declaring-render",
        "control-assistant",
        {},
        artifacts=manager._effective_artifacts("control_assistant", None),
    )
    config_file = project / "config.yml"
    config = yaml.safe_load(config_file.read_text())
    config.setdefault("claude_code", {})["servers"] = _DECLARING_SERVER
    config_file.write_text(yaml.dump(config))
    manager.regenerate_claude_code(project)
    return project


def _rendered_hook_config(project) -> dict:
    import json

    return json.loads((project / ".claude" / "hooks" / "hook_config.json").read_text())


def test_every_declared_tool_classifies_as_a_write_in_the_render(declaring_render):
    """Each declared tool is in the write set and the facility set, never mixed."""
    rendered = _rendered_hook_config(declaring_render)

    for tool in _DECLARED_WRITES:
        name = f"mcp__plc__{tool}"
        assert name in rendered["write_tools"], name
        assert name in rendered["facility_write_tools"], name
        assert name not in rendered["mixed_read_write_tools"], name


def test_a_declaring_servers_other_tools_are_not_writes_in_the_render(declaring_render):
    """No whole-server matcher, and no read tool, rides in with the declaration."""
    rendered = _rendered_hook_config(declaring_render)
    classified = {*rendered["write_tools"], *rendered["facility_write_tools"]}

    assert "mcp__plc__.*" not in classified
    for tool in _DECLARING_READS:
        assert f"mcp__plc__{tool}" not in classified, tool


def test_the_audit_clamp_of_the_render_is_exactly_the_declared_tools(declaring_render, monkeypatch):
    """The middleware's clamp, read from the same file, agrees with the render.

    The clamp is ``write_tools`` minus ``mixed_read_write_tools``; for the
    declaring server that must be the declared tools and nothing else, on the
    verified path the server is launched on.
    """
    from osprey.mcp_server import audit_middleware as am

    monkeypatch.setenv(am.CONFIG_ENV, str(declaring_render / "config.yml"))
    am.reset_audit_state()
    try:
        state = am._resolve_state()
        clamp, source, verified = am._clamp_for(state, "plc")
    finally:
        am.reset_audit_state()

    assert (source, verified) == (am.CLAMP_SOURCE_LOADED, True)
    plc_clamped = {name for name in clamp if name.startswith("mcp__plc__")}
    assert plc_clamped == {f"mcp__plc__{tool}" for tool in _DECLARED_WRITES}
    for tool in _DECLARING_READS:
        assert not am._is_clamped(clamp, f"mcp__plc__{tool}", tool, verified=True), tool
