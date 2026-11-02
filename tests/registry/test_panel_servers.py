"""The MCP entries derived from sidecar panels that open a transport to the agent.

A panel says ``agent_hop`` and ``mcp_path`` once, and three things follow from
that one declaration: the route the terminal admits on the panel token, the
``.mcp.json`` entry that dials it, and the permission and approval wiring of
the tools behind it. This file pins the second and third, with a synthetic
panel throughout — the framework ships no panel that opts in.
"""

from __future__ import annotations

import logging

import pytest

from osprey.agent_runner import write_tools
from osprey.cli.templates.claude_code import (
    config_derived_context,
    panel_declared_context,
)
from osprey.profiles.sidecar_panels import SIDECAR_PANELS, SidecarPanel
from osprey.registry import mcp as registry
from osprey.registry.mcp import (
    _APPROVAL,
    _WRITES_CHECK,
    FRAMEWORK_SERVERS,
    ServerDefinition,
    derive_panel_servers,
    resolve_servers,
)

PANEL_ID = "screens"
CONDITION = f"panel_declared:{PANEL_ID}"


def _panel(**overrides) -> SidecarPanel:
    """A sidecar panel that opens ``/mcp`` to the agent, unless overridden."""
    fields = {
        "name": "Synthetic screens",
        "factory_path": "tests.synthetic:Sidecar",
        "mcp_path": "/mcp",
        "agent_hop": True,
        "mcp_allow": ("list_screens", "read_screen"),
        "mcp_ask": ("push_screen", "delete_screen"),
        "mcp_writes": ("push_screen",),
    }
    fields.update(overrides)
    return SidecarPanel(**fields)


@pytest.fixture()
def merged(monkeypatch) -> ServerDefinition:
    """The synthetic panel's entry, merged into the catalog as at import."""
    (sdef,) = derive_panel_servers({PANEL_ID: _panel()}).values()
    monkeypatch.setitem(FRAMEWORK_SERVERS, PANEL_ID, sdef)
    return sdef


def _entry(config: dict, ctx: dict) -> dict:
    (entry,) = [s for s in resolve_servers(config, ctx) if s["name"] == PANEL_ID]
    return entry


def _ctx(**overrides) -> dict:
    return {"project_root": "/srv/project", "current_python_env": "python", **overrides}


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def test_an_opted_in_panel_derives_one_url_entry():
    derived = derive_panel_servers({PANEL_ID: _panel()})

    assert list(derived) == [PANEL_ID]
    sdef = derived[PANEL_ID]
    assert sdef.name == PANEL_ID
    assert sdef.panel_id == PANEL_ID
    assert sdef.module == ""
    # Both placeholders stay literal and both carry `:-`: Claude Code leaves a
    # bare ${VAR} in place when the variable is unset.
    assert sdef.url == "http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/screens/mcp"
    # The dispatcher entry's two headers: the credential, and the calling
    # session so the proxy can hold it.
    assert sdef.headers == {
        "Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}",
        "X-Osprey-Session": "${OSPREY_SESSION_ID:-}",
    }
    assert sdef.transport == "http"
    assert sdef.condition == CONDITION
    assert sdef.permissions_allow == ["list_screens", "read_screen"]
    assert sdef.permissions_ask == ["push_screen", "delete_screen"]
    assert sdef.hooks_post == []
    assert not sdef.is_external


def test_each_ask_tool_gets_its_own_rule():
    """A write of the machine is governed: the writes check runs ahead of the
    approval prompt, so the switch position decides it. Any other ask tool is
    the dispatcher's shape, approval alone."""
    sdef = derive_panel_servers({PANEL_ID: _panel()})[PANEL_ID]

    assert [rule.matcher for rule in sdef.hooks_pre] == [
        "mcp__screens__push_screen",
        "mcp__screens__delete_screen",
    ]
    push, delete = sdef.hooks_pre
    assert push.hooks == [_WRITES_CHECK, _APPROVAL]
    assert delete.hooks == [_APPROVAL]
    for rule in sdef.hooks_pre:
        assert not rule.keeps_policy
        assert rule.hooks[-1].timeout == 5


def test_a_write_tool_is_governed_by_the_switch_and_an_ask_tool_is_not():
    from osprey.registry.mcp import governed_write_tools

    (sdef,) = derive_panel_servers({PANEL_ID: _panel()}).values()
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(FRAMEWORK_SERVERS, PANEL_ID, sdef)
        servers = resolve_servers({}, _ctx(**{CONDITION: True}))

    governed = governed_write_tools(servers)
    assert "mcp__screens__push_screen" in governed
    assert "mcp__screens__delete_screen" not in governed


def test_a_write_tool_outside_the_ask_list_is_refused():
    with pytest.raises(ValueError, match="mcp_writes"):
        derive_panel_servers({PANEL_ID: _panel(mcp_writes=("read_screen",))})


def test_an_allow_only_panel_derives_no_approval_rule():
    sdef = derive_panel_servers({PANEL_ID: _panel(mcp_ask=(), mcp_writes=())})[PANEL_ID]

    assert sdef.permissions_ask == []
    assert sdef.hooks_pre == []


@pytest.mark.parametrize(
    "panel",
    [
        pytest.param(_panel(agent_hop=False), id="mcp_path-without-agent_hop"),
        pytest.param(_panel(mcp_path=None), id="agent_hop-without-mcp_path"),
        pytest.param(_panel(mcp_path=""), id="empty-mcp_path"),
        # The terminal opens no route for a relative path either: spliced onto
        # the panel id it would name another panel.
        pytest.param(_panel(mcp_path="mcp"), id="relative-mcp_path"),
    ],
)
def test_a_panel_that_opens_no_hop_derives_nothing(panel):
    assert derive_panel_servers({PANEL_ID: panel}) == {}


def test_the_derived_path_is_the_route_the_terminal_admits(monkeypatch):
    """The entry dials exactly what ``panel_tier_routes`` opens for the panel."""
    from osprey.interfaces.web_auth import panel_tier_routes

    monkeypatch.setitem(SIDECAR_PANELS, PANEL_ID, _panel(mcp_path="/agent/mcp"))
    sdef = derive_panel_servers(SIDECAR_PANELS)[PANEL_ID]
    path = sdef.url.removeprefix(registry.PANEL_PROXY_ORIGIN)

    assert path == "/panel/screens/agent/mcp"
    assert {(method, path) for method in ("GET", "POST", "DELETE")} <= panel_tier_routes()


def test_the_derivation_is_pure():
    panels = {PANEL_ID: _panel()}
    before = dict(FRAMEWORK_SERVERS)

    first = derive_panel_servers(panels)
    second = derive_panel_servers(panels)

    assert first == second
    assert first[PANEL_ID] is not second[PANEL_ID]
    assert FRAMEWORK_SERVERS == before
    assert panels == {PANEL_ID: _panel()}


@pytest.mark.parametrize("panel_id", ["my.screens", "screens_", "a b", ""])
def test_a_panel_id_that_cannot_name_a_server_is_refused(panel_id):
    with pytest.raises(ValueError, match="cannot name an MCP server"):
        derive_panel_servers({panel_id: _panel()})


def test_such_an_id_is_fine_on_a_panel_that_opens_no_hop():
    assert derive_panel_servers({"my.screens": _panel(agent_hop=False)}) == {}


# ---------------------------------------------------------------------------
# The merge into the catalog
# ---------------------------------------------------------------------------


def test_the_catalog_holds_the_shipped_panels_entries():
    """Merged once at import: the catalog's panel entries are the derivation."""
    in_catalog = {n: s for n, s in FRAMEWORK_SERVERS.items() if s.panel_id is not None}

    assert in_catalog == derive_panel_servers(SIDECAR_PANELS)


def test_a_panel_cannot_take_a_framework_servers_name():
    with pytest.raises(ValueError, match="existing framework MCP server"):
        derive_panel_servers({"controls": _panel()}, taken=registry._SPELLED_SERVERS)


def test_the_catalog_is_the_spelled_entries_then_the_derived_ones():
    derived = derive_panel_servers(SIDECAR_PANELS)

    assert FRAMEWORK_SERVERS == {**registry._SPELLED_SERVERS, **derived}
    assert not derived.keys() & registry._SPELLED_SERVERS.keys()


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("merged")
def test_the_entry_is_on_when_its_panel_is_declared():
    entry = _entry({}, _ctx(**{CONDITION: True}))

    assert entry["enabled"] is True
    assert entry["url"] == "http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/screens/mcp"
    assert entry["headers"] == {
        "Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}",
        "X-Osprey-Session": "${OSPREY_SESSION_ID:-}",
    }
    assert entry["command"] == ""
    assert entry["permissions_ask"] == ["push_screen", "delete_screen"]
    assert entry["is_custom"] is False


@pytest.mark.usefixtures("merged")
def test_the_entry_is_off_without_its_panel():
    assert _entry({}, _ctx())["enabled"] is False
    assert _entry({}, _ctx(**{CONDITION: False}))["enabled"] is False


@pytest.mark.usefixtures("merged")
def test_enabled_true_cannot_switch_it_on_without_the_panel(caplog):
    """A false condition wins: the route and the token exist only with the panel."""
    with caplog.at_level(logging.WARNING):
        entry = _entry({"servers": {PANEL_ID: {"enabled": True}}}, _ctx())

    assert entry["enabled"] is False
    assert any(
        PANEL_ID in record.message and "does not declare" in record.message
        for record in caplog.records
    )


def test_enabled_true_still_overrides_another_conditioned_server():
    """Only panel-derived entries are held to their condition."""
    servers = resolve_servers({"servers": {"graph": {"enabled": True}}}, _ctx())

    (graph,) = [s for s in servers if s["name"] == "graph"]
    assert graph["enabled"] is True


@pytest.mark.usefixtures("merged")
def test_enabled_false_switches_a_declared_panels_entry_off():
    entry = _entry({"servers": {PANEL_ID: {"enabled": False}}}, _ctx(**{CONDITION: True}))

    assert entry["enabled"] is False


@pytest.mark.usefixtures("merged")
@pytest.mark.parametrize(
    "spec",
    [{"url": "http://elsewhere.example/mcp"}, {"command": "evil-mcp"}],
    ids=["url", "command"],
)
def test_a_spec_cannot_repoint_a_framework_name(caplog, spec):
    (key,) = spec
    with caplog.at_level(logging.WARNING):
        entry = _entry({"servers": {PANEL_ID: spec}}, _ctx(**{CONDITION: True}))

    assert entry["url"] == "http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/screens/mcp"
    assert entry["command"] == ""
    assert any(
        PANEL_ID in record.message and repr(key) in record.message and "ignored" in record.message
        for record in caplog.records
    )


@pytest.mark.usefixtures("merged")
def test_a_spec_cannot_pin_the_bearer_of_a_panel_entry():
    """The catalog declares the header, so a spec's own value never replaces it."""
    spec = {"headers": {"Authorization": "Bearer stolen"}}

    entry = _entry({"servers": {PANEL_ID: spec}}, _ctx(**{CONDITION: True}))

    assert entry["headers"]["Authorization"] == "Bearer ${OSPREY_PANEL_TOKEN:-}"


@pytest.mark.usefixtures("merged")
def test_a_panel_entry_cannot_be_cloned(caplog):
    with caplog.at_level(logging.WARNING):
        servers = resolve_servers(
            {"servers": {"screens2": {"extends": PANEL_ID}}}, _ctx(**{CONDITION: True})
        )

    assert "screens2" not in {s["name"] for s in servers}


# ---------------------------------------------------------------------------
# The condition producer
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("merged")
def test_the_producer_asks_its_predicate_once_per_panel_entry():
    asked: list[str] = []

    def declares(panel_id: str) -> bool:
        asked.append(panel_id)
        return panel_id == PANEL_ID

    produced = panel_declared_context(declares)

    assert produced[CONDITION] is True
    assert asked.count(PANEL_ID) == 1
    assert set(produced) == {
        s.condition for s in FRAMEWORK_SERVERS.values() if s.panel_id is not None
    }
    assert panel_declared_context(lambda _id: False)[CONDITION] is False


def test_the_producer_has_nothing_to_say_about_other_servers(monkeypatch):
    for name in [n for n, s in FRAMEWORK_SERVERS.items() if s.panel_id is not None]:
        monkeypatch.delitem(FRAMEWORK_SERVERS, name)

    assert panel_declared_context(lambda _id: True) == {}


@pytest.mark.usefixtures("merged")
def test_the_render_keys_the_condition_on_the_declared_panel(tmp_path):
    declared = {"web": {"panels": {PANEL_ID: {}}}}
    switched_off = {"web": {"panels": {PANEL_ID: {"enabled": False}}}}

    assert config_derived_context(declared, tmp_path)[CONDITION] is True
    assert config_derived_context(switched_off, tmp_path)[CONDITION] is False
    assert config_derived_context({}, tmp_path)[CONDITION] is False


@pytest.mark.usefixtures("merged")
def test_the_render_carries_the_entry_only_for_a_declaring_config(tmp_path):
    def enabled(config: dict) -> bool:
        ctx = _ctx(**config_derived_context(config, tmp_path))
        return _entry({}, ctx)["enabled"]

    assert enabled({"web": {"panels": {PANEL_ID: {}}}}) is True
    assert enabled({}) is False


# ---------------------------------------------------------------------------
# The read-only path and the shell note
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("merged")
def test_a_derived_ask_tool_is_blocked_on_the_read_only_path(tmp_path):
    ask = ["mcp__screens__push_screen", "mcp__screens__delete_screen"]
    allow = ["mcp__screens__list_screens", "mcp__screens__read_screen"]

    side_effects = write_tools._registry_side_effect_tools()
    disallowed = write_tools.read_only_disallowed_tools(tmp_path)

    for tool in ask:
        assert tool in side_effects
        assert tool in disallowed
    for tool in allow:
        assert tool not in side_effects
        assert tool not in disallowed


def test_a_panel_without_the_hop_blocks_and_opens_nothing(monkeypatch):
    """``mcp_path`` alone: no entry, no route, no tool on any list."""
    from osprey.interfaces.web_auth import panel_tier_routes

    monkeypatch.setitem(SIDECAR_PANELS, PANEL_ID, _panel(agent_hop=False))
    derived = derive_panel_servers(SIDECAR_PANELS)
    monkeypatch.setattr(registry, "FRAMEWORK_SERVERS", {**FRAMEWORK_SERVERS, **derived})

    assert PANEL_ID not in derived
    assert PANEL_ID not in {s["name"] for s in resolve_servers({}, _ctx(**{CONDITION: True}))}
    assert not [t for t in write_tools._registry_side_effect_tools() if "screens" in t]
    assert not [path for _method, path in panel_tier_routes() if path.startswith("/panel/screens")]
    assert CONDITION not in panel_declared_context(lambda _id: True)
