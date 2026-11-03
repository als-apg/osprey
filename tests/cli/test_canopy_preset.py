"""CANOPY in the control-assistant preset family, as the presets resolve.

CANOPY is the control assistant's operator GUI: the root preset shows it as
the landing tab (``web_panels`` plus ``default_panel``; a built-in tab each web
terminal serves from its own sidecar, so the preset states its settings and no
url) and stages its payload before a build (``lifecycle.pre_build``). The
agent's screen server is not the preset's to state: the panel declares it in
the registry, with its tools, and a build renders it wherever the panel is
shown. So the narrow logins (channel finder, knowledge, logbook), which own
their lists and never inherit the panel, carry no server either and say nothing
about it, and no login denies an authoring tool:
every one of them writes into the user's own store and asks first.

The ``canopy-panel-builder`` skill goes with the panel in the same way, though
a skill is a list entry and not derived: the root selects it, the merged
logins inherit it, and the narrow logins, which own their lists, never carry it.

Asserted on the resolved presets, so the facts here are the ones every build
starts from. What a build renders from them is pinned elsewhere (the explicit-
config equivalence deltas).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.profiles.web_panels import BUILTIN_PANELS, SIDECAR_PANELS
from osprey.registry.mcp import FRAMEWORK_SERVERS

SHIPPED_COLLECTION = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/canopy/collections/control-assistant.json"
)

#: Tools run without a prompt: the reads, then the workspace tools.
ALLOW = [
    "check",
    "read_pv",
    "search_pvs",
    "screenshot",
    "get_screen",
    "list_drafts",
    "list_widgets",
    "get_widget",
    "get_schema",
    "list_collections",
    "get_workspace",
    "set_workspace",
]
#: The two that read and rearrange the tabs and tiles of the user's own panel.
WORKSPACE = ["get_workspace", "set_workspace"]
#: Tools that create, edit or delete a screen in the user's own store, or ask
#: for one to be reviewed for the library: prompted on every call.
ASK = [
    "new_screen",
    "patch_node",
    "add_node",
    "remove_node",
    "convert_legacy",
    "delete_draft",
    "request_push",
]

#: The base's tier floor, which every tier inherits.
SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"
DENY_KEY = "claude_code.permissions.deny"

#: The skill that teaches the agent the screen tools; CANOPY's own name for it.
SKILL = "canopy-panel-builder"

#: The merged logins, which inherit CANOPY from the deployment's profile, and the
#: narrow ones (``lists: own``), which own their lists and inherit none of it.
KEEPING = ("tiers/admin", "tiers/readonly", "tiers/readwrite")
EXCLUDING = ("tiers/channel-finder", "tiers/knowledge", "tiers/logbook")


def resolve(name: str) -> BuildProfile:
    """A bundled preset, its ``extends`` chain merged."""
    profile, _ = resolve_build_profile(None, preset=name)
    return profile


@pytest.fixture(scope="module")
def root() -> BuildProfile:
    return resolve("control-assistant")


class TestRootPreset:
    def test_mcp_server_split(self, root: BuildProfile) -> None:
        """The panel declares the server and its tools; the preset states neither."""
        assert "canopy" not in root.mcp_servers
        panel = SIDECAR_PANELS["canopy"]
        assert panel.agent_hop is True
        assert (list(panel.mcp_allow), list(panel.mcp_ask)) == (ALLOW, ASK)
        server = FRAMEWORK_SERVERS["canopy"]
        assert server.panel_id == "canopy"
        assert server.url == "http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/canopy/mcp"
        assert server.headers == {
            "Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}",
            "X-Osprey-Session": "${OSPREY_SESSION_ID:-}",
        }
        assert (server.permissions_allow, server.permissions_ask) == (ALLOW, ASK)

    def test_a_headless_read_only_run_is_refused_the_authoring_tools(self) -> None:
        """The seven are side-effect tools to the registry, the reads are not."""
        from osprey.agent_runner.write_tools import _registry_side_effect_tools

        refused = {tool for tool in _registry_side_effect_tools() if "__canopy__" in tool}
        assert refused == {f"mcp__canopy__{tool}" for tool in ASK}

    def test_allow_and_ask_are_disjoint(self) -> None:
        assert not set(ALLOW) & set(ASK)

    def test_panel(self, root: BuildProfile) -> None:
        assert "canopy" in root.web_panels
        assert root.default_panel == "canopy"

    def test_panel_is_a_builtin_the_terminal_serves_itself(self) -> None:
        panel = SIDECAR_PANELS["canopy"]
        assert "canopy" in BUILTIN_PANELS
        assert panel.factory_path == "osprey.integrations.canopy.sidecar:CanopySidecar"
        assert (panel.label, panel.health_endpoint) == ("CANOPY", "/health")
        assert panel.operator_secret is False
        assert panel.mcp_path == "/mcp"

    def test_panel_asks_for_two_image_checks(self) -> None:
        panel = SIDECAR_PANELS["canopy"]
        assert len(panel.image_checks) == 2

    def test_panel_settings(self, root: BuildProfile) -> None:
        """Exactly the sidecar's settings the preset states live."""
        stated = {
            key: value for key, value in root.config.items() if key.startswith("web.panels.canopy.")
        }
        assert stated == {
            "web.panels.canopy.home_screen": "facility/overview",
            "web.panels.canopy.vole_bytes_max": 268435456,
        }

    @pytest.mark.parametrize("page", ["trace", "correlation", "waveform"])
    def test_the_bare_tools_open_empty(self, root: BuildProfile, page: str) -> None:
        """A tool default would give the bare tool the seeded member's view.

        A collection tool member without a ``query`` takes the tool's default and
        shares its key (``tool:trace``) with the bare launcher, so opening one
        focuses the other. The seeded views live on the collection members.
        """
        assert f"web.panels.canopy.{page}_default_query" not in root.config

    def test_each_seeded_tool_member_carries_its_own_view(self) -> None:
        """Every tool member of the shipped collection names a query of its own."""
        collection = json.loads(SHIPPED_COLLECTION.read_text(encoding="utf-8"))
        tools = [member for member in collection["members"] if "tool" in member]
        assert [member["tool"] for member in tools] == ["trace", "correlation", "waveform"]
        for member in tools:
            assert isinstance(member.get("query"), str) and member["query"], member
            assert len(member["query"].encode()) <= 4096

    @pytest.mark.parametrize(
        "leaf", ["url", "label", "health_endpoint", "path", "rewrite_prefixes", "theme_file"]
    )
    def test_panel_states_nothing_a_builtin_ignores(self, root: BuildProfile, leaf: str) -> None:
        """A builtin's url, label and health endpoint come from the registry.

        ``theme_file`` is a path inside the container and has no default the
        preset could know, so it ships commented.
        """
        assert f"web.panels.canopy.{leaf}" not in root.config

    def test_payload_is_staged_before_a_build(self, root: BuildProfile) -> None:
        steps = {step.name: step for step in root.lifecycle.pre_build}
        assert list(steps) == ["canopy-payload"]
        stage = steps["canopy-payload"]
        assert stage.run == '"{python}" -m osprey.integrations.canopy.stage'
        assert stage.timeout == 30

    def test_deny_floor_does_not_touch_canopy(self, root: BuildProfile) -> None:
        assert root.config[DENY_KEY] == [SETUP_PATCH_TOOL]

    def test_selects_the_panel_builder_skill_once(self, root: BuildProfile) -> None:
        assert root.skills.count(SKILL) == 1


@pytest.fixture(scope="module")
def layers(tmp_path_factory) -> dict[str, BuildProfile]:
    """Every bundled login, loaded over the materialized deployment.

    A merged login is a delta over the deployment's own profile; a narrow one
    owns its lists. Both load as a build loads them, from the deployment repo's
    ``tiers/``.
    """
    from osprey.cli.build_profile_load import load_profile
    from osprey.cli.build_profile_presets import list_tiers
    from tests.cli.test_tier_app_presets import _layered_repo

    repo = _layered_repo(tmp_path_factory.mktemp("canopy-layers") / "my-facility")
    return {f"tiers/{name}": load_profile(repo / "tiers" / f"{name}.yml") for name in list_tiers()}


class TestLogins:
    def test_the_logins_are_the_ones_classified_here(self, layers: dict[str, BuildProfile]) -> None:
        assert sorted(layers) == sorted(KEEPING + EXCLUDING)

    @pytest.mark.parametrize("name", EXCLUDING)
    def test_narrow_logins_carry_neither_panel_nor_server(
        self, layers: dict[str, BuildProfile], name: str
    ) -> None:
        login = layers[name]
        assert "canopy" not in login.web_panels
        # Nothing switches the server off: it is the panel's, and goes with it.
        assert not [key for key in login.config if key.startswith("claude_code.servers.canopy")]
        assert "canopy" not in login.mcp_servers
        assert login.default_panel != "canopy"

    @pytest.mark.parametrize("name", KEEPING)
    def test_merged_logins_inherit_panel_and_server(
        self, layers: dict[str, BuildProfile], name: str
    ) -> None:
        tier = layers[name]
        assert "canopy" in tier.web_panels
        assert tier.default_panel == "canopy"
        assert "canopy" not in tier.mcp_servers
        assert not [key for key in tier.config if key.startswith("claude_code.servers.canopy")]

    @pytest.mark.parametrize("name", EXCLUDING)
    def test_narrow_logins_carry_no_skill_without_the_panel(
        self, layers: dict[str, BuildProfile], name: str
    ) -> None:
        assert SKILL not in layers[name].skills

    @pytest.mark.parametrize("name", KEEPING)
    def test_merged_logins_inherit_the_skill(
        self, layers: dict[str, BuildProfile], name: str
    ) -> None:
        assert layers[name].skills.count(SKILL) == 1

    def test_a_preset_without_the_panel_does_not_select_the_skill(self) -> None:
        profile = resolve("hello-world")
        assert "canopy" not in profile.web_panels
        assert SKILL not in profile.skills

    def test_readonly_denies_no_canopy_tool(self, layers: dict[str, BuildProfile]) -> None:
        """The read-only tier keeps the floor and adds nothing for CANOPY."""
        deny = layers["tiers/readonly"].config[DENY_KEY]
        assert deny == [SETUP_PATCH_TOOL]

    @pytest.mark.parametrize("name", ("tiers/readwrite", "tiers/admin"))
    def test_write_tiers_keep_the_ask_tools_on_ask(
        self, layers: dict[str, BuildProfile], name: str
    ) -> None:
        deny = layers[name].config.get(DENY_KEY) or []
        assert not [entry for entry in deny if entry.startswith("mcp__canopy__")]

    def test_readonly_auto_allows_the_workspace_tools(
        self, layers: dict[str, BuildProfile]
    ) -> None:
        """The panel's server goes with the panel, and the two ride its allow list."""
        tier = layers["tiers/readonly"]
        assert "canopy" in tier.web_panels
        allowed = FRAMEWORK_SERVERS["canopy"].permissions_allow
        assert set(WORKSPACE) <= set(allowed)
        assert not set(WORKSPACE) & set(FRAMEWORK_SERVERS["canopy"].permissions_ask)
        deny = tier.config[DENY_KEY]
        assert not [entry for entry in deny if entry.startswith("mcp__canopy__")]

    @pytest.mark.parametrize("name", EXCLUDING)
    def test_narrow_logins_get_neither_workspace_tool(
        self, layers: dict[str, BuildProfile], name: str
    ) -> None:
        """Without the panel there is no CANOPY server, so no workspace tool at all."""
        login = layers[name]
        assert "canopy" not in login.web_panels
        assert "canopy" not in login.mcp_servers
        stated = [str(value) for value in login.config.values()]
        assert not [text for text in stated for tool in WORKSPACE if tool in text]
