"""What a real ``osprey build`` of the demo cell renders for CANOPY.

``test_canopy_preset.py`` pins the resolved presets. It does not show that the
pieces agree once a whole deployment has been initialised and built: that the
agent reaches CANOPY through its own terminal's panel proxy, and that every
tier keeps CANOPY while every app, which inherits nothing from the profile,
carries none. So this module reads the session's one real
``osprey init --preset demo`` + ``osprey build`` (the shared
``built_demo``) back as parsed YAML and JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from ruamel.yaml import YAML

from osprey.cli.build_profile import resolve_build_profile
from osprey.cli.templates.manager import TemplateManager
from tests._builds import BuiltProject

#: Every test here reads a real build — seconds, not milliseconds.
pytestmark = pytest.mark.slow

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
ASK = [
    "new_screen",
    "patch_node",
    "add_node",
    "remove_node",
    "convert_legacy",
    "delete_draft",
    "request_push",
]

#: The agent's ``.mcp.json`` entry: the terminal's own panel proxy, with every
#: value left as a placeholder for the terminal's environment to fill.
HOP_ENTRY = {
    "type": "http",
    "url": "http://127.0.0.1:${OSPREY_WEB_PORT:-}/panel/canopy/mcp",
    "headers": {
        "Authorization": "Bearer ${OSPREY_PANEL_TOKEN:-}",
        "X-Osprey-Session": "${OSPREY_SESSION_ID:-}",
    },
}

#: The tier and app renders, the panel each lands on, and whether each keeps CANOPY.
RENDER_DEFAULT_PANEL = {
    "readonly": "canopy",
    "readwrite": "canopy",
    "admin": "canopy",
    "knowledge": "okf",
    "logbook": "ariel",
    "channel-finder": "channel-finder",
}
KEEPING = ("readonly", "readwrite", "admin")
EXCLUDING = ("knowledge", "logbook", "channel-finder")
WRITE_TIERS = ("readwrite", "admin")

_ruamel = YAML(typ="rt")


# ── The built cell ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def default_cell(built_demo: BuiltProject) -> Path:
    """The session's shared demo build, read and never written."""
    return built_demo.repo


# ── Reading the artifacts back ───────────────────────────────────────────────


def _config(project: Path, render: str | None = None) -> dict[str, Any]:
    base = project / "build"
    if render is not None:
        base = base / f"{project.name}-{render}"
    return yaml.safe_load((base / "config.yml").read_text(encoding="utf-8"))


def _json(project: Path, render: str | None, relative: str) -> dict[str, Any]:
    base = project / "build"
    if render is not None:
        base = base / f"{project.name}-{render}"
    return json.loads((base / relative).read_text(encoding="utf-8"))


# ── The rendered config ──────────────────────────────────────────────────────


class TestRenderedConfig:
    def test_mcp_split(self, default_cell: Path) -> None:
        """The panel declares the server, so the config spells none of it."""
        assert _json(default_cell, None, ".mcp.json")["mcpServers"]["canopy"] == HOP_ENTRY
        perms = _json(default_cell, None, ".claude/settings.json")["permissions"]
        canopy = {
            tier: [e.removeprefix("mcp__canopy__") for e in entries if "__canopy__" in e]
            for tier, entries in perms.items()
            if isinstance(entries, list)
        }
        assert canopy["allow"] == ALLOW
        assert canopy["ask"] == ASK
        assert canopy["deny"] == []

    def test_every_ask_tool_is_behind_the_approval_hook(self, default_cell: Path) -> None:
        hooks = _json(default_cell, None, ".claude/settings.json")["hooks"]["PreToolUse"]
        gated = {
            rule["matcher"]
            for rule in hooks
            if any("osprey_approval.py" in hook["command"] for hook in rule["hooks"])
        }
        assert {f"mcp__canopy__{tool}" for tool in ASK} <= gated
        assert not {f"mcp__canopy__{tool}" for tool in ALLOW} & gated

    def test_panel_is_the_builtin_with_its_settings(self, default_cell: Path) -> None:
        """The whole block: the builtin's switch and the sidecar's settings, no url."""
        assert _config(default_cell)["web"]["panels"]["canopy"] == {
            "enabled": True,
            "home_screen": "facility/overview",
            "vole_bytes_max": 268435456,
            "theme_file": "data/canopy/theme/osprey.json",
        }

    def test_root_default_panel(self, default_cell: Path) -> None:
        assert _config(default_cell)["web"]["default_panel"] == "canopy"

    def test_root_settings_keep_the_split(self, default_cell: Path) -> None:
        perms = _json(default_cell, None, ".claude/settings.json")["permissions"]
        for tool in ALLOW:
            assert f"mcp__canopy__{tool}" in perms["allow"]
        for tool in ASK:
            assert f"mcp__canopy__{tool}" in perms["ask"]
            assert f"mcp__canopy__{tool}" not in perms.get("deny", [])


class TestTierAndAppRenders:
    @pytest.mark.parametrize("render", sorted(RENDER_DEFAULT_PANEL))
    def test_default_panel_per_render(self, default_cell: Path, render: str) -> None:
        config = _config(default_cell, render)
        assert config["web"]["default_panel"] == RENDER_DEFAULT_PANEL[render]

    @pytest.mark.parametrize("render", KEEPING)
    def test_tiers_carry_canopy(self, default_cell: Path, render: str) -> None:
        config = _config(default_cell, render)
        assert config["web"]["panels"]["canopy"].get("enabled", True) is True
        assert "canopy" not in config["claude_code"].get("servers", {})
        mcp = _json(default_cell, render, ".mcp.json")["mcpServers"]
        assert mcp["canopy"] == HOP_ENTRY

    @pytest.mark.parametrize("render", EXCLUDING)
    def test_apps_without_canopy(self, default_cell: Path, render: str) -> None:
        config = _config(default_cell, render)
        panel = config["web"].get("panels", {}).get("canopy")
        assert panel is None or panel.get("enabled") is False
        assert "canopy" not in (config.get("claude_code") or {}).get("servers", {})
        assert "canopy" not in _json(default_cell, render, ".mcp.json")["mcpServers"]
        perms = _json(default_cell, render, ".claude/settings.json")["permissions"]
        for entries in perms.values():
            if isinstance(entries, list):
                assert not [e for e in entries if e.startswith("mcp__canopy__")]
        settings = _json(default_cell, render, ".claude/settings.json")
        assert "mcp__canopy__" not in json.dumps(settings.get("hooks", {}))

    def test_readonly_settings_ask_for_the_seven_tools(self, default_cell: Path) -> None:
        """A read-only login authors in its own store: asked, like every tier."""
        perms = _json(default_cell, "readonly", ".claude/settings.json")["permissions"]
        assert not [e for e in perms["deny"] if e.startswith("mcp__canopy__")]
        for tool in ASK:
            assert f"mcp__canopy__{tool}" in perms["ask"]
        # The read-only tools stay allowed.
        for tool in ALLOW:
            assert f"mcp__canopy__{tool}" in perms["allow"]

    @pytest.mark.parametrize("render", WRITE_TIERS)
    def test_write_tiers_keep_the_seven_tools_on_ask(self, default_cell: Path, render: str) -> None:
        perms = _json(default_cell, render, ".claude/settings.json")["permissions"]
        assert not [e for e in perms.get("deny", []) if e.startswith("mcp__canopy__")]
        for tool in ASK:
            assert f"mcp__canopy__{tool}" in perms["ask"]


# ── The pre_build hook and the catalog ───────────────────────────────────────


def _hook_steps(preset: str) -> list:
    profile, _ = resolve_build_profile(None, preset=preset)
    return list(profile.lifecycle.pre_build)


def _layer_hook_steps(project: Path, layer: str) -> list:
    """The pre_build steps of a tier file of the built project, loaded as the build loads it."""
    from osprey.cli.build_profile_load import load_profile

    return list(load_profile(project / "tiers" / f"{layer}.yml").lifecycle.pre_build)


class TestPrebuildHook:
    def test_materialized_hook_is_the_payload_step_alone(self, default_cell: Path) -> None:
        with (default_cell / "profile.yml").open("r", encoding="utf-8") as handle:
            profile = _ruamel.load(handle)
        steps = profile["lifecycle"]["pre_build"]
        assert [step["name"] for step in steps] == ["canopy-payload"]
        assert steps[0]["timeout"] == 30
        assert steps[0]["run"] == '"{python}" -m osprey.integrations.canopy.stage'

    @pytest.mark.parametrize("tier", KEEPING)
    def test_tiers_inherit_the_harmless_hook(self, default_cell: Path, tier: str) -> None:
        """Every tier inherits the root's one step, and nothing else.

        Harmless: the payload step is idempotent, so a payload already staged is
        left as it is and running it again for a tier changes nothing.
        """
        root = _hook_steps("demo")
        steps = _layer_hook_steps(default_cell, tier)
        assert steps == root
        assert [step.name for step in steps] == ["canopy-payload"]
        assert "OSPREY_CANOPY" not in steps[0].run and "$" not in steps[0].run

    def test_a_build_carrying_the_hook_renders_every_tier_and_app(self, default_cell: Path) -> None:
        for render in RENDER_DEFAULT_PANEL:
            assert (
                default_cell / "build" / f"{default_cell.name}-{render}" / "config.yml"
            ).is_file()


# ── The data tree the image copies ───────────────────────────────────────────

#: The template's one facility collection (RULINGS D14a), relative to a project.
TEMPLATE_COLLECTION = Path("data/canopy/collections/demo.json")


class TestTheDataTree:
    """The facility collection reaches the project's ``data/`` tree.

    The Dockerfile copies the whole deployment (``COPY . /app/<project>/``) and
    the ``.dockerignore`` names nothing under ``data/``, so there is no copy list
    to extend: a collection in the project's data tree is a collection in the
    image, where the image build checks it and CANOPY serves it.
    """

    def test_the_project_carries_the_template_collection(self, default_cell: Path) -> None:
        template = Path(TemplateManager().template_root) / "apps" / "demo"
        shipped = template / TEMPLATE_COLLECTION
        assert (default_cell / TEMPLATE_COLLECTION).read_bytes() == shipped.read_bytes()

    def test_the_collection_directory_holds_the_one_collection(self, default_cell: Path) -> None:
        directory = default_cell / TEMPLATE_COLLECTION.parent
        assert sorted(p.name for p in directory.iterdir()) == [TEMPLATE_COLLECTION.name]
