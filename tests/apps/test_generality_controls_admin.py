"""An image-container application and six logins build with no framework change.

The deployment root is ``fixtures/six_logins``: a profile with the service axis
keys, a hand tool server and the ``apps:`` block, and six logins under
``tiers/`` with no ``exclude:``. The test assembles the ``controls_admin_like``
application under ``apps/`` and the packaged example facility under ``data/``,
runs the real ``osprey build`` once, and reads what it rendered.

Which tool servers a login runs is one mechanism for every login: the
deployment states them, and a login that does without one switches it off by
name (``claude_code.servers.<name>.enabled: false``). Its other lists add to the
deployment's under ``lists: add`` and are the login's alone under ``lists: own``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey
from osprey.cli.templates.preset_data import FACILITIES_DIRNAME, FACILITY_DATA_DIRNAME
from osprey.deployment.web_terminals.render_naming import render_project
from osprey.profiles.web_panels import panel_spec_enabled
from osprey_connectors.types import tier_writes
from tests.cli._lifecycle_build import invoke_build, rendered_config

FIXTURES = Path(__file__).parent / "fixtures"
DEPLOYMENT = FIXTURES / "six_logins"
APPLICATION = FIXTURES / "controls_admin_like"
FACILITY = Path(osprey.__file__).parent / "templates" / FACILITIES_DIRNAME / "example"

PROJECT = "six-logins"
APP = "controls-admin-like"
SERVICE = "controls-admin"
PANEL = "controls-admin"
HAND_SERVER = "controls_admin"

ADD_LOGINS = ("readonly", "readwrite", "admin")
OWN_LOGINS = ("controls", "knowledge", "logbook")

#: The deployment's own selections, which every ``add`` login starts from.
ROOT_SKILLS = {"diagnose", "session-report"}
ROOT_AGENTS = {"data-visualizer"}
ROOT_PANELS = {"system-health"}


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture deployment, assembled and built once."""
    root = tmp_path_factory.mktemp("generality") / PROJECT
    shutil.copytree(DEPLOYMENT, root)
    shutil.copytree(APPLICATION, root / "apps" / APP)
    # This deployment runs no logbook database, so the facility's logbook
    # vocabulary, which would switch its consumers on, is left out.
    shutil.copytree(
        FACILITY,
        root / "data" / FACILITY_DATA_DIRNAME,
        ignore=shutil.ignore_patterns("ariel"),
    )
    result = invoke_build(root)
    assert result.exit_code == 0, result.output
    return root


def _render_dir(repo: Path, login: str | None) -> Path:
    if login is None:
        return repo / "build"
    _, project_path = render_project(PROJECT, login)
    return repo / project_path


def _config(repo: Path, login: str | None) -> dict[str, Any]:
    if login is None:
        return rendered_config(repo)
    path = _render_dir(repo, login) / "config.yml"
    config: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config


def _names(directory: Path, suffix: str = "") -> set[str]:
    if not directory.is_dir():
        return set()
    return {path.name.removesuffix(suffix) for path in directory.iterdir()}


def _skills(repo: Path, login: str | None) -> set[str]:
    return _names(_render_dir(repo, login) / ".claude" / "skills")


def _agents(repo: Path, login: str | None) -> set[str]:
    return _names(_render_dir(repo, login) / ".claude" / "agents", ".md")


def _rules(repo: Path, login: str | None) -> set[str]:
    return _names(_render_dir(repo, login) / ".claude" / "rules", ".md")


def _output_styles(repo: Path, login: str | None) -> set[str]:
    return _names(_render_dir(repo, login) / ".claude" / "output-styles", ".md")


def _hook_scripts(repo: Path, login: str | None) -> set[str]:
    return _names(_render_dir(repo, login) / ".claude" / "hooks", ".py")


def _servers(repo: Path, login: str | None) -> set[str]:
    path = _render_dir(repo, login) / ".mcp.json"
    return set(json.loads(path.read_text(encoding="utf-8"))["mcpServers"])


def _panels(repo: Path, login: str | None) -> set[str]:
    panels = (_config(repo, login).get("web") or {}).get("panels") or {}
    return {name for name, spec in panels.items() if panel_spec_enabled(spec)}


def _writes(repo: Path, login: str | None) -> str:
    return tier_writes(_config(repo, login)["control_system"])


# --- the application builds end to end ----------------------------------------


def test_the_application_service_renders_from_its_fragment_and_the_axis_keys(repo: Path) -> None:
    compose = repo / "build" / "services" / SERVICE / "docker-compose.yml"
    document = yaml.safe_load(compose.read_text(encoding="utf-8"))
    service = document["services"][SERVICE]

    assert service["image"] == "ghcr.io/example/controls-admin:1.0"
    assert service["network_mode"] == "host"
    assert "ports" not in service
    assert service["environment"]["EPICS_CA_ADDR_LIST"] == "gw.example"
    assert service["environment"]["CONTROLS_ADMIN_SITE"] == "${CONTROLS_ADMIN_SITE}"
    assert service["environment"]["CONTROLS_ADMIN_EXTRA"] == "${CONTROLS_ADMIN_EXTRA}"
    assert service["volumes"] == [
        f"./build/services/{SERVICE}/app-config.yml:/etc/controls-admin/config.yml:ro",
        "/facility/controls/admin:/facility/admin:ro",
        f"{SERVICE}-data:/var/lib/controls-admin",
    ]
    assert f"{SERVICE}-data" in document["volumes"]
    assert SERVICE in rendered_config(repo)["deployed_services"]


def test_the_hand_tool_server_is_declared_once_on_the_root(repo: Path) -> None:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    logins = [
        yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in sorted((repo / "tiers").glob("*.yml"))
    ]

    assert HAND_SERVER in profile["mcp_servers"]
    assert len(logins) == 6
    assert all(HAND_SERVER not in (login.get("mcp_servers") or {}) for login in logins)
    assert all("exclude" not in login for login in logins)


# --- the root render ----------------------------------------------------------


def test_the_root_render_is_the_deployment_s_own_selection(repo: Path) -> None:
    assert _skills(repo, None) == ROOT_SKILLS
    assert _agents(repo, None) == ROOT_AGENTS
    assert _panels(repo, None) == {*ROOT_PANELS, PANEL}
    assert HAND_SERVER in _servers(repo, None)
    assert _writes(repo, None) == "allowed"


# --- the add logins -----------------------------------------------------------


@pytest.mark.parametrize("login", ADD_LOGINS)
def test_an_add_login_switches_the_hand_server_off_and_keeps_the_root_lists(
    repo: Path, login: str
) -> None:
    assert HAND_SERVER not in _servers(repo, login)
    assert _config(repo, login)["claude_code"]["servers"][HAND_SERVER]["enabled"] is False
    assert _skills(repo, login) == ROOT_SKILLS
    assert _agents(repo, login) == ROOT_AGENTS
    assert _rules(repo, login) == _rules(repo, None)
    assert _hook_scripts(repo, login) == _hook_scripts(repo, None)
    assert _output_styles(repo, login) == _output_styles(repo, None)


@pytest.mark.parametrize(
    ("login", "panels", "writes"),
    [
        ("readonly", {*ROOT_PANELS, PANEL}, "never"),
        ("readwrite", {*ROOT_PANELS, "jupyter", PANEL}, "allowed"),
        ("admin", {*ROOT_PANELS, "jupyter", PANEL}, "allowed"),
    ],
)
def test_an_add_login_adds_its_panels_and_states_its_tier(
    repo: Path, login: str, panels: set[str], writes: str
) -> None:
    assert _panels(repo, login) == panels
    assert _writes(repo, login) == writes


# --- the own logins -----------------------------------------------------------


@pytest.mark.parametrize(
    ("login", "skills", "agents", "rules", "styles", "panels", "default_panel", "title"),
    [
        (
            "controls",
            {"controls-diag"},
            {"channel-finder"},
            {"safety", "facility"},
            {"control-operator"},
            {"system-health", PANEL},
            "system-health",
            "Control System Assistant",
        ),
        (
            "knowledge",
            set(),
            {"facility-knowledge"},
            {"safety", "artifacts"},
            {"control-operator"},
            {"okf"},
            "okf",
            "Facility Knowledge Assistant",
        ),
        (
            "logbook",
            {"session-report"},
            set(),
            {"safety", "timezone"},
            set(),
            {"channel-finder"},
            "channel-finder",
            "Logbook Research Assistant",
        ),
    ],
)
def test_an_own_login_takes_every_list_from_its_file_alone(
    repo: Path,
    login: str,
    skills: set[str],
    agents: set[str],
    rules: set[str],
    styles: set[str],
    panels: set[str],
    default_panel: str,
    title: str,
) -> None:
    config = _config(repo, login)
    claude_md = (_render_dir(repo, login) / "CLAUDE.md").read_text(encoding="utf-8")

    assert _skills(repo, login) == skills
    assert _agents(repo, login) == agents
    assert _rules(repo, login) == rules
    assert _output_styles(repo, login) == styles
    assert _panels(repo, login) == panels
    assert config["web"]["default_panel"] == default_panel
    assert f"# OSPREY — {title}" in claude_md


def test_an_own_login_s_channel_finder_mode_reaches_its_render_alone(repo: Path) -> None:
    # The deployment selects no channel finder; the carrier selects the agent
    # and its mode, and the mode is rendered for it.
    assert "pipeline_mode" not in _config(repo, None).get("channel_finder", {})
    assert _config(repo, "controls")["channel_finder"]["pipeline_mode"] == "in_context"


@pytest.mark.parametrize("login", OWN_LOGINS)
def test_an_own_login_runs_its_hooks_over_the_host_guard_floor(repo: Path, login: str) -> None:
    hooks = _hook_scripts(repo, login)

    assert "osprey_approval" in hooks
    # The host's guards reach every login whatever its own list says.
    assert {"osprey_writes_check", "osprey_limits", "osprey_memory_guard"} <= hooks
    # A host hook that is not a guard does not.
    assert "osprey_turn_state" in _hook_scripts(repo, None)
    assert "osprey_turn_state" not in hooks


@pytest.mark.parametrize("login", ("knowledge", "logbook"))
def test_an_own_login_switches_off_each_server_it_does_without(repo: Path, login: str) -> None:
    servers = _servers(repo, login)

    assert HAND_SERVER not in servers
    assert "controls" not in servers
    assert "python" not in servers
    assert _config(repo, login)["claude_code"]["servers"][HAND_SERVER]["enabled"] is False


def test_the_shared_carrier_runs_the_hand_server_read_only_with_the_panel(repo: Path) -> None:
    assert HAND_SERVER in _servers(repo, "controls")
    assert PANEL in _panels(repo, "controls")
    assert _writes(repo, "controls") == "never"


@pytest.mark.parametrize("login", ("knowledge", "logbook"))
def test_an_own_login_that_does_not_show_the_application_gets_no_panel(
    repo: Path, login: str
) -> None:
    assert PANEL not in _panels(repo, login)
    assert _writes(repo, login) == "never"
