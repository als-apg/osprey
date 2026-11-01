"""Tests for the bundled tier and app presets and the control-assistant roster.

A bundled tier (``presets/tiers/<name>.yml``) is a small delta over the
deployment's own profile: write pins, and the panels, skills and privilege
lifts that differ from the host. It carries no deployment fact
and never subtracts, so its top-level keys stay within the tier-kind key set:
the root set minus every deployment key, plus the per-render image keys.

The ``control-assistant`` preset hosts its own multi-user web tier — nginx,
the landing page, one terminal container per person on the roster, and one per
shared app — alongside the full plan stack. Its people each name a TIER, an
access level of the deployment's own agent; its ``shared:`` list names the APPS
everyone on the roster may open beside it.

Three tiers ship in ``presets/tiers/``, each a small delta laid over the
deployment's own profile:

* ``readonly`` — every write surface refuses on every target; built without the EVENTS/BLUESKY panels.
* ``readwrite`` — ``control_system.writes: asked``, so a
  session may write to every machine the deployment configures; channel writes
  pass the ordinary safety chain (writes-check, limits, human approval); adds
  the EVENTS/BLUESKY panels.
* ``admin`` — readwrite's MACHINE and operator panels at the ceiling above it,
  ``control_system.writes: allowed``, plus the privileges the deployment floors off: the ``setup_patch``
  tool, the web Config panel, the scaffold gallery's editors, and the
  ``setup-mode`` skill that drives them.

Three shared logins ship in ``presets/tiers/`` (``lists: own``), each a complete agent definition that
owns its own lists and runs on the deployment's services:

* ``logbook`` — logbook research, with the facility knowledge graph;
* ``knowledge`` — the facility knowledge layer (graph, graph channel finder,
  knowledge bundle);
* ``channel-finder`` — the file-based hierarchical channel finder, no graph.

The tier contract has two halves, and both are asserted wholesale below rather
than key-by-key. The MACHINE axes separate readonly from readwrite: enforcement
(``control_system.writes``, ``never`` against ``asked``) and the
write-oriented panels (EVENTS + BLUESKY, added by every write-armed tier and
absent from readonly). The DEPLOYMENT axes separate admin from both:
``claude_code.permissions.remove_deny`` for the ``setup_patch`` tool,
``web.config_panel.enabled``, ``web.scaffold_gallery.write_enabled``, and the
``setup-mode`` skill. The deployment pins the restricted side of every
deployment axis, so a new tier inherits the floor and has to ask for a
privilege by name.

Tiers and apps are read the way a build reads them: the bundled files laid into
a deployment repo's ``tiers/`` beside the exemplar ``profile.yml``
(the materialized control-assistant deployment), each loaded over that profile.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_cmd import _profile_data_bundle
from osprey.cli.build_profile import BuildProfile, resolve_build_profile
from osprey.cli.build_profile_document import retired_write_answer
from osprey.cli.build_profile_load import _KNOWN_PROFILE_KEYS, load_profile
from osprey.cli.build_profile_presets import _presets_dir, list_presets, list_tiers
from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.qmd_service import DEFAULT_PORT as QMD_DEFAULT_PORT
from osprey.deployment.qmd_service import resolve_qmd_service_config
from osprey.deployment.web_terminals.lint import Finding, lint_web_terminals
from osprey.deployment.web_terminals.naming import web_container_name
from osprey.deployment.web_terminals.ports import resolve_nginx_port
from osprey.deployment.web_terminals.render_naming import derived_renders
from osprey.deployment.web_terminals.roster import roster_entries
from osprey.port_layout import (
    CA_DEFAULT_PORT,
    INDEX_MAX,
    SLOTS_BY_NAME,
    default_port,
    layout_ports,
    resolve_port_base,
)
from osprey.registry.mcp import FRAMEWORK_SERVERS
from osprey.utils.config_writer import config_update_fields
from osprey_connectors.types import configured_targets, tier_writes
from tests._preset_data import create_project_as_built
from tests._renders import tier_render
from tests.cli._lifecycle_build import invoke_build, rendered_config
from tests.cli._template_projects import graph_permission_entries
from tests.fixtures.lifecycle_repo import build_exemplar_repo

TIER_NAMES = ("readonly", "readwrite", "admin")

# Keys only a root profile may spell: inheritance, subtraction and every fact
# that belongs to the deployment rather than to one access level of its agent.
_ROOT_ONLY_KEYS = frozenset(
    {
        "extends",
        "data",
        "project_name",
        "deploy",
        "deploy_services",
        "services",
        "lifecycle",
        "env",
        "environment",
        "dependencies",
        "exclude",
    }
)
# Per-render keys legal in a tier although a root profile never spells them.
_TIER_EXTRA_KEYS = frozenset({"extra_mounts", "seed_base", "image"})
TIER_KIND_KEYS = (_KNOWN_PROFILE_KEYS - _ROOT_ONLY_KEYS) | _TIER_EXTRA_KEYS

# The deployment's services, provider and web-terminal tier are attachment
# facts the build supplies; a bundled tier names none of them.
_ATTACHMENT_CONFIG_KEYS = frozenset({"modules.web_terminals.enabled", "deployed_services"})

#: The tier key: the one write posture a render holds, for every machine.
TIER_WRITES_KEY = "control_system.writes"
#: The retired posture leaves (the flat switch and any connector type's).
_RETIRED_WRITE_KEY = re.compile(r"control_system(\.connector\.[^.]+)?\.writes_enabled")


def _stated_writes(config: dict) -> str | None:
    """The ``control_system.writes`` a file's own ``config:`` states.

    The tier key when it spells it, else what its retired ``writes_enabled``
    leaves amount to (the normalizer's rewrite), else ``None``. Read either
    way so the check holds the posture, not the spelling a shipped file is on.
    """
    if TIER_WRITES_KEY in config:
        return config[TIER_WRITES_KEY]
    return retired_write_answer(config)


def _posture_keys(config: dict) -> set[str]:
    """The keys of a flat ``config:`` block that state the write posture."""
    return {
        key for key in config if key == TIER_WRITES_KEY or _RETIRED_WRITE_KEY.fullmatch(str(key))
    }


def _tier_path(name: str) -> Path:
    return _presets_dir() / "tiers" / f"{name}.yml"


def _load_tier(name: str) -> dict[str, Any]:
    document = yaml.safe_load(_tier_path(name).read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"tiers/{name}.yml is not a mapping"
    return document


def test_tier_kind_key_set_is_derived_from_the_root_set() -> None:
    assert _ROOT_ONLY_KEYS <= _KNOWN_PROFILE_KEYS
    assert not (TIER_KIND_KEYS & _ROOT_ONLY_KEYS)
    assert _TIER_EXTRA_KEYS <= TIER_KIND_KEYS


def test_bundled_tier_directory_holds_exactly_the_three_tiers_and_the_shared_logins() -> None:
    found = sorted(p.stem for p in (_presets_dir() / "tiers").glob("*.yml"))
    assert found == sorted((*TIER_NAMES, *APP_NAMES))


@pytest.mark.parametrize("tier", TIER_NAMES)
def test_tier_keys_within_tier_kind_set(tier: str) -> None:
    keys = set(_load_tier(tier))
    assert keys <= TIER_KIND_KEYS, (
        f"tiers/{tier}.yml spells keys outside the tier kind: {sorted(keys - TIER_KIND_KEYS)}"
    )


@pytest.mark.parametrize("tier", TIER_NAMES)
def test_tier_carries_no_deployment_or_provider_key(tier: str) -> None:
    keys = set(_load_tier(tier))
    assert not (keys & (_ROOT_ONLY_KEYS | {"provider"}))


@pytest.mark.parametrize("tier", TIER_NAMES)
def test_tier_config_is_dotted_and_free_of_attachment_facts(tier: str) -> None:
    config = _load_tier(tier)["config"]
    assert isinstance(config, dict)
    for key, value in config.items():
        assert "." in key, f"tiers/{tier}.yml config key {key!r} is not dotted"
        assert not isinstance(value, dict), f"tiers/{tier}.yml config {key!r} is nested"
    assert not (set(config) & _ATTACHMENT_CONFIG_KEYS)
    assert not any(k.startswith("services.") for k in config)


@pytest.mark.parametrize("tier", TIER_NAMES)
def test_tier_opens_with_its_preset_comment(tier: str) -> None:
    lines = _tier_path(tier).read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# {tier}.yml"
    assert lines[1].startswith("# ") and len(lines[1]) > 2


@pytest.mark.parametrize(
    ("tier", "writes"),
    [
        ("readonly", "never"),
        ("readwrite", "asked"),
        ("admin", "allowed"),
    ],
)
def test_tier_pins_write_posture(tier: str, writes: str) -> None:
    assert _stated_writes(_load_tier(tier)["config"]) == writes


def test_readonly_tier_adds_no_panels_skills_or_privileges() -> None:
    document = _load_tier("readonly")
    assert "web_panels" not in document
    assert "skills" not in document
    config = document["config"]
    assert _posture_keys(config)
    assert set(config) == _posture_keys(config)


def test_readwrite_tier_adds_the_write_panels_only() -> None:
    document = _load_tier("readwrite")
    assert document["web_panels"] == ["events", "bluesky"]
    assert "skills" not in document
    config = document["config"]
    assert _posture_keys(config)
    assert set(config) == _posture_keys(config)


def test_admin_tier_lifts_the_three_deployment_privileges() -> None:
    document = _load_tier("admin")
    config = document["config"]
    assert document["web_panels"] == ["events", "bluesky"]
    assert document["skills"] == ["setup-mode"]
    assert config["claude_code.permissions.remove_deny"] == ["mcp__osprey_workspace__setup_patch"]
    assert config["web.config_panel.enabled"] is True
    assert config["web.scaffold_gallery.write_enabled"] is True


def test_tier_files_are_not_listed_as_deployment_presets() -> None:
    assert not (set(TIER_NAMES) & set(list_presets()))


# ── Bundled app presets ──────────────────────────────────────────────────────
#
# A bundled shared login (``presets/tiers/<name>.yml``, ``lists: own``) is a complete agent definition a
# deployment offers beside its own agent: its own lists, brief, landing panel
# and channel-finder mode, plus config keys laid over the host's. It carries no
# deployment fact (the host's services, provider and data are its by
# attachment), and it states every host tool server it does without as an
# explicit ``claude_code.servers.<name>.enabled: false`` decision, because the
# host's own server switches never reach an app's render.

APP_TITLES = {
    "logbook": "Logbook Research",
    "knowledge": "Facility Knowledge",
    "channel-finder": "Channel Finder",
}
APP_NAMES = tuple(APP_TITLES)
#: The bundled shared logins, sorted: the `lists: own` files in ``presets/tiers/``.
SHARED_LOGIN_NAMES = tuple(sorted(APP_TITLES))


def _person_tiers() -> list[str]:
    """The bundled tiers a person is given: every bundled login but the shared ones."""
    return [name for name in list_tiers() if name not in SHARED_LOGIN_NAMES]


APP_KIND_KEYS = frozenset(
    {
        "name",
        "hooks",
        "rules",
        "skills",
        "agents",
        "output_styles",
        "web_panels",
        "default_panel",
        "claude_md_template",
        "channel_finder_mode",
        "config",
        "mcp_servers",
        "extra_mounts",
        "seed_base",
        "image",
        "provenance",
        "lists",
    }
)
# Every bundled app spells these: the whole definition, nothing inherited.
APP_REQUIRED_KEYS = frozenset(
    {
        "name",
        "hooks",
        "rules",
        "skills",
        "agents",
        "output_styles",
        "web_panels",
        "default_panel",
        "claude_md_template",
        "channel_finder_mode",
        "config",
        "mcp_servers",
        "lists",
    }
)
_APP_LIST_KEYS = ("hooks", "rules", "skills", "agents", "output_styles", "web_panels")
_APP_FORBIDDEN_KEYS = frozenset(
    {"extends", "exclude", "services", "deployed_services", "provider", "data", "deploy_services"}
)

# The tool servers the flagship host runs. An app's render starts from the
# registry defaults, not from the host's switches, so each app names every one
# of these it does without.
HOST_SERVERS = (
    "controls",
    "python",
    "osprey_workspace",
    "ariel",
    "osprey_facility_knowledge",
    "channel-finder",
    "graph",
    "bluesky",
    "health",
)
# Which host servers each app keeps; every other host server is switched off.
APP_SERVERS_ON = {
    "logbook": {"osprey_workspace", "ariel", "graph"},
    "knowledge": {"osprey_workspace", "osprey_facility_knowledge", "channel-finder", "graph"},
    "channel-finder": {"osprey_workspace", "channel-finder"},
}


def _app_path(name: str) -> Path:
    return _presets_dir() / "tiers" / f"{name}.yml"


def _load_app(name: str) -> dict[str, Any]:
    document = yaml.safe_load(_app_path(name).read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"tiers/{name}.yml is not a mapping"
    return document


def _server_switch(name: str) -> str:
    return f"claude_code.servers.{name}.enabled"


def test_bundled_login_directory_holds_exactly_the_three_shared_logins() -> None:
    found = sorted(
        p.stem
        for p in (_presets_dir() / "tiers").glob("*.yml")
        if yaml.safe_load(p.read_text(encoding="utf-8")).get("lists") == "own"
    )
    assert found == sorted(APP_NAMES)
    assert not list((_presets_dir() / "apps").glob("*.yml"))


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_keys_are_the_complete_app_kind_set(app: str) -> None:
    keys = set(_load_app(app))
    assert keys <= APP_KIND_KEYS, (
        f"tiers/{app}.yml spells keys outside the app kind: {sorted(keys - APP_KIND_KEYS)}"
    )
    assert APP_REQUIRED_KEYS <= keys, (
        f"tiers/{app}.yml is not a complete app definition; missing "
        f"{sorted(APP_REQUIRED_KEYS - keys)}"
    )


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_carries_no_inheritance_subtraction_or_deployment_key(app: str) -> None:
    keys = set(_load_app(app))
    assert not (keys & _APP_FORBIDDEN_KEYS), (
        f"tiers/{app}.yml spells {sorted(keys & _APP_FORBIDDEN_KEYS)}"
    )


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_name_is_its_card_title(app: str) -> None:
    assert _load_app(app)["name"] == APP_TITLES[app]


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_opens_with_its_preset_comment(app: str) -> None:
    lines = _app_path(app).read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# {app}.yml"
    assert lines[1].startswith("# ") and len(lines[1]) > 2


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_lists_are_lists_of_names(app: str) -> None:
    document = _load_app(app)
    for key in _APP_LIST_KEYS:
        value = document[key]
        assert isinstance(value, list), f"tiers/{app}.yml {key} is not a list"
        assert all(isinstance(item, str) and item for item in value)
        assert len(value) == len(set(value)), f"tiers/{app}.yml {key} repeats a name"
    assert isinstance(document["mcp_servers"], dict)


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_default_panel_is_one_of_its_panels(app: str) -> None:
    document = _load_app(app)
    assert document["default_panel"] in document["web_panels"]


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_brief_template_ships(app: str) -> None:
    from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog

    template = _load_app(app)["claude_md_template"]
    assert template == f"CLAUDE.{app}.md.j2"
    shipped = Path(__file__).resolve().parents[2] / "src/osprey/templates/claude_code" / template
    assert shipped.is_file(), f"tiers/{app}.yml names a brief that does not ship: {template}"
    names = {
        a.canonical_name
        for a in BuildArtifactCatalog.default()._by_name.values()
        if a.template_path == template
    }
    assert names == {f"claude-md-{app}"}


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_config_is_dotted_and_free_of_attachment_facts(app: str) -> None:
    config = _load_app(app)["config"]
    assert isinstance(config, dict)
    for key, value in config.items():
        assert "." in key, f"tiers/{app}.yml config key {key!r} is not dotted"
        assert not isinstance(value, dict), f"tiers/{app}.yml config {key!r} is nested"
    assert not (set(config) & _ATTACHMENT_CONFIG_KEYS)
    assert not any(k.startswith("services.") for k in config)


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_states_every_host_server_it_switches_off(app: str) -> None:
    config = _load_app(app)["config"]
    for server in HOST_SERVERS:
        key = _server_switch(server)
        if server in APP_SERVERS_ON[app]:
            assert config.get(key, True) is True, f"tiers/{app}.yml switches off {server}"
        else:
            assert config.get(key) is False, (
                f"tiers/{app}.yml does not state {server} switched off ({key}: false)"
            )


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_pins_every_write_key_off_and_lists_no_write_hooks(app: str) -> None:
    document = _load_app(app)
    assert _stated_writes(document["config"]) == "never"
    assert not ({"writes-check", "limits"} & set(document["hooks"]))
    assert not ({"events", "bluesky"} & set(document["web_panels"]))


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_keeps_the_safety_rule(app: str) -> None:
    assert "safety" in _load_app(app)["rules"]


def test_logbook_app_keeps_the_graph_and_the_logbook_surface() -> None:
    document = _load_app("logbook")
    assert document["agents"] == ["facility-knowledge-graph"]
    assert "logbook-deep-research" in document["skills"]
    assert document["web_panels"] == ["ariel"]
    assert document["default_panel"] == "ariel"
    assert document["channel_finder_mode"] is None
    config = document["config"]
    assert config["approval.tools.entry_create"] == "always"
    assert config["approval.tools.entry_publish"] == "always"


def test_knowledge_app_answers_from_graph_channels_and_bundle() -> None:
    document = _load_app("knowledge")
    assert document["channel_finder_mode"] == "graph"
    assert set(document["agents"]) == {
        "channel-finder",
        "facility-knowledge",
        "facility-knowledge-graph",
    }
    assert document["web_panels"] == ["okf", "channel-finder"]
    assert document["default_panel"] == "okf"
    assert document["config"]["claude_code.permissions.deny"] == [
        "mcp__osprey_facility_knowledge__draft_concept"
    ]


def test_channel_finder_app_is_file_based_and_graphless() -> None:
    document = _load_app("channel-finder")
    assert document["channel_finder_mode"] == "hierarchical"
    assert document["agents"] == ["channel-finder"]
    assert "cf-feedback-capture" in document["hooks"]
    assert document["web_panels"] == ["channel-finder"]
    assert "facility-knowledge-graph" not in document["agents"]


@pytest.mark.parametrize("app", APP_NAMES)
def test_app_selections_name_shipped_artifacts(app: str) -> None:
    from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog

    names = set(BuildArtifactCatalog.default()._by_name)
    document = _load_app(app)
    selections = (
        ("hooks", "hooks"),
        ("rules", "rules"),
        ("skills", "skills"),
        ("agents", "agents"),
        ("output_styles", "output-styles"),
    )
    for key, prefix in selections:
        for item in document[key]:
            assert f"{prefix}/{item}" in names, f"tiers/{app}.yml {key} names unknown {item!r}"


def test_app_files_are_not_listed_as_deployment_presets() -> None:
    assert not (set(APP_NAMES) & set(list_presets()))


def test_no_stem_is_both_a_tier_and_an_app() -> None:
    assert not (set(TIER_NAMES) & set(APP_NAMES))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# The enforcement axis the control-system tiers differ on: the tier key, one
# deployment-wide word (``never``, ``asked`` or ``allowed``) a render holds for every
# machine. A loaded layer carries it whichever spelling the shipped file is on,
# because the normalizer rewrites the retired ``writes_enabled`` leaves to it.
WRITES_KEY = TIER_WRITES_KEY
#: The reference deployment's tier files, hand-kept copies of the bundled
#: tiers that other suites validate and parse.
EXEMPLAR_TIERS_DIR = (
    Path(__file__).resolve().parents[1] / "deployment" / "goldens" / "exemplar-profile" / "tiers"
)
# The agent's deployment-editing tool: denied by the deployment for every tier,
# and subtracted back by the admin tier alone. A deny rather than an ask because
# string lists only ever UNION when a tier is laid over the profile — see the
# preset's note.
SETUP_PATCH_TOOL = "mcp__osprey_workspace__setup_patch"
DENY_KEY = "claude_code.permissions.deny"
REMOVE_DENY_KEY = "claude_code.permissions.remove_deny"
# The two browser-side halves of the same privilege: editing the running
# deployment's configuration, and writing to the shared prompt/skill gallery.
CONFIG_PANEL_KEY = "web.config_panel.enabled"
GALLERY_WRITE_KEY = "web.scaffold_gallery.write_enabled"
# The two browser-side floor keys, which the deployment pins false and the
# admin tier flips true. They differ from the tool axis in HOW they differ:
# every tier carries these keys (they are inherited), so the admin delta is a
# value, not a presence. `REMOVE_DENY_KEY` is the opposite — no other tier
# declares it at all — and `DENY_KEY` is neither: the deployment's deny reaches
# every tier verbatim, admin included, which is exactly why the admin tier has
# to subtract it.
ADMIN_LIFTED_FLOOR_KEYS = (CONFIG_PANEL_KEY, GALLERY_WRITE_KEY)
# The skill that drives those surfaces from the agent side.
ADMIN_ONLY_SKILL = "setup-mode"
# The write-oriented panels: added by every write-armed tier's `web_panels`
# list (readwrite, admin) and by none other, so the readonly build genuinely
# lacks them (a tier can only add; `enabled: false` is inert for URL panels).
# Their URL, path and label are NOT preset config — the build projects them
# from the hosting deployment's render (osprey.deployment.reach), which is
# asserted on a real build in test_readwrite_is_told_its_panel_urls.
WRITE_TIER_PANELS = ("events", "bluesky")

# The literal dotted key the hosting preset must carry: the whole web-terminals
# module subtree addressed as one leaf so config_writer sets only this leaf and
# never wholesale-replaces the rendered ``modules`` subtree.
WEB_TERMINALS_KEY = "modules.web_terminals"

# A valid Docker container/object name: must start with an alphanumeric, then
# alphanumerics plus `_`, `.`, `-` (see docker/docker daemon name validation).
DOCKER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

#: The shared apps the flagship roster offers, in the order its ``shared:``
#: list states them — the order their cards and port slots follow.
FLAGSHIP_SHARED = ["logbook", "knowledge", "channel-finder"]


def resolve_preset(name: str) -> BuildProfile:
    """Resolve a bundled deployment preset by name.

    Args:
        name: Bundled preset name (hyphenated or underscored).

    Returns:
        The parsed and validated :class:`BuildProfile`.
    """
    profile, _profile_dir = resolve_build_profile(None, preset=name)
    return profile


def _bundled_layer(kind: str, name: str) -> dict:
    """The document of one bundled tier or app file, as written.

    Args:
        kind: ``"tiers"`` or ``"apps"``, the packaged directory it lives in.
        name: The tier or app name.
    """
    path = _presets_dir() / kind / f"{name}.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _own_config(kind: str, name: str) -> dict:
    """The ``config:`` keys a bundled tier or app spells in its own file.

    Distinct from the loaded layer's ``config``, which carries everything the
    deployment's profile beneath it holds. The checks below are about what a
    tier or app STATES, and a tier that states nothing still loads to the
    profile's whole block.
    """
    return _bundled_layer(kind, name).get("config") or {}


def _layered_repo(dest: Path) -> Path:
    """A control-assistant deployment repo carrying the bundled tiers and apps.

    The exemplar repo is the materialized ``control-assistant`` deployment; its
    ``tiers/`` and ``apps/`` are overwritten here with the bundled files
    themselves, so what loads and builds below is the shipped content laid over
    the shipped deployment, never a hand-kept copy of it.

    Args:
        dest: Directory to materialize the repo into.

    Returns:
        The repo root.
    """
    repo = build_exemplar_repo(dest)
    for path in sorted((_presets_dir() / "tiers").glob("*.yml")):
        shutil.copyfile(path, repo / "tiers" / path.name)
    return repo


@pytest.fixture(scope="module")
def layered_repo(tmp_path_factory) -> Path:
    """The layered deployment repo, materialized once per module."""
    return _layered_repo(tmp_path_factory.mktemp("layered") / "my-facility")


@pytest.fixture(scope="module")
def layers(layered_repo: Path) -> dict[str, BuildProfile]:
    """The deployment's own profile and every tier and app loaded over it.

    Keyed ``"profile"`` and ``"tiers/<name>"``.
    """
    loaded = {"profile": load_profile(layered_repo / "profile.yml")}
    for name in list_tiers():
        loaded[f"tiers/{name}"] = load_profile(layered_repo / "tiers" / f"{name}.yml")
    return loaded


def _render_config_overrides(
    tmp_path: Path, seed: dict, preset: str = "control-assistant", *, config: dict | None = None
) -> dict:
    """Render a ``config:`` layer onto ``seed`` exactly as the build pipeline
    does — via :func:`config_update_fields`, the same dot-notation writer
    ``_apply_config_overrides`` calls — and reload the result as a plain dict.

    Args:
        tmp_path: Per-test temp directory (pytest fixture).
        seed: Pre-existing config contents to render the overrides onto.
        preset: Deployment preset whose overrides to render.
        config: A loaded tier's or app's ``config`` to render instead of
            *preset*'s.

    Returns:
        The reloaded config after applying the overrides.
    """
    config_path = tmp_path / "config.yml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(seed, fh)
    overrides = resolve_preset(preset).config if config is None else config
    config_update_fields(config_path, overrides)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _render_deployable_config(
    tmp_path: Path, repo: Path, preset: str = "control-assistant"
) -> dict:
    """The root config a real build produces from ``preset`` — renders included.

    :func:`_render_config_overrides` renders the preset's ``config:`` layer
    alone. Every build also projects the repo's tier and app files into
    ``modules.web_terminals.renders`` (``derived_renders``), so a config linted
    without that projection is a shape the pipeline never emits. This applies
    the same projection, through the same function the build calls, over a repo
    that carries the bundled tiers and apps, with ``project_name`` = *preset* —
    what a real ``osprey init --preset control-assistant`` proposes from a repo
    named after the preset.
    """
    config_path = tmp_path / "config.yml"
    resolved = resolve_preset(preset)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"system": {}}, fh)
    config_update_fields(config_path, resolved.config)
    config_update_fields(
        config_path,
        {"project_name": preset, **derived_renders(resolved.config, preset, repo)},
    )
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def _build_layered_stack(repo: Path) -> Path:
    """``osprey build`` the layered deployment repo: host, tiers and apps.

    One build renders the host project at ``build/`` and one project per tier
    and app file beside it at ``build/<project_name>-<name>``. Nothing here is
    stubbed: the config a tier or app render reads is the one the pipeline
    actually assembles, and a layer's ``config:`` is applied by the build
    between the base template render and the Claude Code re-render — an
    ordering that decides whether a preset key can reach the agent surface at
    all, and one no config-only helper in this module can observe.

    Args:
        repo: The layered deployment repo (:func:`_layered_repo`).

    Returns:
        The repo root; the renders live under ``<repo>/build/``.
    """

    built = invoke_build(repo)
    assert built.exit_code == 0, built.output
    return repo


def _build_render_stack(repo: Path) -> Path:
    """``osprey init`` the hosting preset, then ``osprey build`` the whole stack.

    The init writes the deployment repo with its tiers and apps materialized;
    the build renders the host project at ``build/`` and one project per tier
    and app beside it.

    Args:
        repo: Directory to materialize the deployment repo into.

    Returns:
        The repo root; the renders live under ``<repo>/build/``.
    """
    from click.testing import CliRunner

    from osprey.cli.init_cmd import init

    created = CliRunner().invoke(init, [str(repo), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    return _build_layered_stack(repo)


def _render_dir(stack: Path, name: str) -> Path:
    """The render of tier or app *name* in a built *stack*."""
    host = rendered_config(stack)
    return stack / "build" / f"{host['project_name']}-{name}"


def _graph_entries(values) -> list[str]:
    return sorted(entry for entry in values if str(entry).startswith("mcp__graph__"))


@pytest.fixture(scope="module")
def built_layered_stack(tmp_path_factory) -> Path:
    """The layered deployment built once, every tier and app included."""
    return _build_layered_stack(_layered_repo(tmp_path_factory.mktemp("stack") / "my-facility"))


# ---------------------------------------------------------------------------
# Hosting preset: multi-user web-terminal block
# ---------------------------------------------------------------------------


class TestControlAssistantWebTier:
    """The tutorial preset hosts its own web tier: block shape, port families
    clear of the tutorial's own service ports, and a lint-clean render."""

    def test_web_terminals_is_one_literal_dotted_key(self) -> None:
        """The block is carried as the single literal ``modules.web_terminals``
        key, never a nested ``modules:`` mapping under ``config:``.

        A nested mapping would deep-merge over the rendered ``modules`` subtree
        and drop sibling modules; the literal dotted key sets only its own leaf.
        """
        base = resolve_preset("control-assistant")
        assert WEB_TERMINALS_KEY in base.config
        assert "modules" not in base.config
        assert "facility" not in base.config
        assert "deploy" not in base.config

    def test_hosts_its_own_web_tier(self) -> None:
        """The preset carries the web-terminals block plus the web-tier
        companion key ``deploy.fqdn`` for the landing URL — the hosting posture
        the tiers and apps below render beside."""
        base = resolve_preset("control-assistant")
        assert WEB_TERMINALS_KEY in base.config
        assert "deploy.fqdn" in base.config

    def test_rendered_config_keeps_sibling_modules(self, tmp_path: Path) -> None:
        """Rendering the overrides adds ``modules.web_terminals`` without
        clobbering a pre-existing sibling ``modules.*`` key or unrelated config.

        This is the whole reason the block is a flat dotted key: config_writer
        writes each dotted key verbatim, so an existing ``modules.event_dispatcher``
        (and any other top-level key) must survive untouched.
        """
        seed = {
            "modules": {
                "event_dispatcher": {"enabled": True, "port": 8010},
            },
            "ports": {"matlab": 8001},
            "system": {"facility_name": "Seed Facility"},
        }
        rendered = _render_config_overrides(tmp_path, seed)

        modules = rendered["modules"]
        # The new block landed.
        assert modules["web_terminals"]["enabled"] is True
        # The sibling module and unrelated keys were not clobbered.
        assert modules["event_dispatcher"] == {"enabled": True, "port": 8010}
        assert rendered["ports"] == {"matlab": 8001}
        assert rendered["system"]["facility_name"] == "Seed Facility"

    def test_rendered_web_terminals_shape(self, tmp_path: Path) -> None:
        """The rendered ``modules.web_terminals`` subtree matches the shipped
        tutorial shape: local image source, a readonly default tier, a roster
        of three people — alice→readwrite, bob→readonly, carol→admin — each
        carrying the tab-title ``display_name`` that visibly marks which
        terminal is which, and three shared apps.

        The roster lists people only. The logbook, knowledge and channel-finder
        cards are apps the deployment offers beside its own agent, named in
        ``shared:``; their containers, port slots, card titles and landing
        section are derived, so nothing about them is spelled as a user, and
        there is no catalog of renders in the profile at all — the build
        projects one from ``tiers/`` and ``apps/``.

        Carol is a person, and the roster's least-privileged-by-default
        property is what her entry pins: ``default_tier`` stays ``readonly``
        even though an admin login exists, so an unrouted visitor lands on the
        read-only tier rather than the one that can rewrite the deployment."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        wt = rendered["modules"]["web_terminals"]

        assert wt["enabled"] is True
        assert wt["image_source"] == "local"
        assert wt["default_tier"] == "readonly"
        # The roster names no port at all: every one of them comes from this
        # deployment's own block, so giving a second deployment its own
        # `deployment.port_base` moves the whole tier with it.
        assert [key for key in wt if "port" in key] == []
        assert resolve_nginx_port(rendered) == default_port("nginx")
        # Password login ships ON, in the demo posture: plain HTTP is only
        # acceptable because the preset's fqdn is loopback. The lifetime is the
        # framework default spelled out, so the preset shows where to change it.
        assert wt["auth"] == {
            "method": "password",
            "allow_insecure_http": True,
            "session_lifetime": 43200,
        }

        assert wt["users"] == [
            {
                "name": "alice",
                "index": 0,
                "tier": "readwrite",
                "display_name": "Control Room (Alice)",
            },
            {
                "name": "bob",
                "index": 1,
                "tier": "readonly",
                "display_name": "Read-Only View (Bob)",
            },
            # The admin login is neither shared nor public: the one account
            # that can rewrite the deployment sits behind its own login.
            {
                "name": "carol",
                "index": 2,
                "tier": "admin",
                "display_name": "Deployment Admin (Carol)",
            },
        ]
        # Every tier the roster names is one the deployment ships.
        assert {user["tier"] for user in wt["users"]} | {wt["default_tier"]} <= set(_person_tiers())

        # Every bundled app, shared with the whole roster, in the stated order.
        assert wt["shared"] == FLAGSHIP_SHARED
        assert sorted(wt["shared"]) == list(SHARED_LOGIN_NAMES)

        # No render projection and no per-card landing section.
        assert "renders" not in wt
        for user in wt["users"]:
            assert not {"access", "login"} & set(user), user

        # People first, under their own heading; the apps after, under theirs.
        assert wt["landing"]["groups"] == [
            {"type": "users", "label": "Users"},
            {"type": "shared", "label": "Shared logins"},
        ]

    def test_preset_spells_the_roster_in_tier_words(self) -> None:
        """Every comment of the preset is copied into the operator's
        ``profile.yml``, so it teaches the current vocabulary only — the
        commented SSO example included, which spells ``roles.<r>.tier``."""
        text = (_presets_dir() / "control-assistant.yml").read_text(encoding="utf-8")
        assert re.findall(r"(?i)landing_group|access: any", text) == []
        assert "roles: {operator: {tier: readwrite}, viewer: {tier: readonly}}" in text

    def test_port_families_clear_tutorial_service_ports(self, tmp_path: Path) -> None:
        """No terminal a roster entry opens — a person's or a shared app's —
        lands on a port one of the tutorial's own services already publishes.

        The containers share the host network namespace, so a family landing on
        a service port would collide at deploy time. The preset no longer pins
        the families itself — that is the point: it writes no port keys, and
        the separation is the block layout's, checked here at the base this
        preset actually resolves for the roster it actually ships. People count
        up from slot 0 and the shared apps down from the top of the block, so
        both ends are checked."""
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        wt = rendered["modules"]["web_terminals"]
        base = resolve_port_base(rendered)
        ports = layout_ports(base)
        # Everything the deployment publishes once, plus the Channel Access
        # port the virtual accelerator keeps outside the block.
        service_ports = {
            port for name, port in ports.items() if not SLOTS_BY_NAME[name].per_index
        } | {CA_DEFAULT_PORT}
        families = [name for name, slot in SLOTS_BY_NAME.items() if slot.per_index]
        assert "web" in families and "channel_finder" in families

        indices = [entry["index"] for entry in roster_entries(wt)]
        assert indices == [0, 1, 2, INDEX_MAX, INDEX_MAX - 1, INDEX_MAX - 2]
        for family in families:
            for index in indices:
                assert default_port(family, index, base) not in service_ports, (
                    f"{family} slot {index} collides with a tutorial service port"
                )

    def test_rendered_config_lints_with_nothing_wrong_but_the_missing_render(
        self, tmp_path: Path, layered_repo: Path
    ) -> None:
        """``lint_web_terminals`` on the freshly-built tutorial config reports
        nothing about the config's SHAPE pre-deploy.

        The tier and app renders do not exist yet at build time. Most of those
        not-yet-rendered paths are WARNINGS. The one an unrouted login lands on
        — the default tier — is an ERROR (``render_privileges_unknown``): its
        privileges cannot be read, so "holds nothing" would be a guess about
        the terminal every unassigned login opens, and `osprey up` gates on
        this belt, where that guess would be fail-open on the deploy path.

        Every finding names the same remedy and it is the command that comes
        next anyway: ``osprey build``. What this test pins is that nothing ELSE
        is reported — every error here is about the render that has not happened
        yet, not about the preset.

        Linted through :func:`_render_deployable_config`, i.e. with the renders
        projection every build writes.
        """
        rendered = _render_deployable_config(tmp_path, layered_repo)

        errors = _errors(lint_web_terminals(rendered))

        assert {finding.code for finding in errors} <= {"web_terminals.render_privileges_unknown"}
        for finding in errors:
            assert "osprey build" in finding.message
        assert any("'readonly'" in finding.message for finding in errors)

    def test_ships_companion_panels_multi_user(self) -> None:
        """Feature parity: multi-user must not shed single-user companion panels.

        The channel-finder panel was once dropped from a hosting preset to dodge
        a per-user port collision (its family was missing from the port-family
        derivation) — the fix is per-user ports, never removing the feature.
        """
        base = resolve_preset("control-assistant")
        assert "channel-finder" in base.web_panels
        assert "ariel" in base.web_panels

    def test_notebook_tab_follows_the_control_surface(
        self, layers: dict[str, BuildProfile]
    ) -> None:
        """The JUPYTER tab is declared once, in the deployment, and reaches
        exactly the renders that keep the controls server.

        A notebook kernel reads and writes through the control target, so the
        tab is machine reach. Every tier is the deployment's own agent and
        inherits it. An app owns its lists, so an app that switches the
        controls server off states no JUPYTER tab of its own. Both groups are
        derived from the bundled tiers and apps on disk rather than listed
        here, so one added later is covered without editing this test.
        """
        assert "jupyter" in resolve_preset("control-assistant").web_panels
        assert _person_tiers(), "no bundled tier"
        for tier in _person_tiers():
            assert "jupyter" in layers[f"tiers/{tier}"].web_panels, tier

        no_control_surface = sorted(
            app
            for app in list(SHARED_LOGIN_NAMES)
            if layers[f"tiers/{app}"].config.get("claude_code.servers.controls.enabled") is False
        )
        assert no_control_surface == ["channel-finder", "knowledge", "logbook"]
        for app in list(SHARED_LOGIN_NAMES):
            expected = app not in no_control_surface
            assert ("jupyter" in layers[f"tiers/{app}"].web_panels) is expected, app

    def test_derived_web_container_names_are_valid_docker_names(self, tmp_path: Path) -> None:
        """The container names derived from the rendered compose project —
        ``<project>-nginx``, ``<project>-auth`` and ``<project>-web-<name>``
        for every roster entry, the shared apps included — all match the
        Docker name grammar (start alphanumeric, no leading dash), which
        ``osprey up`` would otherwise fail the web stack on.
        """
        rendered = _render_config_overrides(tmp_path, {"system": {}})
        project = resolve_project_name(rendered)
        names = (
            f"{project}-nginx",
            f"{project}-auth",
            *(
                web_container_name(project, entry["name"])
                for entry in roster_entries(rendered["modules"]["web_terminals"])
            ),
        )
        for name in names:
            assert DOCKER_NAME_RE.match(name), f"invalid Docker container name: {name!r}"

    def test_rendered_config_satisfies_landing_url(self, tmp_path: Path) -> None:
        """The rendered config passes the exact check ``osprey up`` runs —
        ``_landing_url`` raises without ``deploy.fqdn``, aborting ``osprey up``
        for the otherwise zero-config tutorial."""
        from osprey.deployment.web_terminals.render import TLS_LISTEN_PORT, _landing_url

        rendered = _render_config_overrides(tmp_path, {"system": {}})
        fqdn = rendered["deploy"]["fqdn"]
        nginx_port = resolve_nginx_port(rendered)
        assert nginx_port == default_port("nginx")
        # `tls_port` is required even here, where the preset renders TLS off and
        # the port is never read: the default it would otherwise take is the one
        # value a caller must not guess (see `_external_origin`).
        assert (
            _landing_url(rendered, nginx_port, tls_port=TLS_LISTEN_PORT)
            == f"http://{fqdn}:{nginx_port}"
        )


# ---------------------------------------------------------------------------
# The tiers: single-axis contract; the apps: their own agents
# ---------------------------------------------------------------------------


#: The page that tells a deployer what each tier's screen looks like.
TIERS_PAGE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "source"
    / "how-to"
    / "web-terminal"
    / "multi-user"
    / "tiers.rst"
)
#: The anchor above the page's tier table. Rows are read from it to the next anchor
#: or, when the page has none, to its end.
TIERS_TABLE_ANCHOR = ".. _multi-user-tiers:"
#: A tier row's first cell, ``* - **<tier>** (<login>)``.
_TIER_ROW = re.compile(r"^\s*\* - \*\*(\w+)\*\*")
#: A later cell of the same row, written ``- `` at the row's indent.
_ROW_CELL = re.compile(r"^\s*- (.*)$")
#: What a screen cell says about the write-oriented panels, by their tab labels.
_PANEL_CLAIM = re.compile(r"\b(with|without) the EVENTS and BLUESKY panels\b")


def _tier_screen_cells() -> dict[str, str]:
    """Each tier row's last cell in the tiers page's table, whitespace-collapsed."""
    lines = TIERS_PAGE.read_text(encoding="utf-8").splitlines()
    assert TIERS_TABLE_ANCHOR in lines, f"{TIERS_PAGE} lost its {TIERS_TABLE_ANCHOR} anchor"
    cells: dict[str, list[str]] = {}
    tier = None
    for line in lines[lines.index(TIERS_TABLE_ANCHOR) + 1 :]:
        if line.startswith(".. _"):
            break
        row = _TIER_ROW.match(line)
        if row is not None:
            tier = row.group(1)
            cells[tier] = [""]
        elif tier is not None and (cell := _ROW_CELL.match(line)) is not None:
            cells[tier].append(cell.group(1))
        elif tier is not None and line.strip():
            cells[tier][-1] += " " + line.strip()
    return {name: " ".join(parts[-1].split()) for name, parts in cells.items()}


class TestControlAssistantTiersAndApps:
    """The three tiers: identical renders except for the tier contract; the
    three apps: agents of their own on the deployment's services.

    Over the MACHINE, readonly and readwrite differ on enforcement
    (``writes_enabled``) and the write-oriented panels
    (write-armed tiers only). Over the DEPLOYMENT, admin differs from readwrite
    on the three keys the deployment floors off — ``remove_deny`` for
    ``setup_patch``, the Config panel, the gallery's write surfaces — plus the
    ``setup-mode`` skill that drives them.

    Any difference OUTSIDE those two lists would turn the multi-user story into
    a lie, so both invariants are asserted wholesale rather than key-by-key."""

    def test_the_roster_names_exactly_the_bundled_tiers_and_apps(self) -> None:
        """Every tier the roster can land a login on, and every app it
        shares, is one OSPREY ships — and every one OSPREY ships is reachable
        from the flagship roster, so none is dead weight in a fresh repo."""
        wt = resolve_preset("control-assistant").config[WEB_TERMINALS_KEY]
        assert (
            sorted({user["tier"] for user in wt["users"]} | {wt["default_tier"]}) == _person_tiers()
        )
        assert sorted(wt["shared"]) == list(SHARED_LOGIN_NAMES)

    def test_readonly_disables_writes(self, layers: dict[str, BuildProfile]) -> None:
        profile = layers["tiers/readonly"]
        assert profile.name == "Control Assistant (Read-Only)"
        assert _profile_data_bundle(profile) == "control_assistant"
        assert profile.config.get(WRITES_KEY) == "never"
        assert not _posture_keys(profile.config) - {WRITES_KEY}
        # Flat dotted key, never nested YAML under config:.
        assert "control_system" not in _own_config("tiers", "readonly")

    def test_readwrite_arms_the_simulator(self, layers: dict[str, BuildProfile]) -> None:
        profile = layers["tiers/readwrite"]
        assert profile.name == "Control Assistant (Read-Write)"
        assert _profile_data_bundle(profile) == "control_assistant"
        assert profile.config.get(WRITES_KEY) == "asked"
        assert not _posture_keys(profile.config) - {WRITES_KEY}
        assert "control_system" not in _own_config("tiers", "readwrite")

    def test_admin_lifts_the_deployment_floor(self, layers: dict[str, BuildProfile]) -> None:
        """The admin tier is readwrite plus deployment editing at the ceiling
        above it: it holds ``control_system.writes: allowed`` and adds the
        three keys that lift the floor.

        ``remove_deny`` rather than a bare absence of the deny is the whole
        mechanism — string lists UNION when a tier is laid over the profile, so
        the deployment's deny reaches this tier too and can only be taken back
        by subtraction. The deny is therefore expected to be present HERE as
        well; what differs is that it is cancelled."""
        profile = layers["tiers/admin"]
        assert profile.name == "Control Assistant (Admin)"
        assert _profile_data_bundle(profile) == "control_assistant"
        assert profile.config.get(WRITES_KEY) == "allowed"
        assert not _posture_keys(profile.config) - {WRITES_KEY}
        assert "control_system" not in _own_config("tiers", "admin")

        assert profile.config.get(DENY_KEY) == [SETUP_PATCH_TOOL]
        assert profile.config.get(REMOVE_DENY_KEY) == [SETUP_PATCH_TOOL]
        assert profile.config.get(CONFIG_PANEL_KEY) is True
        assert profile.config.get(GALLERY_WRITE_KEY) is True

        # A full operator desk plus editing: the admin tier adds the same
        # write-oriented panels the readwrite tier does, on top of the profile's.
        base = layers["profile"]
        assert set(profile.web_panels) == set(base.web_panels) | set(WRITE_TIER_PANELS)

    def test_base_floors_the_deployment_privileges(self, layers: dict[str, BuildProfile]) -> None:
        """The deployment ships the restricted side of every deployment axis.

        This is what makes a privilege something a tier has to ask for by
        name: a new tier that says nothing inherits no ``setup_patch`` tool, no
        Config panel and no gallery editors. Both control-system tiers are
        checked too — a floor that only held on the deployment while a tier
        quietly re-enabled a surface would pass a deployment-only assertion and
        still be broken. Every app inherits the same floor."""
        base = resolve_preset("control-assistant")
        assert base.config.get(DENY_KEY) == [SETUP_PATCH_TOOL]
        assert base.config.get(CONFIG_PANEL_KEY) is False
        assert base.config.get(GALLERY_WRITE_KEY) is False
        # The floor is a deny, never a deployment-level `remove_ask`: an
        # inherited `remove_ask` would strip the approval prompt from EVERY
        # tier, admin included, because string lists cannot be subtracted
        # downward.
        assert "claude_code.permissions.remove_ask" not in base.config

        unprivileged = [f"tiers/{name}" for name in ("readonly", "readwrite")]
        unprivileged += [f"tiers/{name}" for name in SHARED_LOGIN_NAMES]
        for key in unprivileged:
            layer = layers[key]
            assert SETUP_PATCH_TOOL in layer.config.get(DENY_KEY), key
            assert REMOVE_DENY_KEY not in layer.config, key
            assert layer.config.get(CONFIG_PANEL_KEY) is False, key
            assert layer.config.get(GALLERY_WRITE_KEY) is False, key
            assert ADMIN_ONLY_SKILL not in layer.skills, key

    def test_admin_differs_from_readwrite_only_on_the_ceiling_and_the_deployment_axis(
        self, layers: dict[str, BuildProfile]
    ) -> None:
        """Admin sits directly on top of readwrite: the ceiling above it, plus
        deployment editing. Asserted wholesale so a further difference cannot
        creep in unnoticed.

        The ceiling and the deployment axes — subtracted before the comparison —
        are the whole difference. The EVENTS/BLUESKY panels are deliberately NOT among them:
        the admin tier is a full operator desk plus deployment editing, so it
        adds the same write-oriented panels readwrite does (asserted with the
        artifact lists — ``web_panels`` is a list, not config, so it never
        appears in the config comparison below)."""
        rw_cfg = dict(layers["tiers/readwrite"].config)
        ad_cfg = dict(layers["tiers/admin"].config)
        # The ceiling: only admin offers the position that does not prompt.
        # Both tiers are otherwise the same desk over the same machines.
        assert rw_cfg.pop(WRITES_KEY) == "asked" and ad_cfg.pop(WRITES_KEY) == "allowed"
        # Deployment axis 1 — the agent's tool. The deployment's deny reaches
        # BOTH tiers identically; what makes admin different is the subtraction.
        assert rw_cfg[DENY_KEY] == [SETUP_PATCH_TOOL]
        assert ad_cfg[DENY_KEY] == [SETUP_PATCH_TOOL]
        assert REMOVE_DENY_KEY not in rw_cfg, "the readwrite tier must not lift the floor"
        assert ad_cfg.pop(REMOVE_DENY_KEY) == [SETUP_PATCH_TOOL]
        # Deployment axes 2 and 3 — the browser surfaces. Inherited by both
        # tiers, so these differ on VALUE rather than presence.
        for key in ADMIN_LIFTED_FLOOR_KEYS:
            assert rw_cfg.pop(key) is False, f"the readwrite tier must not enable {key}"
            assert ad_cfg.pop(key) is True
        # The write-oriented panels ride both tiers' `web_panels` lists, not
        # their config: their URLs are projected by the build, so neither tier
        # may spell one.
        for panel in WRITE_TIER_PANELS:
            for cfg in (rw_cfg, ad_cfg):
                assert not any(key.startswith(f"web.panels.{panel}.") for key in cfg), (
                    f"no tier may pin web.panels.{panel}.* — the build projects it"
                )
        assert ad_cfg == rw_cfg

    def test_tiers_retain_the_deployments_config(self, layers: dict[str, BuildProfile]) -> None:
        """Toggling the write switch must not drop the deployment's own config
        — a representative key survives every tier laid over it.

        The representative is the deployment's session baseline, which is the
        sandbox simulator: the one machine on this deployment where a write is
        harmless, while the stand-in the preset also declares is a target the
        operator switches to on purpose. No tier names ``control_system.type``,
        so every tier keeps that baseline."""
        for tier in _person_tiers():
            assert "control_system.type" not in _own_config("tiers", tier), tier
            assert layers[f"tiers/{tier}"].config.get("control_system.type") == (
                "virtual_accelerator"
            ), tier

    def test_tiers_differ_only_on_the_tier_contract(self, layers: dict[str, BuildProfile]) -> None:
        ro_cfg = dict(layers["tiers/readonly"].config)
        rw_cfg = dict(layers["tiers/readwrite"].config)
        # Axis 1 — enforcement: the write posture, one word on the tier key,
        # `never` against `asked`. No retired per-type leaf survives the
        # load, so nothing else about the posture can differ. The deployment's
        # per-type limits block rides along on every tier by design.
        assert ro_cfg.pop(WRITES_KEY) == "never"
        assert rw_cfg.pop(WRITES_KEY) == "asked"
        assert not _posture_keys(ro_cfg) and not _posture_keys(rw_cfg)
        # Axis 2 — write-oriented panels — is not a config axis at all: the
        # tabs ride the write-armed tiers' `web_panels` lists (asserted in
        # test_tiers_share_every_artifact_list_except_panels_and_the_admin_skill),
        # and their URLs are projected from the hosting deployment's render
        # rather than spelled in any preset.
        for panel in WRITE_TIER_PANELS:
            for cfg in (ro_cfg, rw_cfg):
                assert not any(key.startswith(f"web.panels.{panel}.") for key in cfg), (
                    f"no tier may pin web.panels.{panel}.* — the build projects it"
                )
        # With the tier-contract keys removed, the tiers are identical.
        assert ro_cfg == rw_cfg

    def test_tiers_share_every_artifact_list_except_panels_and_the_admin_skill(
        self, layers: dict[str, BuildProfile]
    ) -> None:
        """No tier is defined by *tool* removal — rules, hooks, agents and
        output styles are the deployment's own on all three tiers (the tier
        boundary is enforcement, not a stripped-down agent). Exactly two lists
        are allowed to differ, and both differ by ADDITION:

        * panels — the write-armed readwrite and admin tiers add the
          write-oriented EVENTS/BLUESKY tabs (their URLs are projected by the
          build, not declared), and the readonly tier is built without them;
        * skills — the admin tier adds ``setup-mode``, the guided workflow
          that edits config.yml and .mcp.json, which is the agent-side half of
          the privilege its config keys turn on.

        A list that SHRANK on a tier would be the failure mode this guards: it
        would make a tier a different agent rather than the same agent under a
        different posture. That is what an app is, and why an app is not a
        tier."""
        base = layers["profile"]
        readonly, readwrite, admin = (
            layers[f"tiers/{n}"] for n in ("readonly", "readwrite", "admin")
        )
        for tier in (readonly, readwrite, admin):
            assert tier.rules == base.rules
            assert tier.hooks == base.hooks
            assert tier.agents == base.agents
            assert tier.output_styles == base.output_styles
        assert readonly.skills == base.skills
        assert readwrite.skills == base.skills
        # UNION, not replacement: the deployment's selection survives underneath.
        assert admin.skills == [*base.skills, ADMIN_ONLY_SKILL]
        assert ADMIN_ONLY_SKILL not in base.skills
        assert readonly.web_panels == base.web_panels
        for tier in (readwrite, admin):
            assert set(tier.web_panels) == set(base.web_panels) | set(WRITE_TIER_PANELS)

    @pytest.mark.parametrize("tier", ["readonly", "readwrite", "admin"])
    def test_tiers_page_names_the_panels_each_tier_declares(
        self, tier: str, layers: dict[str, BuildProfile]
    ) -> None:
        """The tiers page's "What the screen looks like" column says, per tier,
        whether the write-oriented panels are there, and the tier decides it.

        The page is what a deployer reads to choose a login for a teammate. A
        tier that gains or drops the panels changes that answer, so the row
        has to move with it. Read by tab label, because that is what the
        page names."""
        declared = set(WRITE_TIER_PANELS) <= set(layers[f"tiers/{tier}"].web_panels)
        cell = _tier_screen_cells()[tier]
        assert _PANEL_CLAIM.findall(cell) == (["with"] if declared else ["without"]), (
            f"the tiers page's {tier} row says {cell!r}, but the {tier} tier "
            f"{'adds' if declared else 'does not add'} the EVENTS and BLUESKY panels"
        )

    def test_safety_chain_hooks_are_shipped(self, layers: dict[str, BuildProfile]) -> None:
        """The write-capable tier is supervised, not unguarded: the hooks that
        gate a write (writes-check, limits, approval) ship in the deployment
        and reach every tier."""
        base = resolve_preset("control-assistant")
        for hook in ("writes-check", "limits", "approval"):
            assert hook in base.hooks
            for tier in _person_tiers():
                assert hook in layers[f"tiers/{tier}"].hooks, (tier, hook)

    def test_tiers_and_apps_render_attached(self, layers: dict[str, BuildProfile]) -> None:
        """Every tier and app renders attached: ``deploy_services: false`` and
        no web tier of its own, so it builds a terminal image only, the
        injector blocks the deployment's profile carries skip cleanly, and no
        render races the hosting project for the web ports. The hosting
        deployment keeps the self-contained posture.

        The build decides this from where the file sits, so no bundled tier or
        app states either key itself."""
        assert layers["profile"].deploy_services is True
        for kind, names in (("tiers", _person_tiers()), ("tiers", list(SHARED_LOGIN_NAMES))):
            for name in names:
                layer = layers[f"{kind}/{name}"]
                assert layer.deploy_services is False, name
                assert layer.config.get("modules.web_terminals.enabled") is False, name
                own = _bundled_layer(kind, name)
                assert "deploy_services" not in own, name
                assert not any(
                    str(key).startswith("modules.web_terminals") for key in own.get("config") or {}
                ), name

    def test_tiers_and_apps_pin_no_service_address(self) -> None:
        """No tier or app spells where the hosting deployment's services are.

        They render attached, and the build tells each render every
        client-facing fact from the hosting deployment's own render
        (``osprey.deployment.reach``): the graph store's bolt port, the qmd
        sidecar's port, the Postgres, the telemetry store, the bridge, the
        EVENTS and BLUESKY tab URLs. A file that pinned one would be a second
        copy of a number the host already states, free to drift when an
        operator moves the service.

        Read from each bundled file rather than from the loaded layer: the
        profile beneath declares the services the deployment runs, so every
        layer INHERITS those keys. What must stay empty is what a tier or app
        says for itself.
        """
        for kind, names in (("tiers", _person_tiers()), ("tiers", list(SHARED_LOGIN_NAMES))):
            for name in names:
                own = _own_config(kind, name)
                pinned = sorted(
                    key
                    for key in own
                    if str(key).startswith(
                        ("services.", "web.panels.events.", "web.panels.bluesky.")
                    )
                )
                assert pinned == [], f"{kind}/{name} pins {pinned}"
                assert "services" not in own, f"{kind}/{name}"
                assert "services" not in _bundled_layer(kind, name), f"{kind}/{name}"

    @pytest.mark.parametrize("tier", ("readonly", "readwrite", "admin"))
    def test_every_tier_renders_the_graph_server(
        self, built_layered_stack: Path, tier: str
    ) -> None:
        """Every tier's terminal can query the hosting deployment's graph.

        Asserted on a real build rather than on the loaded profile, because the
        claim spans the whole pipeline: the store's port has to be projected
        from the hosting deployment's render (``osprey.deployment.reach``),
        land as a ``services.graphdb`` block in the rendered config through
        ``_apply_config_overrides``, be seen by ``config_derived_context`` as
        ``graphdb_configured``, and only then reach ``resolve_servers`` before
        the Claude Code artifacts are written. A projection applied after
        server resolution would leave every assertion below false.

        The whole main-agent surface is checked — permissions, a launchable
        ``.mcp.json`` entry and the PostToolUse prefix, with nothing behind
        ``ask`` because these tools read — together with the fact that the
        server stops there: it is a main-agent server, so the channel-finder
        subagent's frontmatter must not name one of its tools.
        """
        project = _render_dir(built_layered_stack, tier)
        assert project.is_dir(), f"{tier} was never rendered"

        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        host = rendered_config(built_layered_stack)
        # The store's address, projected from the host, beside the corpus the
        # tier stages for itself (render-local, never stripped).
        assert (config.get("services") or {}).get("graphdb") == {
            "port_host": default_port("graphdb_bolt"),
            "ttl_path": host["services"]["graphdb"]["ttl_path"],
        }

        permissions = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]
        assert _graph_entries(permissions["allow"]) == graph_permission_entries()
        assert _graph_entries(permissions.get("ask") or []) == []

        graph_server = json.loads((project / ".mcp.json").read_text())["mcpServers"]["graph"]
        assert graph_server["args"] == ["-m", "osprey.mcp_server.graph"]

        hook_cfg = json.loads((project / ".claude" / "hooks" / "hook_config.json").read_text())
        assert "mcp__graph__" in hook_cfg["server_prefixes"]
        assert "mcp__graph__" not in hook_cfg["approval_prefixes"]

        frontmatter = yaml.safe_load(
            (project / ".claude" / "agents" / "channel-finder.md")
            .read_text(encoding="utf-8")
            .split("---")[1]
        )
        tools = [entry.strip() for entry in str(frontmatter["tools"]).split(",") if entry.strip()]
        assert _graph_entries(tools) == []

    @pytest.mark.parametrize("tier", ("readonly", "readwrite", "admin"))
    def test_every_tier_denies_monitor_and_enterworktree(
        self, built_layered_stack: Path, tier: str
    ) -> None:
        """Every tier's session refuses a background shell and a new worktree.

        ``Monitor`` runs shell commands in the background and ``EnterWorktree``
        creates a git worktree on disk, so both sit on the interactive write
        floor beside ``Bash`` and ``Edit``: the read-only tier refuses them, and
        the write-capable tiers refuse them too.
        """
        project = _render_dir(built_layered_stack, tier)
        assert project.is_dir(), f"{tier} was never rendered"

        settings = json.loads((project / ".claude" / "settings.json").read_text(encoding="utf-8"))
        deny = settings["permissions"]["deny"]
        assert "Monitor" in deny, f"{tier} does not deny Monitor: {deny}"
        assert "EnterWorktree" in deny, f"{tier} does not deny EnterWorktree: {deny}"

    @pytest.mark.parametrize("tier", ("readonly", "readwrite"))
    def test_projected_facts_land_inside_the_attached_renders_services_map(
        self, built_layered_stack: Path, tier: str
    ) -> None:
        """The projected keys are written *into* the attached render's empty
        ``services`` map, carry the host's values, and leave nothing else behind.

        An attached render scaffolds no services of its own — the app template
        gives it ``services: {}`` — so after the projection that map must hold
        exactly the facts this tier needs about the hosting deployment's
        services, each equal to the hosting render's, and nothing more: not the
        host's own service knobs (image, heap, retention), and no literal
        top-level ``"services.graphdb.port_host"`` string key beside the map,
        which would satisfy ``config.yml`` as YAML and be read by nobody.

        Most entries are the address a consumer in this container dials.
        ``archiver_recorder`` is one exception and is asserted here beside
        them: nothing in a tier dials the recorder, but ``path`` is the host's
        fact THAT it records, which is what ``archive_belongs_to_standin``
        reads to refuse the ``live`` target — a gate that has to hold in a
        multi-user session exactly as it does in a single-user one, and whose
        host-side spelling (``deployed_services``) is empty in every attached
        render. The graph store's ``ttl_path`` is the other: not projected but
        kept, because it names the corpus in the render's OWN data tree, from
        which the build derives this render's search index (the graphdb
        contract's ``render_local`` keys).
        """
        project = _render_dir(built_layered_stack, tier)
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        host = rendered_config(built_layered_stack)

        assert config["services"] == {
            "qmd": {"port": host["services"]["qmd"]["port"]},
            "graphdb": {
                "port_host": host["services"]["graphdb"]["port_host"],
                "ttl_path": host["services"]["graphdb"]["ttl_path"],
            },
            "postgresql": {
                "port_host": host["services"]["postgresql"]["port_host"],
                "username": host["services"]["postgresql"]["username"],
                "database_name": host["services"]["postgresql"]["database_name"],
            },
            # The single lane declares no `target`: it is addressed by the
            # fallback, the simulator the baseline names.
            "bluesky": {"port": host["services"]["bluesky"]["port"]},
            "virtual_accelerator": {"port": host["services"]["virtual_accelerator"]["port"]},
            "live_standin": {"port": host["services"]["live_standin"]["port"]},
            "archiver_recorder": {"path": host["services"]["archiver_recorder"]["path"]},
        }
        assert config["services"]["qmd"]["port"] == QMD_DEFAULT_PORT
        assert [key for key in config if "." in str(key)] == []

    def test_readwrite_is_told_its_panel_urls(self, built_layered_stack: Path) -> None:
        """The EVENTS and BLUESKY tabs the readwrite tier adds are told
        exactly what the hosting render carries for them — where the dispatch
        and bluesky-web injectors derived them — not what any preset spells.
        Every projected leaf matches, absent ones included — the whole entry
        the dispatch and bluesky-web injectors derive (url, path, label, and
        the health endpoint where the service serves one) and nothing more."""
        config = yaml.safe_load(
            (_render_dir(built_layered_stack, "readwrite") / "config.yml").read_text(
                encoding="utf-8"
            )
        )
        host = rendered_config(built_layered_stack)
        for panel in ("events", "bluesky"):
            told = config["web"]["panels"][panel]
            hosts = host["web"]["panels"][panel]
            assert told["url"] == hosts["url"], panel
            for leaf in ("path", "label", "health_endpoint"):
                assert told.get(leaf) == hosts.get(leaf), (panel, leaf)
        readonly = yaml.safe_load(
            (_render_dir(built_layered_stack, "readonly") / "config.yml").read_text(
                encoding="utf-8"
            )
        )
        assert "events" not in readonly["web"]["panels"]
        assert "bluesky" not in readonly["web"]["panels"]

    @pytest.mark.parametrize("render", ("readonly", "readwrite", "logbook"))
    def test_attached_renders_resolve_the_qmd_sidecar(
        self, built_layered_stack: Path, render: str
    ) -> None:
        """Every render's config names the sidecar its hybrid search dials.

        Asserted on a real build and through the client's own resolver, because
        this is the seam the query-time failure lives on: the app template
        enables ``ariel.search_modules.hybrid`` for the render, and the module
        resolves its endpoint from ``services.qmd`` of the *same* config. A
        render with one and not the other is a terminal whose logbook search
        fails on every query, and nothing before this test looked at both
        halves together.
        """
        project = _render_dir(built_layered_stack, render)
        assert project.is_dir(), f"{render} was never rendered"
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))

        hybrid = config["ariel"]["search_modules"]["hybrid"]
        assert hybrid["enabled"] is True, "the template no longer enables hybrid search"

        resolved = resolve_qmd_service_config(config)
        assert resolved is not None, f"{render}: hybrid search is on but no services.qmd"
        assert resolved.port == QMD_DEFAULT_PORT
        assert resolved.base_url == f"http://127.0.0.1:{QMD_DEFAULT_PORT}"

    def test_logbook_app_renders_the_graph_surface(self, built_layered_stack: Path) -> None:
        """The logbook app answers from the logbook AND the facility knowledge
        graph: it dials the host's store and keeps the graph server on.

        Asserted on the real render, because "on" is a pipeline claim: the
        build tells every attached render where the hosting deployment's
        services are, and a store the render knows how to dial is what makes
        the graph server render. So the store's bolt address is projected into
        this render beside its own corpus key, the server is in ``.mcp.json``
        and its read tools are allowed, the graph subagent and the ``safety``
        rule come with the app, and nothing that reaches the machine does: no
        controls, python, bluesky or health server, no JUPYTER tab, no
        control-target picker, and every write switch off.
        """
        project = _render_dir(built_layered_stack, "logbook")
        assert project.is_dir(), "the logbook app was never rendered"
        assert "claude_code.servers.graph.enabled" not in _own_config("tiers", "logbook")

        servers = json.loads((project / ".mcp.json").read_text())["mcpServers"]
        assert set(servers) == {"ariel", "graph", "osprey_workspace"}

        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        graphdb = (config.get("services") or {}).get("graphdb") or {}
        assert graphdb.get("port_host") == default_port("graphdb_bolt"), graphdb

        permissions = json.loads((project / ".claude" / "settings.json").read_text())["permissions"]
        assert _graph_entries(permissions["allow"]) == graph_permission_entries()

        agents = sorted(p.stem for p in (project / ".claude" / "agents").glob("*.md"))
        assert "facility-knowledge-graph" in agents
        rules = sorted(p.stem for p in (project / ".claude" / "rules").glob("*.md"))
        assert "safety" in rules

        assert "jupyter" not in config["web"]["panels"]
        assert config["web"]["control_target_picker"] is False
        section = config["control_system"]
        assert tier_writes(section) == "never"
        assert "writes_enabled" not in section

    def test_knowledge_app_renders_the_knowledge_surface_and_nothing_else(
        self, built_layered_stack: Path
    ) -> None:
        """The knowledge app is the graph, the graph channel finder and the
        knowledge bundle — and no control system, sandbox or logbook.

        Asserted on the real render: the servers in ``.mcp.json`` are the
        three knowledge servers plus the workspace, the channel finder runs in
        graph mode, the three knowledge agents are rendered, the KNOWLEDGE
        panel is the default, and the one bundle-writing tool is denied.
        """
        project = _render_dir(built_layered_stack, "knowledge")
        assert project.is_dir(), "the knowledge app was never rendered"

        servers = json.loads((project / ".mcp.json").read_text())["mcpServers"]
        assert set(servers) == {
            "graph",
            "channel-finder",
            "osprey_facility_knowledge",
            "osprey_workspace",
        }
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        assert config["channel_finder"]["pipeline_mode"] == "graph"
        assert config["web"]["default_panel"] == "okf"
        assert "ariel" not in config["web"]["panels"]
        assert "jupyter" not in config["web"]["panels"]
        assert config["web"]["control_target_picker"] is False

        agents = sorted(p.stem for p in (project / ".claude" / "agents").glob("*.md"))
        assert agents == ["channel-finder", "facility-knowledge", "facility-knowledge-graph"]

        settings = json.loads((project / ".claude" / "settings.json").read_text())
        assert "mcp__osprey_facility_knowledge__draft_concept" in settings["permissions"]["deny"]

    def test_channel_finder_app_renders_the_file_based_finder_and_no_graph(
        self, built_layered_stack: Path
    ) -> None:
        """The channel-finder app is the hierarchical, file-based channel
        finder and nothing else: no graph — the store is not dialled and the
        server is not rendered — no logbook, no control system.
        """
        project = _render_dir(built_layered_stack, "channel-finder")
        assert project.is_dir(), "the channel-finder app was never rendered"

        servers = json.loads((project / ".mcp.json").read_text())["mcpServers"]
        assert set(servers) == {"channel-finder", "osprey_workspace"}
        config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
        assert config["channel_finder"]["pipeline_mode"] == "hierarchical"
        assert config["web"]["default_panel"] == "channel-finder"
        graphdb = (config.get("services") or {}).get("graphdb") or {}
        assert set(graphdb) <= {"ttl_path"}, graphdb
        assert "jupyter" not in config["web"]["panels"]
        assert config["web"]["control_target_picker"] is False

        agents = sorted(p.stem for p in (project / ".claude" / "agents").glob("*.md"))
        assert agents == ["channel-finder"]
        hits = sorted(
            str(path.relative_to(project))
            for path in (project / ".claude").rglob("*")
            if path.is_file()
            and "mcp__graph__" in path.read_text(encoding="utf-8", errors="ignore")
        )
        assert hits == [], f"the channel-finder app dials no graph store but rendered {hits}"


def _shipped_profile(key: str, layers: dict[str, BuildProfile]) -> BuildProfile:
    """The profile one matrix row names: a deployment preset by name, or a
    bundled tier or app (``tiers/<n>``, ``apps/<n>``) loaded over the
    deployment."""
    return layers[key] if key in layers else resolve_preset(key)


def _rendered_section(tmp_path: Path, key: str, layers: dict[str, BuildProfile], section: str):
    """One top-level section the row's ``config:`` renders to.

    The resolver takes the section, not the dotted overrides, so the dotted keys
    have to be written through ``config_update_fields`` first — the same writer
    the build calls — or ``control_system.connector.epics.writes_enabled`` would
    reach it as a top-level string key that nothing reads.
    """
    if key in layers:
        rendered = _render_config_overrides(tmp_path, {"system": {}}, config=layers[key].config)
    else:
        rendered = _render_config_overrides(tmp_path, {"system": {}}, preset=key)
    return rendered.get(section)


def _pinned_rows_match_the_shipped_set(pinned: dict) -> None:
    """Every bundled tier and app has a row, exactly, and every deployment row
    is a shipped deployment preset (the deployment listing itself is pinned in
    ``tests/cli/test_preset_listings.py``)."""
    assert sorted(key for key in pinned if key.startswith("tiers/")) == sorted(
        f"tiers/{name}" for name in list_tiers()
    )
    deployments = {key for key in pinned if "/" not in key}
    assert deployments == {"control-assistant", "hello-world"}
    assert deployments <= set(list_presets())


# ---------------------------------------------------------------------------
# Write posture across every shipped deployment, tier and app
# ---------------------------------------------------------------------------

#: row -> the ``control_system.writes`` it renders. A row is a deployment
#: preset by name, or a bundled tier or app as it loads over the
#: control-assistant deployment (``tiers/<name>``, ``apps/<name>``).
#:
#: The rendered answer rather than the keys a file spells: a row reaches its
#: posture through the tier key, through the retired ``writes_enabled`` leaves
#: the normalizer rewrites to it, or through saying nothing (``never``), and
#: what an operator holds is ``tier_writes`` of the rendered section — one word
#: for every machine the deployment configures, the stand-in included. Stated
#: literally and never derived from the resolver: a table computed from the
#: code under test would pin nothing.
#:
#: Every shipped deployment, tier and app appears, so a new one has to state
#: its posture here before it ships rather than inheriting a silent one.
PINNED_WRITE_POSTURE: dict[str, str] = {
    # The hosting preset arms writes at the prompting position; the read-only
    # tier is what takes them back, and this row is what makes that tier's
    # `never` load-bearing.
    "control-assistant": "asked",
    # No `control_system:` posture of its own; nothing has a write path to lose.
    "hello-world": "never",
    # The write-capable tiers.
    "tiers/admin": "allowed",
    "tiers/readwrite": "asked",
    "tiers/readonly": "never",
    # Every app renders `never`, whatever its host or its file says: the
    # composer pins it.
    "tiers/channel-finder": "never",
    "tiers/knowledge": "never",
    "tiers/logbook": "never",
}


class TestWritePostureMatrix:
    """What each shipped deployment, tier and app means by "may this session
    write".

    The boundary is the tier key, ``control_system.writes``: one word a render
    holds for every machine it configures. The tests below read it the way
    every write surface reads it — ``tier_writes`` on the rendered section —
    rather than through the key a file happens to spell, and hold that no
    retired ``writes_enabled`` leaf survives a render."""

    @pytest.mark.parametrize("row", sorted(PINNED_WRITE_POSTURE))
    def test_every_shipped_row_resolves_the_posture_it_is_pinned_to(
        self, tmp_path: Path, row: str, layers: dict[str, BuildProfile]
    ) -> None:
        # Arrange
        section = _rendered_section(tmp_path, row, layers, "control_system") or {}

        # Act
        resolved = tier_writes(section)

        # Assert
        assert resolved == PINNED_WRITE_POSTURE[row]
        assert not _RETIRED_WRITE_KEY.search(json.dumps(section)), section

    def test_the_shipped_set_is_pinned(self) -> None:
        """A new tier or app must state its posture here before it ships.

        Without this the parametrization above would simply not cover it, and
        the matrix would degrade into a sample as tiers and apps are added."""
        _pinned_rows_match_the_shipped_set(PINNED_WRITE_POSTURE)

    def test_readonly_states_never(self) -> None:
        """The read-only tier states ``never`` in its own file, so no root it
        is laid over can arm it: the tier key is one word, and the tier's wins."""
        own = _own_config("tiers", "readonly")

        assert _stated_writes(own) == "never"
        # Flat dotted keys, never a nested `control_system:` mapping — which
        # would replace the rendered subtree and drop the sibling keys.
        assert "control_system" not in own

    def test_every_app_states_never(self) -> None:
        """Each app states the write boundary in its own file, the same way:
        ``never``, which is also what the composer pins on every app render."""
        stated = {
            app: _stated_writes(_own_config("tiers", app)) for app in list(SHARED_LOGIN_NAMES)
        }

        assert stated == dict.fromkeys(list(SHARED_LOGIN_NAMES), "never")

    @pytest.mark.parametrize("app", ["channel-finder", "knowledge", "logbook"])
    def test_every_app_keeps_the_deployments_targets(
        self, tmp_path: Path, app: str, layers: dict[str, BuildProfile]
    ) -> None:
        """The write pin changes what an app may do, never which machines the
        deployment has: an app's configured targets are the deployment's."""
        (tmp_path / "app").mkdir()
        (tmp_path / "root").mkdir()
        section = _rendered_section(tmp_path / "app", f"tiers/{app}", layers, "control_system")
        hosting = _rendered_section(tmp_path / "root", "profile", layers, "control_system")

        assert configured_targets(section) == configured_targets(hosting)

    @pytest.mark.parametrize(("tier", "ceiling"), [("readwrite", "asked"), ("admin", "allowed")])
    def test_the_write_capable_tiers_state_their_ceiling_and_render_it(
        self, tmp_path: Path, layers: dict[str, BuildProfile], tier: str, ceiling: str
    ) -> None:
        """Only admin offers the position that does not prompt: readwrite's
        ceiling is ``asked``, and each states its word in the tier key."""
        assert layers["tiers/readwrite"].name == "Control Assistant (Read-Write)"
        own = _own_config("tiers", tier)
        assert _stated_writes(own) == ceiling
        assert "control_system" not in own

        section = _rendered_section(tmp_path, f"tiers/{tier}", layers, "control_system")
        assert tier_writes(section) == ceiling

    @pytest.mark.parametrize("tier", ["readonly", "readwrite"])
    def test_the_reference_deployments_tiers_carry_the_bundled_posture(self, tier: str) -> None:
        """A hand-kept tier file states the posture its bundled tier ships.

        Nothing renders these copies, so no build notices when a bundled
        tier's posture moves and the copy does not: this is the one place that
        holds the copy to its source."""
        text = (EXEMPLAR_TIERS_DIR / f"{tier}.yml").read_text(encoding="utf-8")
        assert f"Copied from the bundled `{tier}` tier" in text

        copy = yaml.safe_load(text)["config"]
        assert _stated_writes(copy) == _stated_writes(_own_config("tiers", tier))


# ---------------------------------------------------------------------------
# The title-bar control-target picker
# ---------------------------------------------------------------------------

#: row -> the ``web.control_target_picker`` it states, or None where it states
#: nothing and the key's default (on) holds. Rows as in
#: :data:`PINNED_WRITE_POSTURE`; every shipped deployment, tier and app
#: appears, so a new one has to state its answer here before it ships.
PINNED_CONTROL_TARGET_PICKER: dict[str, bool | None] = {
    "control-assistant": None,
    "hello-world": None,
    "tiers/admin": None,
    "tiers/readonly": None,
    "tiers/readwrite": None,
    "tiers/channel-finder": False,
    "tiers/knowledge": False,
    "tiers/logbook": False,
}


class TestControlTargetPicker:
    """Which shipped deployments, tiers and apps show the control-target
    picker, and why those that do not may leave it out: nothing in them
    reaches the machine."""

    @pytest.mark.parametrize("row", sorted(PINNED_CONTROL_TARGET_PICKER))
    def test_every_shipped_row_states_the_picker_it_is_pinned_to(
        self, tmp_path: Path, row: str, layers: dict[str, BuildProfile]
    ) -> None:
        web = _rendered_section(tmp_path, row, layers, "web") or {}

        assert web.get("control_target_picker") is PINNED_CONTROL_TARGET_PICKER[row]

    def test_the_shipped_set_is_pinned(self) -> None:
        _pinned_rows_match_the_shipped_set(PINNED_CONTROL_TARGET_PICKER)

    @pytest.mark.parametrize(
        "row", sorted(name for name, on in PINNED_CONTROL_TARGET_PICKER.items() if on is False)
    )
    def test_a_row_without_the_picker_reaches_no_machine(
        self, row: str, layers: dict[str, BuildProfile]
    ) -> None:
        """The picker is off only where no tool server and no panel reaches the
        control target: no controls, python or bluesky server, and no JUPYTER
        tab, whose kernels read and write through the target."""
        profile = _shipped_profile(row, layers)

        for server in ("controls", "python", "bluesky"):
            stated = profile.config.get(f"claude_code.servers.{server}.enabled")
            enabled = FRAMEWORK_SERVERS[server].default_enabled if stated is None else stated
            assert enabled is False, server
        assert "jupyter" not in profile.web_panels


# ---------------------------------------------------------------------------
# Web-terminal context overlay (seeding's base.md requirement)
# ---------------------------------------------------------------------------


class TestWebTerminalContextShipped:
    """Every built project — not just the ``control_assistant`` bundle —
    carries the ``docker/web-terminal-context/base.md`` that seeding
    requires. The framework ships a generic FALLBACK from its template root:
    ``modules.web_terminals.enabled`` is a config key any profile can turn
    on, so any bundle may end up seeding a user, and without a baseline
    ``osprey up`` brings up the whole stack and then aborts at the seed step.
    A profile's own ``web-terminal-context/base.md`` overrides the fallback.

    The path is PROJECT-relative (``seeding._CONTEXT_RELPATH``), and in a
    deployment repo the rendered project is the ``build/`` zone — which is why
    the seeding test below places the render there rather than treating it as
    the repo root."""

    def test_built_project_ships_base_md(self, tmp_path: Path) -> None:
        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        project_dir = create_project_as_built(
            TemplateManager(),
            project_name="ctx-ship-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        base_md = project_dir / seeding._CONTEXT_RELPATH / "base.md"
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""

    def test_base_md_is_package_data(self) -> None:
        """The template source is reachable through ``importlib.resources``, so
        it is packaged rather than only present in a source checkout — a build
        from an installed wheel has nothing to copy otherwise."""
        from importlib.resources import files

        base_md = files("osprey.templates").joinpath("claude_code/web-terminal-context/base.md")
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""

    def test_non_control_assistant_bundle_ships_base_md(self, tmp_path: Path) -> None:
        """A bundle that ships no render content of its own still gets the
        framework baseline."""
        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        project_dir = create_project_as_built(
            TemplateManager(),
            project_name="ctx-ship-hello",
            output_dir=tmp_path,
            data_bundle="hello_world",
        )
        base_md = project_dir / seeding._CONTEXT_RELPATH / "base.md"
        assert base_md.is_file()
        assert base_md.read_text(encoding="utf-8").strip() != ""

    def test_non_control_assistant_bundle_seeds_successfully(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A seeded roster on a non-control_assistant bundle reaches the
        CLAUDE.md write instead of aborting on a missing base.md — the
        pre-flight ``RuntimeError`` such a project hit before base.md became
        framework-layer.

        The render is placed where a deployment repo keeps it — as the repo's
        ``build/`` zone — because that is where seeding resolves the overlay
        tree. Seeding from the repo root is the deploy verbs' working
        directory, so this exercises the real pairing rather than a bare
        project directory no deployment has.
        """
        import shutil
        import subprocess

        from osprey.cli.templates.manager import TemplateManager
        from osprey.deployment.web_terminals import seeding

        rendered = create_project_as_built(
            TemplateManager(),
            project_name="ctx-seed-hello",
            output_dir=tmp_path,
            data_bundle="hello_world",
        )
        repo = tmp_path / "repo"
        repo.mkdir()
        project_dir = repo / "build"
        shutil.move(str(rendered), str(project_dir))
        base_content = (project_dir / seeding._CONTEXT_RELPATH / "base.md").read_text(
            encoding="utf-8"
        )

        container = web_container_name("ctx-seed-hello", "alice")
        seeded: list[bytes | None] = []

        def _fake_run(argv, capture_output=True, text=False, env=None, check=False, input=None):  # noqa: ARG001 - the keywords subprocess.run is called with
            if argv[1] == "inspect":
                rc = 0 if argv[-1] == container else 1
                return subprocess.CompletedProcess(argv, returncode=rc, stdout="", stderr="")
            if argv[1] == "exec" and "id -u" in argv[-1]:
                return subprocess.CompletedProcess(
                    argv, returncode=0, stdout="1000:1000\n", stderr=""
                )
            if len(argv) >= 9 and "cat >" in argv[8]:
                seeded.append(input)
            return subprocess.CompletedProcess(argv, returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(seeding.subprocess, "run", _fake_run)
        monkeypatch.setattr(seeding, "get_runtime_command", lambda config=None: ["docker"])
        monkeypatch.setattr(seeding, "runtime_env", lambda config, base_env=None, **kw: {})
        monkeypatch.chdir(repo)

        seeding.seed_user_containers(
            {
                "project_name": "ctx-seed-hello",
                "facility": {"name": "Demo", "prefix": "dls"},
                "system": {"timezone": "UTC"},
                "modules": {
                    "web_terminals": {
                        "enabled": True,
                        "users": ["alice"],
                        "default_tier": "assistant",
                        "renders": {
                            "assistant": tier_render("assistant", "ctx-seed-hello-assistant")
                        },
                    }
                },
            }
        )

        assert seeded == [base_content.encode("utf-8")]
