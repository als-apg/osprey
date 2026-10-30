"""An app render is composed from a complete app file and its host deployment.

The app file owns what the agent IS — its lists, its brief, its panel, its
CLAUDE.md template and its channel-finder mode — and its own ``config:`` sits
over the host's facts: the host's config minus the host's server switches and
minus every write switch. Permissions are kept and unioned, so the host's floor
deny survives into every app. Whatever the two say, an app render cannot write:
the composer pins ``control_system.writes: never``, keeps no retired
``writes_enabled`` leaf, and selects ``writes-check``. Every
other top-level key, the data bundle and the provenance are the host's.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_cmd import _profile_setup_patch_capable
from osprey.cli.build_profile_emit import effective_config_subtree
from osprey.cli.build_profile_load import load_profile
from osprey.cli.build_profile_merge import resolve_profile_document
from osprey.cli.build_profile_presets import _presets_dir, preset_data_bundle
from osprey.cli.templates.claude_code import _graphdb_configured
from osprey.deployment.web_terminals.lint import lint_app_render
from osprey.registry.mcp import resolve_servers
from osprey_connectors.errors import BuildProfileError
from osprey_connectors.types import configured_targets

_SETUP_PATCH = "mcp__osprey_workspace__setup_patch"
_DRAFT_CONCEPT = "mcp__osprey_facility_knowledge__draft_concept"
_APP_RENDER_CODE = "web_terminals.app_render_holds_privilege"

_BUNDLED_APPS = ("logbook", "knowledge", "channel-finder")

_WRITES = "control_system.writes"

_HOST = {
    "name": "Facility",
    "data": "data",
    "provider": "anthropic",
    "model": "opus",
    "env": {"FACILITY": "test"},
    "dispatch": {"max_turns": 5},
    "provenance": {"preset": "control-assistant"},
    "channel_finder_mode": "graph",
    "default_panel": "events",
    "claude_md_template": "CLAUDE.md.j2",
    "hooks": ["hook-log", "target-state", "approval"],
    "skills": ["diagnose", "session-report"],
    "agents": ["channel-finder"],
    "web_panels": ["events", "bluesky"],
    "mcp_servers": {"facility_tool": {"command": "facility-tool"}},
    "config": {
        "control_system.type": "virtual_accelerator",
        "control_system.writes_enabled": True,
        "control_system.connector.epics.writes_enabled": True,
        "control_system.connector.epics.timeout_s": 5.0,
        "claude_code.servers.bluesky.enabled": True,
        "claude_code.permissions.deny": [_SETUP_PATCH],
        "web.ui_mode": "simple",
        "modules.web_terminals.enabled": True,
    },
}

_APP = {
    "name": "Logbook Research",
    "claude_md_template": "CLAUDE.logbook.md.j2",
    "default_panel": "ariel",
    "channel_finder_mode": None,
    "hooks": ["hook-log", "target-state"],
    "rules": ["safety"],
    "skills": ["session-report", "logbook-deep-research"],
    "agents": [],
    "output_styles": ["control-operator"],
    "web_panels": ["ariel"],
    "mcp_servers": {},
    "provenance": {"preset": "logbook"},
    "config": {
        "claude_code.servers.controls.enabled": False,
        "control_system.writes_enabled": False,
        "web.ui_mode": "expert",
    },
}


def _write_yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, app: dict, host: dict | None = None) -> Path:
    """A deployment repo with one app file; returns the app file's path."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True, exist_ok=True)
    _write_yaml(repo / "profile.yml", host if host is not None else _HOST)
    return _write_yaml(repo / "apps" / "logbook.yml", app)


def _resolve(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_profile_document(raw, path, warn=False).raw


def _leaves(config: dict) -> dict:
    """Every leaf the config spells, one dotted path each, in any spelling."""
    leaves: dict = {}

    def walk(node: dict, prefix: tuple[str, ...]) -> None:
        for key, value in node.items():
            path = prefix + tuple(str(key).split("."))
            if isinstance(value, dict) and value:
                walk(value, path)
            else:
                leaves[".".join(path)] = value

    walk(config, ())
    return leaves


def _materialized_repo(repo: Path, host_preset: str, host_config: dict | None = None) -> Path:
    """A deployment extending ``host_preset`` with every bundled app materialized."""
    (repo / "data").mkdir(parents=True)
    host: dict = {"extends": host_preset, "name": "Flagship", "data": "data"}
    if host_config:
        host["config"] = host_config
    _write_yaml(repo / "profile.yml", host)
    (repo / "apps").mkdir()
    for app in _BUNDLED_APPS:
        shutil.copyfile(_presets_dir() / "apps" / f"{app}.yml", repo / "apps" / f"{app}.yml")
    return repo


def _admin_tier_lift() -> dict:
    """The bundled admin tier's ``remove_deny``: the one lift of the setup floor."""
    tier = yaml.safe_load((_presets_dir() / "tiers" / "admin.yml").read_text(encoding="utf-8"))
    key = "claude_code.permissions.remove_deny"
    assert _SETUP_PATCH in tier["config"][key]
    return {key: tier["config"][key]}


@pytest.fixture(scope="module")
def flagship_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A control-assistant deployment with every bundled app materialized."""
    return _materialized_repo(tmp_path_factory.mktemp("flagship") / "repo", "control-assistant")


@pytest.fixture(scope="module", params=["floor", "lifting"])
def host_shape_repo(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """The flagship host, and the same host carrying the admin tier's floor lift."""
    lift = _admin_tier_lift() if request.param == "lifting" else None
    return _materialized_repo(
        tmp_path_factory.mktemp("host") / "repo", "control-assistant", host_config=lift
    )


def _targets(config: dict) -> list[str]:
    return configured_targets(effective_config_subtree(config, ("control_system",)))


def _render_servers(profile) -> list[str]:
    """The MCP servers the render enables, keyed the way the render keys them."""
    claude_code = effective_config_subtree(profile.config, ("claude_code",))
    services = effective_config_subtree(profile.config, ("services",))
    ctx = {"graphdb_configured": _graphdb_configured({"services": services})}
    if profile.channel_finder_mode and "channel-finder" in profile.agents:
        ctx["channel_finder_pipeline"] = profile.channel_finder_mode
    names = [s["name"] for s in resolve_servers(claude_code, ctx) if s["enabled"]]
    return names + [name for name in profile.mcp_servers if name not in names]


# ── what the app file owns ───────────────────────────────────────────────────


def test_lists_and_brief_come_from_the_app_file_alone(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, _APP))

    assert raw["name"] == "Logbook Research"
    assert raw["claude_md_template"] == "CLAUDE.logbook.md.j2"
    assert raw["default_panel"] == "ariel"
    assert raw["skills"] == ["session-report", "logbook-deep-research"]
    assert raw["agents"] == []
    assert raw["web_panels"] == ["ariel"]
    assert raw["rules"] == ["safety"]
    assert raw["output_styles"] == ["control-operator"]


def test_explicit_null_channel_finder_mode_beats_the_host(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, _APP))
    assert "channel_finder_mode" in raw
    assert raw["channel_finder_mode"] is None


def test_an_app_owned_key_the_file_omits_is_not_inherited(tmp_path: Path) -> None:
    app = {k: v for k, v in _APP.items() if k not in ("channel_finder_mode", "agents")}
    raw = _resolve(_repo(tmp_path, app))
    assert raw.get("channel_finder_mode") is None
    assert not raw.get("agents")


def test_writes_check_is_added_to_the_app_hooks(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, _APP))
    assert raw["hooks"][:2] == ["hook-log", "target-state"]
    assert "writes-check" in raw["hooks"]


def test_every_host_guard_reaches_the_app_and_nothing_else_does(tmp_path: Path) -> None:
    host = dict(_HOST, hooks=["hook-log", "approval", "turn-state"])
    raw = _resolve(_repo(tmp_path, dict(_APP, hooks=["target-state"]), host=host))
    assert raw["hooks"] == ["target-state", "approval", "writes-check"]


def test_writes_check_is_not_duplicated(tmp_path: Path) -> None:
    app = dict(_APP, hooks=["target-state", "writes-check"])
    raw = _resolve(_repo(tmp_path, app))
    assert raw["hooks"].count("writes-check") == 1


# ── what the host owns ───────────────────────────────────────────────────────


def test_other_top_level_keys_are_the_hosts_verbatim(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, _APP))

    for key in ("provider", "model", "env", "dispatch", "data", "mcp_servers"):
        assert raw[key] == _HOST[key], key


def test_an_app_model_replaces_the_hosts_and_keeps_the_hosts_provider(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, {**_APP, "model": "claude-sonnet-5"}))

    assert raw["model"] == "claude-sonnet-5"
    assert raw["provider"] == _HOST["provider"]


def test_a_bundled_app_given_a_model_loads_with_it(tmp_path: Path) -> None:
    repo = _materialized_repo(tmp_path / "repo", "control-assistant")
    host = load_profile(repo / "profile.yml")
    app_path = repo / "apps" / "logbook.yml"
    app_path.write_text(
        app_path.read_text(encoding="utf-8") + "\nmodel: claude-sonnet-5\n", encoding="utf-8"
    )

    app = load_profile(app_path)

    assert app.model == "claude-sonnet-5" != host.model
    assert app.provider == host.provider
    assert load_profile(repo / "apps" / "knowledge.yml").model == host.model


@pytest.mark.parametrize("spelled", [{}, {"model": None}], ids=["absent", "null"])
def test_an_app_that_names_no_model_runs_the_hosts(tmp_path: Path, spelled: dict) -> None:
    app = {k: v for k, v in _APP.items() if k != "model"} | spelled
    assert _resolve(_repo(tmp_path, app))["model"] == _HOST["model"]


@pytest.mark.parametrize(
    "config",
    [
        {"claude_code.default_model": "claude-sonnet-5"},
        {"claude_code": {"default_model": "claude-sonnet-5"}},
    ],
    ids=["dotted", "nested"],
)
def test_a_default_model_spelled_in_an_app_config_names_the_app_file(
    tmp_path: Path, config: dict
) -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        _resolve(_repo(tmp_path, {**_APP, "config": config}))

    message = str(excinfo.value)
    assert "the top-level `model:` field sets it" in message
    assert "Remove it from apps/logbook.yml." in message
    assert "profile.yml" not in message


def test_a_default_model_spelled_in_a_tier_config_names_the_tier_file(tmp_path: Path) -> None:
    tier = _write_yaml(
        tmp_path / "repo" / "tiers" / "readonly.yml",
        {"name": "Read only", "config": {"claude_code.default_model": "claude-sonnet-5"}},
    )
    _repo(tmp_path, _APP)

    with pytest.raises(BuildProfileError, match=r"Remove it from tiers/readonly\.yml\."):
        _resolve(tier)


def test_a_tier_model_still_replaces_the_hosts(tmp_path: Path) -> None:
    _repo(tmp_path, _APP)
    tier = _write_yaml(
        tmp_path / "repo" / "tiers" / "readonly.yml",
        {"name": "Read only", "model": "claude-sonnet-5"},
    )

    assert _resolve(tier)["model"] == "claude-sonnet-5"


def test_provenance_is_the_hosts(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, _APP))
    assert raw["provenance"] == {"preset": "control-assistant"}


def test_app_without_provenance_keeps_the_hosts(tmp_path: Path) -> None:
    app = {k: v for k, v in _APP.items() if k != "provenance"}
    raw = _resolve(_repo(tmp_path, app))
    assert raw["provenance"] == {"preset": "control-assistant"}


def test_attachment_facts_still_hold(tmp_path: Path) -> None:
    raw = _resolve(_repo(tmp_path, _APP))
    assert raw["deploy_services"] is False
    assert raw["services"] == {}
    assert raw["config"]["modules.web_terminals.enabled"] is False


def test_composing_does_not_mutate_the_app_document(tmp_path: Path) -> None:
    path = _repo(tmp_path, _APP)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    before = yaml.safe_dump(raw)
    resolve_profile_document(raw, path, warn=False)
    assert yaml.safe_dump(raw) == before


# ── the config: host facts under the app's own block ────────────────────────


def test_config_is_host_facts_under_the_app_config(tmp_path: Path) -> None:
    leaves = _leaves(_resolve(_repo(tmp_path, _APP))["config"])

    assert leaves["web.ui_mode"] == "expert"
    assert leaves["control_system.type"] == "virtual_accelerator"
    assert leaves["control_system.connector.epics.timeout_s"] == 5.0
    assert leaves["claude_code.servers.controls.enabled"] is False


@pytest.mark.parametrize(
    "spelling",
    [
        {"claude_code.servers.bluesky.enabled": True},
        {"claude_code.servers": {"bluesky": {"enabled": True}}},
        {"claude_code": {"servers": {"bluesky": {"enabled": True}}}},
    ],
)
def test_host_server_switches_are_dropped_in_every_spelling(tmp_path: Path, spelling: dict) -> None:
    host = dict(_HOST, config={"web.ui_mode": "simple", **spelling})
    leaves = _leaves(_resolve(_repo(tmp_path, _APP, host=host))["config"])
    assert not any(path.startswith("claude_code.servers.bluesky") for path in leaves)


@pytest.mark.parametrize(
    "spelling",
    [
        {"claude_code.permissions.deny": [_SETUP_PATCH]},
        {"claude_code.permissions": {"deny": [_SETUP_PATCH]}},
        {"claude_code": {"permissions": {"deny": [_SETUP_PATCH]}}},
    ],
)
def test_permissions_are_unioned_in_one_spelling(tmp_path: Path, spelling: dict) -> None:
    """The host floor deny and the app's own deny both reach the render."""
    host = {k: v for k, v in _HOST.items() if k not in ("provenance", "dispatch")}
    host["config"] = spelling
    app = {k: v for k, v in _APP.items() if k != "provenance"}
    app["config"] = {"claude_code.permissions.deny": [_DRAFT_CONCEPT]}
    path = _repo(tmp_path, app, host=host)

    profile = load_profile(path)  # refuses a config spelling one path two ways
    deny = effective_config_subtree(profile.config, ("claude_code",))["permissions"]["deny"]
    assert set(deny) == {_SETUP_PATCH, _DRAFT_CONCEPT}
    assert _profile_setup_patch_capable(profile) is False


@pytest.mark.parametrize(
    "host_writes",
    [
        {"control_system.writes_enabled": True},
        {"control_system": {"writes_enabled": True}},
        {"control_system.connector.epics.writes_enabled": True},
        {"control_system.connector": {"epics": {"writes_enabled": True}}},
        {"control_system.connector.tango.writes_enabled": True},
        {"control_system.connector": {"mypackage.Moat": {"writes_enabled": True}}},
        {"control_system.writes": "allowed"},
        {"control_system": {"writes": "allowed"}},
    ],
)
def test_host_write_switches_never_reach_an_app(tmp_path: Path, host_writes: dict) -> None:
    host = dict(_HOST, config={"web.ui_mode": "simple", **host_writes})
    leaves = _leaves(_resolve(_repo(tmp_path, _APP, host=host))["config"])

    assert leaves[_WRITES] == "never"
    assert not [p for p in leaves if p.endswith("writes_enabled")], leaves


def test_an_old_key_app_that_arms_writes_renders_never(tmp_path: Path) -> None:
    app = dict(
        _APP,
        config={
            "control_system.writes_enabled": True,
            "control_system.connector": {"epics": {"writes_enabled": True}},
        },
    )
    leaves = _leaves(_resolve(_repo(tmp_path, app))["config"])

    assert leaves[_WRITES] == "never"
    assert not [p for p in leaves if p.endswith("writes_enabled")], leaves


def test_an_app_spelling_never_renders_never(tmp_path: Path) -> None:
    app = dict(_APP, config={_WRITES: "never"})
    leaves = _leaves(_resolve(_repo(tmp_path, app))["config"])
    assert leaves[_WRITES] == "never"


def test_a_connector_block_the_host_has_is_kept_without_a_pin(tmp_path: Path) -> None:
    leaves = _leaves(_resolve(_repo(tmp_path, _APP))["config"])
    assert "control_system.connector.epics.writes_enabled" not in leaves
    assert leaves["control_system.connector.epics.timeout_s"] == 5.0


@pytest.mark.parametrize("connector", ["virtual_accelerator", "doocs", "tango"])
def test_no_connector_block_is_created(tmp_path: Path, connector: str) -> None:
    leaves = _leaves(_resolve(_repo(tmp_path, _APP))["config"])
    assert not [p for p in leaves if p.startswith(f"control_system.connector.{connector}.")]


def test_no_retired_leaf_is_pinned_for_a_block_that_held_only_its_switch(
    tmp_path: Path,
) -> None:
    """The tier key alone states an app's posture: no per-type pin keeps a block alive."""
    host = dict(
        _HOST,
        config={
            "control_system.type": "epics",
            "control_system.connector.virtual_accelerator.writes_enabled": True,
        },
    )
    leaves = _leaves(_resolve(_repo(tmp_path, _APP, host=host))["config"])
    connector = {p: v for p, v in leaves.items() if p.startswith("control_system.connector.")}
    assert connector == {}
    assert leaves[_WRITES] == "never"


def test_a_dotted_custom_type_gets_no_per_type_pin(tmp_path: Path) -> None:
    host = dict(
        _HOST,
        config={
            "control_system.type": "mypackage.Moat",
            "control_system.connector": {"mypackage.Moat": {"writes_enabled": True, "url": "x"}},
        },
    )
    config = _resolve(_repo(tmp_path, _APP, host=host))["config"]
    assert not [k for k in config if k.startswith("control_system.connector.mypackage")]
    assert config[_WRITES] == "never"
    assert "control_system.writes_enabled" not in config


@pytest.mark.parametrize(
    "host_config",
    [
        _HOST["config"],
        {
            "control_system.type": "epics",
            "control_system.writes_enabled": True,
            "control_system.connector.epics.address_list": "10.0.0.1",
            "control_system.connector.epics.writes_enabled": True,
        },
        {
            "control_system": {
                "type": "epics",
                "connector": {
                    "epics": {"address_list": "10.0.0.1", "writes_enabled": True},
                    "virtual_accelerator": {"serving": "in_process"},
                },
            }
        },
    ],
    ids=["va-baseline", "epics-baseline", "epics-nested"],
)
def test_the_app_render_has_the_hosts_targets(tmp_path: Path, host_config: dict) -> None:
    host = dict(_HOST, config=host_config)
    render = _resolve(_repo(tmp_path, _APP, host=host))["config"]
    assert _targets(render) == _targets(host_config)


@pytest.mark.parametrize("app", _BUNDLED_APPS)
def test_every_bundled_app_render_has_the_flagship_targets(flagship_repo: Path, app: str) -> None:
    host = load_profile(flagship_repo / "profile.yml")
    profile = load_profile(flagship_repo / "apps" / f"{app}.yml")
    assert _targets(profile.config) == _targets(host.config)
    assert len(_targets(host.config)) >= 2


_SINGLE_TARGET_HOSTS = {
    "epics-only": {
        "control_system.type": "epics",
        "control_system.writes_enabled": True,
        "control_system.connector.epics.address_list": "10.0.0.1",
        "claude_code.permissions.deny": [_SETUP_PATCH],
    },
    "va-only": {
        "control_system.type": "virtual_accelerator",
        "control_system.writes_enabled": True,
        "control_system.connector.virtual_accelerator.writes_enabled": True,
        "claude_code.permissions.deny": [_SETUP_PATCH],
    },
}


@pytest.mark.parametrize("app", _BUNDLED_APPS)
@pytest.mark.parametrize("host_shape", sorted(_SINGLE_TARGET_HOSTS))
def test_every_bundled_app_render_has_a_single_target_hosts_targets(
    tmp_path: Path, host_shape: str, app: str
) -> None:
    """An app's own per-type write pins never count as a connector block."""
    host_config = _SINGLE_TARGET_HOSTS[host_shape]
    host = dict(_HOST, config=host_config)
    path = tmp_path / "repo" / "apps" / f"{app}.yml"
    _repo(tmp_path, {}, host=host)
    shutil.copyfile(_presets_dir() / "apps" / f"{app}.yml", path)

    render = _resolve(path)["config"]
    assert len(_targets(host_config)) == 1
    assert _targets(render) == _targets(host_config)


def test_a_partly_nested_app_deny_composes_and_loads(tmp_path: Path) -> None:
    """``claude_code: {permissions.deny: [...]}`` is one legal spelling, not a dotted name."""
    host = {k: v for k, v in _HOST.items() if k not in ("provenance", "dispatch")}
    host["config"] = {"claude_code.permissions.deny": [_SETUP_PATCH]}
    app = {k: v for k, v in _APP.items() if k != "provenance"}
    app["config"] = {"claude_code": {"permissions.deny": [_DRAFT_CONCEPT]}}
    path = _repo(tmp_path, app, host=host)

    profile = load_profile(path)
    deny = effective_config_subtree(profile.config, ("claude_code",))["permissions"]["deny"]
    assert set(deny) == {_SETUP_PATCH, _DRAFT_CONCEPT}


def test_a_dotted_server_name_is_kept_whole(tmp_path: Path) -> None:
    """A server whose name holds a dot keeps its name under ``claude_code.servers``."""
    app = dict(_APP, config={"claude_code": {"servers": {"my.server": {"enabled": False}}}})
    config = _resolve(_repo(tmp_path, app))["config"]
    assert config["claude_code.servers"] == {"my.server": {"enabled": False}}


# ── the flagship deployment's bundled apps ──────────────────────────────────


@pytest.mark.parametrize("app", _BUNDLED_APPS)
def test_no_app_render_can_patch_its_setup(host_shape_repo: Path, app: str) -> None:
    profile = load_profile(host_shape_repo / "apps" / f"{app}.yml")
    assert _profile_setup_patch_capable(profile) is False


def test_a_lifting_host_is_capable_and_its_apps_are_not(tmp_path: Path) -> None:
    host = dict(
        _HOST,
        config={
            "claude_code.permissions.deny": [_SETUP_PATCH],
            "claude_code": {"permissions": {"remove_deny": [_SETUP_PATCH]}},
        },
    )
    config = _resolve(_repo(tmp_path, _APP, host=host))["config"]
    assert not [p for p in _leaves(config) if p.endswith("remove_deny")]
    assert _SETUP_PATCH in config["claude_code.permissions.deny"]


def test_a_host_without_the_floor_deny_still_yields_a_denying_app(tmp_path: Path) -> None:
    host = dict(_HOST, config={"web.ui_mode": "simple"})
    config = _resolve(_repo(tmp_path, _APP, host=host))["config"]
    assert config["claude_code.permissions.deny"] == [_SETUP_PATCH]


@pytest.mark.parametrize(
    "lift",
    [
        {"claude_code.permissions.remove_deny": [_SETUP_PATCH]},
        {"claude_code": {"permissions": {"remove_deny": [_SETUP_PATCH]}}},
    ],
)
def test_an_app_that_lifts_the_setup_deny_is_refused(tmp_path: Path, lift: dict) -> None:
    app = dict(_APP, config={**_APP["config"], **lift})
    with pytest.raises(BuildProfileError, match="setup_patch"):
        _resolve(_repo(tmp_path, app))


@pytest.mark.parametrize("app", _BUNDLED_APPS)
def test_every_bundled_app_render_is_write_less(flagship_repo: Path, app: str) -> None:
    profile = load_profile(flagship_repo / "apps" / f"{app}.yml")
    assert "writes-check" in profile.hooks
    assert lint_app_render({"hooks": profile.hooks, "config": profile.config}) == []


def test_knowledge_render_denies_setup_patch_and_draft_concept(flagship_repo: Path) -> None:
    profile = load_profile(flagship_repo / "apps" / "knowledge.yml")
    deny = effective_config_subtree(profile.config, ("claude_code",))["permissions"]["deny"]
    assert _SETUP_PATCH in deny
    assert _DRAFT_CONCEPT in deny


def test_logbook_render_servers(flagship_repo: Path) -> None:
    profile = load_profile(flagship_repo / "apps" / "logbook.yml")
    assert sorted(_render_servers(profile)) == ["ariel", "graph", "osprey_workspace"]


def test_logbook_render_keeps_the_host_provenance(flagship_repo: Path) -> None:
    raw = yaml.safe_load((flagship_repo / "apps" / "logbook.yml").read_text(encoding="utf-8"))
    document = resolve_profile_document(raw, flagship_repo / "apps" / "logbook.yml", warn=False)
    assert document.inherited_preset == "control-assistant"
    assert document.raw["channel_finder_mode"] is None


# ── the data bundle is a deployment fact ────────────────────────────────────


@pytest.mark.parametrize("name", ["logbook", "channel_finder", "channel-finder"])
def test_preset_data_bundle_refuses_an_app(name: str) -> None:
    with pytest.raises(BuildProfileError, match="not a deployment"):
        preset_data_bundle(name)


def test_preset_data_bundle_still_answers_for_a_deployment() -> None:
    assert preset_data_bundle("control-assistant") == preset_data_bundle("control_assistant")
    assert isinstance(preset_data_bundle(None), str)


# ── the lint that guards the pins ───────────────────────────────────────────


def _codes(findings) -> list[str]:
    return [finding.code for finding in findings if finding.severity == "error"]


def test_lint_passes_a_pinned_render() -> None:
    render = {
        "hooks": ["target-state", "writes-check"],
        "config": {
            _WRITES: "never",
            "claude_code.permissions.deny": [_SETUP_PATCH],
            "web.config_panel.enabled": False,
            "web.scaffold_gallery.write_enabled": False,
        },
    }
    assert lint_app_render(render) == []


def test_lint_refuses_a_render_that_can_patch_its_setup() -> None:
    findings = lint_app_render({"hooks": ["writes-check"], "config": {_WRITES: "never"}})
    assert _codes(findings) == [_APP_RENDER_CODE]
    assert _SETUP_PATCH in findings[0].message


@pytest.mark.parametrize(
    "config",
    [
        {},
        {_WRITES: "allowed"},
        {"control_system": {"writes": "allowed"}},
        {_WRITES: False},
        {_WRITES: "Never"},
        {_WRITES: "never", "control_system": {"writes": "allowed"}},
        {"control_system": {"writes": "allowed"}, _WRITES: "never"},
        {"control_system": "never"},
        # The pre-PR-4 pin: retired leaves off, no tier key.
        {"control_system.writes_enabled": False},
        {_WRITES: "never", "control_system.writes_enabled": True},
        {_WRITES: "never", "control_system": {"writes_enabled": True}},
        {_WRITES: "never", "control_system.connector.epics.writes_enabled": True},
        {
            _WRITES: "never",
            "control_system.connector": {"mypackage.Moat": {"writes_enabled": "true"}},
        },
        {
            _WRITES: "never",
            "control_system.connector": {"epics": {"writes_enabled": True}},
            "control_system.connector.epics.writes_enabled": False,
        },
    ],
)
def test_lint_refuses_an_armed_or_unpinned_render(config: dict) -> None:
    config = {**config, "claude_code.permissions.deny": [_SETUP_PATCH]}
    findings = lint_app_render({"hooks": ["writes-check"], "config": config})
    assert _codes(findings) == [_APP_RENDER_CODE]
    assert _WRITES in findings[0].message


def test_lint_tolerates_a_retired_leaf_that_is_off() -> None:
    """A retired leaf reading ``false`` arms no retired reader; the tier key still decides."""
    render = {
        "hooks": ["writes-check"],
        "config": {
            _WRITES: "never",
            "control_system.connector": {"epics": {"writes_enabled": False}},
            "claude_code.permissions.deny": [_SETUP_PATCH],
            "web.config_panel.enabled": False,
            "web.scaffold_gallery.write_enabled": False,
        },
    }
    assert lint_app_render(render) == []


def test_lint_refuses_a_render_without_writes_check() -> None:
    findings = lint_app_render(
        {
            "hooks": ["target-state"],
            "config": {
                _WRITES: "never",
                "claude_code.permissions.deny": [_SETUP_PATCH],
            },
        }
    )
    assert _codes(findings) == [_APP_RENDER_CODE]
    assert "writes-check" in findings[0].message
