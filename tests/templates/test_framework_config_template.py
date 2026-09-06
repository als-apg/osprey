"""The framework config template renders the derived keys and nothing else.

``profile.yml`` is the whole declarative input. The one thing it cannot state is
what the BUILD knows — where the project sits, which ports this deployment's
block hands out, what ``providers.yml`` holds, and the keys a profile FIELD
(``provider:``, ``model:``, ``channel_finder_mode:``, ``default_panel:``,
``panel_presets:``) decides. Those are :data:`osprey.cli.derived_keys.DERIVED_KEYS`,
and ``templates/project/config.yml.j2`` is their one writer.

So the claim pinned here is a partition, not a list: every leaf the template
renders is a derived key, an ``api.providers`` entry, or one
``web.panels.<id>.enabled`` from the panel selection — and nothing else. A key
that drifts back into the template would be a second home for a fact the
operator's ``profile.yml`` already states, and the losing copy would be the
silent one.

The rest is what a header must not do: ``web:``, ``services:`` and
``deployed_services:`` are absent rather than empty when the template has
nothing to put in them, because ``interfaces/web_terminal/app.py`` cannot load
a bare ``web:``.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.build_cmd import _ariel_server_enabled
from osprey.cli.derived_keys import is_derived_key
from osprey.cli.templates.manager import TemplateManager, _enable_flags
from osprey.errors import BuildProfileError
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
from osprey.profiles.providers import load_provider_catalog
from osprey.registry.mcp import FRAMEWORK_SERVERS

CONFIG_TEMPLATE = "project/config.yml.j2"

#: A catalog small enough to read in a failure message, with an entry carrying a
#: key the contract does not name — those pass through verbatim.
_CATALOG: dict[str, Any] = {
    "house": {
        "api_key": "${HOUSE_API_KEY}",
        "base_url": "https://gateway.example.org/v1",
        "models": {"haiku": "small", "sonnet": "mid", "opus": "large"},
    },
    "bare": {"base_url": "http://127.0.0.1:8000/v1", "timeout": 30},
}

#: The panel ids the framework serves itself, as the manager hands them over.
_BUILTINS = ["ariel", "channel-finder", "okf", "system-health"]

#: What a hello-world-shaped profile gives the template: no channel-finder
#: agent, no logbook, no web panels. The floor every other case adds to.
_MINIMAL_CTX: dict[str, Any] = {
    "project_name": "demo",
    "project_root": "/repos/demo",
    "default_provider": "anthropic",
    "default_model": "haiku",
    "port_base": DEFAULT_PORT_BASE,
    "osprey_ports": layout_ports(DEFAULT_PORT_BASE),
    "provider_catalog": _CATALOG,
    "builtin_panels": _BUILTINS,
    "selected_web_panels": [],
    "ariel_server_on": False,
}


def _render(**overrides: Any) -> str:
    context = {**_MINIMAL_CTX, **overrides}
    return TemplateManager().jinja_env.get_template(CONFIG_TEMPLATE).render(**context)


def _config(**overrides: Any) -> dict[str, Any]:
    return yaml.safe_load(_render(**overrides))


def _mode_ctx(mode: str) -> dict[str, Any]:
    """The channel-finder half of the context, derived as the manager derives it."""
    return {"channel_finder_mode": mode, "default_pipeline": mode, **_enable_flags(mode)}


def _leaf_keys(node: Any, prefix: str = "") -> set[str]:
    """Every dotted path in a parsed config whose value is not a mapping.

    Leaves rather than every path, because the partition is about facts: an
    intermediate ``claude_code`` or ``ariel.enhancement_modules`` is the shape
    a leaf arrives in, not a key anyone states.
    """
    if not isinstance(node, dict) or not node:
        return {prefix.rstrip(".")} if prefix else set()
    keys: set[str] = set()
    for key, value in node.items():
        keys |= _leaf_keys(value, f"{prefix}{key}.")
    return keys


def _fully_loaded_ctx(mode: str = "hierarchical") -> dict[str, Any]:
    """Everything the template can render at once — the widest key set."""
    return {
        **_mode_ctx(mode),
        "ariel_server_on": True,
        "selected_web_panels": [*_BUILTINS, "events"],
        "default_panel": "channel-finder",
        "panel_presets": {"Machine setup": ["channel-finder", "artifacts"]},
        "environment_python": "/usr/bin/python3",
        "environment_packages": ["numpy"],
        "environment_inherit_exclude": ["osprey"],
    }


# ── The partition ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", sorted(VALID_CHANNEL_FINDER_MODES))
def test_every_rendered_leaf_is_derived(mode: str):
    """The template writes derived keys, the catalog, and the panel selection."""
    panel_enabled = re.compile(r"^web\.panels\.[^.]+\.enabled$")
    stray = sorted(
        key
        for key in _leaf_keys(_config(**_fully_loaded_ctx(mode)))
        if not is_derived_key(key)
        and not key.startswith("api.providers.")
        and not panel_enabled.match(key)
    )
    assert stray == [], (
        f"{CONFIG_TEMPLATE} renders keys no profile field or file supplies: {stray}. "
        "Each belongs in the preset's `config:` block, where it is documented."
    )


@pytest.mark.parametrize(
    "removed",
    ["services", "deployed_services", "cli", "approval", "system", "container_runtime"],
)
def test_removed_sections_are_gone(removed: str):
    """Literal defaults moved to the presets; the template states none of them."""
    assert removed not in _config(**_fully_loaded_ctx())


def test_web_theme_is_not_rendered():
    """`web.theme` is a preset key — the template must not shadow it."""
    assert "theme" not in _config(**_fully_loaded_ctx())["web"]


def test_generated_banner_is_kept():
    rendered = _render(**_fully_loaded_ctx())
    assert "GENERATED" in rendered.splitlines()[0] or "GENERATED" in rendered.splitlines()[1]
    assert "osprey build" in rendered


# ── Headers only when filled ─────────────────────────────────────────────────


def test_minimal_render_has_no_empty_headers():
    """A hello-world-shaped profile gets no `web:`, `services:`, `deployed_services:`.

    A bare `web:` is not an empty tab strip — `_load_panel_config` cannot read
    it at all — so the header has to be absent rather than valueless.
    """
    config = _config()
    for header in ("web", "services", "deployed_services"):
        assert header not in config


def test_minimal_render_is_the_derived_floor():
    """Nothing profile-field-shaped renders without the field that drives it."""
    config = _config()
    assert sorted(config) == [
        "agent_data",
        "api",
        "artifact_server",
        "build_dir",
        "claude_code",
        "execution",
        "file_paths",
        "project_name",
        "project_root",
    ]


def test_project_layout_and_ports_are_derived():
    config = _config()
    assert config["project_name"] == "demo"
    assert config["project_root"] == "/repos/demo"
    assert config["build_dir"] == "./build"
    assert config["agent_data"]["base_dir"] == "var/agent_data"
    assert config["file_paths"] == {
        "api_calls_dir": "api_calls",
        "registry_exports_dir": "registry_exports",
    }
    assert config["artifact_server"] == {"port": layout_ports(DEFAULT_PORT_BASE)["artifact"]}


def test_artifact_port_follows_the_deployments_port_base():
    moved = layout_ports(DEFAULT_PORT_BASE + 1000)
    config = _config(port_base=DEFAULT_PORT_BASE + 1000, osprey_ports=moved)
    assert config["artifact_server"]["port"] == moved["artifact"]


def test_environment_block_is_build_provenance():
    config = _config(
        environment_python="/usr/bin/python3",
        environment_packages=["numpy", "scipy"],
        environment_inherit_exclude=["osprey"],
    )
    assert config["execution"] == {
        "environment": {
            "python": "/usr/bin/python3",
            "packages": ["numpy", "scipy"],
            "inherit_exclude": ["osprey"],
        }
    }
    assert "execution_method" not in config["execution"]


def test_environment_block_defaults_to_an_undeclared_environment():
    assert _config()["execution"]["environment"] == {
        "python": None,
        "packages": [],
        "inherit_exclude": [],
    }


# ── Profile fields ───────────────────────────────────────────────────────────


def test_provider_and_model_come_from_the_profile_fields():
    config = _config(default_provider="cborg", default_model="opus")
    assert config["claude_code"] == {"provider": "cborg", "default_model": "opus"}


def test_model_falls_back_to_the_haiku_tier():
    context = {key: value for key, value in _MINIMAL_CTX.items() if key != "default_model"}
    config = yaml.safe_load(
        TemplateManager().jinja_env.get_template(CONFIG_TEMPLATE).render(**context)
    )
    assert config["claude_code"]["default_model"] == "haiku"


def test_claude_code_carries_no_other_key():
    """Servers, agents and telemetry are the preset's; only the two fields are here."""
    assert sorted(_config(**_fully_loaded_ctx())["claude_code"]) == [
        "default_model",
        "provider",
    ]


# ── Channel finder ───────────────────────────────────────────────────────────


def test_no_channel_finder_section_without_the_agent():
    """The manager derives `default_pipeline` only when the agent is selected."""
    assert "channel_finder" not in _config()


@pytest.mark.parametrize("mode", sorted(VALID_CHANNEL_FINDER_MODES))
def test_pipeline_mode_is_the_profile_field(mode: str):
    assert _config(**_mode_ctx(mode))["channel_finder"]["pipeline_mode"] == mode


@pytest.mark.parametrize("mode", sorted(set(VALID_CHANNEL_FINDER_MODES) - {"graph"}))
def test_each_mode_renders_only_its_own_pipeline(mode: str):
    pipelines = _config(**_mode_ctx(mode))["channel_finder"]["pipelines"]
    assert sorted(pipelines) == [mode]


def test_graph_mode_renders_no_pipeline_block():
    """Graph answers from `services.graphdb`, so the mode is its configuration."""
    channel_finder = _config(**_mode_ctx("graph"))["channel_finder"]
    assert channel_finder["pipelines"] is None
    assert sorted(channel_finder) == ["pipeline_mode", "pipelines"]


def test_in_context_pipeline_values():
    pipeline = _config(**_mode_ctx("in_context"))["channel_finder"]["pipelines"]["in_context"]
    assert pipeline == {
        "database": {
            "type": "template",
            "path": "data/channel_databases/in_context.json",
            "presentation_mode": "template",
        },
        "subagent_model": None,
    }


def test_hierarchical_pipeline_values():
    pipeline = _config(**_mode_ctx("hierarchical"))["channel_finder"]["pipelines"]["hierarchical"]
    assert pipeline == {
        "database": {
            "type": "hierarchical",
            "path": "data/channel_databases/hierarchical.json",
        },
        "feedback": {
            "enabled": True,
            "store_path": "var/agent_data/feedback/hierarchical_feedback.json",
        },
    }


def test_middle_layer_pipeline_values():
    pipeline = _config(**_mode_ctx("middle_layer"))["channel_finder"]["pipelines"]["middle_layer"]
    assert pipeline == {
        "database": {
            "type": "middle_layer",
            "path": "data/channel_databases/middle_layer.json",
        }
    }


def test_channel_finder_carries_no_preset_key():
    """`benchmark:` and the companion web server are the preset's to document."""
    assert sorted(_config(**_mode_ctx("in_context"))["channel_finder"]) == [
        "pipeline_mode",
        "pipelines",
    ]


# ── Web panels ───────────────────────────────────────────────────────────────


def test_selected_builtin_panels_are_enabled():
    config = _config(selected_web_panels=["ariel", "okf"])
    assert config["web"]["panels"] == {
        "ariel": {"enabled": True},
        "okf": {"enabled": True},
    }


def test_custom_panels_are_not_invented_from_the_selection():
    """A non-builtin tab has no block here: an injector or `config:` writes it."""
    config = _config(selected_web_panels=["ariel", "events"])
    assert sorted(config["web"]["panels"]) == ["ariel"]


def test_unselected_builtin_panels_get_no_block():
    """A builtin the profile did not select is somebody else's block to write.

    The frozen control-assistant renders carry `web.panels.bluesky.enabled:
    false` and `web.panels.events.enabled: false` — blocks an injector and the
    preset's `config:` wrote, which `panel_selection_overrides` then annotated
    as not-shown. Writing them here would claim a tab the selection never named.
    """
    config = _config(selected_web_panels=["ariel"])
    assert sorted(config["web"]["panels"]) == ["ariel"]


def test_default_panel_is_the_profile_field():
    config = _config(selected_web_panels=["ariel"], default_panel="ariel")
    assert config["web"]["default_panel"] == "ariel"


def test_panel_presets_are_the_profile_field():
    config = _config(
        selected_web_panels=["ariel"],
        panel_presets={"Logbook review": ["ariel", "artifacts"]},
    )
    assert config["web"]["presets"] == {"Logbook review": ["ariel", "artifacts"]}


def test_web_renders_an_explicit_empty_panels_map_when_a_field_opens_it():
    """A profile that names a layout but selects no builtin still gets a map.

    `panels: {}` is a tab strip with nothing on it; the absent header the
    minimal render produces is a config the panel loader skips entirely.
    """
    config = _config(panel_presets={"Empty": ["artifacts"]})
    assert config["web"]["panels"] == {}


# ── The ARIEL gate ───────────────────────────────────────────────────────────


def test_ariel_blocks_render_when_the_server_is_on():
    config = _config(ariel_server_on=True, default_provider="cborg", default_model="opus")
    assert config["logbook"] == {"composition": {"provider": "cborg"}}
    assert config["ariel"] == {
        "enhancement_modules": {
            "semantic_processor": {"provider": "cborg", "model": {"model_id": "opus"}}
        }
    }


def test_ariel_blocks_are_absent_when_the_server_is_off():
    config = _config(ariel_server_on=False)
    assert "logbook" not in config
    assert "ariel" not in config


@pytest.mark.parametrize(
    "config",
    [
        {"claude_code.servers.ariel.enabled": False},
        {"claude_code.servers.ariel": {"enabled": False}},
        {"claude_code": {"servers": {"ariel": {"enabled": False}}}},
        {"claude_code.servers": {"ariel.enabled": False}},
    ],
    ids=["dotted", "prefix-over-mapping", "nested", "mixed"],
)
def test_ariel_gate_reads_every_spelling(config: dict[str, Any]):
    assert _ariel_server_enabled(SimpleNamespace(config=config)) is False


def test_ariel_gate_honours_an_explicit_enable():
    profile = SimpleNamespace(config={"claude_code.servers.ariel.enabled": True})
    assert _ariel_server_enabled(profile) is True


def test_ariel_gate_falls_back_to_the_registry_default():
    """No `enabled:` anywhere means the server's own default decides."""
    expected = FRAMEWORK_SERVERS["ariel"].default_enabled
    assert _ariel_server_enabled(SimpleNamespace(config={})) is expected
    assert _ariel_server_enabled(SimpleNamespace(config={"approval.enabled": True})) is expected


def test_two_spellings_that_disagree_are_refused():
    """One fact spelled twice, two answers — the build names both, not a winner.

    `spelled_values` walks splits outermost-first and `_config_lookup` (which
    `resolve_servers` reads through) takes the longest dotted match, so picking
    either would gate the template one way and start the server the other.
    """
    profile = SimpleNamespace(
        config={
            "claude_code.servers.ariel.enabled": True,
            "claude_code": {"servers": {"ariel": {"enabled": False}}},
        }
    )
    with pytest.raises(BuildProfileError) as excinfo:
        _ariel_server_enabled(profile)
    message = str(excinfo.value)
    assert "claude_code.servers.ariel.enabled" in message
    assert "True" in message and "False" in message


def test_two_spellings_that_agree_are_accepted():
    """Agreement is the same fact twice, which is untidy but not a contradiction."""
    profile = SimpleNamespace(
        config={
            "claude_code.servers.ariel.enabled": False,
            "claude_code": {"servers": {"ariel": {"enabled": False}}},
        }
    )
    assert _ariel_server_enabled(profile) is False


def test_a_non_boolean_spelling_does_not_make_a_contradiction():
    """Only booleans are answers, so one of them cannot disagree with a string."""
    profile = SimpleNamespace(
        config={
            "claude_code.servers.ariel.enabled": True,
            "claude_code": {"servers": {"ariel": {"enabled": "yes"}}},
        }
    )
    assert _ariel_server_enabled(profile) is True


def test_ariel_gate_ignores_a_non_boolean_enabled():
    """Only a bool is an answer; anything else leaves the registry default."""
    profile = SimpleNamespace(config={"claude_code.servers.ariel.enabled": "yes"})
    assert _ariel_server_enabled(profile) is FRAMEWORK_SERVERS["ariel"].default_enabled


# ── The provider catalog ─────────────────────────────────────────────────────


def test_catalog_renders_verbatim():
    assert _config()["api"]["providers"] == _CATALOG


def test_catalog_keeps_the_files_entry_order():
    """The render is the catalog's own order, so a diff of two builds reads."""
    rendered = list(_config()["api"]["providers"])
    assert rendered == list(_CATALOG)


def test_packaged_catalog_survives_the_round_trip():
    """Every shipped entry, including env-var placeholders and `:` in a value."""
    packaged = load_provider_catalog(None).entries
    assert _config(provider_catalog=packaged)["api"]["providers"] == packaged


def test_no_api_section_without_a_catalog():
    assert "api" not in _config(provider_catalog={})


def test_catalog_renders_an_empty_entry_as_a_mapping():
    """An entry with no leaves must stay a mapping, not become null."""
    assert _config(provider_catalog={"stub": {}})["api"]["providers"] == {"stub": {}}


# ``osprey config --defaults`` no longer renders this template. It reads the
# config-key manifest's `default:` column instead, so what it prints is each
# key's reader's own fallback rather than a set of example values this template
# was rendered with. Its tests live in tests/cli/test_config_view_render.py.
