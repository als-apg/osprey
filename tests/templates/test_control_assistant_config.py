"""Graph mode configures the control-assistant preset through nothing but its name.

The three database-backed paradigms each carry a ``channel_finder.pipelines.<mode>``
block naming the file they read. The graph paradigm reads a store, and that store
is already declared — ``services.graphdb``, deployed by this preset or pointed at a
facility-hosted one through ``uri``. So ``pipeline_mode: graph`` plus that block is
the *entire* configuration: there is no ``pipelines.graph`` block, and graph mode
introduces no config key that the other modes do not already render.

That is a claim about the shape of the rendered file, so it is pinned as one here.
A render is two layers: the framework template (``project/config.yml.j2``), which
writes the keys derived from the profile's fields — ``channel_finder_mode:`` among
them — and the preset's resolved ``config:`` block, which the build lays over it
and which is the same in every mode. The graph render's key set is the in-context
render's key set with the pipeline block removed — nothing added, nothing else
dropped. The corollary is pinned too: the manager derives an ``enable_graph`` flag
for every registered paradigm, and no template reads this one. Should a template
start reading it, the test that walks the template tree fails, and the render
matrix in ``osprey/profiles/config_key_manifest.yml`` (which exercises the ``enable_*``
branches) would need a fourth cell.

The mode-enumerating comment is checked against the registry rather than against
a literal list, so a fifth paradigm fails these tests instead of quietly leaving
the preset's prose one mode short.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

import osprey.profiles
from osprey.build.build_tiers import VALID_CHANNEL_FINDER_MODES
from osprey.cli.build_cmd import _ariel_server_enabled
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_model import BuildProfile
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.templates.manager import TemplateManager, _enable_flags
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports
from osprey.profiles.providers import load_provider_catalog
from osprey.profiles.web_panels import BUILTIN_PANELS

PRESET = "control-assistant"
CONFIG_TEMPLATE = "project/config.yml.j2"
PRESET_PATH = Path(osprey.profiles.__file__).parent / "presets" / f"{PRESET}.yml"


def _profile(mode: str) -> BuildProfile:
    """The preset resolved with *mode* selected, the way ``osprey build --set`` does."""
    profile, _profile_dir = resolve_build_profile(
        None, PRESET, set_pairs=(f"channel_finder_mode={mode}",)
    )
    return profile


def _ctx(profile: BuildProfile) -> dict[str, Any]:
    """The framework template's context for *profile*, derived as the build derives it.

    Only the profile-field-derived half matters here; the rest of the real
    context (ports, interpreter paths) is pinned at the defaults.
    """
    mode = profile.channel_finder_mode or ""
    return {
        "project_name": "demo",
        "project_root": "/repos/demo",
        "default_provider": profile.provider,
        "default_model": profile.model,
        "port_base": DEFAULT_PORT_BASE,
        "osprey_ports": layout_ports(DEFAULT_PORT_BASE),
        "provider_catalog": load_provider_catalog(None).entries,
        "builtin_panels": sorted(BUILTIN_PANELS),
        "selected_web_panels": list(profile.web_panels),
        "default_panel": profile.default_panel,
        "panel_presets": profile.panel_presets,
        "ariel_server_on": _ariel_server_enabled(profile),
        "channel_finder_mode": mode,
        "default_pipeline": mode,
        **_enable_flags(mode),
    }


def _overlay(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """*top* laid over *base*, mapping by mapping — the config-override path's merge."""
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _overlay(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config(mode: str) -> dict[str, Any]:
    """The rendered config for *mode*: framework template, then the preset's ``config:``."""
    profile = _profile(mode)
    rendered = TemplateManager().jinja_env.get_template(CONFIG_TEMPLATE).render(**_ctx(profile))
    return _overlay(yaml.safe_load(rendered) or {}, _expand_dotted(profile.config))


def _dotted_keys(node: Any, prefix: str = "") -> set[str]:
    """Every dotted key path in a parsed config, mappings only."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}{key}"
            keys.add(path)
            keys |= _dotted_keys(value, f"{path}.")
    return keys


def test_graph_mode_is_the_preset_default():
    """The preset selects the graph paradigm by name; the modes below override it."""
    profile, _profile_dir = resolve_build_profile(None, PRESET)
    assert profile.channel_finder_mode == "graph"


def test_graph_mode_selects_the_paradigm_by_name():
    channel_finder = _config("graph")["channel_finder"]
    assert channel_finder["pipeline_mode"] == "graph"


def test_graph_mode_renders_no_pipeline_block():
    pipelines = _config("graph")["channel_finder"].get("pipelines") or {}
    assert "graph" not in pipelines
    assert pipelines == {}, f"graph mode rendered a pipeline block: {sorted(pipelines)}"


def test_graph_mode_configuration_is_the_graph_store():
    """The store the paradigm reads is declared, and deployed by this preset."""
    config = _config("graph")
    assert "graphdb" in config["services"]
    assert "graphdb" in config["deployed_services"]


def test_graph_mode_adds_no_config_key():
    """The graph render is the in-context render minus the pipeline block."""
    graph_keys = _dotted_keys(_config("graph"))
    in_context_keys = _dotted_keys(_config("in_context"))
    expected = {key for key in in_context_keys if not key.startswith("channel_finder.pipelines.")}
    assert graph_keys == expected


def test_enable_graph_is_derived_but_unread():
    """The flag exists for every registered paradigm; no template consumes this one."""
    assert _enable_flags("graph")["enable_graph"] is True

    template_root = Path(TemplateManager().template_root)
    readers = [
        str(path.relative_to(template_root))
        for path in template_root.rglob("*")
        if path.is_file() and _mentions_enable_graph(path)
    ]
    assert readers == [], f"templates reading enable_graph: {readers}"


def _mentions_enable_graph(path: Path) -> bool:
    try:
        return "enable_graph" in path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False


def _comment_above_mode_field() -> str:
    """The contiguous comment block the preset writes directly above ``channel_finder_mode:``."""
    lines = PRESET_PATH.read_text(encoding="utf-8").splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("channel_finder_mode:"))
    comment: list[str] = []
    for line in reversed(lines[:index]):
        if not line.startswith("#"):
            break
        comment.append(line)
    return "\n".join(reversed(comment))


def test_mode_field_comment_names_every_paradigm():
    """The comment beside `channel_finder_mode:` enumerates the registry, not a subset."""
    comment = _comment_above_mode_field()
    assert comment, "no comment above channel_finder_mode: in the preset"
    # A mode counts only as a standalone token: "knowledge graph" in the prose
    # must not stand in for the `graph` paradigm.
    prose = comment.replace("knowledge graph", "")
    missing = [
        mode
        for mode in VALID_CHANNEL_FINDER_MODES
        if not re.search(rf"(?<![\w-]){re.escape(mode)}(?![\w-])", prose)
    ]
    assert missing == [], f"{PRESET_PATH.name} names no paradigm {missing} in {comment!r}"
