"""``web.theme`` drives the rendered terminal theme — one key, both surfaces.

The deployment's ``web.theme`` already decides how the web UI looks. Claude
Code's ``theme`` settings key has scope "any file", so the render can state it
in ``build/.claude/settings.json`` and the terminal follows the same
deployment decision — no second knob, and no seeding into volume state.

The mapping honors the family/id distinction ``theme_config`` maintains: a
concrete theme id (``desy-light``) *pins* a mode, and only a pin is rendered.
A family (``desy``) deliberately leaves light/dark to each viewer's OS — a
terminal cannot follow the OS, so the render stays silent and Claude Code's
own default applies rather than OSPREY inventing a pin the operator never
stated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from osprey.cli.templates.claude_code import config_derived_context
from osprey.cli.templates.manager import TemplateManager


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages, which is the content the render used to reach
    for on its own.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", _bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


def _regen_with_web_theme(tmp_path, theme: str | None):
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        project_name="theme-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    # The preset states a theme of its own, so the absent case has to take it
    # back out rather than simply not writing one.
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    if theme is None:
        (config.get("web") or {}).pop("theme", None)
    else:
        config.setdefault("web", {})["theme"] = theme
    (project_dir / "config.yml").write_text(yaml.dump(config))
    manager.regenerate_claude_code(project_dir)
    return json.loads((project_dir / ".claude" / "settings.json").read_text())


def test_pinned_theme_id_renders_the_terminal_theme(tmp_path):
    settings = _regen_with_web_theme(tmp_path, "desy-light")
    assert settings["theme"] == "light"


def test_family_value_renders_no_terminal_theme(tmp_path):
    """A family pins no mode, so the render must not invent one."""
    settings = _regen_with_web_theme(tmp_path, "desy")
    assert "theme" not in settings


def test_absent_web_theme_renders_no_terminal_theme(tmp_path):
    settings = _regen_with_web_theme(tmp_path, None)
    assert "theme" not in settings


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("dark", "dark"),  # the main family's concrete ids pin too
        ("light", "light"),
        ("high-contrast", None),  # family
        ("no-such-theme", None),  # unknown is not a pin
        (None, None),
    ],
)
def test_context_derivation(tmp_path, configured, expected):
    config = {"web": {"theme": configured}} if configured is not None else {}
    ctx = config_derived_context(config, tmp_path)
    assert ctx["terminal_theme"] == expected
