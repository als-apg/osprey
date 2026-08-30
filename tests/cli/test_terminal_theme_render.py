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

import pytest
import yaml

from osprey.cli.templates.claude_code import config_derived_context
from osprey.cli.templates.manager import TemplateManager


def _regen_with_web_theme(tmp_path, theme: str | None):
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="theme-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    if theme is not None:
        config = yaml.safe_load((project_dir / "config.yml").read_text())
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
