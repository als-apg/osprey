"""``claude_code.thinking.display`` renders Claude Code's ``showThinkingSummaries``.

Interactive sessions read the thinking display from the project settings file,
so the rendered ``.claude/settings.json`` carries ``showThinkingSummaries`` on
every build: ``true`` for ``summarized`` (the default) and ``false`` for
``omitted``. The key is always stated, never left out, so a missing context
value can never silently fall back to Claude Code's own default.
"""

from __future__ import annotations

import json

import pytest
import yaml

from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import create_project_as_built


def _regen_with_display(tmp_path, display):
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
        project_name="thinking-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    claude_code = config.setdefault("claude_code", {})
    if display is None:
        claude_code.pop("thinking", None)
    else:
        claude_code.setdefault("thinking", {})["display"] = display
    (project_dir / "config.yml").write_text(yaml.dump(config))
    manager.regenerate_claude_code(project_dir)
    return (project_dir / ".claude" / "settings.json").read_text()


@pytest.mark.parametrize(
    ("display", "expected"),
    [(None, True), ("summarized", True), ("omitted", False)],
)
def test_display_renders_show_thinking_summaries(tmp_path, display, expected):
    text = _regen_with_display(tmp_path, display)
    settings = json.loads(text)
    assert "showThinkingSummaries" in settings
    assert settings["showThinkingSummaries"] is expected
    assert f'"showThinkingSummaries": {json.dumps(expected)}' in text


def test_rendered_settings_carry_no_jinja_comment_text(tmp_path):
    text = _regen_with_display(tmp_path, "omitted")
    assert "claude_code.thinking.display" not in text
