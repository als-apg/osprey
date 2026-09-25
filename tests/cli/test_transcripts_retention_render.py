"""``claude_code.transcripts.retention_days`` renders Claude Code's ``cleanupPeriodDays``.

Claude Code deletes every session transcript older than ``cleanupPeriodDays``
at startup, 30 days when nothing sets it. The terminals and the dispatch worker
load project settings only, so the rendered ``.claude/settings.json`` is the one
place a deployment can state how long its transcripts are kept. Absent, the
line is absent and Claude Code's own default applies; a value that cannot be a
retention is refused at build rather than dropped, because a dropped value
leaves the 30-day deletion in force.
"""

from __future__ import annotations

import json

import pytest
import yaml

from osprey.cli.templates.claude_code import config_derived_context
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import BuildProfileError
from tests.cli.test_terminal_theme_render import _create_project


def _regen_with_retention(tmp_path, days):
    manager = TemplateManager()
    project_dir = _create_project(
        manager,
        project_name="retention-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    transcripts = config.setdefault("claude_code", {}).setdefault("transcripts", {})
    if days is None:
        transcripts.pop("retention_days", None)
    else:
        transcripts["retention_days"] = days
    (project_dir / "config.yml").write_text(yaml.dump(config))
    manager.regenerate_claude_code(project_dir)
    return (project_dir / ".claude" / "settings.json").read_text()


def test_no_key_renders_no_cleanup_period(tmp_path):
    text = _regen_with_retention(tmp_path, None)
    assert "cleanupPeriodDays" not in text


def test_the_key_renders_cleanup_period_days(tmp_path):
    text = _regen_with_retention(tmp_path, 3650)
    assert '"cleanupPeriodDays": 3650' in text
    assert json.loads(text)["cleanupPeriodDays"] == 3650


@pytest.mark.parametrize("value", [0, -1, True, "30", 1.5])
def test_zero_negative_bool_and_string_are_refused(tmp_path, value):
    config = {"claude_code": {"transcripts": {"retention_days": value}}}
    with pytest.raises(BuildProfileError, match=r"claude_code\.transcripts\.retention_days"):
        config_derived_context(config, tmp_path)
