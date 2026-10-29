"""Claude Code's ``cleanupPeriodDays`` renders as max(``agent_record.retention_days``, 30).

Claude Code deletes every session transcript older than ``cleanupPeriodDays``
at startup. The terminals and the dispatch worker load project settings only,
so the rendered ``.claude/settings.json`` is where a deployment states how long
its transcripts are kept: never shorter than the record keeps them, never below
Claude Code's own 30 days. There is no key of its own, and host settings are
never written.
"""

from __future__ import annotations

import json

import pytest
import yaml

from osprey.cli.templates.claude_code import config_derived_context
from osprey.cli.templates.manager import TemplateManager
from tests._preset_data import create_project_as_built


def _regen_with_retention(tmp_path, days):
    manager = TemplateManager()
    project_dir = create_project_as_built(
        manager,
        project_name="retention-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    config.pop("agent_record.retention_days", None)
    block = config.setdefault("agent_record", {})
    if days is None:
        block.pop("retention_days", None)
    else:
        block["retention_days"] = days
    (project_dir / "config.yml").write_text(yaml.dump(config))
    manager.regenerate_claude_code(project_dir)
    return (project_dir / ".claude" / "settings.json").read_text()


def test_no_record_retention_renders_the_thirty_day_floor(tmp_path):
    text = _regen_with_retention(tmp_path, None)
    assert json.loads(text)["cleanupPeriodDays"] == 30


def test_a_record_retention_above_thirty_is_rendered(tmp_path):
    text = _regen_with_retention(tmp_path, 3650)
    assert '"cleanupPeriodDays": 3650' in text
    assert json.loads(text)["cleanupPeriodDays"] == 3650


def test_a_record_retention_below_thirty_is_floored(tmp_path):
    text = _regen_with_retention(tmp_path, 7)
    assert json.loads(text)["cleanupPeriodDays"] == 30


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, 30),
        ({"agent_record": {"retention_days": 90}}, 90),
        ({"agent_record.retention_days": 90}, 90),
        ({"agent_record": {"retention_days": 29}}, 30),
        ({"agent_record": {"retention_days": 30}}, 30),
        ({"agent_record": {"retention_days": 0}}, 30),
        ({"agent_record": {"retention_days": True}}, 30),
        # ``claude_code.transcripts`` sets nothing.
        ({"claude_code": {"transcripts": {"retention_days": 3650}}}, 30),
    ],
)
def test_cleanup_period_is_the_larger_of_record_retention_and_thirty(config, expected, tmp_path):
    assert config_derived_context(config, tmp_path)["cleanup_period_days"] == expected
