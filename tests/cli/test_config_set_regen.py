"""Tests that `osprey config set-*` commands regenerate Claude Code artifacts.

config.yml is a build-time input — switching control_system.type must re-render
the artifacts that depend on it (e.g. .claude/rules/control-system-safety.md) so
the CLI stays consistent with `osprey build` / `osprey claude regen`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.config_cmd import config
from osprey.cli.templates.manager import TemplateManager
from osprey.utils.config_writer import config_update_fields

PRESETS_DIR = Path(__file__).parents[2] / "src" / "osprey" / "profiles" / "presets"


def _build_project(tmp_path):
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="set-regen",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config_update_fields(project_dir / "config.yml", {"control_system.type": "mock"})
    manager.regenerate_claude_code(project_dir)
    return project_dir


def test_set_control_system_regenerates_safety_rule(tmp_path):
    project_dir = _build_project(tmp_path)
    safety_rule = project_dir / ".claude" / "rules" / "control-system-safety.md"
    before = safety_rule.read_text()

    result = CliRunner().invoke(
        config,
        ["set-control-system", "epics", "--project", str(project_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Regenerated" in result.output
    # The rendered safety rule (which keys off control_system.type) changed.
    assert safety_rule.read_text() != before


def test_set_epics_gateway_invokes_regen(tmp_path, monkeypatch):
    """set-epics-gateway is an independent callsite — verify it triggers regen too."""
    project_dir = _build_project(tmp_path)
    calls = []
    monkeypatch.setattr(
        TemplateManager, "regen_if_drift", lambda self, pd: calls.append(Path(pd)) or []
    )

    result = CliRunner().invoke(
        config,
        ["set-epics-gateway", "--facility", "als", "--project", str(project_dir)],
    )

    assert result.exit_code == 0, result.output
    assert calls, "set-epics-gateway must call regen_if_drift"
    assert calls[0] == project_dir


def test_cli_set_fails_open_when_regen_raises(tmp_path, monkeypatch):
    """A regen error must not fail the command; config.yml stays written."""
    project_dir = _build_project(tmp_path)

    def boom(self, pd):
        raise RuntimeError("regen exploded")

    monkeypatch.setattr(TemplateManager, "regen_if_drift", boom)
    result = CliRunner().invoke(
        config,
        ["set-control-system", "epics", "--project", str(project_dir)],
    )

    assert result.exit_code == 0, result.output
    assert "Could not regenerate" in result.output
    # The config write still landed despite the regen failure.
    cfg = yaml.safe_load((project_dir / "config.yml").read_text())
    assert cfg["control_system"]["type"] == "epics"


@pytest.mark.parametrize("preset", ["hello-world", "control-assistant", "ariel-standalone"])
def test_preset_declares_config_drift_hook(preset):
    """Every shipped preset lists config-drift so the SessionStart guard deploys."""
    data = yaml.safe_load((PRESETS_DIR / f"{preset}.yml").read_text())
    assert "config-drift" in data.get("hooks", []), (
        f"{preset} preset must include the config-drift hook"
    )
