"""Tests for effort level reading from config."""

import re
from pathlib import Path

import pytest
import yaml

from osprey.cli.chat_cmd import chat as chat_command
from osprey.interfaces.web_terminal.routes.websocket import _read_effort_level

_SETTINGS_JS = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "interfaces"
    / "web_terminal"
    / "static"
    / "js"
    / "settings.js"
)


class TestReadEffortLevel:
    """Test _read_effort_level helper."""

    def test_returns_effort_when_present(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text(yaml.dump({"claude_code": {"effort": "high"}}))
        assert _read_effort_level(config) == "high"

    def test_returns_none_when_absent(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text(yaml.dump({"claude_code": {"provider": "cborg"}}))
        assert _read_effort_level(config) is None

    def test_returns_none_when_no_claude_code_section(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text(yaml.dump({"control_system": {"connector": "mock"}}))
        assert _read_effort_level(config) is None

    def test_returns_none_for_none_path(self):
        assert _read_effort_level(None) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _read_effort_level(tmp_path / "nonexistent.yml") is None

    def test_returns_none_for_invalid_yaml(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text(": invalid: yaml: {{{{")
        assert _read_effort_level(config) is None

    def test_returns_none_for_empty_file(self, tmp_path):
        config = tmp_path / "config.yml"
        config.write_text("")
        assert _read_effort_level(config) is None

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "max"])
    def test_all_valid_levels(self, tmp_path, effort):
        config = tmp_path / "config.yml"
        config.write_text(yaml.dump({"claude_code": {"effort": effort}}))
        assert _read_effort_level(config) == effort


def _cli_effort_choices() -> list[str]:
    """The effort vocabulary ``osprey chat --effort`` accepts."""
    for param in chat_command.params:
        if param.name == "effort":
            return list(param.type.choices)
    raise AssertionError("osprey chat has no --effort option")


def test_the_settings_drawer_offers_the_cli_effort_vocabulary() -> None:
    """The drawer's effort select and the CLI flag speak one vocabulary.

    The drawer writes ``claude_code.effort`` straight into the deployment's
    config; a value the CLI would reject is a setting that looks applied and
    then falls back at the next launch, with nothing said. JavaScript cannot
    import the click option, so the list is pinned here instead.
    """
    match = re.search(
        r"'claude_code\.effort':\s*\[([^\]]*)\]", _SETTINGS_JS.read_text(encoding="utf-8")
    )
    assert match, f"no claude_code.effort entry in {_SETTINGS_JS}"
    rendered = re.findall(r"'([^']+)'", match.group(1))
    assert rendered == _cli_effort_choices()
