"""Tests for effort level reading from config."""

import pytest
import yaml

from osprey.interfaces.web_terminal.routes.websocket import _read_effort_level


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
