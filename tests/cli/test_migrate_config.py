"""Tests for 'osprey migrate config' subcommand."""

import textwrap

import pytest
from click.testing import CliRunner
from ruamel.yaml import YAML

from osprey.cli.migrate_cmd import migrate


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def yaml_rt():
    """Round-trip YAML loader for verifying output."""
    y = YAML(typ="rt")
    y.preserve_quotes = True
    return y


def _write_config(tmp_path, content: str):
    """Write a config.yml from a dedented string."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(textwrap.dedent(content), encoding="utf-8")
    return config_path


class TestMigrateConfig:
    def test_migrate_disable_servers(self, runner, yaml_rt, tmp_path):
        """disable_servers list converts to servers with enabled: false."""
        _write_config(
            tmp_path,
            """\
            claude_code:
              disable_servers:
                - accelpapers
            """,
        )
        result = runner.invoke(migrate, ["config", "-p", str(tmp_path), "--apply"])
        assert result.exit_code == 0

        with open(tmp_path / "config.yml") as f:
            data = yaml_rt.load(f)

        cc = data["claude_code"]
        assert "disable_servers" not in cc
        assert cc["servers"]["accelpapers"]["enabled"] is False

    def test_migrate_extra_servers(self, runner, yaml_rt, tmp_path):
        """extra_servers dict moves to servers section."""
        _write_config(
            tmp_path,
            """\
            claude_code:
              extra_servers:
                my-srv:
                  command: echo hello
            """,
        )
        result = runner.invoke(migrate, ["config", "-p", str(tmp_path), "--apply"])
        assert result.exit_code == 0

        with open(tmp_path / "config.yml") as f:
            data = yaml_rt.load(f)

        cc = data["claude_code"]
        assert "extra_servers" not in cc
        assert cc["servers"]["my-srv"]["command"] == "echo hello"

    def test_migrate_disable_agents(self, runner, yaml_rt, tmp_path):
        """disable_agents list converts to agents with enabled: false."""
        _write_config(
            tmp_path,
            """\
            claude_code:
              disable_agents:
                - wiki-search
            """,
        )
        result = runner.invoke(migrate, ["config", "-p", str(tmp_path), "--apply"])
        assert result.exit_code == 0

        with open(tmp_path / "config.yml") as f:
            data = yaml_rt.load(f)

        cc = data["claude_code"]
        assert "disable_agents" not in cc
        assert cc["agents"]["wiki-search"]["enabled"] is False

    def test_migrate_all_combined(self, runner, yaml_rt, tmp_path):
        """All three legacy keys migrate together."""
        _write_config(
            tmp_path,
            """\
            claude_code:
              provider: anthropic
              disable_servers:
                - accelpapers
                - wiki
              extra_servers:
                my-srv:
                  command: echo hi
              disable_agents:
                - wiki-search
            """,
        )
        result = runner.invoke(migrate, ["config", "-p", str(tmp_path), "--apply"])
        assert result.exit_code == 0

        with open(tmp_path / "config.yml") as f:
            data = yaml_rt.load(f)

        cc = data["claude_code"]
        # Legacy keys removed
        for key in ("disable_servers", "extra_servers", "disable_agents"):
            assert key not in cc
        # New keys present
        assert cc["servers"]["accelpapers"]["enabled"] is False
        assert cc["servers"]["wiki"]["enabled"] is False
        assert cc["servers"]["my-srv"]["command"] == "echo hi"
        assert cc["agents"]["wiki-search"]["enabled"] is False
        # Non-legacy keys preserved
        assert cc["provider"] == "anthropic"

    def test_migrate_no_legacy_keys(self, runner, tmp_path):
        """Already new format — informative message, no error."""
        _write_config(
            tmp_path,
            """\
            claude_code:
              provider: anthropic
              servers:
                my-srv:
                  command: echo hi
            """,
        )
        result = runner.invoke(migrate, ["config", "-p", str(tmp_path)])
        assert result.exit_code == 0
        assert "already using new format" in result.output

    def test_migrate_dry_run(self, runner, yaml_rt, tmp_path):
        """--dry-run shows preview without modifying the file."""
        content = textwrap.dedent("""\
            claude_code:
              disable_servers:
                - accelpapers
        """)
        config_path = tmp_path / "config.yml"
        config_path.write_text(content, encoding="utf-8")

        result = runner.invoke(migrate, ["config", "-p", str(tmp_path)])
        assert result.exit_code == 0
        assert "Dry run" in result.output

        # File should be unchanged
        assert config_path.read_text(encoding="utf-8") == content

    def test_migrate_preserves_comments(self, runner, yaml_rt, tmp_path):
        """Comments in config.yml survive the migration."""
        _write_config(
            tmp_path,
            """\
            # Top-level comment
            claude_code:
              # Provider setting
              provider: anthropic
              disable_servers:
                - accelpapers
            """,
        )
        result = runner.invoke(migrate, ["config", "-p", str(tmp_path), "--apply"])
        assert result.exit_code == 0

        text = (tmp_path / "config.yml").read_text(encoding="utf-8")
        assert "# Top-level comment" in text
        assert "# Provider setting" in text
