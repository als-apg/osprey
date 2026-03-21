"""Tests for the osprey build command and build profile system."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from osprey.cli.build_profile import (
    BuildProfile,
    BuildVariant,
    McpServerDef,
    _deep_merge,
    load_profile,
)
from osprey.errors import BuildProfileError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_dir(tmp_path: Path) -> Path:
    """Create a minimal profile directory with overlay sources."""
    # Create overlay source files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "channels.json").write_text('{"pvs": ["SR:DCCT"]}')

    mcp_dir = tmp_path / "mcp_servers" / "test_server"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "__init__.py").write_text("")
    (mcp_dir / "server.py").write_text("# test server")

    return tmp_path


@pytest.fixture()
def minimal_profile_yaml(profile_dir: Path) -> Path:
    """Write a minimal valid profile YAML and return its path."""
    profile = {
        "name": "Test Profile",
        "base_template": "control_assistant",
        "provider": "cborg",
        "model": "haiku",
        "config": {
            "control_system.type": "mock",
        },
        "overlay": {
            "data/channels.json": "data/channel_databases/channels.json",
        },
        "mcp_servers": {
            "test_server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "env": {
                    "CONFIG": "{project_root}/config.yml",
                },
                "permissions": {
                    "allow": ["test_tool"],
                    "ask": ["dangerous_tool"],
                },
            },
        },
        "variants": {
            "prod": {
                "config": {
                    "control_system.type": "epics",
                },
            },
            "test": {
                "config": {
                    "control_system.type": "mock",
                },
            },
        },
    }
    path = profile_dir / "test-profile.yml"
    path.write_text(yaml.dump(profile, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# Profile Loading
# ---------------------------------------------------------------------------


class TestProfileLoading:
    """Tests for load_profile() and YAML parsing."""

    def test_load_minimal_profile(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert profile.name == "Test Profile"
        assert profile.base_template == "control_assistant"
        assert profile.provider == "cborg"
        assert profile.model == "haiku"

    def test_load_profile_not_found(self, tmp_path: Path):
        with pytest.raises(BuildProfileError, match="Profile not found"):
            load_profile(tmp_path / "nonexistent.yml")

    def test_load_profile_invalid_yaml(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yml"
        bad_yaml.write_text("{{invalid yaml: [")
        with pytest.raises(BuildProfileError, match="Invalid YAML"):
            load_profile(bad_yaml)

    def test_load_profile_not_a_mapping(self, tmp_path: Path):
        list_yaml = tmp_path / "list.yml"
        list_yaml.write_text("- item1\n- item2\n")
        with pytest.raises(BuildProfileError, match="must be a YAML mapping"):
            load_profile(list_yaml)

    def test_load_profile_config_parsed(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert profile.config["control_system.type"] == "mock"

    def test_load_profile_mcp_servers_parsed(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert "test_server" in profile.mcp_servers
        server = profile.mcp_servers["test_server"]
        assert server.command == "python"
        assert server.args == ["-m", "test_server"]
        assert server.env == {"CONFIG": "{project_root}/config.yml"}
        assert server.permissions == {"allow": ["test_tool"], "ask": ["dangerous_tool"]}

    def test_load_profile_variants_parsed(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert "prod" in profile.variants
        assert "test" in profile.variants
        assert profile.variants["prod"].config["control_system.type"] == "epics"

    def test_load_profile_defaults(self, tmp_path: Path):
        """Profile with only name should use defaults."""
        simple = tmp_path / "simple.yml"
        simple.write_text("name: Simple\n")
        profile = load_profile(simple)
        assert profile.base_template == "control_assistant"
        assert profile.provider is None
        assert profile.config == {}
        assert profile.overlay == {}
        assert profile.mcp_servers == {}


# ---------------------------------------------------------------------------
# Variant Merging
# ---------------------------------------------------------------------------


class TestVariantMerging:
    """Tests for BuildProfile.with_variant()."""

    def test_variant_overrides_config(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        merged = profile.with_variant("prod")
        assert merged.config["control_system.type"] == "epics"

    def test_variant_preserves_other_config(self, minimal_profile_yaml: Path):
        """Variant should not remove config keys not in the variant."""
        profile = load_profile(minimal_profile_yaml)
        # Add extra config that variant doesn't touch
        profile.config["extra.key"] = "value"
        merged = profile.with_variant("prod")
        assert merged.config["extra.key"] == "value"
        assert merged.config["control_system.type"] == "epics"

    def test_unknown_variant_raises(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        with pytest.raises(BuildProfileError, match="Unknown variant 'nonexistent'"):
            profile.with_variant("nonexistent")

    def test_variant_clears_variants_dict(self, minimal_profile_yaml: Path):
        """After applying a variant, variants dict is emptied to prevent double-apply."""
        profile = load_profile(minimal_profile_yaml)
        merged = profile.with_variant("prod")
        assert merged.variants == {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for BuildProfile.validate()."""

    def test_missing_overlay_source(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            overlay={"nonexistent/file.json": "data/file.json"},
        )
        with pytest.raises(BuildProfileError, match="Overlay source not found"):
            profile.validate(tmp_path)

    def test_path_traversal_blocked(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            overlay={},
        )
        # Manually add a traversal path
        profile.overlay["data/x.json"] = "../../../etc/passwd"
        # Create the source file so we don't hit that error
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "x.json").write_text("{}")
        with pytest.raises(BuildProfileError, match="must be relative without"):
            profile.validate(tmp_path)

    def test_absolute_overlay_destination_blocked(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            overlay={"x.json": "/tmp/evil.json"},
        )
        (tmp_path / "x.json").write_text("{}")
        with pytest.raises(BuildProfileError, match="must be relative without"):
            profile.validate(tmp_path)

    def test_missing_mcp_server_command(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            mcp_servers={"broken": McpServerDef(command="")},
        )
        with pytest.raises(BuildProfileError, match="missing 'command'"):
            profile.validate(tmp_path)

    def test_missing_name_reported(self, tmp_path: Path):
        profile = BuildProfile(name="")
        with pytest.raises(BuildProfileError, match="'name' is required"):
            profile.validate(tmp_path)

    def test_valid_profile_passes(self, profile_dir: Path, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        # Should not raise
        profile.validate(profile_dir)


# ---------------------------------------------------------------------------
# Deep Merge
# ---------------------------------------------------------------------------


class TestDeepMerge:
    """Tests for the _deep_merge helper."""

    def test_simple_override(self):
        result = _deep_merge({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_nested_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 99}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": 99, "c": 2}}

    def test_new_keys_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_no_mutation_of_originals(self):
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}


# ---------------------------------------------------------------------------
# Build Command Helpers
# ---------------------------------------------------------------------------


class TestBuildHelpers:
    """Tests for build_cmd.py helper functions."""

    def test_resolve_placeholders(self):
        from osprey.cli.build_cmd import _resolve_placeholders

        result = _resolve_placeholders("{project_root}/config.yml", Path("/tmp/test"))
        assert result == "/tmp/test/config.yml"

    def test_resolve_placeholders_no_match(self):
        from osprey.cli.build_cmd import _resolve_placeholders

        result = _resolve_placeholders("plain-string", Path("/tmp/test"))
        assert result == "plain-string"

    def test_copy_overlay_path_traversal_guard(self, tmp_path: Path):
        """_copy_overlay_files should reject destinations that escape project root."""
        from osprey.cli.build_cmd import _copy_overlay_files

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "evil.txt").write_text("evil")

        project_path = tmp_path / "project"
        project_path.mkdir()

        overlay = {"evil.txt": "../../../tmp/evil.txt"}
        with pytest.raises(ValueError, match="escapes project root"):
            _copy_overlay_files(profile_dir, project_path, overlay)

    def test_copy_overlay_files(self, tmp_path: Path):
        """_copy_overlay_files should copy files into the project."""
        from osprey.cli.build_cmd import _copy_overlay_files

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "data.json").write_text('{"key": "value"}')

        project_path = tmp_path / "project"
        project_path.mkdir()

        overlay = {"data.json": "config/data.json"}
        _copy_overlay_files(profile_dir, project_path, overlay)

        assert (project_path / "config" / "data.json").exists()
        assert json.loads((project_path / "config" / "data.json").read_text()) == {"key": "value"}

    def test_copy_overlay_directory(self, tmp_path: Path):
        """_copy_overlay_files should handle directory overlays."""
        from osprey.cli.build_cmd import _copy_overlay_files

        profile_dir = tmp_path / "profile"
        src_dir = profile_dir / "server_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("")
        (src_dir / "main.py").write_text("# main")

        project_path = tmp_path / "project"
        project_path.mkdir()

        overlay = {"server_pkg": "_mcp_servers/server_pkg"}
        _copy_overlay_files(profile_dir, project_path, overlay)

        assert (project_path / "_mcp_servers" / "server_pkg" / "__init__.py").exists()
        assert (project_path / "_mcp_servers" / "server_pkg" / "main.py").exists()

    def test_inject_mcp_servers_mcp_json(self, tmp_path: Path):
        """_inject_mcp_servers should add server to .mcp.json."""
        from osprey.cli.build_cmd import _inject_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / ".mcp.json").write_text('{"mcpServers": {}}')
        (project_path / ".claude").mkdir()
        (project_path / ".claude" / "settings.json").write_text(
            '{"permissions": {"allow": [], "ask": []}}'
        )

        servers = {
            "phoebus": McpServerDef(
                command="python",
                args=["-m", "phoebus"],
                env={"CONFIG": "{project_root}/config.yml"},
                permissions={"allow": ["phoebus_launch"], "ask": []},
            ),
        }
        _inject_mcp_servers(project_path, servers)

        mcp_data = json.loads((project_path / ".mcp.json").read_text())
        assert "phoebus" in mcp_data["mcpServers"]
        server_entry = mcp_data["mcpServers"]["phoebus"]
        assert server_entry["command"] == "python"
        assert server_entry["args"] == ["-m", "phoebus"]
        # Verify placeholder resolution
        assert server_entry["env"]["CONFIG"] == f"{project_path}/config.yml"

    def test_inject_mcp_servers_permissions(self, tmp_path: Path):
        """_inject_mcp_servers should add tool permissions to settings.json."""
        from osprey.cli.build_cmd import _inject_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / ".mcp.json").write_text('{"mcpServers": {}}')
        (project_path / ".claude").mkdir()
        (project_path / ".claude" / "settings.json").write_text(
            '{"permissions": {"allow": ["existing_perm"], "ask": []}}'
        )

        servers = {
            "phoebus": McpServerDef(
                command="python",
                args=[],
                permissions={"allow": ["phoebus_launch"], "ask": ["dangerous_op"]},
            ),
        }
        _inject_mcp_servers(project_path, servers)

        settings = json.loads((project_path / ".claude" / "settings.json").read_text())
        assert "mcp__phoebus__phoebus_launch" in settings["permissions"]["allow"]
        assert "existing_perm" in settings["permissions"]["allow"]
        assert "mcp__phoebus__dangerous_op" in settings["permissions"]["ask"]

    def test_inject_mcp_servers_no_duplicate_permissions(self, tmp_path: Path):
        """Running inject twice shouldn't duplicate permission entries."""
        from osprey.cli.build_cmd import _inject_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / ".mcp.json").write_text('{"mcpServers": {}}')
        (project_path / ".claude").mkdir()
        (project_path / ".claude" / "settings.json").write_text(
            '{"permissions": {"allow": ["mcp__phoebus__phoebus_launch"], "ask": []}}'
        )

        servers = {
            "phoebus": McpServerDef(
                command="python",
                args=[],
                permissions={"allow": ["phoebus_launch"], "ask": []},
            ),
        }
        _inject_mcp_servers(project_path, servers)

        settings = json.loads((project_path / ".claude" / "settings.json").read_text())
        # Should appear only once
        assert settings["permissions"]["allow"].count("mcp__phoebus__phoebus_launch") == 1

    def test_apply_config_overrides(self, tmp_path: Path):
        """_apply_config_overrides should update config.yml fields."""
        from osprey.cli.build_cmd import _apply_config_overrides

        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = project_path / "config.yml"
        config_path.write_text(
            dedent("""\
            control_system:
              type: mock
            archiver:
              type: mock_archiver
            """)
        )

        _apply_config_overrides(
            project_path,
            {
                "control_system.type": "epics",
                "system.timezone": "America/Los_Angeles",
            },
        )

        updated = yaml.safe_load(config_path.read_text())
        assert updated["control_system"]["type"] == "epics"
        assert updated["system"]["timezone"] == "America/Los_Angeles"
        # Verify untouched fields preserved
        assert updated["archiver"]["type"] == "mock_archiver"


# ---------------------------------------------------------------------------
# CLI Integration
# ---------------------------------------------------------------------------


class TestBuildCLI:
    """Tests for the Click command integration."""

    def test_build_command_exists(self):
        """Verify build command is registered in the CLI."""
        from click.testing import CliRunner

        from osprey.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--help"])
        assert result.exit_code == 0
        assert "Build a facility-specific assistant" in result.output

    def test_build_command_missing_profile(self):
        """Build should fail if profile file doesn't exist."""
        from click.testing import CliRunner

        from osprey.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["build", "test-proj", "/nonexistent/profile.yml"])
        assert result.exit_code != 0
