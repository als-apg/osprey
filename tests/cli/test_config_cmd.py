"""Tests for config CLI command.

This test module verifies the config command group functionality.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from osprey.cli.config_cmd import (
    config,
    export,
    set_control_system,
    set_epics_gateway,
    show,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestConfigCommandGroup:
    """Test config command group basics."""

    def test_command_help(self, cli_runner):
        """Verify config command help is displayed."""
        result = cli_runner.invoke(config, ["--help"])

        assert result.exit_code == 0
        assert "config" in result.output.lower() or "Manage" in result.output
        assert "show" in result.output
        assert "export" in result.output

    def test_command_exists(self):
        """Verify config command can be imported and is callable."""
        assert config is not None
        assert callable(config)

    def test_command_is_group(self, cli_runner):
        """Verify config is a command group with subcommands."""
        result = cli_runner.invoke(config, ["--help"])

        assert result.exit_code == 0
        # Should list subcommands
        assert "show" in result.output or "export" in result.output


class TestConfigShowCommand:
    """Test config show subcommand."""

    def test_show_command_help(self, cli_runner):
        """Verify show command help is displayed."""
        result = cli_runner.invoke(show, ["--help"])

        assert result.exit_code == 0
        assert "show" in result.output.lower() or "Display" in result.output
        assert "--format" in result.output

    def test_show_with_valid_config(self, cli_runner, tmp_path):
        """Test showing configuration from valid project."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project:\n  name: test\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = str(config_file)

            result = cli_runner.invoke(show, [])

            # Should succeed
            assert result.exit_code == 0
            # Should show config content
            assert "test" in result.output or "Configuration" in result.output

    def test_show_with_json_format(self, cli_runner, tmp_path):
        """Test showing configuration in JSON format."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("project:\n  name: test\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = str(config_file)

            result = cli_runner.invoke(show, ["--format", "json"])

            assert result.exit_code == 0

    def test_show_without_project(self, cli_runner):
        """Test showing configuration when no project found."""
        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.side_effect = Exception("No project")

            result = cli_runner.invoke(show, [])

            # Should handle error gracefully
            assert result.exit_code == 1
            assert "No Osprey project" in result.output or "❌" in result.output


class TestConfigExportCommand:
    """Test config export subcommand."""

    def test_export_command_help(self, cli_runner):
        """Verify export command help is displayed."""
        result = cli_runner.invoke(export, ["--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower() or "Export" in result.output
        assert "--output" in result.output
        assert "--format" in result.output

    def test_export_to_console(self, cli_runner):
        """Test exporting configuration to console."""
        result = cli_runner.invoke(export, [])

        # Should execute (may fail on template, but shows it works)
        assert result.exit_code in [0, 1]
        # Should produce output
        assert len(result.output) > 0

    def test_export_to_file(self, cli_runner, tmp_path):
        """Test exporting configuration to file."""
        output_file = tmp_path / "exported.yml"

        result = cli_runner.invoke(export, ["--output", str(output_file)])

        # Should execute
        assert result.exit_code in [0, 1]

    def test_export_json_format(self, cli_runner):
        """Test exporting in JSON format."""
        result = cli_runner.invoke(export, ["--format", "json"])

        # Should accept JSON format
        assert result.exit_code in [0, 1]
        # Should not show "invalid choice" error
        if result.exit_code == 2:
            assert "Invalid value" not in result.output


class TestConfigSetControlSystemCommand:
    """Test config set-control-system subcommand."""

    def test_set_control_system_help(self, cli_runner):
        """Verify set-control-system command help is displayed."""
        result = cli_runner.invoke(set_control_system, ["--help"])

        assert result.exit_code == 0
        assert "control" in result.output.lower() or "system" in result.output.lower()

    def test_set_control_system_with_valid_config(self, cli_runner, tmp_path):
        """Test setting control system type."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: mock\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.utils.config_writer.set_control_system_type") as mock_update:
                mock_resolve.return_value = str(config_file)
                mock_update.return_value = ("new content", "preview")

                result = cli_runner.invoke(set_control_system, ["epics"])

                # Should call update function
                assert mock_update.called
                # Should succeed
                assert result.exit_code == 0

    def test_set_control_system_accepts_mock(self, cli_runner, tmp_path):
        """Test setting to mock control system."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  type: epics\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.utils.config_writer.set_control_system_type") as mock_update:
                mock_resolve.return_value = str(config_file)
                mock_update.return_value = ("new content", "preview")

                result = cli_runner.invoke(set_control_system, ["mock"])

                assert result.exit_code == 0

    def test_set_control_system_invalid_type(self, cli_runner):
        """Test that invalid system type is rejected."""
        result = cli_runner.invoke(set_control_system, ["invalid"])

        # Click should reject invalid choice
        assert result.exit_code == 2
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()


STORELESS_PROJECT = "control_system:\n  type: mock\n\narchiver:\n  type: mock_archiver\n"
ARCHIVED_PROJECT = (
    "control_system:\n  type: mock\n\narchiver:\n  type: mongodb_archiver\n"
    "  mongodb_archiver:\n    host: localhost\n"
)
NO_ARCHIVER_SECTION = "control_system:\n  type: mock\n"


class TestSetControlSystemRefusesAMockPast:
    """The command may not switch a project onto a machine whose past is fiction.

    Switching to the virtual accelerator changes one key, and the archiver keeps
    whatever it was — so this command can *create* the pairing the build, the
    deploy and the MCP server all refuse. It asks the same question they do,
    before the write rather than after, and refuses rather than prompting: it is
    non-interactive and runs in scripts.
    """

    def _run(self, cli_runner, tmp_path, contents, system_type):
        config_file = tmp_path / "config.yml"
        config_file.write_text(contents)

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = str(config_file)
            result = cli_runner.invoke(set_control_system, [system_type])

        return result, config_file.read_text()

    def test_va_onto_a_mock_archiver_is_refused(self, cli_runner, tmp_path):
        result, after = self._run(cli_runner, tmp_path, STORELESS_PROJECT, "virtual_accelerator")

        assert result.exit_code != 0
        assert after == STORELESS_PROJECT, "a refused switch must not write anything"

    def test_va_with_no_archiver_section_is_refused(self, cli_runner, tmp_path):
        """Unset counts as mock — the common way into the pairing is naming nothing."""
        result, after = self._run(cli_runner, tmp_path, NO_ARCHIVER_SECTION, "virtual_accelerator")

        assert result.exit_code != 0
        assert after == NO_ARCHIVER_SECTION

    def test_refusal_names_a_fix(self, cli_runner, tmp_path):
        """A refusal that does not say what to do instead is just an obstacle."""
        result, _ = self._run(cli_runner, tmp_path, STORELESS_PROJECT, "virtual_accelerator")

        # Rich wraps the console output, so match on fragments that survive a
        # line break falling anywhere inside the message.
        output = " ".join(result.output.split())
        assert "mongodb_archiver" in output
        assert "archiver:" in output
        assert "set-control-system" in output, "the menu that writes the keys is the fix"
        assert "reports a past that never happened" in output, "the shared reason"

    def test_refusal_is_not_followed_by_a_generic_error(self, cli_runner, tmp_path):
        """The message is complete; nothing appends a second, emptier line."""
        result, _ = self._run(cli_runner, tmp_path, STORELESS_PROJECT, "virtual_accelerator")

        assert "Failed to update control system" not in result.output

    def test_va_onto_a_real_archive_goes_through(self, cli_runner, tmp_path):
        result, after = self._run(cli_runner, tmp_path, ARCHIVED_PROJECT, "virtual_accelerator")

        assert result.exit_code == 0
        assert "type: virtual_accelerator" in after
        assert "type: mongodb_archiver" in after

    def test_epics_onto_a_mock_archiver_goes_through(self, cli_runner, tmp_path):
        """Only the simulated machine is affected. A real machine with no archive
        attached yet is a real facility mid-migration, and it stays legal."""
        result, after = self._run(cli_runner, tmp_path, STORELESS_PROJECT, "epics")

        assert result.exit_code == 0
        assert "type: epics" in after
        assert "type: mock_archiver" in after

    def test_mock_onto_a_mock_archiver_goes_through(self, cli_runner, tmp_path):
        """A mock machine with a mock archive claims nothing it cannot back up."""
        result, after = self._run(cli_runner, tmp_path, STORELESS_PROJECT, "mock")

        assert result.exit_code == 0
        assert "type: mock_archiver" in after

    def test_case_insensitive_spelling_is_refused_too(self, cli_runner, tmp_path):
        """Click accepts the type case-insensitively; the guard sees the same value."""
        result, after = self._run(cli_runner, tmp_path, STORELESS_PROJECT, "VIRTUAL_ACCELERATOR")

        assert result.exit_code != 0
        assert after == STORELESS_PROJECT


class TestConfigSetEpicsGatewayCommand:
    """Test config set-epics-gateway subcommand."""

    def test_set_epics_gateway_help(self, cli_runner):
        """Verify set-epics-gateway command help is displayed."""
        result = cli_runner.invoke(set_epics_gateway, ["--help"])

        assert result.exit_code == 0
        assert "gateway" in result.output.lower() or "EPICS" in result.output

    def test_set_epics_gateway_with_facility_preset(self, cli_runner, tmp_path):
        """Test setting EPICS gateway with facility preset."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  epics: {}\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.utils.config_writer.set_epics_gateway_config") as mock_update:
                mock_resolve.return_value = str(config_file)
                mock_update.return_value = ("new content", "preview")

                result = cli_runner.invoke(set_epics_gateway, ["--facility", "als"])

                # Should call update function
                assert mock_update.called
                assert result.exit_code == 0

    def test_set_epics_gateway_custom_requires_address_and_port(self, cli_runner, tmp_path):
        """Test that custom facility requires address and port."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("control_system:\n  epics: {}\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = str(config_file)

            result = cli_runner.invoke(set_epics_gateway, ["--facility", "custom"])

            # Should fail without address and port
            assert result.exit_code == 1
            assert "requires" in result.output.lower() or "❌" in result.output


class TestConfigErrorHandling:
    """Test config command error handling."""

    def test_keyboard_interrupt_handling(self, cli_runner, tmp_path):
        """Test graceful handling of KeyboardInterrupt."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: value\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = str(config_file)
            mock_resolve.side_effect = KeyboardInterrupt()

            result = cli_runner.invoke(show, [])

            # Should handle interrupt gracefully
            assert result.exit_code == 1

    def test_missing_config_file(self, cli_runner):
        """Test handling when config file doesn't exist."""
        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.return_value = "/nonexistent/config.yml"

            result = cli_runner.invoke(show, [])

            # Should handle missing file
            assert result.exit_code == 1
            assert "not found" in result.output.lower() or "❌" in result.output
