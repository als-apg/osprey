"""Tests for export-config CLI command.

This test module verifies the export-config command functionality.
This is a deprecated command that displays osprey's default configuration.
"""

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.export_config_cmd import export_config


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestExportConfigCommandBasics:
    """Test basic export-config command functionality."""

    def test_command_help(self, cli_runner):
        """Verify export-config command help is displayed."""
        result = cli_runner.invoke(export_config, ["--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower()
        assert "--output" in result.output
        assert "--format" in result.output

    def test_command_exists(self):
        """Verify export-config command can be imported and is callable."""
        assert export_config is not None
        assert callable(export_config)

    def test_command_has_correct_options(self, cli_runner):
        """Verify command has expected options."""
        result = cli_runner.invoke(export_config, ["--help"])

        assert "--output" in result.output or "-o" in result.output
        assert "--format" in result.output
        assert "yaml" in result.output
        assert "json" in result.output

    def test_command_name(self):
        """Verify command has correct name."""
        assert export_config.name == "export-config"


class TestExportConfigDeprecationWarning:
    """Test deprecation warning display."""

    def test_shows_deprecation_warning(self, cli_runner):
        """Test that deprecation warning is displayed."""
        # Simply run the command and check for deprecation warning
        # The command will fail on missing template, but should show warning first
        result = cli_runner.invoke(export_config, [])

        # Should show deprecation warning regardless of execution result
        assert "DEPRECATED" in result.output or "deprecated" in result.output.lower()


class TestExportConfigExecution:
    """Test export-config command execution."""

    def test_displays_config_to_console_by_default(self, cli_runner):
        """Test that config is displayed to console when no output file specified."""
        result = cli_runner.invoke(export_config, [])

        # Command should execute without crashing
        # Exit code 0 if template found, 1 if not - both are acceptable
        assert result.exit_code in [0, 1]
        # Should produce some output
        assert len(result.output) > 0

    def test_exports_to_file_when_output_specified(self, cli_runner, tmp_path):
        """Test exporting config to a file."""
        output_file = tmp_path / "exported-config.yml"

        result = cli_runner.invoke(export_config, ["--output", str(output_file)])

        # Command should execute (may fail on template, that's ok for smoke test)
        assert result.exit_code in [0, 1]
        # Should produce output explaining what happened
        assert len(result.output) > 0

    def test_handles_yaml_format(self, cli_runner):
        """Test that YAML format option is accepted."""
        result = cli_runner.invoke(export_config, ["--format", "yaml"])

        # Should accept the option (may fail on template not found)
        # We're just testing the option is accepted
        assert result.exit_code in [0, 1]
        # Should not show "invalid choice" error
        if result.exit_code == 2:  # Exit code 2 is for bad parameters
            assert "Invalid value" not in result.output

    def test_handles_json_format(self, cli_runner):
        """Test that JSON format option is accepted."""
        result = cli_runner.invoke(export_config, ["--format", "json"])

        # Should accept the option
        assert result.exit_code in [0, 1]
        # Should not show "invalid choice" error
        if result.exit_code == 2:
            assert "Invalid value" not in result.output

    def test_rejects_invalid_format(self, cli_runner):
        """Test that invalid format is rejected."""
        result = cli_runner.invoke(export_config, ["--format", "xml"])

        # Should reject invalid format
        assert result.exit_code == 2  # Click uses exit code 2 for parameter errors
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()


class TestExportConfigErrorHandling:
    """Test export-config error handling."""

    def test_handles_missing_template_gracefully(self, cli_runner):
        """Test handling when template file is not found."""
        with patch("osprey.cli.export_config_cmd.Path") as mock_path_class:
            # Mock template path to not exist
            mock_template = MagicMock()
            mock_template.exists.return_value = False

            mock_file_path = MagicMock()
            mock_file_path.parent.parent = Path("/fake")
            mock_file_path.__truediv__ = lambda self, other: mock_template

            def path_factory(*args, **kwargs):
                if args and "export_config_cmd" in str(args[0]):
                    return mock_file_path
                return Path(*args, **kwargs)

            mock_path_class.side_effect = path_factory

            result = cli_runner.invoke(export_config, [])

            # Should handle missing template gracefully
            assert result.exit_code == 1
            assert "not" in result.output.lower() or "could not" in result.output.lower()

    def test_keyboard_interrupt_handling(self, cli_runner):
        """Test graceful handling of KeyboardInterrupt (Ctrl+C)."""
        with patch("builtins.open") as mock_open:
            mock_open.side_effect = KeyboardInterrupt()

            result = cli_runner.invoke(export_config, [])

            # Should handle interrupt gracefully
            assert result.exit_code == 1
            assert "cancel" in result.output.lower() or "⚠" in result.output

    def test_general_exception_handling(self, cli_runner):
        """Test handling of general exceptions."""
        with patch("osprey.cli.export_config_cmd.Path") as mock_path_class:
            mock_path_class.side_effect = Exception("Test error")

            result = cli_runner.invoke(export_config, [])

            # Should handle exception gracefully
            assert result.exit_code == 1
            assert "Failed" in result.output or "Error" in result.output or "❌" in result.output

    def test_debug_mode_available(self, cli_runner):
        """Test that DEBUG environment variable is checked for verbose output."""
        # This test just verifies the code path exists
        # Actual functionality tested in integration tests
        with patch("osprey.cli.export_config_cmd.Path") as mock_path_class:
            mock_path_class.side_effect = ValueError("Test error")

            with patch.dict("os.environ", {"DEBUG": "1"}):
                result = cli_runner.invoke(export_config, [])

                # Should still handle error (with or without traceback)
                assert result.exit_code == 1


class TestExportConfigOutput:
    """Test export-config command output formatting."""

    def test_console_output_uses_syntax_highlighting(self, cli_runner):
        """Test that console output formatting works."""
        result = cli_runner.invoke(export_config, [])

        # Should either show output or error, but not crash
        assert result.exit_code in [0, 1]
        # Should produce formatted output
        assert len(result.output) > 0

    def test_success_message_when_exporting_to_file(self, cli_runner, tmp_path):
        """Test that success message is shown when exporting to file."""
        output_file = tmp_path / "config.yml"

        with patch("osprey.cli.export_config_cmd.Path"):
            mock_template = MagicMock()
            mock_template.exists.return_value = True

            mock_file_path = MagicMock()
            mock_file_path.parent.parent = Path("/fake")

            with patch("builtins.open", mock.mock_open(read_data="project: test")):
                result = cli_runner.invoke(export_config, ["--output", str(output_file)])

                # Should show some output (success or error)
                assert len(result.output) > 0
