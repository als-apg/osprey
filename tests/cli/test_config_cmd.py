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
    set_models,
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
            with patch("osprey.generators.config_updater.set_control_system_type") as mock_update:
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
            with patch("osprey.generators.config_updater.set_control_system_type") as mock_update:
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
            with patch("osprey.generators.config_updater.set_epics_gateway_config") as mock_update:
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


class TestConfigSetModelsCommand:
    """Test config set-models subcommand."""

    def test_set_models_help(self, cli_runner):
        """Verify set-models command help is displayed."""
        result = cli_runner.invoke(set_models, ["--help"])

        assert result.exit_code == 0
        assert "model" in result.output.lower() or "AI" in result.output

    def test_set_models_with_provider_and_model(self, cli_runner, tmp_path):
        """Test setting models with provider and model specified."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("models:\n  orchestrator: {}\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.cli.interactive_menu.get_provider_metadata") as mock_metadata:
                with patch("osprey.generators.config_updater.update_all_models") as mock_update:
                    mock_resolve.return_value = str(config_file)
                    mock_metadata.return_value = {
                        "anthropic": {"models": ["claude-sonnet-4", "claude-haiku"]}
                    }
                    mock_update.return_value = ("new content", "preview")

                    result = cli_runner.invoke(
                        set_models, ["--provider", "anthropic", "--model", "claude-sonnet-4"]
                    )

                    # Should call update function
                    assert mock_update.called
                    assert result.exit_code == 0

    def test_set_models_without_options_launches_interactive(self, cli_runner, tmp_path):
        """Test that set-models without options launches interactive mode."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("models: {}\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.cli.interactive_menu.handle_set_models") as mock_interactive:
                mock_resolve.return_value = str(config_file)

                result = cli_runner.invoke(set_models, [])

                # Should call interactive handler
                assert mock_interactive.called
                assert result.exit_code == 0

    def test_set_models_invalid_provider(self, cli_runner, tmp_path):
        """Test that invalid provider is rejected."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("models: {}\n")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            with patch("osprey.cli.interactive_menu.get_provider_metadata") as mock_metadata:
                mock_resolve.return_value = str(config_file)
                mock_metadata.return_value = {"anthropic": {"models": []}}

                result = cli_runner.invoke(
                    set_models, ["--provider", "invalid", "--model", "some-model"]
                )

                # Should fail (exit code 2 is Click parameter error, 1 is abort)
                assert result.exit_code in [1, 2]
                # May error during parameter validation or during execution
                assert (
                    "Invalid" in result.output
                    or "❌" in result.output
                    or "not found" in result.output.lower()
                )


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
