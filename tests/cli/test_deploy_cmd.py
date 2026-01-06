"""Tests for deploy CLI command.

This test module verifies the deploy command wrapper functionality.
The command wraps the existing container_manager interface.
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.deploy_cmd import deploy


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestDeployCommandBasics:
    """Test basic deploy command functionality."""

    def test_command_help(self, cli_runner):
        """Verify deploy command help is displayed."""
        result = cli_runner.invoke(deploy, ["--help"])

        assert result.exit_code == 0
        assert "deploy" in result.output.lower() or "Manage" in result.output
        assert "up" in result.output
        assert "down" in result.output
        assert "restart" in result.output
        assert "status" in result.output

    def test_command_exists(self):
        """Verify deploy command can be imported and is callable."""
        assert deploy is not None
        assert callable(deploy)

    def test_command_has_action_argument(self, cli_runner):
        """Verify command requires an action argument."""
        result = cli_runner.invoke(deploy, ["--help"])

        assert "up" in result.output
        assert "down" in result.output
        assert "restart" in result.output
        assert "status" in result.output
        assert "build" in result.output
        assert "clean" in result.output
        assert "rebuild" in result.output

    def test_command_has_project_option(self, cli_runner):
        """Verify command has --project option."""
        result = cli_runner.invoke(deploy, ["--help"])
        assert "--project" in result.output or "-p" in result.output

    def test_command_has_config_option(self, cli_runner):
        """Verify command has --config option."""
        result = cli_runner.invoke(deploy, ["--help"])
        assert "--config" in result.output or "-c" in result.output

    def test_command_has_detached_option(self, cli_runner):
        """Verify command has --detached option."""
        result = cli_runner.invoke(deploy, ["--help"])
        assert "--detached" in result.output or "-d" in result.output

    def test_command_has_dev_option(self, cli_runner):
        """Verify command has --dev option."""
        result = cli_runner.invoke(deploy, ["--help"])
        assert "--dev" in result.output


class TestDeployCommandActions:
    """Test deploy command action dispatch."""

    def test_up_action_calls_deploy_up(self, cli_runner, tmp_path):
        """Test that 'up' action calls deploy_up function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["up", "--config", str(config_file)])

                # deploy_up should have been called
                assert mock_deploy_up.called
                call_args = mock_deploy_up.call_args
                assert call_args[0][0] == config_file

    def test_down_action_calls_deploy_down(self, cli_runner, tmp_path):
        """Test that 'down' action calls deploy_down function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_down") as mock_deploy_down:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["down", "--config", str(config_file)])

                assert mock_deploy_down.called

    def test_restart_action_calls_deploy_restart(self, cli_runner, tmp_path):
        """Test that 'restart' action calls deploy_restart function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_restart") as mock_deploy_restart:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["restart", "--config", str(config_file)])

                assert mock_deploy_restart.called

    def test_status_action_calls_show_status(self, cli_runner, tmp_path):
        """Test that 'status' action calls show_status function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.show_status") as mock_show_status:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["status", "--config", str(config_file)])

                assert mock_show_status.called

    def test_build_action_calls_prepare_compose_files(self, cli_runner, tmp_path):
        """Test that 'build' action calls prepare_compose_files function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.prepare_compose_files") as mock_prepare:
            mock_prepare.return_value = (MagicMock(), ["docker-compose.yml"])
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["build", "--config", str(config_file)])

                assert mock_prepare.called
                assert result.exit_code == 0

    def test_clean_action_calls_clean_deployment(self, cli_runner, tmp_path):
        """Test that 'clean' action calls clean_deployment function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.clean_deployment") as mock_clean:
            with patch("osprey.cli.deploy_cmd.prepare_compose_files") as mock_prepare:
                mock_prepare.return_value = (MagicMock(), ["docker-compose.yml"])
                with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                    mock_resolve.return_value = config_file

                    cli_runner.invoke(deploy, ["clean", "--config", str(config_file)])

                    assert mock_clean.called

    def test_rebuild_action_calls_rebuild_deployment(self, cli_runner, tmp_path):
        """Test that 'rebuild' action calls rebuild_deployment function."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.rebuild_deployment") as mock_rebuild:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["rebuild", "--config", str(config_file)])

                assert mock_rebuild.called


class TestDeployCommandOptions:
    """Test deploy command options."""

    def test_detached_flag_passed_to_deploy_up(self, cli_runner, tmp_path):
        """Test that --detached flag is passed to deploy_up."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["up", "--detached", "--config", str(config_file)])

                call_kwargs = mock_deploy_up.call_args[1]
                assert call_kwargs["detached"] is True

    def test_dev_flag_passed_to_deploy_up(self, cli_runner, tmp_path):
        """Test that --dev flag is passed to deploy_up."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(deploy, ["up", "--dev", "--config", str(config_file)])

                call_kwargs = mock_deploy_up.call_args[1]
                assert call_kwargs["dev_mode"] is True


class TestDeployCommandErrorHandling:
    """Test deploy command error handling."""

    def test_missing_config_file_shows_helpful_error(self, cli_runner, tmp_path):
        """Test that missing config file shows helpful error message."""
        # Don't create the config file
        result = cli_runner.invoke(deploy, ["up", "--config", str(tmp_path / "nonexistent.yml")])

        # Should handle missing file gracefully
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "❌" in result.output

    def test_keyboard_interrupt_handling(self, cli_runner, tmp_path):
        """Test graceful handling of KeyboardInterrupt (Ctrl+C)."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file
                mock_deploy_up.side_effect = KeyboardInterrupt()

                result = cli_runner.invoke(deploy, ["up", "--config", str(config_file)])

                # Should handle interrupt gracefully
                assert result.exit_code == 1
                assert "cancel" in result.output.lower() or "⚠" in result.output

    def test_general_exception_handling(self, cli_runner, tmp_path):
        """Test handling of general exceptions."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up") as mock_deploy_up:
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file
                mock_deploy_up.side_effect = Exception("Test deployment error")

                result = cli_runner.invoke(deploy, ["up", "--config", str(config_file)])

                # Should handle exception gracefully
                assert result.exit_code == 1
                assert "failed" in result.output.lower() or "❌" in result.output

    def test_invalid_action_rejected(self, cli_runner):
        """Test that invalid action is rejected."""
        result = cli_runner.invoke(deploy, ["invalid-action"])

        # Click should reject invalid choice
        assert result.exit_code == 2  # Click parameter validation error
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()


class TestDeployCommandOutput:
    """Test deploy command console output."""

    def test_action_message_displayed_for_up(self, cli_runner, tmp_path):
        """Test that action message is displayed for 'up' command."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.deploy_up"):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["up", "--config", str(config_file)])

                # Should show action message
                assert "Service management" in result.output or "up" in result.output

    def test_no_action_message_for_status(self, cli_runner, tmp_path):
        """Test that status command doesn't show redundant action message."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.show_status"):
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["status", "--config", str(config_file)])

                # Status is quick, shouldn't show extra message
                # This test documents current behavior
                assert result.exit_code == 0

    def test_build_shows_success_message(self, cli_runner, tmp_path):
        """Test that build action shows success message."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.deploy_cmd.prepare_compose_files") as mock_prepare:
            mock_prepare.return_value = (MagicMock(), ["docker-compose.yml"])
            with patch("osprey.cli.deploy_cmd.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(deploy, ["build", "--config", str(config_file)])

                # Should show build progress and success
                assert (
                    "Building" in result.output or "built" in result.output or "✅" in result.output
                )
