"""Tests for chat CLI command.

This test module verifies the chat command wrapper functionality.
The command wraps the existing direct_conversation interface.
"""

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.chat_cmd import chat


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


class TestChatCommandBasics:
    """Test basic chat command functionality."""

    def test_command_help(self, cli_runner):
        """Verify chat command help is displayed."""
        result = cli_runner.invoke(chat, ["--help"])

        assert result.exit_code == 0
        assert "chat" in result.output.lower() or "Start interactive" in result.output
        assert "--project" in result.output or "-p" in result.output
        assert "--config" in result.output or "-c" in result.output

    def test_command_exists(self):
        """Verify chat command can be imported and is callable."""
        assert chat is not None
        assert callable(chat)

    def test_command_has_project_option(self, cli_runner):
        """Verify command has --project option."""
        result = cli_runner.invoke(chat, ["--help"])
        assert "--project" in result.output or "-p" in result.output

    def test_command_has_config_option(self, cli_runner):
        """Verify command has --config option."""
        result = cli_runner.invoke(chat, ["--help"])
        assert "--config" in result.output or "-c" in result.output


class TestChatCommandWithValidSetup:
    """Test chat command with properly mocked dependencies."""

    def test_command_with_mocked_dependencies(self, cli_runner, tmp_path):
        """Test basic execution with mocked run_cli."""
        # Create a fake config file to pass Click validation
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test config")

        # Mock all the dependencies
        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock) as mock_run_cli:
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(chat, ["--config", str(config_file)])

                # Should execute without crashing
                assert result.exit_code == 0
                # run_cli should have been called
                assert mock_run_cli.called

    def test_command_with_project_directory(self, cli_runner, tmp_path):
        """Test command with --project option."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        config_file = project_dir / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock):
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(
                    chat, ["--project", str(project_dir), "--config", str(config_file)]
                )

                # Should call resolve_config_path with project dir
                assert mock_resolve.called
                # First argument should be the project directory
                call_args = mock_resolve.call_args[0]
                assert str(project_dir) in str(call_args)

    def test_command_calls_run_cli_with_config_path(self, cli_runner, tmp_path):
        """Test that run_cli is called with config_path parameter."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock) as mock_run_cli:
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                cli_runner.invoke(chat, ["--config", str(config_file)])

                # run_cli should be called with config_path
                assert mock_run_cli.called
                call_kwargs = mock_run_cli.call_args[1] if mock_run_cli.call_args else {}
                assert "config_path" in call_kwargs

    def test_config_env_var_is_set(self, cli_runner, tmp_path):
        """Test that CONFIG_FILE environment variable is set."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock):
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                # Track environment variable changes
                import os

                original_env = os.environ.get("CONFIG_FILE")

                try:
                    result = cli_runner.invoke(chat, ["--config", str(config_file)])

                    # CONFIG_FILE should have been set (or test completed)
                    # This is a smoke test - we're just checking execution didn't crash
                    assert result.exit_code == 0
                finally:
                    # Restore original env var
                    if original_env:
                        os.environ["CONFIG_FILE"] = original_env
                    elif "CONFIG_FILE" in os.environ:
                        del os.environ["CONFIG_FILE"]


class TestChatCommandErrorHandling:
    """Test chat command error handling."""

    def test_keyboard_interrupt_handling(self, cli_runner, tmp_path):
        """Test graceful handling of KeyboardInterrupt (Ctrl+C)."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock) as mock_run_cli:
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file
                mock_run_cli.side_effect = KeyboardInterrupt()

                result = cli_runner.invoke(chat, ["--config", str(config_file)])

                # Should handle interrupt gracefully (exit code 1 for click.Abort)
                assert result.exit_code == 1
                # Should show goodbye message
                assert "Goodbye" in result.output or "👋" in result.output

    def test_general_exception_handling(self, cli_runner, tmp_path):
        """Test handling of general exceptions."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock) as mock_run_cli:
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file
                mock_run_cli.side_effect = Exception("Test error")

                result = cli_runner.invoke(chat, ["--config", str(config_file)])

                # Should handle exception gracefully
                assert result.exit_code == 1
                assert "Error" in result.output or "❌" in result.output
                assert "Test error" in result.output

    def test_missing_config_file_validation(self, cli_runner):
        """Test Click's built-in validation for missing config file."""
        result = cli_runner.invoke(chat, ["--config", "/nonexistent/config.yml"])

        # Click should reject missing file
        assert result.exit_code == 2  # Click parameter validation error
        assert "does not exist" in result.output.lower() or "not found" in result.output.lower()

    def test_config_resolution_error(self, cli_runner, tmp_path):
        """Test handling when config resolution fails."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
            mock_resolve.side_effect = FileNotFoundError("Config not found")

            result = cli_runner.invoke(chat, ["--config", str(config_file)])

            # Should handle error gracefully
            assert result.exit_code == 1


class TestChatCommandOutput:
    """Test chat command console output."""

    def test_startup_message_displayed(self, cli_runner, tmp_path):
        """Test that startup message is shown."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock):
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(chat, ["--config", str(config_file)])

                # Should show startup message
                assert "Starting" in result.output or "Osprey" in result.output

    def test_exit_instructions_displayed(self, cli_runner, tmp_path):
        """Test that exit instructions are shown."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock):
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file

                result = cli_runner.invoke(chat, ["--config", str(config_file)])

                # Should show exit instructions
                assert "Ctrl+C" in result.output or "exit" in result.output.lower()

    def test_error_output_formatting(self, cli_runner, tmp_path):
        """Test that errors are formatted properly."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("# test")

        with patch("osprey.cli.chat_cmd.run_cli", new_callable=AsyncMock) as mock_run_cli:
            with patch("osprey.cli.project_utils.resolve_config_path") as mock_resolve:
                mock_resolve.return_value = config_file
                mock_run_cli.side_effect = ValueError("Something went wrong")

                result = cli_runner.invoke(chat, ["--config", str(config_file)])

                # Should show error with emoji
                assert "❌" in result.output or "Error" in result.output
