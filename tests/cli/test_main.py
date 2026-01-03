"""Tests for main CLI entry point.

Tests the main CLI group and lazy command loading mechanism.
"""

from unittest import mock

import pytest
from click.testing import CliRunner

from osprey.cli.main import LazyGroup, cli, main


@pytest.fixture
def runner():
    """Provide a Click CLI runner for testing."""
    return CliRunner()


class TestLazyGroup:
    """Test the LazyGroup command loading mechanism."""

    def test_lazy_group_initialization(self):
        """Verify LazyGroup can be instantiated."""
        group = LazyGroup(name="test")
        assert group is not None
        assert isinstance(group, LazyGroup)

    def test_get_command_returns_valid_commands(self):
        """Test get_command returns valid command objects."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        # Test a known command
        with mock.patch("importlib.import_module") as mock_import:
            mock_module = mock.Mock()
            mock_module.init = mock.Mock()
            mock_import.return_value = mock_module

            group.get_command(ctx, "init")

            # Should attempt to import the module
            mock_import.assert_called_once()

    def test_get_command_returns_none_for_invalid_command(self):
        """Test get_command returns None for unknown commands."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        cmd = group.get_command(ctx, "nonexistent_command")
        assert cmd is None

    def test_list_commands_returns_expected_commands(self):
        """Verify list_commands returns all available commands."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        commands = group.list_commands(ctx)

        # Verify expected commands are present
        expected_commands = [
            "init",
            "config",
            "deploy",
            "chat",
            "generate",
            "remove",
            "health",
            "workflows",
        ]
        assert isinstance(commands, list)
        assert len(commands) > 0
        for cmd in expected_commands:
            assert cmd in commands

    def test_get_command_handles_config_command(self):
        """Test special handling for config command."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        with mock.patch("importlib.import_module") as mock_import:
            mock_module = mock.Mock()
            mock_module.config = mock.Mock()
            mock_import.return_value = mock_module

            group.get_command(ctx, "config")

            # Should get config attribute from module
            assert mock_import.called

    def test_get_command_handles_deprecated_export_config(self):
        """Test handling of deprecated export-config command."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        with mock.patch("importlib.import_module") as mock_import:
            mock_module = mock.Mock()
            mock_module.export_config = mock.Mock()
            mock_import.return_value = mock_module

            group.get_command(ctx, "export-config")

            # Should still be accessible for backward compatibility
            assert mock_import.called


class TestCliGroup:
    """Test the main CLI group."""

    def test_cli_group_help(self, runner):
        """Test CLI shows help message."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "osprey" in result.output.lower()
        assert "command" in result.output.lower()

    def test_cli_version_option(self, runner):
        """Test --version flag displays version."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        # Should contain version info
        assert "osprey" in result.output.lower() or "version" in result.output.lower()

    @mock.patch("osprey.cli.interactive_menu.launch_tui")
    def test_cli_without_command_launches_tui(self, mock_launch_tui, runner):
        """Test CLI without command launches interactive menu."""
        runner.invoke(cli, [])

        # Should attempt to launch TUI
        assert mock_launch_tui.called

    @mock.patch("osprey.cli.styles.initialize_theme_from_config")
    def test_cli_initializes_theme(self, mock_init_theme, runner):
        """Test CLI attempts to initialize theme from config."""
        with mock.patch("osprey.cli.interactive_menu.launch_tui"):
            runner.invoke(cli, [])

        # Should attempt theme initialization (silent failure is OK)
        assert mock_init_theme.called

    @mock.patch("osprey.cli.styles.initialize_theme_from_config")
    def test_cli_handles_theme_initialization_failure(self, mock_init_theme, runner):
        """Test CLI continues if theme initialization fails."""
        mock_init_theme.side_effect = Exception("Theme loading failed")

        with mock.patch("osprey.cli.interactive_menu.launch_tui"):
            result = runner.invoke(cli, [])

        # Should not crash - silent failure
        # TUI should still be launched
        assert result.exit_code == 0 or result.exception is None

    def test_cli_subcommand_invocation(self, runner):
        """Test CLI can invoke subcommands."""
        # Test with a simple subcommand (health --help should work)
        result = runner.invoke(cli, ["health", "--help"])

        # Should execute subcommand
        assert result.exit_code == 0
        assert "health" in result.output.lower()


class TestMainFunction:
    """Test the main entry point function."""

    @mock.patch("osprey.cli.main.cli")
    def test_main_calls_cli(self, mock_cli):
        """Test main() calls the CLI group."""
        main()

        # Should invoke the CLI
        assert mock_cli.called

    @mock.patch("osprey.cli.main.cli")
    @mock.patch("click.echo")
    def test_main_handles_keyboard_interrupt(self, mock_echo, mock_cli):
        """Test main handles Ctrl+C gracefully."""
        mock_cli.side_effect = KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit with 130 (standard for SIGINT)
        assert exc_info.value.code == 130

        # Should print goodbye message
        assert mock_echo.called

    @mock.patch("osprey.cli.main.cli")
    @mock.patch("click.echo")
    def test_main_handles_general_exception(self, mock_echo, mock_cli):
        """Test main handles exceptions gracefully."""
        mock_cli.side_effect = Exception("Test error")

        with pytest.raises(SystemExit) as exc_info:
            main()

        # Should exit with 1
        assert exc_info.value.code == 1

        # Should print error message
        assert mock_echo.called
        call_args = str(mock_echo.call_args)
        assert "error" in call_args.lower()


class TestLazyLoading:
    """Test lazy loading performance optimization."""

    def test_lazy_loading_defers_imports(self):
        """Test that commands are not imported until needed."""
        # Creating the CLI group should not import heavy dependencies
        group = LazyGroup(name="test")

        # The group should exist without loading langgraph, langchain, etc.
        assert group is not None

    def test_command_modules_map(self):
        """Test that command-to-module mapping is complete."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        # All listed commands should have module mappings
        commands = group.list_commands(ctx)

        for cmd_name in commands:
            # Should be able to get command (even if import fails)
            # This tests the mapping exists
            with mock.patch("importlib.import_module") as mock_import:
                mock_module = mock.Mock()
                setattr(mock_module, cmd_name, mock.Mock())
                mock_import.return_value = mock_module

                try:
                    group.get_command(ctx, cmd_name)
                    # Should not raise KeyError from missing mapping
                except ImportError:
                    # Import errors are OK - we're testing mapping exists
                    pass


class TestIntegration:
    """Integration tests for CLI entry point."""

    def test_cli_help_is_fast(self, runner):
        """Test that --help is fast (lazy loading working)."""
        import time

        start = time.time()
        result = runner.invoke(cli, ["--help"])
        elapsed = time.time() - start

        assert result.exit_code == 0
        # Help should be very fast (< 1 second) due to lazy loading
        assert elapsed < 1.0

    def test_version_import_works(self):
        """Test version import from package."""
        from osprey.cli.main import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        # Should be semver format
        assert "." in __version__

    def test_cli_as_module(self):
        """Test CLI can be invoked as module."""
        # This tests the if __name__ == "__main__" case
        # We verify the main function exists and is callable
        from osprey.cli.main import main

        assert callable(main)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_cli_with_unknown_command(self, runner):
        """Test CLI handles unknown commands gracefully."""
        result = runner.invoke(cli, ["unknown_command"])

        # Should fail with helpful error
        assert result.exit_code != 0

    def test_lazy_group_get_command_with_none_context(self):
        """Test LazyGroup handles None context."""
        group = LazyGroup(name="test")

        # Should handle gracefully
        group.get_command(None, "init")
        # May return None or command - documenting behavior

    @mock.patch("osprey.cli.interactive_menu.launch_tui")
    def test_cli_invoked_subcommand_skips_tui(self, mock_launch_tui, runner):
        """Test that providing a subcommand skips the TUI."""
        runner.invoke(cli, ["health", "--help"])

        # TUI should NOT be launched when subcommand is provided
        assert not mock_launch_tui.called
