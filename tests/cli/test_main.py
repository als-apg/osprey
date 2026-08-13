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
            mock_module.build = mock.Mock()
            mock_import.return_value = mock_module

            group.get_command(ctx, "build")

            # Should attempt to import the module
            mock_import.assert_called_once()

    def test_get_command_returns_none_for_invalid_command(self):
        """Test get_command returns None for unknown commands."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        cmd = group.get_command(ctx, "nonexistent_command")
        assert cmd is None

    def test_list_commands_offers_the_whole_lifecycle(self):
        """The listing is the surface an operator navigates by.

        A verb missing from it is unreachable from `osprey --help` even when it
        resolves, so the lifecycle sequence is asserted as a whole rather than
        sampled — it is the order the help screen teaches.
        """
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        commands = group.list_commands(ctx)

        lifecycle = [
            "init",
            "build",
            "up",
            "down",
            "restart",
            "status",
            "logs",
            "reset",
            "set",
            "validate",
            "chat",
            "config",
            "users",
        ]
        assert isinstance(commands, list)
        for cmd in lifecycle:
            assert cmd in commands

    def test_the_retired_groups_are_gone_from_the_listing(self):
        """A listing that still offered them would advertise commands that do
        not resolve — the failure mode the lazy registry makes easy, since a
        name in the list and a name in the import map are two separate edits."""
        group = LazyGroup(name="test")

        commands = group.list_commands(mock.Mock())

        for retired in ("deploy", "claude"):
            assert retired not in commands

    def test_every_listed_command_actually_resolves(self):
        """The list and the import map have to agree.

        They are two separate literals in the same class, so a verb added to
        one and forgotten in the other produces a help screen naming a command
        that cannot be run.
        """
        import click

        from osprey.cli.main import cli

        ctx = click.Context(cli)
        for name in cli.list_commands(ctx):
            assert cli.get_command(ctx, name) is not None, name

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

    def test_get_command_handles_removed_export_config(self):
        """Test that fully removed export-config command returns None."""
        group = LazyGroup(name="test")
        ctx = mock.Mock()

        cmd = group.get_command(ctx, "export-config")

        # export-config has been fully removed, should return None
        assert cmd is None


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

    def test_cli_without_command_prints_help(self, runner):
        """Bare `osprey` lists what there is to run.

        It used to launch an interactive menu, which existed because the legacy
        verbs took arguments nobody could remember. Every surviving verb is
        zero-argument, so the help IS the menu — and printing it must exit
        cleanly rather than read as a usage error.
        """
        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        assert "Usage: cli" in result.output
        assert "Commands:" in result.output

    @mock.patch("osprey.cli.styles.initialize_theme_from_config")
    def test_cli_initializes_theme(self, mock_init_theme, runner):
        """Test CLI attempts to initialize theme from config."""
        runner.invoke(cli, [])

        # Should attempt theme initialization (silent failure is OK)
        assert mock_init_theme.called

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

    def test_a_subcommand_gets_its_own_help_not_the_groups(self, runner):
        """The group prints its help only when no subcommand was named.

        Otherwise `osprey health --help` would answer with the group's listing
        instead of the verb's own options — the same branch the bare-`osprey`
        case above exercises, read from the other side.
        """
        result = runner.invoke(cli, ["health", "--help"])

        assert result.exit_code == 0
        assert "Usage: cli health" in result.output
