"""Main CLI entry point for Osprey Framework.

This module provides the main CLI group that organizes all osprey
commands under the `osprey` command namespace.

Note: This will become 'osprey' in Phase 8 of the migration.

Performance Note: Uses lazy imports to avoid loading heavy dependencies
(langgraph, langchain, etc.) until a command is actually invoked.
This keeps `osprey --help` fast.
"""

import sys

import click

# Fix Windows console encoding to support Unicode characters (✓, ✗, ⚠️, etc.)
# This must be done before any output that uses Unicode characters
if sys.platform == "win32":
    try:
        # Reconfigure stdout and stderr to use UTF-8 encoding
        # This fixes the 'charmap' codec error on Windows when printing Unicode
        import io

        # Only reconfigure if not already UTF-8
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
            )
    except (AttributeError, OSError):
        # If reconfiguration fails (e.g., no buffer attribute), continue
        # The CLI should still work, just without fancy Unicode characters
        pass

# Import version from osprey package
try:
    from osprey import __version__
except ImportError:
    __version__ = "0.11.4"


# PERFORMANCE OPTIMIZATION: Lazy command loading
# Commands are imported only when invoked, not at module load time.
# This keeps --help fast and avoids loading heavy dependencies unnecessarily.


class LazyGroup(click.Group):
    """Click group that lazily loads subcommands only when invoked."""

    def get_command(self, ctx, cmd_name):
        """Lazily import and return the command when it's invoked."""
        # Map command names to their module paths
        commands = {
            "init": "osprey.cli.init_cmd",
            "deploy": "osprey.cli.deploy_cmd",
            "chat": "osprey.cli.chat_cmd",
            "config": "osprey.cli.config_cmd",
            "export-config": "osprey.cli.export_config_cmd",  # DEPRECATED: kept for backward compat
            "health": "osprey.cli.health_cmd",
            "generate": "osprey.cli.generate_cmd",
            "remove": "osprey.cli.remove_cmd",
            "migrate": "osprey.cli.migrate_cmd",
            "workflows": "osprey.cli.workflows_cmd",  # DEPRECATED: use 'tasks' instead
            "tasks": "osprey.cli.tasks_cmd",
            "claude": "osprey.cli.claude_cmd",
            "eject": "osprey.cli.eject_cmd",
            "channel-finder": "osprey.cli.channel_finder_cmd",
            "ariel": "osprey.cli.ariel",  # ARIEL search service
        }

        if cmd_name not in commands:
            return None

        # Lazy import - only loads when command is actually used
        import importlib

        mod = importlib.import_module(commands[cmd_name])

        # Get the command function from the module
        # Convention: module name without _cmd suffix
        if cmd_name == "config":
            cmd_func = mod.config
        elif cmd_name == "export-config":
            # DEPRECATED: Show warning and redirect to new command
            cmd_func = mod.export_config
        elif cmd_name == "channel-finder":
            cmd_func = mod.channel_finder
        elif cmd_name == "ariel":
            cmd_func = mod.ariel_group
        else:
            cmd_func = getattr(mod, cmd_name)

        return cmd_func

    def list_commands(self, ctx):
        """Return list of available commands (for --help)."""
        # Note: 'workflows' and 'assist' are deprecated but kept in commands dict for backward compat
        return [
            "init",
            "config",
            "deploy",
            "chat",
            "generate",
            "remove",
            "migrate",
            "health",
            "tasks",
            "channel-finder",
            "claude",
            "eject",
            "ariel",
        ]


@click.group(cls=LazyGroup, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="osprey")
@click.pass_context
def cli(ctx):
    """Osprey Framework CLI - Capability-Based Agentic Framework.

    A unified command-line interface for creating, deploying, and interacting
    with intelligent agents built on the Osprey Framework.

    Use 'osprey COMMAND --help' for more information on a specific command.

    Examples:

    \b
      osprey                          Launch interactive menu
      osprey init my-project          Create new project
      osprey config                   Manage configuration (show, export, set)
      osprey generate capability ...  Generate capability from MCP server
      osprey generate mcp-server      Generate demo MCP server
      osprey remove capability ...    Remove capability from project
      osprey deploy up                Start services
      osprey chat                     Interactive conversation
      osprey health                   Check system health
      osprey tasks                    Browse AI assistant tasks
      osprey claude install <task>    Install Claude Code skill
      osprey channel-finder           Interactive channel search
    """
    # Initialize theme from config if available (best-effort, silent failure)
    try:
        from .styles import initialize_theme_from_config

        initialize_theme_from_config()
    except Exception:
        # Silent failure - default theme will be used
        # CLI must work even if theme loading fails
        pass

    # NEW: If no command provided, launch interactive menu
    if ctx.invoked_subcommand is None:
        from .interactive_menu import launch_tui

        launch_tui()


def main():
    """Entry point for the osprey CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nGoodbye!", err=True)
        sys.exit(130)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
