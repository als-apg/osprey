"""Main CLI entry point for Osprey Framework.

This module provides the main CLI group that organizes all osprey
commands under the `osprey` command namespace.
"""

import sys

import click

# Ensure UTF-8 on Windows for Unicode CLI output
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

try:
    from osprey import __version__
except ImportError:
    __version__ = "0.11.5"


class LazyGroup(click.Group):
    """Click group that lazily loads subcommands only when invoked."""

    def get_command(self, ctx, cmd_name):
        """Lazily import and return the command when it's invoked."""
        # Map command names to their module paths
        commands = {
            "init": "osprey.cli.init_cmd",
            "build": "osprey.cli.build_cmd",
            "deploy": "osprey.cli.deploy_cmd",
            "config": "osprey.cli.config_cmd",
            "health": "osprey.cli.health_cmd",
            "claude": "osprey.cli.claude_cmd",
            "eject": "osprey.cli.eject_cmd",
            "channel-finder": "osprey.cli.channel_finder_cmd",
            "ariel": "osprey.cli.ariel",  # ARIEL search service
            "artifacts": "osprey.cli.artifacts_cmd",  # Artifact Gallery
            "web": "osprey.cli.web_cmd",  # Web Terminal
            "prompts": "osprey.cli.prompts_cmd",  # Prompt artifact overrides
            "audit": "osprey.cli.audit_cmd",  # Safety auditor
            "skills": "osprey.cli.skills_cmd",  # Bundled skill management
            "vendor": "osprey.cli.vendor_cmd",  # Vendor asset management
        }

        if cmd_name not in commands:
            return None

        import importlib

        mod = importlib.import_module(commands[cmd_name])

        if cmd_name == "config":
            cmd_func = mod.config
        elif cmd_name == "channel-finder":
            cmd_func = mod.channel_finder
        elif cmd_name == "ariel":
            cmd_func = mod.ariel_group
        elif cmd_name == "artifacts":
            cmd_func = mod.artifacts
        elif cmd_name == "web":
            cmd_func = mod.web
        elif cmd_name == "prompts":
            cmd_func = mod.prompts
        else:
            cmd_func = getattr(mod, cmd_name)

        return cmd_func

    def list_commands(self, ctx):
        """Return list of available commands (for --help)."""
        return [
            "init",
            "build",
            "config",
            "deploy",
            "health",
            "channel-finder",
            "claude",
            "eject",
            "ariel",
            "artifacts",
            "web",
            "prompts",
            "audit",
            "skills",
            "vendor",
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
      osprey deploy up                Start services
      osprey claude regen             Regenerate Claude Code artifacts
      osprey web                      Launch web terminal
      osprey health                   Check system health
      osprey channel-finder           Interactive channel search
    """
    from .styles import initialize_theme_from_config

    initialize_theme_from_config()

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
