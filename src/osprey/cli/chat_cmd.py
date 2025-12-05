"""Interactive chat command.

This module provides the 'osprey chat' command which wraps the existing
direct_conversation CLI interface. It preserves 100% of the original behavior
while providing a cleaner CLI interface.

IMPORTANT: This is a thin wrapper around osprey.interfaces.cli.direct_conversation.
All existing functionality is preserved without modification.
"""

import asyncio

import click

# Import centralized styles
from osprey.cli.styles import Styles, console

# Import existing CLI interface
from osprey.interfaces.cli.direct_conversation import run_cli


@click.command()
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="config.yml",
    help="Configuration file (default: config.yml in project directory)",
)
def chat(project: str, config: str):
    """Start interactive CLI conversation interface.

    Opens an interactive chat session with the agent. The interface
    provides command history, auto-suggestions, and real-time streaming
    of agent responses.

    This command wraps the existing direct_conversation interface,
    preserving all its functionality including:

    \b
      - Real-time status updates during agent processing
      - Approval workflow integration with interrupt handling
      - Rich console formatting with colors and styling
      - Session-based conversation continuity
      - Comprehensive error handling

    Commands within the chat:

    \b
      bye/end - Exit the chat
      Ctrl+L  - Clear screen
      Ctrl+C  - Exit

    Examples:

    \b
      # Start chat in current directory
      $ osprey chat

      # Start chat in specific project
      $ osprey chat --project ~/projects/my-agent

      # Use custom configuration
      $ osprey chat --config my-config.yml

      # Use environment variable
      $ export OSPREY_PROJECT=~/projects/my-agent
      $ osprey chat

    Note: Ensure services are running first (osprey deploy up)
    """
    from .project_utils import resolve_config_path

    console.print("Starting Osprey CLI interface...")
    console.print("   Press Ctrl+C to exit\n")

    try:
        # Resolve config path from project and config args
        config_path = resolve_config_path(project, config)

        # Set CONFIG_FILE environment variable for subprocess execution
        # This is critical for Python executor subprocess to initialize registry
        import os

        os.environ["CONFIG_FILE"] = str(config_path)

        # Call the existing run_cli function with config_path
        # This is the ORIGINAL function from Phase 1.5, behavior unchanged
        asyncio.run(run_cli(config_path=config_path))

    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style=Styles.WARNING)
        raise click.Abort()
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style=Styles.ERROR)
        # Show more details in verbose mode
        import os

        if os.environ.get("DEBUG"):
            import traceback

            console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort()


if __name__ == "__main__":
    chat()
