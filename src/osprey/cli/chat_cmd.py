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
@click.option(
    "--tui", is_flag=True, default=False, help="Launch Terminal User Interface (TUI) instead of CLI"
)
@click.option("--web", is_flag=True, default=False, help="Launch Web Debug UI instead of CLI")
def chat(project: str, config: str, tui: bool, web: bool):
    """Start interactive conversation interface (CLI, TUI, or Web).

    Opens an interactive chat session with the agent. The interface
    provides command history, auto-suggestions, and real-time streaming
    of agent responses.

    Use --tui flag to launch the Terminal User Interface (TUI) instead
    of the default CLI interface. The TUI provides a full-screen terminal
    experience with enhanced visual display.

    Use --web flag to launch the Web Debug UI, a browser-based interface
    for monitoring Osprey events in real-time with filtering and search.

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
      # Start CLI chat in current directory
      $ osprey chat

      # Start TUI chat
      $ osprey chat --tui

      # Start Web Debug UI
      $ osprey chat --web

      # Start chat in specific project
      $ osprey chat --project ~/projects/my-agent

      # Use custom configuration
      $ osprey chat --config my-config.yml

      # Use environment variable
      $ export OSPREY_PROJECT=~/projects/my-agent
      $ osprey chat

    Note: Ensure services are running first (osprey deploy up)
    Note: TUI requires textual package: pip install osprey-framework[tui]
    Note: Web requires fastapi/uvicorn: pip install osprey-framework[web]
    """
    from .project_utils import resolve_config_path

    try:
        # Resolve config path from project and config args
        config_path = resolve_config_path(project, config)

        # Set CONFIG_FILE environment variable for subprocess execution
        # This is critical for Python executor subprocess to initialize registry
        import os

        os.environ["CONFIG_FILE"] = str(config_path)

        if web:
            # Launch Web Debug UI
            console.print("Starting Osprey Web Debug UI...")
            console.print("   URL: http://127.0.0.1:8080")
            console.print("   Press Ctrl+C to stop\n")

            try:
                from osprey.interfaces.web import run_server

                run_server(config_path=config_path)
            except ImportError as e:
                console.print(
                    f"\n❌ Web dependencies not available: {e}\n"
                    "   Install with: pip install osprey-framework[web]",
                    style=Styles.ERROR,
                )
                raise click.Abort() from e
        elif tui:
            # Launch TUI interface
            console.print("Starting Osprey TUI interface...")
            console.print("   Press 'q' or Ctrl+C to exit\n")

            try:
                from osprey.interfaces.tui import run_tui

                asyncio.run(run_tui(config_path=config_path))
            except ImportError as e:
                console.print(
                    f"\n❌ TUI not available: {e}\n"
                    "   Install with: pip install osprey-framework[tui]",
                    style=Styles.ERROR,
                )
                raise click.Abort() from e
        else:
            # Launch CLI interface (original behavior)
            console.print("Starting Osprey CLI interface...")
            console.print("   Press Ctrl+C to exit\n")

            # Call the existing run_cli function with config_path
            # This is the ORIGINAL function from Phase 1.5, behavior unchanged
            asyncio.run(run_cli(config_path=config_path))

    except KeyboardInterrupt:
        console.print("\n\n👋 Goodbye!", style=Styles.WARNING)
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n❌ Error: {e}", style=Styles.ERROR)
        # Show more details in verbose mode
        import os

        if os.environ.get("DEBUG"):
            import traceback

            console.print(traceback.format_exc(), style=Styles.DIM)
        raise click.Abort() from None


if __name__ == "__main__":
    chat()
