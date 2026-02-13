"""
CLI Interface for Channel Finder

Interactive REPL with history, auto-suggestions, and rich output.
Supports both interactive mode (via ChannelFinderCLI) and direct
single-query execution (via direct_query).

These are imported and used by osprey.cli.channel_finder_cmd.
"""

import logging
import os

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table

from osprey.services.channel_finder.service import ChannelFinderService


class ChannelFinderCLI:
    """Command Line Interface for Channel Finder.

    Provides an interactive terminal interface with:
    - Persistent command history
    - Auto-suggestions from previous queries
    - Rich formatted output
    - Key bindings (Ctrl+L to clear, arrow keys for history)
    - Clean, focused user experience

    Can be used standalone (calls initialize() internally) or with
    pre-initialized state when launched from the osprey CLI.
    """

    def __init__(self):
        """Initialize the CLI interface."""
        self.service = None
        self.console = Console()

        # Modern CLI components
        self.prompt_session = None
        self.history_file = os.path.expanduser("~/.channel_finder_history")

        # Create custom key bindings
        self.key_bindings = self._create_key_bindings()

        # Create custom style
        self.prompt_style = Style.from_dict(
            {
                "prompt": "#00aa00 bold",
                "suggestion": "#666666 italic",
            }
        )

    def _create_key_bindings(self):
        """Create custom key bindings for CLI shortcuts."""
        bindings = KeyBindings()

        @bindings.add("c-l")  # Ctrl+L to clear screen
        def _(event):
            """Clear the screen."""
            clear()

        return bindings

    def _create_prompt_session(self):
        """Create a prompt_toolkit session with history and auto-suggestions."""
        return PromptSession(
            history=FileHistory(self.history_file),
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=self.key_bindings,
            style=self.prompt_style,
            mouse_support=False,
            complete_style="multi-column",
            enable_suspend=True,
            reserve_space_for_menu=0,
        )

    async def initialize(self):
        """Initialize the channel finder service and CLI components.

        The registry and config should already be set up before calling
        this method (handled by the osprey CLI wrapper).
        """
        from osprey.utils.config import get_config_builder

        config_builder = get_config_builder()
        facility_name = config_builder.get("facility.name", "Channel Finder")

        self.console.print(f"\n{facility_name}", style="bold cyan")
        self.console.print("=" * 60)

        try:
            self.service = ChannelFinderService()
            self.console.print("Service initialized", style="green")

            # Get pipeline info and statistics
            pipeline_info = self.service.get_pipeline_info()
            stats = pipeline_info.get("statistics", {})
            total_channels = stats.get("total_channels", 0)

            self.console.print(
                f"Pipeline: {pipeline_info.get('pipeline_mode', 'unknown')} mode", style="green"
            )
            self.console.print(f"Database: {total_channels} channels loaded", style="green")
            self.console.print(
                f"Model: {self.service.model_config.get('model_id', 'unknown')}", style="green"
            )
            self.console.print(
                f"Logging: {logging.getLogger().getEffectiveLevel()} level", style="green"
            )

            # Initialize prompt session
            self.prompt_session = self._create_prompt_session()

            self.console.print("\nTips:", style="yellow")
            self.console.print("  Use up/down arrows to navigate command history", style="dim")
            self.console.print("  Press Ctrl+L to clear screen", style="dim")
            self.console.print("  Type 'exit' or 'quit' to exit, or press Ctrl+C", style="dim")
            self.console.print()

        except Exception as e:
            self.console.print(f"Initialization failed: {e}", style="red")
            self.console.print(
                "\nCheck your configuration and API keys to diagnose issues.", style="yellow"
            )
            raise

    async def run(self):
        """Execute the main CLI interaction loop."""
        await self.initialize()

        while True:
            try:
                # Get user input with rich prompt and history
                user_input = await self.prompt_session.prompt_async(
                    HTML("<prompt>Query: </prompt>"), style=self.prompt_style
                )
                user_input = user_input.strip()

                # Exit conditions
                if user_input.lower() in ["exit", "quit", "bye", "end"]:
                    self.console.print("\nGoodbye!", style="yellow")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Process the query
                await self._process_query(user_input)

            except KeyboardInterrupt:
                self.console.print("\nGoodbye!", style="yellow")
                break
            except EOFError:
                self.console.print("\nGoodbye!", style="yellow")
                break
            except Exception as e:
                self.console.print(f"Error: {e}", style="red")
                continue

    async def _process_query(self, query: str):
        """Process a channel finder query and display results."""
        try:
            self.console.print(f'\nProcessing: "{query}"', style="blue")

            result = await self.service.find_channels(query)

            if result.total_channels > 0:
                self._display_results(result)
            else:
                self.console.print("\nNo channels found", style="yellow")
                self.console.print("Tip: Try rephrasing or using different terms", style="dim")

            self.console.print()

        except Exception as e:
            self.console.print(f"\nQuery failed: {e}", style="red")
            self.console.print("Please check your API key and configuration.", style="dim")
            self.console.print()

    def _display_results(self, result):
        """Display query results in a formatted table."""
        self.console.print(f"\nFound {result.total_channels} channel(s)", style="green")

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Channel", style="cyan", no_wrap=True)
        table.add_column("Address", style="yellow")
        table.add_column("Description", style="white")

        for ch in result.channels:
            desc = ch.description or ""
            if len(desc) > 80:
                desc = desc[:77] + "..."

            table.add_row(ch.channel, ch.address, desc)

        self.console.print(table)

        if result.processing_notes:
            self.console.print(f"\n{result.processing_notes}", style="dim")


async def direct_query(query: str, verbose: bool = False):
    """Execute a direct query without interactive mode.

    The registry and config should already be set up before calling
    this function (handled by the osprey CLI wrapper).

    Args:
        query: The query string to execute
        verbose: Enable verbose logging

    Returns:
        Exit code (0 for success, 1 for error/no results, 130 for cancelled)
    """
    console = Console()

    try:
        service = ChannelFinderService()

        console.print(f"\n[cyan]Query:[/cyan] {query}")
        result = await service.find_channels(query)

        if result.total_channels == 0:
            console.print("\n[yellow]No channels found[/yellow]")
            return 1

        console.print(
            f"\n[green]Found {result.total_channels} channel{'s' if result.total_channels != 1 else ''}[/green]\n"
        )

        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Channel", style="cyan", no_wrap=True)
        table.add_column("Address", style="white")
        table.add_column("Description", style="dim")

        for ch in result.channels:
            table.add_row(ch.channel, ch.address, ch.description or "")

        console.print(table)
        console.print()
        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Query cancelled[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        return 1
