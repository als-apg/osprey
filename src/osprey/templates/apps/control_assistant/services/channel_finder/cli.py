#!/usr/bin/env python
"""
CLI Interface for Channel Finder

Command-line interface with history, auto-suggestions, and rich output.
Supports both interactive mode and direct query execution.
"""

import asyncio
import sys
import os
import logging
import argparse
from pathlib import Path

# Imports
from .service import ChannelFinderService
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Use Osprey's config system
from osprey.utils.config import _get_config

# Modern CLI dependencies
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.styles import Style


def _ensure_config_available():
    """
    Ensure Osprey config is available by finding project root.

    This allows the CLI to be run from any directory within the project.
    """
    if os.getenv('CONFIG_FILE'):
        # Already configured
        return

    # Search up the directory tree for config.yml
    current = Path.cwd()

    # Try current directory and parents
    for path in [current] + list(current.parents):
        config_file = path / 'config.yml'
        if config_file.exists():
            os.environ['CONFIG_FILE'] = str(config_file)
            return

    # If not found, assume we're in src/ and go up one level
    project_root = Path(__file__).parent.parent.parent.parent.parent
    config_file = project_root / 'config.yml'
    if config_file.exists():
        os.environ['CONFIG_FILE'] = str(config_file)
        return


def setup_logging(verbose: bool = False):
    """Configure logging with Rich handler for beautiful console output.

    Args:
        verbose: If True, set to DEBUG level. Otherwise use config or INFO.
    """
    _ensure_config_available()
    config_builder = _get_config()

    # Get log level from config or environment
    if verbose or os.getenv("DEBUG"):
        log_level = logging.DEBUG
    else:
        log_level_str = config_builder.get('logging.level', 'INFO')
        log_level = getattr(logging, log_level_str, logging.INFO)

    # Configure Rich handler with clean formatting
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        show_time=False,
        show_level=False,  # Hide log level (INFO, DEBUG, etc.)
        markup=True,
        omit_repeated_times=False
    )

    # Configure root logger with just the message (no logger name, level, etc.)
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[rich_handler],
        force=True  # Override any existing configuration
    )

    # Set level for our application loggers
    for logger_name in ["channel_finder", "channel_finder.pipeline", "channel_finder.service", "LM"]:
        logging.getLogger(logger_name).setLevel(log_level)
        logging.getLogger(logger_name).propagate = True

    # Silence noisy third-party libraries
    for noisy_logger in [
        "httpx", "httpcore", "openai", "anthropic",
        "urllib3", "requests", "google", "ollama",
        "asyncio", "aiofiles"
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


class ChannelFinderCLI:
    """Command Line Interface for Channel Finder.

    Provides an interactive terminal interface with:
    - Persistent command history
    - Auto-suggestions from previous queries
    - Rich formatted output
    - Key bindings (Ctrl+L to clear, arrow keys for history)
    - Clean, focused user experience
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
        self.prompt_style = Style.from_dict({
            'prompt': '#00aa00 bold',
            'suggestion': '#666666 italic',
        })

    def _create_key_bindings(self):
        """Create custom key bindings for CLI shortcuts."""
        bindings = KeyBindings()

        @bindings.add('c-l')  # Ctrl+L to clear screen
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
            complete_style='multi-column',
            enable_suspend=True,
            reserve_space_for_menu=0
        )

    async def initialize(self):
        """Initialize the channel finder service and CLI components."""
        # Ensure config is available
        _ensure_config_available()

        # Set up logging first
        setup_logging(verbose=True)

        # Get facility name from config
        config_builder = _get_config()
        facility_name = config_builder.get('facility.name', 'Channel Finder')

        self.console.print(f"\n🔧 {facility_name}", style="bold cyan")
        self.console.print("="*60)

        try:
            self.service = ChannelFinderService()
            self.console.print("✓ Service initialized", style="green")

            # Get pipeline info and statistics
            pipeline_info = self.service.get_pipeline_info()
            stats = pipeline_info.get('statistics', {})
            total_channels = stats.get('total_channels', 0)

            self.console.print(f"✓ Pipeline: {pipeline_info.get('pipeline_mode', 'unknown')} mode", style="green")
            self.console.print(f"✓ Database: {total_channels} channels loaded", style="green")
            self.console.print(f"✓ Model: {self.service.model_config.get('model_id', 'unknown')}", style="green")
            self.console.print(f"✓ Logging: {logging.getLogger().getEffectiveLevel()} level", style="green")

            # Initialize prompt session
            self.prompt_session = self._create_prompt_session()

            self.console.print("\n💡 Tips:", style="yellow")
            self.console.print("  • Use ↑/↓ arrows to navigate command history", style="dim")
            self.console.print("  • Press Ctrl+L to clear screen", style="dim")
            self.console.print("  • Type 'exit' or 'quit' to exit, or press Ctrl+C", style="dim")
            self.console.print()

        except Exception as e:
            self.console.print(f"✗ Initialization failed: {e}", style="red")
            self.console.print("\nCheck your configuration and API keys to diagnose issues.", style="yellow")
            sys.exit(1)

    async def run(self):
        """Execute the main CLI interaction loop."""
        await self.initialize()

        while True:
            try:
                # Get user input with rich prompt and history
                user_input = await self.prompt_session.prompt_async(
                    HTML('<prompt>🔍 Query: </prompt>'),
                    style=self.prompt_style
                )
                user_input = user_input.strip()

                # Exit conditions
                if user_input.lower() in ["exit", "quit", "bye", "end"]:
                    self.console.print("\n👋 Goodbye!", style="yellow")
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Process the query
                await self._process_query(user_input)

            except KeyboardInterrupt:
                self.console.print("\n👋 Goodbye!", style="yellow")
                break
            except EOFError:
                self.console.print("\n👋 Goodbye!", style="yellow")
                break
            except Exception as e:
                self.console.print(f"❌ Error: {e}", style="red")
                continue

    async def _process_query(self, query: str):
        """Process a channel finder query and display results."""
        try:
            # Show processing indicator
            self.console.print(f"\n⏳ Processing: \"{query}\"", style="blue")

            # Execute query
            result = await self.service.find_channels(query)

            # Display results
            if result.total_channels > 0:
                self._display_results(result)
            else:
                self.console.print("\n✗ No channels found", style="yellow")
                self.console.print("Tip: Try rephrasing or using different terms", style="dim")

            self.console.print()

        except Exception as e:
            self.console.print(f"\n✗ Query failed: {e}", style="red")
            self.console.print("Please check your API key and configuration.", style="dim")
            self.console.print()

    def _display_results(self, result):
        """Display query results in a formatted table."""
        self.console.print(f"\n✓ Found {result.total_channels} channel(s)", style="green")

        # Create a rich table
        table = Table(show_header=True, header_style="bold cyan", border_style="dim")
        table.add_column("Channel", style="cyan", no_wrap=True)
        table.add_column("Address", style="yellow")
        table.add_column("Description", style="white")

        for ch in result.channels:
            # Truncate long descriptions
            desc = ch.description or ""
            if len(desc) > 80:
                desc = desc[:77] + "..."

            table.add_row(ch.channel, ch.address, desc)

        self.console.print(table)

        # Show processing notes
        if result.processing_notes:
            self.console.print(f"\nℹ️  {result.processing_notes}", style="dim")


async def direct_query(query: str, verbose: bool = False):
    """Execute a direct query without interactive mode.

    Args:
        query: The query string to execute
        verbose: Enable verbose logging
    """
    _ensure_config_available()
    setup_logging(verbose=verbose)
    console = Console()

    try:
        # Initialize service
        service = ChannelFinderService()

        # Execute query
        console.print(f"\n[cyan]Query:[/cyan] {query}")
        result = await service.find_channels(query)

        # Display results
        if result.total_channels == 0:
            console.print("\n[yellow]No channels found[/yellow]")
            return 1

        console.print(f"\n[green]Found {result.total_channels} channel{'s' if result.total_channels != 1 else ''}[/green]\n")

        # Create results table
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


async def run_benchmark(args):
    """Run benchmarks using core benchmark system."""
    from .benchmarks import cli as benchmark_cli

    try:
        # Build args for benchmark CLI
        import sys
        benchmark_args = []
        if hasattr(args, 'queries') and args.queries:
            benchmark_args.extend(['--queries', args.queries])
        if hasattr(args, 'model') and args.model:
            benchmark_args.extend(['--model', args.model])

        # Override sys.argv for the benchmark CLI
        original_argv = sys.argv
        sys.argv = ['benchmark'] + benchmark_args

        # Parse args and run the async function directly (we're already in async context)
        parsed_args = benchmark_cli.parse_args()
        await benchmark_cli.main_async(parsed_args)

        # Restore sys.argv
        sys.argv = original_argv
        return 0

    except Exception as e:
        print(f"❌ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1


async def async_main():
    """Async entry point for the CLI application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Channel Finder - Natural language channel search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Benchmark subcommand
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmarks
  channel-finder benchmark

  # Use different model
  channel-finder benchmark --model anthropic/claude-sonnet

  # Run specific queries
  channel-finder benchmark --queries 0:10
        """
    )
    benchmark_parser.add_argument(
        '--queries',
        type=str,
        help='Query selection (e.g., "0:10" or "0,5,10")'
    )
    benchmark_parser.add_argument(
        '--model',
        type=str,
        help='Override model configuration'
    )

    # Global arguments (for query mode)
    parser.add_argument(
        'query',
        nargs='?',
        help='Query to execute (if no command specified, enters query/interactive mode)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Handle benchmark command
    if args.command == 'benchmark':
        exit_code = await run_benchmark(args)
        sys.exit(exit_code)

    # Direct query mode
    if args.query:
        exit_code = await direct_query(args.query, verbose=args.verbose)
        sys.exit(exit_code)

    # Interactive mode
    cli = ChannelFinderCLI()
    await cli.run()


def main():
    """Sync wrapper for CLI entry point (for console_scripts)."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

