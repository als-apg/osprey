"""
Benchmark CLI for Channel Finder

Core benchmark functions for evaluating channel finder performance.
These are imported and used by osprey.cli.channel_finder_cmd.

Functions:
    parse_query_selection: Parse query selection strings
    create_config_override: Apply CLI overrides to config
    run_benchmarks: Execute the benchmark suite
"""

import logging
from datetime import datetime

from rich.panel import Panel
from rich.table import Table

from osprey.cli.styles import Messages, Styles, console
from osprey.services.channel_finder.benchmarks.runner import BenchmarkRunner
from osprey.services.channel_finder.utils.config import get_config


def parse_query_selection(query_arg: str):
    """Parse query selection argument.

    Args:
        query_arg: Query selection string
            - "all": All queries
            - "0:10": Range from 0 to 10
            - "0,5,10": Specific indices

    Returns:
        Query selection in format expected by config
    """
    if query_arg.lower() == "all":
        return "all"

    # Check for range (e.g., "0:10")
    if ":" in query_arg:
        parts = query_arg.split(":")
        if len(parts) == 2:
            try:
                start = int(parts[0])
                end = int(parts[1])
                return {"start": start, "end": end}
            except ValueError as e:
                raise ValueError(
                    f"Invalid range format: {query_arg}. Expected format: 'start:end'"
                ) from e

    # Check for comma-separated indices (e.g., "0,5,10")
    if "," in query_arg:
        try:
            indices = [int(x.strip()) for x in query_arg.split(",")]
            return indices
        except ValueError as e:
            raise ValueError(f"Invalid index list: {query_arg}. Expected format: '0,5,10'") from e

    # Single index
    try:
        return [int(query_arg)]
    except ValueError as e:
        raise ValueError(f"Invalid query selection: {query_arg}") from e


def create_config_override(queries: str = None, model: str = None, dataset: str = None):
    """Apply CLI argument overrides to config."""
    config = get_config()

    # Override query selection
    if queries is not None:
        if "benchmark" not in config["channel_finder"]:
            config["channel_finder"]["benchmark"] = {}
        if "execution" not in config["channel_finder"]["benchmark"]:
            config["channel_finder"]["benchmark"]["execution"] = {}
        config["channel_finder"]["benchmark"]["execution"]["query_selection"] = (
            parse_query_selection(queries)
        )

    # Override model settings
    if model is not None:
        if "model" not in config:
            config["model"] = {}
        config["model"]["model_id"] = model

    # Override benchmark dataset
    if dataset is not None:
        # Override the dataset_path in the active pipeline's benchmark config
        pipeline_mode = config["channel_finder"]["pipeline_mode"]
        if (
            "pipelines" in config["channel_finder"]
            and pipeline_mode in config["channel_finder"]["pipelines"]
        ):
            if "benchmark" not in config["channel_finder"]["pipelines"][pipeline_mode]:
                config["channel_finder"]["pipelines"][pipeline_mode]["benchmark"] = {}
            config["channel_finder"]["pipelines"][pipeline_mode]["benchmark"]["dataset_path"] = (
                dataset
            )


async def run_benchmarks(
    dataset: str = None, queries: str = None, model: str = None, verbose: bool = False
):
    """
    Run channel finder benchmarks.

    Args:
        dataset: Path to custom benchmark dataset (None = use pipeline default)
        queries: Query selection (e.g., "0:10" or "0,5,10")
        model: Override model configuration
        verbose: Show detailed channel finder logs
    """
    # Configure logging level based on verbose flag
    # Suppress INFO logs from channel finder by default
    if not verbose:
        logging.getLogger("osprey").setLevel(logging.WARNING)
        logging.getLogger("channel_finder").setLevel(logging.WARNING)

    # Print banner
    console.print()
    console.print(
        Panel(
            "[bold]🎯 Channel Finder Benchmarking[/bold]\n[dim]Osprey Control Assistant[/dim]",
            style=Styles.HEADER,
            border_style=Styles.BORDER_ACCENT,
            padding=(1, 2),
        )
    )
    console.print()

    # Apply config overrides from CLI arguments
    try:
        create_config_override(queries=queries, model=model, dataset=dataset)
        console.print(f"  {Messages.success('Configuration loaded')}")
    except Exception as e:
        console.print(f"  {Messages.error(f'Failed to load configuration: {e}')}")
        return 1

    # Initialize registry (required for LLM providers)
    try:
        from osprey.registry import initialize_registry

        initialize_registry()
        console.print(f"  {Messages.success('Registry initialized')}")
    except Exception as e:
        console.print(f"  {Messages.error(f'Failed to initialize registry: {e}')}")
        return 1

    # Initialize benchmark runner (reads from config)
    try:
        runner = BenchmarkRunner()
        console.print(f"  {Messages.success('Benchmark runner initialized')}")
    except Exception as e:
        console.print(f"  {Messages.error(f'Failed to initialize runner: {e}')}")
        return 1

    # Display configuration
    config = get_config()
    pipeline_mode = config.get("channel_finder", {}).get("pipeline_mode", "unknown")

    # Get benchmark dataset info
    pipeline_config = config.get("channel_finder", {}).get("pipelines", {}).get(pipeline_mode, {})
    benchmark_dataset = dataset or pipeline_config.get("benchmark", {}).get(
        "dataset_path", "Not configured"
    )

    # Create configuration table
    config_table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
    config_table.add_column("Setting", style=Styles.LABEL)
    config_table.add_column("Value", style=Styles.VALUE)

    config_table.add_row("Pipeline", pipeline_mode)
    config_table.add_row("Dataset", benchmark_dataset)
    if queries:
        config_table.add_row("Queries", queries)
    if model:
        config_table.add_row("Model", model)

    console.print()
    console.print(
        Panel(
            config_table,
            title="[bold]Configuration[/bold]",
            border_style=Styles.BORDER,
            padding=(0, 1),
        )
    )
    console.print()

    console.print(f"  [{Styles.INFO}]⏳ Running benchmarks...[/{Styles.INFO}]")
    console.print()

    # Run benchmarks
    try:
        start_time = datetime.now()

        # Run all enabled datasets (filtered by config overrides above)
        await runner.run_all_enabled_benchmarks()

        elapsed = (datetime.now() - start_time).total_seconds()

        # Create results table
        results_table = Table(show_header=False, box=None, padding=(0, 2), show_edge=False)
        results_table.add_column("Metric", style=Styles.LABEL)
        results_table.add_column("Value", style=Styles.VALUE)

        results_table.add_row("⏱️  Total time", f"{elapsed:.2f}s")
        results_table.add_row("📁 Results location", "data/benchmarks/results/")

        # Display completion message
        console.print()
        console.print(
            Panel(
                results_table,
                title="[bold]✅ Benchmarks Complete[/bold]",
                border_style=Styles.SUCCESS,
                padding=(0, 1),
            )
        )
        console.print()
        console.print(
            f"  [{Styles.INFO}]ℹ️  Check the output files for detailed metrics and results[/{Styles.INFO}]"
        )
        console.print()

        return 0

    except KeyboardInterrupt:
        console.print()
        console.print(f"  {Messages.warning('Benchmark interrupted by user')}")
        console.print()
        return 130
    except Exception as e:
        console.print()
        console.print(f"  {Messages.error(f'Benchmark failed: {e}')}")
        import traceback

        console.print(f"\n[{Styles.DIM}]{traceback.format_exc()}[/{Styles.DIM}]")
        console.print()
        return 1
