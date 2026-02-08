"""Channel Finder CLI command.

Provides the 'osprey channel-finder' command group with subcommands:
- Interactive REPL (default, no subcommand)
- Direct query (osprey channel-finder query "...")
- Benchmarks (osprey channel-finder benchmark)
- Build database (osprey channel-finder build-database)
- Validate database (osprey channel-finder validate)
- Preview database (osprey channel-finder preview)
"""

import asyncio
import os

import click

from osprey.cli.styles import Messages, Styles, console


def _setup_config(project: str | None):
    """Resolve and set CONFIG_FILE from project path.

    Args:
        project: Optional project directory path.

    Raises:
        click.ClickException: If config.yml cannot be found.
    """
    from .project_utils import resolve_config_path

    config_path = resolve_config_path(project)
    if not os.path.exists(config_path):
        raise click.ClickException(
            f"Configuration file not found: {config_path}\n"
            "Run 'osprey init' to create a project, or use --project to specify the project directory."
        )
    os.environ["CONFIG_FILE"] = str(config_path)


def _initialize_registry(verbose: bool = False):
    """Initialize the Osprey registry with appropriate logging.

    Args:
        verbose: If True, show detailed initialization logs.
    """
    import logging

    from osprey.registry import initialize_registry

    if not verbose:
        logging.getLogger("osprey").setLevel(logging.WARNING)
        logging.getLogger("channel_finder").setLevel(logging.WARNING)

    from osprey.utils.log_filter import quiet_logger

    with quiet_logger(
        [
            "REGISTRY",
            "osprey.approval",
            "osprey.services",
            "connector_factory",
            "memory_storage",
        ]
    ):
        initialize_registry(silent=True)


@click.group("channel-finder", invoke_without_command=True)
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging")
@click.pass_context
def channel_finder(ctx, project: str | None, verbose: bool):
    """Channel Finder - natural language channel search.

    Interactive tool for finding control system channels using natural
    language queries. Supports interactive REPL mode, direct queries,
    and benchmarking.

    Examples:

    \b
      osprey channel-finder                           Interactive REPL
      osprey channel-finder query "find BPMs"         Direct query
      osprey channel-finder benchmark                 Run benchmarks
      osprey channel-finder benchmark --queries 0:10  Benchmark subset
    """
    ctx.ensure_object(dict)
    ctx.obj["project"] = project
    ctx.obj["verbose"] = verbose

    # Default action: launch interactive REPL
    if ctx.invoked_subcommand is None:
        try:
            _setup_config(project)
            _initialize_registry(verbose)

            from osprey.services.channel_finder.cli import ChannelFinderCLI

            cli = ChannelFinderCLI()
            asyncio.run(cli.run())
        except click.ClickException:
            raise
        except KeyboardInterrupt:
            console.print("\n\nGoodbye!", style=Styles.WARNING)
        except Exception as e:
            console.print(f"\n{Messages.error(str(e))}")
            raise click.Abort() from None


@channel_finder.command()
@click.argument("query_text")
@click.pass_context
def query(ctx, query_text: str):
    """Execute a single channel finder query.

    QUERY_TEXT is the natural language query to search for channels.

    Examples:

    \b
      osprey channel-finder query "find beam position monitors"
      osprey channel-finder query "show me all vacuum gauges"
      osprey channel-finder -v query "BPM readbacks"
    """
    project = ctx.obj["project"]
    verbose = ctx.obj["verbose"]

    try:
        _setup_config(project)
        _initialize_registry(verbose)

        from osprey.services.channel_finder.cli import direct_query

        exit_code = asyncio.run(direct_query(query_text, verbose=verbose))
        if exit_code:
            raise SystemExit(exit_code)
    except click.ClickException:
        raise
    except SystemExit as e:
        raise SystemExit(e.code) from None
    except KeyboardInterrupt:
        console.print("\n\nQuery cancelled.", style=Styles.WARNING)
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        raise click.Abort() from None


@channel_finder.command()
@click.option("--queries", type=str, help='Query selection (e.g., "all", "0:10", "0,5,10")')
@click.option("--model", type=str, help="Override model (e.g., anthropic/claude-sonnet)")
@click.option("--dataset", type=str, help="Path to custom benchmark dataset JSON file")
@click.option(
    "--verbose",
    "-v",
    "bench_verbose",
    is_flag=True,
    default=False,
    help="Show detailed channel finder logs",
)
@click.pass_context
def benchmark(
    ctx, queries: str | None, model: str | None, dataset: str | None, bench_verbose: bool
):
    """Run channel finder benchmarks.

    Evaluates channel finder performance and accuracy against benchmark
    datasets. Results are saved to data/benchmarks/results/.

    Examples:

    \b
      osprey channel-finder benchmark
      osprey channel-finder benchmark --queries 0:10
      osprey channel-finder benchmark --model anthropic/claude-sonnet
      osprey channel-finder benchmark --dataset data/benchmarks/my_data.json
      osprey channel-finder benchmark --queries 0:10 --model anthropic/claude-sonnet
    """
    project = ctx.obj["project"]
    verbose = ctx.obj["verbose"] or bench_verbose

    try:
        _setup_config(project)

        from osprey.services.channel_finder.benchmarks.cli import run_benchmarks

        exit_code = asyncio.run(
            run_benchmarks(dataset=dataset, queries=queries, model=model, verbose=verbose)
        )
        if exit_code:
            raise SystemExit(exit_code)
    except click.ClickException:
        raise
    except SystemExit as e:
        raise SystemExit(e.code) from None
    except KeyboardInterrupt:
        console.print("\n\nBenchmark cancelled.", style=Styles.WARNING)
        raise click.Abort() from None
    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        raise click.Abort() from None


@channel_finder.command("build-database")
@click.option(
    "--csv",
    type=click.Path(exists=True, dir_okay=False),
    default="data/raw/address_list.csv",
    help="Input CSV file (default: data/raw/address_list.csv)",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default="data/processed/channel_database.json",
    help="Output JSON file (default: data/processed/channel_database.json)",
)
@click.option(
    "--use-llm",
    is_flag=True,
    default=False,
    help="Use LLM to generate descriptive names for standalone channels",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to facility config file (optional, auto-detected if not provided)",
)
def build_database(csv: str, output: str, use_llm: bool, config_path: str | None):
    """Build a channel database from a CSV file.

    Reads a CSV with columns: address, description, family_name, instances, sub_channel.
    Rows with family_name are grouped into templates; rows without are standalone channels.

    Examples:

    \b
      osprey channel-finder build-database
      osprey channel-finder build-database --csv data/raw/channels.csv
      osprey channel-finder build-database --use-llm --config config.yml
      osprey channel-finder build-database --output data/processed/my_db.json
    """
    from pathlib import Path

    from osprey.services.channel_finder.tools.build_database import (
        build_database as do_build,
    )

    csv_path = Path(csv)
    output_path = Path(output)

    try:
        do_build(
            csv_path=csv_path,
            output_path=output_path,
            use_llm=use_llm,
            config_path=Path(config_path) if config_path else None,
        )
    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        raise click.Abort() from None


@channel_finder.command("validate")
@click.option(
    "--database",
    "-d",
    type=click.Path(dir_okay=False),
    default=None,
    help="Path to database file (default: from config)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Show detailed statistics")
@click.option(
    "--pipeline",
    type=click.Choice(["hierarchical", "in_context"]),
    default=None,
    help="Override pipeline type detection (default: auto-detect from config)",
)
@click.pass_context
def validate(ctx, database: str | None, verbose: bool, pipeline: str | None):
    """Validate a channel database JSON file.

    Checks JSON structure, schema validity, and database loading.
    Auto-detects pipeline type (hierarchical vs in_context).

    Examples:

    \b
      osprey channel-finder validate
      osprey channel-finder validate --database data/processed/db.json
      osprey channel-finder validate --verbose
      osprey channel-finder validate --pipeline hierarchical
    """
    project = ctx.obj.get("project")

    try:
        _setup_config(project)
        _initialize_registry(verbose=False)
    except click.ClickException:
        if not database:
            raise
        # If a database path was provided, we can still validate without config

    from osprey.services.channel_finder.tools.validate_database import run_validation

    exit_code = run_validation(database=database, pipeline=pipeline, verbose=verbose)
    if exit_code:
        raise SystemExit(exit_code)


@channel_finder.command("preview")
@click.option(
    "--depth",
    type=int,
    default=3,
    help="Tree depth to display (default: 3, use -1 for unlimited)",
)
@click.option(
    "--max-items",
    type=int,
    default=3,
    help="Maximum items per level (default: 3, use -1 for unlimited)",
)
@click.option(
    "--sections",
    type=str,
    default="tree",
    help="Comma-separated sections: tree,stats,breakdown,samples,all (default: tree)",
)
@click.option(
    "--focus",
    type=str,
    default=None,
    help='Focus on specific path (e.g., "M:QB" for QB family in M system)',
)
@click.option(
    "--database",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Direct path to database file (overrides config, auto-detects type)",
)
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Show complete hierarchy (shorthand for --depth -1 --max-items -1)",
)
@click.pass_context
def preview(
    ctx,
    depth: int,
    max_items: int,
    sections: str,
    focus: str | None,
    database: str | None,
    full: bool,
):
    """Preview a channel database with flexible display options.

    Auto-detects database type (hierarchical, in_context, middle_layer)
    and shows a tree visualization with configurable depth and sections.

    Examples:

    \b
      osprey channel-finder preview
      osprey channel-finder preview --depth 4 --sections tree,stats
      osprey channel-finder preview --database data/processed/db.json
      osprey channel-finder preview --full --sections all
      osprey channel-finder preview --focus M:QB --depth 4
    """
    project = ctx.obj.get("project")

    if not database:
        try:
            _setup_config(project)
            _initialize_registry(verbose=False)
        except click.ClickException:
            raise

    from osprey.services.channel_finder.tools.preview_database import preview_database

    try:
        preview_database(
            depth=depth,
            max_items=max_items,
            sections=sections,
            focus=focus,
            show_full=full,
            db_path=database,
        )
    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        raise click.Abort() from None
