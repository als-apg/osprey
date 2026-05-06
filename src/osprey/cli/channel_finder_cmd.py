"""Channel Finder CLI command.

Provides the 'osprey channel-finder' command group with subcommands:
- Build database (osprey channel-finder build-database)
- Validate database (osprey channel-finder validate)
- Preview database (osprey channel-finder preview)
- Web interface (osprey channel-finder web)
- Benchmark (osprey channel-finder benchmark)
"""

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
            "Run 'osprey build my-project --preset hello-world' to create a project, "
            "or use --project to specify the project directory."
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
            "osprey.services",
            "connector_factory",
        ]
    ):
        initialize_registry(silent=True)


@click.group("channel-finder")
@click.option(
    "--project",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Project directory (default: current directory or OSPREY_PROJECT env var)",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging")
@click.pass_context
def channel_finder(ctx, project: str | None, verbose: bool):
    """Channel Finder - channel database tools.

    Tools for building, validating, previewing, and serving
    control system channel databases.

    Examples:

    \b
      osprey channel-finder build-database
      osprey channel-finder validate
      osprey channel-finder preview
      osprey channel-finder web
    """
    ctx.ensure_object(dict)
    ctx.obj["project"] = project
    ctx.obj["verbose"] = verbose


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
@click.option(
    "--delimiter",
    default=",",
    help="CSV field delimiter (default: ',')",
)
def build_database(csv: str, output: str, use_llm: bool, config_path: str | None, delimiter: str):
    """Build a channel database from a CSV file.

    Reads a CSV with columns: address, description, family_name, instances, sub_channel.
    Rows with family_name are grouped into templates; rows without are standalone channels.

    Examples:

    \b
      osprey channel-finder build-database
      osprey channel-finder build-database --csv data/raw/channels.csv
      osprey channel-finder build-database --delimiter "|"
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
            delimiter=delimiter,
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

    exit_code = run_validation(
        database=database, pipeline=pipeline, verbose=verbose, console=console
    )
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

    Auto-detects database type (hierarchical, in_context)
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
            console=console,
        )
    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        raise click.Abort() from None


@channel_finder.command("web")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8092, type=int, help="Port to run on")
@click.pass_context
def web(ctx, host: str, port: int):
    """Launch the Channel Finder web interface.

    Opens a browser-based interface for exploring, searching, and managing
    control system channels.

    Examples:

    \b
      osprey channel-finder web
      osprey channel-finder web --port 9000
    """
    project = ctx.obj.get("project")
    try:
        _setup_config(project)
    except click.ClickException:
        raise

    import uvicorn

    from osprey.interfaces.channel_finder.app import create_app

    console.print(f"Starting Channel Finder at http://{host}:{port}", style=Styles.SUCCESS)
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


@channel_finder.command("generate")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="data/channel_databases",
    help="Output directory for generated databases (default: data/channel_databases/)",
)
@click.option(
    "--source",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Source hierarchical database (default: built-in template)",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["in_context", "hierarchical", "middle_layer", "all"]),
    default="all",
    help="Format(s) to generate (default: all)",
)
@click.option(
    "--tier",
    type=click.Choice(["1", "2", "3", "none"]),
    default="none",
    help="Tier filter: 1, 2, 3, or none for all channels (default: none)",
)
@click.option(
    "--validate",
    "do_validate",
    is_flag=True,
    default=False,
    help="Verify generated databases load correctly through pipeline database classes",
)
def generate(output_dir: str, source: str | None, fmt: str, tier: str, do_validate: bool):
    """Generate channel databases from a hierarchical template.

    Produces database files from a hierarchical channel template.
    By default, generates all three formats with all channels (no tier
    filtering).

    \b
      - in_context.json    (flat format with aliases)
      - hierarchical.json  (tree format)
      - middle_layer.json  (MML-style with setup blocks)

    Examples:

    \b
      osprey channel-finder generate
      osprey channel-finder generate --tier 1 --format in_context
      osprey channel-finder generate --source my_channels.json
      osprey channel-finder generate --validate
    """
    import json
    from pathlib import Path

    from osprey.services.channel_finder.benchmarks.generator import (
        TIER_1,
        TIER_2,
        TIER_3,
        TierSpec,
        format_hierarchical,
        format_in_context,
        format_middle_layer,
        load_template,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_path = Path(source) if source else None
    tree_data, channels = load_template(source_path)

    if tier == "none":
        all_rings = frozenset(ch["ring"] for ch in channels)
        tier_spec = TierSpec(
            name="all",
            rings=all_rings,
            families=None,
            allowed_subfields=None,
            target_count=len(channels),
        )
    else:
        tier_spec = {"1": TIER_1, "2": TIER_2, "3": TIER_3}[tier]

    format_map = {
        "in_context.json": lambda: format_in_context(channels, tier_spec),
        "hierarchical.json": lambda: format_hierarchical(tree_data, tier_spec),
        "middle_layer.json": lambda: format_middle_layer(channels, tier_spec),
    }

    if fmt != "all":
        filename = f"{fmt}.json"
        format_map = {filename: format_map[filename]}

    for filename, builder in format_map.items():
        path = out / filename
        path.write_text(json.dumps(builder(), indent=2), encoding="utf-8")
        console.print(f"  Generated {path}", style=Styles.SUCCESS)

    console.print(f"\n{len(format_map)} database(s) generated in {out}/", style=Styles.SUCCESS)

    if do_validate:
        console.print("\nValidating generated databases...", style=Styles.INFO)

        from osprey.services.channel_finder.databases.flat import ChannelDatabase
        from osprey.services.channel_finder.databases.hierarchical import (
            HierarchicalChannelDatabase,
        )
        from osprey.services.channel_finder.databases.middle_layer import (
            MiddleLayerDatabase,
        )

        validators = {
            "in_context.json": ChannelDatabase,
            "hierarchical.json": HierarchicalChannelDatabase,
            "middle_layer.json": MiddleLayerDatabase,
        }

        all_valid = True
        for filename in format_map:
            db_class = validators[filename]
            path = out / filename
            try:
                db = db_class(str(path))
                db.load_database()
                stats = db.get_statistics()
                console.print(
                    f"  {filename}: OK ({stats.get('total_channels', '?')} channels)",
                    style=Styles.SUCCESS,
                )
            except Exception as e:
                console.print(f"  {filename}: FAILED - {e}", style="bold red")
                all_valid = False

        if not all_valid:
            raise click.ClickException("Validation failed for one or more databases")
        console.print("\nAll databases validated successfully!", style=Styles.SUCCESS)


def _parse_query_indices(queries_spec: str, total: int) -> list[int]:
    """Parse a query index specification into a list of indices.

    Supports:
      - ``"all"`` -> all indices ``[0, 1, ..., total-1]``
      - ``"0:10"`` -> slice indices ``[0, 1, ..., 9]``
      - ``"0,5,10"`` -> explicit indices ``[0, 5, 10]``

    Args:
        queries_spec: The query specification string.
        total: Total number of available queries.

    Returns:
        Sorted list of integer indices.

    Raises:
        click.BadParameter: If the specification cannot be parsed.
    """
    if queries_spec == "all":
        return list(range(total))
    if ":" in queries_spec:
        parts = queries_spec.split(":")
        if len(parts) != 2:
            raise click.BadParameter(f"Invalid slice format: {queries_spec!r}. Use start:stop.")
        start = int(parts[0])
        stop = int(parts[1])
        return list(range(start, min(stop, total)))
    # Comma-separated indices
    try:
        return sorted(int(i) for i in queries_spec.split(","))
    except ValueError:
        raise click.BadParameter(
            f"Cannot parse query indices: {queries_spec!r}. Use 'all', 'start:stop', or 'i,j,k'."
        ) from None


@channel_finder.command("benchmark")
@click.option(
    "--model",
    required=True,
    help=(
        "LiteLLM-form ``provider/wire_id`` (e.g. anthropic/claude-haiku-4-5, "
        "ollama/gemma3:4b). The provider determines auth and routing; the "
        "wire id is forwarded upstream. Saved BenchmarkRun.model records "
        "the exact string for reproducibility."
    ),
)
@click.option(
    "--queries",
    default="all",
    help="all, or indices like 0:10 or 0,5,10",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose benchmark logging",
)
@click.option(
    "--runs-per-query",
    default=1,
    type=int,
    help="Number of benchmark runs (default: 1)",
)
@click.option(
    "--concurrency",
    default=5,
    type=int,
    help="Max concurrent queries (default: 5)",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to save result JSON files (default: data/benchmarks/results/)",
)
@click.option(
    "--queries-path",
    default=None,
    help="Override benchmark dataset path from config",
)
@click.pass_context
def benchmark(
    ctx,
    model: str,
    queries: str,
    verbose: bool,
    runs_per_query: int,
    concurrency: int,
    output_dir: str | None,
    queries_path: str | None,
):
    """Run channel finder benchmarks against the current project.

    Evaluates channel finder accuracy using the Claude Agent SDK.
    Reads the pipeline mode and benchmark dataset from the project's
    config.yml; the model is passed in directly.

    Examples:

    \b
      osprey channel-finder benchmark --model anthropic/claude-haiku-4-5
      osprey channel-finder benchmark --model ollama/gemma3:4b --queries 0:5
      osprey channel-finder benchmark --model anthropic/claude-haiku-4-5 --runs-per-query 3
    """
    import asyncio
    import logging
    from pathlib import Path

    from osprey.services.channel_finder.benchmarks.models import (
        BenchmarkSuite,
    )
    from osprey.services.channel_finder.benchmarks.runner import (
        BenchmarkRunner,
    )

    # Resolve project directory
    project_dir = Path(ctx.obj.get("project") or os.getcwd())
    config_path = project_dir / "config.yml"
    if not config_path.exists():
        raise click.ClickException(
            f"config.yml not found in {project_dir}\n"
            "Run this from an OSPREY project directory or use --project."
        )

    out_directory = (
        Path(output_dir) if output_dir else project_dir / "data" / "benchmarks" / "results"
    )

    runner = BenchmarkRunner(
        project_dir,
        model=model,
        max_concurrent=concurrency,
        verbose=verbose,
        queries_override=Path(queries_path) if queries_path else None,
    )

    # Load queries and parse index spec
    all_queries = runner.load_queries()
    indices = _parse_query_indices(queries, len(all_queries))

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("osprey").setLevel(logging.DEBUG)

    console.print(
        f"Benchmark: {len(indices)} query/queries x {runs_per_query} run(s) | "
        f"provider={runner.provider} model={runner.model} | "
        f"concurrency={concurrency}",
        style=Styles.INFO,
    )

    try:
        all_runs = []
        for run_idx in range(runs_per_query):
            if runs_per_query > 1:
                console.print(
                    f"\n--- Run {run_idx + 1}/{runs_per_query} ---",
                    style=Styles.INFO,
                )
            run = asyncio.run(
                runner.run_queries(
                    query_indices=indices if queries != "all" else None,
                    output_dir=out_directory,
                )
            )
            all_runs.append(run)

        # Print summary
        console.print(
            f"\n[bold]Benchmark complete:[/bold] {len(all_runs)} run(s) executed",
            style=Styles.SUCCESS,
        )
        for run in all_runs:
            failed_msg = f"  failed={run.num_failed}" if run.num_failed > 0 else ""
            console.print(
                f"  {run.paradigm}: "
                f"F1={run.aggregate_f1:.3f}  "
                f"P={run.aggregate_precision:.3f}  "
                f"R={run.aggregate_recall:.3f}  "
                f"cost=${run.total_cost_usd:.4f}  "
                f"latency={run.avg_latency_s:.1f}s"
                f"{failed_msg}",
            )

        # Save combined suite
        combined = BenchmarkSuite(
            runs=all_runs,
            metadata={
                "provider": runner.provider,
                "model": runner.model,
                "runs_per_query": runs_per_query,
                "query_count": len(indices),
            },
        )
        out_directory.mkdir(parents=True, exist_ok=True)
        suite_path = out_directory / "suite_latest.json"
        combined.to_json(suite_path)
        console.print(f"\nResults saved to {suite_path}", style=Styles.SUCCESS)

    except Exception as e:
        console.print(f"\n{Messages.error(str(e))}")
        raise click.Abort() from None
