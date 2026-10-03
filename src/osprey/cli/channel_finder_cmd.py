"""Channel Finder CLI command.

Provides the 'osprey channel-finder' command group with subcommands:
- Validate database (osprey channel-finder validate)
- Preview database (osprey channel-finder preview)
- Web interface (osprey channel-finder web)
- Benchmark (osprey channel-finder benchmark)
"""

import os

import click

from osprey.build.modes import VALID_CHANNEL_FINDER_MODES
from osprey.cli import output
from osprey.cli.altitude import lift_gate
from osprey.cli.styles import Messages, Styles, console

#: The paradigms ``validate --pipeline`` accepts: every registered paradigm
#: whose store is a database file on disk.
#:
#: ``graph`` is the one deliberate exclusion. A graph store is a service
#: reached over the network, so ``validate`` (which opens a file and checks it)
#: has no file to work on. Derived by subtraction from
#: :data:`~osprey.build.modes.VALID_CHANNEL_FINDER_MODES` so registering a
#: file-backed paradigm opens it up without a second edit, and so the exclusion
#: stays a stated rule rather than a list that silently falls behind.
FILE_DATABASE_PARADIGMS: list[str] = sorted(set(VALID_CHANNEL_FINDER_MODES) - {"graph"})


def _setup_config(project: str | None):
    """Resolve and set CONFIG_FILE from project path.

    Resolution is :func:`osprey.cli.project_utils.resolve_config_path`'s, so this
    group reads the same config ``osprey health`` reports on from the same
    stance: the render of the deployment repo enclosing the working directory,
    or the flat config of a rendered project directory named outright.

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
            "No built deployment was found there, and no deployment repo encloses it. "
            "Run 'osprey init my-project --preset hello-world' to create one, then "
            "'osprey build' from inside it, or name a project with --project."
        )
    os.environ["CONFIG_FILE"] = str(config_path)


def _initialize_registry():
    """Initialize the Osprey registry without its start-up chatter.

    Sets no logger levels of its own: what a run renders is the CLI's altitude
    policy, applied once for every command. The named loggers below are silenced
    for the duration of this call only — registry wiring narrates each component
    it loads, and that transcript belongs to ``-v``, not to a database command
    that happens to need a registry first.
    """
    from osprey.registry import initialize_registry
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
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Deployment repo or rendered project directory. Default: the repo enclosing cwd.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose logging")
@click.pass_context
def channel_finder(ctx, project: str | None, verbose: bool):
    """Channel Finder - channel database tools.

    Tools for validating, previewing, serving and benchmarking
    control system channel databases.

    Examples:

    \b
      osprey channel-finder validate
      osprey channel-finder preview
      osprey channel-finder web
    """
    ctx.ensure_object(dict)
    ctx.obj["project"] = project
    ctx.obj["verbose"] = verbose
    if verbose:
        # The group's own --verbose lifts the CLI altitude gate for this run, so
        # every subcommand under it renders its transcript rather than only
        # warnings and errors. Idempotent, and a no-op when nothing is gated.
        lift_gate()


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
    type=click.Choice(FILE_DATABASE_PARADIGMS),
    default=None,
    help=(
        "Paradigm to validate as; without --database, validates "
        "channel_finder.pipelines.<paradigm>.database.path (default: auto-detect from config)"
    ),
)
@click.pass_context
def validate(ctx, database: str | None, verbose: bool, pipeline: str | None):
    """Validate a channel database JSON file.

    Checks JSON structure, schema validity, and database loading.
    --pipeline names the paradigm; without --database it validates the file
    configured for that paradigm. Without --pipeline the paradigm is
    auto-detected from config, and a graph project, which has no database
    file, is told how to seed and inspect its store instead.

    Examples:

    \b
      osprey channel-finder validate
      osprey channel-finder validate --database data/processed/db.json
      osprey channel-finder validate --verbose
      osprey channel-finder validate --pipeline hierarchical
    """
    if verbose:
        # Lifts the altitude gate for this run, on top of the detailed
        # statistics the flag already asks ``run_validation`` for.
        lift_gate()

    project = ctx.obj.get("project")

    try:
        _setup_config(project)
        _initialize_registry()
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

    Auto-detects the paradigm from config and shows a tree visualization with
    configurable depth and sections. A graph project has no database file: it
    is told how to seed and inspect its store instead.

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
            _initialize_registry()
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
@click.option("--host", default=None, help="Host to bind to (default: from config or 127.0.0.1)")
@click.option(
    "--port",
    default=None,
    type=int,
    help=(
        "Port to run on (default: OSPREY_CHANNEL_FINDER_PORT, then config, "
        "then this deployment's layout port)"
    ),
)
@click.pass_context
def web(ctx, host: str | None, port: int | None):
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
    from osprey.interfaces.common_middleware import WEB_PORT_ENV
    from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV, mint_and_announce
    from osprey.registry.web import resolve_web_server_bind
    from osprey.utils.config import get_config_builder

    # The config _setup_config selected, not the working directory's: under --project they differ.
    # It is read only when a flag is missing, because a fully flagged run may have no config.
    config = get_config_builder().raw_config if host is None or port is None else None
    host, port = resolve_web_server_bind("channel_finder", config, host=host, port=port)

    # Publish the settled port before the app is constructed: cookies ignore
    # ports, so two OSPREY servers on this host share an origin as far as the
    # browser is concerned, and the port is the only thing keeping their session
    # cookies apart. ``session_cookie_name()`` reads it from here.
    os.environ[WEB_PORT_ENV] = str(port)

    # Mint the operator secret in this CLI parent, which becomes the server
    # (direct-serve: uvicorn.run(app) runs in-process). ``mint_and_announce``
    # settles the secret and returns the ``?token=`` login URL — the
    # operator's only way past the auth middleware. ``announce`` is False only
    # when the secret was already supplied by an ancestor/deployment, so a
    # supplied secret is never re-echoed here.
    announce = not (os.environ.get(OPERATOR_SECRET_ENV) or "").strip()
    login_url = mint_and_announce(host, port)

    output.report(f"Starting Channel Finder at http://{host}:{port}")
    if announce:
        # ``output.report`` rather than ``console.print``: the login URL is a
        # single unbroken token, and the plain Rich console wraps it at the
        # terminal width, so a copied line loses the middle of the secret.
        output.report(f"Open: {login_url}")
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


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
        "LiteLLM-form provider/wire_id (e.g. anthropic/claude-haiku-4-5, "
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
    if verbose:
        # One half of what --verbose means here: the altitude gate is lifted, so
        # this run's records are rendered instead of only its warnings. The
        # other half is the level floor set below, which is what lets the
        # framework's DEBUG records be emitted in the first place.
        lift_gate()

    import asyncio
    import logging
    from pathlib import Path

    from osprey.services.channel_finder.benchmarks.models import (
        BenchmarkSuite,
    )
    from osprey.services.channel_finder.benchmarks.runner import (
        BenchmarkRunner,
    )

    from .project_utils import project_config_path, resolve_project_path

    # The group's one resolution rule, not a third spelling of it: the repo
    # enclosing the working directory, or the project directory named outright.
    config_path = project_config_path(resolve_project_path(ctx.obj.get("project")))
    if not config_path.exists():
        raise click.ClickException(
            f"config.yml not found: {config_path}\n"
            "Run this from a built deployment repo, or name one with --project."
        )

    # The runner reads `config.yml` at its own root, so it is handed the
    # directory holding the config — the `build/` render on a host, the project
    # directory itself in a container. Its outputs land beside it for the same
    # reason: a benchmark result is exhaust from that render, not repo source.
    project_dir = config_path.parent

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
        # A level floor, and only that: raising the framework logger is what
        # lets its DEBUG records be emitted at all. Whether an emitted record
        # reaches the terminal is the CLI's altitude policy — the gate lifted at
        # the top of this body.
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
