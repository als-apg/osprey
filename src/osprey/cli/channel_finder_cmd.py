"""Channel Finder CLI command.

Provides the 'osprey channel-finder' command group with subcommands:
- Build database (osprey channel-finder build-database)
- Validate database (osprey channel-finder validate)
- Preview database (osprey channel-finder preview)
- Web interface (osprey channel-finder web)
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
