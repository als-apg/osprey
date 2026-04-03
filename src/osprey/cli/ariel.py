"""ARIEL CLI commands.

Thin CLI wrappers that delegate business logic to
``osprey.services.ariel_search.cli_operations``.

See 04_OSPREY_INTEGRATION.md Sections 13 for specification.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import click

# Import get_config_value at module level for easier patching in tests
from osprey.utils.config import get_config_value
from osprey.utils.logger import get_logger

logger = get_logger("ariel")

if TYPE_CHECKING:
    from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ariel_config() -> dict:
    """Load ARIEL config dict, raising SystemExit if missing."""
    config_dict = get_config_value("ariel", {})
    if not config_dict:
        click.echo("Error: ARIEL not configured in config.yml", err=True)
        raise SystemExit(1)
    return config_dict


def _handle_db_error(e: Exception) -> None:
    """Raise SystemExit on database connection errors, otherwise return."""
    msg = str(e)
    if "connection" in msg.lower() or "connect" in msg.lower():
        click.echo("Error: Cannot connect to the ARIEL database.", err=True)
        click.echo("Make sure the database is running: osprey deploy up", err=True)
        raise SystemExit(1) from None


def _handle_missing_tables(e: Exception) -> None:
    """Raise SystemExit on missing-table errors, otherwise return."""
    msg = str(e)
    if "relation" in msg and "does not exist" in msg:
        click.echo("Error: ARIEL database is not initialized.", err=True)
        click.echo("Run 'osprey ariel migrate' to create the required tables.", err=True)
        raise SystemExit(1) from None


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("ariel")
def ariel_group() -> None:
    """ARIEL search service commands.

    Commands for managing the ARIEL (Agentic Retrieval Interface for
    Electronic Logbooks) search service.
    """


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@ariel_group.command("status")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def status_command(output_json: bool) -> None:
    """Show ARIEL service status.

    Displays database connection, embedding tables, and enhancement stats.
    """
    import json as json_module

    from osprey.services.ariel_search.cli_operations import get_status

    config_dict = get_config_value("ariel", {})
    result = asyncio.run(get_status(config_dict))

    if output_json:
        click.echo(json_module.dumps(result, indent=2))
    else:
        click.echo(f"ARIEL Status: {result['status']}")
        click.echo(f"  {result['message']}")
        if result["status"] != "error":
            click.echo(f"\nDatabase: {result['database']['uri']}")
            click.echo(f"Total Entries: {result['entries']}")
            click.echo("\nEmbedding Tables:")
            for table in result.get("embedding_tables", []):
                active = " (active)" if table["active"] else ""
                click.echo(f"  - {table['table']}: {table['entries']} entries{active}")


@ariel_group.command("migrate")
def migrate_command() -> None:
    """Run ARIEL database migrations.

    Creates required database schema and tables based on enabled modules.
    """
    from osprey.services.ariel_search.cli_operations import run_migrate

    config_dict = _load_ariel_config()
    try:
        asyncio.run(run_migrate(config_dict, progress=click.echo))
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("sync")
@click.option("--limit", type=int, help="Maximum entries to ingest per run")
def sync_command(limit: int | None) -> None:
    """Sync ARIEL database: migrate, incremental ingest, enhance.

    Idempotent — safe to run on every build. Only fetches new entries
    since the last successful ingest run. On a fresh database, runs
    a full ingest.

    Example:
        osprey ariel sync                # Full sync
        osprey ariel sync --limit 1000   # Limit ingest to 1000 entries
    """
    from osprey.services.ariel_search.cli_operations import run_sync
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    config_dict = _load_ariel_config()
    try:
        result = asyncio.run(run_sync(config_dict, limit=limit, progress=click.echo))
        click.echo(
            f"\nSync complete: "
            f"{result.entries_ingested} ingested, "
            f"{result.entries_enhanced} enhanced, "
            f"{result.entries_failed} failed"
        )
        if result.migrations_applied:
            click.echo(f"  Migrations applied: {result.migrations_applied}")
    except DatabaseQueryError as e:
        _handle_missing_tables(e)
        raise
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("ingest")
@click.option("--source", "-s", required=True, help="Source file path or URL")
@click.option(
    "--adapter",
    "-a",
    type=click.Choice(["als_logbook", "jlab_logbook", "ornl_logbook", "generic_json"]),
    default="generic_json",
    help="Adapter type",
)
@click.option("--since", type=click.DateTime(), help="Only ingest entries after this date")
@click.option("--limit", type=int, help="Maximum entries to ingest")
@click.option("--dry-run", is_flag=True, help="Parse entries without storing")
def ingest_command(
    source: str,
    adapter: str,
    since: datetime | None,
    limit: int | None,
    dry_run: bool,
) -> None:
    """Ingest logbook entries from a source file or URL.

    Parses entries from the source using the specified adapter
    and stores them in the ARIEL database. Accepts both local
    file paths and HTTP/HTTPS URLs.
    """
    from osprey.services.ariel_search.cli_operations import run_ingest
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    config_dict = _load_ariel_config()
    try:
        result = asyncio.run(
            run_ingest(config_dict, source, adapter, since, limit, dry_run, progress=click.echo)
        )
        if result.dry_run:
            click.echo(f"\nDry run complete: {result.count} entries would be ingested")
            if result.enhancer_names:
                click.echo(f"Enhancement modules would run: {result.enhancer_names}")
        else:
            click.echo(f"\nIngestion complete: {result.count} entries stored")
            if result.enhancer_names:
                click.echo(f"Enhancement complete: {result.enhanced_count} enhancements applied")
    except DatabaseQueryError as e:
        _handle_missing_tables(e)
        raise
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("watch")
@click.option("--source", "-s", help="Source file path or URL (overrides config)")
@click.option(
    "--adapter",
    "-a",
    type=click.Choice(["als_logbook", "jlab_logbook", "ornl_logbook", "generic_json"]),
    help="Adapter type (overrides config)",
)
@click.option("--once", is_flag=True, help="Run a single poll cycle and exit")
@click.option("--interval", type=int, help="Override poll interval (seconds)")
@click.option("--dry-run", is_flag=True, help="Show what would be ingested without storing")
def watch_command(
    source: str | None,
    adapter: str | None,
    once: bool,
    interval: int | None,
    dry_run: bool,
) -> None:
    """Watch a source for new logbook entries.

    Continuously polls the configured source for new entries and
    ingests them into the ARIEL database. Uses the last successful
    ingestion timestamp to fetch only new entries.

    Requires at least one prior 'osprey ariel ingest' run by default.
    Use --once for a single poll cycle.

    Example:
        osprey ariel watch                         # Watch using config
        osprey ariel watch --once --dry-run        # Preview one cycle
        osprey ariel watch --interval 300          # Poll every 5 minutes
        osprey ariel watch -s https://api/logbook  # Override source URL
    """
    from osprey.services.ariel_search.cli_operations import run_watch
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    config_dict = _load_ariel_config()
    try:
        result = asyncio.run(
            run_watch(config_dict, source, adapter, once, interval, dry_run, progress=click.echo)
        )
        if result is not None:
            prefix = "[dry-run] " if result.dry_run else ""
            click.echo(
                f"\n{prefix}Poll complete: "
                f"{result.entries_added} added, "
                f"{result.entries_failed} failed "
                f"({result.duration_seconds:.1f}s)"
            )
            if result.since:
                click.echo(f"  Since: {result.since.isoformat()}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1) from None
    except DatabaseQueryError as e:
        _handle_missing_tables(e)
        raise
    except KeyboardInterrupt:
        click.echo("\nStopping watcher...")
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("enhance")
@click.option(
    "--module",
    "-m",
    type=click.Choice(["text_embedding", "semantic_processor"]),
    help="Enhancement module to run",
)
@click.option("--force", is_flag=True, help="Re-process already enhanced entries")
@click.option("--limit", type=int, default=100, help="Maximum entries to process")
def enhance_command(module: str | None, force: bool, limit: int) -> None:
    """Run enhancement modules on entries.

    Processes entries that haven't been enhanced yet, or re-processes
    all entries if --force is specified.
    """
    from osprey.services.ariel_search.cli_operations import run_enhance

    config_dict = _load_ariel_config()
    result = asyncio.run(run_enhance(config_dict, module, force, limit, progress=click.echo))
    if result.entries_processed > 0:
        click.echo(f"\nEnhancement complete: {result.entries_processed} entries processed")


@ariel_group.command("models")
def models_command() -> None:
    """List embedding models and their tables.

    Shows all embedding tables in the database and their status.
    """
    from osprey.services.ariel_search.cli_operations import list_models

    config_dict = _load_ariel_config()
    tables = asyncio.run(list_models(config_dict))

    if not tables:
        click.echo("No embedding tables found.")
        return

    click.echo("Embedding Models:")
    for table in tables:
        active = " (active)" if table["is_active"] else ""
        click.echo(f"\n  {table['table_name']}{active}")
        click.echo(f"    Entries: {table['entry_count']}")
        if table["dimension"]:
            click.echo(f"    Dimension: {table['dimension']}")


@ariel_group.command("search")
@click.argument("query")
@click.option("--mode", type=click.Choice(["keyword", "semantic", "rag", "auto"]), default="auto")
@click.option("--limit", type=int, default=10, help="Maximum results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def search_command(query: str, mode: str, limit: int, output_json: bool) -> None:
    """Search the logbook.

    Execute a search query using the ARIEL agent.
    """
    import json as json_module

    from osprey.services.ariel_search.cli_operations import run_search

    config_dict = get_config_value("ariel", {})
    result = asyncio.run(run_search(config_dict, query, mode, limit))

    if output_json:
        click.echo(json_module.dumps(result, indent=2))
    else:
        if result.get("error"):
            click.echo(f"Error: {result['error']}", err=True)
            return

        click.echo(f"Query: {result['query']}")
        click.echo(f"Modes: {', '.join(result['search_modes']) or 'none'}")
        click.echo()

        if result["answer"]:
            click.echo(result["answer"])
            if result["sources"]:
                click.echo(f"\nSources: {', '.join(result['sources'])}")
        else:
            click.echo("No results found.")


@ariel_group.command("reembed")
@click.option("--model", required=True, help="Embedding model name (e.g., nomic-embed-text)")
@click.option("--dimension", type=int, required=True, help="Embedding dimension (e.g., 768)")
@click.option("--batch-size", type=int, default=100, help="Entries per batch")
@click.option("--dry-run", is_flag=True, help="Show what would be done without executing")
@click.option("--force", is_flag=True, help="Overwrite existing embeddings")
def reembed_command(
    model: str,
    dimension: int,
    batch_size: int,
    dry_run: bool,
    force: bool,
) -> None:
    """Re-embed entries with a new or existing model.

    Creates embeddings for all entries using the specified model.
    If the model's embedding table doesn't exist, it will be created.

    Example:
        osprey ariel reembed --model nomic-embed-text --dimension 768
        osprey ariel reembed --model mxbai-embed-large --dimension 1024 --force
    """
    from osprey.services.ariel_search.cli_operations import run_reembed

    config_dict = _load_ariel_config()
    result = asyncio.run(
        run_reembed(config_dict, model, dimension, batch_size, dry_run, force, progress=click.echo)
    )
    if not result.dry_run:
        click.echo("\nRe-embedding complete:")
        click.echo(f"  Processed: {result.processed}")
        click.echo(f"  Skipped (existing): {result.skipped}")
        click.echo(f"  Errors: {result.errors}")


@ariel_group.command("quickstart")
@click.option(
    "--source",
    "-s",
    type=click.Path(exists=True),
    help="Custom logbook JSON file (default: use config or bundled demo data)",
)
def quickstart_command(source: str | None) -> None:
    """Quick setup for ARIEL logbook search.

    Runs the complete setup sequence:
    1. Checks database connection (prompts to run 'osprey deploy up' if down)
    2. Runs database migrations
    3. Ingests demo logbook data (or custom source)

    Example:
        osprey ariel quickstart                    # Use bundled demo data
        osprey ariel quickstart -s my_logbook.json # Use custom data
    """
    from osprey.services.ariel_search.cli_operations import run_quickstart

    config_dict = _load_ariel_config()
    try:
        asyncio.run(run_quickstart(config_dict, source, progress=click.echo))
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("web")
@click.option("--port", "-p", type=int, default=8085, help="Port to run on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def web_command(port: int, host: str, reload: bool) -> None:
    """Launch the ARIEL web interface.

    Starts a FastAPI server providing a web-based search interface
    for ARIEL with support for search, browsing, and entry creation.

    Example:
        osprey ariel web                    # Start on localhost:8085
        osprey ariel web --port 8080        # Custom port
        osprey ariel web --host 0.0.0.0     # Bind to all interfaces
        osprey ariel web --reload           # Development mode with auto-reload
    """
    _load_ariel_config()

    click.echo(f"Starting ARIEL Web Interface on http://{host}:{port}")
    click.echo("Press Ctrl+C to stop\n")

    try:
        from osprey.interfaces.ariel import run_web

        run_web(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        click.echo("\nShutting down...")


@ariel_group.command("purge")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--embeddings-only", is_flag=True, help="Only purge embedding tables, keep entries")
def purge_command(yes: bool, embeddings_only: bool) -> None:
    """Purge all ARIEL data from the database.

    WARNING: This permanently deletes all logbook entries and embeddings!
    Use --embeddings-only to keep entries but clear embedding tables.

    Example:
        osprey ariel purge              # Interactive confirmation
        osprey ariel purge -y           # Skip confirmation
        osprey ariel purge --embeddings-only  # Keep entries, clear embeddings
    """
    from osprey.services.ariel_search.cli_operations import execute_purge, get_purge_info

    config_dict = _load_ariel_config()

    try:
        info = asyncio.run(get_purge_info(config_dict))
    except Exception as e:
        _handle_db_error(e)
        raise

    click.echo("\n⚠️  WARNING: This will permanently delete:")
    if embeddings_only:
        click.echo(f"  - Embedding tables: {info.embedding_tables or '(none)'}")
        click.echo(f"  - Entries will be KEPT ({info.entry_count} entries)")
    else:
        click.echo(f"  - All {info.entry_count} logbook entries")
        click.echo(f"  - All embedding tables: {info.embedding_tables or '(none)'}")
        click.echo("  - All ingestion history")

    if not yes:
        if not click.confirm("\nAre you sure you want to continue?"):
            click.echo("Aborted.")
            return

    try:
        asyncio.run(execute_purge(config_dict, embeddings_only, progress=click.echo))
    except Exception as e:
        _handle_db_error(e)
        raise


__all__ = ["ariel_group"]
