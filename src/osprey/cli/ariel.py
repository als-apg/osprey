"""ARIEL CLI commands.

Thin CLI wrappers that delegate business logic to
``osprey.services.ariel_search.cli_operations``.

See 04_OSPREY_INTEGRATION.md Sections 13 for specification.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING

import click

# Import get_config_value at module level for easier patching in tests
from osprey.utils.config import get_config_value
from osprey.utils.logger import get_logger

from . import output

logger = get_logger("ariel")

if TYPE_CHECKING:
    from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _emit_json_document(payload: object) -> None:
    """Write *payload* to stdout as one JSON document and nothing else.

    This module's machine-output seam, in the shape
    :func:`osprey.health.render.render_json` established: it writes at the
    stream rather than through the renderer, so no human line can reach the
    document. Everything human a ``--json`` run produces goes to stderr
    instead, because the caller holds
    :func:`~osprey.cli.output.machine_mode` open around the whole body.

    :param payload: The already-assembled result to serialize.
    """
    sys.stdout.write(json.dumps(payload, indent=2))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _load_ariel_config() -> dict:
    """Load ARIEL config dict, raising SystemExit if missing."""
    config_dict = get_config_value("ariel", {})
    if not config_dict:
        output.fail(
            "ARIEL is not configured in config.yml",
            None,
            "add an `ariel:` section to config.yml, then run `osprey build`",
        )
        raise SystemExit(1)
    return config_dict


def _handle_db_error(e: Exception) -> None:
    """Raise SystemExit on database connection errors, otherwise return."""
    msg = str(e)
    if "connection" in msg.lower() or "connect" in msg.lower():
        output.fail(
            "cannot connect to the ARIEL database",
            None,
            "start the database with `osprey up`",
        )
        raise SystemExit(1) from None


def _framework_search_modes() -> tuple[str, ...]:
    """Return the search module names the framework registers by default."""
    from osprey.registry.builtins import FrameworkRegistryProvider

    config = FrameworkRegistryProvider().get_registry_config()
    return tuple(registration.name for registration in config.ariel_search_modules)


def _registered_search_modes() -> tuple[str, ...]:
    """Return the ARIEL search module names the registry knows about.

    The project registry is authoritative, so a deployment that registers its
    own search module gets it as a ``--mode`` choice without a code change.
    Building that registry needs a project ``config.yml``; when there is none
    (``--help`` run outside a project directory, say) the framework's own
    baseline registrations stand in. That failure is expected here, so the
    registry loggers are muted while it is probed — a help screen is not the
    place to report a missing project config.

    Returns:
        Search module names, in registry order.
    """
    import logging

    muted = {name: logging.getLogger(name) for name in ("registry", "registry.loader")}
    previous_levels = {name: log.level for name, log in muted.items()}
    try:
        from osprey.registry import get_registry

        for log in muted.values():
            log.setLevel(logging.CRITICAL)
        try:
            names = tuple(module.name for module in get_registry().config.ariel_search_modules)
        finally:
            for name, log in muted.items():
                log.setLevel(previous_levels[name])
        if names:
            return names
    except Exception:
        logger.debug("Registry unavailable; using framework search modules", exc_info=True)

    return _framework_search_modes()


class _SearchModeChoice(click.Choice):
    """Click choice type whose options come from the registry when parsed.

    ``click.Choice`` freezes its options when the decorator runs, which is
    import time — too early to reach the registry, since building one requires
    a project config. Resolving on attribute access defers the lookup to
    parsing and help rendering.
    """

    def __init__(self) -> None:
        """Build a choice type with no fixed option list."""
        self.case_sensitive = True

    @property
    def choices(self) -> tuple[str, ...]:  # type: ignore[override]
        """Registered search module names, resolved on each access."""
        return _registered_search_modes()


def _handle_missing_tables(e: Exception) -> None:
    """Raise SystemExit on missing-table errors, otherwise return."""
    msg = str(e)
    if "relation" in msg and "does not exist" in msg:
        output.fail(
            "the ARIEL database is not initialized",
            None,
            "run `osprey ariel migrate` to create the required tables",
        )
        raise SystemExit(1) from None


def _qmd_resync_pre_step(config_dict: dict) -> None:
    """Bring the qmd markdown mirror up to date before ingesting.

    Entries written straight to Postgres never reach an enhancer, so without
    this pass their content would sit in the database unmirrored until the next
    time something happened to touch them. Failures are reported and swallowed:
    ingestion's job is the database, and a mirror that cannot be written is not
    a reason to abandon it.

    Args:
        config_dict: Raw ``ariel`` config section.
    """
    from osprey.services.ariel_search.cli_operations import resync_qmd_mirror_best_effort

    asyncio.run(resync_qmd_mirror_best_effort(config_dict, progress=output.report))


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
    from osprey.services.ariel_search.cli_operations import get_status

    config_dict = get_config_value("ariel", {})

    # Under ``--json`` the whole body runs in machine mode, so every renderer
    # line goes to stderr and stdout carries one document for a script to parse.
    with output.machine_mode() if output_json else nullcontext():
        result = asyncio.run(get_status(config_dict))

        if output_json:
            _emit_json_document(result)
            return

        output.report(f"ARIEL Status: {result['status']}")
        output.note(result["message"])
        if result["status"] != "error":
            output.report("")
            output.section(
                "",
                {
                    "Database": result["database"]["uri"],
                    "Total entries": result["entries"],
                },
            )
            output.report("")
            output.report("Embedding tables:")
            for table in result.get("embedding_tables", []):
                active = " (active)" if table["active"] else ""
                output.note(f"- {table['table']}: {table['entries']} entries{active}")


@ariel_group.command("migrate")
def migrate_command() -> None:
    """Run ARIEL database migrations.

    Creates required database schema and tables based on enabled modules.
    """
    from osprey.services.ariel_search.cli_operations import run_migrate

    config_dict = _load_ariel_config()
    try:
        asyncio.run(run_migrate(config_dict, progress=output.report))
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
        result = asyncio.run(run_sync(config_dict, limit=limit, progress=output.report))
        output.report("")
        output.report(
            f"Sync complete: "
            f"{result.entries_ingested} ingested, "
            f"{result.entries_enhanced} enhanced, "
            f"{result.entries_failed} failed"
        )
        if result.migrations_applied:
            output.note(f"Migrations applied: {result.migrations_applied}")
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
    _qmd_resync_pre_step(config_dict)
    try:
        result = asyncio.run(
            run_ingest(config_dict, source, adapter, since, limit, dry_run, progress=output.report)
        )
        output.report("")
        if result.dry_run:
            output.report(f"Dry run complete: {result.count} entries would be ingested")
            if result.enhancer_names:
                output.note(f"Enhancement modules would run: {result.enhancer_names}")
        else:
            output.report(f"Ingestion complete: {result.count} entries stored")
            if result.enhancer_names:
                output.note(f"Enhancement complete: {result.enhanced_count} enhancements applied")
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
    # Covers the single --once cycle; in daemon mode run_watch repeats this
    # pre-step at the head of every poll the scheduler makes.
    _qmd_resync_pre_step(config_dict)
    try:
        result = asyncio.run(
            run_watch(config_dict, source, adapter, once, interval, dry_run, progress=output.report)
        )
        if result is not None:
            prefix = "[dry-run] " if result.dry_run else ""
            output.report("")
            output.report(
                f"{prefix}Poll complete: "
                f"{result.entries_added} added, "
                f"{result.entries_updated} updated, "
                f"{result.entries_failed} failed "
                f"({result.duration_seconds:.1f}s)"
            )
            if result.since:
                output.note(f"Since: {result.since.isoformat()}")
    except ValueError as e:
        output.fail(str(e))
        raise SystemExit(1) from None
    except DatabaseQueryError as e:
        _handle_missing_tables(e)
        raise
    except KeyboardInterrupt:
        output.report("")
        output.report("Stopping the watcher.")
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("enhance")
@click.option(
    "--module",
    "-m",
    type=click.Choice(["text_embedding", "semantic_processor", "qmd_export"]),
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
    result = asyncio.run(run_enhance(config_dict, module, force, limit, progress=output.report))
    if result.entries_processed > 0:
        output.report("")
        output.report(f"Enhancement complete: {result.entries_processed} entries processed")


@ariel_group.command("models")
def models_command() -> None:
    """List embedding models and their tables.

    Shows all embedding tables in the database and their status.
    """
    from osprey.services.ariel_search.cli_operations import list_models

    config_dict = _load_ariel_config()
    tables = asyncio.run(list_models(config_dict))

    if not tables:
        output.report("No embedding tables found.")
        return

    output.report("Embedding models:")
    for table in tables:
        active = " (active)" if table["is_active"] else ""
        items: dict[str, object] = {"Entries": table["entry_count"]}
        if table["dimension"]:
            items["Dimension"] = table["dimension"]
        output.report("")
        output.section(f"{table['table_name']}{active}", items)


@ariel_group.command("search")
@click.argument("query")
@click.option("--mode", type=_SearchModeChoice(), default="keyword")
@click.option("--limit", type=int, default=10, help="Maximum results")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def search_command(query: str, mode: str, limit: int, output_json: bool) -> None:
    """Search the logbook.

    Execute a search query using the ARIEL agent.
    """
    from osprey.services.ariel_search.cli_operations import run_search

    config_dict = get_config_value("ariel", {})

    # Under ``--json`` the whole body runs in machine mode, so every renderer
    # line goes to stderr and stdout carries one document for a script to parse.
    with output.machine_mode() if output_json else nullcontext():
        result = asyncio.run(run_search(config_dict, query, mode, limit))

        if output_json:
            _emit_json_document(result)
            return

        if result.get("error"):
            output.fail(str(result["error"]))
            return

        output.report(f"Query: {result['query']}")
        output.report(f"Modes: {', '.join(result['search_modes']) or 'none'}")
        output.report("")

        if result["answer"]:
            output.report(result["answer"])
            if result["sources"]:
                output.report("")
                output.report(f"Sources: {', '.join(result['sources'])}")
        elif result.get("entries"):
            # Direct (non-RAG) modes return entries without a composed answer.
            for idx, entry in enumerate(result["entries"], 1):
                timestamp = entry.get("timestamp", "")[:16].replace("T", " ")
                header = f"{idx}. [{entry.get('entry_id', '?')}] {entry.get('title', '')}"
                output.report(header)
                byline = "   ".join(part for part in (timestamp, entry.get("author", "")) if part)
                if byline:
                    output.note(byline)
        else:
            output.report("No results found.")


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
        run_reembed(
            config_dict, model, dimension, batch_size, dry_run, force, progress=output.report
        )
    )
    if not result.dry_run:
        output.report("")
        output.section(
            "Re-embedding complete:",
            {
                "Processed": result.processed,
                "Skipped (existing)": result.skipped,
                "Errors": result.errors,
            },
        )


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
    1. Checks database connection (prompts to run 'osprey up' if down)
    2. Runs database migrations
    3. Ingests demo logbook data (or custom source)

    Example:
        osprey ariel quickstart                    # Use bundled demo data
        osprey ariel quickstart -s my_logbook.json # Use custom data
    """
    from osprey.services.ariel_search.cli_operations import run_quickstart

    config_dict = _load_ariel_config()
    try:
        asyncio.run(run_quickstart(config_dict, source, progress=output.report))
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

    output.report(f"Starting ARIEL Web Interface on http://{host}:{port}")
    output.note("Press Ctrl+C to stop")
    output.report("")

    try:
        from osprey.interfaces.ariel import run_web

        run_web(host=host, port=port, reload=reload)
    except KeyboardInterrupt:
        output.report("")
        output.report("Shutting down...")


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

    if embeddings_only:
        detail = (
            f"embedding tables: {info.embedding_tables or '(none)'}\n"
            f"the {info.entry_count} logbook entries are kept"
        )
    else:
        detail = (
            f"all {info.entry_count} logbook entries\n"
            f"all embedding tables: {info.embedding_tables or '(none)'}\n"
            "all ingestion history"
        )
    output.warn("this deletes ARIEL data for good", detail)

    if not yes:
        if not click.confirm("\nAre you sure you want to continue?"):
            output.report("Nothing was deleted.")
            return

    try:
        asyncio.run(execute_purge(config_dict, embeddings_only, progress=output.report))
    except Exception as e:
        _handle_db_error(e)
        raise


@ariel_group.command("qmd-resync")
@click.option("--rebuild", is_flag=True, help="Wipe the mirror and re-export every entry")
def qmd_resync_command(rebuild: bool) -> None:
    """Re-export logbook entries the markdown mirror never saw.

    The qmd sidecar searches a markdown mirror of the logbook, and that mirror
    is normally written by the qmd_export enhancement module as entries are
    ingested. Three mutation paths write straight to the database and never
    reach an enhancer, so their content would otherwise never be searchable:

    \b
      - creating a local entry in the ARIEL web interface
        (interfaces/ariel/api/routes.py:395)
      - re-upserting an entry when an attachment is uploaded
        (interfaces/ariel/api/routes.py:533)
      - entry_create upserts from the logbook write service
        (services/ariel_search/service.py:431 and :439)

    This command finds every entry changed since the last run and re-exports
    it. Entries whose content is unchanged are left alone, so a run that finds
    only bookkeeping updates writes nothing and costs the sidecar nothing.

    Ingest and watch already run this pass before their own work, so a routine
    deployment never needs to run it by hand. Reach for --rebuild after
    'osprey ariel purge' or any other wholesale change: it is the only pass
    that clears mirrored files for entries that no longer exist.

    \b
    Example:
        osprey ariel qmd-resync              # Catch up on recent changes
        osprey ariel qmd-resync --rebuild    # Rebuild the mirror from scratch
    """
    from osprey.services.ariel_search.cli_operations import run_qmd_resync

    config_dict = _load_ariel_config()
    try:
        result = asyncio.run(run_qmd_resync(config_dict, rebuild=rebuild, progress=output.report))
    except ValueError as e:
        output.fail(
            "the qmd markdown mirror has nowhere to write",
            str(e),
            "set ariel.enhancement_modules.qmd_export.settings.mirror_path in config.yml",
        )
        raise SystemExit(1) from None
    except Exception as e:
        _handle_db_error(e)
        _handle_missing_tables(e)
        raise

    if result is None:
        output.report("The qmd_export enhancement module is not enabled; nothing to resync")
        return

    rows: list[tuple[str, object]] = []
    if result.rebuild:
        rows.append(("Removed before rebuild", result.removed))
    rows.extend(
        [
            ("Scanned", result.scanned),
            ("Written", result.written),
            ("Unchanged", result.unchanged),
        ]
    )
    if result.failed:
        rows.append(("Failed", result.failed))

    output.report("")
    output.section(f"Mirror: {result.mirror_path}", rows)


__all__ = ["ariel_group"]
