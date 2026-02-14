"""ARIEL CLI commands.

This module provides CLI commands for the ARIEL search service.

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


@click.group("ariel")
def ariel_group() -> None:
    """ARIEL search service commands.

    Commands for managing the ARIEL (Agentic Retrieval Interface for
    Electronic Logbooks) search service.
    """


@ariel_group.command("status")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def status_command(output_json: bool) -> None:
    """Show ARIEL service status.

    Displays database connection, embedding tables, and enhancement stats.
    """
    import json as json_module

    async def _get_status() -> dict:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service

        try:
            # Load config (get_config imported at module level)
            config_dict = get_config_value("ariel", {})
            if not config_dict:
                return {"status": "error", "message": "ARIEL not configured"}

            config = ARIELConfig.from_dict(config_dict)

            service = await create_ariel_service(config)
            async with service:
                # Health check
                healthy, message = await service.health_check()

                # Get enhancement stats
                stats = await service.repository.get_enhancement_stats()

                # Get embedding tables
                tables = await service.repository.get_embedding_tables()

                return {
                    "status": "healthy" if healthy else "unhealthy",
                    "message": message,
                    "database": {
                        "uri": config.database.uri.split("@")[-1]
                        if "@" in config.database.uri
                        else config.database.uri,
                        "connected": healthy,
                    },
                    "entries": stats.get("total_entries", 0),
                    "embedding_tables": [
                        {
                            "table": t.table_name,
                            "entries": t.entry_count,
                            "dimension": t.dimension,
                            "active": t.is_active,
                        }
                        for t in tables
                    ],
                    "enhancement_modules": {
                        "text_embedding": config.is_enhancement_module_enabled("text_embedding"),
                        "semantic_processor": config.is_enhancement_module_enabled(
                            "semantic_processor"
                        ),
                    },
                    "search_modules": {
                        "keyword": config.is_search_module_enabled("keyword"),
                        "semantic": config.is_search_module_enabled("semantic"),
                    },
                    "pipelines": {
                        "rag": config.is_pipeline_enabled("rag"),
                        "agent": config.is_pipeline_enabled("agent"),
                    },
                }

        except Exception as e:
            msg = str(e)
            if "connection" in msg.lower() or "connect" in msg.lower():
                return {
                    "status": "error",
                    "message": "Cannot connect to the ARIEL database. "
                    "Make sure the database is running: osprey deploy up",
                }
            return {"status": "error", "message": msg}

    result = asyncio.run(_get_status())

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

    async def _migrate() -> None:
        from osprey.services.ariel_search import ARIELConfig
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.migrate import run_migrations

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        config = ARIELConfig.from_dict(config_dict)

        click.echo(f"Connecting to database: {config.database.uri.split('@')[-1]}")

        try:
            pool = await create_connection_pool(config.database)
        except Exception as e:
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                click.echo("Error: Cannot connect to the ARIEL database.", err=True)
                click.echo("Make sure the database is running: osprey deploy up", err=True)
                raise SystemExit(1) from None
            raise

        try:
            click.echo("Running migrations...")
            await run_migrations(pool, config)
            click.echo("Migrations complete.")
        finally:
            await pool.close()

    asyncio.run(_migrate())


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

    async def _ingest() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service
        from osprey.services.ariel_search.enhancement import create_enhancers_from_config
        from osprey.services.ariel_search.ingestion import get_adapter

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        # Override source_url from command line
        if "ingestion" not in config_dict:
            config_dict["ingestion"] = {}
        config_dict["ingestion"]["source_url"] = source
        config_dict["ingestion"]["adapter"] = adapter

        config = ARIELConfig.from_dict(config_dict)

        # Get adapter
        adapter_instance = get_adapter(config)

        click.echo(f"Using adapter: {adapter_instance.source_system_name}")
        click.echo(f"Source: {source}")

        # Get enabled enhancement modules
        enhancers = create_enhancers_from_config(config)
        if enhancers:
            click.echo(f"Enhancement modules: {[e.name for e in enhancers]}")

        if dry_run:
            # Just count entries
            count = 0
            async for _entry in adapter_instance.fetch_entries(since=since, limit=limit):
                count += 1
                if count % 100 == 0:
                    click.echo(f"  Parsed {count} entries...")

            click.echo(f"\nDry run complete: {count} entries would be ingested")
            if enhancers:
                click.echo(f"Enhancement modules would run: {[e.name for e in enhancers]}")
            return

        # Full ingestion with enhancement and run tracking
        service = await create_ariel_service(config)
        async with service:
            source_system = adapter_instance.source_system_name
            run_id = await service.repository.start_ingestion_run(source_system)

            count = 0
            enhanced_count = 0
            failed_count = 0

            try:
                async with service.pool.connection() as conn:
                    async for entry in adapter_instance.fetch_entries(since=since, limit=limit):
                        # Store entry
                        await service.repository.upsert_entry(entry)
                        count += 1

                        # Run enabled enhancement modules
                        if enhancers:
                            for enhancer in enhancers:
                                try:
                                    await enhancer.enhance(entry, conn)
                                    await service.repository.mark_enhancement_complete(
                                        entry["entry_id"],
                                        enhancer.name,
                                    )
                                    enhanced_count += 1
                                except Exception as e:
                                    await service.repository.mark_enhancement_failed(
                                        entry["entry_id"],
                                        enhancer.name,
                                        str(e),
                                    )
                                    failed_count += 1

                        if count % 100 == 0:
                            if enhancers:
                                click.echo(f"  Ingested and enhanced {count} entries...")
                            else:
                                click.echo(f"  Ingested {count} entries...")

                await service.repository.complete_ingestion_run(
                    run_id,
                    entries_added=count,
                    entries_updated=0,
                    entries_failed=failed_count,
                )
            except Exception as e:
                await service.repository.fail_ingestion_run(run_id, str(e))
                raise

            click.echo(f"\nIngestion complete: {count} entries stored")
            if enhancers:
                click.echo(f"Enhancement complete: {enhanced_count} enhancements applied")

    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    try:
        asyncio.run(_ingest())
    except DatabaseQueryError as e:
        # Check for "relation does not exist" error indicating missing tables
        if "relation" in str(e) and "does not exist" in str(e):
            click.echo("Error: ARIEL database is not initialized.", err=True)
            click.echo("Run 'osprey ariel migrate' to create the required tables.", err=True)
            raise SystemExit(1) from None
        raise  # Re-raise other database errors
    except Exception as e:
        if "connection" in str(e).lower() or "connect" in str(e).lower():
            click.echo("Error: Cannot connect to the ARIEL database.", err=True)
            click.echo("Make sure the database is running: osprey deploy up", err=True)
            raise SystemExit(1) from None
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
    import signal

    async def _watch() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service
        from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

        # Load config
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        # Apply CLI overrides
        if source or adapter:
            if "ingestion" not in config_dict:
                config_dict["ingestion"] = {}
            if source:
                config_dict["ingestion"]["source_url"] = source
            if adapter:
                config_dict["ingestion"]["adapter"] = adapter

        if interval is not None:
            if "ingestion" not in config_dict:
                config_dict["ingestion"] = {}
            config_dict["ingestion"]["poll_interval_seconds"] = interval

        config = ARIELConfig.from_dict(config_dict)

        if not config.ingestion or not config.ingestion.source_url:
            click.echo(
                "Error: No ingestion source configured. "
                "Set ingestion.source_url in config.yml or use --source.",
                err=True,
            )
            raise SystemExit(1)

        service = await create_ariel_service(config)
        async with service:
            scheduler = IngestionScheduler(
                config=config,
                repository=service.repository,
            )

            if once:
                click.echo(f"Running single poll cycle (source: {config.ingestion.source_url})")
                result = await scheduler.poll_once(dry_run=dry_run)
                prefix = "[dry-run] " if dry_run else ""
                click.echo(
                    f"\n{prefix}Poll complete: "
                    f"{result.entries_added} added, "
                    f"{result.entries_failed} failed "
                    f"({result.duration_seconds:.1f}s)"
                )
                if result.since:
                    click.echo(f"  Since: {result.since.isoformat()}")
                return

            # Daemon mode with signal handling
            poll_secs = config.ingestion.poll_interval_seconds
            click.echo(f"Watching: {config.ingestion.source_url}")
            click.echo(f"Poll interval: {poll_secs}s")
            click.echo("Press Ctrl+C to stop\n")

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(scheduler.stop()))

            await scheduler.start()

    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    try:
        asyncio.run(_watch())
    except DatabaseQueryError as e:
        if "relation" in str(e) and "does not exist" in str(e):
            click.echo("Error: ARIEL database is not initialized.", err=True)
            click.echo("Run 'osprey ariel migrate' to create the required tables.", err=True)
            raise SystemExit(1) from None
        raise
    except KeyboardInterrupt:
        click.echo("\nStopping watcher...")
    except Exception as e:
        if "connection" in str(e).lower() or "connect" in str(e).lower():
            click.echo("Error: Cannot connect to the ARIEL database.", err=True)
            click.echo("Make sure the database is running: osprey deploy up", err=True)
            raise SystemExit(1) from None
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

    async def _enhance() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service
        from osprey.services.ariel_search.enhancement import create_enhancers_from_config

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        config = ARIELConfig.from_dict(config_dict)

        # Get enhancers
        enhancers = create_enhancers_from_config(config)
        if module:
            enhancers = [e for e in enhancers if e.name == module]

        if not enhancers:
            click.echo("No enhancement modules enabled or selected")
            return

        click.echo(f"Enhancement modules: {[e.name for e in enhancers]}")

        service = await create_ariel_service(config)
        async with service:
            # Get entries to enhance
            if force:
                entries = await service.repository.search_by_time_range(limit=limit)
            else:
                entries = await service.repository.get_incomplete_entries(
                    module_name=module,
                    limit=limit,
                )

            click.echo(f"Processing {len(entries)} entries...")

            async with service.pool.connection() as conn:
                for i, entry in enumerate(entries):
                    for enhancer in enhancers:
                        try:
                            await enhancer.enhance(entry, conn)
                            await service.repository.mark_enhancement_complete(
                                entry["entry_id"],
                                enhancer.name,
                            )
                        except Exception as e:
                            await service.repository.mark_enhancement_failed(
                                entry["entry_id"],
                                enhancer.name,
                                str(e),
                            )

                    if (i + 1) % 10 == 0:
                        click.echo(f"  Processed {i + 1} entries...")

            click.echo(f"\nEnhancement complete: {len(entries)} entries processed")

    asyncio.run(_enhance())


@ariel_group.command("models")
def models_command() -> None:
    """List embedding models and their tables.

    Shows all embedding tables in the database and their status.
    """

    async def _list_models() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        config = ARIELConfig.from_dict(config_dict)

        service = await create_ariel_service(config)
        async with service:
            tables = await service.repository.get_embedding_tables()

            if not tables:
                click.echo("No embedding tables found.")
                return

            click.echo("Embedding Models:")
            for table in tables:
                active = " (active)" if table.is_active else ""
                click.echo(f"\n  {table.table_name}{active}")
                click.echo(f"    Entries: {table.entry_count}")
                if table.dimension:
                    click.echo(f"    Dimension: {table.dimension}")

    asyncio.run(_list_models())


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

    async def _search() -> dict:
        from osprey.services.ariel_search import ARIELConfig, SearchMode, create_ariel_service

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            return {"error": "ARIEL not configured"}

        config = ARIELConfig.from_dict(config_dict)

        search_mode = None
        if mode != "auto":
            search_mode = SearchMode[mode.upper()]

        try:
            service = await create_ariel_service(config)
            async with service:
                result = await service.search(
                    query=query,
                    max_results=limit,
                    mode=search_mode,
                )

                return {
                    "query": query,
                    "answer": result.answer,
                    "sources": list(result.sources),
                    "search_modes": [m.value for m in result.search_modes_used],
                    "reasoning": result.reasoning,
                }
        except Exception as e:
            msg = str(e)
            if "connection" in msg.lower() or "connect" in msg.lower():
                return {
                    "error": "Cannot connect to the ARIEL database. "
                    "Make sure the database is running: osprey deploy up"
                }
            if "relation" in msg and "does not exist" in msg:
                return {
                    "error": "Logbook database tables not found. "
                    "Run 'osprey ariel migrate' to create tables, then "
                    "'osprey ariel ingest' to populate data."
                }
            return {"error": msg}

    result = asyncio.run(_search())

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

    async def _reembed() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service
        from osprey.services.ariel_search.database.migration import model_to_table_name
        from osprey.services.ariel_search.enhancement.text_embedding import (
            TextEmbeddingMigration,
        )

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        config = ARIELConfig.from_dict(config_dict)
        table_name = model_to_table_name(model)

        if dry_run:
            click.echo(f"DRY RUN - Would re-embed entries using model: {model}")
            click.echo(f"  Table: {table_name}")
            click.echo(f"  Dimension: {dimension}")
            click.echo(f"  Batch size: {batch_size}")
            click.echo(f"  Force overwrite: {force}")
            return

        service = await create_ariel_service(config)
        async with service:
            # Check if table exists
            tables = await service.repository.get_embedding_tables()
            table_exists = any(t.table_name == table_name for t in tables)

            if not table_exists:
                click.echo(f"Creating embedding table: {table_name}")
                # Create the migration and run it
                migration = TextEmbeddingMigration([(model, dimension)])
                async with service.pool.connection() as conn:
                    await migration.up(conn)
                click.echo(f"  Table created: {table_name}")

            # Get entries to embed
            entry_count = await service.repository.count_entries()
            click.echo(f"Found {entry_count} entries to embed")

            if entry_count == 0:
                click.echo("No entries to embed.")
                return

            # Get embedding provider
            from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

            embedder = OllamaEmbeddingProvider()
            base_url = getattr(config.embedding, "base_url", None) or embedder.default_base_url

            # Process entries in batches
            processed = 0
            skipped = 0
            errors = 0

            async with service.pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Get all entries
                    await cur.execute(
                        "SELECT entry_id, raw_text FROM enhanced_entries ORDER BY entry_id"
                    )
                    rows = await cur.fetchall()

                    batch_texts = []
                    batch_ids = []

                    for entry_id, raw_text in rows:
                        # Check if embedding already exists (unless force)
                        if not force:
                            await cur.execute(
                                f"SELECT 1 FROM {table_name} WHERE entry_id = %s",  # noqa: S608
                                (entry_id,),
                            )
                            if await cur.fetchone():
                                skipped += 1
                                continue

                        batch_texts.append(raw_text or "")
                        batch_ids.append(entry_id)

                        if len(batch_texts) >= batch_size:
                            # Process batch
                            try:
                                embeddings = embedder.execute_embedding(
                                    texts=batch_texts,
                                    model_id=model,
                                    base_url=base_url,
                                )

                                for eid, emb in zip(batch_ids, embeddings, strict=True):
                                    if force:
                                        await cur.execute(
                                            f"""
                                            INSERT INTO {table_name} (entry_id, embedding)
                                            VALUES (%s, %s)
                                            ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding
                                            """,  # noqa: S608
                                            (eid, emb),
                                        )
                                    else:
                                        await cur.execute(
                                            f"""
                                            INSERT INTO {table_name} (entry_id, embedding)
                                            VALUES (%s, %s)
                                            ON CONFLICT (entry_id) DO NOTHING
                                            """,  # noqa: S608
                                            (eid, emb),
                                        )
                                processed += len(batch_ids)
                                click.echo(f"  Processed {processed} entries...")
                            except Exception as e:
                                click.echo(f"  Error in batch: {e}", err=True)
                                errors += len(batch_ids)

                            batch_texts = []
                            batch_ids = []

                    # Process remaining batch
                    if batch_texts:
                        try:
                            embeddings = embedder.execute_embedding(
                                texts=batch_texts,
                                model_id=model,
                                base_url=base_url,
                            )

                            for eid, emb in zip(batch_ids, embeddings, strict=True):
                                if force:
                                    await cur.execute(
                                        f"""
                                        INSERT INTO {table_name} (entry_id, embedding)
                                        VALUES (%s, %s)
                                        ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding
                                        """,  # noqa: S608
                                        (eid, emb),
                                    )
                                else:
                                    await cur.execute(
                                        f"""
                                        INSERT INTO {table_name} (entry_id, embedding)
                                        VALUES (%s, %s)
                                        ON CONFLICT (entry_id) DO NOTHING
                                        """,  # noqa: S608
                                        (eid, emb),
                                    )
                            processed += len(batch_ids)
                        except Exception as e:
                            click.echo(f"  Error in final batch: {e}", err=True)
                            errors += len(batch_ids)

            click.echo("\nRe-embedding complete:")
            click.echo(f"  Processed: {processed}")
            click.echo(f"  Skipped (existing): {skipped}")
            click.echo(f"  Errors: {errors}")

    asyncio.run(_reembed())


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

    async def _quickstart() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.migrate import run_migrations
        from osprey.services.ariel_search.ingestion import get_adapter

        # 1. Load config
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            click.echo("Add an 'ariel:' section to your config.yml file.", err=True)
            raise SystemExit(1)

        # Override source if provided via CLI
        if source:
            if "ingestion" not in config_dict:
                config_dict["ingestion"] = {}
            config_dict["ingestion"]["source_url"] = source
            config_dict["ingestion"]["adapter"] = "generic_json"

        config = ARIELConfig.from_dict(config_dict)

        # 2. Check database connection
        click.echo("Checking database connection...")
        try:
            pool = await create_connection_pool(config.database)
        except Exception as e:
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                click.echo("\nError: Cannot connect to the ARIEL database.", err=True)
                click.echo("Start it with: osprey deploy up", err=True)
                click.echo("Then re-run: osprey ariel quickstart", err=True)
                raise SystemExit(1) from None
            raise
        click.echo("  Database: connected")

        try:
            # 3. Run migrations
            click.echo("Running migrations...")
            applied = await run_migrations(pool, config)
            if applied:
                click.echo(f"  Tables: created ({len(applied)} migrations applied)")
            else:
                click.echo("  Tables: already up to date")

            # 4. Ingest data
            if not config.ingestion or not config.ingestion.source_url:
                click.echo("\nNo ingestion source configured. Skipping data ingestion.")
                click.echo(
                    "Set ingestion.source_url in config.yml or use --source flag.",
                    err=True,
                )
            else:
                click.echo(f"Ingesting data from: {config.ingestion.source_url}")
                adapter_instance = get_adapter(config)

                from osprey.services.ariel_search.enhancement import (
                    create_enhancers_from_config,
                )

                enhancers = create_enhancers_from_config(config)
                if enhancers:
                    click.echo(f"  Enhancement modules: {[e.name for e in enhancers]}")

                service = await create_ariel_service(config)
                async with service:
                    count = 0
                    enhanced_count = 0
                    failed_count = 0

                    async with service.pool.connection() as conn:
                        async for entry in adapter_instance.fetch_entries():
                            await service.repository.upsert_entry(entry)
                            count += 1

                            if enhancers:
                                for enhancer in enhancers:
                                    try:
                                        await enhancer.enhance(entry, conn)
                                        await service.repository.mark_enhancement_complete(
                                            entry["entry_id"],
                                            enhancer.name,
                                        )
                                        enhanced_count += 1
                                    except Exception as e:
                                        await service.repository.mark_enhancement_failed(
                                            entry["entry_id"],
                                            enhancer.name,
                                            str(e),
                                        )
                                        failed_count += 1
                                        logger.debug(
                                            f"Enhancement failed for {entry['entry_id']}: {e}"
                                        )

                    click.echo(f"  Entries: {count} ingested")
                    if enhancers:
                        click.echo(
                            f"  Enhancements: {enhanced_count} applied"
                            + (f", {failed_count} failed" if failed_count else "")
                        )

            # 5. Summary
            enabled_search = config.get_enabled_search_modules()
            click.echo(
                f"\nARIEL quickstart complete!"
                f"\n  Search modules: {', '.join(enabled_search) or 'none'}"
            )
            click.echo('\nTry it: osprey ariel search "What happened with the RF cavity?"')

        finally:
            await pool.close()

    asyncio.run(_quickstart())


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
    # Check if ARIEL is configured
    config_dict = get_config_value("ariel", {})
    if not config_dict:
        click.echo("Error: ARIEL not configured in config.yml", err=True)
        click.echo("Add an 'ariel:' section to your config.yml file.", err=True)
        raise SystemExit(1)

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

    async def _purge() -> None:
        from osprey.services.ariel_search import ARIELConfig
        from osprey.services.ariel_search.database.connection import create_connection_pool

        # Load config (get_config imported at module level)
        config_dict = get_config_value("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        config = ARIELConfig.from_dict(config_dict)

        # Get current counts for display
        pool = await create_connection_pool(config.database)

        try:
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    # Get entry count
                    await cur.execute("SELECT COUNT(*) FROM enhanced_entries")
                    row = await cur.fetchone()
                    entry_count = row[0] if row else 0

                    # Get embedding tables
                    await cur.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                    """)
                    embedding_tables = [r[0] for r in await cur.fetchall()]

            # Show what will be deleted
            click.echo("\n⚠️  WARNING: This will permanently delete:")
            if embeddings_only:
                click.echo(f"  - Embedding tables: {embedding_tables or '(none)'}")
                click.echo(f"  - Entries will be KEPT ({entry_count} entries)")
            else:
                click.echo(f"  - All {entry_count} logbook entries")
                click.echo(f"  - All embedding tables: {embedding_tables or '(none)'}")
                click.echo("  - All ingestion history")

            # Confirm
            if not yes:
                if not click.confirm("\nAre you sure you want to continue?"):
                    click.echo("Aborted.")
                    return

            # Perform purge
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    if embeddings_only:
                        # Only drop embedding tables
                        for table in embedding_tables:
                            await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")  # noqa: S608
                            click.echo(f"  Dropped {table}")
                        click.echo("\n✓ Embedding tables purged. Entries preserved.")
                    else:
                        # Full purge - truncate all tables
                        await cur.execute("TRUNCATE enhanced_entries CASCADE")
                        await cur.execute("TRUNCATE ingestion_runs CASCADE")
                        for table in embedding_tables:
                            await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")  # noqa: S608
                        click.echo("\n✓ All ARIEL data purged.")

        finally:
            await pool.close()

    asyncio.run(_purge())


__all__ = ["ariel_group"]
