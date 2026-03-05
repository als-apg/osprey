"""Business logic for ARIEL CLI commands.

Extracted from ``osprey.cli.ariel`` so that the CLI handlers are thin
wrappers around these service-layer functions.  Each function accepts a
raw config dict (from ``get_config_value("ariel", {})``) and returns a
structured result; the CLI layer handles Click decorators, output
formatting, and ``SystemExit`` translation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    count: int
    enhanced_count: int
    failed_count: int
    dry_run: bool
    enhancer_names: list[str] = field(default_factory=list)


@dataclass
class WatchOnceResult:
    entries_added: int
    entries_failed: int
    duration_seconds: float
    since: datetime | None
    dry_run: bool


@dataclass
class EnhanceResult:
    entries_processed: int
    module_names: list[str]


@dataclass
class ReembedResult:
    processed: int
    skipped: int
    errors: int
    dry_run: bool


@dataclass
class QuickstartResult:
    count: int
    enhanced_count: int
    failed_count: int
    migrations_applied: int
    enabled_search: list[str]


@dataclass
class PurgeInfo:
    entry_count: int
    embedding_tables: list[str]


# ---------------------------------------------------------------------------
# Service functions
# ---------------------------------------------------------------------------

_ProgressCb = Callable[[str], None] | None


async def get_status(config_dict: dict) -> dict:
    """Return ARIEL service status as a plain dict."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service

    if not config_dict:
        return {"status": "error", "message": "ARIEL not configured"}

    try:
        config = ARIELConfig.from_dict(config_dict)
        service = await create_ariel_service(config)
        async with service:
            healthy, message = await service.health_check()
            stats = await service.repository.get_enhancement_stats()
            tables = await service.repository.get_embedding_tables()

            return {
                "status": "healthy" if healthy else "unhealthy",
                "message": message,
                "database": {
                    "uri": (
                        config.database.uri.split("@")[-1]
                        if "@" in config.database.uri
                        else config.database.uri
                    ),
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


async def run_migrate(
    config_dict: dict,
    progress: _ProgressCb = None,
) -> None:
    """Run database migrations."""
    from osprey.services.ariel_search import ARIELConfig
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations

    config = ARIELConfig.from_dict(config_dict)

    if progress:
        progress(f"Connecting to database: {config.database.uri.split('@')[-1]}")

    pool = await create_connection_pool(config.database)

    try:
        if progress:
            progress("Running migrations...")
        await run_migrations(pool, config)
        if progress:
            progress("Migrations complete.")
    finally:
        await pool.close()


async def run_ingest(
    config_dict: dict,
    source: str,
    adapter: str,
    since: datetime | None,
    limit: int | None,
    dry_run: bool,
    progress: _ProgressCb = None,
) -> IngestResult:
    """Ingest logbook entries from a source."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.ingestion import get_adapter

    if "ingestion" not in config_dict:
        config_dict["ingestion"] = {}
    config_dict["ingestion"]["source_url"] = source
    config_dict["ingestion"]["adapter"] = adapter

    config = ARIELConfig.from_dict(config_dict)
    adapter_instance = get_adapter(config)

    if progress:
        progress(f"Using adapter: {adapter_instance.source_system_name}")
        progress(f"Source: {source}")

    enhancers = create_enhancers_from_config(config)
    enhancer_names = [e.name for e in enhancers]
    if enhancers and progress:
        progress(f"Enhancement modules: {enhancer_names}")

    if dry_run:
        count = 0
        async for _entry in adapter_instance.fetch_entries(since=since, limit=limit):
            count += 1
            if count % 100 == 0 and progress:
                progress(f"  Parsed {count} entries...")
        return IngestResult(
            count=count,
            enhanced_count=0,
            failed_count=0,
            dry_run=True,
            enhancer_names=enhancer_names,
        )

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

                    if count % 100 == 0 and progress:
                        if enhancers:
                            progress(f"  Ingested and enhanced {count} entries...")
                        else:
                            progress(f"  Ingested {count} entries...")

            await service.repository.complete_ingestion_run(
                run_id,
                entries_added=count,
                entries_updated=0,
                entries_failed=failed_count,
            )
        except Exception as e:
            await service.repository.fail_ingestion_run(run_id, str(e))
            raise

    return IngestResult(
        count=count,
        enhanced_count=enhanced_count,
        failed_count=failed_count,
        dry_run=False,
        enhancer_names=enhancer_names,
    )


async def run_watch(
    config_dict: dict,
    source: str | None,
    adapter: str | None,
    once: bool,
    interval: int | None,
    dry_run: bool,
    progress: _ProgressCb = None,
) -> WatchOnceResult | None:
    """Watch a source for new logbook entries.

    Returns a ``WatchOnceResult`` when *once* is ``True``.
    In daemon mode runs until stopped and returns ``None``.
    """
    import asyncio
    import signal

    from osprey.services.ariel_search import ARIELConfig, create_ariel_service
    from osprey.services.ariel_search.ingestion.scheduler import IngestionScheduler

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
        raise ValueError(
            "No ingestion source configured. "
            "Set ingestion.source_url in config.yml or use --source."
        )

    service = await create_ariel_service(config)
    async with service:
        scheduler = IngestionScheduler(
            config=config,
            repository=service.repository,
        )

        if once:
            if progress:
                progress(f"Running single poll cycle (source: {config.ingestion.source_url})")

            result = await scheduler.poll_once(dry_run=dry_run)

            return WatchOnceResult(
                entries_added=result.entries_added,
                entries_failed=result.entries_failed,
                duration_seconds=result.duration_seconds,
                since=result.since,
                dry_run=dry_run,
            )

        # Daemon mode
        poll_secs = config.ingestion.poll_interval_seconds
        if progress:
            progress(f"Watching: {config.ingestion.source_url}")
            progress(f"Poll interval: {poll_secs}s")
            progress("Press Ctrl+C to stop\n")

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(scheduler.stop()))

        await scheduler.start()
        return None


async def run_enhance(
    config_dict: dict,
    module: str | None,
    force: bool,
    limit: int,
    progress: _ProgressCb = None,
) -> EnhanceResult:
    """Run enhancement modules on entries."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config

    config = ARIELConfig.from_dict(config_dict)
    enhancers = create_enhancers_from_config(config)
    if module:
        enhancers = [e for e in enhancers if e.name == module]

    if not enhancers:
        if progress:
            progress("No enhancement modules enabled or selected")
        return EnhanceResult(entries_processed=0, module_names=[])

    module_names = [e.name for e in enhancers]
    if progress:
        progress(f"Enhancement modules: {module_names}")

    service = await create_ariel_service(config)
    async with service:
        if force:
            entries = await service.repository.search_by_time_range(limit=limit)
        else:
            entries = await service.repository.get_incomplete_entries(
                module_name=module,
                limit=limit,
            )

        if progress:
            progress(f"Processing {len(entries)} entries...")

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

                if (i + 1) % 10 == 0 and progress:
                    progress(f"  Processed {i + 1} entries...")

    return EnhanceResult(entries_processed=len(entries), module_names=module_names)


async def list_models(config_dict: dict) -> list[dict]:
    """Return embedding model info as a list of dicts."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service

    config = ARIELConfig.from_dict(config_dict)
    service = await create_ariel_service(config)
    async with service:
        tables = await service.repository.get_embedding_tables()
        return [
            {
                "table_name": t.table_name,
                "entry_count": t.entry_count,
                "dimension": t.dimension,
                "is_active": t.is_active,
            }
            for t in tables
        ]


async def run_search(config_dict: dict, query: str, mode: str, limit: int) -> dict:
    """Execute a search query and return the result as a dict."""
    from osprey.services.ariel_search import ARIELConfig, SearchMode, create_ariel_service

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


async def run_reembed(
    config_dict: dict,
    model: str,
    dimension: int,
    batch_size: int,
    dry_run: bool,
    force: bool,
    progress: _ProgressCb = None,
) -> ReembedResult:
    """Re-embed entries with a new or existing model."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service
    from osprey.services.ariel_search.database.migrations import model_to_table_name
    from osprey.services.ariel_search.enhancement.text_embedding import TextEmbeddingMigration

    config = ARIELConfig.from_dict(config_dict)
    table_name = model_to_table_name(model)

    if dry_run:
        if progress:
            progress(f"DRY RUN - Would re-embed entries using model: {model}")
            progress(f"  Table: {table_name}")
            progress(f"  Dimension: {dimension}")
            progress(f"  Batch size: {batch_size}")
            progress(f"  Force overwrite: {force}")
        return ReembedResult(processed=0, skipped=0, errors=0, dry_run=True)

    service = await create_ariel_service(config)
    async with service:
        tables = await service.repository.get_embedding_tables()
        table_exists = any(t.table_name == table_name for t in tables)

        if not table_exists:
            if progress:
                progress(f"Creating embedding table: {table_name}")
            migration = TextEmbeddingMigration([(model, dimension)])
            async with service.pool.connection() as conn:
                await migration.up(conn)
            if progress:
                progress(f"  Table created: {table_name}")

        entry_count = await service.repository.count_entries()
        if progress:
            progress(f"Found {entry_count} entries to embed")

        if entry_count == 0:
            if progress:
                progress("No entries to embed.")
            return ReembedResult(processed=0, skipped=0, errors=0, dry_run=False)

        from osprey.models.embeddings import get_embedding_provider

        embedder = get_embedding_provider(config.embedding.provider)
        base_url = getattr(config.embedding, "base_url", None) or embedder.default_base_url

        processed = 0
        skipped = 0
        errors = 0

        async with service.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT entry_id, raw_text FROM enhanced_entries ORDER BY entry_id"
                )
                rows = await cur.fetchall()

                batch_texts: list[str] = []
                batch_ids: list[str] = []

                for entry_id, raw_text in rows:
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
                        p, e = await _embed_batch(
                            cur,
                            embedder,
                            batch_texts,
                            batch_ids,
                            model,
                            base_url,
                            table_name,
                            force,
                            progress,
                        )
                        processed += p
                        errors += e
                        batch_texts = []
                        batch_ids = []

                if batch_texts:
                    p, e = await _embed_batch(
                        cur,
                        embedder,
                        batch_texts,
                        batch_ids,
                        model,
                        base_url,
                        table_name,
                        force,
                        progress,
                    )
                    processed += p
                    errors += e

    return ReembedResult(processed=processed, skipped=skipped, errors=errors, dry_run=False)


async def _embed_batch(
    cur,
    embedder,
    batch_texts: list[str],
    batch_ids: list[str],
    model: str,
    base_url: str,
    table_name: str,
    force: bool,
    progress: _ProgressCb,
) -> tuple[int, int]:
    """Embed a batch of texts and upsert into the table. Returns (processed, errors)."""
    try:
        embeddings = embedder.execute_embedding(
            texts=batch_texts,
            model_id=model,
            base_url=base_url,
        )

        conflict_clause = (
            "ON CONFLICT (entry_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            if force
            else "ON CONFLICT (entry_id) DO NOTHING"
        )
        for eid, emb in zip(batch_ids, embeddings, strict=True):
            await cur.execute(
                f"""
                INSERT INTO {table_name} (entry_id, embedding)
                VALUES (%s, %s)
                {conflict_clause}
                """,  # noqa: S608
                (eid, emb),
            )
        if progress:
            # Use cumulative count -- caller tracks total
            progress(f"  Processed {len(batch_ids)} entries in batch...")
        return len(batch_ids), 0
    except Exception as e:
        if progress:
            progress(f"  Error in batch: {e}")
        return 0, len(batch_ids)


async def run_quickstart(
    config_dict: dict,
    source: str | None,
    progress: _ProgressCb = None,
) -> QuickstartResult:
    """Run the complete ARIEL quickstart sequence."""
    from osprey.services.ariel_search import ARIELConfig, create_ariel_service
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.migrations import run_migrations
    from osprey.services.ariel_search.enhancement import create_enhancers_from_config
    from osprey.services.ariel_search.ingestion import get_adapter
    from osprey.utils.logger import get_logger

    logger = get_logger("ariel")

    if source:
        if "ingestion" not in config_dict:
            config_dict["ingestion"] = {}
        config_dict["ingestion"]["source_url"] = source
        config_dict["ingestion"]["adapter"] = "generic_json"

    config = ARIELConfig.from_dict(config_dict)

    if progress:
        progress("Checking database connection...")

    pool = await create_connection_pool(config.database)

    if progress:
        progress("  Database: connected")

    count = 0
    enhanced_count = 0
    failed_count = 0
    migrations_applied = 0

    try:
        if progress:
            progress("Running migrations...")

        applied = await run_migrations(pool, config)
        migrations_applied = len(applied) if applied else 0

        if applied and progress:
            progress(f"  Tables: created ({migrations_applied} migrations applied)")
        elif progress:
            progress("  Tables: already up to date")

        if not config.ingestion or not config.ingestion.source_url:
            if progress:
                progress("\nNo ingestion source configured. Skipping data ingestion.")
        else:
            if progress:
                progress(f"Ingesting data from: {config.ingestion.source_url}")
            adapter_instance = get_adapter(config)

            enhancers = create_enhancers_from_config(config)
            if enhancers and progress:
                progress(f"  Enhancement modules: {[e.name for e in enhancers]}")

            service = await create_ariel_service(config)
            async with service:
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
                                    logger.debug(f"Enhancement failed for {entry['entry_id']}: {e}")

                if progress:
                    progress(f"  Entries: {count} ingested")
                    if enhancers:
                        msg = f"  Enhancements: {enhanced_count} applied"
                        if failed_count:
                            msg += f", {failed_count} failed"
                        progress(msg)

        enabled_search = config.get_enabled_search_modules()

        if progress:
            progress(
                f"\nARIEL quickstart complete!"
                f"\n  Search modules: {', '.join(enabled_search) or 'none'}"
            )
            progress('\nTry it: osprey ariel search "What happened with the RF cavity?"')

    finally:
        await pool.close()

    return QuickstartResult(
        count=count,
        enhanced_count=enhanced_count,
        failed_count=failed_count,
        migrations_applied=migrations_applied,
        enabled_search=enabled_search,
    )


async def get_purge_info(config_dict: dict) -> PurgeInfo:
    """Get current counts for purge confirmation display."""
    from osprey.services.ariel_search import ARIELConfig
    from osprey.services.ariel_search.database.connection import create_connection_pool

    config = ARIELConfig.from_dict(config_dict)
    pool = await create_connection_pool(config.database)

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM enhanced_entries")
                row = await cur.fetchone()
                entry_count = row[0] if row else 0

                await cur.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                """)
                embedding_tables = [r[0] for r in await cur.fetchall()]
    finally:
        await pool.close()

    return PurgeInfo(entry_count=entry_count, embedding_tables=embedding_tables)


async def execute_purge(config_dict: dict, embeddings_only: bool, progress: _ProgressCb = None):
    """Execute the actual purge operation."""
    from osprey.services.ariel_search import ARIELConfig
    from osprey.services.ariel_search.database.connection import create_connection_pool

    config = ARIELConfig.from_dict(config_dict)
    pool = await create_connection_pool(config.database)

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if embeddings_only:
                    await cur.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                    """)
                    embedding_tables = [r[0] for r in await cur.fetchall()]
                    for table in embedding_tables:
                        await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")  # noqa: S608
                        if progress:
                            progress(f"  Dropped {table}")
                    if progress:
                        progress("\n✓ Embedding tables purged. Entries preserved.")
                else:
                    await cur.execute("TRUNCATE enhanced_entries CASCADE")
                    await cur.execute("TRUNCATE ingestion_runs CASCADE")
                    await cur.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name LIKE 'text_embeddings_%'
                    """)
                    embedding_tables = [r[0] for r in await cur.fetchall()]
                    for table in embedding_tables:
                        await cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")  # noqa: S608
                    if progress:
                        progress("\n✓ All ARIEL data purged.")
    finally:
        await pool.close()
