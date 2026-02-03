"""ARIEL CLI commands.

This module provides CLI commands for the ARIEL search service.

See 04_OSPREY_INTEGRATION.md Sections 13 for specification.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import click

# Import get_config at module level for easier patching in tests
try:
    from osprey.config import get_config
except ImportError:
    from osprey.utils.config import get_config

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
            config_dict = get_config("ariel", {})
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
                        "uri": config.database.uri.split("@")[-1] if "@" in config.database.uri else config.database.uri,
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
                        "semantic_processor": config.is_enhancement_module_enabled("semantic_processor"),
                    },
                    "search_modules": {
                        "keyword": config.is_search_module_enabled("keyword"),
                        "semantic": config.is_search_module_enabled("semantic"),
                        "rag": config.is_search_module_enabled("rag"),
                    },
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}

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
@click.option("--rollback", is_flag=True, help="Rollback migrations")
def migrate_command(rollback: bool) -> None:
    """Run ARIEL database migrations.

    Creates required database schema and tables based on enabled modules.
    """

    async def _migrate() -> None:
        from osprey.services.ariel_search import ARIELConfig
        from osprey.services.ariel_search.database.connection import create_connection_pool
        from osprey.services.ariel_search.database.migrate import run_migrations

        # Load config (get_config imported at module level)
        config_dict = get_config("ariel", {})
        if not config_dict:
            click.echo("Error: ARIEL not configured in config.yml", err=True)
            raise SystemExit(1)

        config = ARIELConfig.from_dict(config_dict)

        click.echo(f"Connecting to database: {config.database.uri.split('@')[-1]}")

        pool = await create_connection_pool(config.database)

        try:
            if rollback:
                click.echo("Rolling back migrations...")
                # V2: Implement rollback
                click.echo("Rollback not implemented in MVP", err=True)
            else:
                click.echo("Running migrations...")
                await run_migrations(pool, config)
                click.echo("Migrations complete.")
        finally:
            await pool.close()

    asyncio.run(_migrate())


@ariel_group.command("ingest")
@click.option("--source", "-s", required=True, type=click.Path(exists=True), help="Source file path")
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
    """Ingest logbook entries from a source file.

    Parses entries from the source file using the specified adapter
    and stores them in the ARIEL database.
    """

    async def _ingest() -> None:
        from osprey.services.ariel_search import ARIELConfig, create_ariel_service
        from osprey.services.ariel_search.ingestion import get_adapter

        # Load config (get_config imported at module level)
        config_dict = get_config("ariel", {})
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

        if dry_run:
            # Just count entries
            count = 0
            async for _entry in adapter_instance.fetch_entries(since=since, limit=limit):
                count += 1
                if count % 100 == 0:
                    click.echo(f"  Parsed {count} entries...")

            click.echo(f"\nDry run complete: {count} entries would be ingested")
            return

        # Full ingestion
        service = await create_ariel_service(config)
        async with service:
            count = 0
            async for entry in adapter_instance.fetch_entries(since=since, limit=limit):
                await service.repository.upsert_entry(entry)
                count += 1
                if count % 100 == 0:
                    click.echo(f"  Ingested {count} entries...")

            click.echo(f"\nIngestion complete: {count} entries stored")

    asyncio.run(_ingest())


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
        config_dict = get_config("ariel", {})
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
        config_dict = get_config("ariel", {})
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
        config_dict = get_config("ariel", {})
        if not config_dict:
            return {"error": "ARIEL not configured"}

        config = ARIELConfig.from_dict(config_dict)

        search_mode = None
        if mode != "auto":
            search_mode = SearchMode[mode.upper()]

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
        config_dict = get_config("ariel", {})
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


__all__ = ["ariel_group"]
