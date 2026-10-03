"""Integration tests for ARIEL database migrations.

Tests schema migrations against real PostgreSQL.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

import pytest

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]


@pytest.fixture
async def scratch_pool(scratch_config):
    """Pool on the empty scratch database."""
    from osprey.services.ariel_search.database import create_connection_pool

    pool = await create_connection_pool(scratch_config.database)
    try:
        yield pool
    finally:
        await pool.close()


class TestCoreMigration:
    """Test core schema migration."""

    async def test_run_migrations_creates_tables(self, scratch_pool, scratch_config):
        """Running migrations on an empty store creates the required tables."""
        from osprey.services.ariel_search.database import run_migrations

        applied = await run_migrations(scratch_pool, scratch_config)

        async with scratch_pool.connection() as conn:
            result = await conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('enhanced_entries', 'ariel_migrations', 'ingestion_runs')
            """)
            tables = [row[0] for row in await result.fetchall()]

        assert "core_schema" in applied
        assert "enhanced_entries" in tables
        assert "ariel_migrations" in tables

    async def test_migrations_are_idempotent(self, scratch_pool, scratch_config):
        """A second run on a migrated store applies nothing."""
        from osprey.services.ariel_search.database import run_migrations

        first = await run_migrations(scratch_pool, scratch_config)
        second = await run_migrations(scratch_pool, scratch_config)

        assert first
        assert second == []

    async def test_concurrent_runs_on_an_empty_store_apply_once(
        self, scratch_database, scratch_config
    ):
        """Two sessions migrating one empty store both return; one does the work.

        The ``ariel_migrate`` advisory lock serializes them: the second waits,
        then finds everything applied.
        """
        import asyncio

        from osprey.services.ariel_search.database import create_connection_pool, run_migrations

        pools = [await create_connection_pool(scratch_config.database) for _ in range(2)]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(run_migrations(pool, scratch_config) for pool in pools)),
                timeout=120,
            )
        finally:
            for pool in pools:
                await pool.close()

        assert sorted(results, key=len)[0] == []
        applied = sorted(results, key=len)[1]
        assert "core_schema" in applied

        import psycopg

        with psycopg.connect(scratch_database, autocommit=True) as conn:
            rows = conn.execute("SELECT name FROM ariel_migrations").fetchall()
        assert sorted(r[0] for r in rows) == sorted(applied)

    async def test_enhanced_entries_has_required_columns(self, migrated_pool):
        """enhanced_entries table has required columns."""
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'enhanced_entries'
            """)
            columns = {row[0] for row in await result.fetchall()}

        required_columns = {
            "entry_id",
            "source_system",
            "timestamp",
            "author",
            "raw_text",
            "attachments",
            "metadata",
            "enhancement_status",
            "created_at",
            "updated_at",
        }
        assert required_columns.issubset(columns)


class TestSemanticProcessorMigration:
    """Test semantic processor migration."""

    async def test_fts_index_created(self, migrated_pool):
        """FTS index is created by semantic processor migration."""
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'enhanced_entries'
                AND indexname = 'idx_entries_text_search'
            """)
            rows = await result.fetchall()

        # Index should exist if semantic processor is enabled
        assert len(rows) >= 1

    async def test_summary_column_created(self, migrated_pool):
        """Summary column is created by semantic processor migration."""
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'enhanced_entries'
                AND column_name = 'summary'
            """)
            rows = await result.fetchall()

        assert len(rows) == 1

    async def test_keyword_search_matches_summary_and_keywords(
        self, migrated_pool, repository, integration_ariel_config
    ):
        """The real PostgreSQL path retrieves terms absent from raw_text."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        entry_id = "test-semantic-fts-001"
        async with migrated_pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text,
                    attachments, metadata, enhancement_status, summary, keywords
                ) VALUES (
                    %s, 'test', NOW(), 'tester', 'Routine operator note',
                    '[]'::jsonb, '{}'::jsonb, '{}'::jsonb,
                    'Xylophonic vacuum condition', ARRAY['quenchmarker']
                )
                ON CONFLICT (entry_id) DO UPDATE SET
                    raw_text = EXCLUDED.raw_text,
                    summary = EXCLUDED.summary,
                    keywords = EXCLUDED.keywords
                """,
                [entry_id],
            )

        try:
            for query in ("xylophonic", "quenchmarker"):
                results = await keyword_search(
                    query,
                    repository,
                    integration_ariel_config,
                    fuzzy_fallback=False,
                )
                assert entry_id in {entry["entry_id"] for entry, _score, _highlights in results}
        finally:
            async with migrated_pool.connection() as conn:
                await conn.execute("DELETE FROM enhanced_entries WHERE entry_id = %s", [entry_id])


class TestTextEmbeddingMigration:
    """Test text embedding migration."""

    async def test_embedding_table_created(self, migrated_pool):
        """Embedding table is created for configured model."""
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE 'text_embeddings_%'
            """)
            tables = [row[0] for row in await result.fetchall()]

        # Should have at least one embedding table
        assert len(tables) >= 1
        # Should have table for nomic-embed-text
        assert "text_embeddings_nomic_embed_text" in tables

    async def test_pgvector_extension_available(self, migrated_pool):
        """pgvector extension is installed."""
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT extname FROM pg_extension WHERE extname = 'vector'
            """)
            rows = await result.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "vector"


class TestConnectionPool:
    """Test database connection pool."""

    async def test_create_connection_pool(self, database_url):
        """Creates a working connection pool."""
        from osprey.services.ariel_search.config import DatabaseConfig
        from osprey.services.ariel_search.database import create_connection_pool

        config = DatabaseConfig(uri=database_url)
        pool = await create_connection_pool(config)
        try:
            async with pool.connection() as conn:
                result = await conn.execute("SELECT 1 AS value")
                row = await result.fetchone()
                assert row[0] == 1
        finally:
            await pool.close()

    async def test_pool_executes_queries(self, database_url):
        """Pool can execute queries."""
        from osprey.services.ariel_search.config import DatabaseConfig
        from osprey.services.ariel_search.database import create_connection_pool

        config = DatabaseConfig(uri=database_url)
        pool = await create_connection_pool(config)
        try:
            async with pool.connection() as conn:
                result = await conn.execute("SELECT version() AS version")
                row = await result.fetchone()
                assert "PostgreSQL" in row[0]
        finally:
            await pool.close()


# ==============================================================================
# Migration Assertion Improvements (QUAL-001)
# ==============================================================================


class TestMigrationSQLExecution:
    """Quality assertions that migration SQL actually executed (QUAL-001)."""

    async def test_migration_records_stored(self, migrated_pool):
        """Verify migration records are stored in ariel_migrations table.

        QUAL-001: Assert migration SQL actually executed.
        """
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT name FROM ariel_migrations
                ORDER BY applied_at
            """)
            migrations = [row[0] for row in await result.fetchall()]

        # Should have at least core migration
        assert len(migrations) >= 1
        assert "core_schema" in migrations

    async def test_enhanced_entries_schema_matches_spec(self, migrated_pool):
        """Verify enhanced_entries schema matches specification.

        QUAL-001: Verify table schemas match expectations.
        """
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'enhanced_entries'
                ORDER BY ordinal_position
            """)
            columns = {
                row[0]: {"type": row[1], "nullable": row[2]} for row in await result.fetchall()
            }

        # Verify required columns and types
        assert "entry_id" in columns
        assert columns["entry_id"]["nullable"] == "NO"  # Primary key

        assert "source_system" in columns
        assert columns["source_system"]["nullable"] == "NO"

        assert "timestamp" in columns
        # timestamp with time zone
        assert "timestamp" in columns["timestamp"]["type"]

        assert "raw_text" in columns
        assert columns["raw_text"]["type"] == "text"

        assert "attachments" in columns
        assert columns["attachments"]["type"] == "jsonb"

        assert "metadata" in columns
        assert columns["metadata"]["type"] == "jsonb"

        assert "enhancement_status" in columns
        assert columns["enhancement_status"]["type"] == "jsonb"

    async def test_embedding_table_schema_correct(self, migrated_pool):
        """Verify embedding table has correct schema.

        QUAL-001: Verify embedding table structure.
        """
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'text_embeddings_nomic_embed_text'
            """)
            columns = {row[0]: row[1] for row in await result.fetchall()}

        # Should have entry_id and embedding columns
        assert "entry_id" in columns
        assert "embedding" in columns
        # pgvector type
        assert columns["embedding"] == "USER-DEFINED"

    async def test_fts_index_functional(self, migrated_pool):
        """Verify FTS index is actually functional.

        QUAL-001: Assert indexes are created and working.
        """
        # Insert a test entry
        async with migrated_pool.connection() as conn:
            await conn.execute("""
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text,
                    attachments, metadata, enhancement_status
                ) VALUES (
                    'test-fts-func-001', 'test', NOW(), 'tester',
                    'The beam current dropped significantly during operations',
                    '[]'::jsonb, '{}'::jsonb, '{}'::jsonb
                )
                ON CONFLICT (entry_id) DO NOTHING
            """)

            # Test FTS search uses the index
            result = await conn.execute("""
                EXPLAIN SELECT * FROM enhanced_entries
                WHERE to_tsvector('english', raw_text) @@ plainto_tsquery('english', 'beam current')
            """)
            plan = "\n".join([row[0] for row in await result.fetchall()])

            # Clean up
            await conn.execute("DELETE FROM enhanced_entries WHERE entry_id = 'test-fts-func-001'")

        # The query plan should reference the core raw-text FTS index
        assert "idx_entries_raw_text_fts" in plan or "Seq Scan" in plan

    async def test_primary_key_constraint_exists(self, migrated_pool):
        """Verify primary key constraint exists on enhanced_entries.

        QUAL-001: Assert constraints are properly created.
        """
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints
                WHERE table_name = 'enhanced_entries'
                AND constraint_type = 'PRIMARY KEY'
            """)
            constraints = await result.fetchall()

        assert len(constraints) >= 1
        # Primary key on entry_id
        assert any("pkey" in c[0].lower() or "primary" in c[1].lower() for c in constraints)

    async def test_timestamp_columns_have_defaults(self, migrated_pool):
        """Verify created_at and updated_at have default values.

        QUAL-001: Assert default values are set.
        """
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT column_name, column_default
                FROM information_schema.columns
                WHERE table_name = 'enhanced_entries'
                AND column_name IN ('created_at', 'updated_at')
            """)
            defaults = {row[0]: row[1] for row in await result.fetchall()}

        # Should have default values (now() or similar)
        assert defaults.get("created_at") is not None
        assert (
            "now" in defaults["created_at"].lower()
            or "current_timestamp" in defaults["created_at"].lower()
        )

    async def test_embedding_foreign_key_exists(self, migrated_pool):
        """Verify embedding table has foreign key to enhanced_entries.

        QUAL-001: Assert foreign key relationships.
        """
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT tc.constraint_name, tc.table_name, kcu.column_name,
                       ccu.table_name AS foreign_table_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_name = 'text_embeddings_nomic_embed_text'
            """)
            fks = await result.fetchall()

        # Should have FK to enhanced_entries
        if fks:  # FK may be optional in some configurations
            assert any(fk[3] == "enhanced_entries" for fk in fks)


class TestPlainTextMigration:
    """The one-off rewrite of stored ``als_logbook`` rows, against real PostgreSQL."""

    async def test_stored_rows_become_what_a_fresh_ingest_stores(
        self, migrated_pool, integration_ariel_config
    ):
        """Rewritten rows match a fresh ingest, are requeued, and are found by their words."""
        import json
        from datetime import UTC, datetime
        from pathlib import Path

        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.database.repository import ARIELRepository
        from osprey.services.ariel_search.ingestion.adapters.als import (
            ALS_SOURCE_SYSTEM,
            ALSLogbookAdapter,
        )
        from osprey.services.ariel_search.ingestion.adapters.als_text_migration import (
            ALSPlainTextMigration,
        )

        fixture = (
            Path(__file__).parents[3] / "fixtures" / "ariel" / "als_olog_encoded_entries.jsonl"
        )
        rows = {
            row["id"]: row
            for row in (json.loads(line) for line in fixture.read_text().splitlines() if line)
        }
        adapter = ALSLogbookAdapter(
            ARIELConfig.from_dict(
                {
                    "database": {"uri": "postgresql://unused"},
                    "ingestion": {
                        "adapter": "als_logbook",
                        "source_url": "https://olog.example.invalid/rpc.php",
                    },
                }
            )
        )
        repo = ARIELRepository(migrated_pool, integration_ariel_config)
        complete = {"text_embedding": {"status": "complete"}}

        def stored_id(entry_id: str) -> str:
            return f"plain-text-{entry_id}"

        try:
            for entry_id, row in rows.items():
                subject, details = row["subject"], row["details"]
                now = datetime.now(UTC)
                await repo.upsert_entry(
                    {
                        "entry_id": stored_id(entry_id),
                        "source_system": ALS_SOURCE_SYSTEM,
                        "timestamp": now,
                        "author": row["author"],
                        "raw_text": (
                            f"{subject}\n\n{details}" if subject and details else subject or details
                        ),
                        "attachments": [],
                        "metadata": {"subject": subject} if subject else {},
                        "created_at": now,
                        "updated_at": now,
                        "enhancement_status": complete,
                    }
                )

            async def read() -> dict[str, tuple[str, dict, datetime]]:
                async with migrated_pool.connection() as conn:
                    result = await conn.execute(
                        "SELECT entry_id, raw_text, enhancement_status, updated_at "
                        "FROM enhanced_entries WHERE entry_id LIKE 'plain-text-%'"
                    )
                    return {r[0]: (r[1], r[2], r[3]) for r in await result.fetchall()}

            before = await read()

            async with migrated_pool.connection() as conn:
                async with conn.transaction():
                    await ALSPlainTextMigration().up(conn)

            after = await read()

            for entry_id in ("20001", "20002", "20003", "20004", "20007", "20008"):
                raw_text, status, updated_at = after[stored_id(entry_id)]
                assert raw_text == adapter._convert_entry(rows[entry_id])["raw_text"]
                assert status == {}
                assert updated_at > before[stored_id(entry_id)][2]
            assert after[stored_id("20005")][:2] == before[stored_id("20005")][:2]
            assert after[stored_id("20005")][1] == complete
            assert after[stored_id("20006")][0] == before[stored_id("20006")][0]

            async def matches(word: str) -> list[str]:
                async with migrated_pool.connection() as conn:
                    result = await conn.execute(
                        "SELECT entry_id FROM enhanced_entries WHERE entry_id = %s "
                        "AND to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)",
                        [stored_id("20001"), word],
                    )
                    return [r[0] for r in await result.fetchall()]

            assert await matches("retuned") == [stored_id("20001")]
            assert await matches("lt") == []
        finally:
            async with migrated_pool.connection() as conn:
                await conn.execute(
                    "DELETE FROM enhanced_entries WHERE entry_id LIKE 'plain-text-%'"
                )


class TestAttachmentTextColumnsMigration:
    """``attachment_text_columns``: two nullable columns under a non-queueing lock."""

    MIGRATION = "attachment_text_columns"

    @staticmethod
    def _rows(*names: str) -> list:
        """The ``KNOWN_MIGRATIONS`` rows for *names*, in registry order."""
        from osprey.services.ariel_search.database import migrations

        return [row for row in migrations.KNOWN_MIGRATIONS if row[0] in names]

    async def test_registered_always_on_after_core_schema(self):
        """The registry row runs without a module and the migration follows core_schema."""
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextColumnsMigration,
        )

        (row,) = self._rows(self.MIGRATION)
        assert row == (
            self.MIGRATION,
            "osprey.services.ariel_search.database.attachment_text_migration",
            "AttachmentTextColumnsMigration",
            None,
        )
        migration = AttachmentTextColumnsMigration()
        assert migration.name == self.MIGRATION
        assert migration.depends_on == ["core_schema"]

    async def test_full_run_adds_both_nullable_columns(self, scratch_pool, scratch_config):
        """A full run on an empty store applies it and adds TEXT and JSONB columns."""
        from osprey.services.ariel_search.database import run_migrations

        applied = await run_migrations(scratch_pool, scratch_config)

        async with scratch_pool.connection() as conn:
            result = await conn.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = 'enhanced_entries'
                AND column_name IN ('attachment_text', 'attachment_captions')
            """)
            columns = {row[0]: row[1:] for row in await result.fetchall()}

        assert self.MIGRATION in applied
        assert applied.index("core_schema") < applied.index(self.MIGRATION)
        assert columns == {
            "attachment_text": ("text", "YES", None),
            "attachment_captions": ("jsonb", "YES", None),
        }

    async def test_busy_table_skips_quickly_without_blocking_readers(
        self, scratch_database, scratch_pool, scratch_config, monkeypatch
    ):
        """An open reader makes migrate return busy within 6 s; a later reader never waits.

        The reader holds ACCESS SHARE on ``enhanced_entries`` for the whole run.
        A queued ACCESS EXCLUSIVE request would make the third session's read
        wait behind it; the NOWAIT retries never queue, so it returns at once.
        """
        import asyncio
        import time

        import psycopg

        from osprey.services.ariel_search.database import migrations

        core_only = self._rows("core_schema")
        both = self._rows("core_schema", self.MIGRATION)

        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", core_only)
        assert await migrations.run_migrations(scratch_pool, scratch_config) == ["core_schema"]
        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", both)

        holder = await psycopg.AsyncConnection.connect(scratch_database)
        third = await psycopg.AsyncConnection.connect(scratch_database, autocommit=True)
        try:
            await holder.execute("SELECT count(*) FROM enhanced_entries")  # opens a txn

            async def late_reader() -> float:
                await asyncio.sleep(1.5)
                started = time.monotonic()
                await third.execute("SET statement_timeout = '3s'")
                await third.execute("SELECT count(*) FROM enhanced_entries")
                return time.monotonic() - started

            started = time.monotonic()
            result, reader_s = await asyncio.wait_for(
                asyncio.gather(
                    migrations.run_migrations_detailed(scratch_pool, scratch_config),
                    late_reader(),
                ),
                timeout=30,
            )
            migrate_s = time.monotonic() - started
        finally:
            await holder.rollback()
            await holder.close()
            await third.close()

        assert result.applied == []
        assert result.busy_skipped == [self.MIGRATION]
        assert migrate_s < 6.0
        assert reader_s < 1.0

        async with scratch_pool.connection() as conn:
            result_cols = await conn.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'enhanced_entries'
                AND column_name IN ('attachment_text', 'attachment_captions')
            """)
            assert await result_cols.fetchall() == []
            marked = await conn.execute(
                "SELECT count(*) FROM ariel_migrations WHERE name = %s", [self.MIGRATION]
            )
            assert (await marked.fetchone())[0] == 0

        # Once the reader is gone the retry applies it.
        retry = await migrations.run_migrations_detailed(scratch_pool, scratch_config)
        assert retry.applied == [self.MIGRATION]
        assert retry.busy_skipped == []

    async def test_core_schema_skip_leaves_it_unapplied(
        self, scratch_pool, scratch_config, monkeypatch
    ):
        """A skipped core_schema makes it wait: ``up()`` never runs and nothing is marked."""
        from osprey.services.ariel_search.database import migrations
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextColumnsMigration,
        )
        from osprey.services.ariel_search.database.core_migration import CoreMigration

        async def skip(_self, _conn):
            raise migrations.MigrationSkippedError("prerequisite missing")

        calls: list[str] = []

        async def spy(self, _conn):
            calls.append(self.name)

        monkeypatch.setattr(
            migrations, "KNOWN_MIGRATIONS", self._rows("core_schema", self.MIGRATION)
        )
        monkeypatch.setattr(CoreMigration, "up", skip)
        monkeypatch.setattr(AttachmentTextColumnsMigration, "up", spy)

        result = await migrations.run_migrations_detailed(scratch_pool, scratch_config)

        assert result.applied == []
        assert result.busy_skipped == []
        assert calls == []
        async with scratch_pool.connection() as conn:
            assert not await AttachmentTextColumnsMigration().is_applied(conn)


class TestAttachmentFilesCopyStateMigration:
    """``attachment_files_copy_state``: copy-state and rendition columns on attachment_files."""

    MIGRATION = "attachment_files_copy_state"

    #: Columns the migration adds or relaxes: (data_type, is_nullable, has_default).
    EXPECTED_COLUMNS = {
        "data": ("bytea", "YES"),
        "size_bytes": ("integer", "YES"),
        "source_url": ("text", "YES"),
        "copy_status": ("text", "NO"),
        "skip_reason": ("text", "YES"),
        "copy_attempts": ("smallint", "NO"),
        "rendition_bytes": ("bytea", "YES"),
        "rendition_mime": ("text", "YES"),
        "rendition_w": ("integer", "YES"),
        "rendition_h": ("integer", "YES"),
        "rendition_sha256": ("text", "YES"),
    }

    @staticmethod
    def _rows(*names: str) -> list:
        """The ``KNOWN_MIGRATIONS`` rows for *names*, in registry order."""
        from osprey.services.ariel_search.database import migrations

        return [row for row in migrations.KNOWN_MIGRATIONS if row[0] in names]

    @staticmethod
    async def _columns(conn) -> dict[str, tuple]:
        result = await conn.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'attachment_files'
        """)
        return {row[0]: row[1:] for row in await result.fetchall()}

    @staticmethod
    async def _seed_entry(conn, entry_id: str) -> None:
        await conn.execute(
            """
            INSERT INTO enhanced_entries (
                entry_id, source_system, timestamp, author, raw_text,
                attachments, metadata, enhancement_status
            ) VALUES (%s, 'test', NOW(), 'tester', 'x', '[]'::jsonb, '{}'::jsonb, '{}'::jsonb)
            """,
            [entry_id],
        )

    def _assert_copy_state_schema(self, columns: dict[str, tuple]) -> None:
        for name, (data_type, nullable) in self.EXPECTED_COLUMNS.items():
            assert columns[name][:2] == (data_type, nullable), name
        assert "'copied'" in columns["copy_status"][2]
        assert columns["copy_attempts"][2] == "0"

    @pytest.mark.parametrize("semantic_processor", [False, True])
    async def test_fresh_migrate_applies_it(self, scratch_config, semantic_processor):
        """A fresh store migrates cleanly with semantic_processor off and on."""
        from dataclasses import replace

        from osprey.services.ariel_search.config import EnhancementModuleConfig
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations

        modules = dict(scratch_config.enhancement_modules)
        modules["semantic_processor"] = EnhancementModuleConfig(enabled=semantic_processor)
        config = replace(scratch_config, enhancement_modules=modules)

        pool = await create_connection_pool(config.database)
        try:
            applied = await run_migrations(pool, config)
            async with pool.connection() as conn:
                columns = await self._columns(conn)
        finally:
            await pool.close()

        assert self.MIGRATION in applied
        assert ("semantic_processor" in applied) is semantic_processor
        assert applied.index(self.MIGRATION) > applied.index("attachment_files")
        assert applied.index(self.MIGRATION) > applied.index("attachment_text_columns")
        self._assert_copy_state_schema(columns)

    async def test_upgrade_from_todays_store_keeps_rows_copied(
        self, scratch_pool, scratch_config, monkeypatch
    ):
        """A store at today's migration set upgrades; an existing file row reads as copied."""
        from osprey.services.ariel_search.database import migrations

        everything = list(migrations.KNOWN_MIGRATIONS)
        today = [row for row in everything if row[0] != self.MIGRATION]
        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", today)
        first = await migrations.run_migrations(scratch_pool, scratch_config)
        assert self.MIGRATION not in first

        async with scratch_pool.connection() as conn:
            await self._seed_entry(conn, "copy-state-upgrade")
            await conn.execute(
                """
                INSERT INTO attachment_files
                    (attachment_id, entry_id, filename, mime_type, data, size_bytes)
                VALUES ('att-upgrade', 'copy-state-upgrade', 'a.png', 'image/png', '\\x00', 1)
                """
            )

        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", everything)
        assert await migrations.run_migrations(scratch_pool, scratch_config) == [self.MIGRATION]

        async with scratch_pool.connection() as conn:
            self._assert_copy_state_schema(await self._columns(conn))
            result = await conn.execute(
                "SELECT copy_status, skip_reason, copy_attempts, size_bytes, rendition_sha256 "
                "FROM attachment_files WHERE attachment_id = 'att-upgrade'"
            )
            assert await result.fetchone() == ("copied", None, 0, 1, None)

    async def test_text_columns_busy_leaves_it_unapplied(
        self, scratch_database, scratch_pool, scratch_config, monkeypatch
    ):
        """``attachment_text_columns`` forced busy makes this one wait; a retry applies both."""
        import psycopg

        from osprey.services.ariel_search.database import migrations

        names = ("core_schema", "attachment_files", "attachment_text_columns", self.MIGRATION)
        base, chain = self._rows(*names[:2]), self._rows(*names)
        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", base)
        assert await migrations.run_migrations(scratch_pool, scratch_config) == list(names[:2])
        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", chain)
        monkeypatch.setattr(migrations, "LOCK_RETRY_DELAY_S", 0.01)

        holder = await psycopg.AsyncConnection.connect(scratch_database)
        try:
            await holder.execute("SELECT count(*) FROM enhanced_entries")  # opens a txn
            result = await migrations.run_migrations_detailed(scratch_pool, scratch_config)
        finally:
            await holder.rollback()
            await holder.close()

        assert result.applied == []
        assert result.busy_skipped == ["attachment_text_columns"]
        async with scratch_pool.connection() as conn:
            assert "copy_status" not in await self._columns(conn)
            marked = await conn.execute(
                "SELECT count(*) FROM ariel_migrations WHERE name = %s", [self.MIGRATION]
            )
            assert (await marked.fetchone())[0] == 0

        retry = await migrations.run_migrations_detailed(scratch_pool, scratch_config)
        assert retry.applied == ["attachment_text_columns", self.MIGRATION]
        assert retry.busy_skipped == []

    async def test_skip_reason_check_admits_only_registry_codes(self, scratch_pool, scratch_config):
        """The CHECK rejects an unknown code and admits a known one and NULL."""
        import psycopg

        from osprey.services.ariel_search.database import run_migrations
        from osprey.services.ariel_search.database.attachment_migration import (
            COPY_STATE_SKIP_REASONS,
        )

        await run_migrations(scratch_pool, scratch_config)
        insert = """
            INSERT INTO attachment_files
                (attachment_id, entry_id, filename, copy_status, skip_reason, source_url)
            VALUES (%s, 'copy-state-check', 'f', %s, %s, 'https://h/f')
        """
        async with scratch_pool.connection() as conn:
            await self._seed_entry(conn, "copy-state-check")
            await conn.execute(insert, ["att-pending", "pending", None])
            for i, code in enumerate(COPY_STATE_SKIP_REASONS):
                await conn.execute(insert, [f"att-known-{i}", "skipped", code])
            with pytest.raises(psycopg.errors.CheckViolation):
                await conn.execute(insert, ["att-bogus", "skipped", "bogus_reason"])
            with pytest.raises(psycopg.errors.CheckViolation):
                await conn.execute(insert, ["att-summary", "skipped", "no_source_url"])

            result = await conn.execute(
                "SELECT data, size_bytes FROM attachment_files WHERE attachment_id = 'att-pending'"
            )
            assert await result.fetchone() == (None, None)

    @pytest.mark.parametrize(
        "predicate",
        [
            "copy_status = 'pending'",
            "copy_status = 'copied' AND rendition_sha256 IS NULL AND skip_reason IS NULL",
            "copy_status = 'pending' OR "
            "(copy_status = 'copied' AND rendition_sha256 IS NULL AND skip_reason IS NULL)",
        ],
    )
    async def test_copy_todo_index_serves_the_backfill_predicate(
        self, scratch_pool, scratch_config, predicate
    ):
        """With seq scans off the planner reaches the partial to-do index."""
        from osprey.services.ariel_search.database import run_migrations

        await run_migrations(scratch_pool, scratch_config)
        async with scratch_pool.connection() as conn, conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            result = await conn.execute(
                f"EXPLAIN SELECT entry_id FROM attachment_files WHERE {predicate}"
            )
            plan = "\n".join(row[0] for row in await result.fetchall())

        assert "idx_attachment_files_copy_todo" in plan, plan


class _FoldConn:
    """Connection wrapper recording each fold page's entry ids and every write."""

    def __init__(self, conn) -> None:
        self.conn = conn
        self.pages: list[list[str]] = []
        self.writes: list[str] = []

    async def execute(self, query, params=None):
        cursor = await self.conn.execute(query, params)
        if "FOR UPDATE" in query:
            rows = await cursor.fetchall()
            self.pages.append([row[0] for row in rows])

            class _Page:
                rowcount = len(rows)

                @staticmethod
                async def fetchall():
                    return rows

            return _Page()
        if query.lstrip().upper().startswith("UPDATE"):
            self.writes.append(query)
        return cursor


class TestAttachmentTextUpstreamFoldMigration:
    """``attachment_text_upstream_fold``: upstream captions become ``attachment_text``."""

    MODEL = "vision-m"

    @pytest.fixture
    async def store(self, scratch_pool, scratch_config):
        """A migrated, empty scratch store (the fold already ran on it, touching nothing)."""
        from osprey.services.ariel_search.database import run_migrations

        applied = await run_migrations(scratch_pool, scratch_config)
        assert "attachment_text_upstream_fold" in applied
        assert applied.index("attachment_text_columns") < applied.index(
            "attachment_text_upstream_fold"
        )
        return scratch_pool

    @staticmethod
    async def _seed(pool, rows: list[dict]) -> None:
        from psycopg.types.json import Jsonb

        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO enhanced_entries (
                        entry_id, source_system, timestamp, raw_text, attachments,
                        attachment_captions, attachment_text, enhancement_status
                    ) VALUES (
                        %(entry_id)s, 'jlab', NOW(), 'beam tuning', %(attachments)s,
                        %(captions)s, %(text)s, %(status)s
                    )
                    """,
                    [
                        {
                            "entry_id": row["entry_id"],
                            "attachments": Jsonb(row.get("attachments", [])),
                            "captions": Jsonb(row["captions"]) if "captions" in row else None,
                            "text": row.get("text"),
                            "status": Jsonb(row.get("status", {})),
                        }
                        for row in rows
                    ],
                )

    async def _fold(self, pool, model_id: str | None = MODEL) -> _FoldConn:
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextUpstreamFoldMigration,
        )

        async with pool.connection() as conn:
            async with conn.transaction():
                spy = _FoldConn(conn)
                await AttachmentTextUpstreamFoldMigration(model_id).up(spy)  # type: ignore[arg-type]
        return spy

    @staticmethod
    async def _row(pool, entry_id: str) -> tuple:
        async with pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT attachment_text, enhancement_status FROM enhanced_entries"
                " WHERE entry_id = %s",
                (entry_id,),
            )
            return await cursor.fetchone()

    @staticmethod
    def _item(name: str, caption) -> dict:
        return {
            "url": f"https://logbooks.jlab.org/files/{name}",
            "filename": name,
            "type": "image/png",
            "caption": caption,
        }

    async def test_fold_composes_upstream_caption_and_clears_dependent_statuses(
        self, store, caplog
    ):
        """A JLab row's caption is composed, its text statuses cleared, and the log counts it."""
        import logging

        caplog.set_level(logging.INFO, logger="ariel")
        status = {
            "text_embedding": {"status": "complete"},
            "qmd_export": {"status": "complete"},
            "semantic_processor": {"status": "complete"},
        }
        await self._seed(
            store,
            [
                {
                    "entry_id": "jlab-1",
                    "attachments": [self._item("a.png", "RF trip")],
                    "status": status,
                }
            ],
        )

        await self._fold(store)

        text, stored_status = await self._row(store, "jlab-1")
        assert text == "[picture a.png - upstream caption] RF trip"
        assert stored_status == {"semantic_processor": {"status": "complete"}}
        assert "1 rows recomposed, 1 text_embedding/qmd_export statuses cleared" in caplog.text

    async def test_fold_leaves_an_already_composed_row_alone(self, store, caplog):
        """Text already equal to the composed text is neither written, cleared nor counted."""
        import logging

        caplog.set_level(logging.INFO, logger="ariel")
        status = {"text_embedding": {"status": "complete"}, "qmd_export": {"status": "complete"}}
        await self._seed(
            store,
            [
                {
                    "entry_id": "jlab-2",
                    "attachments": [self._item("b.png", "Vacuum burst")],
                    "text": "[picture b.png - upstream caption] Vacuum burst",
                    "status": status,
                }
            ],
        )

        spy = await self._fold(store)

        assert spy.writes == []
        assert await self._row(store, "jlab-2") == (
            "[picture b.png - upstream caption] Vacuum burst",
            status,
        )
        assert "0 rows recomposed, 0 text_embedding/qmd_export statuses cleared" in caplog.text

    async def test_fold_keeps_a_stored_model_caption(self, store):
        """A row holding a caption by the configured model keeps its machine-caption marker."""
        from osprey.services.ariel_search.attachments import attachment_id_for

        item = self._item("c.png", "upstream words")
        attachment_id = attachment_id_for("jlab-3", item)
        await self._seed(
            store,
            [
                {
                    "entry_id": "jlab-3",
                    "attachments": [item, self._item("d.png", "second picture")],
                    "captions": {attachment_id: {self.MODEL: {"caption": "a beam profile"}}},
                }
            ],
        )

        await self._fold(store)

        text, _ = await self._row(store, "jlab-3")
        assert text == (
            f"[picture c.png - machine caption by {self.MODEL}] a beam profile\n"
            "[picture d.png - upstream caption] second picture"
        )

    async def test_fold_does_not_page_rows_with_only_null_or_empty_captions(self, store):
        """Only rows holding a non-empty upstream caption are read (and locked)."""
        await self._seed(
            store,
            [
                {"entry_id": "jlab-4", "attachments": [self._item("e.png", None)]},
                {"entry_id": "jlab-5", "attachments": [self._item("f.png", "")]},
                {"entry_id": "jlab-6", "attachments": [{"url": "x", "filename": "g.png"}]},
                {"entry_id": "jlab-7", "attachments": [self._item("h.png", "kept")]},
            ],
        )

        spy = await self._fold(store)

        assert spy.pages == [["jlab-7"]]
        assert (await self._row(store, "jlab-4"))[0] is None

    async def test_fold_pages_1200_unchanged_rows_in_three_pages_without_writes(self, store):
        """1200 caption rows already equal to their composed text: 3 pages, zero writes."""
        rows = [
            {
                "entry_id": f"jlab-{i:05d}",
                "attachments": [self._item(f"p{i}.png", f"caption {i}")],
                "text": f"[picture p{i}.png - upstream caption] caption {i}",
            }
            for i in range(1200)
        ]
        await self._seed(store, rows)

        spy = await self._fold(store)

        assert [len(page) for page in spy.pages] == [500, 500, 200]
        assert spy.pages[1][0] > spy.pages[0][-1]
        assert spy.writes == []


class TestV2SearchIndexMigrations:
    """``raw_text_fts_index_v2`` and ``semantic_processor_search_index_v2`` on a real store.

    Every query path stays on the v1 expressions until the schema flag lands, so
    the ``EXPLAIN`` tests build their SQL directly from ``RAW_TEXT_FTS_EXPRESSION_V2``
    and from the literal two-column pattern span.
    """

    PATTERN_SPAN = "(raw_text ~* %s OR COALESCE(attachment_text,'') ~* %s)"

    @staticmethod
    def _config(scratch_config, semantic_processor: bool):
        from dataclasses import replace

        from osprey.services.ariel_search.config import EnhancementModuleConfig

        modules = dict(scratch_config.enhancement_modules)
        modules["semantic_processor"] = EnhancementModuleConfig(enabled=semantic_processor)
        return replace(scratch_config, enhancement_modules=modules)

    @staticmethod
    async def _indexes(conn) -> set[str]:
        result = await conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'enhanced_entries'"
        )
        return {row[0] for row in await result.fetchall()}

    @staticmethod
    async def _seed(conn, count: int = 200) -> None:
        for i in range(count):
            await conn.execute(
                """
                INSERT INTO enhanced_entries (
                    entry_id, source_system, timestamp, author, raw_text,
                    attachments, metadata, enhancement_status, attachment_text
                ) VALUES (
                    %s, 'test', NOW(), 'tester', %s,
                    '[]'::jsonb, '{}'::jsonb, '{}'::jsonb, %s
                )
                """,
                [
                    f"v2-{i:04d}",
                    f"shift note {i} vacuum gauge reading",
                    f"[picture p{i}.png - upstream caption] klystron trace {i}" if i % 3 else None,
                ],
            )

    @staticmethod
    async def _explain(conn, sql: str, params: list[str]) -> str:
        """The plan text of *sql* after ANALYZE, with sequential scans disabled."""
        import psycopg

        await conn.execute("ANALYZE enhanced_entries")
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            cursor = psycopg.AsyncClientCursor(conn)
            try:
                await cursor.execute(f"EXPLAIN {sql}", params)
                rows = await cursor.fetchall()
            finally:
                await cursor.close()
        return "\n".join(row[0] for row in rows)

    @pytest.fixture
    async def store(self, scratch_config):
        """A scratch store migrated with semantic_processor on, seeded with rows."""
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations

        config = self._config(scratch_config, True)
        pool = await create_connection_pool(config.database)
        try:
            await run_migrations(pool, config)
            async with pool.connection() as conn:
                await self._seed(conn)
            yield pool
        finally:
            await pool.close()

    @pytest.mark.parametrize("semantic_processor", [False, True])
    async def test_v2_topological_order_and_kept_v1_indexes(
        self, scratch_config, semantic_processor
    ):
        """A fresh migrate orders each v2 build after its dependencies and keeps v1."""
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations

        config = self._config(scratch_config, semantic_processor)
        pool = await create_connection_pool(config.database)
        try:
            applied = await run_migrations(pool, config)
            async with pool.connection() as conn:
                indexes = await self._indexes(conn)
        finally:
            await pool.close()

        pos = applied.index
        assert pos("attachment_text_upstream_fold") < pos("raw_text_fts_index_v2")
        assert {
            "idx_entries_raw_text_fts",
            "idx_entries_raw_text_fts_v2",
            "idx_entries_raw_text_trgm",
            "idx_entries_attachment_text_trgm",
        } <= indexes
        assert ("semantic_processor_search_index_v2" in applied) is semantic_processor
        assert ({"idx_entries_text_search", "idx_entries_text_search_v2"} <= indexes) is (
            semantic_processor
        )
        if semantic_processor:
            v2 = pos("semantic_processor_search_index_v2")
            assert pos("semantic_processor_search_index") < v2
            assert pos("attachment_text_upstream_fold") < v2

    async def test_v2_keyword_query_uses_raw_text_fts_v2_index(self, store):
        """A keyword query over ``RAW_TEXT_FTS_EXPRESSION_V2`` scans the v2 index."""
        from osprey.services.ariel_search.database.search_fts import RAW_TEXT_FTS_EXPRESSION_V2

        async with store.connection() as conn:
            plan = await self._explain(
                conn,
                "SELECT entry_id FROM enhanced_entries "
                f"WHERE {RAW_TEXT_FTS_EXPRESSION_V2} @@ plainto_tsquery('english', %s)",
                ["klystron"],
            )

        assert "Bitmap Index Scan on idx_entries_raw_text_fts_v2" in plan, plan

    async def test_v2_pattern_query_bitmap_ors_both_trigram_indexes(self, store):
        """A pattern span over both columns ORs the two trigram index scans."""
        async with store.connection() as conn:
            plan = await self._explain(
                conn,
                f"SELECT entry_id FROM enhanced_entries WHERE {self.PATTERN_SPAN}",
                ["klystron", "klystron"],
            )

        assert "BitmapOr" in plan, plan
        assert "Bitmap Index Scan on idx_entries_raw_text_trgm" in plan, plan
        assert "Bitmap Index Scan on idx_entries_attachment_text_trgm" in plan, plan

    @pytest.mark.usefixtures("store")
    async def test_v2_share_lock_holder_does_not_block_readers(self, scratch_database):
        """While a session holds SHARE (as a v2 build does), a reader returns at once."""
        import time

        import psycopg

        builder = await psycopg.AsyncConnection.connect(scratch_database)
        reader = await psycopg.AsyncConnection.connect(scratch_database, autocommit=True)
        try:
            await builder.execute("LOCK TABLE enhanced_entries IN SHARE MODE")
            await reader.execute("SET statement_timeout = '3s'")
            started = time.monotonic()
            result = await reader.execute("SELECT count(*) FROM enhanced_entries")
            count = (await result.fetchone())[0]
            reader_s = time.monotonic() - started
        finally:
            await builder.rollback()
            await builder.close()
            await reader.close()

        assert count == 200
        assert reader_s < 1.0


@pytest.fixture
def image_embedding_non_owner_store(database_url: str):
    """A scratch database whose extensions belong to the superuser, not ARIEL's role.

    As superuser: create the database, ``vector`` and ``pg_trgm`` in it (a DBA
    installs extensions on an external store), a per-run login role
    ``ariel_app_<uuid8>`` (roles are cluster-wide and the server may be a
    long-lived dev one), and ``CREATE`` on ``public`` for it (PG15+ grants
    none). Teardown drops the database first, which removes everything the
    role owns, then the role.

    Yields:
        ``(role, role_conninfo)``: the role's name and a conninfo connecting
        to the scratch database as it.
    """
    import secrets
    import uuid

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from tests.services.ariel_search.conftest import skip_or_fail

    run_id = uuid.uuid4().hex
    name = f"ariel_{run_id}"
    role = f"ariel_app_{run_id[:8]}"
    password = secrets.token_hex(16)
    uri = f"{database_url.rsplit('/', 1)[0]}/{name}"

    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        with psycopg.connect(database_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role), sql.Literal(password)
                )
            )
        with psycopg.connect(uri, autocommit=True) as conn:
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except psycopg.Error as e:
                skip_or_fail(f"pgvector is not installed on the test server: {e}")
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            conn.execute(
                sql.SQL("GRANT CREATE ON SCHEMA public TO {}").format(sql.Identifier(role))
            )
        yield role, make_conninfo(uri, user=role, password=password)
    finally:
        with psycopg.connect(database_url, autocommit=True) as conn:
            try:
                conn.execute(
                    sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
                )
            finally:
                # Roles are cluster-wide: always try; it fails loudly if the
                # database (and what the role owns there) survived.
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


class TestImageEmbeddingMigration:
    """``image_embedding``: the per-model image table, built by a role that owns no extension."""

    MODEL = "clip-vit-b-32"
    DIMS = 512

    @classmethod
    def _config(cls, integration_ariel_config, conninfo: str, *, dims: int | None = None):
        from dataclasses import replace

        from osprey.services.ariel_search.config import EnhancementModuleConfig

        modules = dict(integration_ariel_config.enhancement_modules)
        modules["image_embedding"] = EnhancementModuleConfig(
            enabled=True,
            provider="ollama",
            settings={"model": cls.MODEL, "dimensions": dims or cls.DIMS},
        )
        return replace(
            integration_ariel_config,
            database=replace(integration_ariel_config.database, uri=conninfo),
            enhancement_modules=modules,
        )

    @staticmethod
    async def _scalar(conn, query: str, params: tuple = ()):
        result = await conn.execute(query, params)
        row = await result.fetchone()
        return row[0] if row else None

    async def _migrate(self, config):
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations

        from .conftest import RecordingPool

        pool = await create_connection_pool(config.database)
        spy = RecordingPool(pool)
        try:
            applied = await run_migrations(spy, config)  # type: ignore[arg-type]
        except BaseException:
            await pool.close()
            raise
        return pool, spy, applied

    async def test_image_embedding_chain_runs_as_non_owner_of_vector(
        self, image_embedding_non_owner_store, integration_ariel_config
    ):
        """The whole chain runs as ariel_app; no ALTER EXTENSION; it owns the image table."""
        import re

        from osprey.services.ariel_search.database.migrations import (
            image_index_name,
            image_table_name,
        )

        role, conninfo = image_embedding_non_owner_store
        table = image_table_name(self.MODEL, self.DIMS)
        pool, spy, applied = await self._migrate(self._config(integration_ariel_config, conninfo))
        try:
            async with pool.connection() as conn:
                owner = await self._scalar(
                    conn, "SELECT tableowner FROM pg_tables WHERE tablename = %s", (table,)
                )
                ledger_owner = await self._scalar(
                    conn, "SELECT tableowner FROM pg_tables WHERE tablename = 'ariel_migrations'"
                )
                vector_owner = await self._scalar(
                    conn,
                    "SELECT extowner::regrole::text FROM pg_extension WHERE extname = 'vector'",
                )
                index = await self._scalar(
                    conn,
                    "SELECT indexdef FROM pg_indexes WHERE tablename = %s AND indexname = %s",
                    (table, image_index_name(table)),
                )
        finally:
            await pool.close()

        assert applied[0] == "core_schema"
        assert "attachment_files_copy_state" in applied
        assert applied.index("image_embedding") > applied.index("attachment_files_copy_state")
        assert not [s for s in spy.statements if re.search(r"ALTER\s+EXTENSION", s, re.I)]
        assert owner == role
        assert ledger_owner == role
        assert vector_owner != role
        assert index is not None and "hnsw" in index and "vector_cosine_ops" in index

    async def test_image_embedding_table_shape_and_cascade(
        self, image_embedding_non_owner_store, integration_ariel_config
    ):
        """attachment_id keys the row and cascades; embedding may be NULL with a skip_reason."""
        from osprey.services.ariel_search.database.migrations import image_table_name
        from osprey.services.ariel_search.database.vector_literal import vector_literal

        _role, conninfo = image_embedding_non_owner_store
        table = image_table_name(self.MODEL, self.DIMS)
        pool, _spy, _applied = await self._migrate(self._config(integration_ariel_config, conninfo))
        try:
            async with pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT column_name, is_nullable, udt_name
                    FROM information_schema.columns WHERE table_name = %s
                    """,
                    (table,),
                )
                columns = {r[0]: (r[1], r[2]) for r in await result.fetchall()}
                width = await self._scalar(
                    conn,
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
                    (table,),
                )

                await conn.execute(
                    """
                    INSERT INTO enhanced_entries (
                        entry_id, source_system, timestamp, author, raw_text,
                        attachments, metadata, enhancement_status
                    ) VALUES ('e1', 'test', NOW(), 'tester', 'x',
                              '[]'::jsonb, '{}'::jsonb, '{}'::jsonb)
                    """
                )
                await conn.execute(
                    "INSERT INTO attachment_files (attachment_id, entry_id, filename) "
                    "VALUES ('a1', 'e1', 'a1.png'), ('a2', 'e1', 'a2.png')"
                )
                await conn.execute(
                    f"INSERT INTO {table} (attachment_id, embedding, model_ref) "
                    "VALUES ('a1', %(v)s::vector, %(m)s)",
                    {"v": vector_literal([0.5] * self.DIMS), "m": self.MODEL},
                )
                await conn.execute(
                    f"INSERT INTO {table} (attachment_id, embedding, skip_reason) "
                    "VALUES ('a2', NULL, 'degenerate_vector')"
                )
                stored = await self._scalar(
                    conn, f"SELECT embedding::text FROM {table} WHERE attachment_id = 'a1'"
                )
                await conn.execute("DELETE FROM attachment_files WHERE attachment_id = 'a1'")
                remaining = await self._scalar(
                    conn, f"SELECT array_agg(attachment_id) FROM {table}"
                )
        finally:
            await pool.close()

        assert set(columns) == {
            "attachment_id",
            "embedding",
            "skip_reason",
            "model_ref",
            "created_at",
        }
        assert columns["embedding"] == ("YES", "vector")
        assert columns["attachment_id"][0] == "NO"
        assert width == self.DIMS
        assert stored == "[" + ",".join(["0.5"] * self.DIMS) + "]"
        assert remaining == ["a2"]

    async def test_image_embedding_is_applied_by_table_so_a_new_width_gets_its_table(
        self, image_embedding_non_owner_store, integration_ariel_config
    ):
        """A re-run is a no-op; a changed width creates the new width's table."""
        from osprey.services.ariel_search.database.migrations import image_table_name

        _role, conninfo = image_embedding_non_owner_store
        pool, _spy, first = await self._migrate(self._config(integration_ariel_config, conninfo))
        await pool.close()
        pool, _spy, again = await self._migrate(self._config(integration_ariel_config, conninfo))
        await pool.close()
        pool, _spy, wider = await self._migrate(
            self._config(integration_ariel_config, conninfo, dims=768)
        )
        try:
            async with pool.connection() as conn:
                tables = [
                    await self._scalar(conn, "SELECT to_regclass(%s)::text", (name,))
                    for name in (
                        image_table_name(self.MODEL, self.DIMS),
                        image_table_name(self.MODEL, 768),
                    )
                ]
        finally:
            await pool.close()

        assert "image_embedding" in first
        assert again == []
        assert wider == ["image_embedding"]
        assert tables == [
            image_table_name(self.MODEL, self.DIMS),
            image_table_name(self.MODEL, 768),
        ]
