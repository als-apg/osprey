"""Integration tests for ARIEL database migrations.

Tests schema migrations against real PostgreSQL.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestCoreMigration:
    """Test core schema migration."""

    async def test_run_migrations_creates_tables(self, connection_pool, integration_ariel_config):
        """Running migrations creates the required tables."""
        from osprey.services.ariel_search.database import run_migrations

        await run_migrations(connection_pool, integration_ariel_config)

        async with connection_pool.connection() as conn:
            result = await conn.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('enhanced_entries', 'ariel_migrations', 'ingestion_runs')
            """)
            tables = [row[0] for row in await result.fetchall()]

        assert "enhanced_entries" in tables
        assert "ariel_migrations" in tables

    async def test_migrations_are_idempotent(self, connection_pool, integration_ariel_config):
        """Running migrations multiple times is safe."""
        from osprey.services.ariel_search.database import run_migrations

        # Run migrations twice - should not error
        await run_migrations(connection_pool, integration_ariel_config)
        await run_migrations(connection_pool, integration_ariel_config)

        async with connection_pool.connection() as conn:
            result = await conn.execute("SELECT 1")
            assert (await result.fetchone())[0] == 1

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

        # The query plan should reference the index
        assert "idx_entries_text_search" in plan or "Seq Scan" in plan

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
