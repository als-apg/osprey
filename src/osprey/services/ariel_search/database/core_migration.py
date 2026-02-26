"""ARIEL core schema migration.

This module provides the core database schema that is always created,
regardless of which modules are enabled.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migration import BaseMigration

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class CoreMigration(BaseMigration):
    """Core schema migration - always runs.

    Creates:
    - enhanced_entries table (base entry data)
    - ingestion_runs table (track ingestion history)
    - ariel_migrations table (track applied migrations)
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "core_schema"

    @property
    def depends_on(self) -> list[str]:
        """Core schema has no dependencies."""
        return []

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the core schema migration."""
        # pg_trgm extension for fuzzy text search (used by keyword fuzzy fallback)
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enhanced_entries (
                -- Primary identification (facility-specific)
                entry_id        TEXT PRIMARY KEY,
                source_system   TEXT NOT NULL,

                -- Original content (always present)
                timestamp       TIMESTAMPTZ NOT NULL,
                author          TEXT NOT NULL DEFAULT '',
                raw_text        TEXT NOT NULL,
                attachments     JSONB DEFAULT '[]'::jsonb,

                -- Timestamps
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW(),

                -- Facility-specific fields
                metadata        JSONB DEFAULT '{}'::jsonb,

                -- Enhancement tracking (per-module status)
                enhancement_status  JSONB DEFAULT '{}'::jsonb
            )
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_timestamp
            ON enhanced_entries(timestamp DESC)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_author
            ON enhanced_entries(author)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_source
            ON enhanced_entries(source_system)
            """
        )

        # Trigram index for fuzzy matching
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_raw_text_trgm
            ON enhanced_entries USING GIN(raw_text gin_trgm_ops)
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_metadata
            ON enhanced_entries USING GIN(metadata)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_metadata_linked_to
            ON enhanced_entries ((metadata->>'linked_to'))
            """
        )

        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql'
            """
        )

        await conn.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_trigger
                    WHERE tgname = 'update_enhanced_entries_updated_at'
                ) THEN
                    CREATE TRIGGER update_enhanced_entries_updated_at
                        BEFORE UPDATE ON enhanced_entries
                        FOR EACH ROW
                        EXECUTE FUNCTION update_updated_at_column();
                END IF;
            END
            $$
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                id              SERIAL PRIMARY KEY,
                started_at      TIMESTAMPTZ NOT NULL,
                completed_at    TIMESTAMPTZ,
                source_system   TEXT NOT NULL,
                entries_added   INTEGER DEFAULT 0,
                entries_updated INTEGER DEFAULT 0,
                entries_failed  INTEGER DEFAULT 0,
                status          TEXT DEFAULT 'running',
                error_message   TEXT,
                metadata        JSONB DEFAULT '{}'::jsonb
            )
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started
            ON ingestion_runs(started_at DESC)
            """
        )

        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ariel_migrations (
                name        TEXT PRIMARY KEY,
                applied_at  TIMESTAMPTZ DEFAULT NOW(),
                checksum    TEXT
            )
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the core schema migration.

        WARNING: This drops all ARIEL data!
        """
        await conn.execute("DROP TABLE IF EXISTS ariel_migrations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS ingestion_runs CASCADE")
        await conn.execute("DROP TABLE IF EXISTS enhanced_entries CASCADE")
        await conn.execute("DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE")
