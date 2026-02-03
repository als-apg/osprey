"""ARIEL semantic processor migration.

This module provides the database migration for the semantic processor
enhancement module.

See 01_DATA_LAYER.md Section 2.5 for SQL specification.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migration import BaseMigration

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class SemanticProcessorMigration(BaseMigration):
    """Semantic processor enhancement migration.

    Creates:
    - summary column on enhanced_entries
    - keywords column on enhanced_entries
    - pg_trgm extension
    - Full-text search indexes
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "semantic_processor"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the semantic processor migration."""
        # Add columns for semantic processing results
        await conn.execute(
            """
            ALTER TABLE enhanced_entries
            ADD COLUMN IF NOT EXISTS summary TEXT
            """
        )
        await conn.execute(
            """
            ALTER TABLE enhanced_entries
            ADD COLUMN IF NOT EXISTS keywords TEXT[] DEFAULT '{}'
            """
        )

        # Index for keywords array
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_keywords
            ON enhanced_entries USING GIN(keywords)
            """
        )

        # pg_trgm extension for fuzzy text search
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

        # Full-text search index on raw_text and summary
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_text_search
            ON enhanced_entries
            USING GIN(to_tsvector('english', raw_text || ' ' || COALESCE(summary, '')))
            """
        )

        # Trigram index for fuzzy matching
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_text_trgm
            ON enhanced_entries
            USING GIN(raw_text gin_trgm_ops)
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the semantic processor migration."""
        # Drop indexes first
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_trgm")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_keywords")

        # Drop columns
        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS keywords")
        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS summary")

        # Note: We don't drop pg_trgm extension as other things may use it
