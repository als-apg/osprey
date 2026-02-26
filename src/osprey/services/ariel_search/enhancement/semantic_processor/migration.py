"""ARIEL semantic processor migration.

This module provides the database migration for the semantic processor
enhancement module.
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

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_keywords
            ON enhanced_entries USING GIN(keywords)
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_text_search
            ON enhanced_entries
            USING GIN(to_tsvector('english', raw_text || ' ' || COALESCE(summary, '')))
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the semantic processor migration."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_keywords")

        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS keywords")
        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS summary")
