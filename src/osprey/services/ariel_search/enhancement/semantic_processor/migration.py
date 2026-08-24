"""ARIEL semantic processor migration.

This module provides the database migration for the semantic processor
enhancement module.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class SemanticProcessorMigration(BaseMigration):
    """Semantic processor enhancement migration.

    Creates:
    - summary column on enhanced_entries
    - keywords column on enhanced_entries
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

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the semantic processor migration."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_keywords")

        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS keywords")
        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS summary")
