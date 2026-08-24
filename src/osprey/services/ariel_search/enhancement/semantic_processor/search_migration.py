"""Reconcile semantic processor keyword-search indexes.

Deployments that applied the original semantic processor migration already have
its migration record, so they need a separate additive migration to rebuild the
keyword FTS expression and remove the unqueried keyword-array index.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration
from osprey.services.ariel_search.database.search_fts import SEMANTIC_FTS_EXPRESSION

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class SemanticProcessorSearchMigration(BaseMigration):
    """Rebuilds semantic keyword-search indexes for summary and keywords."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "semantic_processor_search_index"

    @property
    def depends_on(self) -> list[str]:
        """Depends on semantic processor columns being present."""
        return ["semantic_processor"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the enriched semantic FTS index migration."""
        await conn.execute(
            """
            CREATE OR REPLACE FUNCTION osprey_text_array_to_string(TEXT[])
            RETURNS TEXT
            LANGUAGE sql
            IMMUTABLE
            PARALLEL SAFE
            RETURNS NULL ON NULL INPUT
            AS $$ SELECT array_to_string($1, ' ') $$
            """
        )
        await conn.execute("DROP INDEX IF EXISTS idx_entries_keywords")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_entries_text_search
            ON enhanced_entries
            USING GIN({SEMANTIC_FTS_EXPRESSION})
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the enriched semantic FTS index migration."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_text_search")
