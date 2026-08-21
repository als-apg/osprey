"""Reconcile the core keyword-search full-text index.

Existing deployments may already have ``core_schema`` marked as applied from
before keyword search had a raw-text FTS index. This additive migration makes
that index appear without mutating the historical migration record.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration
from osprey.services.ariel_search.database.search_fts import RAW_TEXT_FTS_EXPRESSION

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class KeywordSearchFtsMigration(BaseMigration):
    """Adds the always-available raw-text FTS index for keyword search."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "keyword_search_fts_index"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the core enhanced_entries table."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the raw-text FTS index migration."""
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_entries_raw_text_fts
            ON enhanced_entries USING GIN({RAW_TEXT_FTS_EXPRESSION})
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the raw-text FTS index migration."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_raw_text_fts")
