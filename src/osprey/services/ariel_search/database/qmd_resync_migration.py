"""Index supporting the qmd mirror resync scan.

``osprey ariel qmd-resync`` finds the rows it must re-export by asking for
every ``enhanced_entries`` row whose ``updated_at`` is at or after the stored
watermark. Without an index that predicate is a sequential scan of the whole
table on every ingest and every watch cycle, which at ALS scale (~135k rows)
turns a routine pre-step into the most expensive thing in the loop.

The index is purely additive: it creates no column, changes no existing one,
and rolls back by dropping itself.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class QmdResyncMigration(BaseMigration):
    """Adds the ``enhanced_entries.updated_at`` index the resync scan needs.

    Creates:
    - idx_entries_updated_at (btree on enhanced_entries.updated_at)
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "qmd_resync_index"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for the enhanced_entries table."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the qmd resync index migration."""
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_updated_at
            ON enhanced_entries(updated_at)
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the qmd resync index migration."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_updated_at")
