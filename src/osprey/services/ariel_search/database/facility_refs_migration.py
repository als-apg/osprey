"""Column holding the facility references an entry's text quotes.

A span list indexes the ``raw_text`` of its own row, so whatever rewrites
``raw_text`` rewrites the spans. ``NULL`` means the entry was not checked
against a facility file; an empty list means it was and quotes nothing.

The column is nullable with no default, so adding it is a catalog-only change.
The ``ACCESS EXCLUSIVE`` lock it needs is taken with
:func:`~osprey.services.ariel_search.database.migrations.acquire_nonqueueing_lock`,
which never queues: when the table stays busy the migration is skipped as busy
and retried later.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    acquire_nonqueueing_lock,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class FacilityRefsColumnMigration(BaseMigration):
    """Adds ``enhanced_entries.facility_refs`` (JSONB, nullable) -- always runs."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "facility_refs_column"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for the enhanced_entries table."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Add the column under a lock that never queues.

        Raises:
            MigrationBusyError: If ``enhanced_entries`` stayed in use for every
                lock attempt; the runner leaves the migration unapplied.
        """
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        await conn.execute(
            "ALTER TABLE enhanced_entries ADD COLUMN IF NOT EXISTS facility_refs JSONB"
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the column, under the same non-queueing lock."""
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS facility_refs")
