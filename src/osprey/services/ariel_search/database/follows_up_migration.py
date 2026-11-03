"""ARIEL ``follows_up_index``: the expression index behind an entry's follow-ups.

A follow-up names its parent in ``metadata.follows_up``. The thread view lists an
entry's children with ``WHERE metadata->>'follows_up' = %s``; this migration adds
``idx_entries_metadata_follows_up`` on that exact expression, so the lookup stays
an index scan on a large store. It is a separate named migration, not a change to
``core_schema``, so a database that already has ``core_schema`` gets the index on
its next run.

The build takes ``SHARE`` through
:func:`~osprey.services.ariel_search.database.attachment_text_migration.build_kept_indexes`,
so a busy table skips the migration instead of queueing readers behind it. Every
statement is idempotent.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.attachment_text_migration import build_kept_indexes
from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    acquire_nonqueueing_lock,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection

#: The index this migration builds, by name.
FOLLOWS_UP_INDEX = "idx_entries_metadata_follows_up"


class FollowsUpIndexMigration(BaseMigration):
    """Adds the ``(metadata->>'follows_up')`` expression index -- always runs."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "follows_up_index"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for ``enhanced_entries``."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Build the index.

        Raises:
            MigrationBusyError: If ``enhanced_entries`` stayed in use for every
                lock attempt; the runner leaves the migration unapplied.
        """
        await build_kept_indexes(
            conn,
            [
                f"""
                CREATE INDEX IF NOT EXISTS {FOLLOWS_UP_INDEX}
                ON enhanced_entries ((metadata->>'follows_up'))
                """,
            ],
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the index."""
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        await conn.execute(f"DROP INDEX IF EXISTS {FOLLOWS_UP_INDEX}")  # type: ignore[arg-type]
