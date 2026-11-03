"""ARIEL ``subject`` column, the list-view indexes and the ``updated_at`` guard.

``entry_subject_and_keyset`` adds what the mail-client list reads:

- ``enhanced_entries.subject`` (TEXT, nullable, no default) -- a catalog-only
  change, no row is rewritten.
- ``idx_entries_timestamp_entry_id`` on ``(timestamp DESC, entry_id DESC)``, the
  keyset order of the entry list.
- ``idx_entries_author_lower`` and ``idx_entries_source_system_lower`` on
  ``lower(author)`` and ``lower(source_system)``, for case-insensitive facets.

It also replaces the ``update_updated_at_column()`` trigger function from
``core_schema`` with one that leaves ``updated_at`` alone while the transaction
has ``SET LOCAL ariel.preserve_updated_at = 'on'``, so a maintenance backfill
can write a column without making every entry look edited. Outside such a
transaction the function sets ``updated_at = NOW()`` exactly as before.

``ALTER TABLE`` takes ``ACCESS EXCLUSIVE`` and the index builds ``SHARE``, both
through :func:`~osprey.services.ariel_search.database.migrations.acquire_nonqueueing_lock`,
so a busy table skips the migration instead of queueing readers behind it.
Every statement is idempotent.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.attachment_text_migration import build_kept_indexes
from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    acquire_nonqueueing_lock,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection

#: Transaction-local setting that makes the ``updated_at`` trigger keep the stored value.
PRESERVE_UPDATED_AT_SETTING = "ariel.preserve_updated_at"

#: The three indexes this migration builds, by name.
SUBJECT_KEYSET_INDEXES: tuple[str, ...] = (
    "idx_entries_timestamp_entry_id",
    "idx_entries_author_lower",
    "idx_entries_source_system_lower",
)

_GUARDED_TRIGGER_FUNCTION_SQL = f"""
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        IF current_setting('{PRESERVE_UPDATED_AT_SETTING}', true) = 'on' THEN
            RETURN NEW;
        END IF;
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql'
"""

_PLAIN_TRIGGER_FUNCTION_SQL = """
    CREATE OR REPLACE FUNCTION update_updated_at_column()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ language 'plpgsql'
"""


class EntrySubjectAndKeysetMigration(BaseMigration):
    """Adds ``subject``, the keyset and facet indexes, and the guarded trigger -- always runs."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "entry_subject_and_keyset"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for the table and its ``updated_at`` trigger."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Add the column, replace the trigger function, then build the indexes.

        Raises:
            MigrationBusyError: If ``enhanced_entries`` stayed in use for every
                lock attempt; the runner leaves the migration unapplied.
        """
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        await conn.execute("ALTER TABLE enhanced_entries ADD COLUMN IF NOT EXISTS subject TEXT")
        await conn.execute(_GUARDED_TRIGGER_FUNCTION_SQL)  # type: ignore[arg-type]
        await build_kept_indexes(
            conn,
            [
                """
                CREATE INDEX IF NOT EXISTS idx_entries_timestamp_entry_id
                ON enhanced_entries (timestamp DESC, entry_id DESC)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_entries_author_lower
                ON enhanced_entries (lower(author))
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_entries_source_system_lower
                ON enhanced_entries (lower(source_system))
                """,
            ],
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the indexes and the column; restore the unguarded trigger function."""
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        for index in SUBJECT_KEYSET_INDEXES:
            await conn.execute(f"DROP INDEX IF EXISTS {index}")  # type: ignore[arg-type]
        await conn.execute("ALTER TABLE enhanced_entries DROP COLUMN IF EXISTS subject")
        await conn.execute(_PLAIN_TRIGGER_FUNCTION_SQL)  # type: ignore[arg-type]
