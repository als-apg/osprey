"""ARIEL attachment files schema migrations.

``attachment_files`` creates the table storing file binary data (BYTEA)
alongside logbook entries. ``attachment_files_copy_state`` makes that row the
copy state of one picture: whether its bytes were copied, are still pending or
were skipped (and why), plus the viewable rendition encoded from them.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    acquire_nonqueueing_lock,
)

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class AttachmentMigration(BaseMigration):
    """Attachment files migration - always runs.

    Creates:
    - attachment_files table (BYTEA storage for file data)
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "attachment_files"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for enhanced_entries FK."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Apply the attachment files migration."""
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attachment_files (
                attachment_id   TEXT PRIMARY KEY,
                entry_id        TEXT NOT NULL REFERENCES enhanced_entries(entry_id)
                                    ON DELETE CASCADE,
                filename        TEXT NOT NULL,
                mime_type       TEXT,
                data            BYTEA NOT NULL,
                size_bytes      INTEGER NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )

        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_attachment_files_entry_id
            ON attachment_files(entry_id)
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Rollback the attachment files migration."""
        await conn.execute("DROP TABLE IF EXISTS attachment_files CASCADE")


#: The ``skip_reason`` codes the copy-state CHECK admits. Written out as a
#: literal, never derived from the format registry: an applied migration must
#: keep its meaning when the registry grows. A test pins this tuple to the
#: registry's row skip-reason sets, so a new code there fails it and calls for
#: a new migration that replaces the constraint.
COPY_STATE_SKIP_REASONS: tuple[str, ...] = (
    # content: the bytes decide; terminal
    "not_an_image",
    "reserved_format",
    "decoder_failed",
    "format_mismatch",
    "rendition_too_large",
    "not_a_regular_file",
    # config: re-evaluated against the current configuration
    "origin_not_allowed",
    "size_cap",
    "per_entry_limit",
    "copy_on_ingest_mode",
    # source: the source did not deliver; retried by backfill
    "source_gone",
    "source_refused",
    "fetch_failed",
)

#: Name of the CHECK constraint limiting ``skip_reason`` to the codes above.
SKIP_REASON_CONSTRAINT = "attachment_files_skip_reason_known"

#: Partial index over the rows backfill still has work on: pending copies and
#: copied rows with neither a rendition nor a recorded skip.
COPY_TODO_INDEX = "idx_attachment_files_copy_todo"

_SKIP_REASON_SQL = ", ".join(f"'{code}'" for code in COPY_STATE_SKIP_REASONS)


class AttachmentFilesCopyStateMigration(BaseMigration):
    """Copy state and rendition columns on ``attachment_files`` -- always runs.

    Alters attachment_files:
    - data, size_bytes become nullable (a pending or skipped row has no bytes)
    - source_url TEXT
    - copy_status TEXT NOT NULL DEFAULT 'copied' (copied | pending | skipped)
    - skip_reason TEXT, CHECKed against :data:`COPY_STATE_SKIP_REASONS`
    - copy_attempts SMALLINT NOT NULL DEFAULT 0
    - rendition_bytes BYTEA, rendition_mime TEXT, rendition_w INTEGER,
      rendition_h INTEGER, rendition_sha256 TEXT

    Creates the partial index :data:`COPY_TODO_INDEX`.

    Every row stored before this migration holds its bytes, so it reads as
    ``copied`` with no rendition yet. The ``ALTER TABLE`` lock is taken with
    :func:`acquire_nonqueueing_lock`, which never queues behind readers; the
    index is built in the same lock window, as the table is small when this
    runs on an upgrade.
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "attachment_files_copy_state"

    @property
    def depends_on(self) -> list[str]:
        """Needs the table, and the entry text columns.

        Depending on ``attachment_text_columns`` makes "copy_status exists"
        imply "attachment_text exists", so one schema fact covers both.
        """
        return ["attachment_files", "attachment_text_columns"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Alter the table and build the to-do index under a non-queueing lock.

        Raises:
            MigrationBusyError: If ``attachment_files`` stayed in use for every
                lock attempt; the runner leaves the migration unapplied.
        """
        await acquire_nonqueueing_lock(conn, "attachment_files", "ACCESS EXCLUSIVE")
        await conn.execute(
            f"""
            ALTER TABLE attachment_files
                ALTER COLUMN data DROP NOT NULL,
                ALTER COLUMN size_bytes DROP NOT NULL,
                ADD COLUMN IF NOT EXISTS source_url TEXT,
                ADD COLUMN IF NOT EXISTS copy_status TEXT NOT NULL DEFAULT 'copied',
                ADD COLUMN IF NOT EXISTS skip_reason TEXT
                    CONSTRAINT {SKIP_REASON_CONSTRAINT}
                    CHECK (skip_reason IN ({_SKIP_REASON_SQL})),
                ADD COLUMN IF NOT EXISTS copy_attempts SMALLINT NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS rendition_bytes BYTEA,
                ADD COLUMN IF NOT EXISTS rendition_mime TEXT,
                ADD COLUMN IF NOT EXISTS rendition_w INTEGER,
                ADD COLUMN IF NOT EXISTS rendition_h INTEGER,
                ADD COLUMN IF NOT EXISTS rendition_sha256 TEXT
            """
        )
        await conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {COPY_TODO_INDEX}
            ON attachment_files (entry_id)
            WHERE copy_status = 'pending'
               OR (copy_status = 'copied' AND rendition_sha256 IS NULL AND skip_reason IS NULL)
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the added columns and the index, and make the bytes required again.

        Rows without bytes (pending or skipped copies) cannot satisfy the
        restored NOT NULL, so they are deleted; backfill recreates them from
        the entries' attachment lists once the migration is applied again.
        """
        await acquire_nonqueueing_lock(conn, "attachment_files", "ACCESS EXCLUSIVE")
        await conn.execute(f"DROP INDEX IF EXISTS {COPY_TODO_INDEX}")
        await conn.execute("DELETE FROM attachment_files WHERE data IS NULL OR size_bytes IS NULL")
        await conn.execute(
            """
            ALTER TABLE attachment_files
                DROP COLUMN IF EXISTS source_url,
                DROP COLUMN IF EXISTS copy_status,
                DROP COLUMN IF EXISTS skip_reason,
                DROP COLUMN IF EXISTS copy_attempts,
                DROP COLUMN IF EXISTS rendition_bytes,
                DROP COLUMN IF EXISTS rendition_mime,
                DROP COLUMN IF EXISTS rendition_w,
                DROP COLUMN IF EXISTS rendition_h,
                DROP COLUMN IF EXISTS rendition_sha256,
                ALTER COLUMN data SET NOT NULL,
                ALTER COLUMN size_bytes SET NOT NULL
            """
        )
