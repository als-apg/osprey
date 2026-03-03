"""ARIEL attachment files schema migration.

Creates the attachment_files table for storing file binary data (BYTEA)
alongside logbook entries.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration

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
