"""Backfill of the stored ``subject`` column for entries written before it existed.

``entry_subject_backfill`` fills ``enhanced_entries.subject`` for every row where
it is still NULL, with the value
:func:`~osprey.services.ariel_search.subject.derive_subject` gives -- the same
producer the upsert uses, so a backfilled row and a freshly ingested one carry
the same line. Derivation runs in Python, not SQL.

Rows are read in keyset pages by ``entry_id`` (``FOR UPDATE``) and each page is
written with one ``UPDATE ... FROM (VALUES ...)``. The migration runs
``SET LOCAL ariel.preserve_updated_at = 'on'`` once, which the trigger function
from ``entry_subject_and_keyset`` honours, so ``updated_at`` does not move and no
entry looks edited (no qmd resync of the whole logbook). There is no table lock
and no ``DISABLE TRIGGER``; the runner wraps ``up()`` in one transaction, so the
row locks are held until it commits. Only ``subject`` is written: ``metadata``
and ``enhancement_status`` are untouched. An entry with no subject at all gets
``""``, so a second run finds nothing to do.
"""

from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    MigrationSkippedError,
)
from osprey.services.ariel_search.database.subject_migration import (
    PRESERVE_UPDATED_AT_SETTING,
)
from osprey.services.ariel_search.subject import derive_subject
from osprey_connectors.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

logger = get_logger("ariel")

#: Rows read (and locked) per keyset page.
BACKFILL_PAGE_SIZE = 500

_HAS_SUBJECT_COLUMN_SQL = """
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
        AND table_name = 'enhanced_entries'
        AND column_name = 'subject'
    )
"""

_PAGE_SQL = """
    SELECT entry_id, metadata, raw_text
    FROM enhanced_entries
    WHERE subject IS NULL
      AND entry_id > %(after)s
    ORDER BY entry_id
    LIMIT %(limit)s
    FOR UPDATE
"""


def _page_update_sql(rows: int) -> str:
    """One ``UPDATE ... FROM (VALUES ...)`` writing *rows* subjects."""
    values = ", ".join(["(%s::text, %s::text)"] * rows)
    return f"""
        UPDATE enhanced_entries AS e
        SET subject = v.subject
        FROM (VALUES {values}) AS v(entry_id, subject)
        WHERE e.entry_id = v.entry_id
          AND e.subject IS NULL
    """


class EntrySubjectBackfillMigration(BaseMigration):
    """Fills ``subject`` from :func:`derive_subject` where it is NULL -- always runs."""

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "entry_subject_backfill"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the column and the ``updated_at`` guard it relies on."""
        return ["entry_subject_and_keyset"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Backfill every NULL ``subject`` in keyset pages, keeping ``updated_at``.

        Raises:
            MigrationSkippedError: If ``enhanced_entries.subject`` does not exist
                (``entry_subject_and_keyset`` has not run); the runner leaves
                this migration unapplied and retries it on a later run.
        """
        result = await conn.execute(_HAS_SUBJECT_COLUMN_SQL)
        row = await result.fetchone()
        if not (row and row[0]):
            raise MigrationSkippedError("enhanced_entries.subject does not exist yet")

        await conn.execute(f"SET LOCAL {PRESERVE_UPDATED_AT_SETTING} = 'on'")

        backfilled = 0
        after = ""
        while True:
            cursor = await conn.execute(_PAGE_SQL, {"after": after, "limit": BACKFILL_PAGE_SIZE})
            page = await cursor.fetchall()
            if page:
                params: list[Any] = []
                for entry_id, metadata, raw_text in page:
                    params.extend(
                        (entry_id, derive_subject({"metadata": metadata, "raw_text": raw_text}))
                    )
                written = await conn.execute(_page_update_sql(len(page)), params)
                backfilled += max(written.rowcount, 0)
            if len(page) < BACKFILL_PAGE_SIZE:
                break
            after = page[-1][0]
        logger.info(
            f"{self.name}: {backfilled} rows backfilled "
            "(updated_at kept; row locks held until the migration commits)"
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Nothing to undo: the values live in ``subject``.

        Rolling back ``entry_subject_and_keyset`` drops that column; a later
        re-apply of both backfills it again.
        """
