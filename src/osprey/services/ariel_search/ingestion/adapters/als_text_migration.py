"""One-off rewrite of stored ``als_logbook`` rows as plain text.

The ``als_logbook`` adapter stores entry text as plain text. Rows stored before
it did hold the entity-encoded HTML the logbook sent, and the backslash escaping
it adds. This migration cleans those rows once, with the same cleaner the
adapter uses, so a stored row ends up as a fresh ingest of the same entry would
store it.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.database.migrations import BaseMigration
from osprey.services.ariel_search.ingestion.adapters.als import (
    ALS_SOURCE_SYSTEM,
    clean_als_text,
    merge_als_text,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection

logger = get_logger("ariel")

# The predicate is the cleaner's fast path (an ``&``, a ``<``, a backslash before
# a quote mark, or a doubled backslash): no other row can change.
_SELECT_CANDIDATES = r"""
    SELECT entry_id, raw_text, metadata->>'subject'
    FROM enhanced_entries
    WHERE source_system = %s AND raw_text ~ '[&<]|\\[''"\\]'
"""

_REWRITE_ROW = """
    UPDATE enhanced_entries
    SET raw_text = %s,
        metadata = CASE
            WHEN metadata ? 'subject'
            THEN jsonb_set(metadata, '{subject}', to_jsonb(%s::text))
            ELSE metadata
        END,
        enhancement_status = '{}'::jsonb
    WHERE entry_id = %s
"""


def _recleaned(raw_text: str, subject: str | None) -> tuple[str, str | None]:
    """Recompute a stored row's text and subject as the adapter stores them.

    Text that starts with its subject and a blank line is split, cleaned per
    part and merged again; text equal to its subject is the cleaned subject;
    any other text, including a row whose subject was set by explicit source
    metadata, is cleaned as a whole.
    """
    if subject:
        cleaned_subject = clean_als_text(subject)
        prefix = subject + "\n\n"
        if raw_text.startswith(prefix):
            rest = raw_text[len(prefix) :]
            return merge_als_text(cleaned_subject, clean_als_text(rest)), cleaned_subject
        if raw_text == subject:
            return cleaned_subject, cleaned_subject
        return clean_als_text(raw_text), cleaned_subject
    return clean_als_text(raw_text), None


class ALSPlainTextMigration(BaseMigration):
    """Rewrites stored ``als_logbook`` rows as plain text, once.

    Each changed row gets its cleaned ``raw_text`` and ``metadata.subject``, and
    its ``enhancement_status`` is reset to ``{}``. What derives from the text
    then refreshes: the text indexes on the update itself, the qmd mirror
    through the ``updated_at`` trigger, and embeddings and summaries on the next
    enhance pass. A row that is empty without its markup is left as stored.

    The rewrite is one-way: ``down()`` is not implemented.
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "als_logbook_plain_text"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for the enhanced_entries table."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Rewrite every stored row whose text the cleaner changes."""
        cursor = await conn.execute(_SELECT_CANDIDATES, [ALS_SOURCE_SYSTEM])
        rows = await cursor.fetchall()

        rewritten = 0
        left_as_stored = 0
        for entry_id, raw_text, subject in rows:
            new_text, new_subject = _recleaned(raw_text, subject)
            if new_text == raw_text and new_subject == subject:
                continue
            if not new_text.strip():
                left_as_stored += 1
                continue
            await conn.execute(_REWRITE_ROW, [new_text, new_subject, entry_id])
            rewritten += 1

        logger.info(
            f"Rewrote {rewritten} logbook entries as plain text; "
            "their enhancements run again on the next enhance pass"
        )
        if left_as_stored:
            logger.warning(
                f"{left_as_stored} logbook entries left as stored: "
                "their text is empty without markup"
            )
