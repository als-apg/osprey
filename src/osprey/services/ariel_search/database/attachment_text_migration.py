"""Columns holding the searchable text derived from an entry's attachments.

``attachment_text`` is the composed text of an entry's pictures (upstream
captions and machine captions) that search reads next to ``raw_text``;
``attachment_captions`` is the per-attachment, per-model caption store it is
composed from. Both are nullable and carry no default, so adding them is a
catalog-only change: no table rewrite, no row is touched.

``ALTER TABLE`` needs an ``ACCESS EXCLUSIVE`` lock, and a plain request for one
that cannot be granted at once queues and blocks every later reader of
``enhanced_entries``. The lock is therefore taken with
:func:`~osprey.services.ariel_search.database.migrations.acquire_nonqueueing_lock`,
which never queues: when the table stays busy the migration is skipped as
busy and retried later.

The upstream fold then writes ``attachment_text`` for every row whose
attachments already carry a non-empty upstream caption, so those captions are
searchable without waiting for any enhancement module.

``raw_text_fts_index_v2`` then builds the search indexes over both columns: the
v2 full-text index over ``raw_text`` and ``attachment_text`` together, and a
trigram index over ``attachment_text`` so a pattern span over both columns can
combine two index scans. The v1 full-text index is kept next to it.
"""

from typing import TYPE_CHECKING

from osprey.services.ariel_search.attachments.compose import compose_attachment_text
from osprey.services.ariel_search.database.migrations import (
    BaseMigration,
    acquire_nonqueueing_lock,
)
from osprey.services.ariel_search.database.search_fts import (
    ATTACHMENT_TEXT_DOCUMENT,
    RAW_TEXT_FTS_EXPRESSION_V2,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection


class AttachmentTextColumnsMigration(BaseMigration):
    """Adds ``attachment_text`` and ``attachment_captions`` -- always runs.

    Creates:
    - enhanced_entries.attachment_text (TEXT, nullable)
    - enhanced_entries.attachment_captions (JSONB, nullable)
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "attachment_text_columns"

    @property
    def depends_on(self) -> list[str]:
        """Depends on core schema for the enhanced_entries table."""
        return ["core_schema"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Add the two columns under a lock that never queues.

        Raises:
            MigrationBusyError: If ``enhanced_entries`` stayed in use for every
                lock attempt; the runner leaves the migration unapplied.
        """
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        await conn.execute(
            """
            ALTER TABLE enhanced_entries
                ADD COLUMN IF NOT EXISTS attachment_text TEXT,
                ADD COLUMN IF NOT EXISTS attachment_captions JSONB
            """
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the two columns, under the same non-queueing lock."""
        await acquire_nonqueueing_lock(conn, "enhanced_entries", "ACCESS EXCLUSIVE")
        await conn.execute(
            """
            ALTER TABLE enhanced_entries
                DROP COLUMN IF EXISTS attachment_text,
                DROP COLUMN IF EXISTS attachment_captions
            """
        )


logger = get_logger("ariel")

#: Rows of ``enhanced_entries`` holding at least one non-empty upstream caption.
#: The upgrade note's count query uses this same jsonpath.
UPSTREAM_CAPTION_JSONPATH = '$[*] ? (@.caption != null && @.caption != "")'

#: Rows read (and locked) per page of the fold.
FOLD_PAGE_SIZE = 500

#: Enhancement statuses that depend on the searchable text, cleared when it changes.
FOLD_CLEARED_STATUS_KEYS: tuple[str, ...] = ("text_embedding", "qmd_export")

_FOLD_PAGE_SQL = f"""
    SELECT entry_id, attachments, attachment_captions, attachment_text
    FROM enhanced_entries
    WHERE attachments @? '{UPSTREAM_CAPTION_JSONPATH}'
      AND entry_id > %(after)s
    ORDER BY entry_id
    LIMIT %(limit)s
    FOR UPDATE
"""

_FOLD_WRITE_SQL = """
    UPDATE enhanced_entries
    SET attachment_text = %(composed)s
    WHERE entry_id = %(entry_id)s
      AND attachment_text IS DISTINCT FROM %(composed)s
"""

_FOLD_CLEAR_SQL = """
    UPDATE enhanced_entries
    SET enhancement_status = enhancement_status - %(keys)s::text[]
    WHERE entry_id = %(entry_id)s
      AND enhancement_status ?| %(keys)s::text[]
"""


class AttachmentTextUpstreamFoldMigration(BaseMigration):
    """Composes ``attachment_text`` from stored upstream captions -- always runs.

    Pages ``enhanced_entries`` by ``entry_id`` keyset, reading only rows whose
    attachments hold a non-empty upstream caption (a store without any locks
    nothing), each page ``FOR UPDATE``. Every row's text is composed with
    :func:`~osprey.services.ariel_search.attachments.compose.compose_attachment_text`
    from its own attachments and ``attachment_captions`` under the configured
    caption model, so model captions already stored keep their machine-caption
    lines. A row is written only when the composed text is non-empty and differs
    from the stored one; for those rows alone the ``text_embedding`` and
    ``qmd_export`` statuses are cleared, so both re-run on the new text.

    The runner wraps ``up()`` in one transaction, so every touched row stays
    locked until the fold commits; paging bounds memory, not locks.
    """

    def __init__(self, model_id: str | None = None) -> None:
        """Initialize the fold.

        Args:
            model_id: The caption model id (``caption_model_id(config)``) whose
                stored captions are composed next to the upstream ones; None
                when no caption model is configured.
        """
        self.model_id = model_id

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "attachment_text_upstream_fold"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the columns it writes."""
        return ["attachment_text_columns"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Compose and store ``attachment_text`` for every upstream-captioned row."""
        keys = list(FOLD_CLEARED_STATUS_KEYS)
        recomposed = 0
        cleared = 0
        after = ""
        while True:
            cursor = await conn.execute(_FOLD_PAGE_SQL, {"after": after, "limit": FOLD_PAGE_SIZE})
            page = await cursor.fetchall()
            for entry_id, attachments, captions, stored in page:
                composed = compose_attachment_text(entry_id, attachments, captions, self.model_id)
                if not composed or composed == stored:
                    continue
                written = await conn.execute(
                    _FOLD_WRITE_SQL, {"composed": composed, "entry_id": entry_id}
                )
                if written.rowcount != 1:
                    continue
                recomposed += 1
                result = await conn.execute(_FOLD_CLEAR_SQL, {"keys": keys, "entry_id": entry_id})
                cleared += result.rowcount
            if len(page) < FOLD_PAGE_SIZE:
                break
            after = page[-1][0]
        logger.info(
            f"{self.name}: {recomposed} rows recomposed, "
            f"{cleared} text_embedding/qmd_export statuses cleared"
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Nothing to undo: the composed text lives in ``attachment_text``.

        Rolling back ``attachment_text_columns`` drops that column; a later
        re-apply composes it again.
        """


#: ``maintenance_work_mem`` of a v2 index build, set for its transaction only.
INDEX_BUILD_MAINTENANCE_WORK_MEM = "256MB"


async def build_kept_indexes(conn: "AsyncConnection", statements: list[str]) -> None:
    """Run *statements* (index builds) under a ``SHARE`` lock that never queues.

    ``SHARE`` conflicts with no reader, so reads of ``enhanced_entries`` proceed
    during the build while writes wait for it; the lock is taken with
    :func:`~osprey.services.ariel_search.database.migrations.acquire_nonqueueing_lock`,
    so a busy table skips the migration instead of queueing behind a writer.
    ``maintenance_work_mem`` is raised for the migration's transaction alone.

    Raises:
        MigrationBusyError: If ``enhanced_entries`` stayed busy for every attempt.
    """
    await acquire_nonqueueing_lock(conn, "enhanced_entries", "SHARE")
    await conn.execute(f"SET LOCAL maintenance_work_mem = '{INDEX_BUILD_MAINTENANCE_WORK_MEM}'")
    for statement in statements:
        await conn.execute(statement)  # type: ignore[arg-type]


class RawTextFtsIndexV2Migration(BaseMigration):
    """Builds the v2 raw-text search indexes over ``attachment_text`` -- always runs.

    Creates (the v1 ``idx_entries_raw_text_fts`` is kept):
    - idx_entries_raw_text_fts_v2: GIN over ``RAW_TEXT_FTS_EXPRESSION_V2``
    - idx_entries_attachment_text_trgm: trigram GIN over
      ``COALESCE(attachment_text,'')``, next to ``idx_entries_raw_text_trgm``
    """

    @property
    def name(self) -> str:
        """Return migration identifier."""
        return "raw_text_fts_index_v2"

    @property
    def depends_on(self) -> list[str]:
        """Depends on the fold, so the build indexes the folded text once."""
        return ["attachment_text_upstream_fold"]

    async def up(self, conn: "AsyncConnection") -> None:
        """Create both indexes; reads proceed during the build.

        Raises:
            MigrationBusyError: If ``enhanced_entries`` stayed busy for every
                lock attempt; the runner leaves the migration unapplied.
        """
        await build_kept_indexes(
            conn,
            [
                f"""
                CREATE INDEX IF NOT EXISTS idx_entries_raw_text_fts_v2
                ON enhanced_entries USING GIN({RAW_TEXT_FTS_EXPRESSION_V2})
                """,
                f"""
                CREATE INDEX IF NOT EXISTS idx_entries_attachment_text_trgm
                ON enhanced_entries USING GIN(({ATTACHMENT_TEXT_DOCUMENT}) gin_trgm_ops)
                """,
            ],
        )

    async def down(self, conn: "AsyncConnection") -> None:
        """Drop the two v2 indexes; the v1 index stays."""
        await conn.execute("DROP INDEX IF EXISTS idx_entries_raw_text_fts_v2")
        await conn.execute("DROP INDEX IF EXISTS idx_entries_attachment_text_trgm")
