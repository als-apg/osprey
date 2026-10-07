"""ARIEL repository for database operations.

This module provides the ARIELRepository class with async CRUD operations
for the ARIEL database.
"""

import contextlib
import functools
import hashlib
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from osprey.imaging.formats import (
    CAPTION_NOT_DONE_SQL,
    CONFIG_SKIP_REASONS,
    SOURCE_SKIP_REASONS,
    image_table_not_done_sql,
    viewable_sql,
)
from osprey.services.ariel_search.database.search_fts import (
    ATTACHMENT_TEXT_DOCUMENT,
    FTS_CONFIG,
    keyword_search_expressions,
)
from osprey.services.ariel_search.exceptions import (
    DatabaseQueryError,
    ModuleNotEnabledError,
    PatternError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.models import (
    DiagnosticLevel,
    EmbeddingTableInfo,
    EnhancedLogbookEntry,
    SearchDiagnostic,
    enhanced_entry_from_row,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import ARIELConfig

logger = get_logger("ariel")

F = TypeVar("F", bound=Callable[..., Any])


def _pattern_timeout_error(pattern_timeout_seconds: float | None) -> SearchTimeoutError:
    """Build the error a keyword pattern search reports when it was cancelled.

    PostgreSQL signals the same cancellation under two different error classes
    depending on where the statement was when the timeout fired, so the caller
    catches both and reports them identically -- this keeps the message and the
    reported budget in one place.

    Args:
        pattern_timeout_seconds: The budget the statement was given.

    Returns:
        The timeout error to raise, with a zero budget standing in for a
        statement that carried no configured timeout at all.
    """
    return SearchTimeoutError(
        f"Keyword pattern search exceeded {pattern_timeout_seconds}s",
        timeout_seconds=pattern_timeout_seconds or 0,
        operation="keyword_search",
    )


#: One positional placeholder, or an escaped percent sign that must stay as it is.
_POSITIONAL_PLACEHOLDER = re.compile(r"%%|%s")


def named_tsquery(
    sql: str, params: "list[Any] | tuple[Any, ...]", prefix: str = "tq"
) -> tuple[str, dict[str, Any]]:
    """Rewrite positional ``%s`` placeholders into numbered named ones.

    psycopg refuses a statement that mixes positional and named placeholders,
    so SQL built with ``%s`` (the output of ``build_tsquery`` and
    ``build_expanded_tsquery``, or a run of pattern predicates) is renamed
    before it joins a named-only statement. An escaped ``%%`` is left alone,
    so ``%%s`` is never read as a placeholder.

    Args:
        sql: SQL whose placeholders are all positional ``%s``.
        params: One bind value per placeholder, in placeholder order.
        prefix: Name stem; the placeholders become ``%(<prefix>0)s``,
            ``%(<prefix>1)s`` and so on.

    Returns:
        ``(named_sql, bound)`` with ``bound`` mapping each new name to its value.

    Raises:
        ValueError: If the number of placeholders differs from ``len(params)``.
    """
    values = list(params)
    bound: dict[str, Any] = {}

    def _rename(match: "re.Match[str]") -> str:
        if match.group(0) == "%%":
            return "%%"
        index = len(bound)
        if index >= len(values):
            raise ValueError(f"SQL has more %s placeholders than the {len(values)} params given")
        name = f"{prefix}{index}"
        bound[name] = values[index]
        return f"%({name})s"

    named = _POSITIONAL_PLACEHOLDER.sub(_rename, sql)
    if len(bound) != len(values):
        raise ValueError(
            f"SQL has {len(bound)} %s placeholders but {len(values)} params were given"
        )
    return named, bound


def requires_module(module_type: str, module_name: str) -> Callable[[F], F]:
    """Decorator that checks if required module is enabled.

    Args:
        module_type: Type of module ('search' or 'enhancement')
        module_name: Name of the module

    Returns:
        Decorated function that raises ModuleNotEnabledError if module disabled
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: "ARIELRepository", *args: Any, **kwargs: Any) -> Any:
            if module_type == "search":
                enabled = self.config.is_search_module_enabled(module_name)
            elif module_type == "enhancement":
                enabled = self.config.is_enhancement_module_enabled(module_name)
            else:
                enabled = False

            if not enabled:
                raise ModuleNotEnabledError(
                    f"Module '{module_name}' is not enabled. "
                    f"Enable it in config.yml under {module_type}_modules.{module_name}",
                    module_name=module_name,
                )
            return func(self, *args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


#: Failed attempts after which an entry leaves a module's backfill. The count is kept in the
#: module's status object, and a success clears it.
MAX_ENHANCEMENT_ATTEMPTS = 3

#: Seconds a negative schema probe is trusted before the schema is probed again. A positive
#: fact is never re-probed: migrations only ever add these objects.
SCHEMA_FACTS_NEGATIVE_TTL_SECONDS = 60.0


class SchemaFacts(NamedTuple):
    """Optional schema objects a pool's database carries, probed once per pool.

    Attributes:
        has_v2_fts: The v2 search indexes exist -- ``idx_entries_raw_text_fts_v2`` and
            ``idx_entries_attachment_text_trgm``, plus ``idx_entries_text_search_v2``
            when ``semantic_processor`` is enabled.
        has_copy_state: ``attachment_files.copy_status`` exists, so attachment copy
            state can be recorded.
    """

    has_v2_fts: bool
    has_copy_state: bool


#: The ``enhancement_status`` keys of the modules whose input includes ``attachment_text``.
#: Their completion is marked only while the entry still holds the picture text they read.
TEXT_MD5_MODULES: tuple[str, ...] = ("text_embedding", "qmd_export")


def attachment_text_md5(entry: Mapping[str, Any]) -> str:
    """Return the hex md5 of an entry's ``attachment_text`` (empty when absent).

    It equals Postgres ``md5(COALESCE(attachment_text, ''))`` on a UTF-8 database.
    """
    text = entry.get("attachment_text") or ""
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def text_mark_kwargs(
    module_name: str, entry: Mapping[str, Any], has_copy_state: bool
) -> dict[str, str]:
    """Return the keyword arguments of :meth:`ARIELRepository.mark_enhancement_complete`.

    A text module on a store with copy state is marked under the md5 of the
    ``attachment_text`` it read from ``entry``; every other mark takes no keyword.
    """
    if has_copy_state and module_name in TEXT_MD5_MODULES:
        return {"md5": attachment_text_md5(entry)}
    return {}


#: The ``enhancement_status`` keys of the two image modules. A picture whose
#: rendition changes invalidates both, so every writer of a rendition clears them.
IMAGE_STATUS_KEYS: tuple[str, ...] = ("image_caption", "image_embedding")


#: Drops ``%(keys)s`` from one entry's ``enhancement_status``; the ``?|`` guard
#: leaves an entry carrying none of them unwritten.
_CLEAR_STATUS_KEYS_SQL = """
    UPDATE enhanced_entries
    SET enhancement_status = enhancement_status - %(keys)s::text[]
    WHERE entry_id = %(entry_id)s
    AND enhancement_status ?| %(keys)s::text[]
"""


def image_not_done_sql(module_name: str, marker: str) -> str | None:
    """The "picture not done" fragment of an image module, or None for any other module.

    ``image_caption`` reads :data:`CAPTION_NOT_DONE_SQL` (binding ``%(model)s``
    to the marker, the caption model id); ``image_embedding`` reads the
    image table named by its marker.

    Raises:
        ValueError: If the ``image_embedding`` marker is not a plain SQL identifier.
    """
    if module_name == "image_caption":
        return CAPTION_NOT_DONE_SQL
    if module_name == "image_embedding":
        return image_table_not_done_sql(marker)
    return None


#: The marker-aware "still owed" predicate over ``enhanced_entries e``: no key, a
#: key written under another marker, or a pending or failed status not given up.
#: Binds ``%(module)s`` and ``%(marker)s``.
_MARKER_INCOMPLETE_SQL = (
    "(NOT (e.enhancement_status ? %(module)s)"
    " OR e.enhancement_status->%(module)s->>'marker' IS DISTINCT FROM %(marker)s"
    " OR (e.enhancement_status->%(module)s->>'status' IN ('pending', 'failed')"
    " AND NOT COALESCE((e.enhancement_status->%(module)s->>'gave_up')::boolean, false)))"
)

#: Entries the set-based image-module mark selects and locks per batch.
IMAGE_MARK_BATCH_SIZE = 1000


def image_completion_sql(module_name: str, marker: str) -> str:
    """The completion predicate of an image module over ``enhanced_entries e``.

    True when the entry has no picture still being copied and no viewable
    picture the module has not done under ``marker``. An entry with no
    pictures, or only skipped and non-viewable ones, is complete. The
    per-entry and the set-based mark both read this one fragment.

    Args:
        module_name: ``image_caption`` or ``image_embedding``.
        marker: The module's current marker (caption model id, image table).

    Returns:
        A SQL fragment binding the parameters of :func:`image_mark_params`.

    Raises:
        ValueError: If the module is not an image module, or the
            ``image_embedding`` marker is not a plain SQL identifier.
    """
    not_done = image_not_done_sql(module_name, marker)
    if not_done is None:
        raise ValueError(f"not an image module: {module_name!r}")
    return (
        "NOT EXISTS (SELECT 1 FROM attachment_files f"
        " WHERE f.entry_id = e.entry_id"
        f" AND (f.copy_status = 'pending' OR ({viewable_sql('f')} AND {not_done})))"
    )


def image_mark_params(module_name: str, marker: str) -> dict[str, Any]:
    """The named parameters :func:`image_completion_sql` and the mark bind."""
    params: dict[str, Any] = {
        "module": module_name,
        "marker": marker,
        "path": [module_name],
    }
    if module_name == "image_caption":
        params["model"] = marker
    return params


#: The complete object an image-module mark writes under the current marker.
_IMAGE_COMPLETE_SET_SQL = (
    "enhancement_status = jsonb_set(e.enhancement_status, %(path)s::text[],"
    " jsonb_build_object('status', 'complete', 'completed_at', NOW()::text,"
    " 'marker', %(marker)s::text))"
)


#: The module's effective status under the current marker: a stale marker reads as
#: ``pending`` and a missing key as ``pending``. Binds ``%(module)s`` and ``%(marker)s``.
_EFFECTIVE_STATUS_SQL = (
    "CASE WHEN e.enhancement_status->%(module)s->>'marker' IS DISTINCT FROM %(marker)s"
    " THEN 'pending'"
    " ELSE COALESCE(e.enhancement_status->%(module)s->>'status', 'pending') END"
)

#: The module's effective attempt count: 0 under a stale marker or a missing count.
_EFFECTIVE_ATTEMPTS_SQL = (
    "CASE WHEN e.enhancement_status->%(module)s->>'marker' IS DISTINCT FROM %(marker)s"
    " THEN 0"
    " ELSE COALESCE((e.enhancement_status->%(module)s->>'attempts')::int, 0) END"
)

#: Attempts after a failure: a stale marker restarts the count at 1.
_NEXT_ATTEMPTS_SQL = (
    "(CASE WHEN enhancement_status->%(module)s->>'marker' IS DISTINCT FROM %(marker)s"
    " THEN 1"
    " ELSE COALESCE((enhancement_status->%(module)s->>'attempts')::int, 0) + 1 END)"
)

#: Skip codes an outcome may overwrite: config codes are re-evaluated against the
#: current configuration and source codes are retried. Content codes are terminal.
REDECIDABLE_SKIP_REASONS: tuple[str, ...] = tuple(sorted(CONFIG_SKIP_REASONS | SOURCE_SKIP_REASONS))

#: The rows the copy path still has work on -- pending copies and copied rows with
#: neither a rendition nor a recorded skip. Written exactly as the predicate of
#: ``idx_attachment_files_copy_todo`` so the planner can use that partial index.
COPY_TODO_PREDICATE = (
    "copy_status = 'pending' OR "
    "(copy_status = 'copied' AND rendition_sha256 IS NULL AND skip_reason IS NULL)"
)

#: The blob-free columns of an ``attachment_files`` row that attachment readers return.
#: Neither ``data`` nor ``rendition_bytes`` is listed; a reader that needs a blob adds it.
ATTACHMENT_ROW_COLUMNS: tuple[str, ...] = (
    "attachment_id",
    "entry_id",
    "filename",
    "mime_type",
    "size_bytes",
    "source_url",
    "copy_status",
    "skip_reason",
    "copy_attempts",
    "rendition_mime",
    "rendition_w",
    "rendition_h",
    "rendition_sha256",
)

#: Logged once per process by a read path that finds the attachment copy state missing.
ATTACHMENT_SCHEMA_GAP_WARNING = (
    "attachment copy state is missing from the ARIEL database: attachment summaries "
    "fall back to the stored items and renditions are unavailable until "
    "`osprey ariel migrate` and `osprey ariel attachments backfill` run"
)

#: Set once :data:`ATTACHMENT_SCHEMA_GAP_WARNING` has been logged in this process.
_attachment_schema_gap_warned = False


def warn_attachment_schema_gap_once() -> None:
    """Log :data:`ATTACHMENT_SCHEMA_GAP_WARNING` the first time it is called in a process.

    For read paths that have no diagnostics key of their own to report the missing
    copy state through -- the attachment readers when the store is unmigrated, and
    their callers when a reader fails.
    """
    global _attachment_schema_gap_warned
    if _attachment_schema_gap_warned:
        return
    _attachment_schema_gap_warned = True
    logger.warning(ATTACHMENT_SCHEMA_GAP_WARNING)


async def read_attachment_rows(
    repository: Any, entry_ids: list[str]
) -> dict[str, list[dict[str, Any]]] | None:
    """Read the attachment rows of ``entry_ids``, a failing reader read as a store without copy state.

    A ``DatabaseQueryError`` from ``repository.get_attachment_rows`` logs the
    schema-gap warning once and returns ``None``, so the caller's entries keep
    their fallback summaries and a failing reader never costs a result.

    Args:
        repository: The ARIEL repository the rows are read from.
        entry_ids: The entries whose rows are read.

    Returns:
        The rows per entry id, or ``None`` when the store holds no copy state.

    Raises:
        TypeError: If the reader returned something other than None or a dict.
    """
    try:
        mapping = await repository.get_attachment_rows(entry_ids)
    except DatabaseQueryError:
        warn_attachment_schema_gap_once()
        return None
    if mapping is not None and not isinstance(mapping, dict):
        raise TypeError(
            f"get_attachment_rows must return None or a dict, got {type(mapping).__name__}"
        )
    return mapping


#: The diagnostic a search reports while any schema fact is false.
SCHEMA_BEHIND_SEARCH_MESSAGE = "schema behind code: run osprey ariel migrate"

#: The diagnostic source of :data:`SCHEMA_BEHIND_SEARCH_MESSAGE`.
SCHEMA_BEHIND_SEARCH_SOURCE = "ariel.schema"

#: Set once :data:`SCHEMA_BEHIND_SEARCH_MESSAGE` has been logged in this process.
_schema_behind_search_warned = False


def warn_schema_behind_once() -> None:
    """Log :data:`SCHEMA_BEHIND_SEARCH_MESSAGE` the first time it is called in a process.

    For read paths that have no diagnostics key to report the behind schema through.
    """
    global _schema_behind_search_warned
    if _schema_behind_search_warned:
        return
    _schema_behind_search_warned = True
    logger.warning(SCHEMA_BEHIND_SEARCH_MESSAGE)


async def schema_behind_diagnostics(repository: object) -> list[SearchDiagnostic]:
    """Return the schema-behind WARNING for a search answered by `repository`.

    Args:
        repository: The repository that answered the search. Anything without an
            awaitable ``schema_facts`` reports nothing.

    Returns:
        One WARNING diagnostic while any schema fact is false, else an empty list.
    """
    probe = getattr(repository, "schema_facts", None)
    if probe is None:
        return []
    try:
        facts = await probe()
    except Exception as e:  # a store we cannot ask is not reported as behind.
        logger.debug(f"schema facts unavailable for the search diagnostic: {e}")
        return []
    if not isinstance(facts, tuple) or all(facts):
        return []
    return [
        SearchDiagnostic(
            level=DiagnosticLevel.WARNING,
            source=SCHEMA_BEHIND_SEARCH_SOURCE,
            message=SCHEMA_BEHIND_SEARCH_MESSAGE,
        )
    ]


async def image_embedding_table_names(cur: Any) -> list[str]:
    """Return the name of every image-embedding table in the store, sorted.

    The one lookup of the ``image_embeddings_*`` tables, shared by the status
    listing and both purge paths. Matched on the literal prefix, never a
    ``LIKE`` pattern, so ``_`` is not a wildcard and no other table can match.

    Args:
        cur: An open async psycopg cursor.

    Returns:
        Table names in ``public`` that start with the image-table prefix.
    """
    from osprey.services.ariel_search.database.migrations import _IMAGE_TABLE_PREFIX

    await cur.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND left(table_name, %(n)s) = %(prefix)s
        ORDER BY table_name
        """,
        {"n": len(_IMAGE_TABLE_PREFIX), "prefix": _IMAGE_TABLE_PREFIX},
    )
    return [row[0] for row in await cur.fetchall()]


class CopyRendition(NamedTuple):
    """A prepared rendition of a stored picture, as written to ``attachment_files``.

    Attributes:
        data: The rendition bytes.
        mime_type: The rendition's MIME type.
        width: Width in pixels.
        height: Height in pixels.
        sha256: Hex SHA-256 of ``data``.
    """

    data: bytes
    mime_type: str
    width: int
    height: int
    sha256: str


def _rendition_params(rendition: CopyRendition | None) -> dict[str, Any]:
    """Map a rendition (or its absence) onto the ``rendition_*`` named parameters."""
    if rendition is None:
        return {
            "rendition_bytes": None,
            "rendition_mime": None,
            "rendition_w": None,
            "rendition_h": None,
            "rendition_sha256": None,
        }
    return {
        "rendition_bytes": rendition.data,
        "rendition_mime": rendition.mime_type,
        "rendition_w": rendition.width,
        "rendition_h": rendition.height,
        "rendition_sha256": rendition.sha256,
    }


class ARIELRepository:
    """Repository for ARIEL database operations.

    Provides async CRUD operations for enhanced logbook entries.
    Methods are available based on enabled modules in config.

    Attributes:
        pool: Database connection pool
        config: ARIEL configuration
    """

    def __init__(self, pool: "AsyncConnectionPool", config: "ARIELConfig") -> None:
        """Initialize the repository.

        Args:
            pool: Database connection pool
            config: ARIEL configuration
        """
        self.pool = pool
        self.config = config
        self._schema_facts: SchemaFacts | None = None
        self._schema_probed_at: float | None = None
        #: Monotonic clock the negative-probe TTL is measured on; tests replace it.
        self._clock: Callable[[], float] = time.monotonic

    async def schema_facts(self) -> SchemaFacts:
        """Return which optional schema objects this pool's database carries.

        A fact once seen true stays true for the life of the repository. While any
        fact is false the schema is re-probed, at most once every
        :data:`SCHEMA_FACTS_NEGATIVE_TTL_SECONDS`. A probe that fails is logged and
        counts as a negative probe, so a caller degrades to the old-schema path
        instead of failing.

        Returns:
            The cached or freshly probed facts.
        """
        cached = self._schema_facts
        if cached is not None and all(cached):
            return cached
        now = self._clock()
        if (
            cached is not None
            and self._schema_probed_at is not None
            and now - self._schema_probed_at < SCHEMA_FACTS_NEGATIVE_TTL_SECONDS
        ):
            return cached

        probed = await self._probe_schema_facts()
        if cached is not None:
            probed = SchemaFacts(*(old or new for old, new in zip(cached, probed, strict=True)))
        self._schema_facts = probed
        self._schema_probed_at = now
        return probed

    def invalidate_schema_facts(self) -> None:
        """Drop the cached schema facts so the next :meth:`schema_facts` re-probes.

        Called after an in-process migration applied something on this pool, so a
        newly created object is seen at once rather than after the negative TTL.
        """
        self._schema_facts = None
        self._schema_probed_at = None

    async def _probe_schema_facts(self) -> SchemaFacts:
        """Probe the database for every schema fact in one statement.

        Returns:
            The probed facts; all false when the probe fails.
        """
        fts_checks = [
            "to_regclass('idx_entries_raw_text_fts_v2') IS NOT NULL",
            "to_regclass('idx_entries_attachment_text_trgm') IS NOT NULL",
        ]
        if self.config.is_enhancement_module_enabled("semantic_processor"):
            fts_checks.append("to_regclass('idx_entries_text_search_v2') IS NOT NULL")
        sql = f"""
            SELECT
                ({" AND ".join(fts_checks)}) AS has_v2_fts,
                EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = current_schema()
                    AND table_name = 'attachment_files'
                    AND column_name = 'copy_status'
                ) AS has_copy_state
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(sql)
                row = await result.fetchone()
        except Exception as e:  # a failed probe degrades to the old-schema path.
            logger.warning(f"Schema probe failed, treating optional schema as absent: {e}")
            return SchemaFacts(has_v2_fts=False, has_copy_state=False)
        if not row:
            return SchemaFacts(has_v2_fts=False, has_copy_state=False)
        return SchemaFacts(has_v2_fts=bool(row[0]), has_copy_state=bool(row[1]))

    async def get_entry(self, entry_id: str) -> EnhancedLogbookEntry | None:
        """Get a single entry by ID.

        Args:
            entry_id: The entry ID

        Returns:
            EnhancedLogbookEntry or None if not found
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT * FROM enhanced_entries WHERE entry_id = %s",
                        [entry_id],
                    )
                    row = await cur.fetchone()
                    return enhanced_entry_from_row(row) if row else None
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get entry {entry_id}: {e}",
                query=f"SELECT entry_id={entry_id}",
            ) from e

    async def get_entries_by_ids(self, entry_ids: list[str]) -> list[EnhancedLogbookEntry]:
        """Get multiple entries by their IDs.

        Args:
            entry_ids: List of entry IDs

        Returns:
            List of EnhancedLogbookEntry (may be fewer than requested if some not found)
        """
        from psycopg.rows import dict_row

        if not entry_ids:
            return []

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        "SELECT * FROM enhanced_entries WHERE entry_id = ANY(%s)",
                        [entry_ids],
                    )
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get entries by IDs: {e}",
                query=f"SELECT entry_ids=ANY([{len(entry_ids)} ids])",
            ) from e

    async def upsert_entry(self, entry: EnhancedLogbookEntry) -> None:
        """Insert or update an entry.

        Args:
            entry: The entry to upsert
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO enhanced_entries (
                        entry_id, source_system, timestamp, author, raw_text,
                        attachments, metadata, enhancement_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (entry_id) DO UPDATE SET
                        source_system = EXCLUDED.source_system,
                        timestamp = EXCLUDED.timestamp,
                        author = EXCLUDED.author,
                        raw_text = EXCLUDED.raw_text,
                        -- Preserve existing attachments when the incoming upsert
                        -- carries none. A background re-ingestion poll re-fetches an
                        -- already-published entry from the upstream logbook, which has
                        -- no attachments (the adapter write contract carries no
                        -- uploads), so a blind overwrite would erase ARIEL-native
                        -- web-uploaded attachments and orphan their stored blobs. A
                        -- non-empty incoming list still replaces (upstream wins when it
                        -- actually has data).
                        attachments = CASE
                            WHEN EXCLUDED.attachments = '[]'::jsonb
                            THEN enhanced_entries.attachments
                            ELSE EXCLUDED.attachments
                        END,
                        metadata = EXCLUDED.metadata
                    """,
                    [
                        entry["entry_id"],
                        entry["source_system"],
                        entry["timestamp"],
                        entry.get("author", ""),
                        entry["raw_text"],
                        json.dumps(entry.get("attachments", [])),
                        json.dumps(entry.get("metadata", {})),
                        json.dumps(entry.get("enhancement_status", {})),
                    ],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to upsert entry {entry['entry_id']}: {e}",
                query=f"UPSERT entry_id={entry['entry_id']}",
            ) from e

    async def upsert_entry_returning(
        self,
        entry: EnhancedLogbookEntry,
        *,
        conn: "AsyncConnection | None" = None,
    ) -> dict[str, Any]:
        """Insert or update an entry and return its stored picture columns.

        The upsert is the first statement of its transaction, so it takes the
        entry row lock before any ``attachment_files`` row is touched; the row
        stays locked until the caller's transaction ends. An empty incoming
        attachment list keeps the stored list. A non-empty one replaces it, and
        every stored native item (url exactly ``/api/attachments/<id>``) whose
        url the incoming list does not carry is appended after it, in stored
        order, so a re-ingest never drops a web upload.

        Args:
            entry: The entry to upsert.
            conn: Optional caller connection; the transaction nests as a savepoint.

        Returns:
            The stored ``attachments``, ``attachment_text`` and
            ``attachment_captions`` after the write.
        """
        from psycopg.rows import dict_row

        query = f"UPSERT RETURNING entry_id={entry['entry_id']}"
        params = {
            "entry_id": entry["entry_id"],
            "source_system": entry["source_system"],
            "timestamp": entry["timestamp"],
            "author": entry.get("author", ""),
            "raw_text": entry["raw_text"],
            "attachments": json.dumps(entry.get("attachments", [])),
            "metadata": json.dumps(entry.get("metadata", {})),
            "enhancement_status": json.dumps(entry.get("enhancement_status", {})),
        }
        try:
            async with self._connection(conn) as c, c.transaction():
                async with c.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        """
                        INSERT INTO enhanced_entries (
                            entry_id, source_system, timestamp, author, raw_text,
                            attachments, metadata, enhancement_status
                        ) VALUES (
                            %(entry_id)s, %(source_system)s, %(timestamp)s, %(author)s,
                            %(raw_text)s, %(attachments)s, %(metadata)s,
                            %(enhancement_status)s
                        )
                        ON CONFLICT (entry_id) DO UPDATE SET
                            source_system = EXCLUDED.source_system,
                            timestamp = EXCLUDED.timestamp,
                            author = EXCLUDED.author,
                            raw_text = EXCLUDED.raw_text,
                            attachments = CASE
                                WHEN EXCLUDED.attachments = '[]'::jsonb
                                THEN enhanced_entries.attachments
                                WHEN jsonb_typeof(enhanced_entries.attachments) <> 'array'
                                    OR jsonb_typeof(EXCLUDED.attachments) <> 'array'
                                THEN EXCLUDED.attachments
                                ELSE EXCLUDED.attachments || COALESCE((
                                    SELECT jsonb_agg(prev.item ORDER BY prev.pos)
                                    FROM jsonb_array_elements(enhanced_entries.attachments)
                                        WITH ORDINALITY AS prev(item, pos)
                                    WHERE jsonb_typeof(prev.item) = 'object'
                                    AND prev.item->>'url' ~ '^/api/attachments/[^/?#]+$'
                                    AND NOT EXISTS (
                                        SELECT 1
                                        FROM jsonb_array_elements(EXCLUDED.attachments)
                                            AS inc(item)
                                        WHERE jsonb_typeof(inc.item) = 'object'
                                        AND inc.item->>'url' = prev.item->>'url'
                                    )
                                ), '[]'::jsonb)
                            END,
                            metadata = EXCLUDED.metadata
                        RETURNING attachments, attachment_text, attachment_captions
                        """,
                        params,
                    )
                    row = await cur.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to upsert entry {entry['entry_id']}: {e}",
                query=query,
            ) from e
        if row is None:
            raise DatabaseQueryError(
                f"Upsert of entry {entry['entry_id']} returned no row", query=query
            )
        return {
            "attachments": row["attachments"],
            "attachment_text": row["attachment_text"],
            "attachment_captions": row["attachment_captions"],
        }

    async def search_by_time_range(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
        author: str | None = None,
        source_system: str | None = None,
    ) -> list[EnhancedLogbookEntry]:
        """Get entries within a time range, optionally filtered and paginated.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            limit: Maximum entries to return
            offset: Number of entries to skip (for pagination)
            author: Restrict to a single author (exact match)
            source_system: Restrict to a single source system (exact match)

        Returns:
            List of EnhancedLogbookEntry sorted by timestamp descending
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    conditions = []
                    params: list[Any] = []

                    if start is not None:
                        conditions.append("timestamp >= %s")
                        params.append(start)
                    if end is not None:
                        conditions.append("timestamp <= %s")
                        params.append(end)
                    if author is not None:
                        conditions.append("author = %s")
                        params.append(author)
                    if source_system is not None:
                        conditions.append("source_system = %s")
                        params.append(source_system)

                    where_clause = " AND ".join(conditions) if conditions else "TRUE"
                    params.append(limit)
                    params.append(offset)

                    await cur.execute(
                        f"""
                        SELECT * FROM enhanced_entries
                        WHERE {where_clause}
                        ORDER BY timestamp DESC
                        LIMIT %s OFFSET %s
                        """,
                        params,
                    )
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to search by time range: {e}",
                query=f"SELECT time_range=({start}, {end})",
            ) from e

    async def count_entries(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        author: str | None = None,
        source_system: str | None = None,
    ) -> int:
        """Count entries, applying the same optional filters as ``search_by_time_range``.

        Passing no filters counts every entry. The filters mirror
        :meth:`search_by_time_range` so a paginated listing's ``total`` (and the
        ``total_pages`` derived from it) matches the rows the filtered query
        actually returns.

        Args:
            start: Start of time range (inclusive)
            end: End of time range (inclusive)
            author: Restrict to a single author (exact match)
            source_system: Restrict to a single source system (exact match)

        Returns:
            Number of entries matching the filters
        """
        try:
            conditions: list[str] = []
            params: list[object] = []
            if start is not None:
                conditions.append("timestamp >= %s")
                params.append(start)
            if end is not None:
                conditions.append("timestamp <= %s")
                params.append(end)
            if author is not None:
                conditions.append("author = %s")
                params.append(author)
            if source_system is not None:
                conditions.append("source_system = %s")
                params.append(source_system)

            where_clause = " AND ".join(conditions) if conditions else "TRUE"

            async with self.pool.connection() as conn:
                result = await conn.execute(
                    f"SELECT COUNT(*) FROM enhanced_entries WHERE {where_clause}",
                    params,
                )
                row = await result.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to count entries: {e}",
                query="SELECT COUNT(*)",
            ) from e

    async def get_distinct_authors(self) -> list[str]:
        """Get distinct author values from the database.

        Returns:
            Sorted list of unique author names
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    "SELECT DISTINCT author FROM enhanced_entries "
                    "WHERE author IS NOT NULL AND author != '' "
                    "ORDER BY author"
                )
                rows = await result.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get distinct authors: {e}",
                query="SELECT DISTINCT author",
            ) from e

    async def get_distinct_source_systems(self) -> list[str]:
        """Get distinct source_system values from the database.

        Returns:
            Sorted list of unique source system names
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    "SELECT DISTINCT source_system FROM enhanced_entries "
                    "WHERE source_system IS NOT NULL AND source_system != '' "
                    "ORDER BY source_system"
                )
                rows = await result.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get distinct source systems: {e}",
                query="SELECT DISTINCT source_system",
            ) from e

    # === Attachment Methods ===

    async def store_attachment(
        self,
        entry_id: str,
        attachment_id: str,
        filename: str,
        mime_type: str | None,
        data: bytes,
        size_bytes: int,
    ) -> None:
        """Store an attachment file in the database.

        Args:
            entry_id: The entry this attachment belongs to.
            attachment_id: Unique attachment identifier.
            filename: Original filename.
            mime_type: MIME type (e.g. "image/png").
            data: Raw file bytes.
            size_bytes: Size of data in bytes.
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO attachment_files
                        (attachment_id, entry_id, filename, mime_type, data, size_bytes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [attachment_id, entry_id, filename, mime_type, data, size_bytes],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to store attachment {attachment_id}: {e}",
                query=f"INSERT attachment_files attachment_id={attachment_id}",
            ) from e

    # === Attachment Copy-State Methods ===
    #
    # Lock order for every writer: the ``enhanced_entries`` row before any
    # ``attachment_files`` row. Every statement uses named placeholders only.

    @contextlib.asynccontextmanager
    async def _connection(self, conn: "AsyncConnection | None") -> AsyncIterator["AsyncConnection"]:
        """Yield ``conn`` when the caller supplied one, else a pool connection."""
        if conn is not None:
            yield conn
            return
        async with self.pool.connection() as own:
            yield own

    @staticmethod
    async def lock_entry(conn: "AsyncConnection", entry_id: str) -> bool:
        """Take the row lock on an entry, the first step of every copy-state write.

        Args:
            conn: Connection inside an open transaction.
            entry_id: The entry to lock.

        Returns:
            True when the entry exists (and is now locked), False when it is gone.
        """
        result = await conn.execute(
            "SELECT 1 FROM enhanced_entries WHERE entry_id = %(entry_id)s FOR UPDATE",
            {"entry_id": entry_id},
        )
        return await result.fetchone() is not None

    @staticmethod
    async def clear_image_status_keys(conn: "AsyncConnection", entry_id: str) -> bool:
        """Drop the two image-module keys from an entry's ``enhancement_status``.

        Guarded by ``?|``, so an entry carrying neither key is not rewritten.

        Args:
            conn: Connection; the caller holds the entry lock when inside a write.
            entry_id: The entry whose image status is invalidated.

        Returns:
            True when a key was removed.
        """
        result = await conn.execute(
            _CLEAR_STATUS_KEYS_SQL, {"entry_id": entry_id, "keys": list(IMAGE_STATUS_KEYS)}
        )
        return result.rowcount > 0

    @staticmethod
    async def clear_text_status_keys(conn: "AsyncConnection", entry_id: str) -> None:
        """Drop the keys of the modules that read ``attachment_text`` from ``enhancement_status``.

        Called whenever an entry's ``attachment_text`` changes, so those
        modules run again on the new text. Guarded by ``?|``, so an entry
        carrying neither key is not rewritten.

        Args:
            conn: Connection; the caller holds the entry lock when inside a write.
            entry_id: The entry whose text status is invalidated.
        """
        await conn.execute(
            _CLEAR_STATUS_KEYS_SQL, {"entry_id": entry_id, "keys": list(TEXT_MD5_MODULES)}
        )

    @staticmethod
    async def delete_dropped_attachments(
        conn: "AsyncConnection", entry_id: str, keep: list[str]
    ) -> list[str]:
        """Delete an entry's copied-from-source rows whose url left the list.

        Rows of every status go, so a pending or skipped row for a dropped url is
        never retried. Native rows (``source_url`` NULL) are never touched. An empty
        ``keep`` deletes every non-native row of the entry.

        Args:
            conn: Connection holding the entry lock.
            entry_id: The entry.
            keep: The source urls whose rows stay.

        Returns:
            The deleted attachment ids.
        """
        result = await conn.execute(
            """
            DELETE FROM attachment_files
            WHERE entry_id=%(entry_id)s AND source_url IS NOT NULL
            AND source_url <> ALL(%(keep)s)
            RETURNING attachment_id
            """,
            {"entry_id": entry_id, "keep": list(keep)},
        )
        return [row[0] for row in await result.fetchall()]

    async def apply_copy_outcome(
        self,
        entry_id: str,
        attachment_id: str,
        *,
        copy_status: str,
        data: bytes | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        skip_reason: str | None = None,
        rendition: CopyRendition | None = None,
        copy_attempts: int | None = None,
        conn: "AsyncConnection | None" = None,
    ) -> str | None:
        """Write one picture's copy outcome in its own transaction.

        Locks the entry row, then updates the attachment row only while it is
        ``pending`` or carries a re-decidable (config or source) skip code. The
        original and its rendition are written together. When a rendition was
        written, the two image-module status keys are cleared.

        Args:
            entry_id: The entry owning the attachment.
            attachment_id: The attachment row to decide.
            copy_status: The new ``copy_status``.
            data: The stored original, or None.
            mime_type: The sniffed or declared MIME type.
            size_bytes: Stored or observed size.
            skip_reason: A bare skip code, or None.
            rendition: The prepared rendition, or None when not viewable.
            copy_attempts: New attempt count; None leaves it unchanged.
            conn: Optional caller connection; the transaction nests as a savepoint.

        Returns:
            The written ``copy_status``, or None when the row is gone or already
            decided. Zero rows is never an error and never re-inserts a row.
        """
        params: dict[str, Any] = {
            "entry_id": entry_id,
            "attachment_id": attachment_id,
            "copy_status": copy_status,
            "data": data,
            "mime_type": mime_type,
            "size_bytes": size_bytes,
            "skip_reason": skip_reason,
            "copy_attempts": copy_attempts,
            "reasons": list(REDECIDABLE_SKIP_REASONS),
            **_rendition_params(rendition),
        }
        try:
            async with self._connection(conn) as c, c.transaction():
                await self.lock_entry(c, entry_id)
                result = await c.execute(
                    """
                    UPDATE attachment_files SET
                        data = %(data)s,
                        mime_type = %(mime_type)s,
                        size_bytes = %(size_bytes)s,
                        copy_status = %(copy_status)s,
                        skip_reason = %(skip_reason)s,
                        copy_attempts = COALESCE(%(copy_attempts)s::smallint, copy_attempts),
                        rendition_bytes = %(rendition_bytes)s,
                        rendition_mime = %(rendition_mime)s,
                        rendition_w = %(rendition_w)s,
                        rendition_h = %(rendition_h)s,
                        rendition_sha256 = %(rendition_sha256)s
                    WHERE attachment_id = %(attachment_id)s
                    AND (copy_status = 'pending' OR skip_reason = ANY(%(reasons)s))
                    RETURNING attachment_id, copy_status
                    """,
                    params,
                )
                row = await result.fetchone()
                if row is None:
                    return None
                if rendition is not None:
                    await self.clear_image_status_keys(c, entry_id)
                return str(row[1])
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to record copy outcome for {attachment_id}: {e}",
                query=f"UPDATE attachment_files outcome attachment_id={attachment_id}",
            ) from e

    async def apply_render_outcome(
        self,
        entry_id: str,
        attachment_id: str,
        *,
        mime_type: str | None,
        skip_reason: str | None = None,
        rendition: CopyRendition | None = None,
        conn: "AsyncConnection | None" = None,
    ) -> bool:
        """Write the rendition of a ``copied`` row that has none yet.

        Same transaction shape as :meth:`apply_copy_outcome`; never touches ``data``
        or ``copy_status``.

        Args:
            entry_id: The entry owning the attachment.
            attachment_id: The copied row to render.
            mime_type: The sniffed MIME type of the stored original.
            skip_reason: A content skip code when the render failed, else None.
            rendition: The prepared rendition, or None.
            conn: Optional caller connection; the transaction nests as a savepoint.

        Returns:
            True when the row was updated; False when it is gone or already
            rendered or skipped.
        """
        params: dict[str, Any] = {
            "entry_id": entry_id,
            "attachment_id": attachment_id,
            "mime_type": mime_type,
            "skip_reason": skip_reason,
            **_rendition_params(rendition),
        }
        try:
            async with self._connection(conn) as c, c.transaction():
                await self.lock_entry(c, entry_id)
                result = await c.execute(
                    """
                    UPDATE attachment_files SET
                        rendition_bytes = %(rendition_bytes)s,
                        rendition_mime = %(rendition_mime)s,
                        rendition_w = %(rendition_w)s,
                        rendition_h = %(rendition_h)s,
                        rendition_sha256 = %(rendition_sha256)s,
                        skip_reason = %(skip_reason)s,
                        mime_type = %(mime_type)s
                    WHERE attachment_id = %(attachment_id)s
                    AND copy_status = 'copied'
                    AND rendition_sha256 IS NULL
                    AND skip_reason IS NULL
                    RETURNING attachment_id
                    """,
                    params,
                )
                if await result.fetchone() is None:
                    return False
                if rendition is not None:
                    await self.clear_image_status_keys(c, entry_id)
                return True
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to record render outcome for {attachment_id}: {e}",
                query=f"UPDATE attachment_files render attachment_id={attachment_id}",
            ) from e

    async def insert_native_attachment(
        self,
        entry_id: str,
        attachment_id: str,
        *,
        filename: str,
        mime_type: str | None,
        data: bytes,
        skip_reason: str | None = None,
        rendition: CopyRendition | None = None,
        conn: "AsyncConnection | None" = None,
    ) -> None:
        """Store a natively written attachment as a ``copied`` row in one transaction.

        Locks the entry row, inserts the row with its rendition (``source_url``
        NULL marks it native), then clears the two image-module status keys.

        Args:
            entry_id: The entry the attachment belongs to.
            attachment_id: The attachment id.
            filename: Original filename.
            mime_type: The sniffed MIME type.
            data: The original bytes; always kept.
            skip_reason: A content skip code (non-image, reserved format), or None.
            rendition: The prepared rendition, or None when not rendered.
            conn: Optional caller connection; the transaction nests as a savepoint.
        """
        params: dict[str, Any] = {
            "entry_id": entry_id,
            "attachment_id": attachment_id,
            "filename": filename,
            "mime_type": mime_type,
            "data": data,
            "size_bytes": len(data),
            "skip_reason": skip_reason,
            **_rendition_params(rendition),
        }
        try:
            async with self._connection(conn) as c, c.transaction():
                await self.lock_entry(c, entry_id)
                await c.execute(
                    """
                    INSERT INTO attachment_files (
                        attachment_id, entry_id, filename, mime_type, data, size_bytes,
                        source_url, copy_status, skip_reason,
                        rendition_bytes, rendition_mime, rendition_w, rendition_h,
                        rendition_sha256
                    ) VALUES (
                        %(attachment_id)s, %(entry_id)s, %(filename)s, %(mime_type)s,
                        %(data)s, %(size_bytes)s,
                        NULL, 'copied', %(skip_reason)s,
                        %(rendition_bytes)s, %(rendition_mime)s, %(rendition_w)s,
                        %(rendition_h)s, %(rendition_sha256)s
                    )
                    """,
                    params,
                )
                await self.clear_image_status_keys(c, entry_id)
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to store native attachment {attachment_id}: {e}",
                query=f"INSERT attachment_files native attachment_id={attachment_id}",
            ) from e

    async def count_copied_attachments(
        self, entry_id: str, conn: "AsyncConnection | None" = None
    ) -> tuple[int, int]:
        """Return the per-entry copy budget already used.

        Args:
            entry_id: The entry.
            conn: Optional caller connection.

        Returns:
            ``(count, total_bytes)`` of the entry's ``copied`` rows.
        """
        try:
            async with self._connection(conn) as c:
                result = await c.execute(
                    """
                    SELECT count(*), COALESCE(sum(size_bytes), 0)
                    FROM attachment_files
                    WHERE entry_id = %(entry_id)s AND copy_status = 'copied'
                    """,
                    {"entry_id": entry_id},
                )
                row = await result.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to count copied attachments for {entry_id}: {e}",
                query=f"SELECT count attachment_files entry_id={entry_id}",
            ) from e
        return (int(row[0]), int(row[1])) if row else (0, 0)

    async def get_copy_source(self, attachment_id: str) -> tuple[bytes, str | None] | None:
        """Read the stored original of a render-only candidate.

        Args:
            attachment_id: The attachment.

        Returns:
            ``(data, mime_type)`` when the row is ``copied`` with stored bytes, no
            rendition and no skip; otherwise None.
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT data, mime_type FROM attachment_files
                    WHERE attachment_id = %(attachment_id)s
                    AND copy_status = 'copied'
                    AND rendition_sha256 IS NULL
                    AND skip_reason IS NULL
                    AND data IS NOT NULL
                    """,
                    {"attachment_id": attachment_id},
                )
                row = await result.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to read copy source {attachment_id}: {e}",
                query=f"SELECT attachment_files source attachment_id={attachment_id}",
            ) from e
        return (bytes(row[0]), row[1]) if row else None

    async def get_copy_rows(self, entry_id: str) -> list[dict[str, Any]]:
        """Read an entry's attachment rows with their copy state and no blobs.

        Args:
            entry_id: The entry.

        Returns:
            One dict per row: ``attachment_id``, ``source_url``, ``filename``,
            ``mime_type``, ``size_bytes``, ``copy_status``, ``skip_reason``,
            ``copy_attempts``, ``rendition_sha256``, ``has_data``, ``created_at``.
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        """
                        SELECT attachment_id, source_url, filename, mime_type, size_bytes,
                               copy_status, skip_reason, copy_attempts, rendition_sha256,
                               data IS NOT NULL AS has_data, created_at
                        FROM attachment_files
                        WHERE entry_id = %(entry_id)s
                        ORDER BY created_at, attachment_id
                        """,
                        {"entry_id": entry_id},
                    )
                    return [dict(row) for row in await cur.fetchall()]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to read copy rows for {entry_id}: {e}",
                query=f"SELECT attachment_files copy rows entry_id={entry_id}",
            ) from e

    async def get_attachment_rows(
        self, entry_ids: list[str]
    ) -> dict[str, list[dict[str, Any]]] | None:
        """Read the attachment rows of several entries in one statement, without blobs.

        Args:
            entry_ids: The entries.

        Returns:
            ``{entry_id: [row, ...]}`` with each row keyed by :data:`ATTACHMENT_ROW_COLUMNS`;
            an entry with no rows is absent. ``{}`` for no entries, before the schema is
            consulted. None when the store lacks the copy state.
        """
        from psycopg.rows import dict_row

        if not entry_ids:
            return {}
        if not (await self.schema_facts()).has_copy_state:
            warn_attachment_schema_gap_once()
            return None

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        f"""
                        SELECT {", ".join(ATTACHMENT_ROW_COLUMNS)}
                        FROM attachment_files
                        WHERE entry_id = ANY(%(entry_ids)s)
                        ORDER BY entry_id, created_at, attachment_id
                        """,
                        {"entry_ids": list(entry_ids)},
                    )
                    rows = await cur.fetchall()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to read attachment rows: {e}",
                query=f"SELECT attachment_files rows entry_ids=ANY([{len(entry_ids)} ids])",
            ) from e
        mapping: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            mapping.setdefault(row["entry_id"], []).append(dict(row))
        return mapping

    async def get_rendition(self, attachment_id: str) -> dict[str, Any] | None:
        """Read an attachment's rendition with its row columns, never the original.

        Args:
            attachment_id: The attachment.

        Returns:
            The row keyed by :data:`ATTACHMENT_ROW_COLUMNS` plus ``rendition_bytes``, or
            None when there is no such row, it has no rendition, or the store lacks the
            copy state.
        """
        from psycopg.rows import dict_row

        if not (await self.schema_facts()).has_copy_state:
            warn_attachment_schema_gap_once()
            return None

        columns = (*ATTACHMENT_ROW_COLUMNS, "rendition_bytes")
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(
                        f"""
                        SELECT {", ".join(columns)}
                        FROM attachment_files
                        WHERE attachment_id = %(attachment_id)s
                        AND rendition_bytes IS NOT NULL
                        """,
                        {"attachment_id": attachment_id},
                    )
                    row = await cur.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to read rendition {attachment_id}: {e}",
                query=f"SELECT attachment_files rendition attachment_id={attachment_id}",
            ) from e
        return dict(row) if row else None

    async def get_attachment_original(self, attachment_id: str) -> dict[str, Any] | None:
        """Read an attachment's stored original, never its rendition.

        Args:
            attachment_id: The attachment.

        Returns:
            ``filename``, ``mime_type`` and ``data`` of a ``copied`` row with stored
            bytes, or None. When the store lacks the copy state, any row with that id is
            read as stored, so natively uploaded originals stay downloadable.
        """
        from psycopg.rows import dict_row

        if (await self.schema_facts()).has_copy_state:
            sql = """
                SELECT filename, mime_type, data FROM attachment_files
                WHERE attachment_id = %(id)s
                AND copy_status = 'copied'
                AND data IS NOT NULL
            """
        else:
            warn_attachment_schema_gap_once()
            sql = "SELECT filename, mime_type, data FROM attachment_files WHERE attachment_id = %(id)s"
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(sql, {"id": attachment_id})
                    row = await cur.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to read attachment original {attachment_id}: {e}",
                query=f"SELECT attachment_files original attachment_id={attachment_id}",
            ) from e
        return dict(row) if row else None

    async def get_copy_retry_candidates(
        self, after: tuple[datetime, str] | None, limit: int
    ) -> list[tuple[datetime, str]]:
        """List entries holding copy work, newest first below a keyset cursor.

        An entry qualifies when it has a pending row or a copied row with neither a
        rendition nor a skip.

        Args:
            after: ``(timestamp, entry_id)`` of the last entry already visited;
                None starts from the newest.
            limit: Maximum entries returned.

        Returns:
            ``(timestamp, entry_id)`` pairs; the last one is the next cursor.
        """
        params: dict[str, Any] = {"limit": limit}
        cursor_sql = ""
        if after is not None:
            cursor_sql = "AND (e.timestamp, e.entry_id) < (%(after_ts)s, %(after_id)s)"
            params["after_ts"], params["after_id"] = after
        sql = f"""
            SELECT e.timestamp, e.entry_id
            FROM enhanced_entries e
            WHERE EXISTS (
                SELECT 1 FROM attachment_files
                WHERE attachment_files.entry_id = e.entry_id
                AND ({COPY_TODO_PREDICATE})
            )
            {cursor_sql}
            ORDER BY e.timestamp DESC, e.entry_id DESC
            LIMIT %(limit)s
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(sql, params)
                rows = await result.fetchall()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to list copy retry candidates: {e}",
                query="SELECT enhanced_entries copy retry candidates",
            ) from e
        return [(row[0], row[1]) for row in rows]

    async def get_attachment_copy_counts(self) -> tuple[int, dict[str, int]]:
        """Count pending rows and rows per skip code across the store.

        Returns:
            ``(pending, {skip_code: count})``; the dict holds every code present on
            any row, whatever its ``copy_status``.
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT copy_status, skip_reason, count(*)
                    FROM attachment_files
                    GROUP BY copy_status, skip_reason
                    """
                )
                rows = await result.fetchall()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to count attachment copy states: {e}",
                query="SELECT attachment_files GROUP BY copy_status, skip_reason",
            ) from e
        pending = 0
        skipped: dict[str, int] = {}
        for status, reason, count in rows:
            if status == "pending":
                pending += int(count)
            if reason is not None:
                skipped[reason] = skipped.get(reason, 0) + int(count)
        return pending, skipped

    async def get_attachment_bytes(self) -> int:
        """Return the total on-disk size of ``attachment_files`` (with TOAST and indexes)."""
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute("SELECT pg_total_relation_size('attachment_files')")
                row = await result.fetchone()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to size attachment_files: {e}",
                query="SELECT pg_total_relation_size attachment_files",
            ) from e
        return int(row[0]) if row else 0

    # === Enhancement Status Methods ===
    async def get_incomplete_entries(
        self,
        module_name: str | None = None,
        status: str | None = None,
        limit: int = 100,
        *,
        marker: str | None = None,
    ) -> list[EnhancedLogbookEntry]:
        """Get entries with incomplete or failed enhancements.

        With ``module_name`` alone, an entry is returned when the module never ran on it, when
        its status is ``pending``, or when it is ``failed`` and has failed fewer than
        ``MAX_ENHANCEMENT_ATTEMPTS`` times. With ``status`` too, every entry in that state is
        returned.

        With ``module_name`` and ``marker`` (and no ``status``), the module's status is read
        against the current marker: an entry is returned when the key is missing, when its
        stored marker differs, or when it is ``pending``/``failed`` and has not given up. The
        catch-up order puts pending entries first (newest first), then failed ones by
        attempts; a stale marker counts as pending with no attempts. For an image module the
        entry must also hold a viewable picture that is not done, so an entry whose only
        unfinished pictures are still being copied stays incomplete but is not walked.

        Args:
            module_name: Filter by specific module (optional)
            status: Filter by status ('failed', 'pending') (optional)
            limit: Maximum entries to return
            marker: The module's current completion marker (optional)

        Returns:
            List of entries needing enhancement

        Raises:
            ValueError: If an ``image_embedding`` marker is not a plain SQL identifier.
        """
        from psycopg.rows import dict_row

        if module_name and marker is not None and not status:
            return await self._get_incomplete_entries_for_marker(module_name, marker, limit)

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    if module_name and status:
                        await cur.execute(
                            """
                            SELECT * FROM enhanced_entries
                            WHERE enhancement_status->%s->>'status' = %s
                            ORDER BY created_at ASC
                            LIMIT %s
                            """,
                            [module_name, status, limit],
                        )
                    elif module_name:
                        await cur.execute(
                            """
                            SELECT * FROM enhanced_entries
                            WHERE NOT (enhancement_status ? %s)
                               OR enhancement_status->%s->>'status' = 'pending'
                               OR (enhancement_status->%s->>'status' = 'failed'
                                   AND COALESCE((enhancement_status->%s->>'attempts')::int, 0)
                                       < %s)
                            ORDER BY created_at ASC
                            LIMIT %s
                            """,
                            [
                                module_name,
                                module_name,
                                module_name,
                                module_name,
                                MAX_ENHANCEMENT_ATTEMPTS,
                                limit,
                            ],
                        )
                    else:
                        await cur.execute(
                            """
                            SELECT * FROM enhanced_entries
                            ORDER BY created_at ASC
                            LIMIT %s
                            """,
                            [limit],
                        )
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get incomplete entries: {e}",
                query=f"SELECT incomplete module={module_name}",
            ) from e

    async def _get_incomplete_entries_for_marker(
        self, module_name: str, marker: str, limit: int
    ) -> list[EnhancedLogbookEntry]:
        """The marker-aware branch of :meth:`get_incomplete_entries`."""
        from psycopg.rows import dict_row

        not_done = image_not_done_sql(module_name, marker)
        picture_clause = ""
        if not_done is not None:
            picture_clause = f"""
                      AND EXISTS (
                          SELECT 1 FROM attachment_files f
                          WHERE f.entry_id = e.entry_id
                            AND {viewable_sql("f")}
                            AND {not_done}
                      )"""
        sql = f"""
                    SELECT e.* FROM enhanced_entries e
                    WHERE {_MARKER_INCOMPLETE_SQL}
                      {picture_clause}
                    ORDER BY {_EFFECTIVE_STATUS_SQL} = 'failed',
                             {_EFFECTIVE_ATTEMPTS_SQL},
                             e.timestamp DESC,
                             e.entry_id
                    LIMIT %(limit)s
                    """
        params: dict[str, Any] = {"module": module_name, "marker": marker, "limit": limit}
        if module_name == "image_caption":
            params["model"] = marker
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(sql, params)
                    rows = await cur.fetchall()
                    return [enhanced_entry_from_row(row) for row in rows]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get incomplete entries: {e}",
                query=f"SELECT incomplete module={module_name}",
            ) from e

    async def get_enhancement_stats(
        self, markers: Mapping[str, str] | None = None
    ) -> dict[str, Any]:
        """Get statistics about enhancement completion.

        The per-module counts come from one grouped aggregate over
        ``jsonb_each(enhancement_status)``, so the SQL text is fixed and no
        module name is ever spliced into it. A module the store has never seen
        simply has no rows; one a facility registered itself is reported the
        moment its first entry lands, without a code change here.

        ``pending`` keeps the meaning the two hand-written FILTER clauses gave
        it: entries whose status for the module is ``pending``, PLUS entries
        that carry no key for the module at all. The second half cannot be
        counted per module by the aggregate — an entry without the key produces
        no row for it — so it is derived from the total instead. That subtraction
        is why the total rides along in the same statement: two statements on an
        autocommit connection read two snapshots, and an ingest landing between
        them would drive ``pending`` negative. The ``LEFT JOIN ... ON TRUE``
        keeps one row even when no entry carries a status key, so the total is
        still readable from a store that has nothing to group.

        ``markers`` maps a marker module to its current completion marker. For
        those modules any entry whose stored marker differs counts as
        ``pending`` whatever its stored status, and their counts gain
        ``gave_up`` -- the failed entries that stopped retrying, a subset of
        ``failed``. Module names and markers are bound as parameters. A module
        absent from ``markers`` keeps the three-key counts.

        Args:
            markers: Current marker per marker module (optional).

        Returns:
            ``total_entries`` plus, per module, a ``{complete, failed,
            pending}`` count (``gave_up`` too for marker modules). An empty
            store returns ``total_entries`` alone.
        """
        if markers:
            return await self._get_enhancement_stats_for_markers(markers)
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    WITH total AS (
                        SELECT COUNT(*) AS entries FROM enhanced_entries
                    ),
                    per_module AS (
                        SELECT
                            status.key AS module,
                            status.value->>'status' AS state,
                            COUNT(*) AS entries
                        FROM enhanced_entries
                        CROSS JOIN LATERAL jsonb_each(enhancement_status) AS status
                        GROUP BY status.key, status.value->>'status'
                    )
                    SELECT
                        total.entries AS total_entries,
                        per_module.module,
                        per_module.state,
                        per_module.entries
                    FROM total
                    LEFT JOIN per_module ON TRUE
                    """
                )
                rows = await result.fetchall()

                total_entries = rows[0][0] if rows else 0

                by_module: dict[str, dict[str, int]] = {}
                seen: dict[str, int] = {}
                for _total, module, state, entries in rows:
                    if module is None:
                        continue
                    counts = by_module.setdefault(
                        module, {"complete": 0, "failed": 0, "pending": 0}
                    )
                    if state in counts:
                        counts[state] += entries
                    seen[module] = seen.get(module, 0) + entries

                stats: dict[str, Any] = {"total_entries": total_entries}
                for module, counts in by_module.items():
                    counts["pending"] += total_entries - seen[module]
                    stats[module] = counts
                return stats
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get enhancement stats: {e}",
                query="SELECT enhancement_stats",
            ) from e

    async def _get_enhancement_stats_for_markers(
        self, markers: Mapping[str, str]
    ) -> dict[str, Any]:
        """The marker-aware branch of :meth:`get_enhancement_stats`."""
        params: dict[str, Any] = {}
        stale_arms: list[str] = []
        current_arms: list[str] = []
        for i, (module, marker) in enumerate(markers.items()):
            params[f"module_{i}"] = module
            params[f"marker_{i}"] = marker
            stale_arms.append(
                f"WHEN status.key = %(module_{i})s"
                f" AND status.value->>'marker' IS DISTINCT FROM %(marker_{i})s THEN 'pending'"
            )
            current_arms.append(
                f"(status.key = %(module_{i})s"
                f" AND status.value->>'marker' IS NOT DISTINCT FROM %(marker_{i})s)"
            )
        sql = f"""
                    WITH total AS (
                        SELECT COUNT(*) AS entries FROM enhanced_entries
                    ),
                    per_module AS (
                        SELECT
                            status.key AS module,
                            CASE {" ".join(stale_arms)}
                                 ELSE status.value->>'status' END AS state,
                            COUNT(*) AS entries,
                            COUNT(*) FILTER (
                                WHERE status.value->>'status' = 'failed'
                                  AND COALESCE((status.value->>'gave_up')::boolean, false)
                                  AND ({" OR ".join(current_arms)})
                            ) AS gave_up
                        FROM enhanced_entries
                        CROSS JOIN LATERAL jsonb_each(enhancement_status) AS status
                        GROUP BY 1, 2
                    )
                    SELECT
                        total.entries AS total_entries,
                        per_module.module,
                        per_module.state,
                        per_module.entries,
                        per_module.gave_up
                    FROM total
                    LEFT JOIN per_module ON TRUE
                    """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(sql, params)
                rows = await result.fetchall()
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get enhancement stats: {e}",
                query="SELECT enhancement_stats",
            ) from e

        total_entries = rows[0][0] if rows else 0
        by_module: dict[str, dict[str, int]] = {}
        seen: dict[str, int] = {}
        for _total, module, state, entries, gave_up in rows:
            if module is None:
                continue
            counts = by_module.setdefault(module, {"complete": 0, "failed": 0, "pending": 0})
            if module in markers:
                counts.setdefault("gave_up", 0)
                counts["gave_up"] += gave_up or 0
            if state in ("complete", "failed", "pending"):
                counts[state] += entries
            seen[module] = seen.get(module, 0) + entries

        stats: dict[str, Any] = {"total_entries": total_entries}
        for module, counts in by_module.items():
            counts["pending"] += total_entries - seen[module]
            stats[module] = counts
        return stats

    async def mark_enhancement_complete(
        self,
        entry_id: str,
        module_name: str,
        *,
        marker: str | None = None,
        md5: str | None = None,
    ) -> None:
        """Mark an enhancement as complete for an entry.

        Args:
            entry_id: The entry ID
            module_name: The enhancement module name
            marker: The module's current completion marker, stored in the
                complete object when given (optional)
            md5: The md5 of the ``attachment_text`` the module read (optional). When
                given, the entry is marked only while its ``attachment_text`` still
                hashes to it; otherwise nothing is written and the module stays owed.

        Raises:
            ValueError: If both ``marker`` and ``md5`` are given.
        """
        if md5 is not None:
            if marker is not None:
                raise ValueError("mark_enhancement_complete takes marker or md5, not both")
            try:
                async with self.pool.connection() as conn:
                    await conn.execute(
                        """
                        UPDATE enhanced_entries
                        SET enhancement_status = jsonb_set(
                            enhancement_status,
                            ARRAY[%(module)s::text],
                            jsonb_build_object('status', 'complete', 'completed_at', NOW()::text)
                        )
                        WHERE entry_id = %(entry_id)s
                          AND md5(COALESCE(attachment_text, '')) = %(md5)s
                        """,
                        {"module": module_name, "entry_id": entry_id, "md5": md5},
                    )
            except Exception as e:
                raise DatabaseQueryError(
                    f"Failed to mark enhancement complete: {e}",
                    query=f"UPDATE entry_id={entry_id} module={module_name}",
                ) from e
            return
        if marker is not None:
            try:
                async with self.pool.connection() as conn:
                    await conn.execute(
                        """
                        UPDATE enhanced_entries
                        SET enhancement_status = jsonb_set(
                            enhancement_status,
                            %(path)s::text[],
                            jsonb_build_object(
                                'status', 'complete',
                                'completed_at', NOW()::text,
                                'marker', %(marker)s::text
                            )
                        )
                        WHERE entry_id = %(entry_id)s
                        """,
                        {"path": [module_name], "entry_id": entry_id, "marker": marker},
                    )
            except Exception as e:
                raise DatabaseQueryError(
                    f"Failed to mark enhancement complete: {e}",
                    query=f"UPDATE entry_id={entry_id} module={module_name}",
                ) from e
            return
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE enhanced_entries
                    SET enhancement_status = jsonb_set(
                        enhancement_status,
                        %s,
                        jsonb_build_object('status', 'complete', 'completed_at', NOW()::text)
                    )
                    WHERE entry_id = %s
                    """,
                    [[module_name], entry_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark enhancement complete: {e}",
                query=f"UPDATE entry_id={entry_id} module={module_name}",
            ) from e

    async def mark_image_module_complete(
        self,
        entry_id: str,
        module_name: str,
        marker: str,
        *,
        conn: "AsyncConnection | None" = None,
    ) -> bool:
        """Mark an image module complete for one entry, only while it is complete.

        One transaction: the entry row lock, then an ``UPDATE`` guarded by
        :func:`image_completion_sql`, so a picture whose copy finished, or a
        native writer that committed while the mark waited for the lock, keeps
        the module owed.

        Args:
            entry_id: The entry to mark.
            module_name: ``image_caption`` or ``image_embedding``.
            marker: The module's current marker, stored in the complete object.
            conn: Optional caller connection; the transaction nests as a savepoint.

        Returns:
            True when the entry was marked complete.

        Raises:
            ValueError: If the module is not an image module or the marker is
                not a plain SQL identifier where one is spliced.
        """
        completion = image_completion_sql(module_name, marker)
        params = {**image_mark_params(module_name, marker), "entry_id": entry_id}
        try:
            async with self._connection(conn) as c, c.transaction():
                if not await self.lock_entry(c, entry_id):
                    return False
                result = await c.execute(
                    f"""
                    UPDATE enhanced_entries e
                    SET {_IMAGE_COMPLETE_SET_SQL}
                    WHERE e.entry_id = %(entry_id)s
                      AND {completion}
                    """,
                    params,
                )
                return result.rowcount > 0
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark image module complete: {e}",
                query=f"UPDATE image mark entry_id={entry_id} module={module_name}",
            ) from e

    async def mark_image_module_complete_batch(
        self,
        module_name: str,
        marker: str,
        *,
        after: str = "",
        limit: int = IMAGE_MARK_BATCH_SIZE,
        conn: "AsyncConnection | None" = None,
    ) -> list[str]:
        """Mark complete, set-based, the next batch of owed entries with nothing left to do.

        Selects (``FOR UPDATE SKIP LOCKED``) up to ``limit`` entries past the
        keyset cursor ``after`` that are still owed under ``marker`` and satisfy
        :func:`image_completion_sql`, then marks them with the same predicate
        re-checked, in one transaction. Entries already complete under the
        current marker never match, so the caller's loop ends.

        Args:
            module_name: ``image_caption`` or ``image_embedding``.
            marker: The module's current marker.
            after: Keyset cursor: only entry ids sorting after it are visited.
                ``''`` on the first call, then the last id the previous call
                returned.
            limit: Batch size.
            conn: Optional caller connection; the transaction nests as a savepoint.

        Returns:
            The marked entry ids in cursor order, so the last one is the next
            ``after``. Fewer than ``limit`` means the walk is done.

        Raises:
            ValueError: If the module is not an image module or the marker is
                not a plain SQL identifier where one is spliced.
        """
        completion = image_completion_sql(module_name, marker)
        params = {**image_mark_params(module_name, marker), "after": after, "limit": limit}
        try:
            async with self._connection(conn) as c, c.transaction():
                result = await c.execute(
                    f"""
                    SELECT e.entry_id FROM enhanced_entries e
                    WHERE {_MARKER_INCOMPLETE_SQL}
                      AND {completion}
                      AND e.entry_id > %(after)s
                    ORDER BY e.entry_id
                    LIMIT %(limit)s
                    FOR UPDATE SKIP LOCKED
                    """,
                    params,
                )
                picked = [str(row[0]) for row in await result.fetchall()]
                if not picked:
                    return []
                result = await c.execute(
                    f"""
                    UPDATE enhanced_entries e
                    SET {_IMAGE_COMPLETE_SET_SQL}
                    WHERE e.entry_id = ANY(%(ids)s)
                      AND {completion}
                    RETURNING e.entry_id
                    """,
                    {**params, "ids": picked},
                )
                marked = {str(row[0]) for row in await result.fetchall()}
                return [entry_id for entry_id in picked if entry_id in marked]
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to batch-mark image module complete: {e}",
                query=f"UPDATE image batch mark module={module_name}",
            ) from e

    async def mark_enhancement_failed(
        self,
        entry_id: str,
        module_name: str,
        error: str,
        *,
        marker: str | None = None,
    ) -> int:
        """Mark an enhancement as failed for an entry and count the attempt.

        The count is kept in the module's status object; a status written before the count
        existed counts as no attempts.

        With ``marker``, the failure object also stores the marker and ``gave_up`` (true once
        the count reaches ``MAX_ENHANCEMENT_ATTEMPTS``), and a stored marker that differs from
        ``marker`` restarts the count at 1.

        Args:
            entry_id: The entry ID
            module_name: The enhancement module name
            error: Error message
            marker: The module's current completion marker (optional)

        Returns:
            The attempt count now stored, or 0 when no entry has that id.
        """
        if marker is not None:
            try:
                async with self.pool.connection() as conn:
                    result = await conn.execute(
                        f"""
                        UPDATE enhanced_entries
                        SET enhancement_status = jsonb_set(
                            enhancement_status,
                            %(path)s::text[],
                            jsonb_build_object(
                                'status', 'failed',
                                'failed_at', NOW()::text,
                                'error', %(error)s::text,
                                'attempts', {_NEXT_ATTEMPTS_SQL},
                                'gave_up', {_NEXT_ATTEMPTS_SQL} >= %(max_attempts)s,
                                'marker', %(marker)s::text
                            )
                        )
                        WHERE entry_id = %(entry_id)s
                        RETURNING (enhancement_status->%(module)s->>'attempts')::int
                        """,
                        {
                            "path": [module_name],
                            "module": module_name,
                            "error": error[:500],
                            "marker": marker,
                            "max_attempts": MAX_ENHANCEMENT_ATTEMPTS,
                            "entry_id": entry_id,
                        },
                    )
                    row = await result.fetchone()
                    return int(row[0]) if row else 0
            except Exception as e:
                raise DatabaseQueryError(
                    f"Failed to mark enhancement failed: {e}",
                    query=f"UPDATE entry_id={entry_id} module={module_name}",
                ) from e
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    UPDATE enhanced_entries
                    SET enhancement_status = jsonb_set(
                        enhancement_status,
                        %s::text[],
                        jsonb_build_object(
                            'status', 'failed',
                            'failed_at', NOW()::text,
                            'error', %s::text,
                            'attempts', COALESCE((enhancement_status->%s->>'attempts')::int, 0) + 1
                        )
                    )
                    WHERE entry_id = %s
                    RETURNING (enhancement_status->%s->>'attempts')::int
                    """,
                    [[module_name], error[:500], module_name, entry_id, module_name],
                )
                row = await result.fetchone()
                return int(row[0]) if row else 0
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark enhancement failed: {e}",
                query=f"UPDATE entry_id={entry_id} module={module_name}",
            ) from e

    async def get_embedding_tables(self) -> list[EmbeddingTableInfo]:
        """Discover all embedding tables in the database.

        Returns:
            List of EmbeddingTableInfo for each embedding table
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'text_embeddings_%'
                    """
                )
                rows = await result.fetchall()

                tables: list[EmbeddingTableInfo] = []
                active_model = self.config.get_search_model()

                for row in rows:
                    table_name = row[0]

                    count_result = await conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count_row = await count_result.fetchone()
                    entry_count = int(count_row[0]) if count_row else 0

                    dim_result = await conn.execute(
                        """
                        SELECT atttypmod
                        FROM pg_attribute
                        WHERE attrelid = %s::regclass
                        AND attname = 'embedding'
                        """,
                        [table_name],
                    )
                    dim_row = await dim_result.fetchone()
                    dimension = int(dim_row[0]) if dim_row and dim_row[0] > 0 else None

                    is_active = False
                    if active_model:
                        from osprey.services.ariel_search.database.migrations import (
                            model_to_table_name,
                        )

                        is_active = table_name == model_to_table_name(active_model)

                    tables.append(
                        EmbeddingTableInfo(
                            table_name=table_name,
                            entry_count=entry_count,
                            dimension=dimension,
                            is_active=is_active,
                        )
                    )

                return tables
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get embedding tables: {e}",
                query="SELECT embedding tables",
            ) from e

    async def get_image_embedding_tables(self) -> list[EmbeddingTableInfo]:
        """Discover every image-embedding table in the database.

        Kept apart from :meth:`get_embedding_tables`, which lists the text
        tables only. ``entry_count`` here counts the stored picture vectors (a
        row with a ``skip_reason`` and no vector is not counted); ``is_active``
        marks the table the enabled ``image_embedding`` block names, and is
        False for every table while that block is off or misconfigured.

        Returns:
            One EmbeddingTableInfo per image table, sorted by name.
        """
        from osprey.services.ariel_search.database.migrations import image_embedding_target
        from osprey.services.ariel_search.exceptions import ModuleConfigError

        active_table: str | None = None
        if self.config.is_enhancement_module_enabled("image_embedding"):
            try:
                active_table = image_embedding_target(
                    self.config.get_enhancement_module_config("image_embedding") or {}
                ).table
            except ModuleConfigError:
                active_table = None

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    names = await image_embedding_table_names(cur)
                    tables: list[EmbeddingTableInfo] = []
                    for table_name in names:
                        await cur.execute(f"SELECT COUNT(embedding) FROM {table_name}")
                        count_row = await cur.fetchone()
                        await cur.execute(
                            """
                            SELECT atttypmod FROM pg_attribute
                            WHERE attrelid = %s::regclass AND attname = 'embedding'
                            """,
                            [table_name],
                        )
                        dim_row = await cur.fetchone()
                        tables.append(
                            EmbeddingTableInfo(
                                table_name=table_name,
                                entry_count=int(count_row[0]) if count_row else 0,
                                dimension=(int(dim_row[0]) if dim_row and dim_row[0] > 0 else None),
                                is_active=table_name == active_table,
                            )
                        )
                    return tables
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get image embedding tables: {e}",
                query="SELECT image embedding tables",
            ) from e

    async def validate_search_model_table(self, model: str) -> None:
        """Validate that the embedding table for the model exists.

        Args:
            model: Model name to validate

        Raises:
            ConfigurationError: If table does not exist
        """
        from osprey.services.ariel_search.database.migrations import model_to_table_name
        from osprey.services.ariel_search.exceptions import ConfigurationError

        table_name = model_to_table_name(model)

        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                    """,
                    [table_name],
                )
                row = await result.fetchone()
                if not row or not row[0]:
                    raise ConfigurationError(
                        f"Embedding table '{table_name}' does not exist. "
                        f"The model '{model}' is configured for semantic search "
                        "but migrations have not been run. Run 'osprey ariel migrate'.",
                        config_key="search_modules.semantic.model",
                    )
        except ConfigurationError:
            raise
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to validate embedding table: {e}",
                query=f"SELECT table exists {table_name}",
            ) from e

    @requires_module("enhancement", "text_embedding")
    async def store_text_embedding(
        self,
        entry_id: str,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """Store a text embedding for an entry.

        Args:
            entry_id: The entry ID
            embedding: The embedding vector
            model_name: The model name (determines table)
        """
        from osprey.services.ariel_search.database.migrations import model_to_table_name

        table_name = model_to_table_name(model_name)

        try:
            async with self.pool.connection() as conn:
                # Format embedding as PostgreSQL array
                embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                await conn.execute(
                    f"""
                    INSERT INTO {table_name} (entry_id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (entry_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                    """,
                    [entry_id, embedding_str],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to store text embedding: {e}",
                query=f"INSERT {table_name} entry_id={entry_id}",
            ) from e

    @requires_module("search", "keyword")
    async def keyword_search(
        self,
        where_clauses: list[str],
        params: list[Any],
        search_text: str,
        max_results: int = 10,
        include_highlights: bool = True,
        *,
        tsquery_sql: str | None = None,
        tsquery_params: list[Any] | None = None,
        pattern_timeout_seconds: float | None = None,
        v2: bool = False,
    ) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
        """Execute keyword search using full-text search.

        Args:
            where_clauses: SQL WHERE conditions
            params: Query parameters
            search_text: Original search text for highlighting
            max_results: Maximum results to return
            include_highlights: Include highlighted snippets
            tsquery_sql: Pre-built tsquery expression carrying its own ``%s``
                placeholders, used in place of the plain ``plainto_tsquery``
                for both the rank and the headline. ``None`` keeps the plain
                single-term path.
            tsquery_params: Parameters for ONE occurrence of `tsquery_sql`. They
                are spliced twice when highlights are requested and once
                otherwise, always ahead of `params`.
            pattern_timeout_seconds: Wall-clock budget for a statement carrying
                ``~*`` pattern predicates. When given, the statement runs inside
                a transaction with a local ``statement_timeout``; ``None`` runs
                it exactly as an ordinary keyword search does.
            v2: The ``has_v2_fts`` schema fact the caller built `where_clauses`
                under, so ``ts_rank`` and ``ts_headline`` use the same expression
                as the match.

        Returns:
            List of (entry, score, highlights) tuples

        Raises:
            ValueError: If `pattern_timeout_seconds` rounds down to 0 ms, which
                PostgreSQL reads as "no timeout at all".
            SearchTimeoutError: If the statement exceeded its timeout.
            PatternError: If PostgreSQL refused to compile a pattern.
            DatabaseQueryError: If the query failed for any other reason.
        """
        import contextlib

        import psycopg
        from psycopg.rows import dict_row

        statement_timeout: str | None = None
        if pattern_timeout_seconds is not None:
            timeout_ms = int(pattern_timeout_seconds * 1000)
            if timeout_ms < 1:
                raise ValueError(
                    "pattern_timeout_seconds must be at least 0.001; "
                    f"{pattern_timeout_seconds} renders as 0ms, which disables the timeout"
                )
            statement_timeout = f"{timeout_ms}ms"

        try:
            async with self.pool.connection() as conn, contextlib.AsyncExitStack() as stack:
                if statement_timeout is not None:
                    await stack.enter_async_context(conn.transaction())
                    await conn.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (statement_timeout,),
                    )
                async with conn.cursor(row_factory=dict_row) as cur:
                    where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"

                    fts_expression, headline_document = keyword_search_expressions(
                        self.config, v2=v2
                    )
                    if tsquery_sql is None:
                        query_expression = f"plainto_tsquery({FTS_CONFIG}, %s)"
                        query_params: list[Any] = [search_text]
                    else:
                        query_expression = f"({tsquery_sql})"
                        query_params = list(tsquery_params or [])

                    # A pattern-only statement carries no tsquery, so every row
                    # ranks 0 and the tie needs breaking to stay reproducible.
                    order_by = (
                        "rank DESC, timestamp DESC"
                        if tsquery_sql is None and not search_text.strip()
                        else "rank DESC"
                    )
                    if include_highlights:
                        query = f"""
                            SELECT e.*,
                                   ts_rank(
                                       {fts_expression},
                                       {query_expression}
                                   ) AS rank,
                                   ts_headline({FTS_CONFIG}, {headline_document}, {query_expression},
                                       'StartSel=<b>, StopSel=</b>, MaxFragments=3'
                                   ) AS headline
                            FROM enhanced_entries e
                            WHERE {where_sql}
                            ORDER BY {order_by}
                            LIMIT %s
                        """
                        all_params = query_params + query_params + params + [max_results]
                    else:
                        query = f"""
                            SELECT e.*,
                                   ts_rank(
                                       {fts_expression},
                                       {query_expression}
                                   ) AS rank,
                                   NULL AS headline
                            FROM enhanced_entries e
                            WHERE {where_sql}
                            ORDER BY {order_by}
                            LIMIT %s
                        """
                        all_params = query_params + params + [max_results]

                    await cur.execute(query, all_params)
                    rows = await cur.fetchall()

                    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
                    for row in rows:
                        # Row is now a dict, extract rank and headline, pass rest to factory
                        rank = float(row.pop("rank", 0.0) or 0.0)
                        headline = row.pop("headline", "") or ""

                        entry = enhanced_entry_from_row(row)
                        highlights = [headline] if headline else []
                        results.append((entry, rank, highlights))

                    return results

        except psycopg.errors.QueryCanceled as e:
            raise _pattern_timeout_error(pattern_timeout_seconds) from e
        except psycopg.errors.InvalidRegularExpression as e:
            # A statement_timeout that fires while the regex engine is running is
            # reported by PostgreSQL as SQLSTATE 2201B ("operation cancelled"),
            # not as QueryCanceled. It is the timeout the caller asked for, not a
            # malformed pattern, and must be classified the same way.
            if "cancel" in str(e).lower():
                raise _pattern_timeout_error(pattern_timeout_seconds) from e
            raise PatternError(str(e)) from e
        except Exception as e:
            raise DatabaseQueryError(
                f"Keyword search failed: {e}",
                query=f"KEYWORD SEARCH: {search_text}",
            ) from e

    async def caption_matches(
        self,
        entry_ids: list[str],
        model_id: str | None,
        *,
        tsquery_sql: str | None = None,
        tsquery_params: "list[Any] | tuple[Any, ...]" = (),
        pattern_bodies: "list[str] | tuple[str, ...]" = (),
        query_original: str | None = None,
        query_flattened: str | None = None,
        min_fraction: float | None = None,
    ) -> dict[str, list[str]]:
        """Find the attachments of some entries whose caption matches a query.

        Two caption sources are searched in one statement: model captions in
        ``attachment_captions`` under `model_id` (caption plus visible text; an
        ``{error}`` record carries no caption and never matches), and upstream
        captions on the ``attachments`` items, joined to ``attachment_files`` by
        ``source_url`` for their attachment id. The statement runs under the
        keyword pattern ``statement_timeout``.

        The match predicate takes one of two forms. Without `min_fraction` it
        is the keyword form: the caption text satisfies `tsquery_sql` or any
        pattern body (``~*``). With `min_fraction` it is only a lexeme-coverage
        count: with ``n(x)`` the number of distinct lexemes of ``x``, the
        caption's lexemes shared with `query_flattened`, capped at
        ``n(query_original)``, must reach ``ceil(min_fraction * n(query_original))``,
        and a query with no lexemes matches nothing.

        Args:
            entry_ids: The entries to search.
            model_id: The caption model whose captions count; None omits model
                captions and searches upstream captions only.
            tsquery_sql: A tsquery expression with positional ``%s`` placeholders.
            tsquery_params: One bind value per placeholder of `tsquery_sql`.
            pattern_bodies: PostgreSQL AREs matched case-insensitively.
            query_original: The text the caller typed (coverage form).
            query_flattened: The expanded query text; defaults to
                `query_original` (coverage form).
            min_fraction: Required coverage in ``(0, 1]``; selects the coverage form.

        Returns:
            ``{entry_id: [attachment_id, ...]}`` with ids sorted and unique; an
            entry with no match is absent. ``{}`` without touching the database
            for no entries, no predicate, or a store lacking the copy state.

        Raises:
            ValueError: If `min_fraction` is outside ``(0, 1]``, or the
                placeholders of `tsquery_sql` disagree with `tsquery_params`.
            SearchTimeoutError: If the statement exceeded the pattern timeout.
            PatternError: If PostgreSQL refused to compile a pattern.
            DatabaseQueryError: If the query failed for any other reason.
        """
        import psycopg
        from psycopg.rows import dict_row

        from osprey.services.ariel_search.search.keyword import KeywordSearchSettings

        if min_fraction is not None and not 0 < min_fraction <= 1:
            raise ValueError(f"min_fraction must be in (0, 1], got {min_fraction!r}")

        text = "t.caption_text"
        params: dict[str, Any] = {"entry_ids": list(entry_ids)}
        if min_fraction is not None:
            if query_original is None:
                return {}

            def _lexemes(document: str) -> str:
                return f"tsvector_to_array(to_tsvector({FTS_CONFIG}, {document}))"

            n_orig = f"cardinality({_lexemes('%(q_orig)s::text')})"
            predicate = (
                f"({n_orig} > 0 AND LEAST(cardinality(ARRAY("
                f"SELECT unnest({_lexemes(text)}) "
                f"INTERSECT SELECT unnest({_lexemes('%(q_flat)s::text')}))), {n_orig}) "
                f">= ceil(%(f)s::float8 * {n_orig}))"
            )
            params["q_orig"] = query_original
            params["q_flat"] = query_original if query_flattened is None else query_flattened
            params["f"] = float(min_fraction)
        else:
            legs: list[str] = []
            if tsquery_sql is not None:
                named, bound = named_tsquery(tsquery_sql, tsquery_params, prefix="tq")
                legs.append(f"to_tsvector({FTS_CONFIG}, {text}) @@ ({named})")
                params.update(bound)
            if pattern_bodies:
                named, bound = named_tsquery(
                    " OR ".join([f"{text} ~* %s"] * len(pattern_bodies)),
                    pattern_bodies,
                    prefix="pat",
                )
                legs.append(named)
                params.update(bound)
            if not legs:
                return {}
            predicate = "(" + " OR ".join(legs) + ")"

        if not entry_ids:
            return {}
        if not (await self.schema_facts()).has_copy_state:
            return {}

        selects: list[str] = []
        if model_id is not None:
            params["model"] = model_id
            selects.append(
                f"""
                SELECT e.entry_id, c.k AS attachment_id
                FROM enhanced_entries e
                CROSS JOIN LATERAL jsonb_each(
                    CASE WHEN jsonb_typeof(e.attachment_captions) = 'object'
                         THEN e.attachment_captions ELSE '{{}}'::jsonb END
                ) AS c(k, v)
                CROSS JOIN LATERAL (
                    SELECT COALESCE(c.v->%(model)s->>'caption', '') || ' ' ||
                           COALESCE(c.v->%(model)s->>'visible_text', '') AS caption_text
                ) AS t
                WHERE e.entry_id = ANY(%(entry_ids)s)
                  AND jsonb_typeof(c.v->%(model)s) = 'object'
                  AND c.v->%(model)s ? 'caption'
                  AND {predicate}
                """
            )
        selects.append(
            f"""
            SELECT e.entry_id, af.attachment_id
            FROM enhanced_entries e
            CROSS JOIN LATERAL jsonb_array_elements(
                CASE WHEN jsonb_typeof(e.attachments) = 'array'
                     THEN e.attachments ELSE '[]'::jsonb END
            ) AS a(item)
            JOIN attachment_files af
              ON af.entry_id = e.entry_id AND af.source_url = a.item->>'url'
            CROSS JOIN LATERAL (
                SELECT COALESCE(a.item->>'caption', '') AS caption_text
            ) AS t
            WHERE e.entry_id = ANY(%(entry_ids)s)
              AND {predicate}
            """
        )
        query = (
            "SELECT m.entry_id, m.attachment_id FROM ("
            + " UNION ".join(selects)
            + ") AS m ORDER BY m.entry_id, m.attachment_id"
        )

        timeout_seconds = KeywordSearchSettings.from_ariel_config(
            self.config
        ).pattern_timeout_seconds
        timeout_ms = int(timeout_seconds * 1000)
        if timeout_ms < 1:
            raise ValueError(
                "pattern_timeout_seconds must be at least 0.001; "
                f"{timeout_seconds} renders as 0ms, which disables the timeout"
            )

        try:
            async with self.pool.connection() as conn, conn.transaction():
                await conn.execute(
                    "SELECT set_config('statement_timeout', %s, true)", (f"{timeout_ms}ms",)
                )
                async with conn.cursor(row_factory=dict_row) as cur:
                    await cur.execute(query, params)
                    rows = await cur.fetchall()
        except psycopg.errors.QueryCanceled as e:
            raise _pattern_timeout_error(timeout_seconds) from e
        except psycopg.errors.InvalidRegularExpression as e:
            if "cancel" in str(e).lower():
                raise _pattern_timeout_error(timeout_seconds) from e
            raise PatternError(str(e)) from e
        except Exception as e:
            raise DatabaseQueryError(
                f"Caption match failed: {e}",
                query=f"CAPTION MATCHES entry_ids=ANY([{len(entry_ids)} ids])",
            ) from e

        matches: dict[str, list[str]] = {}
        for row in rows:
            matches.setdefault(row["entry_id"], []).append(row["attachment_id"])
        return matches

    @requires_module("search", "keyword")
    async def fuzzy_search(
        self,
        search_text: str,
        threshold: float = 0.3,
        max_results: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        *,
        v2: bool = False,
    ) -> list[tuple[EnhancedLogbookEntry, float, list[str]]]:
        """Execute fuzzy search using pg_trgm similarity.

        Args:
            search_text: Text to search for
            threshold: Minimum similarity threshold (0-1)
            max_results: Maximum results to return
            start_date: Filter entries after this time
            end_date: Filter entries before this time
            v2: The ``has_v2_fts`` schema fact. When true an entry scores the
                better of its ``raw_text`` and its ``attachment_text`` similarity,
                so a long caption never dilutes a ``raw_text`` match; when false
                only ``raw_text`` exists to compare.

        Returns:
            List of (entry, score, highlights) tuples
        """
        from psycopg.rows import dict_row

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    if v2:
                        sim_sql = (
                            "GREATEST(similarity(raw_text, %s), "
                            f"similarity({ATTACHMENT_TEXT_DOCUMENT}, %s))"
                        )
                        sim_params: list[Any] = [search_text, search_text]
                    else:
                        sim_sql = "similarity(raw_text, %s)"
                        sim_params = [search_text]
                    where_clauses = [f"{sim_sql} >= %s"]
                    params: list[Any] = [*sim_params, threshold]

                    if start_date:
                        where_clauses.append("timestamp >= %s")
                        params.append(start_date)
                    if end_date:
                        where_clauses.append("timestamp <= %s")
                        params.append(end_date)

                    where_sql = " AND ".join(where_clauses)

                    query = f"""
                        SELECT e.*, {sim_sql} AS sim
                        FROM enhanced_entries e
                        WHERE {where_sql}
                        ORDER BY sim DESC
                        LIMIT %s
                    """
                    all_params = sim_params + params + [max_results]

                    await cur.execute(query, all_params)
                    rows = await cur.fetchall()

                    results: list[tuple[EnhancedLogbookEntry, float, list[str]]] = []
                    for row in rows:
                        sim = float(row.pop("sim", 0.0) or 0.0)
                        entry = enhanced_entry_from_row(row)
                        results.append((entry, sim, []))

                    return results

        except Exception as e:
            raise DatabaseQueryError(
                f"Fuzzy search failed: {e}",
                query=f"FUZZY SEARCH: {search_text}",
            ) from e

    @requires_module("search", "semantic")
    async def semantic_search(
        self,
        query_embedding: list[float],
        model_name: str,
        max_results: int = 10,
        similarity_threshold: float = 0.5,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        author: str | None = None,
        source_system: str | None = None,
    ) -> list[tuple[EnhancedLogbookEntry, float]]:
        """Execute semantic similarity search using pgvector.

        Args:
            query_embedding: Query embedding vector
            model_name: Model name for table lookup
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity threshold
            start_date: Filter entries after this time
            end_date: Filter entries before this time
            author: Filter by author name (ILIKE match)
            source_system: Filter by source system (exact match)

        Returns:
            List of (entry, similarity) tuples
        """
        from psycopg.rows import dict_row

        from osprey.services.ariel_search.database.migrations import model_to_table_name

        table_name = model_to_table_name(model_name)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor(row_factory=dict_row) as cur:
                    where_clauses = ["1 - (emb.embedding <=> %s::vector) >= %s"]
                    params: list[Any] = [embedding_str, similarity_threshold]

                    if start_date:
                        where_clauses.append("e.timestamp >= %s")
                        params.append(start_date)
                    if end_date:
                        where_clauses.append("e.timestamp <= %s")
                        params.append(end_date)
                    if author:
                        where_clauses.append("e.author ILIKE %s")
                        params.append(f"%{author}%")
                    if source_system:
                        where_clauses.append("e.source_system = %s")
                        params.append(source_system)

                    where_sql = " AND ".join(where_clauses)

                    query = f"""
                        SELECT e.*, 1 - (emb.embedding <=> %s::vector) AS similarity
                        FROM enhanced_entries e
                        JOIN {table_name} emb ON e.entry_id = emb.entry_id
                        WHERE {where_sql}
                        ORDER BY similarity DESC
                        LIMIT %s
                    """

                    all_params = [embedding_str] + params + [max_results]

                    await cur.execute(query, all_params)
                    rows = await cur.fetchall()

                    results: list[tuple[EnhancedLogbookEntry, float]] = []
                    for row in rows:
                        similarity = float(row.pop("similarity", 0.0) or 0.0)
                        entry = enhanced_entry_from_row(row)
                        results.append((entry, similarity))

                    return results

        except Exception as e:
            raise DatabaseQueryError(
                f"Semantic search failed: {e}",
                query=f"SEMANTIC SEARCH model={model_name}",
            ) from e

    async def start_ingestion_run(self, source_system: str) -> int:
        """Record the start of an ingestion run.

        Args:
            source_system: Source system identifier

        Returns:
            The ingestion run ID
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    INSERT INTO ingestion_runs (started_at, source_system, status)
                    VALUES (NOW(), %s, 'running')
                    RETURNING id
                    """,
                    [source_system],
                )
                row = await result.fetchone()
                if not row:
                    raise DatabaseQueryError(
                        "Failed to start ingestion run: no ID returned",
                        query="INSERT ingestion_runs",
                    )
                return int(row[0])
        except DatabaseQueryError:
            raise
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to start ingestion run: {e}",
                query="INSERT ingestion_runs",
            ) from e

    async def complete_ingestion_run(
        self,
        run_id: int,
        entries_added: int,
        entries_updated: int,
        entries_failed: int,
    ) -> None:
        """Mark an ingestion run as successfully completed.

        Args:
            run_id: The ingestion run ID
            entries_added: Number of new entries added
            entries_updated: Number of existing entries updated
            entries_failed: Number of entries that failed
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE ingestion_runs
                    SET completed_at = NOW(),
                        status = 'success',
                        entries_added = %s,
                        entries_updated = %s,
                        entries_failed = %s
                    WHERE id = %s
                    """,
                    [entries_added, entries_updated, entries_failed, run_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to complete ingestion run {run_id}: {e}",
                query=f"UPDATE ingestion_runs id={run_id}",
            ) from e

    async def fail_ingestion_run(self, run_id: int, error_message: str) -> None:
        """Mark an ingestion run as failed.

        Args:
            run_id: The ingestion run ID
            error_message: Error description
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute(
                    """
                    UPDATE ingestion_runs
                    SET completed_at = NOW(),
                        status = 'failed',
                        error_message = %s
                    WHERE id = %s
                    """,
                    [error_message[:500], run_id],
                )
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to mark ingestion run {run_id} as failed: {e}",
                query=f"UPDATE ingestion_runs id={run_id}",
            ) from e

    async def get_last_successful_run(self, source_system: str) -> datetime | None:
        """Get the incremental-poll watermark of one source: the last successful run's start.

        The start, not the completion, is the watermark: an entry written
        upstream while a run was fetching is newer than the run's start but
        may be older than its completion, so the next poll's ``since`` must
        not pass it. Failed runs never move the watermark.

        Args:
            source_system: Source system identifier

        Returns:
            Start timestamp of the latest successful run, or None if no runs found
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT MAX(started_at) FROM ingestion_runs
                    WHERE source_system = %s AND status = 'success'
                    """,
                    [source_system],
                )
                row = await result.fetchone()
                if row and row[0]:
                    ts: datetime = row[0]
                    return ts
                return None
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get last successful run: {e}",
                query=f"SELECT MAX(started_at) source_system={source_system}",
            ) from e

    async def get_last_ingestion(self) -> datetime | None:
        """Get the completion time of the last successful ingestion run, any source.

        The facility-wide status time, as opposed to
        :meth:`get_last_successful_run`, which is one source's incremental-poll
        watermark and reads the run's start instead. Two surfaces report this value:
        the ARIEL dashboard status panel and the ``osprey ariel status`` CLI
        command, both by way of ``ARIELSearchService.get_status``.

        Returns:
            Completion timestamp of the most recent successful run across all
            source systems, or None if no run has ever succeeded.
        """
        try:
            async with self.pool.connection() as conn:
                result = await conn.execute(
                    """
                    SELECT MAX(completed_at) FROM ingestion_runs
                    WHERE status = 'success'
                    """
                )
                row = await result.fetchone()
                if row and row[0]:
                    ts: datetime = row[0]
                    return ts
                return None
        except Exception as e:
            raise DatabaseQueryError(
                f"Failed to get last ingestion: {e}",
                query="SELECT MAX(completed_at) all sources",
            ) from e

    async def health_check(self) -> tuple[bool, str]:
        """Check database connectivity and basic health.

        Returns:
            Tuple of (healthy, message)
        """
        try:
            async with self.pool.connection() as conn:
                await conn.execute("SELECT 1")
            return (True, "Database connected")
        except Exception as e:
            return (False, f"Database unreachable: {e}")
