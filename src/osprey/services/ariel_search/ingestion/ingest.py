"""Store one upstream logbook entry: text, attachment rows, pictures, enhancers.

:func:`ingest_one` is the single per-entry path of every ingest caller (the
polling scheduler, ``osprey ariel ingest`` and quickstart). Each phase opens
its own short-lived pool connection, so no connection is held while pictures
are fetched or enhancers call out.

Order per entry:

1. Sidecar metadata is merged into the entry (failures are non-fatal).
2. One transaction: the upsert ``RETURNING`` the stored picture columns takes
   the entry row lock first; a nested savepoint then records the attachment
   rows, deletes the rows whose url left the list, prunes their captions and
   recomposes ``attachment_text``. The transaction commits once, so the
   entry's text is stored before any picture is fetched.
3. ``copy_entry`` fetches, sniffs and renders the entry's pictures.
4. The inline enhancers run.

A failure of the upsert or of its transaction raises: the text was not
stored. A failure inside the savepoint rolls back only the savepoint, keeps
the text and is reported through :attr:`EntryIngestOutcome.attachments_recorded`;
picture and enhancer failures stay inside the outcome as well.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from osprey.services.ariel_search.attachments.copy import copy_entry, record_and_compose
from osprey.services.ariel_search.database.repository import text_mark_kwargs
from osprey.services.ariel_search.exceptions import DatabaseQueryError
from osprey.services.ariel_search.ingestion.metadata_attachment import (
    extract_metadata_from_attachments,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.attachments.copy import CopyRun
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

__all__ = ["SCHEMA_BEHIND_WARNING", "EntryIngestOutcome", "ingest_one"]

logger = get_logger("ariel.ingestion")

#: Logged once per run while the store lacks the attachment copy state.
SCHEMA_BEHIND_WARNING = (
    "schema behind code: entries stored now need `osprey ariel attachments backfill` after migrate"
)

#: The runs that already logged :data:`SCHEMA_BEHIND_WARNING`; a run is one ``CopyRun``.
_schema_behind_warned: weakref.WeakSet[CopyRun] = weakref.WeakSet()


@dataclass(frozen=True)
class EntryIngestOutcome:
    """What one :func:`ingest_one` call did after the entry's text was stored.

    Attributes:
        enhanced: Enhancers that ran and were marked complete.
        enhancer_failed: Enhancers that raised (each marked failed on the entry).
        attachments_recorded: False when recording the attachment rows failed
            and rolled back; the text is stored and the backfill record pass
            recovers the rows. True otherwise, including on a store whose
            schema predates the copy state, where nothing is recorded by design.
    """

    enhanced: int = 0
    enhancer_failed: int = 0
    attachments_recorded: bool = True


async def ingest_one(
    entry: EnhancedLogbookEntry,
    adapter: FacilityAdapter,
    repo: ARIELRepository,
    enhancers: Sequence[BaseEnhancementModule],
    config: ARIELConfig,
    copy_run: CopyRun | None,
) -> EntryIngestOutcome:
    """Store one entry, record and copy its pictures, then run the enhancers.

    On a store without the attachment copy state the old upsert runs as it
    did before (no attachment rows, no composed text, no picture fetch), and
    one WARNING per run says the entries need a backfill after migrating.

    After a successful store the entry dict carries ``attachment_text`` and
    ``attachment_captions`` as the row holds them.

    Args:
        entry: The entry as the adapter produced it; updated in place.
        adapter: The adapter that produced the entry.
        repo: The repository whose pool every phase borrows a connection from.
        enhancers: The inline enhancement modules, run in order.
        config: The ARIEL config.
        copy_run: The run's shared fetch state; one per poll, ingest or quickstart.

    Returns:
        The enhancer counts and whether the attachment rows were recorded.

    Raises:
        DatabaseQueryError: The upsert or its transaction failed, so the
            entry's text was not stored.
    """
    entry_id = entry["entry_id"]
    await extract_metadata_from_attachments(entry, adapter=adapter)

    if not (await repo.schema_facts()).has_copy_state:
        await repo.upsert_entry(entry)
        _warn_schema_behind(copy_run)
        return await _run_enhancers(
            entry, repo, enhancers, attachments_recorded=True, has_copy_state=False
        )

    recorded = await _store(entry, adapter, repo, config)

    if copy_run is not None:
        try:
            await copy_entry(repo, entry_id, config, copy_run)
        except Exception as exc:
            logger.warning("%s: picture copy failed (%s); the next poll retries it", entry_id, exc)

    return await _run_enhancers(
        entry, repo, enhancers, attachments_recorded=recorded, has_copy_state=True
    )


async def _store(
    entry: EnhancedLogbookEntry,
    adapter: FacilityAdapter,
    repo: ARIELRepository,
    config: ARIELConfig,
) -> bool:
    """Upsert the entry and record its attachments in one transaction.

    Returns:
        Whether the attachment savepoint committed.

    Raises:
        DatabaseQueryError: The upsert or the enclosing transaction failed.
    """
    entry_id = entry["entry_id"]
    recorded = True
    try:
        async with repo.pool.connection() as conn, conn.transaction():
            row = await repo.upsert_entry_returning(entry, conn=conn)
            text = row.get("attachment_text")
            captions = row.get("attachment_captions")
            try:
                async with conn.transaction():
                    await record_and_compose(conn, entry_id, row, config, adapter)
                    text, captions = await _stored_picture_text(conn, entry_id)
            except Exception as exc:
                recorded = False
                logger.warning(
                    "%s: attachments not recorded (%s); run osprey ariel attachments backfill",
                    entry_id,
                    exc,
                )
    except DatabaseQueryError:
        raise
    except Exception as exc:
        raise DatabaseQueryError(
            f"Failed to store entry {entry_id}: {exc}",
            query=f"INGEST entry_id={entry_id}",
        ) from exc

    stored = cast("dict[str, Any]", entry)
    stored["attachment_text"] = text
    stored["attachment_captions"] = captions
    return recorded


async def _stored_picture_text(conn: Any, entry_id: str) -> tuple[str | None, Any]:
    """Read the composed ``attachment_text`` and ``attachment_captions`` of a locked entry."""
    result = await conn.execute(
        """
        SELECT attachment_text, attachment_captions
        FROM enhanced_entries WHERE entry_id = %(entry_id)s
        """,
        {"entry_id": entry_id},
    )
    row = await result.fetchone()
    if row is None:
        return None, None
    if isinstance(row, Mapping):
        return row["attachment_text"], row["attachment_captions"]
    return row[0], row[1]


async def _run_enhancers(
    entry: EnhancedLogbookEntry,
    repo: ARIELRepository,
    enhancers: Sequence[BaseEnhancementModule],
    *,
    attachments_recorded: bool,
    has_copy_state: bool = False,
) -> EntryIngestOutcome:
    """Run each inline enhancer on its own connection and mark its status on the entry.

    A ``runs_inline=False`` module is skipped without a status write: it runs
    only in the catch-up. On a store with copy state a text module is marked
    under the md5 of the ``attachment_text`` it read, so a caption that changed
    meanwhile leaves the module owed.
    """
    enhanced = 0
    failed = 0
    for enhancer in enhancers:
        if not getattr(enhancer, "runs_inline", True):
            continue
        mark_kwargs = text_mark_kwargs(enhancer.name, entry, has_copy_state)
        try:
            async with repo.pool.connection() as conn:
                await enhancer.enhance(entry, conn)
            await repo.mark_enhancement_complete(entry["entry_id"], enhancer.name, **mark_kwargs)
            enhanced += 1
        except Exception as exc:
            failed += 1
            try:
                await repo.mark_enhancement_failed(entry["entry_id"], enhancer.name, str(exc))
            except Exception:
                logger.exception(
                    "%s: could not mark enhancer %s failed", entry["entry_id"], enhancer.name
                )
    return EntryIngestOutcome(
        enhanced=enhanced,
        enhancer_failed=failed,
        attachments_recorded=attachments_recorded,
    )


def _warn_schema_behind(copy_run: CopyRun | None) -> None:
    """Log :data:`SCHEMA_BEHIND_WARNING` once for ``copy_run`` (every call without one)."""
    if copy_run is not None:
        if copy_run in _schema_behind_warned:
            return
        _schema_behind_warned.add(copy_run)
    logger.warning(SCHEMA_BEHIND_WARNING)
