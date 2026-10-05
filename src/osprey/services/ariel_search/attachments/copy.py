"""The attachment copy step: which recorded attachments ARIEL fetches.

The helpers here are pure decisions over a recorded ``attachment_files`` row
and the current configuration, so a configuration change (another
``copy_on_ingest`` mode, a larger cap, a wider origin set) is honoured by the
next backfill without re-fetching anything it still refuses.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import posixpath
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlsplit

from psycopg.types.json import Jsonb

from osprey.services.ariel_search.attachments import (
    attachment_id_for,
    fetchable_url,
    is_native_item,
)
from osprey.services.ariel_search.attachments import prepare as _prepare
from osprey.services.ariel_search.attachments.compose import (
    caption_model_id,
    compose_attachment_text,
)
from osprey.services.ariel_search.attachments.fetch import (
    COPY_ENTRY_DEADLINE,
    FetchOutcome,
    fetch_attachment_bytes,
    is_file_source,
    make_fetch_session,
    origin_of,
)
from osprey.services.ariel_search.attachments.formats import (
    CONFIG_SKIP_REASONS,
    OCTET_STREAM,
    SOURCE_SKIP_REASONS,
    is_markup,
    sniff,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    import aiohttp
    from psycopg import AsyncConnection

    from osprey.services.ariel_search.config import ARIELConfig, AttachmentsConfig, Origin
    from osprey.services.ariel_search.database.repository import ARIELRepository, CopyRendition
    from osprey.services.ariel_search.ingestion.base import FacilityAdapter

__all__ = [
    "COPY_ENTRY_DEADLINE",
    "COPY_MAX_ATTEMPTS",
    "COPY_MAX_PER_ENTRY",
    "COPY_PENDING_MAX_AGE",
    "CopyEntryReport",
    "CopyRun",
    "HostBreaker",
    "copy_entry",
    "eligible_for_mode",
    "fetch_attachment_bytes",
    "record_and_compose",
    "record_rows",
    "still_skipped",
    "validated_declared_type",
]

logger = get_logger("ariel")

#: Filename recorded when neither the upstream item nor its url names one.
DEFAULT_FILENAME = "attachment"

#: Longest declared type stored; anything longer is recorded as NULL.
MAX_DECLARED_TYPE_LEN = 100

_DECLARED_TYPE_RE = re.compile(r"[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*")


def validated_declared_type(t: object) -> str | None:
    """Return the upstream-declared MIME type in its stored form.

    Args:
        t: The type an upstream attachment item declares, of any type.

    Returns:
        The type lower-cased when it is a string of at most 100 characters
        shaped ``type/subtype`` (letters, digits and ``.+-``, each part opening
        with a letter or digit); otherwise ``None``. Parameters such as
        ``; charset=…`` and surrounding whitespace make a type invalid.
    """
    if not isinstance(t, str):
        return None
    lowered = t.lower()
    if len(lowered) > MAX_DECLARED_TYPE_LEN:
        return None
    if _DECLARED_TYPE_RE.fullmatch(lowered) is None:
        return None
    return lowered


def eligible_for_mode(declared: object, mode: str) -> bool:
    """Whether ``copy_on_ingest`` mode ``mode`` fetches an item declared as ``declared``.

    Args:
        declared: The declared type, raw or validated; an invalid one counts as
            missing.
        mode: ``images``, ``all`` or ``none``.

    Returns:
        ``all`` fetches everything and ``none`` nothing. ``images`` fetches an
        item declared ``image/*``, declared ``application/octet-stream`` or with
        no usable declared type; the magic-byte sniff decides afterwards.
    """
    if mode == "all":
        return True
    if mode != "images":
        return False
    valid = validated_declared_type(declared)
    return valid is None or valid == OCTET_STREAM or valid.startswith("image/")


def _attachments_config(config: ARIELConfig | AttachmentsConfig) -> AttachmentsConfig:
    from osprey.services.ariel_search.config import AttachmentsConfig

    # Imported here: the config module imports the attachments package.
    if isinstance(config, AttachmentsConfig):
        return config
    return config.attachments


def still_skipped(
    row: Mapping[str, Any],
    config: ARIELConfig | AttachmentsConfig,
    origins: frozenset[Origin],
    *,
    file_source: bool,
) -> str | None:
    """Re-decide the configuration skips for one recorded attachment row.

    Only what the row recorded is read — ``source_url``, ``mime_type`` (the
    declared or sniffed type) and ``size_bytes`` (for a size skip, the
    observed size capped at ``cap + 1``) — never its stored ``skip_reason``, so
    a row skipped under an older configuration is fetched once the current
    one allows it.

    Args:
        row: An ``attachment_files`` row as ``get_copy_rows`` returns it.
        config: The ARIEL config (or its ``attachments`` block).
        origins: The origin set the adapter may fetch from.
        file_source: Whether the adapter reads a local file source
            (``is_file_source(adapter)``); a relative path there has no origin
            and is never ``origin_not_allowed``, while on an http source a url
            without an origin is.

    Returns:
        The configuration skip code that still applies — ``copy_on_ingest_mode``,
        ``origin_not_allowed`` or ``size_cap`` — or ``None`` when the row may be
        fetched. ``per_entry_limit`` is never returned: it depends on sibling
        rows and is decided by the per-entry budget check.
    """
    attachments = _attachments_config(config)
    mode = attachments.copy_on_ingest
    if mode == "none":
        return "copy_on_ingest_mode"

    # The origin rule checks absolute http(s) urls against the origin set. A
    # relative path on a file source has no origin and is confined by the
    # fetcher instead; on any other source a url without an origin cannot be
    # fetched at all (``fetchable_url`` never records one), so it is refused here
    # rather than sent to the network.
    source_url = row.get("source_url")
    origin = origin_of(source_url)
    if origin is None:
        if not file_source:
            return "origin_not_allowed"
    elif origin not in origins:
        return "origin_not_allowed"

    size = row.get("size_bytes")
    cap = attachments.max_file_mb * 1024 * 1024
    if isinstance(size, int) and size > cap:
        return "size_cap"

    if not eligible_for_mode(row.get("mime_type"), mode):
        return "copy_on_ingest_mode"
    return None


def _filename_for(item: Mapping[str, Any], url: str) -> str:
    """Return the upstream filename, else the url's basename, else ``attachment``."""
    upstream = item.get("filename")
    if upstream is not None and not isinstance(upstream, (Mapping, list, bool)):
        name = str(upstream).strip()
        if name:
            return name
    base = posixpath.basename(unquote(urlsplit(url).path).replace("\\", "/")).strip()
    return base or DEFAULT_FILENAME


def _attachment_list(attachments: Any) -> list[Any]:
    """Return the locked JSONB attachment list as a Python list (empty when unusable)."""
    if isinstance(attachments, str):
        try:
            attachments = json.loads(attachments)
        except ValueError:
            return []
    if isinstance(attachments, Sequence) and not isinstance(attachments, (str, bytes)):
        return list(attachments)
    return []


async def record_rows(
    conn: AsyncConnection,
    entry_id: str,
    attachments: Any,
    config: ARIELConfig | AttachmentsConfig,
    adapter: FacilityAdapter,
) -> list[str]:
    """Record an ``attachment_files`` row for every fetchable upstream attachment.

    Works on the JSONB list the caller holds locked (the entry row lock is
    taken first); it never reads the list itself. Per item: a non-mapping item
    or a missing, empty or non-string url is skipped; a native item
    (``/api/attachments/<id>``) never gets a row nor enters ``keep``; a url
    that fails ``fetchable_url`` for the adapter's source kind gets no row
    (its summary reads ``no_source_url``). Every other item gets, unless its
    id already has a row, a ``pending`` row when the ``copy_on_ingest`` mode
    may fetch its declared type, else a ``skipped / copy_on_ingest_mode`` row
    keeping the declared type; in ``none`` mode every such item is skipped.
    Both inserts are ``ON CONFLICT DO NOTHING``, so copied and content-skipped
    rows are never re-fetched. When any row was inserted, the two image
    status keys are cleared.

    Args:
        conn: Connection inside the transaction holding the entry lock.
        entry_id: The entry the attachments belong to.
        attachments: The entry's locked ``attachments`` JSONB value.
        config: The ARIEL config (or its ``attachments`` block).
        adapter: The adapter the caller holds; decides whether relative paths
            are fetchable.

    Returns:
        ``keep``: the source urls rows are keyed on, in list order.
    """
    from osprey.services.ariel_search.database.repository import ARIELRepository

    mode = _attachments_config(config).copy_on_ingest
    file_source = is_file_source(adapter)
    keep: list[str] = []
    inserted = False
    for item in _attachment_list(attachments):
        if not isinstance(item, Mapping):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not url or is_native_item(item):
            continue
        if not fetchable_url(url, file_source=file_source):
            continue
        attachment_id = attachment_id_for(entry_id, item)
        if attachment_id is None:
            continue
        if url not in keep:
            keep.append(url)
        declared = validated_declared_type(item.get("type"))
        eligible = mode != "none" and eligible_for_mode(declared, mode)
        result = await conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, source_url,
                copy_status, skip_reason
            ) VALUES (
                %(attachment_id)s, %(entry_id)s, %(filename)s, %(mime_type)s,
                %(source_url)s, %(copy_status)s, %(skip_reason)s
            )
            ON CONFLICT (attachment_id) DO NOTHING
            """,
            {
                "attachment_id": attachment_id,
                "entry_id": entry_id,
                "filename": _filename_for(item, url),
                "mime_type": declared,
                "source_url": url,
                "copy_status": "pending" if eligible else "skipped",
                "skip_reason": None if eligible else "copy_on_ingest_mode",
            },
        )
        if result.rowcount > 0:
            inserted = True
    if inserted:
        await ARIELRepository.clear_image_status_keys(conn, entry_id)
    return keep


async def record_and_compose(
    conn: AsyncConnection,
    entry_id: str,
    locked_row: Mapping[str, Any],
    config: ARIELConfig,
    adapter: FacilityAdapter,
) -> list[str]:
    """Record an entry's attachment rows and recompose its ``attachment_text``.

    The one body both recovery paths run on a locked entry: ingest inside its
    savepoint and backfill per locked row. In order: :func:`record_rows`;
    when the locked list is non-empty, delete the source rows whose url left
    ``keep``; prune the deleted ids from ``attachment_captions``; compose
    ``attachment_text``; write both only when they differ from the row; and,
    when ``attachment_text`` changed, drop the ``text_embedding`` and
    ``qmd_export`` status keys so those modules re-read it.

    Args:
        conn: Connection inside the transaction holding the entry lock.
        entry_id: The entry.
        locked_row: The locked row's ``attachments``, ``attachment_text`` and
            ``attachment_captions``.
        config: The ARIEL config.
        adapter: The adapter the caller holds.

    Returns:
        ``keep`` as :func:`record_rows` returns it.
    """
    from osprey.services.ariel_search.database.repository import ARIELRepository

    attachments = _attachment_list(locked_row.get("attachments"))
    keep = await record_rows(conn, entry_id, attachments, config, adapter)

    stored_captions = locked_row.get("attachment_captions")
    captions: dict[str, Any] | None = (
        dict(stored_captions) if isinstance(stored_captions, Mapping) else None
    )
    if attachments:
        deleted = await ARIELRepository.delete_dropped_attachments(conn, entry_id, keep)
        if captions is not None:
            for attachment_id in deleted:
                captions.pop(attachment_id, None)

    text = compose_attachment_text(entry_id, attachments, captions, caption_model_id(config))
    captions_param = Jsonb(captions) if captions is not None else None
    await conn.execute(
        """
        UPDATE enhanced_entries
        SET attachment_text = %(text)s, attachment_captions = %(captions)s::jsonb
        WHERE entry_id = %(entry_id)s
        AND (attachment_text, attachment_captions)
            IS DISTINCT FROM (%(text)s::text, %(captions)s::jsonb)
        """,
        {"entry_id": entry_id, "text": text, "captions": captions_param},
    )
    if text != locked_row.get("attachment_text"):
        await ARIELRepository.clear_text_status_keys(conn, entry_id)
    return keep


# -- copy_entry: fetch, classify and render one entry's pictures ---------------

#: The most ``copied`` rows one entry may hold.
COPY_MAX_PER_ENTRY = 20

#: Host-up transient attempts after which a row becomes ``skipped / fetch_failed``.
COPY_MAX_ATTEMPTS = 5

#: A connect-class failure on a row recorded longer ago than this ends it ``fetch_failed``.
COPY_PENDING_MAX_AGE = timedelta(days=7)

#: Concurrent fetches one run allows.
COPY_CONCURRENCY = 4

#: Consecutive transient failures from one host that trip its breaker.
BREAKER_THRESHOLD = 5

#: Seconds a tripped host waits before one probe fetch is let through.
BREAKER_COOLDOWN_S = 60.0

#: Per-entry byte budget, as a multiple of the per-file cap.
BYTE_BUDGET_FILES = 4

_MB = 1024 * 1024


class HostBreaker:
    """The per-host circuit breaker of one copy run.

    ``threshold`` CONSECUTIVE transient failures from one host trip it; any
    answer from the host (a fetched file, or a definite refusal such as a 404)
    resets the count, so scattered 503/429s never trip it. While tripped no
    fetch to the host is allowed until ``cooldown_s`` has passed; then exactly
    one probe goes through, whose answer closes the breaker and whose
    transient failure re-trips it for another cooldown.
    """

    def __init__(
        self,
        threshold: int = BREAKER_THRESHOLD,
        cooldown_s: float = BREAKER_COOLDOWN_S,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._failures: dict[Origin, int] = {}
        self._tripped_at: dict[Origin, float] = {}
        self._probing: set[Origin] = set()

    def is_open(self, host: Origin) -> bool:
        """Whether the host's breaker is tripped (probe or not)."""
        return host in self._tripped_at

    def allow(self, host: Origin) -> bool:
        """Whether a fetch to ``host`` may go out now; a granted probe is marked in flight."""
        tripped_at = self._tripped_at.get(host)
        if tripped_at is None:
            return True
        if host in self._probing or self._clock() - tripped_at < self.cooldown_s:
            return False
        self._probing.add(host)
        return True

    def record(self, host: Origin, *, transient: bool) -> None:
        """Record one fetch outcome from ``host``."""
        if not transient:
            self._failures.pop(host, None)
            self._tripped_at.pop(host, None)
            self._probing.discard(host)
            return
        if host in self._probing:
            self._probing.discard(host)
            self._tripped_at[host] = self._clock()
            return
        count = self._failures.get(host, 0) + 1
        self._failures[host] = count
        if count >= self.threshold and host not in self._tripped_at:
            self._tripped_at[host] = self._clock()

    def release(self, host: Origin) -> None:
        """Forget a probe that ended without an outcome (deadline or cancellation)."""
        self._probing.discard(host)


@dataclass(eq=False)
class CopyRun:
    """What one poll, ingest or backfill run shares across its ``copy_entry`` calls.

    An async context manager: entering opens the run's one fetch session
    (``make_fetch_session(adapter)``), which every fetch of the run uses, and
    leaving closes it. The timing constants are fields so tests can scale them.

    Attributes:
        adapter: The ingestion adapter whose connector and SSL context carry fetches.
        origins: The origin set fetches may reach (``origins_for(adapter, config)``).
        semaphore: The run's concurrency limit.
        breaker: The per-host breaker; built from the two breaker fields when omitted.
        entry_deadline_s: Seconds one entry's fetch candidates may take to fetch and
            render. Render-only candidates are rendered before it starts and
            are not bounded by it.
        pending_max_age: Age after which a connect-class failure ends a row.
        breaker_threshold: Consecutive transient failures that trip a host.
        breaker_cooldown_s: Seconds before a tripped host gets one probe.
        max_attempts: Host-up transient attempts before ``fetch_failed``.
        max_per_entry: Most ``copied`` rows per entry.
        session: The run's fetch session, set while the run is entered.
        render_available: False once the render worker proved unavailable; the
            rest of the run stores originals without renditions.
    """

    adapter: FacilityAdapter
    origins: frozenset[Origin]
    semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(COPY_CONCURRENCY)
    )
    breaker: HostBreaker | None = None
    entry_deadline_s: float = COPY_ENTRY_DEADLINE
    pending_max_age: timedelta = COPY_PENDING_MAX_AGE
    breaker_threshold: int = BREAKER_THRESHOLD
    breaker_cooldown_s: float = BREAKER_COOLDOWN_S
    max_attempts: int = COPY_MAX_ATTEMPTS
    max_per_entry: int = COPY_MAX_PER_ENTRY
    session: aiohttp.ClientSession | None = None
    render_available: bool = True
    _render_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _owns_session: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.breaker is None:
            self.breaker = HostBreaker(self.breaker_threshold, self.breaker_cooldown_s)

    async def __aenter__(self) -> CopyRun:
        if self.session is None:
            self.session = make_fetch_session(self.adapter)
            self._owns_session = True
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._owns_session and self.session is not None:
            await self.session.close()
            self.session = None
            self._owns_session = False

    def _stop_rendering(self, exc: BaseException) -> None:
        if self.render_available:
            logger.warning(
                "Render worker unavailable (exit code %s): %s; storing pictures without "
                "renditions for the rest of this run",
                getattr(exc, "exit_code", None),
                exc,
            )
        self.render_available = False


@dataclass
class CopyEntryReport:
    """What one :func:`copy_entry` call did.

    Attributes:
        fetches: Fetch calls made.
        copied: Rows written ``copied`` by a fetch.
        rendered: Render-only rows given a rendition or a content skip.
        skipped: Rows written ``skipped``, by code.
        pending: Fetch candidates left ``pending`` (transient, deadline, breaker).
    """

    fetches: int = 0
    copied: int = 0
    rendered: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    pending: int = 0

    def _skip(self, code: str) -> None:
        self.skipped[code] = self.skipped.get(code, 0) + 1


@dataclass
class _Flight:
    """One fetch candidate's progress, read by the deadline handling."""

    row: dict[str, Any]
    phase: str = "queued"  # queued | fetching | storing | done
    sent: bool = False


@dataclass
class _Budget:
    """The byte budget left for this entry's stored originals."""

    bytes_left: int

    def take(self, size: int) -> bool:
        if size > self.bytes_left:
            return False
        self.bytes_left -= size
        return True


def _is_render_only(row: Mapping[str, Any]) -> bool:
    return (
        row.get("copy_status") == "copied"
        and row.get("rendition_sha256") is None
        and row.get("skip_reason") is None
        and bool(row.get("has_data"))
    )


def _is_fetch_candidate(row: Mapping[str, Any], retry_skipped: bool) -> bool:
    status = row.get("copy_status")
    if status == "pending":
        return True
    return (
        retry_skipped
        and status == "skipped"
        and row.get("skip_reason") in (CONFIG_SKIP_REASONS | SOURCE_SKIP_REASONS)
    )


def _declared_picture(row: Mapping[str, Any]) -> bool:
    """Whether the row was declared a picture: ``image/*`` or an image-extension basename.

    A ``source_refused`` row is one: that skip is written only for a declared
    picture, and it replaces the stored mime with the refused page's, so the
    row itself is the remaining evidence of the declaration.
    """
    if row.get("skip_reason") == "source_refused":
        return True
    declared = validated_declared_type(row.get("mime_type"))
    if declared is not None and declared.startswith("image/"):
        return True
    url = row.get("source_url")
    if not isinstance(url, str):
        return False
    base = posixpath.basename(unquote(urlsplit(url).path).replace("\\", "/"))
    guessed, _ = mimetypes.guess_type(base)
    return bool(guessed and guessed.startswith("image/"))


def _too_old(row: Mapping[str, Any], max_age: timedelta) -> bool:
    created = row.get("created_at")
    if not isinstance(created, datetime):
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return created < datetime.now(UTC) - max_age


def rendition_of(prepared: _prepare.PreparedPicture) -> CopyRendition | None:
    """The rendition of ``prepared`` as the repository stores it, or ``None`` when none was made."""
    from osprey.services.ariel_search.database.repository import CopyRendition

    if not prepared.has_rendition:
        return None
    return CopyRendition(
        data=cast(bytes, prepared.rendition_bytes),
        mime_type=cast(str, prepared.rendition_mime),
        width=cast(int, prepared.rendition_w),
        height=cast(int, prepared.rendition_h),
        sha256=cast(str, prepared.rendition_sha256),
    )


async def _in_list_order(
    repo: ARIELRepository, entry_id: str, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Order rows as the entry's JSONB list does; rows it does not name keep their order last."""
    entry = await repo.get_entry(entry_id)
    attachments = entry.get("attachments") if entry else None
    positions: dict[str, int] = {}
    for index, item in enumerate(_attachment_list(attachments)):
        if not isinstance(item, Mapping):
            continue
        attachment_id = attachment_id_for(entry_id, item)
        if attachment_id is not None:
            positions.setdefault(attachment_id, index)
    last = len(positions) + len(rows)
    return sorted(rows, key=lambda r: positions.get(r["attachment_id"], last))


async def _write_skip(
    repo: ARIELRepository,
    entry_id: str,
    row: Mapping[str, Any],
    code: str,
    report: CopyEntryReport,
    *,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    copy_attempts: int | None = None,
) -> None:
    """Write ``skipped / code``; a row already carrying that exact skip is left alone."""
    report._skip(code)
    if (
        row.get("copy_status") == "skipped"
        and row.get("skip_reason") == code
        and mime_type is None
        and copy_attempts is None
    ):
        return
    await repo.apply_copy_outcome(
        entry_id,
        row["attachment_id"],
        copy_status="skipped",
        skip_reason=code,
        mime_type=mime_type if mime_type is not None else row.get("mime_type"),
        size_bytes=size_bytes if size_bytes is not None else row.get("size_bytes"),
        copy_attempts=copy_attempts,
    )


async def _charge_attempt(
    repo: ARIELRepository,
    entry_id: str,
    row: Mapping[str, Any],
    copy_run: CopyRun,
    report: CopyEntryReport,
) -> None:
    """Charge one host-up transient attempt; the last allowed one ends the row ``fetch_failed``.

    A retried ``source_refused`` row keeps its skip instead: its stored mime is
    the refused page's, so as a ``pending`` row it would be judged by that mime
    and lose the declared picture. Backfill retries it again.
    """
    if row.get("skip_reason") == "source_refused":
        await _write_skip(repo, entry_id, row, "source_refused", report)
        return
    base = int(row.get("copy_attempts") or 0) if row.get("copy_status") == "pending" else 0
    attempts = base + 1
    if attempts >= copy_run.max_attempts:
        await _write_skip(repo, entry_id, row, "fetch_failed", report, copy_attempts=attempts)
        return
    report.pending += 1
    await repo.apply_copy_outcome(
        entry_id,
        row["attachment_id"],
        copy_status="pending",
        mime_type=row.get("mime_type"),
        size_bytes=row.get("size_bytes"),
        copy_attempts=attempts,
    )


async def _record_transient(
    repo: ARIELRepository,
    entry_id: str,
    row: Mapping[str, Any],
    outcome: FetchOutcome,
    copy_run: CopyRun,
    report: CopyEntryReport,
) -> None:
    if row.get("skip_reason") == "source_refused":
        # Left as it is, for the same reason as in _charge_attempt.
        await _write_skip(repo, entry_id, row, "source_refused", report)
    elif outcome.host_up:
        await _charge_attempt(repo, entry_id, row, copy_run, report)
    elif _too_old(row, copy_run.pending_max_age):
        await _write_skip(repo, entry_id, row, "fetch_failed", report)
    else:
        # A connect-class failure burns no attempt: the row stays as it is.
        report.pending += 1


async def _store_fetched(
    repo: ARIELRepository,
    entry_id: str,
    row: Mapping[str, Any],
    data: bytes,
    mode: str,
    budget: _Budget,
    deadline_at: float,
    copy_run: CopyRun,
    report: CopyEntryReport,
) -> None:
    """Decide and write the outcome of a fetched file (the mode-dependent outcome table)."""
    sniffed = sniff(data)
    size = len(data)
    if is_markup(sniffed) and _declared_picture(row):
        # A login or proxy page served where a picture was declared: the source
        # did not deliver, so backfill retries it once access is fixed.
        await _write_skip(
            repo, entry_id, row, "source_refused", report, mime_type=sniffed.mime, size_bytes=size
        )
        return
    if not sniffed.is_image and mode != "all":
        await _write_skip(
            repo,
            entry_id,
            row,
            "copy_on_ingest_mode",
            report,
            mime_type=sniffed.mime,
            size_bytes=size,
        )
        return
    if not budget.take(size):
        await _write_skip(
            repo, entry_id, row, "per_entry_limit", report, mime_type=sniffed.mime, size_bytes=size
        )
        return

    mime_type = sniffed.mime
    skip_reason = sniffed.skip_reason
    rendition = None
    if sniffed.is_image:
        skip_reason = None
        async with copy_run._render_lock:
            loop = asyncio.get_running_loop()
            if copy_run.render_available and loop.time() < deadline_at:
                try:
                    prepared = await _prepare.prepare_picture(data)
                except _prepare.RenderUnavailable as exc:
                    copy_run._stop_rendering(exc)
                else:
                    mime_type = prepared.mime_type
                    skip_reason = prepared.skip_reason
                    rendition = rendition_of(prepared)
    report.copied += 1
    await repo.apply_copy_outcome(
        entry_id,
        row["attachment_id"],
        copy_status="copied",
        data=data,
        mime_type=mime_type,
        size_bytes=size,
        skip_reason=skip_reason,
        rendition=rendition,
    )


async def _fetch_one(
    repo: ARIELRepository,
    entry_id: str,
    flight: _Flight,
    *,
    cap: int,
    mode: str,
    budget: _Budget,
    deadline_at: float,
    copy_run: CopyRun,
    report: CopyEntryReport,
) -> None:
    row = flight.row
    url = row["source_url"]
    host = origin_of(url)
    breaker = cast(HostBreaker, copy_run.breaker)
    loop = asyncio.get_running_loop()

    def _on_sent() -> None:
        flight.sent = True

    # The slot is taken here, not inside the fetcher, so the breaker is asked
    # only once a fetch can really go out: a tripped host gets no queued fetches.
    async with copy_run.semaphore:
        # A grant on an open breaker is the probe; only the probe holder may release it.
        probe = host is not None and breaker.is_open(host)
        if host is not None and not breaker.allow(host):
            flight.phase = "done"
            report.pending += 1
            return
        flight.phase = "fetching"
        report.fetches += 1
        try:
            outcome = await fetch_attachment_bytes(
                url,
                cap,
                copy_run.origins,
                copy_run.adapter,
                session=copy_run.session,
                total=max(deadline_at - loop.time(), 0.001),
                on_sent=_on_sent,
            )
        except BaseException:
            if host is not None and probe:
                breaker.release(host)
            raise
        # Recorded before the slot is released, so the next queued fetch to
        # this host already sees the outcome when it asks the breaker.
        if host is not None:
            breaker.record(host, transient=outcome.transient)
    flight.phase = "storing"

    if outcome.ok:
        await _store_fetched(
            repo,
            entry_id,
            row,
            cast(bytes, outcome.data),
            mode,
            budget,
            deadline_at,
            copy_run,
            report,
        )
    elif outcome.transient:
        await _record_transient(repo, entry_id, row, outcome, copy_run, report)
    else:
        code = outcome.code or "source_refused"
        await _write_skip(
            repo,
            entry_id,
            row,
            code,
            report,
            size_bytes=outcome.observed_size if code == "size_cap" else None,
        )
    flight.phase = "done"


async def _render_stored(
    repo: ARIELRepository,
    entry_id: str,
    row: Mapping[str, Any],
    copy_run: CopyRun,
    report: CopyEntryReport,
) -> None:
    """Render a ``copied`` row's stored original; never touches ``data`` or ``copy_status``."""
    source = await repo.get_copy_source(row["attachment_id"])
    if source is None:
        return
    data, _stored_mime = source
    if not copy_run.render_available and sniff(data).is_image:
        return
    async with copy_run._render_lock:
        try:
            prepared = await _prepare.prepare_picture(data)
        except _prepare.RenderUnavailable as exc:
            copy_run._stop_rendering(exc)
            return
    if await repo.apply_render_outcome(
        entry_id,
        row["attachment_id"],
        mime_type=prepared.mime_type,
        skip_reason=prepared.skip_reason,
        rendition=rendition_of(prepared),
    ):
        report.rendered += 1


async def copy_entry(
    repo: ARIELRepository,
    entry_id: str,
    config: ARIELConfig | AttachmentsConfig,
    copy_run: CopyRun,
    *,
    retry_skipped: bool = False,
) -> CopyEntryReport:
    """Fetch, classify and render one entry's recorded pictures.

    Reads the entry's rows with no lock and takes them in JSONB list order.
    Render-only candidates (``copied``, stored bytes, no rendition, no skip)
    skip the config check, the budget and the network in every mode: their
    stored original goes through ``prepare_picture`` and the render-only
    UPDATE. Fetch candidates (``pending``; with ``retry_skipped`` also the
    config and source skips) first pass :func:`still_skipped`, whose code is
    written with no network call; then the database budget
    (``max_per_entry`` copied rows, ``4 × cap`` bytes) decides which fit, the
    rest getting ``per_entry_limit`` (a slot a fitted row leaves unused by
    ending non-copied is not handed on in this call; ``per_entry_limit`` is a
    config skip, so backfill reconsiders those rows). The fitting rows are fetched
    concurrently under the run's semaphore and per-host breaker, within
    ``entry_deadline_s``; each outcome is its own transaction.

    Args:
        repo: The ARIEL repository.
        entry_id: The entry whose pictures are copied.
        config: The ARIEL config (or its ``attachments`` block).
        copy_run: The run's shared adapter, origins, semaphore, breaker and session.
        retry_skipped: Also retry the config- and source-skipped rows (backfill).

    Returns:
        A :class:`CopyEntryReport` of what was done.
    """
    attachments = _attachments_config(config)
    report = CopyEntryReport()
    rows = await repo.get_copy_rows(entry_id)
    if not rows:
        return report
    rows = await _in_list_order(repo, entry_id, rows)

    for row in rows:
        if _is_render_only(row):
            await _render_stored(repo, entry_id, row, copy_run, report)

    file_source = is_file_source(copy_run.adapter)
    to_fetch: list[dict[str, Any]] = []
    for row in rows:
        if not _is_fetch_candidate(row, retry_skipped):
            continue
        # A source skip's mime_type is what the source sent instead of the
        # file (a login page), not the attachment's kind, so the
        # mode check of its retry leaves the kind to the sniff after the fetch.
        decided = (
            {**row, "mime_type": None} if row.get("skip_reason") in SOURCE_SKIP_REASONS else row
        )
        code = still_skipped(decided, attachments, copy_run.origins, file_source=file_source)
        if code is not None:
            await _write_skip(repo, entry_id, row, code, report)
        else:
            to_fetch.append(row)
    if not to_fetch:
        return report

    cap = attachments.max_file_mb * _MB
    count, used = await repo.count_copied_attachments(entry_id)
    budget = _Budget(bytes_left=BYTE_BUDGET_FILES * cap - used)
    slots = max(0, copy_run.max_per_entry - count) if budget.bytes_left > 0 else 0
    for row in to_fetch[slots:]:
        await _write_skip(repo, entry_id, row, "per_entry_limit", report)
    fits = to_fetch[:slots]
    if not fits:
        return report

    loop = asyncio.get_running_loop()
    deadline_at = loop.time() + copy_run.entry_deadline_s
    flights = [_Flight(row) for row in fits]
    tasks = {
        asyncio.create_task(
            _fetch_one(
                repo,
                entry_id,
                flight,
                cap=cap,
                mode=attachments.copy_on_ingest,
                budget=budget,
                deadline_at=deadline_at,
                copy_run=copy_run,
                report=report,
            )
        ): flight
        for flight in flights
    }
    try:
        _done, late = await asyncio.wait(tasks, timeout=copy_run.entry_deadline_s)
        # At the deadline, fetches still queued or in flight are cancelled; a
        # picture already fetched is stored (a render in flight finishes, later
        # ones are stored without a rendition).
        for task in late:
            if tasks[task].phase in ("queued", "fetching"):
                task.cancel()
        if late:
            await asyncio.wait(late)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for task, flight in tasks.items():
        if task.cancelled():
            if flight.sent:
                await _charge_attempt(repo, entry_id, flight.row, copy_run, report)
            else:
                report.pending += 1
    for task in tasks:
        if not task.cancelled() and task.exception() is not None:
            raise cast(BaseException, task.exception())
    return report
