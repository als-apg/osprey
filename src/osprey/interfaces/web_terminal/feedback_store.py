"""Filesystem store for web-terminal feedback submissions.

Each submission is two JSON documents in one flat directory:

``ctx-<utc-ms>-<rand>.json``
    The bulky context bundle (collected panels, scrollback, transcript
    excerpts). Written **first**.
``fb-<utc-ms>-<rand>.json``
    The small header (identity, timestamps, channel, excerpt, and a
    ``context_file`` pointer). Written **last**, so an interrupted write leaves
    at most an orphan context file — never a header promising a context that
    was never stored. Readers drop orphans; the header alone is the record.

Both documents carry the record id *inside* the document as well as in their
filename, because the CLI reads a whole store by concatenating files
(``find … -exec cat {} +``) and filenames do not survive concatenation.

Writes go through :func:`_atomic_write`: serialize into a hidden temporary file
in the same directory, then ``os.replace``. ``os.replace`` is atomic per file on
POSIX, so a concurrent reader sees a complete document or no document — never a
truncation. It is *not* atomic across the two files, which is exactly why the
ordering above matters. Temporary names are dot-prefixed and ``.tmp``-suffixed
so they can never be matched by :data:`HEADER_GLOB` or :data:`CONTEXT_GLOB`.

The store is bounded by a byte ceiling (see :func:`prune_store`). Only bulk is
ever reclaimed: context documents are deleted oldest-first and their headers are
rewritten with ``context_pruned: true``, so the submission history stays
complete however small the ceiling is.

The module is deliberately free of application imports (pure standard library):
the CLI reads stores out of container volumes with no web-terminal app in
scope, and reuses the naming contract defined here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONTEXT_GLOB",
    "CONTEXT_PREFIX",
    "HEADER_GLOB",
    "HEADER_PREFIX",
    "context_filename",
    "header_filename",
    "list_headers",
    "new_record_id",
    "prune_store",
    "write_record",
]

HEADER_PREFIX = "fb-"
"""Filename prefix of a record header (and of the record id itself)."""

CONTEXT_PREFIX = "ctx-"
"""Filename prefix of a record's context document."""

HEADER_GLOB = "fb-*.json"
"""Glob that matches every committed header and nothing else."""

CONTEXT_GLOB = "ctx-*.json"
"""Glob that matches every committed context document and nothing else."""


def _now_ms() -> int:
    """Return the current UTC time as whole milliseconds since the epoch.

    Split out so tests can freeze the clock without patching :mod:`time`.
    """
    return int(time.time() * 1000)


def new_record_id() -> str:
    """Allocate a record id of the form ``fb-<utc-ms>-<8 hex>``.

    The millisecond prefix makes ids sort chronologically; the random suffix
    keeps two submissions made in the same millisecond distinct.
    """
    return f"{HEADER_PREFIX}{_now_ms()}-{uuid.uuid4().hex[:8]}"


def header_filename(record_id: str) -> str:
    """Return the header filename for *record_id* (the id is its stem)."""
    return f"{record_id}.json"


def context_filename(record_id: str) -> str:
    """Return the context-document filename paired with *record_id*."""
    return f"{CONTEXT_PREFIX}{record_id.removeprefix(HEADER_PREFIX)}.json"


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    """Serialize *data* to *path* as JSON, atomically.

    Writes a hidden ``.<name>.…​.tmp`` file in ``path.parent``, flushes and
    fsyncs it, then ``os.replace``s it over *path*. On any failure the
    temporary file is removed and the error re-raised, so a broken write never
    leaves debris behind. ``path.parent`` must already exist.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, default=str)
            handle.flush()
            with suppress(OSError):
                os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_record(
    feedback_dir: Path,
    header: dict[str, Any],
    context: dict[str, Any],
    *,
    record_id: str | None = None,
) -> str:
    """Write one feedback record into *feedback_dir* and return its id.

    *feedback_dir* is created if absent. The context document is written first
    and the header last; both are stamped with the allocated id, and the header
    additionally carries a ``context_file`` pointer at its paired context
    document. Those two keys are authoritative — values supplied by the caller
    under the same names are overwritten. The caller's dicts are not mutated.

    *record_id* lets the caller allocate the id (with :func:`new_record_id`)
    before the header exists, which is what a caller who has to *name* the
    record inside the record needs — the web-terminal route digests a payload
    whose truncation marker carries the id, and a digest of a payload naming a
    different id would not match the bytes the operator pasted. Omitted, the id
    is minted here as before. Either way it is the id both documents are stamped
    and filed under, and the id returned.

    Raises:
        ValueError: if *record_id* does not start with :data:`HEADER_PREFIX`.
            The id **is** the header's filename stem, so an id without the
            prefix files a header that :data:`HEADER_GLOB` cannot see — the
            submission would vanish from ``list_headers`` and the pruner, while
            its context document stayed behind as a prunable orphan.
    """
    if record_id is not None and not record_id.startswith(HEADER_PREFIX):
        raise ValueError(
            f"record_id must start with {HEADER_PREFIX!r} (it is the header filename stem); "
            f"got {record_id!r}"
        )

    feedback_dir = Path(feedback_dir)
    feedback_dir.mkdir(parents=True, exist_ok=True)

    record_id = record_id or new_record_id()
    context_name = context_filename(record_id)

    context_doc: dict[str, Any] = {"id": record_id}
    context_doc.update({k: v for k, v in context.items() if k != "id"})
    _atomic_write(feedback_dir / context_name, context_doc)

    header_doc: dict[str, Any] = {"id": record_id, "context_file": context_name}
    header_doc.update({k: v for k, v in header.items() if k not in ("id", "context_file")})
    _atomic_write(feedback_dir / header_filename(record_id), header_doc)

    return record_id


def _file_size(path: Path) -> int:
    """Return the size of *path* in bytes, or ``0`` if it cannot be stat'ed."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _context_sort_key(path: Path) -> tuple[int, int, str]:
    """Return the oldest-first ordering key for a context document.

    Age comes from the ``<utc-ms>`` component of the filename, not from the
    filesystem's mtime, so a store restored from a backup (or copied between
    volumes) still prunes in submission order. A context whose name carries no
    numeric timestamp cannot be placed on that timeline, so it sorts *after*
    every dateable one: foreign files are the last thing to be reclaimed.
    """
    stem = path.stem.removeprefix(CONTEXT_PREFIX)
    timestamp = stem.partition("-")[0]
    if timestamp.isdigit():
        return (0, int(timestamp), path.name)
    return (1, 0, path.name)


def _mark_context_pruned(header_path: Path) -> int:
    """Rewrite *header_path* with ``context_pruned: true``; return its size delta.

    A header that is absent (the deleted context was an orphan), unreadable, or
    not a JSON object is left alone and reports a delta of ``0`` — pruning must
    reclaim space even when a neighbouring file is damaged.
    """
    try:
        document = json.loads(header_path.read_text())
    except FileNotFoundError:
        return 0
    except (OSError, ValueError):
        logger.debug(
            "feedback store: not flagging unreadable header %s", header_path, exc_info=True
        )
        return 0
    if not isinstance(document, dict):
        logger.debug("feedback store: not flagging non-object header %s", header_path)
        return 0

    before = _file_size(header_path)
    document["context_pruned"] = True
    try:
        _atomic_write(header_path, document)
    except OSError:
        logger.warning(
            "feedback store: could not flag pruned header %s", header_path, exc_info=True
        )
        return 0
    return _file_size(header_path) - before


def prune_store(feedback_dir: Path, max_bytes: int) -> list[str]:
    """Delete context documents until *feedback_dir* fits in *max_bytes*.

    The measured size is the store's own files only — everything matched by
    :data:`HEADER_GLOB` or :data:`CONTEXT_GLOB` — so unrelated debris and
    in-flight temporary files never trigger a prune. While the store is over
    the ceiling, the oldest remaining context document (by the ``<utc-ms>``
    component of its name) is deleted and its header rewritten through
    :func:`_atomic_write` with ``context_pruned: true``. Headers are never
    deleted: the submission history survives at full length, only its bulk is
    reclaimed, and a store of headers alone can legitimately end up over the
    ceiling with nothing left to prune.

    Returns the ids whose context was deleted, oldest first. An orphan context
    (no header — an interrupted write) is reclaimed like any other and its id
    reported, but no header is written for it. A missing or non-directory
    *feedback_dir*, and a store already under the ceiling, both prune nothing.
    """
    directory = Path(feedback_dir)
    if not directory.is_dir():
        return []

    total = sum(
        _file_size(path) for glob in (HEADER_GLOB, CONTEXT_GLOB) for path in directory.glob(glob)
    )
    if total <= max_bytes:
        return []

    pruned: list[str] = []
    for context_path in sorted(directory.glob(CONTEXT_GLOB), key=_context_sort_key):
        if total <= max_bytes:
            break
        reclaimed = _file_size(context_path)
        try:
            context_path.unlink()
        except OSError:
            logger.warning(
                "feedback store: could not prune context %s", context_path, exc_info=True
            )
            continue
        total -= reclaimed
        record_id = f"{HEADER_PREFIX}{context_path.stem.removeprefix(CONTEXT_PREFIX)}"
        pruned.append(record_id)
        total += _mark_context_pruned(directory / header_filename(record_id))

    return pruned


def list_headers(feedback_dir: Path) -> list[dict[str, Any]]:
    """Return every readable header in *feedback_dir*, sorted by record id.

    Only :data:`HEADER_GLOB` is read: context documents and in-flight temporary
    files are ignored. A missing directory (or a path that is not a directory)
    yields an empty list — an unused store is a normal state, not an error — as
    does a header that cannot be parsed as a JSON object, which is logged at
    debug level and skipped so one damaged file cannot hide the rest.
    """
    directory = Path(feedback_dir)
    if not directory.is_dir():
        return []

    headers: list[dict[str, Any]] = []
    for path in sorted(directory.glob(HEADER_GLOB)):
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            logger.debug("feedback store: skipping unreadable header %s", path, exc_info=True)
            continue
        if not isinstance(document, dict):
            logger.debug("feedback store: skipping non-object header %s", path)
            continue
        document.setdefault("id", path.stem)
        headers.append(document)

    headers.sort(key=lambda doc: str(doc.get("id", "")))
    return headers
