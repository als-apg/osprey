"""Byte-stable markdown mirror writer for the qmd sidecar.

The qmd sidecar indexes a directory of markdown files. This module renders an
``enhanced_entries`` row into one such file and writes it to a sharded mirror
tree rooted at a configured directory::

    <mirror_root>/YYYY/MM/<percent-encoded entry_id>.md

Three properties of this module are load-bearing:

**Body text, not frontmatter.** qmd indexes document *content*; it does not
index YAML frontmatter as searchable metadata. Author, timestamp, source system
and scalar facility metadata are therefore rendered as ordinary prose in the
body, where they are retrievable.

**Byte-compare before write.** :func:`write_entry` reads the existing file and
returns without touching it when the rendered bytes are identical. An exporter
that rewrote unchanged entries would turn qmd's free "nothing changed" scan into
a full reindex and re-embed pass on every tick — at ALS scale (~135k entries)
that is tens of minutes of work per tick instead of ~12 seconds. Rendering is
therefore fully deterministic: no wall-clock stamps, no dict-iteration order.

**Lossless, invertible filenames.** ``entry_id`` is percent-encoded so that any
identifier — including ``a/b``, ``..``, unicode, and control characters — maps
to exactly one filesystem path, and :func:`entry_id_from_path` maps that path
back to the original identifier for Postgres hydration of qmd hits. Identifiers
whose encoding will not fit a filename (see :data:`MAX_ENCODED_ID_CHARS`) are
refused rather than mangled, so a mirrored path always inverts exactly.

Encoding keeps only ``[a-z0-9-_~]`` literal. Uppercase letters are encoded so
that two identifiers differing only in case cannot collide on a case-insensitive
filesystem (macOS, and any bind-mounted host directory on it); dots are encoded
so that no filename begins with ``.`` and is skipped as a dotfile.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

# Maximum size of one rendered document, including the truncation marker.
BODY_CAP_BYTES = 256 * 1024

# Appended in place of the content dropped when a document exceeds the cap.
TRUNCATION_MARKER = "\n\n[truncated]\n"

# Markdown suffix every mirrored file carries.
MARKDOWN_SUFFIX = ".md"

# Shard used when an entry carries no usable timestamp.
UNKNOWN_SHARD = ("unknown", "unknown")

# Longest encoded identifier that becomes a filename. Keeps the name plus the
# ``.md`` suffix and the temp-file decoration inside the 255-byte NAME_MAX of
# ext4/APFS. Splitting longer ids across nested directories would preserve
# invertibility but blow past macOS's 1024-byte PATH_MAX, so the limit is
# enforced here instead: an id this long is refused loudly rather than mirrored
# under a name hydration could not invert. One byte outside the literal set
# costs three characters, so this admits any 200-character ASCII id and any
# 33-character accented one — far beyond real logbook identifiers, which are
# short numeric or alphanumeric keys.
MAX_ENCODED_ID_CHARS = 200

# Characters kept literal in an encoded entry_id. Deliberately excludes
# uppercase letters (case-insensitive filesystems) and ``.`` (dotfiles).
_LITERAL_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyz0123456789-_~")

# Control characters are stripped before indexing; tab, newline and carriage
# return survive because they carry document structure.
_CONTROL_TRANSLATION: dict[int, None] = {
    codepoint: None for codepoint in range(0x20) if codepoint not in (0x09, 0x0A, 0x0D)
}
_CONTROL_TRANSLATION[0x7F] = None


def sanitize_text(value: Any) -> str:
    """Coerce a value to text that is safe to write into a markdown file.

    Invalid UTF-8 (including lone surrogates) is replaced rather than raising,
    and NUL plus every other C0/C1 control character is stripped. Tab, newline
    and carriage return are preserved.

    Args:
        value: Any value from a database row. ``None`` becomes an empty string;
            non-string values are rendered with ``str()``.

    Returns:
        Text containing only valid UTF-8 and no control characters.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    cleaned = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return cleaned.translate(_CONTROL_TRANSLATION)


def encode_entry_id(entry_id: str) -> str:
    """Percent-encode an entry identifier into a single filename component.

    The encoding is total and invertible: every byte outside ``[a-z0-9-_~]`` is
    written as ``%XX``, so path separators, ``..``, leading dots, spaces,
    unicode and control bytes all survive a round trip through the filesystem.

    Args:
        entry_id: Raw ``enhanced_entries.entry_id`` value.

    Returns:
        A filename-safe encoding of ``entry_id``, without the ``.md`` suffix.

    Raises:
        ValueError: If ``entry_id`` is empty, or if its encoding exceeds
            :data:`MAX_ENCODED_ID_CHARS`. Both indicate an identifier that
            cannot be mirrored without breaking the invertible-name contract;
            callers should log and skip the row rather than mangle its name.
    """
    raw = entry_id.encode("utf-8", errors="replace")
    if not raw:
        raise ValueError("entry_id must not be empty")

    encoded = "".join(chr(byte) if byte in _LITERAL_BYTES else f"%{byte:02X}" for byte in raw)
    if len(encoded) > MAX_ENCODED_ID_CHARS:
        raise ValueError(
            f"entry_id is too long to mirror: encodes to {len(encoded)} "
            f"characters, limit is {MAX_ENCODED_ID_CHARS}"
        )
    return encoded


def decode_entry_id(encoded: str) -> str:
    """Invert :func:`encode_entry_id`.

    Args:
        encoded: An encoded identifier, without the ``.md`` suffix. Use
            :func:`entry_id_from_path` when starting from a mirrored path.

    Returns:
        The original ``entry_id``.
    """
    return unquote(encoded, encoding="utf-8", errors="replace")


def entry_id_from_path(path: Path | str, mirror_root: Path | str | None = None) -> str:
    """Recover the ``entry_id`` that produced a mirrored markdown file.

    This is the hydration helper: qmd returns file paths, and ARIEL needs the
    Postgres primary key to fetch the authoritative row.

    Args:
        path: Path of a mirrored markdown file. Only its filename carries the
            identifier, so a path expressed in the sidecar container's mount
            namespace decodes as well as a host path.
        mirror_root: Optional root to check ``path`` against. Pass it when the
            path and the root share a namespace and the caller wants foreign
            paths rejected; omit it when they do not.

    Returns:
        The original ``entry_id``.

    Raises:
        ValueError: If ``path`` carries no ``.md`` suffix, or if
            ``mirror_root`` is given and ``path`` does not sit under it.
    """
    candidate = Path(path)
    if mirror_root is not None and not candidate.is_relative_to(Path(mirror_root)):
        raise ValueError(f"path is outside the mirror root: {path!r}")
    if candidate.suffix != MARKDOWN_SUFFIX:
        raise ValueError(f"path is not a mirrored markdown file: {path!r}")
    return decode_entry_id(candidate.name[: -len(MARKDOWN_SUFFIX)])


def render_entry(entry: Mapping[str, Any]) -> str:
    """Render one ``enhanced_entries`` row to markdown.

    Author, timestamp, source system and scalar ``metadata`` values are emitted
    as body prose because qmd does not index frontmatter. The output is a pure
    function of the row: rendering the same row twice yields identical bytes.

    Args:
        entry: An ``enhanced_entries`` row, typically an
            :class:`~osprey.services.ariel_search.models.EnhancedLogbookEntry`.
            Missing fields degrade to ``unknown`` rather than raising.

    Returns:
        The markdown document, always ending in a newline and never exceeding
        :data:`BODY_CAP_BYTES` including :data:`TRUNCATION_MARKER`.
    """
    entry_id = sanitize_text(entry.get("entry_id")).strip()
    author = sanitize_text(entry.get("author")).strip()
    source = sanitize_text(entry.get("source_system")).strip()
    raw_text = sanitize_text(entry.get("raw_text")).strip()
    stamp = _format_timestamp(entry.get("timestamp"))

    lines = [
        f"# Entry {entry_id or 'unidentified'}",
        "",
        f"Logged by {author or 'unknown'} on {stamp}.",
        f"Source: {source or 'unknown'}.",
    ]
    metadata_line = _render_metadata(entry.get("metadata"))
    if metadata_line:
        lines.append(metadata_line)
    lines.extend(["", raw_text])

    document = "\n".join(lines).rstrip("\n") + "\n"
    return _apply_cap(document)


def mirror_path(mirror_root: Path | str, entry: Mapping[str, Any]) -> Path:
    """Resolve the mirror path an entry's markdown file must occupy.

    Args:
        mirror_root: Root directory of the markdown mirror. It need not exist.
        entry: An ``enhanced_entries`` row. ``entry_id`` is required;
            ``timestamp`` selects the ``YYYY/MM`` shard and falls back to
            ``unknown/unknown``.

    Returns:
        The resolved absolute path for the entry, ``<root>/YYYY/MM/<name>.md``.

    Raises:
        ValueError: If ``entry_id`` is missing, empty or too long to encode, or
            if the resolved path would escape ``mirror_root``.
    """
    root = Path(mirror_root).resolve()
    entry_id = entry.get("entry_id")
    if not isinstance(entry_id, str):
        entry_id = "" if entry_id is None else str(entry_id)

    shard = _shard(entry.get("timestamp"))
    name = encode_entry_id(entry_id) + MARKDOWN_SUFFIX

    resolved = root.joinpath(*shard, name).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes the mirror root: {entry_id!r}")
    return resolved


def write_entry(mirror_root: Path | str, entry: Mapping[str, Any]) -> bool:
    """Write an entry's markdown file, skipping writes that change nothing.

    The rendered bytes are compared against the file already on disk. When they
    match, the file is left completely untouched — same contents, same mtime —
    so qmd's next scan sees an unchanged corpus and does no reindex work. When
    they differ, the file is replaced atomically so the sidecar never indexes a
    half-written document.

    Args:
        mirror_root: Root directory of the markdown mirror. Parent directories
            are created on demand, but only when a write actually happens.
        entry: An ``enhanced_entries`` row to mirror.

    Returns:
        ``True`` if the file was created or updated, ``False`` if the on-disk
        content already matched. Callers touch the sidecar's update marker only
        when this returns ``True``.

    Raises:
        ValueError: If the entry has no usable ``entry_id`` or its path would
            escape ``mirror_root``.
        OSError: If the file cannot be written.
    """
    path = mirror_path(mirror_root, entry)
    data = render_entry(entry).encode("utf-8")

    if _matches_on_disk(path, data):
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    # Leading dot keeps a partial write out of qmd's index if it ever scans
    # mid-write; the pid keeps concurrent writers from sharing a temp file.
    temp_path = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise
    return True


def _matches_on_disk(path: Path, data: bytes) -> bool:
    """Report whether ``path`` already holds exactly ``data``.

    Args:
        path: Candidate file path, which may not exist.
        data: The bytes that would be written.

    Returns:
        ``True`` only if the file exists and its content is byte-identical. Any
        read failure returns ``False`` so the write is attempted and surfaces
        the real error.
    """
    try:
        if path.stat().st_size != len(data):
            return False
        return path.read_bytes() == data
    except OSError:
        return False


def _shard(timestamp: Any) -> tuple[str, ...]:
    """Compute the ``YYYY/MM`` shard components for an entry timestamp.

    Args:
        timestamp: A ``datetime``, an ISO-8601 string, or anything else.

    Returns:
        Two path components, or :data:`UNKNOWN_SHARD` when no date is usable.
    """
    moment = _coerce_datetime(timestamp)
    if moment is None:
        return UNKNOWN_SHARD
    return (f"{moment.year:04d}", f"{moment.month:02d}")


def _format_timestamp(timestamp: Any) -> str:
    """Render an entry timestamp as UTC prose for the document body.

    Args:
        timestamp: A ``datetime``, an ISO-8601 string, or anything else.

    Returns:
        ``YYYY-MM-DD HH:MM:SS UTC``, or ``unknown time`` when unusable.
    """
    moment = _coerce_datetime(timestamp)
    if moment is None:
        return "unknown time"
    return moment.strftime("%Y-%m-%d %H:%M:%S UTC")


def _coerce_datetime(timestamp: Any) -> datetime | None:
    """Normalize a timestamp to an aware UTC ``datetime``.

    Naive values are read as UTC, matching the ``TIMESTAMPTZ`` column's storage.

    Args:
        timestamp: A ``datetime``, an ISO-8601 string, or anything else.

    Returns:
        The UTC-normalized datetime, or ``None`` if it cannot be interpreted.
    """
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError:
            return None
    if not isinstance(timestamp, datetime):
        return None
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _render_metadata(metadata: Any) -> str:
    """Render scalar facility metadata as one line of searchable prose.

    Nested structures are skipped: they belong to the Postgres row, and folding
    them into text would make the rendering sensitive to serialization detail.
    Keys are sorted so the output does not depend on dict ordering.

    Args:
        metadata: The row's ``metadata`` JSONB value, usually a dict.

    Returns:
        A single line such as ``Category: ops. Level: normal.``, or an empty
        string when no scalar values are present.
    """
    if not isinstance(metadata, Mapping):
        return ""

    fragments = []
    for key in sorted(str(k) for k in metadata):
        value = metadata[key]
        if not isinstance(value, str | int | float | bool):
            continue
        text = sanitize_text(value).strip()
        if not text:
            continue
        fragments.append(f"{sanitize_text(key).strip()}: {text}.")
    return " ".join(fragments)


def _apply_cap(document: str) -> str:
    """Truncate an oversized document and mark the truncation.

    Args:
        document: The fully rendered markdown document.

    Returns:
        ``document`` unchanged when it fits, otherwise a UTF-8-clean prefix
        ending in :data:`TRUNCATION_MARKER`, with the total at or below
        :data:`BODY_CAP_BYTES`.
    """
    encoded = document.encode("utf-8")
    if len(encoded) <= BODY_CAP_BYTES:
        return document

    budget = BODY_CAP_BYTES - len(TRUNCATION_MARKER.encode("utf-8"))
    # errors="ignore" drops a multi-byte character split by the cut.
    return encoded[:budget].decode("utf-8", errors="ignore") + TRUNCATION_MARKER
