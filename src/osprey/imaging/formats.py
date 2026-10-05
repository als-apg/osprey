"""The picture format registry: magic-byte sniffing, skip reasons and viewability.

Pure and dependency-free (standard library only). Classification reads bytes,
never decodes them: the in-process sniff looks at the first
:data:`SNIFF_BYTES` bytes for a magic number and never calls Pillow, so a
hostile file is classified without any decoder touching it. Decoding happens
only inside the isolated render worker, which opens with ``formats=ACCEPTED``
and checks the format Pillow reports against :func:`admitted_formats`.

Classification order:

1. magic classes (accepted images, then reserved binary formats);
2. ``markup`` -- an HTML page, the usual body of a login or proxy page served
   where a picture was expected;
3. the reserved svg row (``<svg``, ``<?xml``);
4. the reserved text row -- every byte of the window is printable UTF-8 (this
   is how csv and other text data is classed);
5. unknown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

SNIFF_BYTES = 32
"""Bytes the magic-number check reads."""

TEXT_WINDOW_BYTES = 64
"""Bytes read for the markup, svg and text checks only."""

_MAX_LEADING_WHITESPACE = 32
_UTF8_BOM = b"\xef\xbb\xbf"
_ASCII_WHITESPACE = b" \t\n\r\x0b\x0c"

RENDITION_MAX_BYTES = int(3.5 * 1024 * 1024)
"""Largest rendition stored for a picture.

Renditions are what agents see, so they are kept small; above this size the
worker re-encodes once smaller, else the picture is the content skip
``rendition_too_large``.
"""

RENDITION_MAX_SIDE = 1024
"""Longest side of a rendition, in pixels."""

RENDITION_MODES: frozenset[str] = frozenset({"RGB", "RGBA", "L", "LA"})
"""Modes a rendition is encoded in."""

Kind = Literal["image", "reserved", "markup", "unknown"]
Status = Literal["accepted", "reserved"]

OCTET_STREAM = "application/octet-stream"
HTML_MIME = "text/html"
TEXT_MIME = "text/plain"
SVG_MIME = "image/svg+xml"


@dataclass(frozen=True)
class FormatRow:
    """One registry row.

    Attributes:
        name: Row name (``png``, ``svg``, ``video`` ...).
        status: ``accepted`` rows are rendered; ``reserved`` rows are known
            formats that are stored but never rendered.
        mimes: Every MIME type the row's magic class can sniff to; the first
            is the canonical one.
        pillow_formats: The Pillow ``Image.format`` values this magic class
            admits (empty for reserved rows). A decoded format outside the set
            is the content skip ``format_mismatch``.
    """

    name: str
    status: Status
    mimes: tuple[str, ...]
    pillow_formats: frozenset[str] = frozenset()


ROWS: dict[str, FormatRow] = {
    row.name: row
    for row in (
        FormatRow("png", "accepted", ("image/png",), frozenset({"PNG"})),
        FormatRow("jpeg", "accepted", ("image/jpeg",), frozenset({"JPEG", "MPO"})),
        FormatRow("gif", "accepted", ("image/gif",), frozenset({"GIF"})),
        FormatRow("webp", "accepted", ("image/webp",), frozenset({"WEBP"})),
        FormatRow("bmp", "accepted", ("image/bmp",), frozenset({"BMP", "DIB"})),
        FormatRow("tiff", "accepted", ("image/tiff",), frozenset({"TIFF"})),
        FormatRow("svg", "reserved", (SVG_MIME,)),
        FormatRow("pdf", "reserved", ("application/pdf",)),
        FormatRow("text", "reserved", (TEXT_MIME,)),
        FormatRow("heif", "reserved", ("image/heic", "image/avif")),
        FormatRow(
            "video",
            "reserved",
            ("video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"),
        ),
        FormatRow(
            "audio",
            "reserved",
            ("audio/mp4", "audio/mpeg", "audio/flac", "audio/ogg", "audio/wav"),
        ),
        FormatRow(
            "archive",
            "reserved",
            (
                "application/zip",
                "application/gzip",
                "application/x-bzip2",
                "application/x-xz",
                "application/x-7z-compressed",
                "application/vnd.rar",
            ),
        ),
    )
}

_ROW_BY_MIME: dict[str, FormatRow] = {mime: row for row in ROWS.values() for mime in row.mimes}

IMAGE_MIMES: frozenset[str] = frozenset(
    row.mimes[0] for row in ROWS.values() if row.status == "accepted"
)
"""Canonical MIME types of the accepted rows -- the only viewable types."""

ACCEPTED: tuple[str, ...] = ("PNG", "JPEG", "GIF", "WEBP", "BMP", "DIB", "TIFF")
"""Pillow opener names passed as ``Image.open(fp, formats=ACCEPTED)``.

MPO is not listed: Pillow reaches it through the JPEG opener and has no MPO
opener of its own, and naming an unregistered opener makes ``Image.open``
raise.
"""


@dataclass(frozen=True)
class SniffResult:
    """Outcome of :func:`sniff`.

    Attributes:
        kind: ``image`` (an accepted row), ``reserved`` (a known format that is
            never rendered), ``markup`` (an HTML page) or ``unknown``.
        mime: The sniffed MIME type (``application/octet-stream`` when unknown).
        row: The registry row name, or ``None`` for markup and unknown bytes.
        skip_reason: ``None`` for an image, ``reserved_format`` for a reserved
            row, else ``not_an_image``.
    """

    kind: Kind
    mime: str
    row: str | None
    skip_reason: str | None

    @property
    def is_image(self) -> bool:
        """Whether the bytes are an accepted picture format."""
        return self.kind == "image"


def _row_result(row_name: str, mime: str) -> SniffResult:
    row = ROWS[row_name]
    if row.status == "accepted":
        return SniffResult("image", mime, row.name, None)
    return SniffResult("reserved", mime, row.name, "reserved_format")


_UNKNOWN = SniffResult("unknown", OCTET_STREAM, None, "not_an_image")
_MARKUP = SniffResult("markup", HTML_MIME, None, "not_an_image")

# (prefix, row, mime) checked with startswith on the first SNIFF_BYTES bytes.
_PREFIX_MAGIC: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpeg", "image/jpeg"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
    (b"II*\x00", "tiff", "image/tiff"),
    (b"MM\x00*", "tiff", "image/tiff"),
    (b"%PDF", "pdf", "application/pdf"),
    (b"PK\x03\x04", "archive", "application/zip"),
    (b"PK\x05\x06", "archive", "application/zip"),
    (b"PK\x07\x08", "archive", "application/zip"),
    (b"\x1f\x8b", "archive", "application/gzip"),
    (b"\xfd7zXZ\x00", "archive", "application/x-xz"),
    (b"7z\xbc\xaf\x27\x1c", "archive", "application/x-7z-compressed"),
    (b"Rar!\x1a\x07", "archive", "application/vnd.rar"),
    (b"\x1aE\xdf\xa3", "video", "video/webm"),
    (b"ID3", "audio", "audio/mpeg"),
    (b"fLaC", "audio", "audio/flac"),
    (b"OggS", "audio", "audio/ogg"),
)

# RIFF containers: the form type at bytes 8..12 decides the row.
_RIFF_FORMS: dict[bytes, tuple[str, str]] = {
    b"WEBP": ("webp", "image/webp"),
    b"WAVE": ("audio", "audio/wav"),
    b"AVI ": ("video", "video/x-msvideo"),
}

# ISO base media (``ftyp`` at offset 4): the major brand decides the row.
_FTYP_BRANDS: dict[bytes, tuple[str, str]] = {
    b"heic": ("heif", "image/heic"),
    b"heix": ("heif", "image/heic"),
    b"mif1": ("heif", "image/heic"),
    b"msf1": ("heif", "image/heic"),
    b"avif": ("heif", "image/avif"),
    b"avis": ("heif", "image/avif"),
    b"qt  ": ("video", "video/quicktime"),
    b"M4A ": ("audio", "audio/mp4"),
    b"M4B ": ("audio", "audio/mp4"),
}

# BITMAPFILEHEADER is followed by a DIB header whose first field is its own
# size; checking it keeps a text file that starts with "BM" out of the bmp row.
_BMP_DIB_HEADER_SIZES = frozenset({12, 16, 40, 52, 56, 64, 108, 124})

_MARKUP_PREFIXES: tuple[bytes, ...] = (b"<!doctype html", b"<html", b"<head", b"<body")
_SVG_PREFIXES: tuple[bytes, ...] = (b"<svg", b"<?xml", b"<!doctype svg")


def _magic(head: bytes) -> SniffResult | None:
    if head[:4] == b"RIFF" and head[8:12] in _RIFF_FORMS:
        return _row_result(*_RIFF_FORMS[head[8:12]])
    if head[4:8] == b"ftyp":
        row_name, mime = _FTYP_BRANDS.get(head[8:12], ("video", "video/mp4"))
        return _row_result(row_name, mime)
    if head[:2] == b"BM" and int.from_bytes(head[14:18], "little") in _BMP_DIB_HEADER_SIZES:
        return _row_result("bmp", "image/bmp")
    if head[:3] == b"BZh" and head[3:4] in b"123456789" and head[3:4]:
        return _row_result("archive", "application/x-bzip2")
    for prefix, row_name, mime in _PREFIX_MAGIC:
        if head.startswith(prefix):
            return _row_result(row_name, mime)
    return None


def _stripped_window(window: bytes) -> bytes:
    """The window with a UTF-8 BOM and up to 32 bytes of ASCII whitespace removed."""
    if window.startswith(_UTF8_BOM):
        window = window[len(_UTF8_BOM) :]
    skip = 0
    while (
        skip < _MAX_LEADING_WHITESPACE and skip < len(window) and window[skip] in _ASCII_WHITESPACE
    ):
        skip += 1
    return window[skip:]


def _is_printable_text(window: bytes, *, truncated: bool) -> bool:
    """Whether every byte of the window is printable UTF-8 text.

    When the data continues past the window, a multi-byte character cut by the
    window edge is not held against it.
    """
    if window.startswith(_UTF8_BOM):
        window = window[len(_UTF8_BOM) :]
    if not window:
        return False
    try:
        text = window.decode("utf-8")
    except UnicodeDecodeError as exc:
        if not (truncated and exc.reason == "unexpected end of data" and exc.end == len(window)):
            return False
        text = window[: exc.start].decode("utf-8")
    return all(ch.isprintable() or ch in "\t\n\r" for ch in text)


def sniff(data: bytes) -> SniffResult:
    """Classify bytes by magic number; never decodes and never calls Pillow.

    Args:
        data: The picture bytes, or at least their first
            :data:`TEXT_WINDOW_BYTES` bytes.

    Returns:
        The :class:`SniffResult`; ``kind == "image"`` only for an accepted row.
    """
    head = bytes(data[:SNIFF_BYTES])
    found = _magic(head)
    if found is not None:
        return found

    window = bytes(data[:TEXT_WINDOW_BYTES])
    lowered = _stripped_window(window).lower()
    if lowered.startswith(_MARKUP_PREFIXES):
        return _MARKUP
    if lowered.startswith(_SVG_PREFIXES):
        return _row_result("svg", SVG_MIME)
    if _is_printable_text(window, truncated=len(data) > TEXT_WINDOW_BYTES):
        return _row_result("text", TEXT_MIME)
    return _UNKNOWN


def is_markup(result: SniffResult | None) -> bool:
    """Whether a sniff result is a web page (HTML) rather than a picture.

    This is the one answer to "is this a web page": a source that serves an
    HTML page where a picture was expected (a login or proxy page) is refused.
    """
    return isinstance(result, SniffResult) and result.kind == "markup"


def _bare_mime(mime: str | None) -> str:
    if not mime:
        return ""
    return mime.split(";", 1)[0].strip().lower()


def kind(mime: str | None) -> Kind:
    """The registry kind of a stored or sniffed MIME type.

    ``image`` only for the accepted rows; ``reserved`` for a reserved row;
    ``markup`` for HTML; else ``unknown``.
    """
    bare = _bare_mime(mime)
    if bare == HTML_MIME:
        return "markup"
    row = _ROW_BY_MIME.get(bare)
    if row is None:
        return "unknown"
    return "image" if row.status == "accepted" else "reserved"


def admitted_formats(mime: str | None) -> frozenset[str]:
    """The Pillow formats a sniffed MIME's magic class admits (empty if none)."""
    row = _ROW_BY_MIME.get(_bare_mime(mime))
    return row.pillow_formats if row is not None else frozenset()


# -- skip reasons --------------------------------------------------------------

CONTENT_SKIP_REASONS: frozenset[str] = frozenset(
    {
        "not_an_image",
        "reserved_format",
        "decoder_failed",
        "format_mismatch",
        "rendition_too_large",
        "not_a_regular_file",
    }
)
"""Terminal: the bytes themselves decide, so a retry gives the same answer."""

CONFIG_SKIP_REASONS: frozenset[str] = frozenset(
    {"origin_not_allowed", "size_cap", "per_entry_limit", "copy_on_ingest_mode"}
)
"""Re-evaluated against the current configuration."""

SOURCE_SKIP_REASONS: frozenset[str] = frozenset({"source_gone", "source_refused", "fetch_failed"})
"""The source did not deliver the picture; retried by backfill."""

ROW_SKIP_REASONS: frozenset[str] = CONTENT_SKIP_REASONS | CONFIG_SKIP_REASONS | SOURCE_SKIP_REASONS
"""Every code a stored ``attachment_files.skip_reason`` may hold."""

SUMMARY_ONLY_REASONS: frozenset[str] = frozenset({"no_source_url"})
"""Codes that appear in attachment summaries only, never on a stored row."""

_BACKFILL = "osprey ariel attachments backfill"

_SKIP_REASON_TEXT: dict[str, str] = {
    "not_an_image": "the file is not a picture; it is stored for download but never shown",
    "reserved_format": (
        "the file is a format that is not rendered (svg, pdf, text data, video, audio "
        "or an archive); it is stored for download but never shown"
    ),
    "decoder_failed": (
        f"the picture could not be decoded; run `{_BACKFILL} --retry-decoder-failed` to try again"
    ),
    "format_mismatch": ("the picture's content does not match the format its first bytes announce"),
    "rendition_too_large": (
        "the picture is too large to prepare for viewing; the original is stored for download"
    ),
    "not_a_regular_file": "the source path is not a regular file (a link, directory or device)",
    "origin_not_allowed": (
        "the picture's host is not an allowed origin; add it to "
        f"`ariel.attachments.allowed_origins`, then run `{_BACKFILL}`"
    ),
    "size_cap": (
        "the picture is larger than `ariel.attachments.max_file_mb`; raise it, "
        f"then run `{_BACKFILL}`"
    ),
    "per_entry_limit": (
        "the entry already holds its per-entry budget of pictures (a count limit and four "
        "times `ariel.attachments.max_file_mb` in bytes)"
    ),
    "copy_on_ingest_mode": (
        "the file is not a picture and `ariel.attachments.copy_on_ingest` is `images`; set it "
        f"to `all`, then run `{_BACKFILL}`"
    ),
    "source_gone": (
        f"the picture is no longer at its source; run `{_BACKFILL}` once it is restored"
    ),
    "source_refused": (
        "the server refused the picture or returned a web page instead of it (a login or "
        f"proxy page); fix access to the source, then run `{_BACKFILL}` to retry"
    ),
    "fetch_failed": (
        f"the picture could not be fetched from its source; run `{_BACKFILL}` to retry"
    ),
    "no_source_url": "the attachment has no usable source address, so there is nothing to copy",
}


def skip_reason_text(code: str) -> str:
    """Human text for a skip code, naming the config key or command that acts on it."""
    text = _SKIP_REASON_TEXT.get(code)
    if text is not None:
        return text
    return f"skipped ({code})"


# -- viewable -------------------------------------------------------------------


def _field(row: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(name)
    return getattr(row, name, None)


def is_viewable(row: Mapping[str, Any] | Any) -> bool:
    """Whether a stored attachment row is a picture whose finished copy exists.

    ``copy_status == 'copied' AND skip_reason IS NULL AND kind(mime_type) ==
    'image' AND rendition_sha256 IS NOT NULL``; accepts a mapping or an object
    with those attributes. :data:`VIEWABLE_SQL` is the same predicate in SQL.
    """
    return (
        _field(row, "copy_status") == "copied"
        and _field(row, "skip_reason") is None
        and kind(_field(row, "mime_type")) == "image"
        and _field(row, "rendition_sha256") is not None
    )


def _is_sql_identifier(name: str) -> bool:
    """Whether *name* is a plain ASCII SQL identifier: a letter or underscore, then word chars."""
    return name.isascii() and name.isidentifier()


def viewable_sql(alias: str | None = None) -> str:
    """The :func:`is_viewable` predicate as a SQL fragment over ``attachment_files``.

    Args:
        alias: Optional table alias to qualify the columns with.

    Raises:
        ValueError: If ``alias`` is not a plain SQL identifier.
    """
    if alias is None:
        prefix = ""
    elif _is_sql_identifier(alias):
        prefix = f"{alias}."
    else:
        raise ValueError(f"not a SQL identifier: {alias!r}")
    mimes = ", ".join(f"'{mime}'" for mime in sorted(IMAGE_MIMES))
    return (
        f"({prefix}copy_status = 'copied' AND {prefix}skip_reason IS NULL"
        f" AND {prefix}mime_type IN ({mimes})"
        f" AND {prefix}rendition_sha256 IS NOT NULL)"
    )


VIEWABLE_SQL = viewable_sql()
"""SQL fragment matching exactly the rows :func:`is_viewable` accepts."""

CAPTION_NOT_DONE_SQL = (
    "NOT (COALESCE(attachment_captions->f.attachment_id, '{}'::jsonb) ? %(model)s)"
)
"""SQL fragment: picture ``f`` has no caption entry under the ``%(model)s`` model.

Evaluated over ``attachment_files f`` correlated with an ``enhanced_entries``
row. The COALESCE keeps a picture "not done" both when ``attachment_captions``
is NULL and when it has no key for the picture; either one alone would make
the clause NULL, which reads as done.
"""


def image_table_not_done_sql(table: str) -> str:
    """SQL fragment: picture ``f`` has no row in the image embedding table ``table``.

    Args:
        table: The image embedding table name, spliced as an identifier.

    Raises:
        ValueError: If ``table`` is not a plain SQL identifier.
    """
    if not _is_sql_identifier(table):
        raise ValueError(f"not a SQL identifier: {table!r}")
    return f"NOT EXISTS (SELECT 1 FROM {table} t WHERE t.attachment_id = f.attachment_id)"
