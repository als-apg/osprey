"""Shared attachment processing for ARIEL.

Standalone functions used by both MCP tools and REST API for validating,
reading, and storing file attachments.
"""

from __future__ import annotations

import hashlib
import mimetypes
import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import AttachmentInfo

logger = get_logger("ariel")

#: Every attachment id: native ids carry 12 hex digits, copied ids 24.
#: Match it with :func:`re.fullmatch`; ``$`` alone also accepts a trailing newline.
ATTACHMENT_ID_RE = r"^att-[0-9a-f]{12}(?:[0-9a-f]{12})?$"

#: The url a native (web/MCP-uploaded) item carries; group 1 is its attachment id.
_NATIVE_URL_RE = re.compile(r"/api/attachments/([^/?#]+)")

#: Copied ids keep this many hex digits of the sha256.
_COPIED_ID_HEX = 24

#: Used when ``ariel.attachments.max_file_mb`` is unset.
DEFAULT_MAX_ATTACHMENT_MB = 10


class AttachmentValidationError(Exception):
    """Raised when an attachment fails validation."""


def max_attachment_bytes() -> int:
    """Largest attachment an entry may carry, in bytes.

    Attachments are stored as ``BYTEA`` in the same Postgres the logbook lives
    in, so both directions of this number are a site storage decision: a
    facility that attaches raw scope traces raises it and pays for the space, a
    facility that wants the database small lowers it.

    Read on every call — never cached at import — so a test and a re-primed
    process see the value their config declares. The ``ariel.attachments``
    block is parsed by ``AttachmentsConfig``; a block that does not parse is
    logged and the default cap kept, so the bound never disappears.
    """
    default = DEFAULT_MAX_ATTACHMENT_MB * 1024 * 1024
    try:
        from osprey.utils.config import get_config_value

        section = get_config_value("ariel.attachments", {})
    except Exception:
        logger.debug("No config available for ariel.attachments", exc_info=True)
        return default
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        logger.warning(
            "ariel.attachments must be a mapping (got %r); using %d MB",
            section,
            DEFAULT_MAX_ATTACHMENT_MB,
        )
        return default

    # Imported here: the config module imports this package for its default.
    from osprey.services.ariel_search.config import AttachmentsConfig

    try:
        config = AttachmentsConfig.from_dict(section)
    except ValueError as exc:
        # The block is refused at config parse; the size check still keeps a bound.
        logger.warning("%s; using %d MB", exc, DEFAULT_MAX_ATTACHMENT_MB)
        return default
    return config.max_file_mb * 1024 * 1024


def validate_file_size(size: int, filename: str) -> None:
    """Validate that a file is within the size limit.

    Args:
        size: File size in bytes.
        filename: Filename for error messages.

    Raises:
        AttachmentValidationError: If the file exceeds
            ``ariel.attachments.max_file_mb``.
    """
    limit = max_attachment_bytes()
    if size > limit:
        max_mb = limit / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise AttachmentValidationError(
            f"File '{filename}' is {actual_mb:.1f} MB, exceeds {max_mb:.0f} MB limit."
        )


def guess_mime_type(filename: str) -> str | None:
    """Guess MIME type from filename extension.

    Args:
        filename: Filename with extension.

    Returns:
        MIME type string or None if unknown.
    """
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type


def generate_attachment_id() -> str:
    """Generate a unique attachment ID.

    Returns:
        String like "att-a1b2c3d4e5f6".
    """
    return f"att-{uuid.uuid4().hex[:12]}"


def read_local_file(path: str | Path) -> tuple[bytes, str, str | None]:
    """Read a local file and return its data, filename, and MIME type.

    Args:
        path: Path to the file.

    Returns:
        Tuple of (data, filename, mime_type).

    Raises:
        AttachmentValidationError: If file doesn't exist or exceeds size limit.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise AttachmentValidationError(f"File not found: {path}")

    if not file_path.is_file():
        raise AttachmentValidationError(f"Not a file: {path}")

    size = file_path.stat().st_size
    validate_file_size(size, file_path.name)

    data = file_path.read_bytes()
    filename = file_path.name
    mime_type = guess_mime_type(filename)

    return data, filename, mime_type


async def store_native_attachment(
    repository: ARIELRepository,
    entry_id: str,
    *,
    filename: str,
    declared_mime: str | None,
    data: bytes,
) -> AttachmentInfo:
    """Store one natively written attachment and return its JSONB item.

    The bytes are always kept, whatever ``ariel.attachments.copy_on_ingest``
    says. When the schema records copy state, the bytes are sniffed and prepared
    with no transaction open, then written as one ``copied`` row carrying its
    rendition (or its content skip reason) in a single transaction that locks the
    entry row first and clears the two image-module status keys. A render worker
    that cannot run leaves the row copied with no rendition, for the poll's render
    step to finish. A schema without copy state gets the plain attachment row and
    no preparation; backfill does that work later.

    Native writers never run enhancers and never fetch.

    Args:
        repository: ARIEL repository for database storage.
        entry_id: The entry the attachment belongs to.
        filename: Original filename.
        declared_mime: MIME type the uploader declared, or None.
        data: The attachment bytes.

    Returns:
        The ``AttachmentInfo`` item linking the stored attachment.
    """
    attachment_id = generate_attachment_id()
    facts = await repository.schema_facts()
    if not facts.has_copy_state:
        await repository.store_attachment(
            entry_id=entry_id,
            attachment_id=attachment_id,
            filename=filename,
            mime_type=declared_mime,
            data=data,
            size_bytes=len(data),
        )
        return {
            "url": f"/api/attachments/{attachment_id}",
            "type": declared_mime,
            "filename": filename,
        }

    # Imported here: the repository and the render stack import this package.
    from osprey.services.ariel_search.attachments import prepare as _prepare
    from osprey.services.ariel_search.attachments.copy import rendition_of
    from osprey.services.ariel_search.attachments.formats import sniff
    from osprey.services.ariel_search.database.repository import CopyRendition

    rendition: CopyRendition | None = None
    try:
        prepared = await _prepare.prepare_picture(data)
    except _prepare.RenderUnavailable as exc:
        logger.warning(
            "Render worker unavailable for attachment %s of %s (%s); stored without a rendition",
            attachment_id,
            entry_id,
            exc,
        )
        mime_type = sniff(data).mime
        skip_reason = None
    else:
        mime_type = prepared.mime_type
        skip_reason = prepared.skip_reason
        rendition = rendition_of(prepared)

    await repository.insert_native_attachment(
        entry_id,
        attachment_id,
        filename=filename,
        mime_type=mime_type,
        data=data,
        skip_reason=skip_reason,
        rendition=rendition,
    )
    return {
        "url": f"/api/attachments/{attachment_id}",
        "type": declared_mime or mime_type,
        "filename": filename,
    }


async def process_attachments_for_entry(
    entry_id: str,
    file_paths: list[str],
    repository: ARIELRepository,
) -> list[AttachmentInfo]:
    """Read, validate, and store attachments for an entry.

    Every file is validated before any is stored; each is then stored through
    :func:`store_native_attachment`.

    Args:
        entry_id: The entry ID to associate attachments with.
        file_paths: List of local file paths.
        repository: ARIEL repository for database storage.

    Returns:
        List of AttachmentInfo dicts for the entry's attachments JSONB.

    Raises:
        AttachmentValidationError: If any file fails validation.
    """
    files = [read_local_file(path) for path in file_paths]
    return [
        await store_native_attachment(
            repository,
            entry_id,
            filename=filename,
            declared_mime=mime_type,
            data=data,
        )
        for data, filename, mime_type in files
    ]


def _item_url(item: Any) -> str | None:
    """Return the item's url when it is a non-empty string, else None."""
    if not isinstance(item, Mapping):
        return None
    url = item.get("url")
    if not isinstance(url, str) or not url:
        return None
    return url


def is_native_item(item: Any) -> bool:
    """Return whether a JSONB attachment item points at a natively stored attachment.

    A native item's url is exactly ``/api/attachments/<id>``, with no query,
    fragment or further path segment.
    """
    url = _item_url(item)
    return url is not None and _NATIVE_URL_RE.fullmatch(url) is not None


def attachment_id_for(entry_id: str, item: Any) -> str | None:
    """Map a JSONB attachment item to its ``attachment_files`` row id.

    A native item maps to the id in its url. Any other non-empty string url maps
    to the deterministic ``att-<sha256(entry_id \\0 url)[:24]>``, so re-ingesting
    the same entry finds the same row. An empty, missing or non-string url has
    no row and returns None.
    """
    url = _item_url(item)
    if url is None:
        return None
    native = _NATIVE_URL_RE.fullmatch(url)
    if native is not None:
        return native.group(1)
    digest = hashlib.sha256(f"{entry_id}\0{url}".encode()).hexdigest()
    return f"att-{digest[:_COPIED_ID_HEX]}"


def _has_dot_dot_segment(path: str) -> bool:
    """Return whether any ``/`` or ``\\`` separated segment is ``..`` (percent-decoded too)."""
    for candidate in (path, unquote(path)):
        if ".." in re.split(r"[/\\]", candidate):
            return True
    return False


def fetchable_url(url: str, *, file_source: bool) -> bool:
    """Return whether an upstream attachment url may ever be fetched.

    Fetchable is an absolute ``http``/``https`` url with a host or, when the
    source is a generic file source, a relative path that can be confined under
    the source's attachment directory: no scheme, no leading ``/`` or ``\\``,
    no NUL. A ``..`` path segment is never fetchable.
    """
    if not isinstance(url, str) or not url or "\0" in url:
        return False
    if _has_dot_dot_segment(url):
        return False
    parts = urlsplit(url)
    if parts.scheme:
        return parts.scheme.lower() in ("http", "https") and bool(parts.hostname)
    if not file_source:
        return False
    if url.startswith(("/", "\\")) or ":" in url.split("/", 1)[0]:
        return False
    return True
