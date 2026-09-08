"""Shared attachment processing for ARIEL.

Standalone functions used by both MCP tools and REST API for validating,
reading, and storing file attachments.
"""

from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import AttachmentInfo

logger = get_logger("ariel")

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
    process see the value their config declares.
    """
    megabytes = DEFAULT_MAX_ATTACHMENT_MB
    try:
        from osprey.utils.config import get_config_value

        configured = get_config_value("ariel.attachments.max_file_mb", DEFAULT_MAX_ATTACHMENT_MB)
    except Exception:
        logger.debug("No config available for ariel.attachments.max_file_mb", exc_info=True)
        configured = None
    if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
        megabytes = configured
    elif configured is not None and configured != DEFAULT_MAX_ATTACHMENT_MB:
        logger.warning(
            "ariel.attachments.max_file_mb must be an integer >= 1 (got %r); using %d MB",
            configured,
            DEFAULT_MAX_ATTACHMENT_MB,
        )
    return megabytes * 1024 * 1024


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


async def process_attachments_for_entry(
    entry_id: str,
    file_paths: list[str],
    repository: ARIELRepository,
) -> list[AttachmentInfo]:
    """Read, validate, and store attachments for an entry.

    Args:
        entry_id: The entry ID to associate attachments with.
        file_paths: List of local file paths.
        repository: ARIEL repository for database storage.

    Returns:
        List of AttachmentInfo dicts for the entry's attachments JSONB.

    Raises:
        AttachmentValidationError: If any file fails validation.
    """
    # Pre-validate all files before storing anything
    files: list[tuple[bytes, str, str | None]] = []
    for path in file_paths:
        data, filename, mime_type = read_local_file(path)
        files.append((data, filename, mime_type))

    # Store all attachments
    attachment_infos: list[AttachmentInfo] = []
    for data, filename, mime_type in files:
        attachment_id = generate_attachment_id()
        await repository.store_attachment(
            entry_id=entry_id,
            attachment_id=attachment_id,
            filename=filename,
            mime_type=mime_type,
            data=data,
            size_bytes=len(data),
        )
        attachment_infos.append(
            {
                "url": f"/api/attachments/{attachment_id}",
                "type": mime_type,
                "filename": filename,
            }
        )

    return attachment_infos
