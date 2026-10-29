"""Attachment rows and logbook entries, as the ARIEL route suites build them.

:func:`viewable_row` is a stored, copied attachment with a JPEG rendition, under
:data:`ATT_ID`; :func:`attachment_entry` is an entry carrying JSONB attachment
items, and :func:`attachment_row` the ``attachment_files`` row for one of them.
"""

from __future__ import annotations

import io
from datetime import datetime

ATT_ID = "att-0123456789ab"
PNG_URL = "https://logbook.invalid/pic.png"


def jpeg_bytes() -> bytes:
    """A small JPEG, the kind of rendition a viewable attachment carries."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 6), (30, 200, 30)).save(buf, "JPEG")
    return buf.getvalue()


def viewable_row(**overrides) -> dict:
    """A copied attachment row with a JPEG rendition, *overrides* applied."""
    row = {
        "attachment_id": ATT_ID,
        "entry_id": "e-1",
        "filename": "beam.png",
        "mime_type": "image/png",
        "copy_status": "copied",
        "skip_reason": None,
        "rendition_mime": "image/jpeg",
        "rendition_sha256": "ab" * 32,
        "rendition_bytes": jpeg_bytes(),
    }
    row.update(overrides)
    return row


def attachment_entry(attachments, **extra):
    """An entry dict carrying the given JSONB attachment items."""
    entry = {
        "entry_id": "e-att",
        "source_system": "Test",
        "timestamp": datetime(2024, 1, 1, 12, 0, 0),
        "author": "op",
        "raw_text": "text",
        "attachments": attachments,
        "metadata": {},
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "updated_at": datetime(2024, 1, 1, 12, 0, 0),
        "summary": None,
        "keywords": [],
    }
    entry.update(extra)
    return entry


def attachment_row(entry_id, item, *, mime_type, viewable):
    """An ``attachment_files`` row for an item, copied, viewable or not."""
    from osprey.services.ariel_search.attachments import attachment_id_for

    return {
        "attachment_id": attachment_id_for(entry_id, item),
        "entry_id": entry_id,
        "filename": item.get("filename"),
        "mime_type": mime_type,
        "copy_status": "copied",
        "skip_reason": None,
        "source_url": item.get("url"),
        "rendition_sha256": "ab" * 32 if viewable else None,
    }
