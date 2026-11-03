"""The stored subject line of a logbook entry.

Facilities name the subject differently: the ALS adapter stores
``metadata["subject"]``, the JLab, ORNL and generic adapters store
``metadata["title"]``, and an untitled entry has only its text. ARIEL derives
one subject per entry when it is written and stores it in the
``enhanced_entries.subject`` column, so every reader shows the same line.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["SUBJECT_MAX_CHARS", "derive_subject", "entry_subject"]

#: Longest subject taken from an entry's text.
SUBJECT_MAX_CHARS = 200

_METADATA_KEYS = ("subject", "title")


def derive_subject(entry: Mapping[str, Any]) -> str:
    """Return the subject line of ``entry``.

    The first non-empty value of ``metadata.subject``, then ``metadata.title``,
    then the first non-blank line of ``raw_text`` (stripped, at most
    :data:`SUBJECT_MAX_CHARS` characters), and ``""`` when there is none.

    Args:
        entry: An entry mapping with optional ``metadata`` and ``raw_text``.

    Returns:
        The subject, possibly empty.
    """
    metadata = entry.get("metadata")
    if isinstance(metadata, Mapping):
        for key in _METADATA_KEYS:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raw_text = entry.get("raw_text")
    if isinstance(raw_text, str):
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:SUBJECT_MAX_CHARS]
    return ""


def entry_subject(entry: Mapping[str, Any]) -> str:
    """The entry's stored subject, derived from the entry when none is stored.

    A store that predates the ``subject`` column (or an entry not yet
    backfilled) has no stored value; deriving it with :func:`derive_subject`
    keeps every reader showing the same line.
    """
    stored = entry.get("subject")
    if isinstance(stored, str) and stored.strip():
        return stored
    return derive_subject(entry)
