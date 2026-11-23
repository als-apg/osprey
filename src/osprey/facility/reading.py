"""The checked read of a built ``facility.json``, shared by every framework reader.

A reader outside the panel engine reads the facility document through
:func:`load_facility`, or checks a document it already holds with
:func:`check_header`. Both compare the header against :data:`READS` (an exact
string compare) and refuse anything else with
:class:`~osprey.facility.errors.FacilityFileError`, naming the rebuild. The
wording is the panel engine's (``osprey.facility.panels.document``), so an
operator reads one sentence whichever reader met the file. Only an absent file
is an absence: :func:`load_facility` lets :class:`FileNotFoundError` through and
the caller decides what absence means.

It checks the header and that the document is an object, nothing more; the
build validates the document against the schema.

Stdlib only, besides the error it raises.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.errors import FacilityFileError

__all__ = ["READS", "REBUILD", "check_header", "load_facility"]

#: The facility-document headers this reader reads, compared as exact strings.
READS: frozenset[str] = frozenset({"osprey.facility.facility/4"})

#: The sentence every refusal ends with; a framework reader that words its own
#: refusal of the facility document ends it with this too.
REBUILD = "-- rebuild with osprey build"

_JSON_TYPE_WORDS: Mapping[type, str] = {
    dict: "an object",
    list: "a list",
    str: "a string",
    int: "a number",
    float: "a number",
    bool: "a boolean",
    type(None): "null",
}


def _type_word(value: Any) -> str:
    return _JSON_TYPE_WORDS.get(type(value), type(value).__name__)


def _refuse(source: str, problem: str) -> FacilityFileError:
    return FacilityFileError(f"{source} {problem} {REBUILD}")


def _expected() -> str:
    return " or ".join(sorted(READS))


def check_header(document: object, source: str) -> None:
    """Refuse a document that is not an object or whose header is not in :data:`READS`.

    Args:
        document: The parsed document.
        source: What to name the document in a refusal, normally its path.

    Raises:
        FacilityFileError: The document is not a JSON object, carries no
            ``schema`` header, or carries one outside :data:`READS`.
    """
    if not isinstance(document, Mapping):
        raise _refuse(source, f"is {_type_word(document)}, not a JSON object")
    if "schema" not in document:
        found = "a document without a schema header"
    else:
        header = document["schema"]
        if isinstance(header, str) and header in READS:
            return
        plain = isinstance(header, str) and header and header == header.strip()
        found = header if plain else repr(header)
    raise FacilityFileError(f"{source} is {found}; this reader reads {_expected()} {REBUILD}")


def load_facility(path: Path | str) -> dict[str, Any]:
    """Read and check the facility document at ``path``.

    Args:
        path: The ``facility.json`` file.

    Returns:
        The document, a JSON object whose header is in :data:`READS`.

    Raises:
        FileNotFoundError: No file is there.
        FacilityFileError: The file cannot be read, is not JSON, is not a JSON
            object, or carries a header outside :data:`READS`.
    """
    path = Path(path)
    source = str(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise _refuse(source, f"cannot be read: {exc}") from exc
    try:
        document: dict[str, Any] = json.loads(text)
    except ValueError as exc:
        raise _refuse(source, f"is not JSON: {exc}") from exc
    check_header(document, source)
    return document
