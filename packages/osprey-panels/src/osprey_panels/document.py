"""The facility document as the panel engine reads it.

The engine reads the built ``facility.json`` and nothing else of the description:
this module owns the document shape the engine depends on. It checks the header
against :data:`READS` (an exact string compare) and the shape of every table the
engine reads, and refuses anything else with :class:`FacilityDocumentError`. It
does not validate the document against the schema; the build does that.

A further address of a wiring record (under ``also``) reads every record fact
from the channel of its record's ``address``: the description states that list
(``osprey_facility.RECORD_FACTS``) and :data:`RECORD_FACTS` mirrors it.

Stdlib only: the panels carry no dependency on the description code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "END_KINDS",
    "FACILITY_FILE",
    "READS",
    "RECORD_FACTS",
    "FacilityDocument",
    "FacilityDocumentError",
    "end_identity",
]

#: The facility-document headers this reader reads, compared as exact strings.
READS: frozenset[str] = frozenset({"osprey.facility.facility/4"})

#: The name of the built facility document.
FACILITY_FILE = "facility.json"

#: The channel slots a further address of a wiring record reads from its record.
RECORD_FACTS: frozenset[str] = frozenset(
    {
        "attributes",
        "command",
        "idle",
        "options",
        "pair",
        "precision",
        "role",
        "shape",
        "signal",
        "simulation",
        "started_by",
        "stopped_by",
        "tags",
        "timeout_s",
        "tolerance",
        "unit",
        "value_type",
    }
)

#: The kinds one relation end may name; an end names exactly one.
END_KINDS: tuple[str, ...] = ("device", "place", "group")

#: Each record kind with the document table that holds it and its id key.
_RECORD_KINDS: tuple[tuple[str, str, str], ...] = (
    ("place", "places", "id"),
    ("device", "devices", "id"),
    ("channel", "channels", "id"),
    ("group", "groups", "id"),
    ("model", "models", "name"),
)

#: Tables that must be present, with each row's key.
_REQUIRED_TABLES: tuple[tuple[str, str], ...] = (("signals", "name"), ("relation_types", "name"))

#: Record tables whose rows are checked (every one but ``models``); an absent one
#: reads as empty.
_CHECKED_RECORD_TABLES: tuple[tuple[str, str], ...] = tuple(
    (table, key) for kind, table, key in _RECORD_KINDS if kind != "model"
)

_REBUILD = "-- rebuild with osprey build"

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


class FacilityDocumentError(ValueError):
    """The facility document is not one this reader reads."""


def _refuse(source: str, problem: str) -> FacilityDocumentError:
    return FacilityDocumentError(f"{source} {problem} {_REBUILD}")


def _expected() -> str:
    return " or ".join(sorted(READS))


def _check_header(document: Mapping[str, Any], source: str) -> None:
    if "schema" not in document:
        found = "a document without a schema header"
    else:
        header = document["schema"]
        if isinstance(header, str) and header in READS:
            return
        plain = isinstance(header, str) and header and header == header.strip()
        found = header if plain else repr(header)
    raise FacilityDocumentError(f"{source} is {found}; this reader reads {_expected()} {_REBUILD}")


def _check_rows(document: Mapping[str, Any], table: str, key: str, source: str) -> None:
    rows = document[table]
    if not isinstance(rows, list):
        raise _refuse(source, f"has table {table!r} as {_type_word(rows)}, not a list")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _refuse(
                source, f"has row {index} of table {table!r} as {_type_word(row)}, not an object"
            )
        if not isinstance(row.get(key), str):
            raise _refuse(
                source, f"has row {index} of table {table!r} without a string {key!r} key"
            )


class FacilityDocument:
    """A checked facility document, indexed by record kind.

    Args:
        document: The facility document.
        source: What to name the document in a refusal, normally its path.

    Raises:
        FacilityDocumentError: The header is not in :data:`READS`, ``signals`` or
            ``relation_types`` is missing, or a table the engine reads is not a
            list of objects with a string ``name``/``id`` key.
    """

    def __init__(self, document: Mapping[str, Any], *, source: str = "<document>") -> None:
        if not isinstance(document, Mapping):
            raise _refuse(source, f"is {_type_word(document)}, not a JSON object")
        _check_header(document, source)
        for table, key in _REQUIRED_TABLES:
            if table not in document:
                raise _refuse(source, f"has no {table!r} table")
            _check_rows(document, table, key, source)
        for table, key in _CHECKED_RECORD_TABLES:
            if table in document:
                _check_rows(document, table, key, source)
        self._document = MappingProxyType(document)
        self._by_kind: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for kind, table, key in _RECORD_KINDS:
            rows = document.get(table)
            indexed: dict[str, Mapping[str, Any]] = {}
            for row in rows if isinstance(rows, list) else ():
                if isinstance(row, Mapping) and row.get(key) is not None:
                    indexed.setdefault(str(row[key]), MappingProxyType(row))
            self._by_kind[kind] = tuple(indexed.values())
            if kind == "channel":
                self._channels = indexed
        self._record_of: dict[str, str] = {}
        for model in self._by_kind["model"]:
            wiring = model.get("wiring")
            for entry in wiring if isinstance(wiring, list) else ():
                address = entry.get("address") if isinstance(entry, Mapping) else None
                if not isinstance(address, str):
                    continue
                also = entry.get("also")
                self._record_of.setdefault(address, address)
                for name in also if isinstance(also, list) else ():
                    if isinstance(name, str):
                        self._record_of.setdefault(name, address)
        self._signals = tuple(MappingProxyType(row) for row in document["signals"])
        self._relation_types = tuple(MappingProxyType(row) for row in document["relation_types"])

    @classmethod
    def from_file(cls, path: Path) -> FacilityDocument | None:
        """Read the facility document at ``path``.

        Returns:
            The document, or ``None`` when no file is there.

        Raises:
            FacilityDocumentError: The file cannot be read, is not JSON, is not a
                JSON object, or fails a check of the constructor.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            raise _refuse(str(path), f"cannot be read: {exc}") from exc
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise _refuse(str(path), f"is not JSON: {exc}") from exc
        return cls(document, source=str(path))

    @property
    def document(self) -> Mapping[str, Any]:
        """The document, read-only."""
        return self._document

    def of_kind(self, kind: str) -> tuple[Mapping[str, Any], ...]:
        """The rows of one record kind: the first per id key, in document order.

        Raises:
            KeyError: ``kind`` is not a record kind.
        """
        return self._by_kind[kind]

    def record_address(self, address: str) -> str:
        """The address of the wiring record naming ``address`` under ``also``; ``address`` itself otherwise."""
        return self._record_of.get(address, address)

    def channel(self, address: str) -> Mapping[str, Any] | None:
        """The channel at ``address``, read as its wiring record states it.

        A further address keeps every slot but :data:`RECORD_FACTS` from its
        own row and reads those from the channel of its record's ``address``.

        Returns:
            The channel as read, read-only; the channel's own row for any
            address no record names under ``also``; ``None`` when no channel
            has that address.
        """
        row = self._channels.get(address)
        own = self.record_address(address)
        held = self._channels.get(own)
        if row is None or own == address or held is None:
            return row
        read = {key: value for key, value in row.items() if key not in RECORD_FACTS}
        read.update((key, value) for key, value in held.items() if key in RECORD_FACTS)
        return MappingProxyType(read)

    def signal_rows(self) -> tuple[Mapping[str, Any], ...]:
        """The rows of the ``signals`` table, in document order."""
        return self._signals

    def relation_types(self) -> tuple[Mapping[str, Any], ...]:
        """The rows of the ``relation_types`` table, in document order."""
        return self._relation_types


def end_identity(end: Mapping[str, Any]) -> tuple[str, str] | None:
    """The ``(kind, id)`` a relation end names, or ``None`` when it names no kind.

    An end naming more than one kind reads as its first in :data:`END_KINDS` order.
    """
    for kind in END_KINDS:
        if end.get(kind) is not None:
            return kind, str(end[kind])
    return None
