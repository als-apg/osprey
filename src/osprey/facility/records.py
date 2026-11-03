"""The records of a facility file, looked up by the id a reference names.

A reference names a channel by its channel id (its address) or a device by its
device id. Nothing else resolves: not a name, a label or a place. One string
may be both a channel id and a device id, since ids are unique per kind; it
then resolves to both records, the channel first.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from osprey.utils.facility import FacilityFileError

__all__ = ["FacilityRecord", "FacilityRecords", "RecordKind"]

#: The kinds of record a reference resolves to.
RecordKind = Literal["channel", "device"]

#: Each kind with the facility-file list that holds its records, in lookup order.
_LISTS: tuple[tuple[RecordKind, str], ...] = (("channel", "channels"), ("device", "devices"))


@dataclass(frozen=True)
class FacilityRecord:
    """One record of the facility file that a reference resolves to.

    Attributes:
        kind: ``channel`` or ``device``.
        id: The record's id, the string the reference names.
        record: The record as the facility file holds it.

    Raises:
        ValueError: ``kind`` is neither ``channel`` nor ``device``.
    """

    kind: RecordKind
    id: str
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in get_args(RecordKind):
            raise ValueError(f"a facility record is a channel or a device, not a {self.kind!r}")


class FacilityRecords:
    """The channel and device records of one facility file, keyed by id.

    Args:
        document: The facility file, with its ``channels`` and ``devices`` lists.
    """

    def __init__(self, document: Mapping[str, Any]) -> None:
        index: dict[str, list[FacilityRecord]] = {}
        for kind, key in _LISTS:
            for record in document.get(key) or ():
                if isinstance(record, Mapping) and isinstance(record.get("id"), str):
                    index.setdefault(record["id"], []).append(
                        FacilityRecord(kind, record["id"], record)
                    )
        self._index = {target: tuple(records) for target, records in index.items()}

    @classmethod
    def from_file(cls, path: Path) -> FacilityRecords | None:
        """Read the records of the facility file at ``path``.

        Args:
            path: The facility file.

        Returns:
            The records, or ``None`` when no file is there.

        Raises:
            FacilityFileError: The file is there but cannot be read, is not
                JSON, or is not a JSON object.
        """
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            raise FacilityFileError(f"The facility file {path} cannot be read: {exc}") from exc
        try:
            document = json.loads(text)
        except ValueError as exc:
            raise FacilityFileError(f"The facility file {path} is not JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise FacilityFileError(f"The facility file {path} is not a JSON object")
        return cls(document)

    def resolve(self, target: str) -> tuple[FacilityRecord, ...]:
        """The records whose id is ``target``: the channel first, then the device.

        Args:
            target: A channel id or a device id.

        Returns:
            The matching records; empty when the file holds neither.
        """
        return self._index.get(target, ())

    def targets(self) -> frozenset[str]:
        """Every string that resolves to at least one record."""
        return frozenset(self._index)

    def describe(self, target: str) -> list[dict[str, Any]]:
        """What a reader shows of each record ``target`` resolves to.

        Args:
            target: A channel id or a device id.

        Returns:
            One ``{kind, id, unit?, description?}`` per record :meth:`resolve`
            returns, in that order; ``unit`` and ``description`` appear only
            when the record states them. Empty for a target that resolves to
            nothing.
        """
        described: list[dict[str, Any]] = []
        for found in self.resolve(target):
            entry: dict[str, Any] = {"kind": found.kind, "id": found.id}
            for key in ("unit", "description"):
                value = found.record.get(key)
                if isinstance(value, str) and value:
                    entry[key] = value
            described.append(entry)
        return described
