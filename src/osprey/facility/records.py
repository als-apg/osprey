"""The one read model of a built facility file: its records by kind and id.

A :class:`FacilityRecords` is built once over a finished document: the build's
return value, the last stage's ``validated.document``, or a render's
``facility.json``. It is never kept across build stages, which replace parts of
the document. It indexes a finished document only; a reader that must refuse an
unfinished file checks the raw lists through :attr:`FacilityRecords.document`.

It holds the document's own record mappings, each behind one read-only proxy
built at index time, and copies nothing: two lookups of one record return the
same proxy. Nested values are the file's own objects, and no reader writes them.
Every id is indexed as its string. A record without its id slot is skipped, and
a duplicate id within one kind is won by the first record in document order.

Every answer but one is a lookup by a stated id, and every multi-record answer
has a stated order. Lookup by id serves five kinds; resolution answers channels
and devices only:

- :meth:`FacilityRecords.find` lists records in ``RECORD_KINDS`` order: place,
  device, channel, group, model.
- :meth:`FacilityRecords.resolve` and :meth:`FacilityRecords.describe` list them
  in ``LINK_KINDS`` order: channel, then device.

``find``, ``resolve``, ``targets`` and ``describe`` match exact ids only, never a
name or a label. :meth:`FacilityRecords.lookup` is the one name-aware entry: it
finds places and devices by id, then by a ``names`` entry or a
``scheme_names[].name``, then (places only) by a flat last id segment, each
record once at its first match.

:meth:`FacilityRecords.resolve_link` is the one link resolver: a knowledge-page
link target resolves as :meth:`~FacilityRecords.resolve` does, and a target
``place/<id>`` that resolves to nothing there names the place ``<id>``. The
place form stays outside ``LINK_KINDS`` and :meth:`~FacilityRecords.targets`.

A finished document holds these invariants, each enforced by a build stage:

- I1, ids are unique per kind. The ``load`` stage stops on a duplicate within a
  layer (``layer-duplicate``, ``sources.py`` ``_check_duplicates``, which keeps
  the first record) and the ``combine`` stage merges one id across layers. So
  :meth:`~FacilityRecords.record` and :meth:`~FacilityRecords.require` are
  single-valued, while one string may name records of several kinds.
- I2, an address is wired at most once. The ``compute`` stage stops on a second
  wiring record (``wiring-conflict``).
- I3, every record carries its id slot and every id a record names exists. The
  ``schema`` and ``references`` stages enforce it, so a lookup by a stated id
  never needs a spelling fallback.
- I4, a channel's devices are its ``on.device`` and its ``endpoint_of``, each
  once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

from osprey.facility.places import resolve_place_name
from osprey.utils.facility import FacilityFileError

__all__ = [
    "LINK_KINDS",
    "RECORD_KINDS",
    "ChannelSelection",
    "FacilityRecord",
    "FacilityRecords",
    "LinkKind",
    "PLACE_LINK_PREFIX",
    "RecordKind",
    "group_kind",
    "describe_records",
    "is_family",
]

#: The kinds of record the read model serves.
RecordKind = Literal["place", "device", "channel", "group", "model"]

#: Each record kind with the facility-file list that holds it and its id slot.
#: This order is the order of :meth:`FacilityRecords.find`.
RECORD_KINDS: tuple[tuple[RecordKind, str, str], ...] = (
    ("place", "places", "id"),
    ("device", "devices", "id"),
    ("channel", "channels", "id"),
    ("group", "groups", "id"),
    ("model", "models", "name"),
)

#: The kind of a group that states none.
DEFAULT_GROUP_KIND = "family"


def group_kind(group: Mapping[str, Any]) -> str:
    """A group's kind: the one it states, else ``family``."""
    return str(group.get("kind") or DEFAULT_GROUP_KIND)


def is_family(group: Mapping[str, Any]) -> bool:
    """Whether a group is a family, the only kind of group a view treats as one.

    A family describes its members and files as a Family in the middle-layer
    index; a group of another kind (a ``section``) does neither.
    """
    return group_kind(group) == DEFAULT_GROUP_KIND


#: The kinds of record a reference may name.
LinkKind = Literal["channel", "device"]

#: The link kinds, in the order of :meth:`FacilityRecords.resolve`.
LINK_KINDS: tuple[LinkKind, ...] = get_args(LinkKind)

#: The target prefix of a knowledge-page link that names a place,
#: ``facility:place/<id>``; see :meth:`FacilityRecords.resolve_link`.
PLACE_LINK_PREFIX = "place/"

#: How a refusal words the JSON type a facility file holds instead of an object.
_JSON_TYPE_WORDS: Mapping[type, str] = {
    list: "a list",
    str: "a string",
    int: "a number",
    float: "a number",
    bool: "a boolean",
    type(None): "null",
}


@dataclass(frozen=True)
class FacilityRecord:
    """One record of the facility file with its kind.

    Attributes:
        kind: ``place``, ``device``, ``channel``, ``group`` or ``model``.
        id: The record's id, the string its id slot holds.
        record: The record as the facility file holds it, read-only.

    Raises:
        ValueError: ``kind`` is none of the five.
    """

    kind: RecordKind
    id: str
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in get_args(RecordKind):
            raise ValueError(f"a facility record's kind is one of five, not {self.kind!r}")


@dataclass(frozen=True)
class ChannelSelection:
    """A named selection over the channels of a facility file; its facets AND.

    Attributes:
        model: A model's name.
        place: A place's id; the selection takes that place and every place under it.
        group: A group's id.
        signal: The ``signal`` a channel states.
        engine_role: The ``engine_role`` of the wiring record naming a channel's address.
        plane: The ``plane`` of the wiring record naming a channel's address.

    Raises:
        ValueError: The selection names neither a model nor a place, a value is
            not a string, or an engine_role or plane is outside the facility schema's
            vocabulary.
    """

    model: str | None = None
    place: str | None = None
    group: str | None = None
    signal: str | None = None
    engine_role: str | None = None
    plane: str | None = None

    def __post_init__(self) -> None:
        for value in (
            self.model,
            self.place,
            self.group,
            self.signal,
            self.engine_role,
            self.plane,
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"a selection's values are strings, not {value!r}")
        if self.model is None and self.place is None:
            raise ValueError("a selection names a model or a place")
        if self.engine_role is not None or self.plane is not None:
            self._check_wiring_vocabulary()

    def _check_wiring_vocabulary(self) -> None:
        """Refuse an ``engine_role`` or ``plane`` the facility schema does not state."""
        from osprey.facility.schema import core

        if self.engine_role is not None and self.engine_role not in {
            m.value for m in core.EngineRoleEnum
        }:
            raise ValueError(
                f"a selection's engine_role is an engine role, not {self.engine_role!r}"
            )
        if self.plane is not None and self.plane not in {m.value for m in core.PlaneEnum}:
            raise ValueError(f"a selection's plane is a wiring plane, not {self.plane!r}")


class FacilityRecords:
    """The records of one finished facility document, by kind and id.

    An absent or non-list kind has no records; the constructor never refuses.

    Args:
        document: The facility file.
    """

    def __init__(self, document: Mapping[str, Any]) -> None:
        self._source = document
        self._document = MappingProxyType(document)
        self._by_kind: dict[RecordKind, dict[str, Mapping[str, Any]]] = {}
        for kind, key, slot in RECORD_KINDS:
            rows = document.get(key)
            indexed: dict[str, Mapping[str, Any]] = {}
            for record in rows if isinstance(rows, list) else ():
                if isinstance(record, Mapping) and record.get(slot) is not None:
                    indexed.setdefault(str(record[slot]), MappingProxyType(record))
            self._by_kind[kind] = indexed
        channels_on: dict[str, list[Mapping[str, Any]]] = {}
        for channel in self._by_kind["channel"].values():
            on_device = (channel.get("on") or {}).get("device")
            for device in dict.fromkeys((on_device, *(channel.get("endpoint_of") or ()))):
                if device is not None:
                    channels_on.setdefault(str(device), []).append(channel)
        self._channels_on = {device: tuple(held) for device, held in channels_on.items()}
        self._targets = frozenset(id_ for kind in LINK_KINDS for id_ in self._by_kind[kind])
        self._wiring: dict[str, tuple[str, Mapping[str, Any]]] = {}
        for name, model in self._by_kind["model"].items():
            wiring = model.get("wiring")
            for entry in wiring if isinstance(wiring, list) else ():
                if isinstance(entry, Mapping) and entry.get("address") is not None:
                    self._wiring.setdefault(str(entry["address"]), (name, entry))
        self._by_name: dict[str, set[tuple[RecordKind, str]]] = {}
        for kind in _LOOKUP_KINDS:
            for id_, record in self._by_kind[kind].items():
                for name in _record_names(record):
                    self._by_name.setdefault(name, set()).add((kind, id_))

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
            raise FacilityFileError(
                f"The facility file {path} is not a JSON object: "
                f"it holds {_JSON_TYPE_WORDS[type(document)]}"
            )
        return cls(document)

    @property
    def document(self) -> Mapping[str, Any]:
        """The mapping this read model was built from, read-only."""
        return self._document

    def facility_bytes(self) -> bytes:
        """The bytes of the facility file this read model was built from.

        Serialised by :func:`osprey.facility.render.facility_bytes` from the
        document handed in, not a copy of it, so a build's renders, which each
        build their own read model over the one document, share its one
        serialisation.
        """
        from osprey.facility.render import facility_bytes

        source = self._source
        return facility_bytes(source if isinstance(source, dict) else dict(source))

    def record(self, kind: RecordKind, id: str) -> Mapping[str, Any] | None:
        """The one record of ``kind`` with ``id``, or ``None``."""
        return self._by_kind[kind].get(id)

    def require(self, kind: RecordKind, id: str) -> Mapping[str, Any]:
        """The one record of ``kind`` with ``id``, for a caller that needs it.

        Raises:
            KeyError: The file holds no such record; the message names both.
        """
        found = self.record(kind, id)
        if found is None:
            raise KeyError(f"the facility file holds no {kind} {id!r}")
        return found

    def of_kind(self, kind: RecordKind) -> tuple[Mapping[str, Any], ...]:
        """Every indexed record of ``kind``, in document order."""
        return tuple(self._by_kind[kind].values())

    def ids(self, kind: RecordKind) -> frozenset[str]:
        """The ids of ``kind``."""
        return frozenset(self._by_kind[kind])

    def find(self, id: str) -> tuple[FacilityRecord, ...]:
        """Every record, of any kind, with ``id``, in ``RECORD_KINDS`` order."""
        return tuple(
            FacilityRecord(kind, id, self._by_kind[kind][id])
            for kind, _key, _slot in RECORD_KINDS
            if id in self._by_kind[kind]
        )

    def lookup(self, text: str) -> tuple[tuple[str, FacilityRecord], ...]:
        """The places and devices ``text`` names, each with how it matched.

        Matches come in three groups: ``"id"`` (the record's id is ``text``),
        ``"name"`` (a ``names`` entry or a ``scheme_names[].name`` is ``text``,
        exact string), then ``"segment"`` (places only: the flat-name rule of
        :func:`~osprey.facility.places.resolve_place_name`). Within a group,
        places come before devices, each kind in id order. A record appears
        once, at its first match. :meth:`find`, :meth:`resolve`,
        :meth:`targets` and :meth:`describe` stay exact-id.

        Args:
            text: An id, a name, or a flat place name.

        Returns:
            ``(match, record)`` entries in that order; empty when nothing matches.
        """
        rank = {kind: index for index, (kind, _key, _slot) in enumerate(RECORD_KINDS)}
        hits: list[tuple[str, RecordKind, str]] = [
            ("id", kind, text) for kind in _LOOKUP_KINDS if text in self._by_kind[kind]
        ]
        named = sorted(self._by_name.get(text, ()), key=lambda hit: (rank[hit[0]], hit[1]))
        hits += [("name", kind, id_) for kind, id_ in named]
        segment = resolve_place_name(text, self._by_kind["place"])
        segments = [segment] if isinstance(segment, str) else segment or []
        hits += [("segment", "place", id_) for id_ in segments]
        found: list[tuple[str, FacilityRecord]] = []
        seen: set[tuple[RecordKind, str]] = set()
        for match, kind, id_ in hits:
            if (kind, id_) not in seen:
                seen.add((kind, id_))
                found.append((match, FacilityRecord(kind, id_, self._by_kind[kind][id_])))
        return tuple(found)

    def channels_on(self, device_id: str) -> tuple[Mapping[str, Any], ...]:
        """The channels on a device: its ``on.device`` and ``endpoint_of`` channels.

        Args:
            device_id: A device id.

        Returns:
            Each channel once, in document order; empty for a device with none
            and for an unknown id.
        """
        return self._channels_on.get(device_id, ())

    def wired(self, address: str) -> Mapping[str, Any] | None:
        """The one wiring record naming an address.

        An address is wired at most once: the ``compute`` stage stops on a second
        wiring record (``compute.py`` ``_addresses_wired_twice``, ``wiring-conflict``).

        Args:
            address: A channel's address.

        Returns:
            The wiring record as the facility file holds it; ``None`` for an
            address no model wires.
        """
        held = self._wiring.get(address)
        return None if held is None else held[1]

    def select_channels(self, selection: ChannelSelection) -> tuple[Mapping[str, Any], ...]:
        """The channel records a selection matches, in document order.

        ``model`` takes a channel a wiring record of that model names, or whose
        ``on.device`` has that computed ``model``. ``place`` takes a channel whose
        place (its device's ``place``, else its own ``on.place``) is the named
        place or lies under it in the ``/``-separated tree of place ids. ``group``
        takes a channel whose ``on.device`` is a member. ``signal`` is equality.
        ``engine_role`` and ``plane`` match the wiring record naming the channel's
        address, never the channel's own ``role`` slot: with ``model`` set, that
        model's record; otherwise the one record any model holds. An unwired address
        matches neither.

        Args:
            selection: The facets, which AND.

        Returns:
            The matching channels; empty when nothing matches.

        Raises:
            ValueError: A named model, place or group is no record of the file.
        """
        wired: frozenset[str] = frozenset()
        members: frozenset[str] = frozenset()
        named_records: tuple[tuple[RecordKind, str | None], ...] = (
            ("model", selection.model),
            ("place", selection.place),
            ("group", selection.group),
        )
        for kind, named in named_records:
            if named is not None and named not in self._by_kind[kind]:
                raise ValueError(f"the facility file holds no {kind} {named!r}")
        if selection.model is not None:
            wiring = self._by_kind["model"][selection.model].get("wiring") or ()
            wired = frozenset(str(entry["address"]) for entry in wiring)
        if selection.group is not None:
            group = self._by_kind["group"][selection.group]
            members = frozenset(str(member) for member in group.get("members") or ())

        def matches(channel: Mapping[str, Any]) -> bool:
            on = channel.get("on") or {}
            on_device = on.get("device")
            device = self._by_kind["device"].get(str(on_device), {}) if on_device else {}
            if selection.model is not None and not (
                str(channel["id"]) in wired or device.get("model") == selection.model
            ):
                return False
            if selection.place is not None:
                place = device.get("place") or on.get("place")
                if place is None or not _is_at_or_under(str(place), selection.place):
                    return False
            if selection.group is not None and str(on_device) not in members:
                return False
            if selection.signal is not None and channel.get("signal") != selection.signal:
                return False
            if selection.engine_role is None and selection.plane is None:
                return True
            held = self._wiring.get(str(channel["id"]))
            if held is None or selection.model not in (None, held[0]):
                return False
            entry = held[1]
            return selection.engine_role in (None, entry.get("engine_role")) and (
                selection.plane in (None, entry.get("plane"))
            )

        return tuple(channel for channel in self._by_kind["channel"].values() if matches(channel))

    def resolve(self, target: str) -> tuple[FacilityRecord, ...]:
        """The records a reference target names: the channel first, then the device.

        Only a channel id or a device id resolves, by exact id; never a name, a
        label, a former address, a place, a group or a model.

        Args:
            target: A channel id or a device id.

        Returns:
            The matching records, in ``LINK_KINDS`` order; empty when dangling.
        """
        return tuple(
            FacilityRecord(kind, target, self._by_kind[kind][target])
            for kind in LINK_KINDS
            if target in self._by_kind[kind]
        )

    def resolve_link(self, target: str) -> tuple[FacilityRecord, ...]:
        """The records a knowledge-page link target names.

        The exact :meth:`resolve` answer wins, so a channel or device whose id is
        literally ``place/<id>`` keeps its link. Otherwise a target
        ``place/<id>`` names the place ``<id>``.

        Args:
            target: A link's target text, verbatim.

        Returns:
            :meth:`resolve`'s records when there are any; else the one place a
            ``place/<id>`` target names; else empty.
        """
        found = self.resolve(target)
        if found or not target.startswith(PLACE_LINK_PREFIX):
            return found
        place_id = target[len(PLACE_LINK_PREFIX) :]
        record = self._by_kind["place"].get(place_id)
        return () if record is None else (FacilityRecord("place", place_id, record),)

    def targets(self) -> frozenset[str]:
        """Every string :meth:`resolve` resolves to at least one record."""
        return self._targets

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
        return describe_records(self.resolve(target))


def describe_records(found: Iterable[FacilityRecord]) -> list[dict[str, Any]]:
    """What a reader shows of each record: ``{kind, id, unit?, description?}``.

    Args:
        found: Records, as :meth:`FacilityRecords.resolve` or
            :meth:`FacilityRecords.resolve_link` returned them.

    Returns:
        One entry per record, in the order given; ``unit`` and ``description``
        appear only when the record states them as a non-empty string.
    """
    described: list[dict[str, Any]] = []
    for record in found:
        entry: dict[str, Any] = {"kind": record.kind, "id": record.id}
        for key in ("unit", "description"):
            value = record.record.get(key)
            if isinstance(value, str) and value:
                entry[key] = value
        described.append(entry)
    return described


_LOOKUP_KINDS: tuple[RecordKind, ...] = ("place", "device")


def _record_names(record: Mapping[str, Any]) -> set[str]:
    """A record's ``names`` entries and ``scheme_names[].name`` values, as strings."""
    names = {str(name) for name in record.get("names") or () if name is not None}
    for alias in record.get("scheme_names") or ():
        if isinstance(alias, Mapping) and alias.get("name") is not None:
            names.add(str(alias["name"]))
    return names


def _is_at_or_under(place: str, named: str) -> bool:
    """Whether ``place`` is the ``named`` place or lies under it in the tree of place ids."""
    return place == named or place.startswith(f"{named}/")
