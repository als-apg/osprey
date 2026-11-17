"""The panel graph: a built facility file read with its rule tables.

:func:`load_graph` reads the facility file through
:class:`~osprey.facility.records.FacilityRecords` and the rule tables through
:func:`~osprey.facility.panels.tables.read_tables`, then builds every index
once. :class:`PanelGraph` is read-only; each accessor returns fresh lists.

A thing is a device, a place or a group of the facility file, the three kinds
a relation end may name (:data:`~osprey.facility.relations.END_KINDS`). One id
can name records of two kinds; an accessor given only an id takes the first
kind in that order (device, place, group), and ``kind=`` picks another.

What the graph leaves out, everywhere:

- a thing whose id ``excluded.csv`` lists (``kind`` ``thing``), every channel
  ``on`` it, every relation with it at an end, its group memberships, its
  poses; an excluded place hides every place within it (by id, transitively)
  and every device placed in any of them, with all of the above;
- a channel ``excluded.csv`` lists (``kind`` ``channel``) and a channel whose
  name starts with ``VA:`` (simulation only), with their record types, pairs
  and health rules; a pair whose readback is left out is left out;
- a pair ``pairs.csv`` does not state: the table is the one source of pairs;
- what a document ``excluded.csv`` lists (``kind`` ``document``) is the
  source of: a citation, a relation claim, a pose;
- a ``superseded`` relation claim. ``accepted`` and ``contested`` claims are
  edges, their standing on the edge. A relation kind the framework does not
  know is an edge under its own key like any other.

The ring frame is the frame of ``positions.csv`` that carries the most rows
with an empty ``derived_assumes`` (on a
tie, the rows name no ring frame: :func:`load_graph` refuses and asks for
one), unless :func:`load_graph` is told which.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from osprey.facility import FACILITY_FILE
from osprey.facility.panels.tables import (
    POSITIONS,
    TABLE_FILES,
    DrawnAs,
    HealthRule,
    Pair,
    PanelTableError,
    PanelTables,
    PicturePart,
    Pose,
    RecordType,
    read_tables,
)
from osprey.facility.records import FacilityRecords
from osprey.facility.relations import END_KINDS, end_identity

__all__ = [
    "SIMULATION_PREFIX",
    "Channel",
    "Citation",
    "Edge",
    "PanelGraph",
    "Thing",
    "ThingKind",
    "load_graph",
]

#: The prefix of a channel only the simulation serves; no screen shows one.
SIMULATION_PREFIX = "VA:"

ThingKind = Literal["device", "place", "group"]

#: ``(kind, id)``: one thing.
_Key = tuple[str, str]


@dataclass(frozen=True)
class Citation:
    """A document a fact is read from."""

    document: str
    version: str | None = None
    state: str | None = None
    locator: str | None = None


@dataclass(frozen=True)
class Thing:
    """A device, place or group.

    Attributes:
        id: The record's id.
        kind: ``device``, ``place`` or ``group``.
        cls: A device's class; ``None`` for a place, a group or a device with none.
        place: A device's place; a place's parent (its id without the last
            ``/`` segment, when that place exists); ``None`` for a group.
        citations: The documents the record cites.
        level: A place's ``level`` (the facility's word for its depth of the
            tree, such as ``rack``); ``None`` for a device, a group or a place
            with none.
    """

    id: str
    kind: ThingKind
    cls: str | None
    place: str | None
    citations: tuple[Citation, ...] = ()
    level: str | None = None


@dataclass(frozen=True)
class Channel:
    """A channel of the facility file as a screen binds it.

    Attributes:
        name: The channel's address.
        drawn_as: How ``record_types.csv`` says to draw it; ``None`` with no row.
        writable: Whether the facility file's role is ``setpoint``.
        alias: Another name the facility file states for it (the first of its
            ``names``, sorted), or ``None``.
        thing: The id of the thing it is ``on``.
        thing_kind: That thing's kind.
        record_type: The record type ``record_types.csv`` states, or ``None``.
        unit: The facility file's unit, or ``None``.
        description: The facility file's description, or ``None``.
        tolerance: The facility file's ``tolerance`` as ``(term, amount)``,
            ``term`` being ``absolute`` or ``relative``; ``None`` with none.
    """

    name: str
    drawn_as: DrawnAs | None
    writable: bool
    alias: str | None
    thing: str
    thing_kind: ThingKind
    record_type: str | None = None
    unit: str | None = None
    description: str | None = None
    tolerance: tuple[str, float] | None = None


@dataclass(frozen=True)
class Edge:
    """One relation claim seen from one of its ends.

    Attributes:
        kind: The claim's relation type.
        claim: The claim's id.
        outgoing: Whether this end is the claim's subject.
        other: The id of the other end.
        other_kind: The other end's kind.
        standing: ``accepted`` or ``contested``.
        basis: ``stated`` or ``inferred``.
        port: This end's port, or ``None``.
        other_port: The other end's port, or ``None``.
        unit: The rack unit either end names, or ``None``.
        attributes: The claim's attributes, read-only.
        source: The document the claim cites, or ``None``.
    """

    kind: str
    claim: str
    outgoing: bool
    other: str
    other_kind: ThingKind
    standing: str
    basis: str
    port: str | None = None
    other_port: str | None = None
    unit: int | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict, hash=False)
    source: Citation | None = None


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _citation(source: Any) -> Citation | None:
    if not isinstance(source, Mapping) or source.get("document") is None:
        return None
    return Citation(
        document=str(source["document"]),
        version=_text(source.get("version")),
        state=_text(source.get("state")),
        locator=_text(source.get("locator")),
    )


def _tolerance(record: Any) -> tuple[str, float] | None:
    """The one ``absolute`` or ``relative`` term a built tolerance states, or ``None``."""
    if not isinstance(record, Mapping):
        return None
    for term in ("absolute", "relative"):
        amount = record.get(term)
        if isinstance(amount, int | float) and not isinstance(amount, bool):
            return (term, float(amount))
    return None


def _ring_frame(poses: list[Pose]) -> str | None:
    """The frame with the most rows that assume nothing, or ``None`` with no such row.

    Raises:
        PanelTableError: Two frames tie for the most.
    """
    ranked = Counter(pose.frame for pose in poses if not pose.derived_assumes).most_common()
    if not ranked:
        return None
    tied = [frame for frame, count in ranked if count == ranked[0][1]]
    if len(tied) > 1:
        raise PanelTableError(
            POSITIONS,
            None,
            f"frames {', '.join(sorted(tied))} tie for the ring frame with {ranked[0][1]} "
            "rows each; pass the ring frame to load_graph",
        )
    return ranked[0][0]


class PanelGraph:
    """The things, channels, relations, poses and rules screens are drawn from.

    Built by :func:`load_graph`; see the module docstring for what it leaves out.
    """

    def __init__(
        self,
        records: FacilityRecords,
        tables: PanelTables,
        ring_frame: str | None = None,
        *,
        content_hash: str | None = None,
    ) -> None:
        self._records = records
        self._tables = tables
        self._content_hash = content_hash
        excluded = {kind: set[str]() for kind in ("thing", "channel", "document")}
        for row in tables.excluded:
            excluded[row.kind].add(row.id)
        self._excluded_documents = frozenset(excluded["document"])
        self._things = self._read_things(records, excluded["thing"])
        self._children: dict[str, list[Thing]] = {}
        for thing in self._things.values():
            if thing.kind != "group" and thing.place is not None:
                self._children.setdefault(thing.place, []).append(thing)
        self._members: dict[str, tuple[str, ...]] = {
            str(group["id"]): tuple(str(member) for member in group.get("members") or ())
            for group in records.of_kind("group")
            if ("group", str(group["id"])) in self._things
        }
        self._read_channels(records, tables, excluded["channel"])
        self._read_edges(records)
        self._read_poses(tables, ring_frame)
        self._rules: dict[str, list[HealthRule]] = {}
        for rule in tables.health:
            if rule.channel in self._channel:
                self._rules.setdefault(rule.channel, []).append(rule)
        self._read_pairs(tables)
        self._pictures: dict[str, list[PicturePart]] = {}
        for part in tables.pictures:
            if part.document not in self._excluded_documents:
                self._pictures.setdefault(part.device_class, []).append(part)

    # -- building -------------------------------------------------------------

    def _citations(self, record: Mapping[str, Any]) -> tuple[Citation, ...]:
        cited = (_citation(entry) for entry in record.get("citations") or ())
        return tuple(
            citation
            for citation in cited
            if citation is not None and citation.document not in self._excluded_documents
        )

    def _read_things(self, records: FacilityRecords, excluded: set[str]) -> dict[_Key, Thing]:
        every_place = {str(place["id"]) for place in records.of_kind("place")}
        hidden = {
            place
            for place in every_place
            if any(
                "/".join(place.split("/")[:depth]) in excluded
                for depth in range(1, place.count("/") + 2)
            )
        }
        places = every_place - hidden
        things: dict[_Key, Thing] = {}
        for kind in END_KINDS:
            for record in records.of_kind(kind):  # type: ignore[arg-type]
                thing_id = str(record["id"])
                if thing_id in excluded or (kind == "place" and thing_id in hidden):
                    continue
                if kind == "device" and _text(record.get("place")) in hidden:
                    continue
                if kind == "device":
                    place = _text(record.get("place"))
                elif kind == "place" and "/" in thing_id:
                    place = thing_id.rsplit("/", 1)[0]
                else:
                    place = None
                things[(kind, thing_id)] = Thing(
                    id=thing_id,
                    kind=kind,  # type: ignore[arg-type]
                    cls=_text(record.get("class")) if kind == "device" else None,
                    place=place if place in places else None,
                    citations=self._citations(record),
                    level=_text(record.get("level")) if kind == "place" else None,
                )
        return things

    def _read_channels(
        self, records: FacilityRecords, tables: PanelTables, excluded: set[str]
    ) -> None:
        types = {row.channel: row for row in tables.record_types}
        self._types: dict[str, RecordType] = {}
        self._channel: dict[str, Channel] = {}
        self._channels: dict[_Key, list[Channel]] = {}
        for record in records.of_kind("channel"):
            name = str(record["id"])
            if name in excluded or name.startswith(SIMULATION_PREFIX):
                continue
            on = record.get("on") or {}
            holder = end_identity(on) if isinstance(on, Mapping) else None
            if holder is None or holder not in self._things:
                continue
            row = types.get(name)
            names = sorted(str(alias) for alias in record.get("names") or ())
            channel = Channel(
                name=name,
                drawn_as=row.drawn_as if row is not None else None,
                writable=record.get("role") == "setpoint",
                alias=names[0] if names else None,
                thing=holder[1],
                thing_kind=holder[0],  # type: ignore[arg-type]
                record_type=row.record_type if row is not None else None,
                unit=_text(record.get("unit")),
                description=_text(record.get("description")),
                tolerance=_tolerance(record.get("tolerance")),
            )
            self._channel[name] = channel
            if row is not None:
                self._types[name] = row
            holders = dict.fromkeys(
                [holder, *(("device", str(device)) for device in record.get("endpoint_of") or ())]
            )
            for key in holders:
                if key in self._things:
                    self._channels.setdefault(key, []).append(channel)

    def _read_edges(self, records: FacilityRecords) -> None:
        self._edges: dict[_Key, dict[str, list[Edge]]] = {}
        for claim in records.document.get("relations") or ():
            if not isinstance(claim, Mapping) or claim.get("standing") == "superseded":
                continue
            ends = [claim.get("subject"), claim.get("object")]
            if not all(isinstance(end, Mapping) for end in ends):
                continue
            keys = [end_identity(end) for end in ends]
            if any(key is None or key not in self._things for key in keys):
                continue
            source = _citation(claim.get("source"))
            if source is not None and source.document in self._excluded_documents:
                continue
            units = [end.get("unit") for end in ends if end.get("unit") is not None]
            for index in (0, 1):
                here, there = keys[index], keys[1 - index]
                assert here is not None and there is not None
                edge = Edge(
                    kind=str(claim.get("type")),
                    claim=str(claim.get("id")),
                    outgoing=index == 0,
                    other=there[1],
                    other_kind=there[0],  # type: ignore[arg-type]
                    standing=str(claim.get("standing") or "accepted"),
                    basis=str(claim.get("basis") or "stated"),
                    port=_text(ends[index].get("port")),
                    other_port=_text(ends[1 - index].get("port")),
                    unit=int(units[0]) if units else None,
                    attributes=MappingProxyType(dict(claim.get("attributes") or {})),
                    source=source,
                )
                self._edges.setdefault(here, {}).setdefault(edge.kind, []).append(edge)

    def _read_poses(self, tables: PanelTables, ring_frame: str | None) -> None:
        thing_ids = {thing_id for _kind, thing_id in self._things}
        kept = [
            pose
            for pose in tables.positions
            if pose.thing in thing_ids and pose.document not in self._excluded_documents
        ]
        if ring_frame is None:
            ring_frame = _ring_frame(kept)
        self.ring_frame: str | None = ring_frame
        """The frame whose ``s_m`` is the ring position."""
        self._poses: dict[str, list[Pose]] = {}
        for pose in kept:
            ringed = dataclasses.replace(pose, in_ring_frame=pose.frame == ring_frame)
            self._poses.setdefault(pose.thing, []).append(ringed)
        for poses in self._poses.values():
            poses.sort(key=lambda pose: not pose.in_ring_frame)

    def _read_pairs(self, tables: PanelTables) -> None:
        self._pairs: dict[str, Pair] = {}
        for pair in tables.pairs:
            if pair.setpoint in self._channel and pair.readback in self._channel:
                self._pairs[pair.setpoint] = pair

    # -- reading --------------------------------------------------------------

    def _key(self, thing_id: str, kind: str | None = None) -> _Key:
        for candidate in (kind,) if kind is not None else END_KINDS:
            if (candidate, thing_id) in self._things:
                return (candidate, thing_id)
        where = f"{kind} " if kind is not None else ""
        raise KeyError(f"the panel graph holds no {where}thing {thing_id!r}")

    @property
    def content_hash(self) -> str:
        """The SHA-256 hex digest of what the graph was read from.

        :func:`load_graph` hashes the facility file's bytes and the rule
        tables' files, so the digest is the same on every machine. A graph
        built from records and tables directly hashes their canonical text
        instead, on first use.
        """
        if self._content_hash is None:
            digest = hashlib.sha256()
            text = json.dumps(
                self._records.document, sort_keys=True, ensure_ascii=False, default=str
            )
            digest.update(text.encode("utf-8"))
            digest.update(repr(self._tables).encode("utf-8"))
            self._content_hash = digest.hexdigest()
        return self._content_hash

    def things(self) -> list[Thing]:
        """Every thing: devices, then places, then groups, each in file order."""
        return list(self._things.values())

    def thing(self, thing_id: str, kind: ThingKind | None = None) -> Thing:
        """The thing with *thing_id*.

        Raises:
            KeyError: No such thing, or it is excluded.
        """
        return self._things[self._key(thing_id, kind)]

    def neighbours(self, thing_id: str, kind: ThingKind | None = None) -> dict[str, list[Edge]]:
        """The thing's edges by relation kind, both directions, in file order.

        Raises:
            KeyError: No such thing, or it is excluded.
        """
        edges = self._edges.get(self._key(thing_id, kind), {})
        return {relation: list(found) for relation, found in edges.items()}

    def channels(self, thing_id: str, kind: ThingKind | None = None) -> list[Channel]:
        """The channels on the thing (``on`` it or, for a device, ``endpoint_of`` it).

        Raises:
            KeyError: No such thing, or it is excluded.
        """
        return list(self._channels.get(self._key(thing_id, kind), ()))

    def channel(self, name: str) -> Channel:
        """The channel *name*.

        Raises:
            KeyError: No such channel, or it is left out.
        """
        try:
            return self._channel[name]
        except KeyError:
            raise KeyError(f"the panel graph holds no channel {name!r}") from None

    def members(self, group_id: str) -> list[Thing]:
        """The group's members the graph holds, in the group's order.

        Raises:
            KeyError: No such group, or it is excluded.
        """
        self._key(group_id, "group")
        found = []
        for member in self._members[group_id]:
            try:
                found.append(self.thing(member))
            except KeyError:
                continue
        return found

    def within(self, place_id: str, *, deep: bool = False) -> list[Thing]:
        """The devices and places whose place is *place_id*.

        Args:
            place_id: A place.
            deep: Also everything within those places, depth first.

        Raises:
            KeyError: No such place, or it is excluded.
        """
        self._key(place_id, "place")
        found: list[Thing] = []
        self._collect(place_id, deep, found)
        return found

    def _collect(self, place_id: str, deep: bool, found: list[Thing]) -> None:
        for thing in self._children.get(place_id, ()):
            found.append(thing)
            if deep and thing.kind == "place":
                self._collect(thing.id, deep, found)

    def pose(self, thing_id: str) -> Pose | None:
        """The thing's ring-frame pose, else its first pose, else ``None``."""
        poses = self._poses.get(thing_id)
        return poses[0] if poses else None

    def poses(self, thing_id: str) -> list[Pose]:
        """Every pose of the thing, the ring-frame row first, then file order."""
        return list(self._poses.get(thing_id, ()))

    def rule(self, channel: str) -> list[HealthRule]:
        """Every health row for the channel, all sources, in file order."""
        return list(self._rules.get(channel, ()))

    def rules(self) -> list[HealthRule]:
        """Every health row of every channel the graph holds, by channel then file order."""
        return [row for rows in self._rules.values() for row in rows]

    def record_type(self, channel: str) -> RecordType | None:
        """The ``record_types.csv`` row of a channel the graph holds, or ``None``."""
        return self._types.get(channel)

    def pictures(self, device_class: str) -> list[PicturePart]:
        """The ``pictures.csv`` rows of a device class, in file order."""
        return list(self._pictures.get(device_class, ()))

    def pair(self, setpoint: str) -> Pair | None:
        """The readback ``pairs.csv`` shows the setpoint beside, or ``None``."""
        return self._pairs.get(setpoint)


def load_graph(facility: Path, panels_dir: Path, *, ring_frame: str | None = None) -> PanelGraph:
    """Read a built facility file and its rule tables into a :class:`PanelGraph`.

    Args:
        facility: The facility file, or the directory holding ``facility.json``.
        panels_dir: The directory holding the rule tables.
        ring_frame: The frame whose ``s_m`` is the ring position; ``None``
            finds it in ``positions.csv``.

    Returns:
        The graph.

    Raises:
        FileNotFoundError: No facility file is there.
        FacilityFileError: The facility file cannot be read.
        PanelTableError: A rule table breaks its schema, or two frames tie
            for the ring frame and none is passed.
    """
    path = Path(facility)
    if path.is_dir():
        path = path / FACILITY_FILE
    records = FacilityRecords.from_file(path)
    if records is None:
        raise FileNotFoundError(f"no facility file at {path}")
    tables = read_tables(panels_dir)
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    for name in TABLE_FILES:
        digest.update(b"\0" + name.encode("utf-8") + b"\0")
        table = Path(panels_dir) / name
        digest.update(table.read_bytes() if table.is_file() else b"")
    return PanelGraph(records, tables, ring_frame=ring_frame, content_hash=digest.hexdigest())
