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
  source of: a citation, a relation claim, a pose, a health row, a pair, a
  record type;
- a ``superseded`` relation claim. ``accepted`` and ``contested`` claims are
  edges, their standing on the edge. A relation kind the framework does not
  know is an edge under its own key like any other.

A channel that is ``on`` no thing the file holds, and is no kept device's
endpoint, is kept as a channel of no thing (:meth:`PanelGraph.unheld`). A
channel ``on`` a left-out thing, or with no ``on`` and only left-out
endpoints, is left out with them.

A value of a row the graph cannot read (a rack unit that is no integer, an
attributes field that is no mapping, members that are no list) never stops
the graph: the row is kept without it, or with it as text where it is shown
as text, and :attr:`PanelGraph.unread` counts such values.

The ring frame is the frame of ``positions.csv`` that carries the most rows
with an empty ``derived_assumes`` (on a
tie, the rows name no ring frame: :func:`load_graph` refuses and asks for
one), unless :func:`load_graph` is told which.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from osprey.facility import FACILITY_FILE
from osprey.facility.panels.tables import (
    CALIBRATION_BASES,
    CURVE_PREFIX,
    TABLE_FILES,
    WALKED_RELATIONS,
    Calibration,
    CalibrationKind,
    CardRecipe,
    CardRule,
    CardSection,
    ChainStep,
    ChannelRole,
    ChannelUnit,
    CurvePoint,
    DrawnAs,
    FrontRow,
    HealthRule,
    Metric,
    Pair,
    PanelTables,
    PicturePart,
    Pose,
    ProbeElement,
    RecordType,
    StripSymbol,
    read_tables,
)
from osprey.facility.records import FacilityRecords
from osprey.facility.relations import END_KINDS, end_identity

__all__ = [
    "CalibrationState",
    "PART_RELATION",
    "PLACE_RELATION",
    "SIMULATION_PREFIX",
    "Channel",
    "Citation",
    "Edge",
    "PanelGraph",
    "RoleHit",
    "Thing",
    "ThingKind",
    "load_graph",
]

#: The relation kind that states a thing's place.
PLACE_RELATION = "located_in"

#: The relation kind, from the facility vocabulary, that makes a device a component of
#: another device; a part with no place of its own is in the place of its whole.
PART_RELATION = "part_of"

#: A port's trailing number, which a role pattern's ``{n}`` stands for.
_PORT_NUMBER = re.compile(r"(\d+)$")

#: The prefix of a channel only the simulation serves; no screen shows one.
SIMULATION_PREFIX = "VA:"

ThingKind = Literal["device", "place", "group"]

#: ``(kind, id)``: one thing.
_Key = tuple[str, str]


@dataclass(frozen=True)
class RoleHit:
    """A channel that plays a ``channel_roles.csv`` role for a device.

    Attributes:
        role: The row.
        channel: The channel.
        owner: The device the channel is on: the device itself, a part of it, or
            the device a walked relation names.
        port: The walked relation's object port, or ``None``.
        join: For a channel of another thing reached through a relation claim
            with basis ``inferred`` (a walked relation or ``part_of``): that
            claim's rule, ``""`` when it states none; ``None`` for a stated join
            and for the device's own channels.
    """

    role: ChannelRole
    channel: str
    owner: str
    port: str | None = None
    join: str | None = None


def _join(edges: Iterable[Edge]) -> str | None:
    """The rule of the first inferred claim among *edges* (``""`` when it states
    none); ``None`` when every one is stated."""
    for edge in edges:
        if edge.basis == "inferred":
            return edge.rule or ""
    return None


def _takes(pattern: str, thing_id: str) -> bool:
    """Whether a ``devices`` pattern takes *thing_id*: a glob, or with ``!`` first every
    id the rest does not match."""
    if pattern.startswith("!"):
        return not fnmatch.fnmatchcase(thing_id, pattern[1:])
    return fnmatch.fnmatchcase(thing_id, pattern)


def substitute(pattern: str, port: str | None) -> str | None:
    """*pattern* with ``{port}`` and ``{n}`` filled from a relation's object port.

    ``None`` where the pattern needs what the port does not give: no port, or for
    ``{n}`` a port with no trailing number. The port is matched literally. The
    facility repository's role producer does the same, so a card and the table
    agree on what a pattern means.
    """
    if "{port}" in pattern:
        if not port:
            return None
        pattern = pattern.replace("{port}", re.sub(r"([*?\[])", r"[\1]", port))
    if "{n}" in pattern:
        match = _PORT_NUMBER.search(port or "")
        if not match:
            return None
        pattern = pattern.replace("{n}", match.group(1))
    return pattern


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
        short_name: The short name ``names.csv`` states for it (a strip's
            label), or ``None``.
        model: The model a device belongs to, or ``None``.
    """

    id: str
    kind: ThingKind
    cls: str | None
    place: str | None
    citations: tuple[Citation, ...] = ()
    level: str | None = None
    short_name: str | None = None
    model: str | None = None
    #: For a group: its kind as stated, ``family`` when absent (only a family is a
    #: set); ``None`` for a device or a place.
    group_kind: str | None = None
    #: A device's attributes as ``(name, value)`` pairs, values as text (a list
    #: value gives one pair per item), in name order; empty for a place or a group.
    attributes: tuple[tuple[str, str], ...] = ()
    #: A device's ``s`` along its model's beam path, as the build computed or a
    #: source stated it; ``None`` without one.
    s_m: float | None = None


@dataclass(frozen=True)
class Channel:
    """A channel of the facility file as a screen binds it.

    Attributes:
        name: The channel's address.
        drawn_as: How ``record_types.csv`` says to draw it; ``None`` with no row.
        writable: Whether the facility file's role is ``setpoint``.
        alias: Another name the facility file states for it (the first of its
            ``names``, sorted), or ``None``.
        thing: The id of the thing it is ``on``; ``None`` when it is on no thing
            the graph holds.
        thing_kind: That thing's kind, or ``None``.
        record_type: The record type ``record_types.csv`` states, or ``None``.
        unit: The facility file's unit, or ``None``.
        description: The facility file's description, or ``None``.
        tolerance: The facility file's ``tolerance`` as ``(term, amount)``,
            ``term`` being ``absolute`` or ``relative``; ``None`` with none.
        signal: The signal role the facility file states for it (its
            ``signal``), or ``None``.
    """

    name: str
    drawn_as: DrawnAs | None
    writable: bool
    alias: str | None
    thing: str | None
    thing_kind: ThingKind | None
    record_type: str | None = None
    unit: str | None = None
    description: str | None = None
    tolerance: tuple[str, float] | None = None
    signal: str | None = None


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
        unit: The rack unit either end names, as an integer; ``None`` when it
            names none or one that is no integer.
        unit_text: The unit as the end states it, when it is no integer
            (``A5``); ``None`` otherwise.
        attributes: The claim's attributes, read-only.
        source: The document the claim cites, or ``None``.
        rule: For an inferred claim, the rule it was inferred by, when stated.
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
    unit_text: str | None = None
    rule: str | None = None


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


def _unit(value: Any) -> tuple[int | None, str | None]:
    """A rack unit as ``(number, None)``, or ``(None, text)`` when it is no integer."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value, None
    if isinstance(value, float) and value.is_integer():
        return int(value), None
    text = str(value).strip()
    try:
        return int(text), None
    except ValueError:
        return None, text


def _beam_frames(poses: list[Pose], model_of: Mapping[str, str | None]) -> dict[str | None, str]:
    """Per model (``None``: the devices that name none), the frame its devices' ``s_m``
    is the beam position in: the frame with the most of their rows that state an
    ``s_m`` and assume nothing; a tie goes to the frame with the most such rows of
    all devices, then to the first by name. A frame where fewer of them are stated
    (a girder's own frame) is a local frame."""
    counts: dict[str | None, Counter[str]] = {}
    overall: Counter[str] = Counter()
    for pose in poses:
        if pose.thing in model_of and pose.s_m is not None and not pose.derived_assumes:
            counts.setdefault(model_of[pose.thing], Counter())[pose.frame] += 1
            overall[pose.frame] += 1
    return {
        model: min(counted, key=lambda frame: (-counted[frame], -overall[frame], frame))
        for model, counted in counts.items()
    }


@dataclass(frozen=True)
class CalibrationState:
    """What the records say of one calibration kind of one device.

    Attributes:
        kind: The kind.
        subject: The device.
        in_force: The claim in force: the newest accepted claim no other claim
            supersedes, or ``None``. A contested claim is never in force: it is
            no base value, shown only as contested.
        state: ``contested`` when the claim in force says so or two accepted
            claims in force of one basis differ by more than the kind's
            tolerance; else the basis of the claim in force; else ``none``.
        below_wanted: The claim in force ranks under the kind's wanted basis
            (in :data:`~osprey.facility.panels.tables.CALIBRATION_BASES`, a
            display order), or there is none.
        design: The newest accepted ``design`` claim, in force or not, the
            value a deviation is taken from; ``None`` with none.
        earlier: The other claims of the kind for the device, newest first.
        rivals: The accepted claims in force that disagree with the claim in
            force, with it excluded.
        points: The points of the claim in force when its value is a curve.
        contested: The contested claims no other claim supersedes, newest first.
    """

    kind: CalibrationKind
    subject: str
    in_force: Calibration | None
    state: str
    below_wanted: bool
    design: Calibration | None
    earlier: tuple[Calibration, ...]
    rivals: tuple[Calibration, ...]
    points: tuple[CurvePoint, ...] = ()
    contested: tuple[Calibration, ...] = ()


def calibration_numbers(value: str) -> tuple[float, ...] | None:
    """The numbers of a claim's value: one for a number, several for ``a;b;c``;
    ``None`` for a curve or text that is no number."""
    if value.startswith(CURVE_PREFIX):
        return None
    try:
        return tuple(float(part) for part in value.split(";"))
    except ValueError:
        return None


def _disagree(a: Calibration, b: Calibration, tolerance: tuple[str, float] | None) -> bool:
    """Whether claim *a* differs from claim *b* (the one in force) by more than
    *tolerance* in any number: ``abs`` in the kind's unit, ``rel`` a fraction of
    *b*'s number. Claims with no tolerance, or values that are no numbers, are
    not compared."""
    x, y = calibration_numbers(a.value), calibration_numbers(b.value)
    if tolerance is None or x is None or y is None or len(x) != len(y):
        return False
    term, amount = tolerance
    return any(
        abs(p - q) > (amount * abs(q) if term == "rel" else amount)
        for p, q in zip(x, y, strict=True)
    )


def _value_key(value: object) -> str:
    """A discrete value as one text: a whole number without its point."""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


#: The ``class`` of a ``cards.csv`` row that matches a device of any class.
ANY_CLASS = "*"


def _number_or_none(value: Any) -> float | None:
    """A JSON number as a float; anything else (a bool included) ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


#: The frame name of a model's beam path: positions in it are its devices' ``s``.
MODEL_FRAME = "model {}"


def _attribute_pairs(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """A device record's ``attributes`` map as sorted ``(name, value)`` text pairs;
    a list value gives one pair per item, a map value none."""
    found = record.get("attributes")
    if not isinstance(found, Mapping):
        return ()
    pairs: set[tuple[str, str]] = set()
    for name, value in found.items():
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, bool):
                pairs.add((str(name), "true" if item else "false"))
            elif isinstance(item, (str, int, float)):
                pairs.add((str(name), str(item)))
    return tuple(sorted(pairs))


class PanelGraph:
    """The things, channels, relations, poses and rules screens are drawn from.

    Built by :func:`load_graph`; see the module docstring for what it leaves out.
    """

    def __init__(
        self,
        records: FacilityRecords,
        tables: PanelTables,
        *,
        content_hash: str | None = None,
    ) -> None:
        self._records = records
        self._lengths: dict[str, float] | None = None
        self._tables = tables
        self._content_hash = content_hash
        self._under_names: dict[str, list[Thing]] | None = None
        self.unread = 0
        """How many values of rows the graph could not read (see the module docstring)."""
        excluded = {kind: set[str]() for kind in ("thing", "channel", "document")}
        for row in tables.excluded:
            excluded[row.kind].add(row.id)
        self._excluded_documents = frozenset(excluded["document"])
        self._things = self._read_things(records, excluded["thing"])
        short = {
            row.thing: row.short_name
            for row in tables.names
            if row.document not in self._excluded_documents
        }
        for key, thing in self._things.items():
            if thing.id in short:
                self._things[key] = dataclasses.replace(thing, short_name=short[thing.id])
        self._children: dict[str, list[Thing]] = {}
        for thing in self._things.values():
            if thing.kind != "group" and thing.place is not None:
                self._children.setdefault(thing.place, []).append(thing)
        self._members: dict[str, tuple[str, ...]] = {}
        for group in records.of_kind("group"):
            if ("group", str(group["id"])) not in self._things:
                continue
            members = group.get("members") or ()
            if not isinstance(members, list | tuple):
                self.unread += 1
                members = ()
            self._members[str(group["id"])] = tuple(str(member) for member in members)
        self._read_channels(records, tables, excluded["channel"])
        self._read_edges(records)
        self._read_claims()
        self._read_wholes()
        self._read_poses(tables)
        self._rules: dict[str, list[HealthRule]] = {}
        for rule in tables.health:
            if rule.channel in self._channel and rule.source not in self._excluded_documents:
                self._rules.setdefault(rule.channel, []).append(rule)
        self._read_pairs(tables)
        self._pictures: dict[str, list[PicturePart]] = {}
        self._held_by: dict[str, frozenset[str]] = {}
        for part in tables.pictures:
            if part.document not in self._excluded_documents:
                self._pictures.setdefault(part.device_class, []).append(part)
        self._read_calibrations(tables)
        self._labels: dict[tuple[str, str], str] = {
            (row.channel, _value_key(row.value)): row.label for row in tables.state_labels
        }
        self._words: dict[str, dict[str, str]] = {}
        for label in tables.state_labels:
            self._words.setdefault(label.channel, {})[_value_key(label.value)] = label.label
        self._units = {row.channel: row for row in tables.units}
        self._roles_of: dict[str, list[ChannelRole]] = {}
        for role in tables.channel_roles:
            if role.source not in self._excluded_documents:
                self._roles_of.setdefault(role.cls, []).append(role)
        self._recipes: dict[str, list[CardRecipe]] = {}
        for recipe in tables.card_recipes:
            self._recipes.setdefault(recipe.cls, []).append(recipe)
        self._sections: dict[str, list[CardSection]] = {}
        for section in sorted(tables.card_sections, key=lambda r: r.order):
            self._sections.setdefault(section.recipe, []).append(section)
        self._symbols: dict[str, list[StripSymbol]] = {}
        for symbol in tables.strip_symbols:
            self._symbols.setdefault(symbol.cls, []).append(symbol)
        self._fronts: dict[str, list[FrontRow]] = {}
        for front in sorted(tables.fronts, key=lambda r: r.order):
            self._fronts.setdefault(front.recipe, []).append(front)
        self._steps: dict[tuple[str, str], list[ChainStep]] = {}
        for step in sorted(tables.chains, key=lambda r: r.step):
            self._steps.setdefault((step.recipe, step.chain), []).append(step)

    def _read_calibrations(self, tables: PanelTables) -> None:
        self._kinds = {(kind.kind, kind.cls): kind for kind in tables.calibration_kinds}
        self._under: dict[str, frozenset[str]] = {}
        self._kinds_of: dict[str, list[CalibrationKind]] = {}
        for kind in tables.calibration_kinds:  # classes in file order
            self._kinds_of.setdefault(kind.cls, [])
        for kind in sorted(
            tables.calibration_kinds, key=lambda k: (k.order is None, k.order or 0.0, k.kind)
        ):
            self._kinds_of.setdefault(kind.cls, []).append(kind)
        self._claims: dict[tuple[str, str], list[Calibration]] = {}
        for claim in tables.calibrations:
            if claim.document not in self._excluded_documents:
                self._claims.setdefault((claim.subject, claim.kind), []).append(claim)
        self._superseded = {
            claim.supersedes
            for claims in self._claims.values()
            for claim in claims
            if claim.supersedes is not None
        }
        self._points: dict[str, list[CurvePoint]] = {}
        for point in tables.calibration_curves:
            self._points.setdefault(point.id, []).append(point)

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
        withheld = self._withheld_parts(records, excluded, hidden)
        self.withheld = len(withheld)
        """How many devices are left out as part of an excluded thing (at any depth)."""
        things: dict[_Key, Thing] = {}
        self._left_out: set[_Key] = set()
        for kind in END_KINDS:
            for record in records.of_kind(kind):  # type: ignore[arg-type]
                thing_id = str(record["id"])
                if (
                    thing_id in excluded
                    or (kind == "place" and thing_id in hidden)
                    or (kind == "device" and thing_id in withheld)
                ):
                    self._left_out.add((kind, thing_id))
                    continue
                if kind == "device" and _text(record.get("place")) in hidden:
                    self._left_out.add((kind, thing_id))
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
                    model=_text(record.get("model")) if kind == "device" else None,
                    group_kind=(_text(record.get("kind")) or "family") if kind == "group" else None,
                    attributes=_attribute_pairs(record) if kind == "device" else (),
                    s_m=_number_or_none(record.get("s")) if kind == "device" else None,
                )
        return things

    @staticmethod
    def _withheld_parts(records: FacilityRecords, excluded: set[str], hidden: set[str]) -> set[str]:
        """The devices ``part_of`` an excluded device (at any depth), which are left out too.

        A device is excluded by name or by a place it is in; the ``part_of``
        claims that are not superseded are followed down from those, each
        device once, so a loop ends.
        """
        gone = {
            str(record["id"])
            for record in records.of_kind("device")
            if str(record["id"]) in excluded or _text(record.get("place")) in hidden
        }
        parts: dict[str, list[str]] = {}
        for claim in records.document.get("relations") or ():
            if not isinstance(claim, Mapping) or claim.get("type") != PART_RELATION:
                continue
            if claim.get("standing") == "superseded":
                continue
            ends = [claim.get("subject"), claim.get("object")]
            if not all(isinstance(end, Mapping) and "device" in end for end in ends):
                continue
            parts.setdefault(str(ends[1]["device"]), []).append(str(ends[0]["device"]))  # type: ignore[index]
        withheld: set[str] = set()
        stack = list(gone)
        while stack:
            for part in parts.get(stack.pop(), ()):
                if part not in gone and part not in withheld:
                    withheld.add(part)
                    stack.append(part)
        return withheld

    def _read_channels(
        self, records: FacilityRecords, tables: PanelTables, excluded: set[str]
    ) -> None:
        types = {
            row.channel: row
            for row in tables.record_types
            if row.source not in self._excluded_documents
        }
        self._types: dict[str, RecordType] = {}
        self._channel: dict[str, Channel] = {}
        self._channels: dict[_Key, list[Channel]] = {}
        self._unheld: list[Channel] = []
        for record in records.of_kind("channel"):
            name = str(record["id"])
            if name in excluded or name.startswith(SIMULATION_PREFIX):
                continue
            on = record.get("on") or {}
            holder = end_identity(on) if isinstance(on, Mapping) and on else None
            if holder is not None and holder not in self._things:
                if holder in self._left_out:
                    continue  # on a thing no screen shows
                holder = None  # on nothing the file holds
            endpoints = [("device", str(device)) for device in record.get("endpoint_of") or ()]
            if holder is None and endpoints and all(e in self._left_out for e in endpoints):
                continue
            row = types.get(name)
            names = sorted(str(alias) for alias in record.get("names") or ())
            channel = Channel(
                name=name,
                drawn_as=row.drawn_as if row is not None else None,
                writable=record.get("role") == "setpoint",
                alias=names[0] if names else None,
                thing=holder[1] if holder is not None else None,
                thing_kind=holder[0] if holder is not None else None,  # type: ignore[arg-type]
                record_type=row.record_type if row is not None else None,
                unit=_text(record.get("unit")),
                description=_text(record.get("description")),
                tolerance=_tolerance(record.get("tolerance")),
                signal=_text(record.get("signal")),
            )
            self._channel[name] = channel
            if row is not None:
                self._types[name] = row
            held = False
            for key in dict.fromkeys([*([holder] if holder is not None else []), *endpoints]):
                if key in self._things:
                    self._channels.setdefault(key, []).append(channel)
                    held = True
            if not held:
                self._unheld.append(channel)

    def _read_edges(self, records: FacilityRecords) -> None:
        self._edges: dict[_Key, dict[str, list[Edge]]] = {}
        for claim in records.document.get("relations") or ():
            if not isinstance(claim, Mapping) or claim.get("standing") == "superseded":
                continue
            subject, obj = claim.get("subject"), claim.get("object")
            if not (isinstance(subject, Mapping) and isinstance(obj, Mapping)):
                continue
            ends: list[Mapping[str, Any]] = [subject, obj]
            keys = [end_identity(end) for end in ends]
            if any(key is None or key not in self._things for key in keys):
                continue
            source = _citation(claim.get("source"))
            if source is not None and source.document in self._excluded_documents:
                continue
            units = [end.get("unit") for end in ends if end.get("unit") is not None]
            unit, unit_text = _unit(units[0]) if units else (None, None)
            if unit_text is not None:
                self.unread += 1
            attributes = claim.get("attributes") or {}
            if not isinstance(attributes, Mapping):
                self.unread += 1
                attributes = {}
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
                    unit=unit,
                    attributes=MappingProxyType(dict(attributes)),
                    source=source,
                    unit_text=unit_text,
                    rule=_text(claim.get("rule")),
                )
                self._edges.setdefault(here, {}).setdefault(edge.kind, []).append(edge)

    def _read_claims(self) -> None:
        """Every thing a ``located_in`` claim puts in a place, by place, in graph order."""
        self._claimed: dict[str, list[Thing]] = {}
        self._claims_of: dict[_Key, list[str]] = {}
        for key, thing in self._things.items():
            if thing.kind == "group":
                continue
            for edge in self._edges.get(key, {}).get(PLACE_RELATION, ()):
                if not edge.outgoing or edge.other_kind != "place":
                    continue
                places = self._claims_of.setdefault(key, [])
                if edge.other not in places:
                    places.append(edge.other)
                    self._claimed.setdefault(edge.other, []).append(thing)

    def _own_places(self, key: _Key) -> tuple[str, ...]:
        """The places a thing has by itself: its built place, else its claims' places."""
        place = self._things[key].place
        return (place,) if place is not None else tuple(self._claims_of.get(key, ()))

    def _read_wholes(self) -> None:
        """Every device with no place of its own, put in the place of its nearest whole.

        The walk follows outgoing accepted ``part_of`` edges to devices, breadth first in
        file order, and stops at the first whole that has a built place or a
        claim; a device it reaches twice is not walked again, so a loop ends.
        """
        self._whole_of: dict[_Key, tuple[str, tuple[str, ...]]] = {}
        self._derived: dict[str, list[Thing]] = {}
        self._through: dict[str, list[Thing]] = {}
        for key, thing in self._things.items():
            if thing.kind != "device" or self._own_places(key):
                continue
            seen = {key}
            level = [key]
            found: tuple[_Key, tuple[str, ...]] | None = None
            while level and found is None:
                upper: list[_Key] = []
                for here in level:
                    for edge in self._edges.get(here, {}).get(PART_RELATION, ()):
                        whole = (edge.other_kind, edge.other)
                        if (
                            not edge.outgoing
                            or edge.other_kind != "device"
                            or edge.standing != "accepted"
                            or whole in seen
                        ):
                            continue
                        seen.add(whole)
                        places = self._own_places(whole)
                        if places:
                            found = (whole, places)
                            break
                        upper.append(whole)
                    if found is not None:
                        break
                level = upper
            if found is None:
                continue
            self._whole_of[key] = (found[0][1], found[1])
            self._through.setdefault(found[0][1], []).append(thing)
            for place in found[1]:
                self._derived.setdefault(place, []).append(thing)

    def _read_poses(self, tables: PanelTables) -> None:
        thing_ids = {thing_id for _kind, thing_id in self._things}
        kept = [
            pose
            for pose in tables.positions
            if pose.thing in thing_ids and pose.document not in self._excluded_documents
        ]
        model_of = {
            thing.id: thing.model for thing in self._things.values() if thing.kind == "device"
        }
        self._beam_frame: dict[str | None, str] = _beam_frames(kept, model_of)
        self.beam_frames: frozenset[str] = frozenset(self._beam_frame.values())
        """The frames some model's (or the model-less devices') positions are in."""
        self._poses: dict[str, list[Pose]] = {}
        for pose in kept:
            if pose.thing in model_of:  # a device: its own model's frame
                frame = self._beam_frame.get(model_of[pose.thing])
                on_beam = pose.frame == frame
            else:  # a place: any beam frame
                on_beam = pose.frame in self.beam_frames
            self._poses.setdefault(pose.thing, []).append(
                dataclasses.replace(pose, on_beam=on_beam)
            )
        for poses in self._poses.values():
            poses.sort(key=lambda pose: not pose.is_beam_position)

    def _read_pairs(self, tables: PanelTables) -> None:
        self._pairs: dict[str, Pair] = {}
        for pair in tables.pairs:
            if pair.source in self._excluded_documents:
                continue
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

    def unheld(self) -> list[Channel]:
        """The channels on no thing the graph holds, in file order."""
        return list(self._unheld)

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

    def contents(self, place_id: str) -> list[Thing]:
        """What is in a place: the one definition every screen and roll-up uses.

        The devices and places whose built place is *place_id* or a place
        within it (any depth, depth first), then those a ``located_in`` claim
        puts in it or in a place within it, then the parts with no place of
        their own whose nearest placed whole is there (:meth:`whole_of`); each
        thing once.

        Raises:
            KeyError: No such place, or it is excluded.
        """
        found = self.within(place_id, deep=True)
        seen = {(thing.kind, thing.id) for thing in found}
        places = [place_id, *(thing.id for thing in found if thing.kind == "place")]
        for source in (self._claimed, self._derived):
            for place in places:
                for thing in source.get(place, ()):
                    if (thing.kind, thing.id) not in seen:
                        seen.add((thing.kind, thing.id))
                        found.append(thing)
        return found

    def claimed(self, place_id: str) -> list[Thing]:
        """The things a ``located_in`` claim puts directly in the place, in graph order."""
        return list(self._claimed.get(place_id, ()))

    def derived(self, place_id: str) -> list[Thing]:
        """The parts :meth:`whole_of` puts directly in the place, in graph order."""
        return list(self._derived.get(place_id, ()))

    def parts_through(self, device_id: str) -> list[Thing]:
        """The parts :meth:`whole_of` places through the device (it is their nearest
        placed whole), in graph order."""
        return list(self._through.get(device_id, ()))

    def whole_of(self, thing_id: str) -> tuple[str, tuple[str, ...]] | None:
        """``(whole, places)`` for a device placed only as part of a whole, else ``None``.

        The placement is derived: a device with a built place or a
        ``located_in`` claim of its own keeps it and has none.
        """
        return self._whole_of.get(("device", thing_id))

    def place_of(self, thing_id: str, kind: ThingKind | None = None) -> str | None:
        """The one place a thing stands in for its crumbs and strips: its built place,
        else, for a part placed by its whole (:meth:`whole_of`), the whole's first
        place; else ``None``."""
        key = self._key(thing_id, kind)
        place = self._things[key].place
        if place is not None:
            return place
        whole = self._whole_of.get(key)
        return whole[1][0] if whole is not None else None

    def placed(self, thing_id: str, kind: ThingKind | None = None) -> bool:
        """Whether the thing is in some place: built, claimed, or as part of a whole."""
        key = self._key(thing_id, kind)
        return bool(self._own_places(key)) or key in self._whole_of

    def unplaced(self) -> list[Thing]:
        """Every device in no place by :meth:`contents`, in graph order."""
        return [
            thing
            for key, thing in self._things.items()
            if thing.kind == "device" and not self._own_places(key) and key not in self._whole_of
        ]

    def beam_position(self, thing: Thing) -> tuple[str, float] | None:
        """Where a thing is along a beam path, as ``(frame, s)``: its pose in its
        model's beam frame (:attr:`Pose.is_beam_position`; a place: in any beam frame),
        else a device record's ``s`` in its model's frame (:data:`MODEL_FRAME`), else
        ``None``."""
        pose = self.pose(thing.id)
        if pose is not None and pose.is_beam_position:
            return pose.frame, float(pose.s_m)  # type: ignore[arg-type]
        if thing.s_m is not None and thing.model is not None:
            return MODEL_FRAME.format(thing.model), thing.s_m
        return None

    def frame_length(self, frame: str) -> float | None:
        """The length a closed beam path (a ring) wraps at, for positions in *frame*:
        the ``length`` of a model the facility file marks ``periodic``; ``None`` for a
        path the beam passes once, or one whose length is not stated. A pose frame
        takes the length of the model it is the beam frame of."""
        if self._lengths is None:
            rings: dict[str, float] = {}
            for model in self._records.document.get("models") or ():
                if not isinstance(model, Mapping) or model.get("periodic") is not True:
                    continue
                length = _number_or_none(model.get("length"))
                if length:
                    rings[MODEL_FRAME.format(model.get("name"))] = length
            for name, posed in self._beam_frame.items():
                found = rings.get(MODEL_FRAME.format(name)) if name is not None else None
                if found is not None:
                    rings[posed] = found
            self._lengths = rings
        return self._lengths.get(frame)

    def pose(self, thing_id: str) -> Pose | None:
        """The thing's beam-frame pose, else its first pose, else ``None``."""
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

    def pictures(self, device_class: str, device: str | None = None) -> list[PicturePart]:
        """The ``pictures.csv`` rows of a device class, in file order; with *device*,
        only those that apply to it: a row with no ``under``, or one whose ``under``
        place holds the device (:meth:`contents`); and a row with no ``devices``, or
        one whose ``devices`` pattern takes the device's id (``!`` first: every
        device but those it matches). A row under a place the graph does not hold
        applies to no device."""
        rows = list(self._pictures.get(device_class, ()))
        if device is None:
            return rows
        return [
            r
            for r in rows
            if (r.under is None or device in self._held(r.under))
            and (r.devices is None or _takes(r.devices, device))
        ]

    def _held(self, place_id: str) -> frozenset[str]:
        """The ids of the devices in a place (:meth:`contents`), once per place."""
        if place_id not in self._held_by:
            try:
                things = self.contents(place_id)
            except KeyError:
                things = []
            self._held_by[place_id] = frozenset(t.id for t in things if t.kind == "device")
        return self._held_by[place_id]

    def metrics(self, view: str, level: str | None) -> list[Metric]:
        """The rows of ``metrics.csv`` for *view* (``place`` or ``rollup``) of a place
        of *level*: those naming that level or none, by ``order``; a row whose source
        is an excluded document is left out."""
        rows = [
            row
            for row in self._tables.metrics
            if row.view == view
            and row.level in (None, level)
            and row.source not in self._excluded_documents
        ]
        return sorted(rows, key=lambda row: row.order)

    def calibration_kinds(self, cls: str | None = None) -> list[CalibrationKind]:
        """The calibration kinds of a device class, by ``order``, wherever they apply
        (:meth:`calibration_kinds_of` asks for one device); every kind with ``None``."""
        if cls is None:
            return [kind for kinds in self._kinds_of.values() for kind in kinds]
        return list(self._kinds_of.get(cls, ()))

    def calibration_kind(self, kind: str, cls: str) -> CalibrationKind:
        """The calibration kind *kind* of the device class *cls*.

        Raises:
            KeyError: No such kind of that class.
        """
        return self._kinds[kind, cls]

    def applies(self, kind: CalibrationKind, device_id: str) -> bool:
        """Whether *kind* is a calibration kind of the device: the device is of the
        kind's class, and belongs to the model or lies in the place (its
        :meth:`contents`) the kind's ``under`` names, when it names one.

        Raises:
            KeyError: No such device.
        """
        device = self.thing(device_id, "device")
        return device.cls == kind.cls and self._under_rank(device, kind.under) is not None

    def _under_rank(self, device: Thing, under: str | None) -> int | None:
        """How specific *under* is for the device: 0 for everywhere, 1 for its model,
        2 and up for a place it lies in, deeper places higher; ``None`` when *under*
        names neither its model nor a place it lies in."""
        if under is None:
            return 0
        if device.id in self._lying_under(under):
            return 2 + under.count("/")
        return 1 if device.model == under else None

    def _lying_under(self, place_id: str) -> frozenset[str]:
        if place_id not in self._under:
            held = ("place", place_id) in self._things
            found = self.contents(place_id) if held else []
            self._under[place_id] = frozenset(t.id for t in found if t.kind == "device")
        return self._under[place_id]

    def calibration_kinds_of(self, device_id: str) -> list[CalibrationKind]:
        """The calibration kinds of a device (:meth:`applies`), by ``order``.

        Raises:
            KeyError: No such device.
        """
        cls = self.thing(device_id, "device").cls
        return [kind for kind in self._kinds_of.get(cls or "", ()) if self.applies(kind, device_id)]

    def calibration(self, device_id: str, kind: str) -> CalibrationState:
        """What the records say of *kind* for a device: the claims whose subject is the
        device, one of its channels, or a part placed through it (:meth:`parts_through`).

        Raises:
            KeyError: No such device, or *kind* is no kind of it (:meth:`applies`).
        """
        cls = self.thing(device_id, "device").cls
        found = self._kinds[kind, cls or ""]
        if not self.applies(found, device_id):
            raise KeyError(f"{kind!r} is no calibration kind of {device_id!r}")
        subjects = [
            device_id,
            *(channel.name for channel in self.channels(device_id, "device")),
            *(part.id for part in self.parts_through(device_id)),
        ]
        order = {id(c): n for n, c in enumerate(self._tables.calibrations)}
        claims = sorted(
            (claim for subject in subjects for claim in self._claims.get((subject, kind), ())),
            key=lambda c: (c.date or "", order[id(c)]),
            reverse=True,
        )  # newest first; a later row wins a tie
        live = [c for c in claims if c.id not in self._superseded and c.standing != "superseded"]
        accepted = [c for c in live if c.standing == "accepted"]
        in_force = accepted[0] if accepted else None  # a contested claim is no base value
        disputed = tuple(c for c in live if c.standing == "contested")
        rivals = tuple(
            c
            for c in accepted
            if in_force is not None
            and c is not in_force
            and c.basis == in_force.basis
            and _disagree(c, in_force, found.tolerance)
        )
        if rivals or disputed:
            state = "contested"
        elif in_force is None:
            state = "none"
        else:
            state = in_force.basis
        rank = CALIBRATION_BASES.index  # a display order, not a precedence
        below = in_force is None or rank(in_force.basis) < rank(found.wanted_basis)
        design = next((c for c in claims if c.basis == "design" and c.standing == "accepted"), None)
        points = (
            tuple(self._points.get(in_force.id, ()))
            if in_force is not None and in_force.value.startswith(CURVE_PREFIX)
            else ()
        )
        return CalibrationState(
            kind=found,
            subject=device_id,
            in_force=in_force,
            state=state,
            below_wanted=below,
            design=design,
            earlier=tuple(c for c in claims if c is not in_force),
            rivals=rivals,
            points=points,
            contested=disputed,
        )

    def probe(self, device_id: str) -> tuple[ProbeElement, ...]:
        """The probe picture of a device: the ``probe_pictures.csv`` rows of its class,
        per element (per curve of a plot) the one whose ``under`` is the most
        specific for the device (:meth:`applies` says which apply), in the file
        order of the rows with no ``under``, then of the others. Rows citing an
        excluded document are left out.

        Raises:
            KeyError: No such device.
        """
        device = self.thing(device_id, "device")
        chosen: dict[tuple[str, str | None], tuple[int, int, ProbeElement]] = {}
        for n, row in enumerate(self._tables.probe_pictures):
            if row.cls != device.cls or row.document in self._excluded_documents:
                continue
            rank = self._under_rank(device, row.under)
            if rank is None:
                continue
            key = (row.element, row.label if row.shape == "plot" else None)
            first = chosen[key][1] if key in chosen else n
            if key not in chosen or rank > chosen[key][0]:
                chosen[key] = (rank, first, row)
        return tuple(row for _rank, _n, row in sorted(chosen.values(), key=lambda c: c[1]))

    def signal_channels(self, device_id: str, signal: str) -> tuple[str, ...]:
        """The channels that state *signal* for a device, in channel order, from the
        first of these that has any: (1) the device the ``port`` of the device's
        relations to devices names, when it names a device (the control system's own
        records); (2) the device itself; (3) the devices those relations name (any
        relation type in :data:`~osprey.facility.panels.tables.WALKED_RELATIONS`).
        Empty when none states it.

        Raises:
            KeyError: No such device.
        """
        near = self.neighbours(device_id, "device")
        edges = [
            edge
            for relation in WALKED_RELATIONS
            for edge in near.get(relation, ())
            if edge.outgoing and edge.other_kind == "device"
        ]
        ports = [
            edge.other_port
            for edge in edges
            if edge.other_port is not None and ("device", edge.other_port) in self._things
        ]
        for level in (ports, [device_id], [edge.other for edge in edges]):
            found = tuple(
                c.name
                for each in level
                for c in self.channels(each, "device")
                if c.signal == signal
            )
            if found:
                return found
        return ()

    def probe_channel(self, device_id: str, signal: str) -> str | None:
        """The first channel :meth:`signal_channels` finds, or ``None``.

        Raises:
            KeyError: No such device.
        """
        return next(iter(self.signal_channels(device_id, signal)), None)

    def picture_matches(self, device_id: str, signal: str) -> list[tuple[str, str]]:
        """What a ``pictures.csv`` row's *signal* matches for a device, as ``(device,
        channel)`` pairs from the first of these levels that has any: (1) the device
        itself; (2) its parts, the devices ``part_of`` it (one level); (3) the devices
        the ``port`` of its relations to devices names (any relation type a
        ``channel_roles.csv`` scope may walk, :data:`~osprey.facility.panels.tables.WALKED_RELATIONS`);
        (4) the devices those relations name. *signal* is a signal name, matched
        against a channel's stated signal, or a pattern with ``*``, ``?`` or ``[``,
        matched against channel names. Matches on more than one device of a level
        are all returned: the caller says which it shows (:meth:`picture_channel`)
        and ``check`` notes them.

        Raises:
            KeyError: No such device.
        """
        pattern = any(mark in signal for mark in "*?[")
        edges = self.neighbours(device_id, "device")
        parts = [
            edge.other
            for edge in edges.get(PART_RELATION, ())
            if not edge.outgoing and edge.other_kind == "device"
        ]
        walked = [
            edge
            for relation in WALKED_RELATIONS
            for edge in edges.get(relation, ())
            if edge.outgoing and edge.other_kind == "device"
        ]
        ports = [
            edge.other_port
            for edge in walked
            if edge.other_port is not None and ("device", edge.other_port) in self._things
        ]
        for level in ([device_id], parts, ports, [edge.other for edge in walked]):
            found = [
                (each, channel.name)
                for each in dict.fromkeys(level)
                for channel in self.channels(each, "device")
                if (
                    fnmatch.fnmatchcase(channel.name, signal)
                    if pattern
                    else channel.signal == signal
                )
            ]
            if found:
                return found
        return []

    def picture_channel(self, device_id: str, signal: str) -> str | None:
        """The channel a ``pictures.csv`` row's *signal* shows on a device: the first
        of :meth:`picture_matches`, or ``None`` when nothing matches.

        Raises:
            KeyError: No such device.
        """
        found = self.picture_matches(device_id, signal)
        return found[0][1] if found else None

    def state_words(self, channel: str) -> dict[str, str]:
        """Every word ``state_labels.csv`` states for *channel*, by value (``"0"``,
        ``"1"``), in file order; empty when it states none."""
        return dict(self._words.get(channel, {}))

    def state_value(self, channel: str, word: str) -> str | None:
        """The value whose stated word for *channel* is *word*, or ``None``."""
        return next((v for v, w in self._words.get(channel, {}).items() if w == word), None)

    def unit(self, channel: str) -> ChannelUnit | None:
        """The ``units.csv`` row of *channel*, or ``None``."""
        return self._units.get(channel)

    def recipe(self, device_id: str) -> CardRecipe | None:
        """The ``card_recipes.csv`` row of a device: the first, in file order, of its
        class's rows whose ``under`` holds the device and whose ``devices`` pattern
        matches its id (an empty one holds and matches every device); ``None`` when
        none does (the default recipe).

        Raises:
            KeyError: No such device.
        """
        thing = self.thing(device_id, "device")
        return next(
            (
                r
                for r in self._recipes.get(thing.cls or "", [])
                if (r.under is None or device_id in self._held(r.under))
                and (r.devices is None or fnmatch.fnmatchcase(device_id, r.devices))
            ),
            None,
        )

    def strip_symbol(self, thing: Thing) -> StripSymbol | None:
        """The ``symbols.csv`` row that draws a device on a card's strip: the first of
        its class's rows whose pattern matches its id (an empty one matches any)."""
        return next(
            (
                r
                for r in self._symbols.get(thing.cls or "", [])
                if r.devices is None or fnmatch.fnmatchcase(thing.id, r.devices)
            ),
            None,
        )

    def card_sections(self, recipe: str) -> list[CardSection]:
        """The ``card_sections.csv`` rows of *recipe*, by ``order``."""
        return list(self._sections.get(recipe, ()))

    def fronts(self, recipe: str) -> list[FrontRow]:
        """The ``fronts.csv`` rows of *recipe*, by ``order``."""
        return list(self._fronts.get(recipe, ()))

    def card_rules(self) -> tuple[CardRule, ...]:
        """The ``cards.csv`` rows, in file order."""
        return self._tables.cards

    def card_channels(self, device_id: str) -> bool:
        """Whether a device has a channel its card shows: one on it or that it is an
        endpoint of (:meth:`channels`), or one a ``channel_roles.csv`` row walks to
        from it (:meth:`roles`: its parts, what reads it out or controls it)."""
        return bool(self.channels(device_id, "device")) or bool(self.roles(device_id))

    def card_drawing(self, device_id: str) -> bool:
        """Whether a device has a drawing: a ``card_recipes.csv`` row, or a
        ``pictures.csv`` row that applies to it."""
        thing = self.thing(device_id, "device")
        if self.recipe(device_id) is not None:
            return True
        return thing.cls is not None and bool(self.pictures(thing.cls, device_id))

    def card_rule(self, device_id: str) -> CardRule | None:
        """The ``cards.csv`` row that decides a device's card: the first whose class
        (``*`` matches any), place (the device's place, or its whole's, is it or
        within it), id pattern, attribute and conditions (``when``) all hold."""
        try:
            thing = self.thing(device_id, "device")
        except KeyError:
            return None
        for row in self._tables.cards:
            if row.cls not in (ANY_CLASS, thing.cls):
                continue
            if row.under:
                place = self.place_of(thing.id, "device") or ""
                if place != row.under and not place.startswith(row.under + "/"):
                    continue
            if row.devices and not fnmatch.fnmatchcase(thing.id, row.devices):
                continue
            if row.attribute is not None and row.attribute not in thing.attributes:
                continue
            if "no-channel" in (row.when or ()) and self.card_channels(thing.id):
                continue
            if "no-drawing" in (row.when or ()) and self.card_drawing(thing.id):
                continue
            return row
        return None

    def has_card(self, device_id: str) -> bool:
        """Whether a device gets a card: as its :meth:`card_rule` says, else by the
        default, a card when it has a channel its card shows or a drawing."""
        row = self.card_rule(device_id)
        if row is not None:
            return row.action == "keep"
        return self.card_channels(device_id) or self.card_drawing(device_id)

    def named_under(self, owner: str) -> list[Thing]:
        """The devices whose id is *owner* and one more ``:`` or ``.`` segment (the
        sockets of a panel named ``<panel>:A1``), in id order; indexed once."""
        if self._under_names is None:
            found: dict[str, list[Thing]] = {}
            for thing in self._things.values():
                if thing.kind != "device":
                    continue
                cut = max(thing.id.rfind(":"), thing.id.rfind("."))
                if cut > 0:
                    found.setdefault(thing.id[:cut], []).append(thing)
            self._under_names = {k: sorted(v, key=lambda t: t.id) for k, v in found.items()}
        return list(self._under_names.get(owner, ()))

    def chain(self, recipe: str, chain: str) -> list[ChainStep]:
        """The hops of a recipe's chain, by ``step``."""
        return list(self._steps.get((recipe, chain), ()))

    def hop(self, thing_id: str, kind: ThingKind, step: ChainStep) -> list[Edge]:
        """The claims a chain hop walks from a thing: those of its relation in its
        direction, superseded ones left out, whose other end its ``to`` takes; one
        per other end, an accepted claim before a contested one (a claim that several
        layers state is one end)."""
        edges = self.neighbours(thing_id, kind).get(step.relation, ())
        found: dict[str, Edge] = {}
        for edge in edges:
            if edge.outgoing is not (step.direction == "out") or edge.standing == "superseded":
                continue
            if not self._takes(step.to, edge):
                continue
            held = found.get(edge.other)
            if held is None or (held.standing == "contested" and edge.standing != "contested"):
                found[edge.other] = edge
        return [found[other] for other in sorted(found)]

    def _takes(self, to: str | None, edge: Edge) -> bool:
        if not to:
            return True
        if to in ("device", "place"):
            return edge.other_kind == to
        cls, _, pattern = to.partition(":")
        if edge.other_kind != "device":
            return False
        try:
            other = self.thing(edge.other, "device")
        except KeyError:
            return False
        return other.cls == cls and (not pattern or fnmatch.fnmatchcase(edge.other, pattern))

    def roles(self, device_id: str) -> list[RoleHit]:
        """The channels that play a role for a device, in ``channel_roles.csv``
        order, each row's channels in channel order.

        A ``device`` row matches the device's own channels and a ``parts`` row
        those of its parts (``part_of`` it, one level); every match plays the role.
        A row whose scope is a relation type (:data:`~osprey.facility.panels.tables.ROLE_SCOPES`:
        ``read_out_by``, ``controlled_by``, ``interlocked_by``, ...) walks each such
        relation of the device to a device, fills the pattern from that relation's object port
        (:func:`substitute`) and matches that device's channels: exactly one
        channel over all walked relations plays the role, more than one is
        contested and none plays it.

        Raises:
            KeyError: No such device.
        """
        thing = self.thing(device_id, "device")
        rows = self._roles_of.get(thing.cls or "", [])
        if not rows:
            return []
        edges = self.neighbours(device_id, "device")
        part_edges = [
            e for e in edges.get(PART_RELATION, ()) if not e.outgoing and e.other_kind == "device"
        ]
        parts = [e.other for e in part_edges]
        own = [c.name for c in self.channels(device_id, "device")]
        found: list[RoleHit] = []
        for role in rows:
            if role.scope == "device":
                found += [
                    RoleHit(role, c, device_id) for c in own if fnmatch.fnmatchcase(c, role.signal)
                ]
            elif role.scope == "parts":
                found += [
                    RoleHit(
                        role, c.name, part, None, _join(e for e in part_edges if e.other == part)
                    )
                    for part in dict.fromkeys(parts)
                    for c in self.channels(part, "device")
                    if fnmatch.fnmatchcase(c.name, role.signal)
                ]
            else:
                walked = [e for e in edges.get(role.scope, ()) if e.outgoing]
                hits = {
                    (c.name, e.other, e.other_port)
                    for e in walked
                    if e.outgoing and e.other_kind == "device"
                    for pattern in [substitute(role.signal, e.other_port)]
                    if pattern is not None
                    for c in self.channels(e.other, "device")
                    if fnmatch.fnmatchcase(c.name, pattern)
                }
                if len({h[0] for h in hits}) == 1:
                    name, owner, port = min(hits, key=lambda h: (h[1], h[2] or ""))
                    on = (e for e in walked if e.other == owner and e.other_port == port)
                    found.append(RoleHit(role, name, owner, port, _join(on)))
        return found

    def state_label(self, channel: str, value: object) -> str | None:
        """The word ``state_labels.csv`` states for *channel* at *value* (``1``,
        ``1.0`` and ``"1"`` are one value), or ``None``."""
        return self._labels.get((channel, _value_key(value)))

    def pair(self, setpoint: str) -> Pair | None:
        """The readback ``pairs.csv`` shows the setpoint beside, or ``None``."""
        return self._pairs.get(setpoint)


def load_graph(facility: Path, panels_dir: Path) -> PanelGraph:
    """Read a built facility file and its rule tables into a :class:`PanelGraph`.

    Args:
        facility: The facility file, or the directory holding ``facility.json``.
        panels_dir: The directory holding the rule tables.

    Returns:
        The graph.

    Raises:
        FileNotFoundError: No facility file is there.
        FacilityFileError: The facility file cannot be read.
        PanelTableError: A rule table breaks its schema.
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
    return PanelGraph(records, tables, content_hash=digest.hexdigest())
