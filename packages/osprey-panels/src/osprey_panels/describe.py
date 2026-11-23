"""What each screen contains: one pure function per view kind, graph in, view out.

:func:`targets` lists every screen a graph gets, each addressed by a
:class:`~osprey_panels.view.Target` (view kind, thing kind, id). The
view functions return :class:`~osprey_panels.view.View` objects whose
links only point at those targets; a thing with no screen is named in a
:class:`~osprey_panels.view.Fact` instead.

Which things get which screens:

- ``card``: every device that has a channel its card shows or a drawing, or that
  a ``cards.csv`` row keeps (and none it drops), every place, every group;
- ``set``: every group whose members are devices of one class;
- ``place``: every place with anything within it;
- ``rack``: every place some ``located_in`` claim names a unit in (the claim is
  data; no level or place name is written here);
- ``rollup`` (by ``place`` and by ``system``) and ``checkout``: every place
  with a rule-bearing channel within it, at any depth.

Views carry channel names, never values or states: whoever draws a view binds
the channels and asks :mod:`~osprey_panels.health` for the states.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import itertools
import math
import os
import re
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from osprey_panels.health import disagrees, rule_channels, supported, unplaced_channels
from osprey_panels.model import (
    MODEL_FRAME,
    PART_RELATION,
    PLACE_RELATION,
    CalibrationState,
    Channel,
    Citation,
    Edge,
    PanelGraph,
    RoleHit,
    Thing,
    ThingKind,
    calibration_numbers,
)
from osprey_panels.order import name_order
from osprey_panels.tables import (
    CONTROLS,
    CURVE_PREFIX,
    DRAWN_AS,
    Calibration,
    CalibrationKind,
    CardRule,
    CardSection,
    ChainStep,
    ChannelRole,
    FrontRow,
    Pose,
    RecordType,
)
from osprey_panels.view import (
    ENTRIES_IN_ONE_COLUMN,
    Band,
    BlockRow,
    CalibrationCell,
    CalibrationRow,
    Calibrations,
    CalibrationTallies,
    CalibrationTally,
    CardFoot,
    CardHead,
    CardStrip,
    Chain,
    ChainBlock,
    ChannelCount,
    ChannelGrid,
    CheckRow,
    Connector,
    Control,
    ControlGroup,
    CurvePlot,
    Elevation,
    Fact,
    FarEnd,
    Front,
    FrontGroup,
    Grid,
    GridCell,
    GridRow,
    Group,
    Item,
    KeyTile,
    Light,
    LimitLadder,
    Link,
    Node,
    Operate,
    PartFacts,
    Picture,
    Placement,
    Probe,
    ProbePart,
    RackUnit,
    RingMember,
    RingStrip,
    Section,
    Strip,
    StripElement,
    StripMember,
    Target,
    Tile,
    Totals,
    TrayEntry,
    ValueRow,
    View,
    ViewKind,
)

__all__ = [
    "GENERIC_SYMBOL",
    "NOT_PLACED",
    "NO_POSITION",
    "NO_SYSTEM",
    "PLACE_RELATION",
    "RACK_ORDER",
    "ENUMERATED_OUTPUTS",
    "action_word",
    "chain_states",
    "strip_states",
    "RELATION_TITLES",
    "CARD_SECTIONS",
    "DEFAULT_RECIPE",
    "HOP_CONTESTED",
    "HOP_FOUND",
    "HOP_NOT_STATED",
    "STRIP_NOWHERE",
    "STRIP_BY_PLACE",
    "STRIP_PLACED",
    "checkout",
    "describe",
    "channel_list",
    "device_card",
    "health_rollup",
    "light",
    "others",
    "place_view",
    "rack_elevation",
    "set_overview",
    "targets",
]

#: The id of the screens of what is in no place (:meth:`PanelGraph.unplaced`):
#: a roll-up and a checkout, addressed as a place that the graph does not hold.
NOT_PLACED = "not placed"
#: The id of the one home screen: the way in, above every top place.
HOME = "facility"
HOME_TARGET = Target("home", "place", HOME)
HOME_LINK = Link("home", HOME_TARGET)
#: The ``by`` of a home page that lists the devices of one class in no place.
UNPLACED_BY = "unplaced"
NO_CLASS = "no class stated"

#: Section titles of a card's neighbours: relation kind -> (title on the
#: subject's card, title on the object's card). A kind not listed is titled by
#: its own name on both.
RELATION_TITLES: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "powered_by": ("fed by", "feeds"),
        "controlled_by": ("controlled by", "controls"),
        "read_out_by": ("read out by", "reads out"),
        "interlocked_by": ("protected by", "protects"),
        "triggered_by": ("triggered by", "triggers"),
        "cabled_to": ("cabled to", "cabled to"),
        "networked_to": ("network", "network"),
        "part_of": ("part of", "parts"),
        "supplied_by": ("supplied by", "supplies"),
        "same_as": ("also known as", "also known as"),
        "bounded_by": ("bounded by", "bounds"),
    }
)

#: How a rack elevation lists its units: ``top-down`` is the highest unit
#: number first, as a rack is numbered from the bottom (unit 1 lowest).
RACK_ORDER = "top-down"

#: The symbol of a class the symbol mapping does not name.
GENERIC_SYMBOL = "generic"

#: The node of a by-system roll-up that holds devices with no system key.
NO_SYSTEM = "no system"


def _system(thing: Thing) -> str:
    """The system a device's roll-up by system puts it in: its class, else :data:`NO_SYSTEM`."""
    return thing.cls if thing.cls is not None else NO_SYSTEM


#: A tray entry's reason when nothing more is known than that it has no position.
NO_POSITION = "no position"
_NOT_IN = "not placed"
_NO_THING = "on no device"
_OTHERS = "{} other things in this place"
_RACKS = "racks"
_OTHER = "other"
_MORE = "+{} more"
_BY_SYSTEM = "by system"
_RING = "ring"
#: The most chips one group of a place view lists; the rest are counted in a link.
MAX_CHIPS = 24
_WITHOUT = "{} without a rule"
_OWN = "{} itself"
_DRAWN_RANK = {drawn: rank for rank, drawn in enumerate(DRAWN_AS)}
_EMPTY: Mapping[str, str] = MappingProxyType({})


# -- the index of which screens exist --------------------------------------------


@dataclass(frozen=True)
class _Index:
    """Which screens a graph gets, apart from racks (those depend on a parameter)."""

    carded: frozenset[str]  # devices with a card
    listed: frozenset[str]  # devices with more channels than a card lists: a channels view
    place_views: frozenset[str]
    sets: frozenset[str]
    ruled: frozenset[str]  # places with a rule-bearing channel in their contents
    readbacks: frozenset[str]  # every channel some setpoint is paired with
    racked: frozenset[str]  # places some located_in claim names a unit in
    unplaced_ruled: bool  # a device in no place has a rule-bearing channel
    unplaced_channels: bool  # a device in no place has a channel
    systems: Mapping[str, frozenset[str]]  # place -> systems with a rule channel in it
    # set -> its members' one class
    set_class: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    # device -> its sets
    sets_of: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    # a device with no card -> the thing whose card draws it, or None (not stated)
    owners: Mapping[str, tuple[str, ThingKind] | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    # a device whose card lists counts, not its channels, and has rule-bearing
    # channels -> the pages of its own roll-up
    paged: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    # a set over one page -> its pages
    set_pages: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    # class -> the devices in no place of that class that get a card
    loose: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))


_INDEXES: weakref.WeakKeyDictionary[PanelGraph, _Index] = weakref.WeakKeyDictionary()


def _index(graph: PanelGraph) -> _Index:
    """The graph's screen index, built once per graph in one pass over it."""
    found = _INDEXES.get(graph)
    if found is None:
        found = _INDEXES[graph] = _build_index(graph)
    return found


def _build_index(graph: PanelGraph) -> _Index:
    carded: set[str] = set()
    listed: set[str] = set()
    place_views: set[str] = set()
    sets: set[str] = set()
    ruled: set[str] = set()
    readbacks: set[str] = set()
    racked: set[str] = set()
    systems: dict[str, set[str]] = {}
    set_class: dict[str, str] = {}
    sets_of: dict[str, list[str]] = {}
    owners: dict[str, tuple[str, ThingKind] | None] = {}
    paged: dict[str, int] = {}
    set_pages: dict[str, int] = {}
    for thing in graph.things():
        if thing.kind == "group":
            members = graph.members(thing.id)
            classes = {member.cls if member.kind == "device" else None for member in members}
            # a set: a family (a section is things that belong together, never one),
            # every member a device of one class
            if thing.group_kind == "family" and len(classes) == 1 and None not in classes:
                sets.add(thing.id)
                set_class[thing.id] = next(iter(classes))  # type: ignore[assignment]
                for member in members:
                    sets_of.setdefault(member.id, []).append(thing.id)
                if len(members) > SET_PAGE:
                    pages = len(_set_pages(graph, [m for m in members if m.kind == "device"]))
                    if pages > 1:
                        set_pages[thing.id] = pages
            continue
        channels = graph.channels(thing.id, thing.kind)
        for channel in channels:
            pair = graph.pair(channel.name)
            if pair is not None:
                readbacks.add(pair.readback)
        for edge in graph.neighbours(thing.id, thing.kind).get(PLACE_RELATION, ()):
            if edge.outgoing and edge.other_kind == "place" and edge.unit is not None:
                racked.add(edge.other)
        if thing.kind == "device":
            has_card = graph.has_card(thing.id)
            if has_card:
                carded.add(thing.id)
            else:
                owners[thing.id] = _owner(graph, thing.id, graph.card_rule(thing.id))
            shown = _listed(graph, thing.id, channels)
            # over the cap, or with driver-internal records: those are read only there
            if has_card and (len(shown) > GRID_CAP or len(shown) < len(channels)):
                listed.add(thing.id)
                ruled_here = len(rule_channels(graph, thing.id, "device"))
                if ruled_here:  # no card reads them all: a roll-up of its own does
                    paged[thing.id] = -(-ruled_here // DEVICE_PAGE)
            if thing.place is not None:
                place_views.add(thing.place)
        elif thing.place is not None:
            place_views.add(thing.place)
        if any(graph.rule(channel.name) for channel in channels):
            # the places graph.contents puts it in: built, claimed, or through its whole
            claims = [
                edge.other
                for edge in graph.neighbours(thing.id, thing.kind).get(PLACE_RELATION, ())
                if edge.outgoing and edge.other_kind == "place"
            ]
            whole = graph.whole_of(thing.id) if thing.kind == "device" else None
            starts = [thing.place if thing.kind == "device" else thing.id, *claims]
            starts += list(whole[1]) if whole is not None else []
            system = _system(thing) if thing.kind == "device" else None
            for place in starts:
                while place is not None and (place not in ruled or system is not None):
                    if system is not None and system in systems.setdefault(place, set()):
                        break
                    ruled.add(place)
                    if system is not None:
                        systems[place].add(system)
                    place = graph.thing(place, "place").place
    unplaced = graph.unplaced()
    return _Index(
        frozenset(carded),
        frozenset(listed),
        frozenset(place_views),
        frozenset(sets),
        frozenset(ruled),
        frozenset(readbacks),
        frozenset(racked),
        bool(unplaced_channels(graph)),
        any(graph.channels(thing.id, "device") for thing in unplaced),
        MappingProxyType({place: frozenset(found) for place, found in systems.items()}),
        MappingProxyType(set_class),
        MappingProxyType({device: tuple(found) for device, found in sets_of.items()}),
        MappingProxyType(_drawn_owners(owners)),
        MappingProxyType(paged),
        MappingProxyType(set_pages),
        MappingProxyType(
            {
                cls: tuple(found)
                for cls, found in sorted(_unplaced_carded(graph, frozenset(carded)).items())
            }
        ),
    )


def unset_groups(graph: PanelGraph) -> list[tuple[str, tuple[str, ...]]]:
    """The families that are no set only because some members are no device of a
    stated class while the others share one: each with those members, in group
    order. A family of several classes, or a section, is no set by design."""
    found = []
    for thing in graph.things():
        if thing.kind != "group" or thing.group_kind != "family":
            continue
        members = graph.members(thing.id)
        classes = {m.cls for m in members if m.kind == "device" and m.cls is not None}
        odd = tuple(m.id for m in members if m.kind != "device" or m.cls is None)
        if len(classes) == 1 and odd:
            found.append((thing.id, odd))
    return found


def _unplaced_carded(graph: PanelGraph, carded: frozenset[str]) -> dict[str, list[str]]:
    """The devices in no place that get a card, by class (:data:`NO_CLASS` for none)."""
    found: dict[str, list[str]] = {}
    for thing in graph.unplaced():
        if thing.id in carded:
            found.setdefault(thing.cls or NO_CLASS, []).append(thing.id)
    return found


def _owner(graph: PanelGraph, device: str, row: CardRule | None) -> tuple[str, ThingKind] | None:
    """The thing a ``cards.csv`` drop row's relation names at *device* (its
    ``owner_to`` taking it, as a chain hop's ``to``): one thing, else ``None`` (none,
    or several: no owner is stated, as for a device the default leaves cardless)."""
    if row is None or row.owner_relation is None or row.owner_direction is None:
        return None
    to = row.owner_to or None
    step = ChainStep(
        DEFAULT_RECIPE, "owner", 1, 0, row.owner_relation, row.owner_direction, to, None, None,
        "owner", None, None,
    )  # fmt: skip
    found = {(e.other, e.other_kind) for e in graph.hop(device, "device", step)}
    return next(iter(found)) if len(found) == 1 else None


def _drawn_owners(
    owners: Mapping[str, tuple[str, ThingKind] | None],
) -> dict[str, tuple[str, ThingKind] | None]:
    """Each no-card device's owner, followed past owners that have no card either."""
    out: dict[str, tuple[str, ThingKind] | None] = {}
    for device, owner in owners.items():
        seen = {device}
        while owner is not None and owner[1] == "device" and owner[0] in owners:
            if owner[0] in seen:
                owner = None
                break
            seen.add(owner[0])
            owner = owners[owner[0]]
        out[device] = owner
    return out


def owner_link(index: _Index, thing_id: str) -> Link | None:
    """For a device with no card and an owner, a link named by it to its owner's card."""
    owner = index.owners.get(thing_id)
    if owner is None:
        return None
    return Link(thing_id, Target("card", owner[1], owner[0]))


def drawn_on(graph: PanelGraph) -> list[tuple[str, Target]]:
    """Each device with no card and an owner, with its owner's card, in
    id order; one with no owner stated gets no entry."""
    index = _index(graph)
    return [
        (thing, Target("card", owner[1], owner[0]))
        for thing, owner in sorted(index.owners.items())
        if owner is not None
    ]


def card_rule_counts(graph: PanelGraph) -> list[tuple[CardRule, int]]:
    """Each ``cards.csv`` row, in file order, with how many devices it decides (a
    device is decided by the first row that selects it)."""
    decided = Counter(
        graph.card_rule(thing.id) for thing in graph.things() if thing.kind == "device"
    )
    return [(row, decided[row]) for row in graph.card_rules()]


def card_counts(graph: PanelGraph) -> tuple[int, int, int]:
    """How many devices get no card, how many of them have no owner stated, and how
    many get none by the default alone (no ``cards.csv`` row selects them)."""
    index = _index(graph)
    by_default = sum(graph.card_rule(device) is None for device in index.owners)
    ownerless = sum(owner is None for owner in index.owners.values())
    return len(index.owners), ownerless, by_default


def _existing(graph: PanelGraph, index: _Index, target: Target) -> Target | None:
    """*target* when it is a screen of the graph, else ``None``."""
    return target if _has(graph, index, target) else None


def _applies_in(graph: PanelGraph, kind: CalibrationKind, group_id: str) -> bool:
    """Whether *kind* is a calibration kind of a device member of the set."""
    return any(m.kind == "device" and graph.applies(kind, m.id) for m in graph.members(group_id))


def _has(graph: PanelGraph, index: _Index, target: Target) -> bool:
    """Whether *target* is a screen of the graph."""
    view, kind, thing_id = target.view, target.kind, target.id
    if view == "home":
        if target.by == UNPLACED_BY:
            return (kind, thing_id) == ("place", HOME) and target.system in index.loose
        return target == HOME_TARGET
    if (kind, thing_id) == ("place", NOT_PLACED) and not _holds(graph, thing_id, "place"):
        if view == "rollup":
            return target.by == "place" and index.unplaced_ruled
        return view == "checkout" and index.unplaced_channels
    if view == "card":
        return thing_id in index.carded if kind == "device" else _holds(graph, thing_id, kind)
    if view == "channels":
        return kind == "device" and thing_id in index.listed
    if view == "set":
        pages = index.set_pages.get(thing_id, 1)
        page_ok = target.system is None or (
            target.system.isdigit() and 2 <= int(target.system) <= pages
        )
        return kind == "group" and thing_id in index.sets and page_ok
    if view == "calibration":
        cls = index.set_class.get(thing_id) if kind == "group" else None
        return cls is not None and any(
            found.kind == target.system and _applies_in(graph, found, thing_id)
            for found in graph.calibration_kinds(cls)
        )
    if view == "rollup" and kind == "device":
        pages = index.paged.get(thing_id, 0)
        return target.by == "device" and (
            target.system is None
            and pages > 0
            or target.system is not None
            and target.system.isdigit()
            and 2 <= int(target.system) <= pages
        )
    if kind != "place":
        return False
    if view == "place":
        return thing_id in index.place_views
    if view == "rack":
        return thing_id in index.racked
    if view == "rollup":
        if target.system is not None:
            return target.by == "system" and target.system in index.systems.get(thing_id, ())
        return target.by in ("place", "system") and thing_id in index.ruled
    return view == "checkout" and thing_id in index.ruled


def _holds(graph: PanelGraph, thing_id: str, kind: ThingKind) -> bool:
    try:
        graph.thing(thing_id, kind)
    except KeyError:
        return False
    return True


def targets(graph: PanelGraph) -> list[Target]:
    """Every screen the graph gets, by thing in graph order, then by view kind.

    Args:
        graph: The panel graph.

    Returns:
        The screens' addresses, each once.
    """
    index = _index(graph)
    found: list[Target] = [HOME_TARGET, *(_unplaced_page(cls) for cls in index.loose)]
    for thing in graph.things():
        candidates = [Target("card", thing.kind, thing.id)]
        if thing.kind == "device":
            candidates.append(Target("channels", "device", thing.id))
            candidates += device_rollup_pages(index, thing.id)
        if thing.kind == "group":
            candidates += set_pages(index, thing.id)
            cls = index.set_class.get(thing.id)
            if cls is not None:
                candidates += [
                    Target("calibration", "group", thing.id, system=found.kind)
                    for found in graph.calibration_kinds(cls)
                    if _applies_in(graph, found, thing.id)
                ]
        elif thing.kind == "place":
            candidates += [
                Target("place", "place", thing.id),
                Target("rack", "place", thing.id),
                Target("rollup", "place", thing.id, by="place"),
                Target("rollup", "place", thing.id, by="system"),
                *(
                    Target("rollup", "place", thing.id, by="system", system=system)
                    for system in sorted(index.systems.get(thing.id, ()))
                ),
                Target("checkout", "place", thing.id),
            ]
        found += [target for target in candidates if _has(graph, index, target)]
    for target in _UNPLACED_TARGETS:
        if _has(graph, index, target):
            found.append(target)
    return found


_UNPLACED_ROLLUP = Target("rollup", "place", NOT_PLACED, by="place")
_UNPLACED_CHECKOUT = Target("checkout", "place", NOT_PLACED)
_UNPLACED_TARGETS = (_UNPLACED_ROLLUP, _UNPLACED_CHECKOUT)


# -- shared pieces ---------------------------------------------------------------


def _ref(graph: PanelGraph, index: _Index, thing: Thing) -> Link | Fact:
    """A link to the thing's card, or a fact naming it when it has none."""
    target = Target("card", thing.kind, thing.id)
    if _has(graph, index, target):
        return Link(thing.id, target)
    owned = owner_link(index, thing.id)
    if owned is not None and _has(graph, index, owned.target):
        return owned  # drawn on its owner's card
    return Fact(thing.cls or thing.kind, thing.id)


def _place_link(
    graph: PanelGraph, index: _Index, place_id: str, prefer: Target | None = None
) -> Link:
    """A link to *prefer* (a view of the place) when it exists, else to the place's card."""
    if prefer is not None and _has(graph, index, prefer):
        return Link(place_id, prefer)
    return Link(place_id, Target("card", "place", place_id))


def _crumbs(
    graph: PanelGraph,
    index: _Index,
    place_id: str | None,
    view: ViewKind = "card",
    by: str | None = None,
) -> tuple[Link, ...]:
    """The home screen, then links to *place_id* and every place around it,
    outermost first."""
    chain: list[str] = []
    while place_id is not None:
        chain.append(place_id)
        place_id = graph.thing(place_id, "place").place
    return (HOME_LINK,) + tuple(
        _place_link(graph, index, place, Target(view, "place", place, by=by))
        for place in reversed(chain)
    )


def _label(channel: Channel) -> str:
    """What a value row reads: the signal the facility states for the channel, else its name."""
    return channel.signal.replace("_", " ") if channel.signal else channel.name


def _value_row(graph: PanelGraph, channel: Channel) -> ValueRow:
    pair = graph.pair(channel.name) if channel.writable else None
    return ValueRow(
        label=_label(channel),
        channel=channel.name,
        readback=pair.readback if pair is not None else None,
        tolerance=channel.tolerance,
        writable=channel.writable,
        drawn_as=channel.drawn_as,
        inferred=pair is not None and pair.basis == "inferred",
        unit=channel.unit,
        readback_unit=graph.channel(pair.readback).unit if pair is not None else None,
    )


def light(graph: PanelGraph, channel: str) -> Light:
    """The light of a rule-bearing channel, with what its rows say about it."""
    rows = graph.rule(channel)
    return Light(
        channel=channel,
        inferred=bool(rows) and all(row.basis == "inferred" for row in rows),
        disagree=disagrees(graph, channel),
        unsupported=any(not supported(row.value) for row in rows),
    )


def _citation_text(citation: Citation) -> str:
    parts = [
        f"version {citation.version}" if citation.version else None,
        citation.state,
        citation.locator,
    ]
    return ", ".join(part for part in parts if part) or citation.document


def _number(value: float) -> str:
    return f"{value:g}"


#: The roles of a shared supply a member's card shows: its values, never its presses.
_SHARED_SHOWN = frozenset({"value", "entry", "word", "lamp"})
_SHARED_TITLE = "shared supply {}"
_SHARED_LABEL = "{} (shared)"


def _shared(graph: PanelGraph, index: _Index, thing: Thing) -> list[tuple[str, list[Item]]]:
    """For a device that is an endpoint of channels another device holds (a family's
    shared supply), one section per such holder: a link to the holder's card, then
    the holder's roles among those channels, each read-only and labelled as shared.
    The holder's presses stay on its own card."""
    if thing.kind != "device":
        return []
    held: dict[str, set[str]] = {}
    for channel in graph.channels(thing.id, "device"):
        if channel.thing_kind == "device" and channel.thing not in (None, thing.id):
            held.setdefault(channel.thing, set()).add(channel.name)  # type: ignore[arg-type]
    out: list[tuple[str, list[Item]]] = []
    for holder, names in held.items():
        rows: list[Item] = []
        seen: set[str] = set()
        for hit in graph.roles(holder):
            if hit.channel not in names or hit.channel in seen:
                continue
            if hit.role.shown_as not in _SHARED_SHOWN:
                continue
            seen.add(hit.channel)
            row = _value_row(graph, graph.channel(hit.channel))
            rows.append(
                dataclasses.replace(
                    row, label=_SHARED_LABEL.format(_row_word(hit.role)), writable=False
                )
            )
        if not rows:
            continue
        ref = _ref(graph, index, graph.thing(holder, "device"))
        out.append((_SHARED_TITLE.format(holder), [ref, *rows]))
    return out


_OTHER_FRAME = "placed along another beam path ({})"


def _beam_of(graph: PanelGraph, thing: Thing) -> tuple[str, float] | None:
    """A device's ``(frame, s)`` (:meth:`PanelGraph.beam_position`), else, for a part
    placed by its whole, the whole's."""
    found = graph.beam_position(thing)
    if found is not None:
        return found
    whole = graph.whole_of(thing.id)
    return graph.beam_position(graph.thing(whole[0], "device")) if whole is not None else None


def _unwrapped(positions: Mapping[str, float], length: float | None) -> dict[str, float]:
    """*positions* along one beam path, read on past the end of a ring: when a gap
    between two of them is over half the ring's *length*, the stretch crosses the
    ring's start, and those before the gap are moved one length on, so it is drawn
    in one piece. A whole ring has no such gap and a line (``length`` ``None``) is
    drawn as stated."""
    out = dict(positions)
    if not length or len(out) < 2:
        return out
    ordered = sorted(out.values())
    gap, before = max((b - a, a) for a, b in itertools.pairwise(ordered))
    if gap > length / 2:
        for key, s_m in out.items():
            if s_m <= before:
                out[key] = s_m + length
    return out


def _by_whole(graph: PanelGraph, thing: Thing) -> tuple[str, str] | None:
    """``(place, whole)`` for a device placed only by its whole, else ``None``."""
    whole = graph.whole_of(thing.id) if thing.kind == "device" else None
    return (whole[1][0], whole[0]) if whole is not None else None


def _pose_of(graph: PanelGraph, thing: Thing) -> Pose | None:
    """A device's pose, else, for a part placed by its whole (:meth:`PanelGraph.whole_of`),
    the whole's: a part is where its whole is."""
    pose = graph.pose(thing.id)
    if pose is not None or thing.kind != "device":
        return pose
    whole = graph.whole_of(thing.id)
    return graph.pose(whole[0]) if whole is not None else None


def _beam_s(graph: PanelGraph, thing: Thing) -> float | None:
    """The thing's position along its beam path (:meth:`PanelGraph.beam_position`)."""
    found = graph.beam_position(thing)
    return found[1] if found is not None else None


def _pose_text(pose: Pose) -> str:
    parts = []
    if pose.s_m is not None:
        parts.append(f"s {pose.s_m:.3f} m")
    xyz = [
        f"{axis} {_number(value)}"
        for axis, value in (("x", pose.x_mm), ("y", pose.y_mm), ("z", pose.z_mm))
        if value is not None
    ]
    if xyz:
        parts.append(" ".join(xyz) + " mm")
    if pose.side_mm is not None:
        parts.append(f"side {_number(pose.side_mm)} mm")
    if pose.height_mm is not None:
        parts.append(f"height {_number(pose.height_mm)} mm")
    return " · ".join(parts) or "no coordinates"


def _pose_fact(pose: Pose) -> Fact:
    citation = (
        Citation(pose.document, pose.version, pose.state, pose.locator)
        if pose.document is not None
        else None
    )
    return Fact(
        pose.frame,
        _pose_text(pose),
        citation=citation,
        caveat=pose.derived_assumes or None,
        basis=pose.basis,
    )


def _sections(pairs: Iterable[tuple[str, list[Item]]]) -> tuple[Section, ...]:
    """Sections from (title, items), each empty one left out."""
    return tuple(Section(title, tuple(items)) for title, items in pairs if items)


# -- device card -----------------------------------------------------------------


def _signals(
    graph: PanelGraph, index: _Index, channels: list[Channel]
) -> tuple[list[Item], list[Item]]:
    """The card's signals (pairs, then readable channels by how they are drawn) and settings."""
    paired: list[Item] = []
    settings: list[Item] = []
    readable: list[Channel] = []
    for channel in channels:
        if channel.writable:
            row = _value_row(graph, channel)
            (paired if row.readback is not None else settings).append(row)
        elif channel.name not in index.readbacks:
            readable.append(channel)
    readable.sort(key=lambda channel: _DRAWN_RANK[channel.drawn_as or "untyped"])
    return paired + [_value_row(graph, channel) for channel in readable], settings


def _neighbours(graph: PanelGraph, index: _Index, thing: Thing) -> list[tuple[str, list[Item]]]:
    """One section per relation kind and direction present; equal titles share one."""
    edges = graph.neighbours(thing.id, thing.kind)
    edges.pop(PLACE_RELATION, None)  # shown under "where" and "within"
    kinds = [kind for kind in RELATION_TITLES if kind in edges]
    kinds += sorted(kind for kind in edges if kind not in RELATION_TITLES)
    found: dict[str, list[Item]] = {}
    for kind in kinds:
        subject_title, object_title = RELATION_TITLES.get(kind, (kind, kind))
        for outgoing, title in ((True, subject_title), (False, object_title)):
            for edge in edges[kind]:
                if edge.outgoing is outgoing:
                    found.setdefault(title, []).append(_edge_item(graph, index, title, edge))
    return list(found.items())


def _edge_item(graph: PanelGraph, index: _Index, title: str, edge: Edge) -> Item:
    target = Target("card", edge.other_kind, edge.other)
    if _has(graph, index, target):
        return Link(edge.other, target, standing=edge.standing, basis=edge.basis)
    owned = owner_link(index, edge.other)
    if owned is not None and _has(graph, index, owned.target):
        return dataclasses.replace(owned, standing=edge.standing, basis=edge.basis)
    return Fact(title, edge.other, citation=edge.source, standing=edge.standing, basis=edge.basis)


def _placement(graph: PanelGraph, index: _Index, edge: Edge) -> Item:
    """A ``located_in`` claim as a placement: its rack view when there is one, else a card."""
    target = Target("card", edge.other_kind, edge.other)
    rack = Target("rack", "place", edge.other)
    if edge.other_kind == "place" and _has(graph, index, rack):
        target = rack
    if not _has(graph, index, target):
        return Fact(
            PLACE_RELATION,
            edge.other,
            citation=edge.source,
            standing=edge.standing,
            basis=edge.basis,
        )
    return Placement(
        Link(edge.other, target, standing=edge.standing, basis=edge.basis),
        edge.unit,
        _size(edge.attributes.get("size_u")),
        edge.source,
    )


def _where(graph: PanelGraph, index: _Index, thing: Thing) -> list[Item]:
    """The built place, every ``located_in`` claim of the thing, then its poses.

    A device with neither is in the place of its nearest placed whole: that
    line reads "in <place>, as part of <whole>", is inferred, and links to the
    whole.
    """
    items: list[Item] = []
    if thing.place is not None:
        label = "place" if thing.kind == "device" else "parent"
        items.append(Link(label, Target("card", "place", thing.place)))
    whole = graph.whole_of(thing.id) if thing.kind == "device" else None
    if whole is not None:
        text = f"in {', '.join(whole[1])}, as part of {whole[0]}"
        target = Target("card", "device", whole[0])
        if _has(graph, index, target):
            items.append(Link(text, target, basis="inferred"))
        else:
            items.append(Fact("where", text, basis="inferred"))
    if thing.kind != "group":
        claims = graph.neighbours(thing.id, thing.kind).get(PLACE_RELATION, ())
        items += [_placement(graph, index, e) for e in claims if e.outgoing]
        items += [_pose_fact(pose) for pose in graph.poses(thing.id)]
    if thing.kind == "device":
        items += _beam_fact(graph, thing)
    return items


_BEAM_TEXT = "s {:.3f} m along {}, from {}"
_BEAM_OTHER = "the facility file states s {:.3f} m along {}"
#: How far apart two stated positions of one device may be before the card says so, m.
BEAM_AGREE_M = 0.01


def _beam_fact(graph: PanelGraph, thing: Thing) -> list[Item]:
    """Which position along the beam the device's strips use, and where it is read
    from: its pose in its model's beam frame, else the facility file's ``s``, else
    its whole's; with the other stated one beside it when the two differ."""
    pose = graph.pose(thing.id)
    found = graph.beam_position(thing)
    by = None
    if found is None:
        whole = graph.whole_of(thing.id)
        found = _beam_of(graph, thing)
        by = whole[0] if whole is not None and found is not None else None
    if found is None:
        return []
    frame, s_m = found
    if by is not None:
        source = f"its whole {by}"
    elif pose is not None and pose.is_beam_position:
        source = "its pose in positions.csv"
    else:
        source = "the facility file"
    caveat = None
    posed = source.startswith("its pose") and thing.model is not None
    if posed and thing.s_m is not None and abs(thing.s_m - s_m) > BEAM_AGREE_M:
        caveat = _BEAM_OTHER.format(thing.s_m, MODEL_FRAME.format(thing.model))
    return [Fact("beam position", _BEAM_TEXT.format(s_m, frame, source), caveat=caveat)]


def _within(graph: PanelGraph, index: _Index, place: Thing) -> list[Item]:
    """A place's direct contents, then what a ``located_in`` claim puts in it besides,
    then the parts in it only as part of a whole (inferred)."""
    children = graph.within(place.id)
    items: list[Item] = [_ref(graph, index, child) for child in children]
    seen = {(child.kind, child.id) for child in children}
    for edge in graph.neighbours(place.id, "place").get(PLACE_RELATION, ()):
        key = (edge.other_kind, edge.other)
        if edge.outgoing or key in seen:
            continue
        seen.add(key)
        ref = _ref(graph, index, graph.thing(edge.other, edge.other_kind))
        if isinstance(ref, Link):
            items.append(Link(ref.label, ref.target, standing=edge.standing, basis=edge.basis))
        else:
            items.append(dataclasses.replace(ref, standing=edge.standing, basis=edge.basis))
    for part in graph.derived(place.id):
        if (part.kind, part.id) in seen:
            continue
        seen.add((part.kind, part.id))
        ref = _ref(graph, index, part)
        items.append(dataclasses.replace(ref, basis="inferred"))
    return items


def _pictures(graph: PanelGraph, thing: Thing) -> list[Item]:
    """A device's drawing per view of its class, each row with the channel its
    ``signal`` resolves to on this device (:meth:`PanelGraph.picture_channel`)."""
    if thing.kind != "device" or thing.cls is None:
        return []
    views: dict[str, list] = {}
    for part in graph.pictures(thing.cls, thing.id):
        if part.signal is not None:
            part = dataclasses.replace(part, channel=graph.picture_channel(thing.id, part.signal))
        views.setdefault(part.view, []).append(part)
    found = graph.strip_symbol(thing)
    symbol = found.symbol if found is not None else None
    return [Picture(thing.cls, view, tuple(parts), symbol) for view, parts in views.items()]


#: The section a pickup's cross-section is drawn in.
PROBE_SECTION = "pickup, looking downstream"
#: The ``state`` of an anchor row whose placement the data infers.
_INFERRED_STATE = "inferred"


def _probe(graph: PanelGraph, thing: Thing) -> list[Item]:
    """A device's probe picture, each element with the channels of its signals; or nothing."""
    if thing.kind != "device":
        return []
    rows = graph.probe(thing.id)
    if not rows:
        return []

    def channel(signal: str | None) -> str | None:
        return graph.probe_channel(thing.id, signal) if signal is not None else None

    parts = tuple(ProbePart(row, channel(row.signal), channel(row.signal_y)) for row in rows)
    inferred = any(r.shape == "anchor" and r.state == _INFERRED_STATE for r in rows)
    return [Probe(thing.cls or "", parts, inferred)]


_SCREEN_TITLES = {
    ("place", None): "place view",
    ("rack", None): "rack elevation",
    ("rollup", "place"): "health by place",
    ("rollup", "system"): "health by system",
    ("checkout", None): "checkout",
    ("set", None): "set overview",
}


def _screens(graph: PanelGraph, index: _Index, thing: Thing) -> list[Item]:
    """Links to the thing's other screens."""
    if thing.kind == "group":
        candidates = [Target("set", "group", thing.id)]
    elif thing.kind == "place":
        candidates = [
            Target("place", "place", thing.id),
            Target("rack", "place", thing.id),
            Target("rollup", "place", thing.id, by="place"),
            Target("rollup", "place", thing.id, by="system"),
            Target("checkout", "place", thing.id),
        ]
    else:
        return []
    return [
        Link(_SCREEN_TITLES[(target.view, target.by)], target)
        for target in candidates
        if _has(graph, index, target)
    ]


def _sources(thing: Thing) -> list[Item]:
    return [
        Fact(citation.document, _citation_text(citation), citation=citation)
        for citation in thing.citations
    ]


#: The most documents the sources line names; the rest are counted.
FOOT_DOCUMENTS = 4


def _card_foot(thing: Thing, where: list[Item], sets: tuple[Link, ...] = ()) -> CardFoot:
    """Where the device is and what it is read from, one line each.

    The where line: each place a ``located_in`` claim names with its unit or units,
    opening that place's rack view (two places read "<a> or <b>", and a contested
    claim says so), else the place, opening its card; then "in <place>, as part of
    <whole>" for a part, then its position per frame. The sources line: the count of
    statements (its own citations, its placement and pose claims) and the documents
    they cite; each statement's full provenance is a line of the hover text."""
    segments: list[tuple[str, Link | None]] = []
    citations: list[Citation] = list(thing.citations)
    places: dict[Target, tuple[Link, list[str]]] = {}
    contested = False
    for item in where:
        if isinstance(item, Placement):
            link, units = places.setdefault(item.link.target, (item.link, []))
            if item.unit is not None and str(item.unit) not in units:
                units.append(str(item.unit))
            contested |= item.link.standing == "contested"
        if isinstance(item, Placement | Fact) and item.citation is not None:
            citations.append(item.citation)
    for n, (link, units) in enumerate(places.values()):
        word = "unit" if len(units) == 1 else "units"
        text = f"{link.target.id} {word} {', '.join(units)}" if units else link.target.id
        segments.append((("or " if n else "") + text, link))
    if not places:
        segments += [(i.target.id, i) for i in where if isinstance(i, Link)][:1]
    frames: set[str] = set()
    for item in where:
        if isinstance(item, Link) and item.basis == "inferred":
            segments.append((item.label, item))
        elif isinstance(item, Fact) and item.label == "where":
            segments.append((item.text, None))
        elif isinstance(item, Fact) and item.label not in frames:
            frames.add(item.label)
            segments.append((f"{item.label} {item.text}", None))
    if contested:
        segments.append((_CONTESTED_WORD, None))
    if not segments:
        segments.append((_NOT_STATED, None))
    documents = list(dict.fromkeys(c.document for c in citations))
    named = ", ".join(documents[:FOOT_DOCUMENTS])
    if len(documents) > FOOT_DOCUMENTS:
        named += f" and {len(documents) - FOOT_DOCUMENTS} more"
    noun = "statement" if len(citations) == 1 else "statements"
    source_noun = "source" if len(documents) == 1 else "sources"
    sources = (
        f"{len(citations)} {noun} from {len(documents)} {source_noun}: {named}"
        if citations
        else "no source stated"
    )
    provenance = tuple(f"{c.document}: {_citation_text(c)}" for c in citations)
    return CardFoot(tuple(segments), sources, provenance, tuple(where), tuple(citations), sets)


def device_card(
    graph: PanelGraph,
    thing_id: str,
    kind: ThingKind | None = None,
) -> View:
    """The card of a device, a place or a group.

    Sections, each left out when empty: ``screens`` (a place's or group's
    other screens), ``within`` (a place's direct contents, then what a
    ``located_in`` claim puts in it besides, with the claim's standing),
    ``members`` (a group's), ``signals``, ``settings``, ``health``, one per
    relation kind and direction (titled from :data:`RELATION_TITLES`;
    ``located_in`` is never one), ``picture``, ``where`` (the built place,
    a :class:`Placement` per ``located_in`` claim, the poses), ``sources``.

    Args:
        graph: The panel graph.
        thing_id: The thing.
        kind: Picks between things that share an id.

    Raises:
        KeyError: No such thing, or it is excluded.
    """
    thing = graph.thing(thing_id, kind)
    index = _index(graph)
    if thing.kind == "device":
        extra = [
            (PROBE_SECTION, _probe(graph, thing)),
            ("picture", _pictures(graph, thing)),
            ("calibration", _calibration(graph, index, thing)),
        ]
        # up to the place's view where it has one: the way back to where it sits
        crumbs = _crumbs(graph, index, graph.place_of(thing.id, "device"), "place")
        sets = tuple(
            Link(group, Target("set", "group", group)) for group in index.sets_of.get(thing.id, ())
        )
        foot = _card_foot(thing, _where(graph, index, thing), sets)
        sections = _drawn_card(graph, index, thing, extra, [("foot", [foot])])
        return View(
            Target("card", "device", thing.id), thing.id, crumbs, sections, alias=thing.short_name
        )
    channels = graph.channels(thing.id, thing.kind)
    signals, settings = _signals(graph, index, channels)
    within: list[Item] = []
    members: list[Item] = []
    if thing.kind == "place":
        within = _within(graph, index, thing)
    elif thing.kind == "group":
        members = [_ref(graph, index, member) for member in graph.members(thing.id)]
    ruled = [channel.name for channel in channels if graph.rule(channel.name)]
    health: list[Item] = [light(graph, name) for name in ruled[:HEALTH_CAP]]
    if len(ruled) > HEALTH_CAP:  # past one screen's rules: the roll-up reads them all
        rest = Target("rollup", "place", thing.id, by="place")
        more = _MORE_LIGHTS.format(len(ruled) - HEALTH_CAP)
        health.append(Link(more, rest) if _has(graph, index, rest) else Fact("health", more))
    sources = _sources(thing)
    sections = _sections(
        [
            ("screens", _screens(graph, index, thing)),
            ("within", within),
            ("members", members),
            ("signals", signals),
            ("settings", settings),
            ("health", health),
            *_neighbours(graph, index, thing),
            (PROBE_SECTION, _probe(graph, thing)),
            ("calibration", _calibration(graph, index, thing)),
            ("picture", _pictures(graph, thing)),
            ("where", _where(graph, index, thing)),
            ("sources", sources),
        ]
    )
    crumbs = (
        _crumbs(graph, index, graph.place_of(thing.id, thing.kind))
        if thing.kind != "group"
        else (HOME_LINK,)
    )
    target = Target("card", thing.kind, thing.id)
    return View(target, thing.id, crumbs, sections, alias=thing.short_name)


# -- the drawn card --------------------------------------------------------------

#: The section titles of a drawn card that are no numbered drawing: the title line,
#: the key tiles, the operate block and the all-channels grid.
CARD_SECTIONS: Mapping[str, str] = MappingProxyType(
    {
        "head": "head",
        "tiles": "tiles",
        "relations": "relations",
        "operate": "operate",
        "channels": "all channels",
    }
)
#: The most key tiles a card shows, and how many the default recipe takes from roles.
MAX_TILES = 5
DEFAULT_TILES = 4
#: The default recipe's chain: the relations of the device, each a block (a dashed
#: "not stated" one where the device has none), with the word its block carries
#: and the word on the arrow from the device's side.
DEFAULT_CHAIN: tuple[tuple[str, str, str], ...] = (
    (PLACE_RELATION, "place", "holds"),
    ("powered_by", "power", "feeds"),
    ("networked_to", "network", "network"),
    ("controlled_by", "controller", "controlled by"),
)
#: Relations the default chain never draws as blocks: a name claim is no hop.
_NO_BLOCK = frozenset({"same_as"})
_NOT_STATED = "not stated"
_CHAIN_FOOT = (
    "The chain is drawn from the device's place, power, network and controller; any "
    "other relation the device has is one more block."
)
_DEFAULT_COLUMNS = 4
FOCUS_ROWS = 4  # the most value roles a chain's device block shows
_CONTESTED_WORD = "contested"
#: The most channels a card lists in its grid; a device with more gets a ``channels``
#: view, and its card the counts and a link to it. The densest mock grid that reads.
GRID_CAP = 60
#: The role ``channel_roles.csv`` gives a driver's own records (a private signal, a
#: record field, a driver's diagnostics): never offered on a card, nor listed in its grid.
INTERNAL_ROLE = "internal"
_NO_ACTIONS = "no operator actions stated"
#: The most lights a place's or group's card draws for its own rule-bearing channels;
#: the rest are counted, opening the roll-up. One light is a few rules; this many keep
#: a card well inside one screen's rule budget.
HEALTH_CAP = GRID_CAP
_MORE_LIGHTS = "{} more with a health rule, in the roll-up"
_INTERNAL_COUNT = "{} driver-internal"
#: The ``many`` words of ``chains.csv`` that say a set of things is expected.
MANY_WORDS = frozenset({"yes", "1"})
#: Record types whose setpoint is a whole number: an output of two or more states.
ENUMERATED_OUTPUTS = frozenset({"bo", "mbbo"})
_DEFAULT_PRESS = 1.0  # what a command a record types writes, with no role saying otherwise


def _filled(template: str, graph: PanelGraph, thing: Thing) -> str:
    """A recipe's sentence with each ``{relation}`` slot the first thing the device's
    claims of that relation name (outgoing), else "not stated"."""
    edges = graph.neighbours(thing.id, thing.kind)

    def slot(name: str) -> str:
        found = [e.other for e in edges.get(name, ()) if e.outgoing]
        if name == PLACE_RELATION and not found and thing.place is not None:
            found = [thing.place]
        return found[0] if found else _NOT_STATED

    out, rest = [], template
    while "{" in rest and "}" in rest.split("{", 1)[1]:
        before, after = rest.split("{", 1)
        name, rest = after.split("}", 1)
        out.append(before + slot(name.strip()))
    out.append(rest)
    return "".join(out)


def _words(graph: PanelGraph, channel: str) -> tuple[tuple[str, str], ...]:
    return tuple(graph.state_words(channel).items())


def _role_unit(graph: PanelGraph, hit: RoleHit) -> tuple[str | None, float | None]:
    """A role's channel's unit and divisor: the role's own where it states them, else
    ``units.csv``'s, else the facility file's unit."""
    stated = graph.unit(hit.channel)
    unit = hit.role.unit or (stated.unit if stated is not None else None)
    scale = hit.role.scale or (stated.scale if stated is not None else None)
    if unit is None:
        try:
            unit = graph.channel(hit.channel).unit
        except KeyError:
            unit = None
    return unit, scale


def _role_title(role: str) -> str:
    return role.replace("-", " ").replace("_", " ").capitalize()


def _role_word(row: ChannelRole) -> str:
    """What a card calls a role row's channel, on a tile or over a group: the row's
    ``label`` as stated, else its signal's ``title``, else the role's name as a
    title. A choice's label is the word of its own segment, so a choice is called
    by its role."""
    if row.label and row.shown_as != "choice":
        return row.label
    return row.title or _role_title(row.role)


def _plate_word(row: ChannelRole) -> str:
    """A nameplate role's name in the device's block: its ``label`` as stated, else
    its signal's ``title``, else the quantity its signal names (``model`` for
    ``model_readback``)."""
    if row.label or row.title:
        return row.label or row.title or ""
    head, _, word = row.role.rpartition("_")
    return _role_title(head if head and word in ("readback", "status") else row.role).lower()


def _row_word(row: ChannelRole) -> str:
    """A role row's name on a value row of a block: its ``label`` as stated, else
    its signal's ``title`` as stated, else the role's name in lower case."""
    if row.label and row.shown_as != "choice":
        return row.label
    return row.title or _role_title(row.role).lower()


#: A card key number's measures over every channel of its role on the device.
_OVER_ALL = ("count_true", "max", "sum")


def _key_tiles(graph: PanelGraph, thing: Thing, hits: list[RoleHit]) -> list[KeyTile]:
    """The tiles: the ``metrics.csv`` rows of view ``card`` of the device's class
    whose ``level`` is its recipe or empty, by order, each
    the first channel of its role (``signal``) or not served yet; with none, the
    first :data:`DEFAULT_TILES` roles that exactly one of the device's own
    channels plays and that show a value, a word or a lamp, identity roles left
    out."""
    by_role: dict[str, list[RoleHit]] = {}
    for hit in hits:
        by_role.setdefault(hit.role.role, []).append(hit)
    recipe = graph.recipe(thing.id)
    name = recipe.recipe if recipe is not None else None
    rows = [
        m
        for m in graph.metrics("card", name)
        if m.cls == thing.cls and (m.level is None or m.level == name)
    ][:MAX_TILES]
    tiles: list[KeyTile] = []
    for row in rows:
        found = by_role.get(row.signal, [])
        if not found:
            tiles.append(KeyTile(row.title, None, row.measure, row.unit))
            continue
        hit = found[0]
        unit, scale = _role_unit(graph, hit)
        if row.measure in _OVER_ALL:  # over every channel of the role on the device
            own = [h for h in found if h.owner == thing.id] or found
            members = tuple(dict.fromkeys(h.channel for h in own))
            counted = row.measure == "count_true"
            tiles.append(
                KeyTile(
                    row.title,
                    hit.channel,
                    row.measure,
                    None if counted else row.unit or unit,
                    None if counted else scale,
                    row.precision,
                    channels=members,
                )
            )
            continue
        shown = "word" if row.measure == "word" else "value"
        if shown == "word" and hit.role.shown_as == "lamp":
            shown = "lamp"
        tiles.append(
            KeyTile(
                row.title,
                hit.channel,
                shown,
                row.unit or unit,
                scale,
                row.precision,
                _words(graph, hit.channel),
            )
        )
    if rows:
        return tiles
    for role, found in by_role.items():
        if len(tiles) == DEFAULT_TILES:
            break
        # Only the device's own channels of a role count: a neighbour's channel
        # its roles reach is no figure of this device, and two own channels of
        # one role are no single figure either.
        own = [h for h in found if h.owner == thing.id]
        if len(own) != 1:
            continue
        hit = own[0]
        first = hit.role
        if role in graph.identity or role == INTERNAL_ROLE or first.shown_as in CONTROLS:
            continue
        unit, scale = _role_unit(graph, hit)
        tiles.append(
            KeyTile(
                _role_word(first),
                hit.channel,
                first.shown_as,
                unit,
                scale,
                None,
                _words(graph, hit.channel),
            )
        )
    return tiles


def _relation_block(
    graph: PanelGraph,
    index: _Index,
    edge: Edge,
    word: str,
    arrow: str | None,
    contested: bool = False,
) -> ChainBlock:
    """A block for the thing one claim of the device names."""
    item = _edge_item(graph, index, word, edge)
    link = item if isinstance(item, Link) else None
    try:
        other = graph.thing(edge.other, edge.other_kind)
        kind = other.cls or other.level or edge.other_kind
    except KeyError:
        kind = edge.other_kind
    rows: list[BlockRow] = []
    if edge.kind == PLACE_RELATION:
        unit = edge.unit_text or (str(edge.unit) if edge.unit is not None else None)
        rows.append(BlockRow("unit", text=unit or _NOT_STATED))
    elif edge.other_port is not None or edge.port is not None:
        rows.append(BlockRow("port", text=edge.other_port or edge.port))
    else:
        rows.append(BlockRow("kind", text=kind))
    standing = _CONTESTED_WORD if contested else None
    if edge.kind == PLACE_RELATION and kind != edge.other_kind:  # the place's level: rack
        word = kind
    title = f"{word.capitalize()}  {edge.other}"
    return ChainBlock(title, link, tuple(rows), arrow=arrow, standing=standing)


def _default_chain(
    graph: PanelGraph, index: _Index, thing: Thing, hits: list[RoleHit]
) -> tuple[Chain, set[str]]:
    """The default recipe's chain and the channels its blocks show: the device's
    place and power left of it, then the device (focus, its identity roles), then
    its network, its controller and every other relation it claims, one block per
    thing; a dashed "not stated" block where it claims no place, power, network or
    controller."""
    edges = graph.neighbours(thing.id, thing.kind)
    before: list[ChainBlock] = []
    after: list[ChainBlock] = []
    for relation, word, arrow in DEFAULT_CHAIN:
        to = "place" if relation == PLACE_RELATION else "device"
        step = ChainStep(
            DEFAULT_RECIPE, word, 1, 0, relation, "out", to, None, None, word, None, None
        )
        found = graph.hop(thing.id, thing.kind, step)
        side = before if relation in (PLACE_RELATION, "powered_by") else after
        if not found:
            side.append(
                ChainBlock(f"{word.capitalize()}: {_NOT_STATED}", None, dashed=True, arrow=arrow)
            )
        contested = len(found) > 1  # two things after the filter; a claim twice is one
        for edge in found:
            side.append(_relation_block(graph, index, edge, word, arrow, contested))
    shown = {relation for relation, _w, _a in DEFAULT_CHAIN}
    for outgoing in (True, False):  # what the device claims, then what claims it
        for kind in sorted(edges):
            if kind in _NO_BLOCK or (outgoing and kind in shown) or kind == PLACE_RELATION:
                continue
            word = _relation_word(kind, outgoing)
            after.extend(
                _relation_block(graph, index, edge, word, word)
                for edge in _one_per_thing([e for e in edges[kind] if e.outgoing is outgoing])
            )
    # a role of a nameplate signal says what the device is: its own block, no tile
    identity = [h for h in hits if h.role.role in graph.identity and h.owner == thing.id]
    rows = tuple(
        BlockRow(_plate_word(h.role), h.channel, shown_as="value", words=_words(graph, h.channel))
        for h in identity
    )
    if not rows:
        rows = (BlockRow("class", text=thing.cls or thing.kind),)
    device = ChainBlock(thing.id, None, rows, focus=True)
    # the arrow between two blocks carries the word of the one nearer the device
    blocks = [replace_arrow(b, b.arrow) for b in before] + [device]
    for block in after:
        blocks[-1] = replace_arrow(blocks[-1], block.arrow)
        blocks.append(replace_arrow(block, None))
    return Chain(tuple(blocks), _CHAIN_FOOT), {h.channel for h in identity}


def _relation_word(kind: str, outgoing: bool) -> str:
    """A relation's word seen from the device: its subject's title, or its object's."""
    titles = RELATION_TITLES.get(kind, (kind.replace("_", " "), kind.replace("_", " ")))
    return titles[0] if outgoing else titles[1]


def replace_arrow(block: ChainBlock, arrow: str | None) -> ChainBlock:
    return dataclasses.replace(block, arrow=arrow)


def _one_per_thing(edges: list[Edge]) -> list[Edge]:
    """The first claim per other end, contested ones kept beside accepted ones."""
    seen: dict[tuple[str, str], Edge] = {}
    for edge in edges:
        key = (edge.other_kind, edge.other)
        if key not in seen or (seen[key].unit is None and edge.unit is not None):
            seen[key] = edge
    return list(seen.values())


def _press(graph: PanelGraph, hit: RoleHit) -> float | None:
    """What a press of a role row writes: its own value, else the value whose state
    word is its label, else ``None``."""
    if hit.role.press_value is not None:
        return hit.role.press_value
    if hit.role.label is None:
        return None
    value = graph.state_value(hit.channel, hit.role.label)
    return _as_number(value) if value is not None else None


def _short_channel(channel: str, device: str) -> str:
    return channel[len(device) + 1 :] if channel.startswith(device + ":") else channel


_SEGMENTS = re.compile(r"[:_]")


def _apart(names: list[str]) -> list[str]:
    """What tells *names* apart: each without the segments (split at ``:`` and
    ``_``; a record's own suffix such as ``-SP`` stays) every one of them starts
    and ends with; a name left empty, or two left alike, keep the whole name."""
    split = [_SEGMENTS.split(name) for name in names]
    if len(split) < 2:
        return list(names)
    head = 0
    while all(len(parts) > head + 1 and parts[head] == split[0][head] for parts in split):
        head += 1
    tail = 0
    while all(
        len(parts) > head + tail + 1 and parts[-1 - tail] == split[0][-1 - tail] for parts in split
    ):
        tail += 1
    cut = []
    for name, parts in zip(names, split, strict=True):
        kept = parts[head : len(parts) - tail]
        cut.append(_rejoin(name, parts, head, len(parts) - tail) if kept else name)
    return cut if len(set(cut)) == len(cut) else list(names)


def _rejoin(name: str, parts: list[str], start: int, stop: int) -> str:
    """The text of ``parts[start:stop]`` as it stands in *name*, its separators kept."""
    at = sum(len(p) + 1 for p in parts[:start])
    length = sum(len(p) for p in parts[start:stop]) + (stop - start - 1)
    return name[at : at + length]


def _told_apart(controls: list[Control], device: str, *, keep_word: bool) -> list[Control]:
    """Controls whose labels repeat, each labelled by what tells its channel apart
    from the others of that label (:func:`_apart`); with *keep_word*, after its
    word ("Enable CH2"). Labels that are all distinct are kept."""
    by_label: dict[str, list[int]] = {}
    for n, control in enumerate(controls):
        by_label.setdefault(control.label, []).append(n)
    found = list(controls)
    for label, places in by_label.items():
        if len(places) < 2:
            continue
        names = [_short_channel(controls[n].channel, device) for n in places]
        for n, part in zip(places, _apart(names), strict=True):
            text = f"{label} {part}" if keep_word else part
            found[n] = dataclasses.replace(controls[n], label=text)
    return found


def _as_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _press_text(value: float | None) -> str:
    return _NOT_STATED if value is None else f"{value:g}"


def _operate(
    graph: PanelGraph, thing: Thing, hits: list[RoleHit], channels: list[Channel]
) -> tuple[Operate | None, set[str], list[str]]:
    """The operate block: the control roles of the device grouped by role (a choice
    by role and channel), each group with the line of what it writes. With no
    control role, from the writable channels: a two- or more-state output with
    stated words is a choice of them, one with a single stated word an action that
    writes that word's value, a command an action that writes 1, any other output
    with no stated word nothing, a text record nothing (Apply writes only a number;
    it stays in the channels view), any other channel an entry.

    A press is offered only with a value a source states; the third item is the
    channels of the presses left out for want of one (they stay in the grid)."""
    groups: list[ControlGroup] = []
    used: set[str] = set()
    unstated: list[str] = []
    # a role a channel's signal gives it says what it is, not how it is pressed
    controls = [h for h in hits if h.role.shown_as in CONTROLS and h.role.stated]
    status = {h.role.role: h.channel for h in hits if h.role.shown_as not in CONTROLS}
    if controls:
        keyed: dict[tuple[str, str, str], list[RoleHit]] = {}
        for hit in controls:
            channel_key = hit.channel if hit.role.shown_as == "choice" else ""
            keyed.setdefault((hit.role.shown_as, hit.role.role, channel_key), []).append(hit)
        for (kind, role, _channel), found in keyed.items():
            title = _role_word(found[0].role)
            labelled = all(h.role.label for h in found)
            joined = [h for h in found if h.join is not None]  # never pressed: read only
            found = [h for h in found if h.join is None]
            if kind != "choice":  # a choice's segments keep their values' order
                found.sort(key=lambda h: name_order(h.channel))
            items = []
            for hit in found:
                press = _press(graph, hit) if kind != "entry" else None
                if kind != "entry" and press is None:
                    unstated.append(hit.channel)
                    continue
                shade = _status(graph, hit, status) if kind == "choice" else None
                label = hit.role.label or hit.role.title or _role_title(role)
                inferred = hit.role.press_value is not None and hit.role.basis == "inferred"
                items.append(Control(label, hit.channel, press, shade or hit.channel, inferred))
                used.add(hit.channel)
            if kind == "choice" and len(items) == 1:  # one segment is no choice
                kind, items = _one_segment(graph, items[0])
            if kind in ("entry", "action"):
                # one word for several channels says nothing: each what tells it apart
                # a stated label stays as the word in front of what tells them apart
                items = _told_apart(items, thing.id, keep_word=kind == "action" or labelled)
            if kind == "choice":
                items = _distinct(items)
            if items:
                groups.append(
                    ControlGroup(kind, title, tuple(items), _writes(kind, items, thing.id))
                )
            if joined:
                shown = {h.channel: h for h in joined}  # one read-only value per channel
                rule = "; ".join(dict.fromkeys(h.join or RULE_NOT_STATED for h in joined))
                groups.append(
                    ControlGroup(
                        WITHHELD,
                        title,
                        tuple(Control(_role_word(h.role), c) for c, h in shown.items()),
                        JOIN_WITHHELD,
                        rule,
                    )
                )
                used.update(shown)
        return (Operate(tuple(groups)) if groups else None), used, unstated
    actions: list[Control] = []
    entries: list[Control] = []
    for channel in (c for c in channels if c.writable and c.drawn_as != "text"):
        # a text record is never an entry: Apply writes only a number
        words = graph.state_words(channel.name)
        row = graph.record_type(channel.name)
        enumerated = row is not None and row.record_type in ENUMERATED_OUTPUTS
        stated = [
            Control(w, channel.name, number, channel.name)
            for v, w in words.items()
            if (number := _as_number(v)) is not None
        ]
        if enumerated and len(stated) > 1:  # a choice of its stated states
            stated = _distinct(stated)
            title = action_word(channel.name)
            groups.append(
                ControlGroup("choice", title, tuple(stated), _writes("choice", stated, thing.id))
            )
        elif enumerated and len(stated) == 1:  # one stated state: an action writing it
            actions.append(stated[0])
        elif channel.drawn_as == "command":
            # 1 is inferred unless the channel's own states name it
            stated_one = graph.state_label(channel.name, _DEFAULT_PRESS) is not None
            word = _action_word(channel.name, graph)
            actions.append(Control(word, channel.name, _DEFAULT_PRESS, inferred=not stated_one))
        elif enumerated:  # an output with no stated word: no value to write is stated
            unstated.append(channel.name)
            continue
        else:
            entries.append(Control(_short_channel(channel.name, thing.id), channel.name))
        used.add(channel.name)
    if actions:
        actions = _told_apart(actions, thing.id, keep_word=True)
        groups.insert(
            0, ControlGroup("action", None, tuple(actions), _writes("action", actions, thing.id))
        )
    if entries:
        groups.append(
            ControlGroup("entry", None, tuple(entries), _writes("entry", entries, thing.id))
        )
    return (Operate(tuple(groups)) if groups else None), used, unstated


#: A group of presses withheld: the card reaches their channel only through an
#: inferred join, so they are shown read only with this line.
WITHHELD = "withheld"
JOIN_WITHHELD = "join inferred, command not offered"


def _one_segment(graph: PanelGraph, control: Control) -> tuple[str, list[Control]]:
    """A choice role with one stated segment: a choice of every state its channel's
    words state, when they state two or more; else an action that writes its value,
    with the live status beside it."""
    stated = [
        Control(w, control.channel, number, control.status)
        for v, w in graph.state_words(control.channel).items()
        if (number := _as_number(v)) is not None
    ]
    if len(stated) > 1 and any(c.press_value == control.press_value for c in stated):
        return "choice", stated
    return "action", [control]


@dataclass(frozen=True)
class PressCount:
    """The presses the cards of some devices offer, by how their value is known."""

    stated: int = 0
    inferred: int = 0
    left_out: int = 0  # no value stated or inferred: not offered
    left_out_cards: int = 0
    withheld: int = 0  # reached through an inferred join: shown read only


def press_counts(graph: PanelGraph, devices: Iterable[str]) -> PressCount:
    """How many presses the cards of *devices* offer with a stated value, with an
    inferred one, leave out because no source states what they write (and on how
    many devices), and withhold because their channel is reached only through an
    inferred join."""
    stated = inferred = count = cards = withheld = 0
    for device in devices:
        thing = graph.thing(device, "device")
        hits = graph.roles(device)
        channels = _listed(graph, device, graph.channels(device, thing.kind))
        operate, _used, left = _operate(graph, thing, hits, channels)
        for group in operate.groups if operate is not None else ():
            if group.kind == WITHHELD:
                withheld += len(group.controls)
                continue
            for control in group.controls if group.kind != "entry" else ():
                inferred += control.inferred
                stated += not control.inferred
        count += len(left)
        cards += bool(left)
    return PressCount(stated, inferred, count, cards, withheld)


def tile_less(graph: PanelGraph, devices: Iterable[str]) -> tuple[int, int]:
    """How many classes, and devices, the default recipe draws with no key tile: no
    ``metrics.csv`` card row and no role that shows a value (the data side's to add)."""
    classes: set[str] = set()
    count = 0
    for device in devices:
        thing = graph.thing(device, "device")
        if graph.recipe(device) is None and not _key_tiles(graph, thing, graph.roles(device)):
            classes.add(thing.cls or thing.kind)
            count += 1
    return len(classes), count


def _status(graph: PanelGraph, hit: RoleHit, status: Mapping[str, str]) -> str | None:
    """The channel whose value says which segment of a choice is in force: the
    readback the facility file pairs with the command, else the channel of the role
    named as the command's without ``-cmd``, else ``None`` (the command itself)."""
    pair = graph.pair(hit.channel)
    if pair is not None:
        return pair.readback
    return status.get(hit.role.role.removesuffix("-cmd"))


def _action_word(channel: str, graph: PanelGraph) -> str:
    """A command's word: the state word of the value it writes, else its name's."""
    return graph.state_label(channel, _DEFAULT_PRESS) or action_word(channel)


#: A command name's ending that says "this is a command", not what it does; the
#: button's word is the name without it.
_COMMAND_ENDING = re.compile(r"[-_.]?(?:cmd|command|cmnd)$", re.IGNORECASE)


def action_word(channel: str) -> str:
    """The action a command channel's name says: its last segment (after the last
    ``:``), a command ending (``_Cmd``, ``-CMD``) cut off, split at ``_``, ``-``,
    ``.``, a lower-to-upper case step and a number, in sentence case (a word all in
    capitals kept): ``resetRelay1`` gives ``Reset relay 1``, ``HV_On-Cmd`` gives
    ``HV on``. A name that is all ending reads whole."""
    segment = channel.rsplit(":", 1)[-1]
    text = _COMMAND_ENDING.sub("", segment)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", text)
    words = [w for w in re.split(r"[-_. ]+", text) if w]
    if not words:
        return segment
    words = [w if w.isupper() else w.lower() for w in words]
    first = words[0]
    return " ".join([first[:1].upper() + first[1:], *words[1:]])


#: What a press's words add when its value is inferred: never written silently.
INFERRED_NOTE = "(value inferred, not stated in the database)"


def press_words(control: Control) -> str:
    """``<value>``, with :data:`INFERRED_NOTE` after an inferred one."""
    value = _press_text(control.press_value)
    return f"{value} {INFERRED_NOTE}" if control.inferred else value


def _distinct(controls: list[Control]) -> list[Control]:
    """Segments that share one word say nothing apart: each gets its value beside it."""
    if len({c.label for c in controls}) == len(controls):
        return controls
    return [
        dataclasses.replace(c, label=f"{c.label} = {_press_text(c.press_value)}") for c in controls
    ]


def _writes(kind: str, controls: list[Control], device: str) -> str:
    """The muted line under a group: what it writes, and when; an inferred value
    says so beside it."""
    names = ", ".join(dict.fromkeys(_short_channel(c.channel, device) for c in controls))
    if kind == "entry" and len(controls) > ENTRIES_IN_ONE_COLUMN:
        return "type a value, check it beside the live one, Apply writes each entry's own channel"
    if kind == "entry":
        return f"type a value, check it beside the live one, Apply writes it ({names})"
    if any(c.inferred for c in controls):
        each = "; ".join(
            f"{_short_channel(c.channel, device)} = {press_words(c)}" for c in controls
        )
        return f"writes {each}; " + (
            "stays until changed" if kind == "choice" else "once per press"
        )
    presses = ", ".join(
        c.label if c.label.endswith(f"= {_press_text(c.press_value)}")
        else f"{c.label} = {_press_text(c.press_value)}"
        for c in controls
    )  # fmt: skip
    if kind == "choice":
        return f"writes {names}: {presses}; stays until changed"
    if len({c.channel for c in controls}) == len(controls) and all(
        c.press_value == controls[0].press_value for c in controls
    ):
        return f"each writes its channel = {_press_text(controls[0].press_value)} once ({names})"
    return f"writes {names}: {presses}, once per press"


#: The recipe of a device that no ``card_recipes.csv`` row takes; its sections and
#: chains may still be stated under this name.
DEFAULT_RECIPE = "default"
#: How a chain hop came out: every thing found, or why not.
HOP_FOUND, HOP_NOT_STATED, HOP_CONTESTED, HOP_NO_START = (
    "found",
    "not stated",
    "contested",
    "its start is not stated",
)


def hop_from(graph: PanelGraph, starts: list[tuple[str, ThingKind]], step: ChainStep) -> list[Edge]:
    """A hop walked from each thing its start step reached, the targets united, one
    claim per target (in target order). The caller judges the count: one target is
    found, several are contested unless the hop is ``many``, none is not stated. A
    claim's own standing does not make a hop contested; only two distinct targets
    after the hop's ``to`` filter do."""
    found: dict[tuple[str, str], Edge] = {}
    for thing_id, kind in starts:
        for edge in graph.hop(thing_id, kind, step):
            found.setdefault((edge.other, edge.other_kind), edge)
    return [found[key] for key in sorted(found)]


def _port_word(step: ChainStep, edge: Edge) -> str | None:
    """The word on the arrow into a hop's block: the port its ``port_from`` names
    (the claim's object or subject end), or ``port_from`` itself as a word."""
    if step.port_from == "object_port":
        port = edge.other_port if edge.outgoing else edge.port
    elif step.port_from == "subject_port":
        port = edge.port if edge.outgoing else edge.other_port
    else:
        return step.port_from
    return port.rsplit(":", 1)[-1] if port else None  # a port named as a channel: its last part


def _block_rows(
    graph: PanelGraph, hits: list[RoleHit], thing_id: str, kind: ThingKind, roles: list[str]
) -> tuple[list[BlockRow], set[str]]:
    """The lines of a block: each role, found on the block's thing (as a role the
    card's device plays there, else one the thing plays itself), else as a role the
    card's device plays on a channel of its own, else not served; with no role, the
    thing's class."""
    rows: list[BlockRow] = []
    used: set[str] = set()
    own: list[RoleHit] | None = None
    for role in roles:
        found = [h for h in hits if h.role.role == role and h.owner == thing_id]
        if not found:
            if own is None:
                try:
                    own = graph.roles(thing_id) if kind == "device" else []
                except KeyError:
                    own = []
            found = [h for h in own if h.role.role == role]
        if not found:
            found = [h for h in hits if h.role.role == role]
        if not found:
            word = graph.titles.get(role) or _role_title(role).lower()
            rows.append(BlockRow(word, None, shown_as="value"))
            continue
        hit = found[0]
        shown = hit.role.shown_as if hit.role.shown_as in ("value", "word", "lamp") else "value"
        unit, scale = _role_unit(graph, hit)
        rows.append(
            BlockRow(
                _row_word(hit.role), hit.channel, shown_as=shown, unit=unit, scale=scale,
                words=_words(graph, hit.channel),
            )
        )  # fmt: skip
        used.add(hit.channel)
    if not roles:
        try:
            other = graph.thing(thing_id, kind)
            rows.append(BlockRow("class", text=other.cls or other.level or other.kind))
        except KeyError:
            pass
    return rows, used


#: The half width of a card strip's window when its section states none, m.
STRIP_HALF_M = 3.0
_PLACE_BRACKET = "in {}, place in it not stated"
#: How a card's strip came out for a device.
STRIP_PLACED, STRIP_BY_PLACE, STRIP_NOWHERE = "placed", "by its place", "not stated"


def _beam_order(graph: PanelGraph, frame: str) -> list[tuple[float, Thing]]:
    """Every device placed along the beam path *frame* (:func:`_beam_of`), by its
    position; built once per graph."""
    frames = _BEAM_ORDERS.get(graph)
    if frames is None:
        frames = {}
        for thing in graph.things():
            if thing.kind == "device":
                found = _beam_of(graph, thing)
                if found is not None:
                    frames.setdefault(found[0], []).append((found[1], thing))
        for listed in frames.values():
            listed.sort(key=lambda pair: (pair[0], pair[1].id))
        _BEAM_ORDERS[graph] = frames
    return frames.get(frame, [])


_BEAM_ORDERS: weakref.WeakKeyDictionary[PanelGraph, dict[str, list[tuple[float, Thing]]]] = (
    weakref.WeakKeyDictionary()
)


def _where_on_strip(
    graph: PanelGraph, thing: Thing
) -> tuple[float, tuple[str, float, float] | None, str] | None:
    """Where a device is along the beam, as ``(s, by, frame)``: its own position on
    a beam path (``by`` ``None``; :func:`_beam_of`, so a part takes its whole's), else
    the stretch of the nearest place around it that has one, as ``by = (place, lo,
    hi)`` with ``s`` its middle: the place's own position and those of the devices
    in it, along the beam path most of them are on, read on across a ring's start;
    else ``None``."""
    found = _beam_of(graph, thing)
    if found is not None:
        return found[1], None, found[0]
    place = graph.place_of(thing.id, "device")
    while place is not None:
        try:
            here = graph.thing(place, "place")
        except KeyError:
            return None
        stretch = _place_stretch(graph, here)
        if stretch is not None:
            frame, lo, hi = stretch
            return (lo + hi) / 2, (place, lo, hi), frame
        place = here.place
    return None


def _place_stretch(graph: PanelGraph, place: Thing) -> tuple[str, float, float] | None:
    """``(frame, lo, hi)``: where a place and what it holds lie along a beam path."""
    found = [graph.beam_position(place)]
    found += [_beam_of(graph, t) for t in graph.contents(place.id) if t.kind == "device"]
    frames = Counter(where[0] for where in found if where is not None)
    if not frames:
        return None
    frame = frames.most_common(1)[0][0]
    positions = {
        str(n): where[1] for n, where in enumerate(found) if where is not None and where[0] == frame
    }
    read = _unwrapped(positions, graph.frame_length(frame)).values()
    return frame, min(read), max(read)


def strip_states(graph: PanelGraph, devices: Iterable[str]) -> dict[str, Counter[str]]:
    """Per recipe that draws a strip, how many of *devices* are placed, placed only
    by a place around them, or not placed (:func:`strip_state`)."""
    found: dict[str, Counter[str]] = {}
    for device in devices:
        recipe = graph.recipe(device)
        name = recipe.recipe if recipe is not None else DEFAULT_RECIPE
        if any(r.kind == "strip" for r in graph.card_sections(name)):
            thing = graph.thing(device, "device")
            found.setdefault(name, Counter())[strip_state(graph, thing)] += 1
    return found


def strip_state(graph: PanelGraph, thing: Thing) -> str:
    """:data:`STRIP_PLACED`, :data:`STRIP_BY_PLACE` or :data:`STRIP_NOWHERE`."""
    where = _where_on_strip(graph, thing)
    if where is None:
        return STRIP_NOWHERE
    return STRIP_PLACED if where[1] is None else STRIP_BY_PLACE


def card_strip(
    graph: PanelGraph, index: _Index, thing: Thing, section: CardSection
) -> CardStrip | None:
    """A card's strip: the devices of the section's ``members`` classes that
    ``symbols.csv`` draws, within ``window_m`` of the device along the beam, each
    with the first channel of the section's first role. A device placed only by a
    place around it is a dashed bracket over that place's stretch, the window
    widened to hold it; one placed by neither has no strip (``None``)."""
    where = _where_on_strip(graph, thing)
    if where is None:
        return None
    centre, by, frame = where
    half = section.window_m or STRIP_HALF_M
    lo, hi = centre - half, centre + half
    if by is not None:
        lo, hi = min(lo, by[1]), max(hi, by[2])
    classes = {c for c in (section.members or "").split(";") if c}
    role = next((r for r in (section.roles or "").split(";") if r), None)
    members: list[StripMember] = []
    seen: set[str] = set()
    ring = graph.frame_length(frame)
    for stated, other in _beam_order(graph, frame):
        # on a ring the window reads on across its start: a device just before it
        # is drawn just below the window's centre, not a ring's length away
        s_m = centre + _ring_offset(stated - centre, ring) if ring else stated
        if s_m < lo or s_m > hi or (other.cls not in classes and other.id != thing.id):
            continue
        symbol = graph.strip_symbol(other)
        if symbol is None:
            continue
        seen.add(other.id)
        members.append(_strip_member(graph, index, other, s_m, symbol, role, thing.id))
    bracket = None
    if by is not None:
        bracket = (by[1], by[2], _PLACE_BRACKET.format(by[0]))
        symbol = graph.strip_symbol(thing)
        if symbol is not None and thing.id not in seen:  # drawn in the bracket's middle
            members.append(_strip_member(graph, index, thing, centre, symbol, role, thing.id))
    members.sort(key=lambda m: (m.s_m, m.thing))
    return CardStrip((lo, hi), tuple(members), bracket, section.caption)


def _ring_offset(delta: float, length: float) -> float:
    """*delta* along a ring of *length*, taken the short way: in [-length/2, length/2)."""
    return (delta + length / 2) % length - length / 2


def _strip_member(graph, index, other: Thing, s_m: float, symbol, role, focus: str) -> StripMember:
    channel = unit = None
    if role is not None:
        hit = next((h for h in graph.roles(other.id) if h.role.role == role), None)
        if hit is not None:
            channel, unit = hit.channel, _role_unit(graph, hit)[0]
    link = None
    if other.id != focus:
        item = _ref(graph, index, other)
        link = item if isinstance(item, Link) else None
    label = other.short
    return StripMember(
        other.id, label, s_m, symbol.symbol, symbol.lane, link, channel, unit, other.id == focus
    )


def recipe_chain(
    graph: PanelGraph,
    index: _Index,
    thing: Thing,
    hits: list[RoleHit],
    recipe: str,
    section: CardSection,
) -> tuple[Chain, set[str], str]:
    """A recipe's chain for a device: the device's block (in focus, its value roles
    no hop or front shows that one channel of it plays), then one block per hop in step order, each the thing the hop
    reaches with its roles, the arrow into it carrying its port; a hop that reaches
    nothing is a dashed "not stated" block, one that reaches two distinct things
    stacks them dashed as contested, ``many`` stacks every thing it reaches. A hop
    whose start reached several things walks from each (:func:`hop_from`).
    Returns the chain, the channels it shows and how it came out (:data:`HOP_FOUND`
    when every hop found its things, else the first hop's title and why)."""
    steps = graph.chain(recipe, section.members or "")
    reached: dict[int, list[tuple[str, ThingKind]]] = {0: [(thing.id, thing.kind)]}
    blocks: list[ChainBlock] = []
    used: set[str] = set()
    state = HOP_FOUND
    for step in steps:
        n = int(step.step)
        start = int(step.from_step) if step.from_step is not None else n - 1
        froms = reached.get(start) or []
        roles = [r for r in (step.roles or "").split(";") if r]
        edges = hop_from(graph, froms, step)
        if not froms:
            why = HOP_NO_START
        elif not edges:
            why = HOP_NOT_STATED
        elif step.many not in MANY_WORDS and len(edges) > 1:
            why = HOP_CONTESTED
        else:
            why = HOP_FOUND
        if why != HOP_FOUND and state == HOP_FOUND:
            state = f"{step.title}: {why}"
        reached[n] = [(e.other, e.other_kind) for e in edges]
        if not edges:
            blocks.append(ChainBlock(f"{step.title}: {_NOT_STATED}", None, dashed=True))
            continue
        made: list[ChainBlock] = []
        for edge in edges:
            rows, shown = _block_rows(graph, hits, edge.other, edge.other_kind, roles)
            used |= shown
            item = _edge_item(graph, index, step.title, edge)
            link = item if isinstance(item, Link) else None
            standing = _CONTESTED_WORD if why == HOP_CONTESTED else None
            made.append(
                ChainBlock(
                    f"{step.title}  {edge.other}", link, tuple(rows), standing=standing,
                    inferred=inferred_rule([edge]),
                )
            )  # fmt: skip
        port = _port_word(step, edges[0]) or _relation_word(step.relation, step.direction == "out")
        blocks.append(dataclasses.replace(made[0], stack=tuple(made[1:]), arrow_in=port))
    shown_roles = {r for s in steps for r in (s.roles or "").split(";") if r}
    shown_roles |= {  # a front of the recipe shows its roles on its connectors
        r
        for row in graph.card_sections(recipe)
        if row.kind == "front"
        for r in (row.roles or "").split(";")
        if r
    }
    # a role several of the device's channels play is one per connector or member:
    # its place is the front, a grid or the channel list, never rows on this block
    played = Counter(h.role.role for h in hits if h.owner == thing.id)
    mine = [
        h for h in hits
        if h.owner == thing.id and h.role.role not in shown_roles
        and played[h.role.role] == 1
        and h.role.shown_as in ("value", "word", "lamp")
        and h.role.role not in graph.identity and h.role.role != INTERNAL_ROLE
    ][:FOCUS_ROWS]  # fmt: skip
    labels = [_row_word(h.role) for h in mine]
    focus_rows = tuple(
        BlockRow(
            label, h.channel, shown_as=h.role.shown_as,
            unit=_role_unit(graph, h)[0], scale=_role_unit(graph, h)[1], words=_words(graph, h.channel),
        )
        for h, label in zip(mine, labels, strict=True)
    )  # fmt: skip
    used |= {h.channel for h in mine}
    device = ChainBlock(
        f"{thing.cls or thing.kind}  {thing.id}", None, focus_rows or (BlockRow("class", text=thing.cls or thing.kind),), focus=True
    )  # fmt: skip
    ordered = [device, *blocks]
    # the arrow between two blocks carries the word of the hop into the later one
    for i in range(len(ordered) - 1):
        ordered[i] = dataclasses.replace(ordered[i], arrow=ordered[i + 1].arrow_in)
    return Chain(tuple(ordered), section.caption), used, state


#: How a connector of a front came out: something on it, nothing stated, or
#: two things or more on one side; a reading has nothing to be on it.
FRONT_STATED, FRONT_NONE, FRONT_CONTESTED = "stated", "nothing stated", "contested"
FRONT_READING = "reading"
_PORTS_NOT_STATED = "port count and assignment not stated"
_LOOSE = "on no stated port: {}"


def _numbered(pattern: str) -> re.Pattern[str]:
    """A ``{n}`` glob (``*:OutletStatus{n}``) as a regex that captures the number."""
    head, _, tail = pattern.partition("{n}")

    def glob(text: str) -> str:
        return re.escape(text).replace(r"\*", ".*").replace(r"\?", ".")

    return re.compile(glob(head) + r"(\d+)" + glob(tail))


def _near(
    graph: PanelGraph, thing_id: str, relation: str | None, direction: str | None
) -> list[Edge]:
    """The claims of *relation* in *direction* at a thing, superseded ones left out;
    ``out``: the thing is the subject, ``in``: the object; none for no relation."""
    if not relation:
        return []
    try:
        edges = graph.neighbours(thing_id, "device").get(relation, ())
    except KeyError:
        return []
    return [
        e
        for e in edges
        if e.outgoing is (direction == "out") and e.standing != "superseded" and e.other != thing_id
    ]


def _socket(graph: PanelGraph, device: str, connector: str) -> str | None:
    """The thing ``<device>:<connector>`` or ``<device>.<connector>``, if it exists."""
    for sep in (":", "."):
        try:
            return graph.thing(f"{device}{sep}{connector}", "device").id
        except KeyError:
            continue
    return None


def front_numbers(graph: PanelGraph, device: str, row: FrontRow) -> list[str]:
    """The connectors of a row: one (no ``{n}``), else the numbers 1..``count`` when
    stated, else those of the device's served channels matching ``channel``, else
    those of the socket things that exist, else those of the ports the relation
    states; in number order, never as text."""
    if "{n}" not in row.connector:
        return [""]
    listed = _row_roles(row)
    if listed and not row.channel:  # numbered by the channels of its first role served
        numbers = {
            int(n)
            for h in graph.roles(device)
            if h.owner == device
            and h.role.role == listed[0]
            and (n := _last_number(h.channel)) is not None
        }
        return [str(k) for k in sorted(numbers)]
    if row.count is not None:
        return [str(k) for k in range(1, int(row.count) + 1)]
    found: set[int] = set()
    if row.channel:
        rx = _numbered(row.channel)
        found = {
            int(m.group(1)) for c in graph.channels(device, "device") if (m := rx.fullmatch(c.name))
        }
    elif row.far_direction == "socket":
        rx = re.compile(r"[:.]" + _numbered(row.connector).pattern)
        found = {
            int(m.group(1))
            for t in graph.named_under(device)
            if (m := rx.fullmatch(t.id[len(device) :]))
        }
    elif row.far_port and "{n}" in row.far_port:
        rx = re.compile(re.escape(row.far_port).replace(re.escape("{n}"), r"(\d+)"))
        found = {
            int(m.group(1))
            for e in _near(graph, device, row.far_relation, row.far_direction)
            if e.port and (m := rx.fullmatch(e.port))
        }
    return [str(k) for k in sorted(found)]


#: The hover text of an inferred relation that states no rule.
RULE_NOT_STATED = "inferred; the rule is not stated"


def inferred_rule(edges: Iterable[Edge]) -> str | None:
    """For relations any of which rests on an inferred claim: the rule(s) it was
    inferred by (:data:`RULE_NOT_STATED` when none is stated); ``None`` when every
    one is stated. One rule for every hop, connector and member."""
    rules = [e.rule or RULE_NOT_STATED for e in edges if e.basis == "inferred"]
    return "; ".join(dict.fromkeys(rules)) if rules else None


def _far_end(
    graph: PanelGraph, index: _Index, thing_id: str, port: str | None, inferred: str | None = None
) -> FarEnd:
    try:
        item = _ref(graph, index, graph.thing(thing_id, "device"))
    except KeyError:
        item = None
    return FarEnd(thing_id, item if isinstance(item, Link) else None, port, inferred)


def front_sides(
    graph: PanelGraph, index: _Index, device: str, row: FrontRow, n: str
) -> tuple[tuple[FarEnd, ...], ...]:
    """What is on connector *n*, per side: a socket thing has two sides (its own
    relation, then the relations naming it), a reading (a row with no far
    relation) none, any other connector one; on a side one thing per thing, in id
    order."""
    if row.far_relation is None:
        return ()
    if row.far_direction == "socket":
        socket = _socket(graph, device, row.connector.replace("{n}", n))
        if socket is None:
            return ((),)
        sides = []
        for d in ("out", "in"):
            near = _near(graph, socket, row.far_relation, d)
            ends = {e.other: e.other_port for e in near}
            sides.append(
                tuple(
                    _far_end(
                        graph, index, other, port,
                        inferred_rule(e for e in near if e.other == other),
                    )
                    for other, port in sorted(ends.items())
                )
            )  # fmt: skip
        return tuple(sides)
    port = (row.far_port or "").replace("{n}", n)
    near = [
        e
        for e in _near(graph, device, row.far_relation, row.far_direction)
        if not port or e.port == port
    ]
    ends = {e.other: e.other_port for e in near}
    return (
        tuple(
            _far_end(graph, index, o, p, inferred_rule(e for e in near if e.other == o))
            for o, p in sorted(ends.items())
        ),
    )


def front_state(sides: tuple[tuple[FarEnd, ...], ...]) -> str:
    """:data:`FRONT_READING` for a connector with no side, else
    :data:`FRONT_CONTESTED` when a side has two things or more, else
    :data:`FRONT_STATED` when a side has one, else :data:`FRONT_NONE`."""
    if not sides:
        return FRONT_READING
    if any(len(side) > 1 for side in sides):
        return FRONT_CONTESTED
    return FRONT_STATED if any(sides) else FRONT_NONE


def _row_roles(row: FrontRow) -> list[str]:
    """The section roles a ``fronts.csv`` row lists (empty: every section role)."""
    return [r for r in (row.roles or "").split(";") if r]


def _last_number(text: str) -> str | None:
    found = re.findall(r"\d+", text)
    return str(int(found[-1])) if found else None


def card_front(
    graph: PanelGraph,
    index: _Index,
    thing: Thing,
    hits: list[RoleHit],
    recipe: str,
    section: CardSection,
) -> Front:
    """A card's front: each ``fronts.csv`` row of the recipe a connector or a
    numbered range, by group; on each its state lamp (the row's ``channel``), what is
    on it per side, the section's roles of the connector (the device's channels of
    that role whose last number is the connector's, else the role on the one thing
    on it) and its presses (only with a stated or inferred value). Things the rows'
    relations name on no stated port are drawn apart, by class. A group with no
    connector is left out, unless clients on no stated port are drawn for it; so
    is a literal connector whose row lists roles when neither its lamp nor any of
    them binds on the device."""
    roles = [r for r in (section.roles or "").split(";") if r]
    own = [h for h in hits if h.owner == thing.id and h.role.role in roles]
    channels = {c.name for c in graph.channels(thing.id, "device")}
    groups: list[FrontGroup] = []
    presses: list[Control] = []
    ported: set[str] = set()
    entries: dict[str, list[Control]] = {}
    titles: dict[str, str] = {}  # each role's group title
    loose_rows: list[FrontRow] = []
    client_groups: set[tuple[str, str]] = set()
    for (face, group), rows in _grouped(graph.fronts(recipe)):
        connectors: list[Connector] = []
        for row in rows:
            numbers = front_numbers(graph, thing.id, row)
            numbered_by_role = bool(row.roles) and not row.channel
            apart = row.apart != "no"  # "no": clients on no stated port are not drawn
            if (
                "{n}" in row.connector
                and row.far_direction == "in"
                and not numbered_by_role
                and apart
            ):
                # clients of a range; a range numbered by a role has no clients apart
                loose_rows.append(row)
                client_groups.add((face, group))
            for n in numbers:
                sides = front_sides(graph, index, thing.id, row, n)
                lamp = None
                if row.channel:
                    pattern = row.channel.replace("{n}", n)
                    lamp = next(
                        (c for c in sorted(channels) if fnmatch.fnmatchcase(c, pattern)), None
                    )
                carried = [r for r in roles if r in _row_roles(row)] if row.roles else roles
                literal = row.connector if not n and row.roles else None
                rows_, controls, typed = _connector_roles(
                    graph, own, carried, n, sides, lamp, literal
                )
                if literal is not None and lamp is None and not (rows_ or controls or typed):
                    continue  # a literal connector nothing of this device binds to
                ported |= {end.thing for side in sides for end in side}
                presses += controls
                label = (row.label or row.connector).replace("{n}", n)
                for hit in typed:
                    word = _role_word(hit.role)
                    titles.setdefault(hit.role.role, word)
                    entries.setdefault(hit.role.role, []).append(
                        Control(f"{label} · {word}", hit.channel)
                    )
                connectors.append(Connector(label, row.kind, lamp, sides, rows_, tuple(controls)))
        empty = None if connectors else _PORTS_NOT_STATED
        groups.append(FrontGroup(face, group, tuple(connectors), empty))
    loose = _loose(graph, index, thing.id, loose_rows, ported, roles)
    # a group with no connector is drawn only as the clients' switch face whose
    # ports no source states; a range with nothing in it is no part of this unit
    groups = [g for g in groups if g.connectors or (loose and (g.face, g.title) in client_groups)]
    return Front(
        tuple(groups),
        loose,
        _LOOSE.format(_PORTS_NOT_STATED) if loose else None,
        _front_writes(presses, thing.id) if presses else None,
        section.caption,
        tuple(
            ControlGroup("entry", titles[role], tuple(found), _writes("entry", found, thing.id))
            for role, found in entries.items()
        ),
    )


def _grouped(rows: list[FrontRow]) -> list[tuple[tuple[str, str], list[FrontRow]]]:
    out: dict[tuple[str, str], list[FrontRow]] = {}
    for row in rows:  # by order already
        out.setdefault((row.face, row.group), []).append(row)
    return list(out.items())


def _connector_roles(
    graph: PanelGraph,
    own: list[RoleHit],
    roles: list[str],
    n: str,
    sides: tuple[tuple[FarEnd, ...], ...],
    lamp: str | None,
    literal: str | None = None,
) -> tuple[tuple[BlockRow, ...], list[Control], list[RoleHit]]:
    """The values and presses of *roles* on one connector: the device's channels of
    the role whose last number is *n*, or, on a literal connector whose row lists
    its roles, whose name carries the *literal*."""
    rows: list[BlockRow] = []
    controls: list[Control] = []
    typed: list[RoleHit] = []
    single = [end.thing for side in sides for end in side]
    for role in roles:
        if literal is not None:
            found = [h for h in own if h.role.role == role and literal in h.channel]
        else:
            found = [h for h in own if h.role.role == role and n and _last_number(h.channel) == n]
        if not found and len(single) == 1:  # the role on the one thing on it
            try:
                found = [h for h in graph.roles(single[0]) if h.role.role == role][:1]
            except KeyError:
                found = []
            found = [h for h in found if h.role.shown_as not in CONTROLS]
        for hit in found:
            if hit.role.shown_as in CONTROLS:
                if hit.role.shown_as == "entry":
                    typed.append(hit)  # drawn under the front, named by its connector
                    continue
                press = _press(graph, hit)
                if press is None:
                    continue  # no value stated: no press (the card's count has it)
                inferred = hit.role.press_value is not None and hit.role.basis == "inferred"
                label = hit.role.label or hit.role.title or _role_title(role)
                controls.append(Control(label, hit.channel, press, None, inferred))
            elif hit.channel != lamp or graph.state_words(hit.channel):
                shown = hit.role.shown_as if hit.role.shown_as != "lamp" else "word"
                unit, scale = _role_unit(graph, hit)
                rows.append(
                    BlockRow(
                        _row_word(hit.role), hit.channel, shown_as=shown, unit=unit,
                        scale=scale, words=_words(graph, hit.channel),
                    )
                )  # fmt: skip
                break  # one value per role and connector
    return tuple(rows), controls, typed


def _loose(
    graph: PanelGraph,
    index: _Index,
    device: str,
    rows: list[FrontRow],
    ported: set[str],
    roles: list[str],
) -> tuple[tuple[str, tuple[tuple[FarEnd, tuple[BlockRow, ...]], ...]], ...]:
    """The devices the relations of the rows of numbered ranges name as clients of
    the device (``in``) on no stated port, by class (in class order), each with the
    section's roles it plays itself."""
    seen: dict[str, Edge] = {}
    for row in rows:
        for edge in _near(graph, device, row.far_relation, row.far_direction):
            if edge.port is None and edge.other not in ported and edge.other_kind == "device":
                seen.setdefault(edge.other, edge)
    by_class: dict[str, list[tuple[FarEnd, tuple[BlockRow, ...]]]] = {}
    for other in sorted(seen, key=name_order):
        try:
            cls = graph.thing(other, seen[other].other_kind).cls or seen[other].other_kind
        except KeyError:
            cls = seen[other].other_kind
        found: list[BlockRow] = []
        if seen[other].other_kind == "device":
            for hit in graph.roles(other):
                if hit.role.role in roles and hit.role.shown_as not in CONTROLS:
                    shown = "word" if hit.role.shown_as == "lamp" else hit.role.shown_as
                    found.append(
                        BlockRow(
                            _row_word(hit.role), hit.channel, shown_as=shown,
                            words=_words(graph, hit.channel),
                        )
                    )  # fmt: skip
                    break
        end = _far_end(graph, index, other, seen[other].other_port)
        by_class.setdefault(cls, []).append((end, tuple(found)))
    return tuple(
        (cls, tuple(things))
        for cls, things in sorted(by_class.items(), key=lambda kv: name_order(kv[0]))
    )


def _front_writes(controls: list[Control], device: str) -> str:
    """One line for every press of a front: each writes its own connector's channel,
    the values by word, an inferred one said so."""
    values = list(dict.fromkeys((c.label, c.press_value, c.inferred) for c in controls))
    words = ", ".join(
        f"{label} = {press_words(Control(label, '', value, None, inferred))}"
        for label, value, inferred in values
    )
    names = list(dict.fromkeys(_short_channel(c.channel, device) for c in controls))
    span = names[0] if len(names) == 1 else f"{names[0]} … {names[-1]}"
    return f"each button writes its own connector's channel ({span}) once: {words}"


def front_states(graph: PanelGraph, devices: Iterable[str]) -> dict[str, Counter[str]]:
    """Per recipe with a front, how many connectors came out each way
    (:func:`front_state`, readings apart), as its card draws them, and its devices
    by whether every connector that is not a reading, some or none has something
    stated on it (``all``, ``some``, ``none``; ``readings only`` when every one is a
    reading)."""
    index = _index(graph)
    found: dict[str, Counter[str]] = {}
    for device in devices:
        recipe = graph.recipe(device)
        name = recipe.recipe if recipe is not None else DEFAULT_RECIPE
        sections = [r for r in graph.card_sections(name) if r.kind == "front"]
        if not graph.fronts(name) or not sections:
            continue
        counts = found.setdefault(name, Counter())
        thing = graph.thing(device, "device")
        front = card_front(graph, index, thing, graph.roles(device), name, sections[0])
        states = Counter(front_state(c.sides) for group in front.groups for c in group.connectors)
        counts.update(states)
        total = sum(states.values()) - states[FRONT_READING]  # a reading has no far end
        whole = total and states[FRONT_STATED] == total
        some = states[FRONT_STATED] or states[FRONT_CONTESTED]
        if not total and states[FRONT_READING]:
            counts["devices: readings only"] += 1  # nothing on its front is looked for
            continue
        counts["devices: all" if whole else "devices: some" if some else "devices: none"] += 1
    return found


_NO_CURVE = "no {} stated for this device"
_NO_OPERATING_POINT = (
    "operating point not drawn: the conversion from set point to current is not stated"
)


def card_curve(graph: PanelGraph, thing: Thing, section: CardSection) -> CurvePlot:
    """A card's curve: the points of the claim in force of the calibration kind the
    section's ``members`` names, for the device; with none, the note that none is
    stated. The operating point is not drawn (no table states the conversion)."""
    kind_id = section.members or ""
    kinds = [k for k in graph.calibration_kinds_of(thing.id) if k.kind == kind_id]
    title = kinds[0].title if kinds else kind_id
    units = ((kinds[0].unit or "") if kinds else "").split(";")
    x_unit = units[0] or None
    y_unit = units[1] if len(units) > 1 and units[1] else None
    if not kinds:
        return CurvePlot(title, (), x_unit, y_unit, None, _NO_CURVE.format(title.lower()))
    found = graph.calibration(thing.id, kind_id)
    points = tuple(sorted((p.x, p.y) for p in found.points))
    claim = found.in_force
    source = None
    if claim is not None and claim.document:
        source = f"{claim.document} {claim.version or ''}".strip()
    if not points and found.contested:  # a contested curve is not drawn as the one in force
        return CurvePlot(title, (), x_unit, y_unit, None, _CURVE_CONTESTED.format(title.lower()))
    if not points:
        return CurvePlot(title, (), x_unit, y_unit, source, _NO_CURVE.format(title.lower()))
    return CurvePlot(title, points, x_unit, y_unit, source, _NO_OPERATING_POINT)


_CURVE_CONTESTED = "the {} is contested: no claim in force, none drawn"
_NO_OWNER = "its {} not stated"
_NOT_ON_DRAWING = "where it sits on {}'s drawing"
_NO_DRAWING = "a drawing of {}"


def part_section(
    graph: PanelGraph,
    index: _Index,
    thing: Thing,
    hits: list[RoleHit],
    recipe: str,
    section: CardSection,
) -> list[Item]:
    """A part's section: its owner (the thing the section's ``focus`` relation
    names) drawn with the part in focus (the owner's picture rows whose ``signal``
    matches a channel of the part), then what is stated about the part (its
    relations) and what is not (its owner, its place on the drawing, a drawing at
    all, each hop of the recipe's chains that found nothing)."""
    relation = section.focus or PART_RELATION
    owners = sorted(
        {e.other for e in _near(graph, thing.id, relation, "out") if e.other_kind == "device"}
    )
    missing: list[str] = []
    items: list[Item] = []
    if len(owners) != 1:
        missing.append(_NO_OWNER.format(relation.replace("_", " ")))
    else:
        owner = graph.thing(owners[0], "device")
        mine = [c.name for c in graph.channels(thing.id, "device")]
        pictures = _pictures(graph, owner)
        if not pictures:
            missing.append(_NO_DRAWING.format(owner.id))
        focused = False
        for picture in pictures:
            assert isinstance(picture, Picture)
            focus = tuple(
                part.part
                for part in picture.parts
                if part.signal and any(fnmatch.fnmatchcase(c, part.signal) for c in mine)
            )
            focused |= bool(focus)
            items.append(dataclasses.replace(picture, focus=focus))
        if pictures and not focused:
            missing.append(_NOT_ON_DRAWING.format(owner.id))
    stated: list[Link | Fact] = []
    edges = graph.neighbours(thing.id, thing.kind)
    for kind in sorted(edges):
        if kind in _NO_BLOCK or kind == PLACE_RELATION:
            continue
        for edge in _one_per_thing(edges[kind]):
            word = _relation_word(kind, edge.outgoing)
            item = _edge_item(graph, index, word, edge)
            label = f"{word}: {edge.other}"
            stated.append(
                dataclasses.replace(item, label=label)
                if isinstance(item, Link)
                else Fact(word, edge.other)
            )
    for row in graph.card_sections(recipe):
        if row.kind == "chain" and row.members:
            _chain, _used, state = recipe_chain(graph, index, thing, hits, recipe, row)
            if state != HOP_FOUND:
                missing.append(state)
    items.append(PartFacts(tuple(stated), tuple(dict.fromkeys(missing))))
    return items


_THRESHOLD_PREFIX = "threshold-"


def limit_ladder(
    graph: PanelGraph, thing: Thing, hits: list[RoleHit], section: CardSection
) -> LimitLadder:
    """A ladder: the section's first role is the value, the others its set points
    in their order; each the device's first channel of that role. A set point no
    channel serves is left out and named."""
    roles = [r for r in (section.roles or "").split(";") if r]
    own: dict[str, RoleHit] = {}
    for hit in hits:
        if hit.owner == thing.id:
            own.setdefault(hit.role.role, hit)
    value = own.get(roles[0]) if roles else None
    marks: list[tuple[str, str]] = []
    missing: list[str] = []
    for role in roles[1:]:
        found = own.get(role)
        title = (
            found.role.label
            or found.role.title
            or _role_title(role.removeprefix(_THRESHOLD_PREFIX))
            if found is not None
            else _role_title(role.removeprefix(_THRESHOLD_PREFIX))
        )
        if found is None:
            missing.append(title)
        else:
            marks.append((title, found.channel))
    unit = _role_unit(graph, value)[0] if value is not None else None
    return LimitLadder(
        value.channel if value is not None else None, unit, tuple(marks), tuple(missing),
        section.caption,
    )  # fmt: skip


def ladder_states(graph: PanelGraph, devices: Iterable[str]) -> dict[str, Counter[str]]:
    """Per recipe with a ladder, its devices drawn whole and those with a set point
    or the value not served (by what is missing)."""
    found: dict[str, Counter[str]] = {}
    for device in devices:
        recipe = graph.recipe(device)
        name = recipe.recipe if recipe is not None else DEFAULT_RECIPE
        rows = [r for r in graph.card_sections(name) if r.kind == "ladder"]
        if not rows:
            continue
        thing = graph.thing(device, "device")
        hits = graph.roles(device)
        for row in rows:
            ladder = limit_ladder(graph, thing, hits, row)
            missing = list(ladder.left_out) + ([] if ladder.value else ["value"])
            key = f"not served: {', '.join(missing)}" if missing else HOP_FOUND
            found.setdefault(name, Counter())[key] += 1
    return found


#: How a ring-strip orders its members: by ring position, or by their number when
#: one has no position.
RING_BY_S, RING_BY_NUMBER = "s", "number"
_SAME_PLACE = "same place"


def ring_group(graph: PanelGraph, thing: Thing, recipe: str, members: str) -> list[Thing]:
    """The members of a ring-strip: ``same place`` (the devices of the card's recipe
    in the device's place, the device among them) or ``<relation> <direction>`` (the
    devices that relation names at the device)."""
    if members == _SAME_PLACE:
        place = graph.place_of(thing.id, "device")
        if place is None:
            return [thing]
        try:
            here = graph.within(place)
        except KeyError:
            return [thing]
        return [
            t
            for t in here
            if t.kind == "device" and getattr(graph.recipe(t.id), "recipe", None) == recipe
        ] or [thing]
    relation, _, direction = members.partition(" ")
    found: dict[str, Thing] = {}
    for edge in _near(graph, thing.id, relation, direction or "out"):
        if edge.other_kind == "device":
            try:
                found[edge.other] = graph.thing(edge.other, "device")
            except KeyError:
                continue
    return list(found.values())


def ring_strip(
    graph: PanelGraph, index: _Index, thing: Thing, recipe: str, section: CardSection
) -> RingStrip | None:
    """A card's ring-strip: the section's group (:func:`ring_group`), one column each,
    by ring position when every member has one, else in number order (the caption
    says so); each with the first of the section's roles it plays (a role not served
    leaves its column empty), the device in focus; ``None`` for no member."""
    members = ring_group(graph, thing, recipe, section.members or "")
    if not members:
        return None
    relation, _, direction = (section.members or "").partition(" ")
    resting: dict[str, str | None] = {}
    if section.members != _SAME_PLACE:  # members a relation names: on what basis
        near = _near(graph, thing.id, relation, direction or "out")
        resting = {m.id: inferred_rule(e for e in near if e.other == m.id) for m in members}
    roles = [r for r in (section.roles or "").split(";") if r]
    where = {m.id: _beam_s(graph, m) for m in members}
    if all(s_m is not None for s_m in where.values()):
        ordered = RING_BY_S
        members.sort(key=lambda m: (where[m.id], m.id))
    else:
        ordered = RING_BY_NUMBER
        members.sort(key=lambda m: name_order(m.id))
    out = []
    for member in members:
        rows: list[BlockRow] = []
        for hit in graph.roles(member.id):
            if hit.role.role in roles and hit.role.shown_as not in CONTROLS:
                unit, scale = _role_unit(graph, hit)
                rows.append(
                    BlockRow(
                        _row_word(hit.role), hit.channel, shown_as=hit.role.shown_as,
                        unit=unit, scale=scale, words=_words(graph, hit.channel),
                    )
                )  # fmt: skip
                break
        focus = member.id == thing.id
        item = None if focus else _ref(graph, index, member)
        label = member.short
        out.append(
            RingMember(
                member.id,
                label,
                item if isinstance(item, Link) else None,
                tuple(rows),
                focus,
                resting.get(member.id),
            )  # fmt: skip
        )
    labels = [m.label for m in out]
    if len(set(labels)) < len(labels):  # one short name for two members: each its id
        out = [dataclasses.replace(m, label=m.thing) for m in out]
    places: list[tuple[str, int]] = []
    held = _ring_places(graph, members)
    if held:  # by place, in ring order; those with none last
        first: dict[str, int] = {}
        for i, member in enumerate(members):
            if held.get(member.id) is not None:
                first.setdefault(held[member.id] or "", i)
        if ordered == RING_BY_NUMBER:  # no ring position: the places in number order
            named = sorted(first, key=name_order)
            first = {place: n for n, place in enumerate(named)}
        last = len(members)
        rank = {
            m.id: (first[held[m.id] or ""] if held.get(m.id) is not None else last, i)
            for i, m in enumerate(members)
        }
        out.sort(key=lambda m: rank[m.thing])
        members.sort(key=lambda m: rank[m.id])
        for member in members:
            run = held.get(member.id) or _PLACE_NOT_STATED
            if places and places[-1][0] == run:
                places[-1] = (run, places[-1][1] + 1)
            else:
                places.append((run, 1))
        start = 0
        for _run, k in places:  # inside a place a member's short name, its id on hover
            short = _short_labels(members[start : start + k])
            out[start : start + k] = [
                dataclasses.replace(m, label=short[m.thing]) for m in out[start : start + k]
            ]
            start += k
    caption = section.caption
    inferred = [m for m in out if m.inferred]
    if inferred and 2 * len(inferred) > len(out):  # said once, not under every member
        rules = "; ".join(dict.fromkeys(m.inferred or "" for m in inferred))
        said = (
            _ALL_INFERRED.format(len(out), rules)
            if len(inferred) == len(out)
            else _MOSTLY_INFERRED.format(rules)
        )
        caption = "; ".join(c for c in (caption, said) if c)
        marked = len(inferred) < len(out)
        out = [dataclasses.replace(m, inferred=None, stated=marked and not m.inferred) for m in out]
    if ordered == RING_BY_NUMBER and not (caption or "").lower().startswith("in number order"):
        caption = "; ".join(c for c in (_RING_NUMBER_NOTE, caption) if c)
    lamps = any(r.shown_as == "lamp" for m in out for r in m.rows)
    counted = f"giving {_role_title(roles[0]).lower()} now" if roles and lamps else None
    return RingStrip(tuple(out), ordered, caption, tuple(places), counted)


_RING_NUMBER_NOTE = "in number order: positions not stated"
_ALL_INFERRED = "all {} inferred: {}"
_MOSTLY_INFERRED = "inferred unless marked stated: {}"


def _short_labels(members: list[Thing]) -> dict[str, str]:
    """Each member's name inside its place: its short name, else the last segment of
    its id; members that would share one name each keep what follows the part of
    their ids they all share (``A:P1``, ``B:P1``)."""
    first = {m.id: m.short for m in members}
    seen = Counter(first.values())
    out = dict(first)
    for name, n in seen.items():
        if n > 1:
            ids = [i for i, v in first.items() if v == name]
            cut = len(os.path.commonprefix(ids))
            out.update({i: i[cut:] or i for i in ids})
    return out


#: A ring-strip runs by place at the finest level of the place tree that gives at
#: most this many places, so a row holds a group, not one member.
RING_PLACES_MOST = 8


def _ring_places(graph: PanelGraph, members: list[Thing]) -> dict[str, str | None]:
    """Each member's place at the finest level of the place tree whose places number
    at most :data:`RING_PLACES_MOST` and at least two (``None`` for a member in no
    place); empty when no level groups them."""
    paths: dict[str, list[str]] = {}
    for member in members:
        path: list[str] = []
        place = member.place
        while place is not None and place not in path:
            path.insert(0, place)
            try:
                place = graph.thing(place, "place").place
            except KeyError:
                break
        paths[member.id] = path
    deepest = max((len(p) for p in paths.values()), default=0)
    for depth in range(deepest, 0, -1):
        at = {
            m: (p[depth - 1] if len(p) >= depth else (p[-1] if p else None))
            for m, p in paths.items()
        }
        count = len({v for v in at.values() if v is not None})
        if 1 < count <= RING_PLACES_MOST:
            return at
    return {}


_PLACE_NOT_STATED = "place not stated"


def chain_states(graph: PanelGraph, devices: Iterable[str]) -> dict[str, Counter[str]]:
    """Per ``<recipe>/<chain>`` drawn on the cards of *devices*, how many came out
    each way: :data:`HOP_FOUND`, or the first hop's title and why it did not."""
    index = _index(graph)
    found: dict[str, Counter[str]] = {}
    for device in devices:
        thing = graph.thing(device, "device")
        recipe = graph.recipe(device)
        name = recipe.recipe if recipe is not None else DEFAULT_RECIPE
        rows = [r for r in graph.card_sections(name) if r.kind == "chain" and r.members]
        if not rows:
            continue
        hits = graph.roles(device)
        for row in rows:
            _chain, _used, state = recipe_chain(graph, index, thing, hits, name, row)
            found.setdefault(f"{name}/{row.members}", Counter())[state] += 1
    return found


def _internal(graph: PanelGraph, device: str) -> set[str]:
    """The device's channels the data marks as driver-internal (:data:`INTERNAL_ROLE`)."""
    return {
        h.channel for h in graph.roles(device) if h.role.role == INTERNAL_ROLE and h.owner == device
    }


def _listed(graph: PanelGraph, device: str, channels: list[Channel]) -> list[Channel]:
    """The device's channels a card or its list shows: all but the driver-internal."""
    internal = _internal(graph, device)
    return [c for c in channels if c.name not in internal]


def _owner_name(graph: PanelGraph, channel: str) -> str:
    """The class and id of the device a channel belongs to."""
    owner = graph.channel(channel).thing
    if owner is None:
        return "no device"
    try:
        thing = graph.thing(owner, "device")
    except KeyError:
        return owner
    return f"{thing.cls or 'device'} {thing.id}"


def _card_channels(
    graph: PanelGraph, thing: Thing, hits: list[RoleHit], channels: list[Channel]
) -> tuple[tuple[str, str], ...]:
    """Every channel a card is about, as (short name, channel): the device's own (not
    the driver-internal), then, of each other device its roles reach (a controller,
    an interlock), every channel beside the ones a role names (the same name up to
    its last ``:``: the controller's channels for this device's port), named short
    from that device."""
    rows = [(_short_channel(c.name, thing.id), c.name) for c in channels]
    seen = {c.name for c in channels}
    branches: dict[str, set[str]] = {}
    for hit in hits:
        if hit.owner != thing.id and hit.role.role != INTERNAL_ROLE:
            branches.setdefault(hit.owner, set()).add(hit.channel.rsplit(":", 1)[0] + ":")
    for owner, prefixes in branches.items():
        try:
            others = _listed(graph, owner, graph.channels(owner, "device"))
        except KeyError:
            continue
        for channel in others:
            if channel.name not in seen and channel.name.startswith(tuple(prefixes)):
                seen.add(channel.name)
                rows.append((_short_channel(channel.name, owner), channel.name))
    return _shared_cut(rows)


def _shared_cut(rows: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Rows whose short name is still the whole channel name (a name that does not
    start with its device's id) lose the ``:``-segments they all share."""
    whole = [short for short, channel in rows if short == channel and ":" in channel]
    if len(whole) < 2:
        return tuple(rows)
    parts = [name.split(":")[:-1] for name in whole]
    shared = 0
    while all(len(p) > shared and p[shared] == parts[0][shared] for p in parts):
        shared += 1
    if not shared:
        return tuple(rows)
    cut = ":".join(parts[0][:shared]) + ":"
    return tuple(
        (short.removeprefix(cut) if short == channel else short, channel) for short, channel in rows
    )


def _channel_count(
    thing: Thing, rows: tuple[tuple[str, str], ...], owners: Mapping[str, str], hidden: int
) -> ChannelCount:
    """The counts line for a device over :data:`GRID_CAP`: its own channels, each
    other device's (*owners*: channel to its device's class and id), the
    driver-internal records."""
    counts: Counter[str] = Counter()
    for _short, channel in rows:
        counts[owners.get(channel, f"{thing.cls or thing.kind} {thing.id}")] += 1
    parts = [f"{n} on {who}" for who, n in counts.items()]
    if hidden:
        parts.append(_INTERNAL_COUNT.format(hidden))
    link = Link("open the full list", Target("channels", "device", thing.id))
    return ChannelCount(len(rows), tuple(parts), link)


def channel_list(graph: PanelGraph, device_id: str) -> View:
    """A device's ``channels`` view: every channel its card is about (its own, and
    those of the devices its roles reach, as the card's grid takes them), grouped
    by the role it plays, then by record type, then the driver-internal records
    last; each row its name, value and unit. The one view that is a plain list by
    design.

    Raises:
        KeyError: No such device.
    """
    thing = graph.thing(device_id, "device")
    index = _index(graph)
    hits = graph.roles(device_id)
    role_of: dict[str, ChannelRole] = {}
    for hit in hits:
        role_of.setdefault(hit.channel, hit.role)
    own = graph.channels(device_id, "device")
    names = {c.name for c in own}
    reached = [
        graph.channel(name)
        for _short, name in _card_channels(graph, thing, hits, _listed(graph, device_id, own))
        if name not in names
    ]
    groups: dict[str, list[Item]] = {}
    last: list[Item] = []
    for channel in (*own, *reached):
        role = role_of.get(channel.name)
        if role is not None and role.role == INTERNAL_ROLE:
            last.append(_value_row(graph, channel))
            continue
        if role is not None:
            plate = role.role in graph.identity
            title = _plate_word(role).capitalize() if plate and not role.label else _role_word(role)
        else:
            row = graph.record_type(channel.name)
            title = (
                f"record type {row.record_type}" if row is not None else "record type not stated"
            )
        groups.setdefault(title, []).append(_value_row(graph, channel))
    roled = [k for k in groups if not k.startswith("record type")]
    typed = sorted(k for k in groups if k.startswith("record type"))
    unstated = f"record type {_NOT_STATED}"
    sections = [
        Section(k, tuple(groups[k]), "role" if k in roled else "record type")
        for k in (*roled, *typed)
        if k != unstated
    ]  # a stated group of its grouping each; the rest stand alone after them
    if unstated in groups:
        sections.append(Section(unstated, tuple(groups[unstated])))
    if last:
        sections.append(Section(_INTERNAL_COUNT.format(len(last)), tuple(last)))
    crumbs = (
        *_crumbs(graph, index, graph.place_of(thing.id, "device")),
        Link(thing.id, Target("card", "device", thing.id)),
    )
    return View(
        Target("channels", "device", device_id),
        f"{device_id} · all channels",
        crumbs,
        tuple(sections),
        alias=thing.short_name,
    )


def _drawn_card(
    graph: PanelGraph,
    index: _Index,
    thing: Thing,
    extra: list[tuple[str, list[Item]]],
    after: list[tuple[str, list[Item]]],
) -> tuple[Section, ...]:
    """The sections of a device's drawn card: the title line, the key tiles, the
    relations chain, the drawings *extra* holds (probe, picture, calibration), the
    operate block, every channel not drawn above, then *after* (where it is, its
    sources)."""
    recipe = graph.recipe(thing.id)
    cls = thing.cls or thing.kind
    kind = f"{cls} · {recipe.recipe} recipe" if recipe is not None else f"{cls} · default recipe"
    description = None
    if recipe is not None and recipe.description:
        description = _filled(recipe.description, graph, thing)
    hits = graph.roles(thing.id)
    every = graph.channels(thing.id, thing.kind)
    channels = _listed(graph, thing.id, every)  # the driver-internal never shown
    hidden = len(every) - len(channels)  # read only in the full list
    shared = _shared(graph, index, thing)
    drawn_shared = {
        row.channel for _title, items in shared for row in items if isinstance(row, ValueRow)
    }
    # once, under its supply: shown there, so not hidden and no reason for the full list
    channels = [c for c in channels if c.name not in drawn_shared]
    tiles = _key_tiles(graph, thing, hits)
    name = recipe.recipe if recipe is not None else DEFAULT_RECIPE
    drawings: list[tuple[str, list[Item]]] = []
    nowhere = False
    for row in graph.card_sections(name):
        if row.kind == "chain" and row.members:
            chain, _used, _state = recipe_chain(graph, index, thing, hits, name, row)
            drawings.append((row.title, [chain]))
        elif row.kind == "section" and row.focus:
            drawings.append((row.title, part_section(graph, index, thing, hits, name, row)))
        elif row.kind == "curve":
            drawings.append((row.title, [card_curve(graph, thing, row)]))
        elif row.kind == "ladder":
            drawings.append((row.title, [limit_ladder(graph, thing, hits, row)]))
        elif row.kind == "ring-strip":
            ring = ring_strip(graph, index, thing, name, row)
            if ring is not None:
                drawings.append((row.title, [ring]))
        elif row.kind == "front":
            drawings.append((row.title, [card_front(graph, index, thing, hits, name, row)]))
        elif row.kind == "strip":
            strip = card_strip(graph, index, thing, row)
            if strip is None:
                nowhere = True
            else:
                drawings.append((row.title, [strip]))
    if not graph.card_sections(name):  # no recipe rows: the built-in relations chain
        chain, _shown = _default_chain(graph, index, thing, hits)
        drawings.append((CARD_SECTIONS["relations"], [chain]))
    fronts = [item for _title, items in drawings for item in items if isinstance(item, Front)]
    on_front = {
        control.channel
        for item in fronts
        for group in item.groups
        for connector in group.connectors
        for control in connector.controls
    } | {control.channel for item in fronts for group in item.entries for control in group.controls}
    # a channel another device holds (a family's shared supply) is pressed on its card
    theirs = {c.name for c in every if c.thing_kind == "device" and c.thing not in (None, thing.id)}
    operate_hits = [h for h in hits if h.channel not in on_front | theirs]  # drawn on the front
    operate_channels = [c for c in channels if c.name not in on_front | theirs]  # nor as fallback
    operate, _in_operate, _unstated = _operate(graph, thing, operate_hits, operate_channels)
    columns = (
        int(recipe.channels_cols)
        if recipe is not None and recipe.channels_cols
        else _DEFAULT_COLUMNS
    )
    rows = _card_channels(graph, thing, hits, channels)
    listing: list[Item] = []
    if len(rows) > GRID_CAP:
        own = {c.name for c in channels}
        owners = {c: _owner_name(graph, c) for _s, c in rows if c not in own}
        listing = [_channel_count(thing, rows, owners, hidden)]
    elif rows or hidden:
        full = Link("in the full list", Target("channels", "device", thing.id)) if hidden else None
        listing = [ChannelGrid(rows, columns, hidden, full)]
    pairs: list[tuple[str, list[Item]]] = [
        (
            CARD_SECTIONS["head"],
            [CardHead(kind, description, recipe is None, nowhere, _by_whole(graph, thing))],
        ),
        (CARD_SECTIONS["tiles"], list(tiles)),
        *drawings,
        *shared,
        *extra,
        # a device whose presses all stand on its front has no operate block to say none
        (
            CARD_SECTIONS["operate"],
            [operate or Operate((), _NO_ACTIONS)] if every and (operate or not on_front) else [],
        ),
        (CARD_SECTIONS["channels"], listing),
        *after,
    ]
    return _sections(pairs)


# -- calibration -----------------------------------------------------------------

_WANTED = "{} wanted"
_FIRST = "first value"
_REPLACES = "replaces {}"
_EARLIER = "{} earlier"
_FROM_DESIGN = "{:+.2f} % from design"
_ALSO_CLAIMED = "also claimed: {}"
_CONTESTED_CLAIMS = "contested, not in force: {}"
_POINTS = "{} points"
_CONTESTED = "two sources disagree"
_LIVE = "EPICS"


def _with_unit(value: str, unit: str | None) -> str:
    """A claim's value as text: a tuple's numbers joined by ``·``, with the unit."""
    text = " · ".join(value.split(";"))
    return f"{text} {unit}" if unit else text


def _claim_text(claim: Calibration, kind: CalibrationKind, points: int = 0) -> str:
    if claim.value.startswith(CURVE_PREFIX):
        return _POINTS.format(points) if points else claim.value.removeprefix(CURVE_PREFIX)
    return _with_unit(claim.value, claim.unit or kind.unit)


def calibration_row(
    graph: PanelGraph, device_id: str, kind: CalibrationKind, link: Link | None = None
) -> CalibrationRow:
    """The card row of *kind* for a device: the value in force, its basis, date,
    source and history, or that there is none yet and which basis is wanted."""
    found = graph.calibration(device_id, kind.kind)
    claim = found.in_force
    channels = _held_in(graph, device_id, kind)
    disputed = ", ".join(
        f"{_claim_text(c, kind)} ({' '.join(t for t in (c.document, c.version) if t) or _NOT_STATED})"
        for c in found.contested
    )
    if claim is None:
        if found.contested:  # claimed, but contested: shown so, never as a base value
            return CalibrationRow(
                kind.title, None, None, _CONTESTED_CLAIMS.format(disputed), "contested",
                _CONTESTED, True, None, None, None, channels, link,
            )  # fmt: skip
        chip = _LIVE if channels else _WANTED.format(kind.wanted_basis)
        return CalibrationRow(
            kind.title, None, None, None, "none", chip, True, None, None, None, channels, link
        )
    value = _claim_text(claim, kind, len(found.points))
    note = None
    if found.contested:
        note = _CONTESTED_CLAIMS.format(disputed)
    elif found.rivals:
        note = _ALSO_CLAIMED.format(", ".join(_claim_text(c, kind) for c in found.rivals))
    elif found.design is not None and claim is not found.design:
        mine, design = calibration_numbers(claim.value), calibration_numbers(found.design.value)
        if mine and design and len(mine) == len(design) == 1 and design[0]:
            note = _FROM_DESIGN.format((mine[0] / design[0] - 1) * 100)  # of its size
    if found.points:
        unit = (kind.unit or "").split(";")[0]  # a curve's unit is x;y
        ends = ", ".join(f"{p.x:g} {unit}".strip() for p in (found.points[0], found.points[-1]))
        value = f"{value} ({ends})"
    if found.state == "contested":
        chip = _CONTESTED
    else:
        chip = f"{claim.basis} · {claim.method}" if claim.method else claim.basis
    replaced = next((c for c in found.earlier if c.id == claim.supersedes), None)
    if not found.earlier:
        history = _FIRST
    elif replaced is not None:
        history = _REPLACES.format(f"{replaced.basis} {_claim_text(replaced, kind)}")
        if len(found.earlier) > 1:
            history += " · " + _EARLIER.format(len(found.earlier))
    else:
        history = _EARLIER.format(len(found.earlier))
    source = " ".join(t for t in (claim.document, claim.version) if t) or None
    return CalibrationRow(
        kind.title,
        claim.conditions,
        value,
        note,
        found.state,
        _LIVE if channels else chip,
        found.below_wanted,
        claim.date,
        source,
        history,
        channels,
        link,
    )


def _held_in(graph: PanelGraph, device_id: str, kind: CalibrationKind) -> tuple[str, ...]:
    """The device's channels that state the kind's ``held_in`` signal, in channel order."""
    if kind.held_in is None:
        return ()
    return graph.signal_channels(device_id, kind.held_in)


def _calibration(graph: PanelGraph, index: _Index, thing: Thing) -> list[Item]:
    """A device's calibration block: a row per kind of it (:meth:`PanelGraph.applies`),
    each linking to the kind's view over the first set of the device's class it is
    in; or nothing."""
    if thing.kind != "device" or thing.cls is None:
        return []
    kinds = graph.calibration_kinds_of(thing.id)
    if not kinds:
        return []
    group = next(
        (g for g in index.sets_of.get(thing.id, ()) if index.set_class[g] == thing.cls), None
    )

    def link(kind: CalibrationKind) -> Link | None:
        if group is None:
            return None
        return Link(
            f"{group} {kind.title}", Target("calibration", "group", group, system=kind.kind)
        )

    return [
        Calibrations(tuple(calibration_row(graph, thing.id, kind, link(kind)) for kind in kinds))
    ]


_TALLY = "{} {} / {}"
_MORE_TALLIES = "more calibration kinds"
#: A tally's states, worst first.
TALLY_STATES = ("contested", "none", "below_wanted", "at_wanted")


def _tally_state(found: CalibrationState) -> str:
    if found.state == "contested":
        return "contested"
    if found.in_force is None:
        return "none"
    return "below_wanted" if found.below_wanted else "at_wanted"


def _calibration_summary(
    graph: PanelGraph, index: _Index, place_id: str, more: Target | None = None
) -> list[Item]:
    """The place's calibration chips (:class:`CalibrationTallies`): per kind title,
    over every class of a device under the place whose kind it is, how many are at
    their wanted basis uncontested, of how many; the worst state present; the kind's
    view over the set holding most of them. Titles with no value under the place
    are counted, not shown. Nothing when no kind applies under the place."""
    devices = [t for t in graph.contents(place_id) if t.kind == "device" and t.cls is not None]
    groups: dict[str, list[tuple[CalibrationKind, Thing]]] = {}
    rank: dict[str, tuple[float, str]] = {}
    for kind in graph.calibration_kinds():
        mine = [d for d in devices if graph.applies(kind, d.id)]
        if not mine:
            continue
        title = kind.tally_title
        groups.setdefault(title, []).extend((kind, d) for d in mine)
        order = (kind.order if kind.order is not None else math.inf, title)
        rank[title] = min(rank.get(title, order), order)
    if not groups:
        return []
    tallies: list[CalibrationTally] = []
    not_started = 0
    for title in sorted(groups, key=lambda t: rank[t]):
        pairs = groups[title]
        found = [(kind, d, graph.calibration(d.id, kind.kind)) for kind, d in pairs]
        if all(f.in_force is None for _k, _d, f in found):
            not_started += 1
            continue
        states = [_tally_state(f) for _k, _d, f in found]
        here = Counter(
            (group, kind.kind) for kind, d, _f in found for group in index.sets_of.get(d.id, ())
            if index.set_class.get(group) == d.cls
        )  # fmt: skip
        good = states.count("at_wanted")
        link = None
        if here:
            group, kind_id = min(here, key=lambda g: (-here[g], g))
            target = Target("calibration", "group", group, system=kind_id)
            link = Link(_TALLY.format(title, good, len(found)), target)
        worst = min(states, key=TALLY_STATES.index)
        basis = pairs[0][0].wanted_basis
        tallies.append(CalibrationTally(title, good, len(found), worst, basis, link))
    folded = Link(_MORE_TALLIES, more) if more is not None else None
    return [CalibrationTallies(tuple(tallies), not_started, folded)]


_DESIGN_TEXT = "design"
_NO_VALUE_SHORT = "no value"
_DEVIATION = "{:+.2f} %"
_VIEW_TITLE = "{} {}"
_CONTESTED_SHORT = "contested"


def calibration_cell(graph: PanelGraph, device_id: str, kind: CalibrationKind) -> CalibrationCell:
    """What a member's records say of *kind*, as a calibration view's cell shows it."""
    found = graph.calibration(device_id, kind.kind)
    channels = _held_in(graph, device_id, kind)
    claim = found.in_force
    if claim is None:
        if found.contested:
            return CalibrationCell(_CONTESTED_SHORT, "contested", channels)
        return CalibrationCell(_NO_VALUE_SHORT, "none", channels)
    if claim is found.design:
        return CalibrationCell(_DESIGN_TEXT, found.state, channels)
    design = calibration_numbers(found.design.value) if found.design is not None else None

    def shown(c: Calibration) -> str:
        mine = calibration_numbers(c.value)
        if mine and design and len(mine) == len(design) == 1 and design[0]:
            return _DEVIATION.format((mine[0] / design[0] - 1) * 100)
        return _claim_text(c, kind, len(found.points))

    text = " / ".join(shown(c) for c in (claim, *found.rivals))
    return CalibrationCell(text, found.state, channels)


def calibration_overview(graph: PanelGraph, group_id: str, kind_id: str) -> View:
    """A calibration kind over a set: the set overview's grid of its members of the
    kind's class (:func:`set_overview`), each cell what the member's records say
    (:func:`calibration_cell`). Static: it binds no channel but a ``held_in`` kind's.

    Raises:
        KeyError: No such group or kind.
    """
    graph.thing(group_id, "group")
    index = _index(graph)
    kind = graph.calibration_kind(kind_id, index.set_class.get(group_id) or "")
    members = [
        m for m in graph.members(group_id) if m.kind == "device" and graph.applies(kind, m.id)
    ]

    def cell(member: Thing) -> GridCell:
        found = calibration_cell(graph, member.id, kind)
        return GridCell(_ref(graph, index, member), (), None, found, member.short_name)

    grid = Grid(None, _member_rows(_member_groups(graph, members), cell), kind.title)
    group_link = Link(group_id, Target("set", "group", group_id))
    sections = (Section("group", (group_link,)), Section("members", (grid,)))
    target = Target("calibration", "group", group_id, system=kind_id)
    return View(target, _VIEW_TITLE.format(group_id, kind.title), (HOME_LINK,), sections)


# -- set overview ----------------------------------------------------------------


def _primary_signal(graph: PanelGraph, members: list[Thing]) -> str | None:
    """The stated signal of the setpoint of the pair most members have; ties by name."""
    count: Counter[str] = Counter()
    for member in members:
        signals = {
            channel.signal
            for channel in graph.channels(member.id, "device")
            if channel.writable and channel.signal and graph.pair(channel.name) is not None
        }
        count.update(signals)
    if not count:
        return None
    most = max(count.values())
    return min(signal for signal, n in count.items() if n == most)


def _row_places(graph: PanelGraph, members: list[Thing]) -> dict[str, str | None]:
    """Each member's row: its nearest place that has a sibling of the same level among
    the members' places (their places and every place around them), else its own place."""

    def own(member: Thing) -> str | None:
        if member.place is not None:
            return member.place
        claims = graph.neighbours(member.id, "device").get(PLACE_RELATION, ())
        return next((e.other for e in claims if e.outgoing and e.other_kind == "place"), None)

    def chain(place: str | None) -> list[Thing]:
        found = []
        while place is not None:
            thing = graph.thing(place, "place")
            found.append(thing)
            place = thing.place
        return found

    chains = {member.id: chain(own(member)) for member in members}
    every = {place.id: place for found in chains.values() for place in found}
    by_parent: dict[tuple[str | None, str | None], set[str]] = {}
    for place in every.values():
        by_parent.setdefault((place.place, place.level), set()).add(place.id)
    rows: dict[str, str | None] = {}
    for member in members:
        places = chains[member.id]
        sibling = next((p.id for p in places if len(by_parent[(p.place, p.level)]) > 1), None)
        rows[member.id] = sibling or (places[0].id if places else None)
    return rows


def set_overview(graph: PanelGraph, group_id: str, page: int = 1) -> View:
    """A group's devices as a grid: one row per place, the members in it in beam order.

    The row of a member is its nearest place (built, else the first a claim
    names) that has a sibling of the same level among the members' places, in
    practice the sector; the members with no place are a last row ``not
    placed``. Rows go in the graph's place order, members within a row by
    ring position (``s_m``, those with none after), then id. A cell is the
    member: its card, its rule-bearing channels (one light), and for the
    set's primary pair (:func:`_primary_signal`: keyed on the signal the
    facility states for the setpoint, never on the channel's name) the
    member's setpoint with readback and tolerance. No other channel is in
    the grid; the member's card has them.

    Raises:
        KeyError: No such group, or it is excluded.
    """
    graph.thing(group_id, "group")
    index = _index(graph)
    members = [member for member in graph.members(group_id) if member.kind == "device"]
    signal = _primary_signal(graph, members)
    pages = set_pages(index, group_id)
    if not 1 <= page <= len(pages):
        raise KeyError(f"the set {group_id} has no page {page}")
    groups = _set_pages(graph, members)[page - 1] if len(pages) > 1 else None

    def cell(member: Thing) -> GridCell:
        pair = None
        if signal is not None:
            for channel in graph.channels(member.id, "device"):
                if channel.writable and channel.signal == signal and graph.pair(channel.name):
                    pair = _value_row(graph, channel)
                    break
        channels = tuple(rule_channels(graph, member.id, "device"))
        tile = None
        if not channels and pair is None:  # no light, no pair: what its card shows first
            tiles = _key_tiles(graph, member, graph.roles(member.id))
            tile = next((t for t in tiles if t.channel is not None and not t.channels), None)
        return GridCell(
            _ref(graph, index, member), channels, pair, short_name=member.short_name, tile=tile
        )

    grid = Grid(signal, _member_rows(groups or _member_groups(graph, members), cell))
    group_link = Link(group_id, Target("card", "group", group_id))
    sections: tuple[Section, ...] = (Section("group", (group_link,)),)
    if len(pages) > 1:  # over one screen's budget: paged by its rows
        sections += (
            Section(
                _PAGES,
                tuple(Link(_PAGE.format(n, len(pages)), t) for n, t in enumerate(pages, 1)),
            ),
        )
    cls = index.set_class.get(group_id)
    views = [
        Link(
            kind.tally_title,  # the section's heading says "calibration"
            Target("calibration", "group", group_id, system=kind.kind),
        )
        for kind in (graph.calibration_kinds(cls) if cls is not None else ())
        if _applies_in(graph, kind, group_id)
    ]
    if views:
        sections += (Section("calibration", tuple(views)),)
    sections += (Section("members", (grid,)),)
    title = group_id if len(pages) == 1 else f"{group_id} · {_PAGE.format(page, len(pages))}"
    return View(pages[page - 1], title, (HOME_LINK,), sections)


def _member_groups(graph: PanelGraph, members: list[Thing]) -> list[tuple[str | None, list[Thing]]]:
    """A set's members by row (see :func:`set_overview`): each row's place (``None``
    for the members in no place, last) and its members in beam order."""
    rows_of = _row_places(graph, members)

    def beam(member: Thing) -> tuple[float, str]:
        s_m = _beam_s(graph, member)
        return (s_m if s_m is not None else math.inf, member.id)

    order = {thing.id: n for n, thing in enumerate(graph.things()) if thing.kind == "place"}
    grouped: dict[str | None, list[Thing]] = {}
    for member in members:
        grouped.setdefault(rows_of[member.id], []).append(member)
    keys = sorted((k for k in grouped if k is not None), key=lambda k: order[k])
    return [
        (key, sorted(grouped[key], key=beam))
        for key in ([*keys, None] if None in grouped else keys)
    ]


def _member_rows(
    groups: list[tuple[str | None, list[Thing]]], cell: Callable[[Thing], GridCell]
) -> tuple[GridRow, ...]:
    """Grid rows of *groups* (from :func:`_member_groups`), each cell from *cell*."""
    rows: list[GridRow] = []
    for key, members in groups:
        label: Link | Fact = (
            Link(key, Target("card", "place", key)) if key is not None else Fact("row", _NOT_IN)
        )
        rows.append(GridRow(label, tuple(cell(m) for m in members)))
    return tuple(rows)


#: The most members one page of a set view holds. Each cell is a light and, with a
#: pair, its field and deviation light: a few rules; this many keep a page inside
#: one screen's rule budget. A larger set is paged by its rows.
SET_PAGE = 96


def _set_pages(
    graph: PanelGraph, members: list[Thing]
) -> list[list[tuple[str | None, list[Thing]]]]:
    """A set's rows in pages of at most :data:`SET_PAGE` members: whole rows while
    they fit, a row longer than a page split across pages."""
    pages: list[list[tuple[str | None, list[Thing]]]] = [[]]
    held = 0
    for key, row in _member_groups(graph, members):
        while row:
            room = SET_PAGE - held
            if room <= 0 or (len(row) > room and held and len(row) <= SET_PAGE):
                pages.append([])  # a row that fits a page of its own starts one
                held, room = 0, SET_PAGE
            take, row = row[:room], row[room:]
            pages[-1].append((key, take))
            held += len(take)
    return pages if pages[0] else [[]]


def set_pages(index: _Index, group_id: str) -> list[Target]:
    """The pages of a set view, first to last."""
    return [
        Target("set", "group", group_id, system=None if n == 1 else str(n))
        for n in range(1, index.set_pages.get(group_id, 1) + 1)
    ]


# -- place view ------------------------------------------------------------------


def _band(graph: PanelGraph, place: Thing, view: str) -> Band | None:
    """The key numbers ``metrics.csv`` states for *view* of *place*: per row, the
    channels in the place (its own and its contents', :meth:`PanelGraph.contents`)
    that state the row's signal, on things of the row's class when it names one."""
    rows = graph.metrics(view, place.level)
    if not rows:
        return None
    things = [place, *graph.contents(place.id)]
    tiles = []
    for row in rows:
        chosen: list[Channel] = []
        for thing in things:
            if row.cls is not None and thing.cls != row.cls:
                continue
            chosen += [c for c in graph.channels(thing.id, thing.kind) if c.signal == row.signal]
        chosen = sorted({c.name: c for c in chosen}.values(), key=lambda c: name_order(c.name))
        pairs: list[tuple[str, str, str, float]] = []
        if row.measure == "count_in_tolerance":
            for channel in chosen:
                pair = graph.pair(channel.name)
                if channel.writable and pair is not None and channel.tolerance is not None:
                    term, amount = channel.tolerance
                    pairs.append((channel.name, pair.readback, term, amount))
            chosen = []
        names = tuple(c.name for c in chosen)
        tiles.append(
            Tile(row.title, row.measure, row.unit, row.caption, names, tuple(pairs), row.precision)
        )
    return Band(tuple(tiles))


def _annotations(
    graph: PanelGraph, thing: Thing, signals: Iterable[str]
) -> tuple[tuple[str, str | None], ...]:
    """Per stated signal, the device's first channel (by name) stating it, with its unit."""
    channels = sorted(graph.channels(thing.id, "device"), key=lambda c: name_order(c.name))
    found = []
    for signal in signals:
        channel = next((c for c in channels if c.signal == signal), None)
        if channel is not None:
            found.append((channel.name, channel.unit))
    return tuple(found)


def place_view(
    graph: PanelGraph,
    place_id: str,
    *,
    annotate: Mapping[str, tuple[str, ...]] = MappingProxyType({}),
) -> View:
    """The devices of a place's contents in beam order, the rest in a tray or counted.

    Only a device ``symbols.csv`` draws belongs on the beamline (a class meant to
    show without a picture of its own is given ``generic`` there), on the beam
    or, for lane ``above`` or ``below``, the lane off it. Such a device with a true ring position
    (:attr:`Pose.is_beam_position`) is a strip element; with none it is in
    the tray with ``no position`` or what its position assumes. Every other
    device, a ring position or not, is counted on one line, ``<n> other
    things in this place``, that opens the place's card (which lists them). Sections:
    ``strip``; ``places within``: the direct child places grouped by what the
    data states of them (``racks``: some claim names a unit in it; else its
    level; else ``other``), racks first and ``other`` last, each group at
    most :data:`MAX_CHIPS` with the rest counted in a link to the place's
    card; then ``by system``: the rows of the place's roll-up by system. The
    crumbs lead out to the parent.

    Args:
        graph: The panel graph.
        place_id: The place.
        annotate: Symbol -> the stated signals whose values its elements
            show: per signal, the device's first channel (by name) that
            states it, with its unit.

    Raises:
        KeyError: No such place, or it is excluded.
    """
    place = graph.thing(place_id, "place")
    index = _index(graph)
    elements: list[StripElement] = []
    tray: list[TrayEntry] = []
    others = 0
    devices = [t for t in graph.contents(place_id) if t.kind == "device"]
    beams = {t.id: _beam_of(graph, t) for t in devices}
    # one beam path per strip: the frame most of its drawn devices are placed in
    frames = Counter(
        found[0]
        for t in devices
        if graph.strip_symbol(t) is not None
        for found in [beams[t.id]]
        if found
    )
    frame = frames.most_common(1)[0][0] if frames else None
    s_of = _unwrapped(
        {t.id: found[1] for t in devices for found in [beams[t.id]] if found and found[0] == frame},
        graph.frame_length(frame) if frame is not None else None,
    )
    for thing in devices:
        ref = _ref(graph, index, thing)
        link = ref if isinstance(ref, Link) else None
        pose = _pose_of(graph, thing)
        s_m = s_of.get(thing.id)
        cls = thing.cls
        drawn_as = graph.strip_symbol(thing)
        drawn = cls is not None and drawn_as is not None
        if drawn_as is not None and cls is not None and s_m is not None:
            symbol = drawn_as.symbol
            lane = "beam" if drawn_as.lane == "beam" else "above"
            channels = tuple(rule_channels(graph, thing.id, "device"))
            elements.append(
                StripElement(
                    thing.id,
                    cls,
                    s_m,
                    symbol,
                    lane,
                    link,
                    channels,
                    short=thing.short_name,
                    annotations=_annotations(graph, thing, annotate.get(symbol, ())),
                )
            )
        elif drawn:
            reason = pose.derived_assumes if pose is not None and pose.derived_assumes else None
            elsewhere = beams[thing.id]
            if reason is None and elsewhere is not None:
                reason = _OTHER_FRAME.format(elsewhere[0])
            tray.append(TrayEntry(thing.id, thing.cls, reason or NO_POSITION, link))
        else:
            others += 1
    elements.sort(key=lambda element: (element.s_m, element.thing))
    extent = (elements[0].s_m, max(e.s_m for e in elements)) if elements else None
    card = Target("card", "place", place_id)
    rest = Link(_OTHERS.format(others), card) if others else None
    strip = Strip(tuple(elements), tuple(tray), extent, rest)
    groups: dict[str, list[Node]] = {}
    for child in graph.within(place_id):
        if child.kind != "place":
            continue
        kind = _RACKS if child.id in index.racked else child.level or _OTHER
        # a rack opens its elevation, where its units are; any other place its view
        view: ViewKind = "rack" if child.id in index.racked else "place"
        link = _place_link(graph, index, child.id, Target(view, "place", child.id))
        short = child.id.removeprefix(place_id + "/")
        groups.setdefault(kind, []).append(
            _node(short, Link(short, link.target), rule_channels(graph, child.id, "place"))
        )
    order = sorted(groups, key=lambda kind: (kind != _RACKS, kind == _OTHER, kind))
    within: list[Item] = []
    for kind in order:
        nodes = groups[kind]
        more = None
        if len(nodes) > MAX_CHIPS:
            more = Link(_MORE.format(len(nodes) - MAX_CHIPS), card)
        within.append(Group(kind, "chips", tuple(nodes[:MAX_CHIPS]), more))
    systems: list[Item] = []
    if place_id in index.ruled:
        *_band_, section = health_rollup(graph, place_id, "system").sections
        (root,) = section.items
        systems = [Group(_BY_SYSTEM, "rows", root.children)]  # type: ignore[union-attr]
    band = _band(graph, place, "place")
    rollup = _existing(graph, index, Target("rollup", "place", place_id, by="place"))
    sections = _sections(
        [
            ("key numbers", [band] if band is not None else []),
            ("calibration", _calibration_summary(graph, index, place_id, rollup)),
            ("strip", [strip]),
            ("places within", within),
            (_BY_SYSTEM, systems),
            (SETS_SECTION, _sets_within(index, _devices_in(graph, place_id))),
        ]
    )
    crumbs = _crumbs(graph, index, place.place, "place")
    return View(Target("place", "place", place_id), place_id, crumbs, sections)


# -- rack elevation --------------------------------------------------------------


def _size(value: object) -> int | float | None:
    """A ``size_u`` attribute as a number: an integer stays one; text that is no number is ``None``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def rack_elevation(graph: PanelGraph, rack_id: str) -> View:
    """A rack's devices: those a ``located_in`` claim puts at a unit in it, then the rest.

    Every distinct (device, unit) that a claim into the rack names is one
    unit, whatever the device's built place: a device claimed into two racks
    is in both, and a device whose claims name two units in one rack is
    drawn at both, each marked ``conflicting``. A panel shows the model's
    contradiction rather than pick a side. Claims from several sources that
    agree on one (device, unit) are one unit carrying all their citations.
    Units are listed in :data:`RACK_ORDER` (the highest unit first), ties by
    id, each with the claims' standing and the first ``size_u`` that parses
    as a number and the half (``side`` ``L`` or ``R``) of a half-width unit. The
    devices whose built place is the rack and that no unit claim
    positions are ``unpositioned``, by id, and so are those a claim puts in the
    rack with no unit, or at a unit that is no integer (with the unit's text).
    Each entry links to the device's card and carries a :class:`Light` per
    rule channel of the device and of the parts placed through it, each once.

    Raises:
        KeyError: No such place, or it is excluded.
    """
    rack = graph.thing(rack_id, "place")
    index = _index(graph)

    def lights(thing: Thing) -> tuple[Light, ...]:
        """The device's rule channels and those of the parts placed through it, each once."""
        names = dict.fromkeys(rule_channels(graph, thing.id, "device"))
        for part in graph.parts_through(thing.id):
            names.update(dict.fromkeys(rule_channels(graph, part.id, "device")))
        return tuple(light(graph, name) for name in names)

    # (device, unit) -> its claims into this rack, in graph order
    slots: dict[tuple[str, int], list[Edge]] = {}
    for claim in graph.neighbours(rack_id, "place").get(PLACE_RELATION, ()):
        if not claim.outgoing and claim.other_kind == "device" and claim.unit is not None:
            slots.setdefault((claim.other, claim.unit), []).append(claim)
    units_of = Counter(device for device, _unit in slots)
    positioned: list[tuple[int, str, RackUnit]] = []
    for (device, position), claims in slots.items():
        thing = graph.thing(device, "device")
        sizes = [_size(claim.attributes.get("size_u")) for claim in claims]
        citations = dict.fromkeys(claim.source for claim in claims if claim.source is not None)
        contested = any(claim.standing == "contested" for claim in claims)
        sides = [claim.attributes.get("side") for claim in claims]
        side = next((str(v) for v in sides if v in ("L", "R")), None)
        entry = RackUnit(
            position,
            _ref(graph, index, thing),
            lights(thing),
            next((size for size in sizes if size is not None), None),
            "contested" if contested else "accepted",
            conflicting=units_of[device] > 1,
            citations=tuple(citations),
            side=side,
        )
        positioned.append((-position, device, entry))
    loose: dict[str, RackUnit] = {
        thing.id: RackUnit(None, _ref(graph, index, thing), lights(thing))
        for thing in graph.within(rack_id)
        if thing.kind == "device" and thing.id not in units_of
    }
    texts: dict[str, list[str]] = {}
    for claim in graph.neighbours(rack_id, "place").get(PLACE_RELATION, ()):
        if claim.outgoing or claim.other_kind != "device" or claim.other in units_of:
            continue
        # claimed in with no unit, or a unit that is no integer: unpositioned here
        stated = texts.setdefault(claim.other, [])
        if claim.unit_text is not None:
            stated.append(claim.unit_text)
    for device, stated in texts.items():
        thing = graph.thing(device, "device")
        text = ", ".join(dict.fromkeys(stated)) or None
        if device in loose and text is None:
            continue
        loose[device] = RackUnit(None, _ref(graph, index, thing), lights(thing), unit_text=text)
    unpositioned = list(loose.items())
    elevation = Elevation(
        tuple(entry[-1] for entry in sorted(positioned, key=lambda e: e[:2])),
        tuple(entry[-1] for entry in sorted(unpositioned, key=lambda e: e[0])),
        RACK_ORDER,
    )
    crumbs = _crumbs(graph, index, rack.place, "place")
    sections = (Section("units", (elevation,)),)
    return View(Target("rack", "place", rack_id), rack_id, crumbs, sections)


# -- health roll-up --------------------------------------------------------------


def _node(
    label: str, link: Link | None, channels: Iterable[str], children: tuple[Node, ...] = ()
) -> Node:
    read = tuple(dict.fromkeys(channels))
    return Node(label, link, read, not read, children)


def _device_node(graph: PanelGraph, index: _Index, thing: Thing) -> Node:
    """A device in a roll-up: its card, or, when its card cannot read its rule-bearing
    channels (it lists counts), the first page of its own roll-up."""
    ref: Link | Fact = _ref(graph, index, thing)
    if thing.id in index.paged:
        ref = Link(thing.id, Target("rollup", "device", thing.id, by="device"))
    return _node(
        thing.id, ref if isinstance(ref, Link) else None, rule_channels(graph, thing.id, "device")
    )


#: The most rule-bearing channels one page of a device's roll-up reads. Its lights are
#: drawn as parts, each one rule over as many channels as one rule holds; this many
#: keep a page well inside one screen's rule budget.
DEVICE_PAGE = 400
_PAGES = "pages"
_PAGE = "page {} of {}"


def device_rollup_pages(index: _Index, device_id: str) -> list[Target]:
    """The pages of a device's own roll-up, first to last; none for most devices."""
    pages = index.paged.get(device_id, 0)
    return [
        Target("rollup", "device", device_id, by="device", system=None if n == 1 else str(n))
        for n in range(1, pages + 1)
    ]


def device_rollup(graph: PanelGraph, device_id: str, page: int = 1) -> View:
    """One page of a device's own roll-up: :data:`DEVICE_PAGE` of its rule-bearing
    channels in name order, read by the root, which has no children (the layout draws
    them as parts, one light per rule). Every rule-bearing channel of the device is on
    exactly one page. With more than one page, a ``pages`` section links them all.

    Raises:
        KeyError: The device has no roll-up of its own, or no such page.
    """
    thing = graph.thing(device_id, "device")
    index = _index(graph)
    pages = device_rollup_pages(index, device_id)
    if not 1 <= page <= len(pages):
        raise KeyError(f"{device_id} has no roll-up page {page}")
    ruled = sorted(rule_channels(graph, device_id, "device"))
    part = ruled[(page - 1) * DEVICE_PAGE : page * DEVICE_PAGE]
    label = device_id if len(pages) == 1 else f"{device_id} · {_PAGE.format(page, len(pages))}"
    root = _node(label, None, part)
    sections: tuple[Section, ...] = ()
    if len(pages) > 1:
        links = tuple(
            Link(_PAGE.format(n, len(pages)), target) for n, target in enumerate(pages, 1)
        )
        sections += (Section(_PAGES, links),)
    crumbs = (
        *_crumbs(graph, index, graph.place_of(thing.id, "device"), "rollup", "place"),
        Link(device_id, Target("card", "device", device_id)),
    )
    target = pages[page - 1]
    return View(target, f"{label} health", crumbs, (*sections, Section("roll-up", (root,))))


def _unplaced_node(graph: PanelGraph, index: _Index) -> Node | None:
    """The child "not placed" of a root roll-up: every device in no place, linked down."""
    if not index.unplaced_ruled:
        return None
    return _node(NOT_PLACED, Link(NOT_PLACED, _UNPLACED_ROLLUP), unplaced_channels(graph))


def _place_children(graph: PanelGraph, index: _Index, root_id: str, by: str) -> list[Node]:
    """A by-place roll-up's children: what is built directly in the place, then what a
    claim puts directly in it and is not built within it, then the parts placed
    directly in it by their whole that are not there already."""
    children: list[Node] = []
    direct = graph.within(root_id)
    seen = {(thing.kind, thing.id) for thing in graph.within(root_id, deep=True)}
    claimed = [t for t in graph.claimed(root_id) if (t.kind, t.id) not in seen]
    seen |= {(thing.kind, thing.id) for thing in claimed}
    parts = [t for t in graph.derived(root_id) if (t.kind, t.id) not in seen]
    for child in [*direct, *claimed, *parts]:
        if child.kind == "place":
            link = _place_link(graph, index, child.id, Target("rollup", "place", child.id, by))
            children.append(_node(child.id, link, rule_channels(graph, child.id, "place")))
        else:
            children.append(_device_node(graph, index, child))
    return children


#: The fewest children on a ring for a place's roll-up to draw them as arcs.
MIN_RING_ARCS = 2


def on_a_ring(graph: PanelGraph, place_id: str) -> bool:
    """Whether a place's roll-up draws its children as arcs of a ring: at least
    :data:`MIN_RING_ARCS` of them lie along one beam path the facility file marks
    ``periodic`` (:meth:`PanelGraph.frame_length`), and they go round it: no gap
    between their positions, taken round the ring, is half its length or more (a
    part of the ring, such as one sector, is no ring)."""
    spans, length = _child_spans(graph, place_id)
    if not length or len(spans) < MIN_RING_ARCS:
        return False
    found = sorted(s for span in spans.values() for s in span)
    gaps = [b - a for a, b in zip(found, found[1:], strict=False)]
    gaps.append(found[0] + length - found[-1])
    return max(gaps) < length / 2


def _child_spans(graph: PanelGraph, place_id: str) -> tuple[dict[str, list[float]], float | None]:
    """The positions of what each child place of *place_id* holds along the beam
    path most of them are on, and that path's length (``None`` for a line)."""
    placed: dict[str, list[tuple[str, float]]] = {}
    for child in graph.within(place_id):
        if child.kind != "place":
            continue
        found = [
            where
            for thing in graph.contents(child.id)
            for where in (graph.beam_position(thing),)
            if where is not None
        ]
        if found:
            placed[child.id] = found
    frames = Counter(frame for found in placed.values() for frame, _s in found)
    frame = frames.most_common(1)[0][0] if frames else None
    spans = {
        child: sorted(s for f, s in found if f == frame)
        for child, found in placed.items()
        if any(f == frame for f, _s in found)
    }
    return spans, graph.frame_length(frame) if frame is not None else None


def ring_arcs(graph: PanelGraph, place_id: str, *, ring_only: bool = False) -> list[str]:
    """The child places of *place_id* that lie on the ring, in beam order.

    A child lies on the ring when what it holds (:meth:`PanelGraph.contents`)
    has a position along the beam path most of them are on. Its span starts after
    the widest gap between its positions, taken round the ring when the path's
    length is stated (:meth:`PanelGraph.frame_length`), so a child across the
    ring's start is ordered by where its span begins, not by its position nearest
    zero.
    """
    spans, length = _child_spans(graph, place_id)
    if ring_only and not length:
        return []

    def start(found: list[float]) -> float:
        gaps = [(b - a, b) for a, b in zip(found, found[1:], strict=False)]
        if length:
            gaps.append((found[0] + length - found[-1], found[0]))  # round the ring's start
        else:
            gaps.append((math.inf, found[0]))  # a line: a span starts at its lowest
        return max(gaps)[1]

    return sorted(spans, key=lambda child: (start(spans[child]), child))


def _beam_key(graph: PanelGraph, label: str) -> tuple[float, str]:
    """Beam order for a node of a thing with a beam position, else name order after."""
    try:
        s_m = _beam_s(graph, graph.thing(label))
    except KeyError:
        s_m = None
    return (s_m if s_m is not None else math.inf, label)


def _ordered(
    graph: PanelGraph,
    children: list[Node],
    rest: Target,
    *,
    sort: bool = True,
    tail: Node | None = None,
) -> list[Node]:
    """A roll-up's children: those with a rule (in beam, else name, order unless *sort*
    is false), then *tail*, then one row ``<n> without a rule`` that opens *rest*."""
    ruled = [node for node in children if not node.no_rule]
    if sort:
        ruled.sort(key=lambda node: _beam_key(graph, node.label))
    if tail is not None:
        ruled.append(tail)
    without = sum(node.no_rule for node in children)
    if without:
        label = _WITHOUT.format(without)
        ruled.append(Node(label, Link(label, rest), (), True))
    return ruled


def _unplaced_rollup(graph: PanelGraph, index: _Index) -> View:
    """The roll-up of what is in no place: one node per device, then the channels on no thing."""
    children = [_device_node(graph, index, thing) for thing in graph.unplaced()]
    unheld = [channel.name for channel in graph.unheld() if graph.rule(channel.name)]
    tail = _node(_NO_THING, None, unheld) if unheld else None
    children = _ordered(graph, children, _UNPLACED_CHECKOUT, tail=tail)
    root = _node(NOT_PLACED, None, unplaced_channels(graph), tuple(children))
    return View(
        _UNPLACED_ROLLUP, f"{NOT_PLACED} health", (HOME_LINK,), (Section("roll-up", (root,)),)
    )


#: The section of a place view and a roll-up of one system that opens the sets.
SETS_SECTION = "sets"


def _sets_within(index: _Index, devices: Iterable[str]) -> list[Item]:
    """A link to the set view of every set one of *devices* is a member of, in set id
    order."""
    found = {group for device in devices for group in index.sets_of.get(device, ())}
    return [Link(group, Target("set", "group", group)) for group in sorted(found, key=name_order)]


def _devices_in(graph: PanelGraph, place_id: str) -> list[str]:
    return [thing.id for thing in graph.contents(place_id) if thing.kind == "device"]


def _system_rollup(
    graph: PanelGraph,
    index: _Index,
    place: Thing,
    system: str,
    systems: Mapping[str | None, list[Node]],
) -> View:
    """One system within a place: a node per device of the system."""
    key = None if system == NO_SYSTEM and None in systems else system
    devices = list(systems.get(key, ()))
    children = _ordered(graph, devices, Target("card", "place", place.id))
    root = _node(system, None, (c for node in devices for c in node.channels), tuple(children))
    crumbs = (
        *_crumbs(graph, index, place.place, "rollup", "system"),
        _place_link(graph, index, place.id, Target("rollup", "place", place.id, by="system")),
    )
    target = Target("rollup", "place", place.id, by="system", system=system)
    title = f"{place.id} {system} health"
    # the sets of this system's devices here: a set's members, not its class name,
    # since the draw settings may group classes into one system
    sets = _sections([(SETS_SECTION, _sets_within(index, (n.label for n in devices)))])
    return View(target, title, crumbs, (*sets, Section("roll-up", (root,))))


def health_rollup(
    graph: PanelGraph,
    root_id: str,
    by: str = "place",
    *,
    system_of: Callable[[Thing], str | None] | None = None,
    system: str | None = None,
) -> View:
    """The channels each node of a place's roll-up reads: the root and one level below it.

    What a place holds is :meth:`PanelGraph.contents`. By ``place``, the
    nodes are the place's direct children (built there, or claimed there and
    built elsewhere); a child place links down to its own roll-up (its card
    when it has none), a device to its card. By ``system``, the devices of
    the place's contents are grouped by ``system_of`` (default: the device's
    class), the groups in name order and :data:`NO_SYSTEM` last; a group node
    carries the union of its devices' channels and no children, and links to
    the roll-up of that system within the place (with the default grouping).
    With *system*, the roll-up is that one system within the place: a node
    per device of it. A node that reads no rule-bearing channel is present
    and marked ``no_rule``. No node below the root has children: one screen
    shows one level. The children with a rule come first, in beam order (a
    device with a true ring position) then name order, by system in name
    order; then one row ``<n> without a rule`` for all the others, which opens
    the place's card, where they are listed. The place's own rule-bearing
    channels are a child of their own, ``<place> itself``, so a screen whose
    root is too long for one rule still shows them live. By ``place``, a
    section ``by system`` before the tree holds the rows of the place's
    roll-up by system, each opening it. By ``place``, a place named in *ring*
    has a section ``ring`` before them: a group of form ``ring`` holding the
    nodes of its children on the ring (:func:`ring_arcs`), in beam order,
    those without a rule too.

    A root place (one with no parent) has one more child, ``not placed``:
    every device in no place, linked to their own roll-up, whose channels the
    root reads too. :data:`NOT_PLACED` as *root_id* is that roll-up.

    Raises:
        KeyError: No such place, or it is excluded.
        ValueError: *by* is neither ``place`` nor ``system``.
    """
    if by not in ("place", "system"):
        raise ValueError(f"a roll-up goes by 'place' or 'system', not by {by!r}")
    index = _index(graph)
    if root_id == NOT_PLACED and not _holds(graph, root_id, "place"):
        if by != "place":
            raise KeyError(f"the {NOT_PLACED} roll-up goes by place only")
        return _unplaced_rollup(graph, index)
    place = graph.thing(root_id, "place")
    children: list[Node] = []
    if by == "place":
        children = _place_children(graph, index, root_id, by)
    else:
        key = system_of if system_of is not None else (lambda thing: thing.cls)
        systems: dict[str | None, list[Node]] = {}
        for thing in graph.contents(root_id):
            if thing.kind == "device":
                systems.setdefault(key(thing), []).append(_device_node(graph, index, thing))
        if system is not None:
            return _system_rollup(graph, index, place, system, systems)
        names = sorted(name for name in systems if name is not None)
        for name in [*names, None] if None in systems else names:
            read = (channel for node in systems[name] for channel in node.channels)
            label = name if name is not None else NO_SYSTEM
            down = Target("rollup", "place", root_id, by="system", system=label)
            link = Link(label, down) if system_of is None and _has(graph, index, down) else None
            children.append(_node(label, link, read))
    channels = rule_channels(graph, root_id, "place")
    unplaced = _unplaced_node(graph, index) if place.place is None else None
    if unplaced is not None:
        channels += unplaced.channels
    rest = Target("card", "place", root_id)
    own = [c.name for c in graph.channels(root_id, "place") if graph.rule(c.name)]
    if own:  # the place's own channels are no child's: a node of their own
        label = _OWN.format(root_id)
        children.insert(0, _node(label, Link(label, rest), own))
    arcs: list[Node] = []
    if by == "place" and on_a_ring(graph, root_id):
        on_ring = {node.label: node for node in children}
        arcs = [on_ring[label] for label in ring_arcs(graph, root_id) if label in on_ring]
    children = _ordered(graph, children, rest, sort=by == "place", tail=unplaced)
    root = _node(root_id, None, channels, tuple(children))
    crumbs = _crumbs(graph, index, place.place, "rollup", by)
    title = f"{root_id} health by {by}"
    band = _band(graph, place, "rollup")
    head: tuple[Section, ...] = (Section("key numbers", (band,)),) if band is not None else ()
    if by == "place":
        head += _sections([("calibration", _calibration_summary(graph, index, root_id))])
    if arcs:
        head += (Section(_RING, (Group(_RING, "ring", tuple(arcs)),)),)
    if by == "system":  # the other way down: the sets of the place's devices
        head += _sections([(SETS_SECTION, _sets_within(index, _devices_in(graph, root_id)))])
    if by == "place" and root_id in index.ruled:  # beside the places: the same by system
        *_band_, section = health_rollup(graph, root_id, "system").sections
        (system_root,) = section.items
        rows = Group(_BY_SYSTEM, "rows", system_root.children)  # type: ignore[union-attr]
        head += (Section(_BY_SYSTEM, (rows,)),)
    return View(
        Target("rollup", "place", root_id, by=by),
        title,
        crumbs,
        (*head, Section("roll-up", (root,))),
    )


# -- home ------------------------------------------------------------------------

_TOP_PLACES = "places"
_NOT_PLACED_SECTION = "not placed"
_OTHER_GROUPS = "other groups"
_MACHINES = "machines"
_OTHER_PLACES = "other places"
_BY_CLASS = "by class"


def home(graph: PanelGraph) -> View:
    """The one screen above every other: the ways down, by place and by system.

    Sections, each left out when empty: ``places``, one chip per top place (a
    place in no other), opening its place view, else its roll-up by place, else
    its card, the machines first (a top place that holds a device of a model)
    and the other places below them; ``sets``, one chip per set view, under its class, then
    ``other groups``: each group with no set view (its members of several
    classes, or no device), opening its card; ``not placed``, the roll-up of what is in no
    place when there is one, then one chip per class of the devices in no place
    that get a card, with their count, each opening that class's index
    (:func:`unplaced_index`), which opens each card. A device in no place is
    reached that way or from no screen at all.

    Under a write scope (the graph's ``write_scope``, set by
    :func:`~osprey_panels.emit.scoped`) it lists only what the scope keeps
    and says once, first, which places the write keeps to.
    """
    index = _index(graph)
    scope = graph.write_scope
    machines = _machine_places(graph)
    if scope is not None:  # a kept place at the top is a machine when it holds a model's device
        machines = frozenset(
            place
            for place in scope.places
            if graph.thing(place, "place").place not in scope.places
            and any(t.kind == "device" and t.model for t in graph.contents(place))
        )
    tops: list[Node] = []
    others: list[Node] = []
    for thing in graph.things():
        if thing.kind != "place":
            continue
        if scope is not None:  # the kept places whose parent is not kept stand at the top
            if thing.id not in scope.places or thing.place in scope.places:
                continue
        elif thing.place is not None:
            continue
        link = None
        for view, by in (("place", None), ("rollup", "place"), ("card", None)):
            target = Target(view, "place", thing.id, by=by)  # type: ignore[arg-type]
            if _has(graph, index, target):
                link = Link(thing.id, target)
                break
        (tops if thing.id in machines or not machines else others).append(_node(thing.id, link, ()))
    by_class: dict[str, list[Node]] = {}
    for group in sorted(index.sets, key=name_order):
        if scope is not None and group not in scope.groups:
            continue
        cls = index.set_class[group]
        set_link = Link(group, Target("set", "group", group))
        by_class.setdefault(cls, []).append(_node(group, set_link, ()))
    # a group of several classes, or of no device, has no set view: its card
    unset = sorted(
        t.id
        for t in graph.things()
        if t.kind == "group" and t.id not in index.sets and (scope is None or t.id in scope.groups)
    )
    groups = [_node(group, Link(group, Target("card", "group", group)), ()) for group in unset]
    not_placed: list[Item] = []
    if _has(graph, index, _UNPLACED_ROLLUP):
        not_placed.append(Link(f"{NOT_PLACED} health", _UNPLACED_ROLLUP))
    classes = [
        _node(f"{cls} ({len(found)})", Link(cls, _unplaced_page(cls)), ())
        for cls, found in _loose_kept(graph, index).items()
    ]
    if classes:  # one index per class: every card of a device in no place is reached
        not_placed.append(Group(_BY_CLASS, "chips", tuple(classes)))
    places: list[Item] = []
    if scope is not None:
        places.append(Fact(_SCOPED, _SCOPED_TEXT.format(", ".join(scope.under))))
    if tops:
        places.append(Group(_MACHINES if machines else _TOP_PLACES, "chips", tuple(tops)))
    if others:
        places.append(Group(_OTHER_PLACES, "chips", tuple(others)))
    sets: list[Item] = [
        Group(cls, "chips", tuple(by_class[cls])) for cls in sorted(by_class, key=name_order)
    ]
    if groups:
        sets.append(Group(_OTHER_GROUPS, "chips", tuple(groups)))
    sections = _sections(
        [(_TOP_PLACES, places), (SETS_SECTION, sets), (_NOT_PLACED_SECTION, not_placed)]
    )
    return View(HOME_TARGET, HOME, (), sections)


_SCOPED = "this write"
_SCOPED_TEXT = "keeps to {}; the screens outside it are not written"


def _loose_kept(graph: PanelGraph, index: _Index) -> dict[str, tuple[str, ...]]:
    """The devices in no place that get a card, by class; under a write scope, only
    the kept ones."""
    scope = graph.write_scope
    out: dict[str, tuple[str, ...]] = {}
    for cls, found in index.loose.items():
        kept = tuple(d for d in found if scope is None or d in scope.devices)
        if kept:
            out[cls] = kept
    return out


def _machine_places(graph: PanelGraph) -> frozenset[str]:
    """The top places that hold a device of a model (the facility's machines)."""
    found: set[str] = set()
    for thing in graph.things():
        if thing.kind != "device" or thing.model is None or thing.place is None:
            continue
        top = thing.place
        while (parent := graph.thing(top, "place").place) is not None:
            top = parent
        found.add(top)
    return frozenset(found)


def _unplaced_page(cls: str) -> Target:
    return Target("home", "place", HOME, by=UNPLACED_BY, system=cls)


def unplaced_index(graph: PanelGraph, cls: str) -> View:
    """The devices of one class in no place that get a card: one chip each, opening it.

    Raises:
        KeyError: No device of that class is in no place with a card.
    """
    index = _index(graph)
    loose = _loose_kept(graph, index)
    if cls not in loose:
        raise KeyError(f"no device of class {cls!r} is in no place with a card")
    chips = tuple(
        _node(device, Link(device, Target("card", "device", device)), ()) for device in loose[cls]
    )
    section = Section(_NOT_PLACED_SECTION, (Group(cls, "chips", chips),))
    title = f"{cls} · {NOT_PLACED}"
    return View(_unplaced_page(cls), title, (HOME_LINK,), (section,))


# -- checkout --------------------------------------------------------------------


def _sourced(graph: PanelGraph, channel: str) -> bool:
    record_type = graph.record_type(channel)
    pair = graph.pair(channel)
    return bool(
        (record_type is not None and record_type.source)
        or (pair is not None and pair.source)
        or any(row.source for row in graph.rule(channel))
    )


def _check_rows(graph: PanelGraph, things: Iterable[Thing]) -> list[CheckRow]:
    rows: dict[str, CheckRow] = {}
    for thing in things:
        if thing.kind != "device":
            continue
        for channel in graph.channels(thing.id, "device"):
            if channel.name in rows:
                continue
            rows[channel.name] = _check_row(graph, channel, thing.id)
    return list(rows.values())


def _typed(row: RecordType | None) -> bool:
    """Whether a ``record_types.csv`` row types the channel: it states a record type
    whose drawing (stated, or its role's when left empty) is not ``untyped``."""
    return row is not None and row.drawn_as != "untyped"


def _check_row(graph: PanelGraph, channel: Channel, thing: str | None) -> CheckRow:
    return CheckRow(
        channel=channel.name,
        thing=thing,
        on_device=thing is not None,
        typed=_typed(graph.record_type(channel.name)),
        has_rule=bool(graph.rule(channel.name)),
        has_source=_sourced(graph, channel.name),
    )


def _totals(found: list[CheckRow]) -> Totals:
    return Totals(
        channels=len(found),
        on_device=sum(row.on_device for row in found),
        typed=sum(row.typed for row in found),
        has_rule=sum(row.has_rule for row in found),
        has_source=sum(row.has_source for row in found),
    )


def checkout(graph: PanelGraph, place_id: str) -> View:
    """One row per channel of the devices a place holds (:meth:`PanelGraph.contents`), with totals.

    A row says whether the channel is on a device, typed (a ``record_types.csv``
    row states its record type, not drawn ``untyped``), rule-bearing, and sourced (a record type, pair
    or health row names a source). Sections: ``channels``, ``totals``, then
    ``places within``: links to each child place's checkout (its card when it
    has none), so a table cut on screen leaves no row unreachable. A root
    place also links to the checkout of what is in no place; that checkout is
    :data:`NOT_PLACED` as *place_id*.

    Raises:
        KeyError: No such place, or it is excluded.
    """
    index = _index(graph)
    if place_id == NOT_PLACED and not _holds(graph, place_id, "place"):
        found = _check_rows(graph, graph.unplaced())
        sections = _sections([("channels", list(found)), ("totals", [_totals(found)])])
        return View(_UNPLACED_CHECKOUT, f"{NOT_PLACED} checkout", (HOME_LINK,), sections)
    place = graph.thing(place_id, "place")
    found = _check_rows(graph, graph.contents(place_id))
    if place.place is None:
        found += [_check_row(graph, channel, None) for channel in graph.unheld()]
    children: list[Item] = [
        _place_link(graph, index, child.id, Target("checkout", "place", child.id))
        for child in graph.within(place_id)
        if child.kind == "place"
    ]
    if place.place is None and _has(graph, index, _UNPLACED_CHECKOUT):
        children.append(Link(NOT_PLACED, _UNPLACED_CHECKOUT))
    sections = _sections(
        [("channels", list(found)), ("totals", [_totals(found)]), ("places within", children)]
    )
    crumbs = _crumbs(graph, index, place.place, "checkout")
    return View(Target("checkout", "place", place_id), f"{place_id} checkout", crumbs, sections)


# -- any view --------------------------------------------------------------------


def describe(
    graph: PanelGraph,
    target: Target,
    *,
    annotate: Mapping[str, tuple[str, ...]] = MappingProxyType({}),
    system_of: Callable[[Thing], str | None] | None = None,
) -> View:
    """The view a target addresses, each parameter passed to the function that takes it.

    Raises:
        KeyError: The target's thing is not in the graph.
        ValueError: The target's view kind is unknown.
    """
    if target.view == "home" and target.by == UNPLACED_BY:
        return unplaced_index(graph, target.system or "")
    if target.view == "home":
        return home(graph)
    if target.view == "card":
        return device_card(graph, target.id, target.kind)
    if target.view == "channels":
        return channel_list(graph, target.id)
    if target.view == "set":
        return set_overview(graph, target.id, int(target.system or 1))
    if target.view == "calibration":
        return calibration_overview(graph, target.id, target.system or "")
    if target.view == "place":
        return place_view(graph, target.id, annotate=annotate)
    if target.view == "rack":
        return rack_elevation(graph, target.id)
    if target.view == "rollup" and target.kind == "device":
        return device_rollup(graph, target.id, int(target.system or 1))
    if target.view == "rollup":
        return health_rollup(
            graph,
            target.id,
            target.by,  # type: ignore[arg-type]
            system_of=system_of,
            system=target.system,
        )
    if target.view == "checkout":
        return checkout(graph, target.id)
    raise ValueError(f"no view kind {target.view!r}")


def others(view: View, graph: PanelGraph | None = None) -> tuple[dict[str, Any], ...]:
    """What a device's screen shows of other things, on purpose: each thing an item
    of the view is about other than the device (a neighbour on the strip, a block of
    the chain, a far end), the kind of item that draws it, and the channels of it
    the screen binds; with *graph*, also each channel an item binds that is on
    another thing although the item names none (a row of a channel list the
    device's roles reach). A checker reads it to tell a channel drawn as a
    neighbour's from a channel on the wrong screen. Empty for a screen not about
    a device.

    Returns:
        One entry per thing and kind of item, sorted: ``{"thing", "as", "channels"}``.
    """
    if view.id.kind != "device" or view.id.view not in ("card", "channels"):
        return ()
    found: dict[tuple[str, str], set[str]] = {}

    def about(node: Any) -> str | None:
        thing = getattr(node, "thing", None)
        if isinstance(thing, str):
            return thing
        link = getattr(node, "link", None)
        if isinstance(node, ChainBlock) and isinstance(link, Link):
            return link.target.id
        return None

    def channels(node: Any) -> set[str]:
        bound = {getattr(node, "channel", None), *getattr(node, "channels", ())}
        for row in getattr(node, "rows", ()):
            bound |= {getattr(row, "channel", None), *getattr(row, "channels", ())}
        return {c for c in bound if isinstance(c, str)}

    def owner(channel: str) -> str | None:
        try:
            return graph.channel(channel).thing if graph is not None else None
        except KeyError:
            return None

    def walk(node: Any, named: bool = False) -> None:
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            thing = about(node)
            kind = re.sub(r"(?<!^)(?=[A-Z])", " ", type(node).__name__).lower()
            bound = channels(node)
            if thing is not None and thing != view.id.id and bound:
                found.setdefault((thing, kind), set()).update(bound)
            elif thing is None and not named:  # an item that names no thing: by owner
                own = {getattr(node, "channel", None), *getattr(node, "channels", ())}
                for channel in (c for c in own if isinstance(c, str)):
                    held = owner(channel)
                    if held is not None and held != view.id.id:
                        found.setdefault((held, kind), set()).add(channel)
            for f in dataclasses.fields(node):
                walk(getattr(node, f.name), named or thing is not None)
        elif isinstance(node, (tuple, list)):
            for item in node:
                walk(item, named)

    walk(view.sections)
    return tuple(
        {"thing": thing, "as": kind, "channels": sorted(bound)}
        for (thing, kind), bound in sorted(found.items())
    )
