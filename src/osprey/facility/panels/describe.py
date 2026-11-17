"""What each screen contains: one pure function per view kind, graph in, view out.

:func:`targets` lists every screen a graph gets, each addressed by a
:class:`~osprey.facility.panels.view.Target` (view kind, thing kind, id). The
view functions return :class:`~osprey.facility.panels.view.View` objects whose
links only point at those targets; a thing with no screen is named in a
:class:`~osprey.facility.panels.view.Fact` instead.

Which things get which screens:

- ``card``: every device that has a channel, a pose, or a relation other
  than its place relation (``located_in`` its own place), every place, every
  group; a citation alone gives no card;
- ``set``: every group whose members are devices of one class;
- ``place``: every place with anything within it;
- ``rack``: every place some ``located_in`` claim names a unit in, and every
  place whose ``level`` is one of the ``rack_classes`` passed in (both
  markers are data; no level or place name is written here);
- ``rollup`` (by ``place`` and by ``system``) and ``checkout``: every place
  with a rule-bearing channel within it, at any depth.

Views carry channel names, never values or states: whoever draws a view binds
the channels and asks :mod:`~osprey.facility.panels.health` for the states.
"""

from __future__ import annotations

import dataclasses
import math
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from osprey.facility.panels.health import disagrees, rule_channels, supported, unplaced_channels
from osprey.facility.panels.model import (
    PLACE_RELATION,
    Channel,
    Citation,
    Edge,
    PanelGraph,
    Thing,
    ThingKind,
)
from osprey.facility.panels.tables import DRAWN_AS, Pose
from osprey.facility.panels.view import (
    CheckRow,
    Elevation,
    Fact,
    Grid,
    GridCell,
    GridRow,
    Item,
    Light,
    Link,
    Node,
    Picture,
    Placement,
    RackUnit,
    Section,
    Strip,
    StripElement,
    Target,
    Totals,
    TrayEntry,
    ValueRow,
    View,
    ViewKind,
)

__all__ = [
    "GENERIC_SYMBOL",
    "NOT_PLACED",
    "NO_SYSTEM",
    "PLACE_RELATION",
    "RACK_ORDER",
    "RELATION_TITLES",
    "checkout",
    "describe",
    "device_card",
    "health_rollup",
    "light",
    "place_view",
    "rack_elevation",
    "set_overview",
    "targets",
]

#: The id of the screens of what is in no place (:meth:`PanelGraph.unplaced`):
#: a roll-up and a checkout, addressed as a place that the graph does not hold.
NOT_PLACED = "not placed"

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


_NO_POSITION = "no position"
_NOT_IN = "not placed"
_NO_THING = "on no device"
_OTHERS = "{} other things in this place"
_WITHOUT = "{} without a rule"
_OWN = "{} itself"
_DRAWN_RANK = {drawn: rank for rank, drawn in enumerate(DRAWN_AS)}
_EMPTY: Mapping[str, str] = MappingProxyType({})


# -- the index of which screens exist --------------------------------------------


@dataclass(frozen=True)
class _Index:
    """Which screens a graph gets, apart from racks (those depend on a parameter)."""

    carded: frozenset[str]  # devices with a card
    place_views: frozenset[str]
    sets: frozenset[str]
    ruled: frozenset[str]  # places with a rule-bearing channel in their contents
    readbacks: frozenset[str]  # every channel some setpoint is paired with
    racked: frozenset[str]  # places some located_in claim names a unit in
    unplaced_ruled: bool  # a device in no place has a rule-bearing channel
    unplaced_channels: bool  # a device in no place has a channel
    systems: Mapping[str, frozenset[str]]  # place -> systems with a rule channel in it


_INDEXES: weakref.WeakKeyDictionary[PanelGraph, _Index] = weakref.WeakKeyDictionary()


def _index(graph: PanelGraph) -> _Index:
    """The graph's screen index, built once per graph in one pass over it."""
    found = _INDEXES.get(graph)
    if found is None:
        found = _INDEXES[graph] = _build_index(graph)
    return found


def _is_place_edge(graph: PanelGraph, thing: Thing, edge: Edge) -> bool:
    """Whether the edge is the thing's own place relation, seen from either end."""
    if edge.kind != PLACE_RELATION:
        return False
    if edge.outgoing:
        return edge.other_kind == "place" and edge.other == thing.place
    if thing.kind != "place" or edge.other_kind != "device":
        return False
    return graph.thing(edge.other, "device").place == thing.id


def _neighbour_edges(graph: PanelGraph, thing: Thing) -> dict[str, list[Edge]]:
    """The thing's edges by kind, without its place relation."""
    kept: dict[str, list[Edge]] = {}
    for kind, edges in graph.neighbours(thing.id, thing.kind).items():
        others = [edge for edge in edges if not _is_place_edge(graph, thing, edge)]
        if others:
            kept[kind] = others
    return kept


def _build_index(graph: PanelGraph) -> _Index:
    carded: set[str] = set()
    place_views: set[str] = set()
    sets: set[str] = set()
    ruled: set[str] = set()
    readbacks: set[str] = set()
    racked: set[str] = set()
    systems: dict[str, set[str]] = {}
    for thing in graph.things():
        if thing.kind == "group":
            members = graph.members(thing.id)
            classes = {member.cls if member.kind == "device" else None for member in members}
            if len(classes) == 1 and None not in classes:
                sets.add(thing.id)
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
            if channels or graph.pose(thing.id) is not None or _neighbour_edges(graph, thing):
                carded.add(thing.id)
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
        frozenset(place_views),
        frozenset(sets),
        frozenset(ruled),
        frozenset(readbacks),
        frozenset(racked),
        bool(unplaced_channels(graph)),
        any(graph.channels(thing.id, "device") for thing in unplaced),
        MappingProxyType({place: frozenset(found) for place, found in systems.items()}),
    )


def _has(
    graph: PanelGraph, index: _Index, target: Target, rack_classes: frozenset[str] = frozenset()
) -> bool:
    """Whether *target* is a screen of the graph."""
    view, kind, thing_id = target.view, target.kind, target.id
    if (kind, thing_id) == ("place", NOT_PLACED) and not _holds(graph, thing_id, "place"):
        if view == "rollup":
            return target.by == "place" and index.unplaced_ruled
        return view == "checkout" and index.unplaced_channels
    if view == "card":
        return thing_id in index.carded if kind == "device" else _holds(graph, thing_id, kind)
    if view == "set":
        return kind == "group" and thing_id in index.sets
    if kind != "place":
        return False
    if view == "place":
        return thing_id in index.place_views
    if view == "rack":
        if thing_id in index.racked:
            return True
        return _holds(graph, thing_id, "place") and graph.thing(thing_id, "place").level in (
            rack_classes
        )
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


def targets(graph: PanelGraph, *, rack_classes: frozenset[str] = frozenset()) -> list[Target]:
    """Every screen the graph gets, by thing in graph order, then by view kind.

    Args:
        graph: The panel graph.
        rack_classes: The place levels that mark a rack.

    Returns:
        The screens' addresses, each once.
    """
    index = _index(graph)
    found: list[Target] = []
    for thing in graph.things():
        candidates = [Target("card", thing.kind, thing.id)]
        if thing.kind == "group":
            candidates.append(Target("set", "group", thing.id))
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
        found += [target for target in candidates if _has(graph, index, target, rack_classes)]
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
    """Links to *place_id* and every place around it, outermost first."""
    chain: list[str] = []
    while place_id is not None:
        chain.append(place_id)
        place_id = graph.thing(place_id, "place").place
    return tuple(
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
    return Fact(title, edge.other, citation=edge.source, standing=edge.standing, basis=edge.basis)


def _placement(graph: PanelGraph, index: _Index, edge: Edge, rack_classes: frozenset[str]) -> Item:
    """A ``located_in`` claim as a placement: its rack view when there is one, else a card."""
    target = Target("card", edge.other_kind, edge.other)
    rack = Target("rack", "place", edge.other)
    if edge.other_kind == "place" and _has(graph, index, rack, rack_classes):
        target = rack
    if not _has(graph, index, target, rack_classes):
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


def _where(
    graph: PanelGraph, index: _Index, thing: Thing, rack_classes: frozenset[str]
) -> list[Item]:
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
        if _has(graph, index, target, rack_classes):
            items.append(Link(text, target, basis="inferred"))
        else:
            items.append(Fact("where", text, basis="inferred"))
    if thing.kind != "group":
        claims = graph.neighbours(thing.id, thing.kind).get(PLACE_RELATION, ())
        items += [_placement(graph, index, e, rack_classes) for e in claims if e.outgoing]
        items += [_pose_fact(pose) for pose in graph.poses(thing.id)]
    return items


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
    if thing.kind != "device" or thing.cls is None:
        return []
    views: dict[str, list] = {}
    for part in graph.pictures(thing.cls):
        views.setdefault(part.view, []).append(part)
    return [Picture(thing.cls, view, tuple(parts)) for view, parts in views.items()]


_SCREEN_TITLES = {
    ("place", None): "place view",
    ("rack", None): "rack elevation",
    ("rollup", "place"): "health by place",
    ("rollup", "system"): "health by system",
    ("checkout", None): "checkout",
    ("set", None): "set overview",
}


def _screens(
    graph: PanelGraph, index: _Index, thing: Thing, rack_classes: frozenset[str]
) -> list[Item]:
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
        if _has(graph, index, target, rack_classes)
    ]


def device_card(
    graph: PanelGraph,
    thing_id: str,
    kind: ThingKind | None = None,
    *,
    rack_classes: frozenset[str] = frozenset(),
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
        rack_classes: The place levels that mark a rack, as passed to :func:`targets`.

    Raises:
        KeyError: No such thing, or it is excluded.
    """
    thing = graph.thing(thing_id, kind)
    index = _index(graph)
    channels = graph.channels(thing.id, thing.kind)
    signals, settings = _signals(graph, index, channels)
    within: list[Item] = []
    members: list[Item] = []
    if thing.kind == "place":
        within = _within(graph, index, thing)
    elif thing.kind == "group":
        members = [_ref(graph, index, member) for member in graph.members(thing.id)]
    health: list[Item] = [
        light(graph, channel.name) for channel in channels if graph.rule(channel.name)
    ]
    sources: list[Item] = [
        Fact(citation.document, _citation_text(citation), citation=citation)
        for citation in thing.citations
    ]
    sections = _sections(
        [
            ("screens", _screens(graph, index, thing, rack_classes)),
            ("within", within),
            ("members", members),
            ("signals", signals),
            ("settings", settings),
            ("health", health),
            *_neighbours(graph, index, thing),
            ("picture", _pictures(graph, thing)),
            ("where", _where(graph, index, thing, rack_classes)),
            ("sources", sources),
        ]
    )
    crumbs = _crumbs(graph, index, thing.place) if thing.kind != "group" else ()
    return View(Target("card", thing.kind, thing.id), thing.id, crumbs, sections)


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


def set_overview(graph: PanelGraph, group_id: str) -> View:
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
    rows_of = _row_places(graph, members)

    def beam(member: Thing) -> tuple[float, str]:
        pose = graph.pose(member.id)
        ring = pose is not None and pose.is_ring_position
        return (pose.s_m if ring else math.inf, member.id)  # type: ignore[union-attr]

    def cell(member: Thing) -> GridCell:
        pair = None
        if signal is not None:
            for channel in graph.channels(member.id, "device"):
                if channel.writable and channel.signal == signal and graph.pair(channel.name):
                    pair = _value_row(graph, channel)
                    break
        channels = tuple(rule_channels(graph, member.id, "device"))
        return GridCell(_ref(graph, index, member), channels, pair)

    order = {thing.id: n for n, thing in enumerate(graph.things()) if thing.kind == "place"}
    grouped: dict[str | None, list[Thing]] = {}
    for member in members:
        grouped.setdefault(rows_of[member.id], []).append(member)
    keys = sorted((k for k in grouped if k is not None), key=lambda k: order[k])
    rows: list[GridRow] = []
    for key in [*keys, None] if None in grouped else keys:
        label: Link | Fact = (
            Link(key, Target("card", "place", key)) if key is not None else Fact("row", _NOT_IN)
        )
        rows.append(GridRow(label, tuple(cell(m) for m in sorted(grouped[key], key=beam))))
    grid = Grid(signal, tuple(rows))
    group_link = Link(group_id, Target("card", "group", group_id))
    sections = (Section("group", (group_link,)), Section("members", (grid,)))
    return View(Target("set", "group", group_id), group_id, (), sections)


# -- place view ------------------------------------------------------------------


def place_view(
    graph: PanelGraph,
    place_id: str,
    *,
    symbols: Mapping[str, str] = _EMPTY,
    above: frozenset[str] = frozenset(),
) -> View:
    """The devices of a place's contents in beam order, the rest in a tray or counted.

    A device with a true ring position (:attr:`Pose.is_ring_position`) is a
    strip element. A device with none whose class has a symbol (it belongs
    on the beamline) is in the tray with ``no position`` or what its position
    assumes. Every other device is counted on one line, ``<n> other things in
    this place``, that opens the place's card (which lists them). Sections:
    ``strip``, then ``places within`` (links to the direct child places); the
    crumbs lead out to the parent.

    Args:
        graph: The panel graph.
        place_id: The place.
        symbols: Device class -> symbol; a class not named is ``generic``.
        above: The symbols drawn on the lane above the beam.

    Raises:
        KeyError: No such place, or it is excluded.
    """
    place = graph.thing(place_id, "place")
    index = _index(graph)
    elements: list[StripElement] = []
    tray: list[TrayEntry] = []
    others = 0
    for thing in graph.contents(place_id):
        if thing.kind != "device":
            continue
        link = _ref(graph, index, thing)
        link = link if isinstance(link, Link) else None
        pose = graph.pose(thing.id)
        if pose is not None and pose.is_ring_position:
            symbol = symbols.get(thing.cls or "", GENERIC_SYMBOL)
            lane = "above" if symbol in above else "beam"
            channels = tuple(rule_channels(graph, thing.id, "device"))
            elements.append(
                StripElement(thing.id, thing.cls, pose.s_m, symbol, lane, link, channels)  # type: ignore[arg-type]
            )
        elif thing.cls is not None and thing.cls in symbols:
            reason = pose.derived_assumes if pose is not None and pose.derived_assumes else None
            tray.append(TrayEntry(thing.id, thing.cls, reason or _NO_POSITION, link))
        else:
            others += 1
    elements.sort(key=lambda element: (element.s_m, element.thing))
    extent = (elements[0].s_m, max(e.s_m for e in elements)) if elements else None
    rest = Link(_OTHERS.format(others), Target("card", "place", place_id)) if others else None
    strip = Strip(tuple(elements), tuple(tray), extent, rest)
    children: list[Item] = [
        _place_link(graph, index, child.id, Target("place", "place", child.id))
        for child in graph.within(place_id)
        if child.kind == "place"
    ]
    sections = _sections([("strip", [strip]), ("places within", children)])
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
    as a number. The devices whose built place is the rack and that no unit claim
    positions are ``unpositioned``, by id, and so are those a claim puts in the
    rack at a unit that is no integer (with the unit's text).
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
        entry = RackUnit(
            position,
            _ref(graph, index, thing),
            lights(thing),
            next((size for size in sizes if size is not None), None),
            "contested" if contested else "accepted",
            conflicting=units_of[device] > 1,
            citations=tuple(citations),
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
        text = claim.unit_text
        if text is not None:
            texts.setdefault(claim.other, []).append(text)
    for device, stated in texts.items():
        thing = graph.thing(device, "device")
        text = ", ".join(dict.fromkeys(stated))
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
    ref = _ref(graph, index, thing)
    return _node(
        thing.id, ref if isinstance(ref, Link) else None, rule_channels(graph, thing.id, "device")
    )


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


def _beam_key(graph: PanelGraph, label: str) -> tuple[float, str]:
    """Beam order for a node of a thing with a true ring position, else name order after."""
    pose = graph.pose(label)
    ring = pose is not None and pose.is_ring_position
    return (pose.s_m if ring else math.inf, label)  # type: ignore[union-attr]


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
    return View(_UNPLACED_ROLLUP, f"{NOT_PLACED} health", (), (Section("roll-up", (root,)),))


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
    return View(target, title, crumbs, (Section("roll-up", (root,)),))


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
    root is too long for one rule still shows them live.

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
    children = _ordered(graph, children, rest, sort=by == "place", tail=unplaced)
    root = _node(root_id, None, channels, tuple(children))
    crumbs = _crumbs(graph, index, place.place, "rollup", by)
    title = f"{root_id} health by {by}"
    return View(
        Target("rollup", "place", root_id, by=by), title, crumbs, (Section("roll-up", (root,)),)
    )


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


def _check_row(graph: PanelGraph, channel: Channel, thing: str | None) -> CheckRow:
    return CheckRow(
        channel=channel.name,
        thing=thing,
        on_device=thing is not None,
        typed=channel.drawn_as is not None and channel.drawn_as != "untyped",
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

    A row says whether the channel is on a device, typed (``drawn_as``
    other than ``untyped``), rule-bearing, and sourced (a record type, pair
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
        return View(_UNPLACED_CHECKOUT, f"{NOT_PLACED} checkout", (), sections)
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
    rack_classes: frozenset[str] = frozenset(),
    symbols: Mapping[str, str] = _EMPTY,
    above: frozenset[str] = frozenset(),
    system_of: Callable[[Thing], str | None] | None = None,
) -> View:
    """The view a target addresses, each parameter passed to the function that takes it.

    Raises:
        KeyError: The target's thing is not in the graph.
        ValueError: The target's view kind is unknown.
    """
    if target.view == "card":
        return device_card(graph, target.id, target.kind, rack_classes=rack_classes)
    if target.view == "set":
        return set_overview(graph, target.id)
    if target.view == "place":
        return place_view(graph, target.id, symbols=symbols, above=above)
    if target.view == "rack":
        return rack_elevation(graph, target.id)
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
