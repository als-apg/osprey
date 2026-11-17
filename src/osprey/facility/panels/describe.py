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

from osprey.facility.panels.health import rule_channels, supported
from osprey.facility.panels.model import Channel, Citation, Edge, PanelGraph, Thing, ThingKind
from osprey.facility.panels.tables import DRAWN_AS, Pose
from osprey.facility.panels.view import (
    CheckRow,
    Elevation,
    Fact,
    Grid,
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

#: The relation kind that states a thing's place; shown under "where", not as a neighbour.
PLACE_RELATION = "located_in"

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

_NO_POSITION = "no position"
_DRAWN_RANK = {drawn: rank for rank, drawn in enumerate(DRAWN_AS)}
_EMPTY: Mapping[str, str] = MappingProxyType({})


# -- the index of which screens exist --------------------------------------------


@dataclass(frozen=True)
class _Index:
    """Which screens a graph gets, apart from racks (those depend on a parameter)."""

    carded: frozenset[str]  # devices with a card
    place_views: frozenset[str]
    sets: frozenset[str]
    ruled: frozenset[str]  # places with a rule-bearing channel within
    readbacks: frozenset[str]  # every channel some setpoint is paired with
    racked: frozenset[str]  # places some located_in claim names a unit in


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
            place = thing.place if thing.kind == "device" else thing.id
            while place is not None and place not in ruled:
                ruled.add(place)
                place = graph.thing(place, "place").place
    return _Index(
        frozenset(carded),
        frozenset(place_views),
        frozenset(sets),
        frozenset(ruled),
        frozenset(readbacks),
        frozenset(racked),
    )


def _has(
    graph: PanelGraph, index: _Index, target: Target, rack_classes: frozenset[str] = frozenset()
) -> bool:
    """Whether *target* is a screen of the graph."""
    view, kind, thing_id = target.view, target.kind, target.id
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
                Target("checkout", "place", thing.id),
            ]
        found += [target for target in candidates if _has(graph, index, target, rack_classes)]
    return found


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


def _label(thing_id: str, channel: str) -> str:
    """The channel's name without its thing's id, when it starts with it."""
    return channel.removeprefix(f"{thing_id}:")


def _value_row(graph: PanelGraph, thing_id: str, channel: Channel) -> ValueRow:
    pair = graph.pair(channel.name) if channel.writable else None
    return ValueRow(
        label=_label(thing_id, channel.name),
        channel=channel.name,
        readback=pair.readback if pair is not None else None,
        tolerance=channel.tolerance,
        writable=channel.writable,
        drawn_as=channel.drawn_as,
        inferred=pair is not None and pair.basis == "inferred",
    )


def light(graph: PanelGraph, channel: str) -> Light:
    """The light of a rule-bearing channel, with what its rows say about it."""
    rows = graph.rule(channel)
    by_value: dict[str, dict[str | None, set[str]]] = {}
    for row in rows:
        by_value.setdefault(row.value, {}).setdefault(row.source, set()).add(row.state)
    disagree = any(
        a != b and states_a != states_b
        for sources in by_value.values()
        for a, states_a in sources.items()
        for b, states_b in sources.items()
    )
    return Light(
        channel=channel,
        inferred=bool(rows) and all(row.basis == "inferred" for row in rows),
        disagree=disagree,
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
    graph: PanelGraph, index: _Index, thing: Thing, channels: list[Channel]
) -> tuple[list[Item], list[Item]]:
    """The card's signals (pairs, then readable channels by how they are drawn) and settings."""
    paired: list[Item] = []
    settings: list[Item] = []
    readable: list[Channel] = []
    for channel in channels:
        if channel.writable:
            row = _value_row(graph, thing.id, channel)
            (paired if row.readback is not None else settings).append(row)
        elif channel.name not in index.readbacks:
            readable.append(channel)
    readable.sort(key=lambda channel: _DRAWN_RANK[channel.drawn_as or "untyped"])
    return paired + [_value_row(graph, thing.id, channel) for channel in readable], settings


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
    """The built place, every ``located_in`` claim of the thing, then its poses."""
    items: list[Item] = []
    if thing.place is not None:
        label = "place" if thing.kind == "device" else "parent"
        items.append(Link(label, Target("card", "place", thing.place)))
    if thing.kind != "group":
        claims = graph.neighbours(thing.id, thing.kind).get(PLACE_RELATION, ())
        items += [_placement(graph, index, e, rack_classes) for e in claims if e.outgoing]
        items += [_pose_fact(pose) for pose in graph.poses(thing.id)]
    return items


def _within(graph: PanelGraph, index: _Index, place: Thing) -> list[Item]:
    """A place's direct contents, then what a ``located_in`` claim puts in it besides."""
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
    signals, settings = _signals(graph, index, thing, channels)
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


def _by_suffix(member: Thing, channels: list[Channel]) -> dict[str, Channel]:
    prefix = f"{member.id}:"
    return {
        channel.name[len(prefix) :]: channel
        for channel in channels
        if channel.name.startswith(prefix)
    }


def _cell(graph: PanelGraph, thing_id: str, channel: Channel) -> ValueRow | Light:
    if not channel.writable and graph.rule(channel.name):
        return light(graph, channel.name)
    return _value_row(graph, thing_id, channel)


def set_overview(graph: PanelGraph, group_id: str) -> View:
    """A group's devices as a grid: one row per member, one column per common channel.

    Rows go by ring position when every member has a true one, else by id.
    Columns are the channel suffixes (the name after ``<member id>:``) every
    member has, in the first row's channel order; a suffix that is, on every
    member, the readback paired with another column's setpoint is shown
    beside that setpoint instead of as a column. A rule-bearing readable
    channel is a :class:`Light`, any other a :class:`ValueRow`.

    Raises:
        KeyError: No such group, or it is excluded.
    """
    graph.thing(group_id, "group")
    index = _index(graph)
    members = [member for member in graph.members(group_id) if member.kind == "device"]
    poses = [graph.pose(member.id) for member in members]
    if members and all(pose is not None and pose.is_ring_position for pose in poses):
        ranked = sorted(zip(members, poses, strict=True), key=lambda mp: (mp[1].s_m, mp[0].id))  # type: ignore[union-attr]
        members = [member for member, _pose in ranked]
    else:
        members.sort(key=lambda member: member.id)
    suffixes = [_by_suffix(member, graph.channels(member.id, "device")) for member in members]
    common = [s for s in suffixes[0] if all(s in held for held in suffixes)] if suffixes else []

    def paired_readbacks(held: dict[str, Channel]) -> set[str]:
        pairs = (graph.pair(held[s].name) for s in common if held[s].writable)
        return {pair.readback for pair in pairs if pair is not None}

    readbacks = [paired_readbacks(held) for held in suffixes]
    columns = [
        s
        for s in common
        if not all(held[s].name in rbs for held, rbs in zip(suffixes, readbacks, strict=True))
    ]
    cells = tuple(
        tuple(_cell(graph, member.id, held[s]) for s in columns)
        for member, held in zip(members, suffixes, strict=True)
    )
    grid = Grid(tuple(_ref(graph, index, member) for member in members), tuple(columns), cells)
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
    """The devices within a place (any depth) in beam order, with the rest in a tray.

    A device with a true ring position (:attr:`Pose.is_ring_position`) is a
    strip element; any other is in the tray with ``no position`` or what its
    position assumes. Sections: ``strip``, then ``places within`` (links to
    the direct child places); the crumbs lead out to the parent.

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
    for thing in graph.within(place_id, deep=True):
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
        else:
            reason = pose.derived_assumes if pose is not None and pose.derived_assumes else None
            tray.append(TrayEntry(thing.id, thing.cls, reason or _NO_POSITION, link))
    elements.sort(key=lambda element: (element.s_m, element.thing))
    extent = (elements[0].s_m, max(e.s_m for e in elements)) if elements else None
    strip = Strip(tuple(elements), tuple(tray), extent)
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
    positions are ``unpositioned``, by id. Each entry links to the device's
    card and carries a :class:`Light` per channel the device's roll-up reads.

    Raises:
        KeyError: No such place, or it is excluded.
    """
    rack = graph.thing(rack_id, "place")
    index = _index(graph)

    def lights(thing: Thing) -> tuple[Light, ...]:
        return tuple(light(graph, name) for name in rule_channels(graph, thing.id, "device"))

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
    unpositioned = [
        (thing.id, RackUnit(None, _ref(graph, index, thing), lights(thing)))
        for thing in graph.within(rack_id)
        if thing.kind == "device" and thing.id not in units_of
    ]
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


def health_rollup(
    graph: PanelGraph,
    root_id: str,
    by: str = "place",
    *,
    system_of: Callable[[Thing], str | None] | None = None,
) -> View:
    """The channels each node of a place's roll-up reads: the root and one level below it.

    By ``place``, the nodes are the place's direct children; a child place
    links down to its own roll-up (its card when it has none), a device to
    its card. By ``system``, the devices within the place (any depth) are
    grouped by ``system_of`` (default: the device's class), the groups in
    name order and :data:`NO_SYSTEM` last; a group node carries the union of
    its devices' channels and no children, and has no screen of its own. A
    node that reads no rule-bearing channel is present and marked ``no_rule``.
    No node below the root has children: one screen shows one level.

    Raises:
        KeyError: No such place, or it is excluded.
        ValueError: *by* is neither ``place`` nor ``system``.
    """
    if by not in ("place", "system"):
        raise ValueError(f"a roll-up goes by 'place' or 'system', not by {by!r}")
    place = graph.thing(root_id, "place")
    index = _index(graph)
    children: list[Node] = []
    if by == "place":
        for child in graph.within(root_id):
            if child.kind == "place":
                link = _place_link(graph, index, child.id, Target("rollup", "place", child.id, by))
                children.append(_node(child.id, link, rule_channels(graph, child.id, "place")))
            else:
                children.append(_device_node(graph, index, child))
    else:
        key = system_of if system_of is not None else (lambda thing: thing.cls)
        systems: dict[str | None, list[Node]] = {}
        for thing in graph.within(root_id, deep=True):
            if thing.kind == "device":
                systems.setdefault(key(thing), []).append(_device_node(graph, index, thing))
        names = sorted(name for name in systems if name is not None)
        for name in [*names, None] if None in systems else names:
            read = (channel for node in systems[name] for channel in node.channels)
            children.append(_node(name if name is not None else NO_SYSTEM, None, read))
    root = _node(root_id, None, rule_channels(graph, root_id, "place"), tuple(children))
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


def checkout(graph: PanelGraph, place_id: str) -> View:
    """One row per channel of the devices within a place (any depth), with totals.

    A row says whether the channel is on a device (always, here), typed
    (``drawn_as`` other than ``untyped``), rule-bearing, and sourced (a
    record type, pair or health row names a source). Sections: ``channels``,
    ``totals``, then ``places within``: links to each child place's checkout
    (its card when it has none), so a table cut on screen leaves no row
    unreachable.

    Raises:
        KeyError: No such place, or it is excluded.
    """
    place = graph.thing(place_id, "place")
    index = _index(graph)
    rows: dict[str, CheckRow] = {}
    for thing in graph.within(place_id, deep=True):
        if thing.kind != "device":
            continue
        for channel in graph.channels(thing.id, "device"):
            if channel.name in rows:
                continue
            rows[channel.name] = CheckRow(
                channel=channel.name,
                thing=thing.id,
                on_device=True,
                typed=channel.drawn_as is not None and channel.drawn_as != "untyped",
                has_rule=bool(graph.rule(channel.name)),
                has_source=_sourced(graph, channel.name),
            )
    found = list(rows.values())
    totals = Totals(
        channels=len(found),
        on_device=sum(row.on_device for row in found),
        typed=sum(row.typed for row in found),
        has_rule=sum(row.has_rule for row in found),
        has_source=sum(row.has_source for row in found),
    )
    children = [
        _place_link(graph, index, child.id, Target("checkout", "place", child.id))
        for child in graph.within(place_id)
        if child.kind == "place"
    ]
    sections = _sections(
        [("channels", list(found)), ("totals", [totals]), ("places within", children)]
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
        return health_rollup(graph, target.id, target.by, system_of=system_of)  # type: ignore[arg-type]
    if target.view == "checkout":
        return checkout(graph, target.id)
    raise ValueError(f"no view kind {target.view!r}")
