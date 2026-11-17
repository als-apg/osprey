"""View descriptions: what a screen contains, with no pixel, colour or format.

A :class:`View` is a tree of :class:`Section` objects whose items are the
dataclasses below. Every item names channels, never values: whoever draws a
view binds the channels and computes the states itself. A :class:`Link`
points at another screen by its :class:`Target`; a neighbour that gets no
screen is a :class:`Fact` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from osprey.facility.panels.model import Citation, ThingKind
from osprey.facility.panels.tables import PicturePart

__all__ = [
    "VIEW_KINDS",
    "CheckRow",
    "Elevation",
    "Fact",
    "Grid",
    "GridCell",
    "GridRow",
    "Item",
    "Light",
    "Link",
    "Node",
    "Picture",
    "Placement",
    "RackUnit",
    "Section",
    "Strip",
    "StripElement",
    "Target",
    "Totals",
    "TrayEntry",
    "ValueRow",
    "View",
    "ViewKind",
]

ViewKind = Literal["card", "set", "place", "rack", "rollup", "checkout"]

#: Every view kind, in the order screens of one thing are listed.
VIEW_KINDS: tuple[ViewKind, ...] = ("card", "set", "place", "rack", "rollup", "checkout")


@dataclass(frozen=True)
class Target:
    """The address of one screen.

    Attributes:
        view: The view kind.
        kind: The kind of the thing the screen is of; one id can name two kinds.
        id: The thing's id.
        by: A roll-up's grouping, ``place`` or ``system``; ``None`` for every
            other view kind.
    """

    view: ViewKind
    kind: ThingKind
    id: str
    by: str | None = None


@dataclass(frozen=True)
class Link:
    """A way to another screen.

    Attributes:
        label: What the link reads.
        target: The screen it opens.
        standing: For a relation, the claim's standing as the graph gives it
            (``accepted`` or ``contested``); ``None`` otherwise.
        basis: For a relation, ``stated`` or ``inferred``; ``None`` otherwise.
    """

    label: str
    target: Target
    standing: str | None = None
    basis: str | None = None


@dataclass(frozen=True)
class Fact:
    """A statement shown as text.

    Attributes:
        label: What the fact is about.
        text: The fact.
        citation: The document it is read from, or ``None``.
        caveat: What the fact assumes, shown beside it; ``None`` when nothing.
        standing: For a relation, as on :class:`Link`.
        basis: For a relation or a pose, ``stated`` or ``inferred``.
    """

    label: str
    text: str
    citation: Citation | None = None
    caveat: str | None = None
    standing: str | None = None
    basis: str | None = None


@dataclass(frozen=True)
class ValueRow:
    """A channel shown as a value, with the readback beside a setpoint.

    Attributes:
        label: What the row reads.
        channel: The channel shown (the setpoint, for a pair).
        readback: The channel that reads the setpoint back, or ``None``.
        tolerance: The channel's ``(term, amount)`` tolerance, or ``None``.
        writable: Whether the channel is a setpoint.
        drawn_as: How ``record_types.csv`` says to draw the channel, or ``None``.
        inferred: Whether the pair is inferred rather than stated.
    """

    label: str
    channel: str
    readback: str | None
    tolerance: tuple[str, float] | None
    writable: bool
    drawn_as: str | None = None
    inferred: bool = False


@dataclass(frozen=True)
class Light:
    """A health light for one rule-bearing channel.

    Attributes:
        channel: The channel whose state the light shows.
        inferred: Every health row of the channel is inferred.
        disagree: Rows from different sources give one value different states.
        unsupported: Some row of the channel has a form the engine cannot
            evaluate, so the light can read unknown where the rule would not.
    """

    channel: str
    inferred: bool = False
    disagree: bool = False
    unsupported: bool = False


@dataclass(frozen=True)
class Placement:
    """One ``located_in`` claim of a thing, shown under "where".

    Attributes:
        link: The place (its rack view when it has one, else its card), with
            the claim's standing and basis.
        unit: The rack unit the claim names, or ``None``.
        size_u: The claim's ``size_u`` when it parses as a number, else ``None``.
        citation: The document the claim cites, or ``None``.
    """

    link: Link
    unit: int | None
    size_u: int | float | None
    citation: Citation | None


@dataclass(frozen=True)
class GridCell:
    """One member of a set overview.

    Attributes:
        member: The member's card, or a fact when it has none.
        channels: The member's rule-bearing channels: its one light reads
            their worst state, as a roll-up node of the member would.
        pair: The member's setpoint of the set's primary pair, with its
            readback and tolerance; ``None`` when the member has none.
    """

    member: Link | Fact
    channels: tuple[str, ...]
    pair: ValueRow | None = None


@dataclass(frozen=True)
class GridRow:
    """The members of a set overview that share one place.

    Attributes:
        label: The place (a link to its card), or a fact ``not placed``.
        cells: The members, in beam order (``s_m``, then id).
    """

    label: Link | Fact
    cells: tuple[GridCell, ...]


@dataclass(frozen=True)
class Grid:
    """A set's members by place.

    Attributes:
        signal: The signal the facility states for the setpoint of the set's
            primary pair (the pair most members have), or ``None``.
        rows: One per place, in the graph's place order; the members with
            no place last.
    """

    signal: str | None
    rows: tuple[GridRow, ...]


@dataclass(frozen=True)
class StripElement:
    """A device at its true ring position.

    Attributes:
        thing: The device's id.
        cls: The device's class.
        s_m: Its ring position.
        symbol: The symbol it is drawn as; ``generic`` with none mapped.
        lane: ``above`` or ``beam``.
        link: Its card, or ``None`` when it has no screen.
        channels: The device's rule-bearing channels, whose worst state the
            symbol shows; none, and the symbol has no state to show.
    """

    thing: str
    cls: str | None
    s_m: float
    symbol: str
    lane: str
    link: Link | None
    channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrayEntry:
    """A device within the place that has no true ring position.

    Attributes:
        thing: The device's id.
        cls: The device's class.
        reason: ``no position``, or what its position assumes.
        link: Its card, or ``None`` when it has no screen.
    """

    thing: str
    cls: str | None
    reason: str
    link: Link | None


@dataclass(frozen=True)
class Strip:
    """A place's devices in beam order.

    Attributes:
        elements: The devices with a true ring position, by ``s_m`` then id.
        tray: The devices with none, in the place's order.
        extent_m: The least and greatest ``s_m``; ``None`` with no element.
    """

    elements: tuple[StripElement, ...]
    tray: tuple[TrayEntry, ...]
    extent_m: tuple[float, float] | None


@dataclass(frozen=True)
class Picture:
    """The parts drawn on one view of a device class.

    Attributes:
        device_class: The class.
        view: ``end`` or ``side``.
        parts: The ``pictures.csv`` rows of that view, in file order.
    """

    device_class: str
    view: str
    parts: tuple[PicturePart, ...]


@dataclass(frozen=True)
class RackUnit:
    """One device placed in a rack.

    Attributes:
        position: The rack unit the place relation names, or ``None``.
        link: The device's card, or a fact when it has no screen.
        lights: One per channel its roll-up reads.
        size_u: The height in units the place relation's ``size_u`` states,
            when it parses as a number; ``None`` otherwise.
        standing: The unit claims' standing: ``contested`` when any claim for
            this (device, unit) is, else ``accepted``; ``None`` for an
            unpositioned device.
        conflicting: The same device has claims naming more than one unit
            in this rack; each unit is drawn, none is picked.
        citations: The documents of every claim for this (device, unit),
            each once, in graph order.
    """

    position: int | None
    link: Link | Fact
    lights: tuple[Light, ...]
    size_u: int | float | None = None
    standing: str | None = None
    conflicting: bool = False
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Elevation:
    """A rack's devices.

    Attributes:
        units: The devices whose place relation names a unit, in *order*
            (ties by id).
        unpositioned: The devices in the rack with no unit, by id.
        order: ``top-down``: the highest unit number first.
    """

    units: tuple[RackUnit, ...]
    unpositioned: tuple[RackUnit, ...]
    order: str


@dataclass(frozen=True)
class Node:
    """One node of a health roll-up.

    Attributes:
        label: The node's name.
        link: The screen one level down, or ``None``.
        channels: The rule-bearing channels the node's roll-up reads.
        no_rule: The node reads no rule-bearing channel.
        children: The nodes below it, on this screen.
    """

    label: str
    link: Link | None
    channels: tuple[str, ...]
    no_rule: bool
    children: tuple[Node, ...] = ()


@dataclass(frozen=True)
class CheckRow:
    """What one expected channel has.

    Attributes:
        channel: The channel.
        thing: The device it is on.
        on_device: It is on a device.
        typed: ``record_types.csv`` types it as other than ``untyped``.
        has_rule: It has a health row.
        has_source: A rule-table row for it (record type, pair, health) names
            its source.
    """

    channel: str
    thing: str
    on_device: bool
    typed: bool
    has_rule: bool
    has_source: bool


@dataclass(frozen=True)
class Totals:
    """How many checkout rows have each property."""

    channels: int
    on_device: int
    typed: int
    has_rule: int
    has_source: int


Item = (
    ValueRow
    | Light
    | Link
    | Fact
    | Placement
    | Grid
    | Strip
    | Picture
    | Elevation
    | Node
    | CheckRow
    | Totals
)


@dataclass(frozen=True)
class Section:
    """A titled run of items."""

    title: str
    items: tuple[Item, ...]


@dataclass(frozen=True)
class View:
    """One screen's content.

    Attributes:
        id: Its address.
        title: What it is titled.
        crumbs: Links from the outermost place in.
        sections: Its sections, each non-empty, in order.
    """

    id: Target
    title: str
    crumbs: tuple[Link, ...]
    sections: tuple[Section, ...]
