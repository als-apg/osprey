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
from osprey.facility.panels.tables import PicturePart, ProbeElement

__all__ = [
    "VIEW_KINDS",
    "Band",
    "Tile",
    "CheckRow",
    "Elevation",
    "Fact",
    "Grid",
    "GridCell",
    "GridRow",
    "Group",
    "Item",
    "Light",
    "Link",
    "Node",
    "Picture",
    "Probe",
    "ProbePart",
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

ViewKind = Literal["card", "set", "calibration", "place", "rack", "rollup", "checkout"]

#: Every view kind, in the order screens of one thing are listed.
VIEW_KINDS: tuple[ViewKind, ...] = (
    "card",
    "set",
    "calibration",
    "place",
    "rack",
    "rollup",
    "checkout",
)


@dataclass(frozen=True)
class Target:
    """The address of one screen.

    Attributes:
        view: The view kind.
        kind: The kind of the thing the screen is of; one id can name two kinds.
        id: The thing's id.
        by: A roll-up's grouping, ``place`` or ``system``; ``None`` for every
            other view kind.
        system: For a roll-up by ``system``, the one system it shows within
            the place; ``None`` for the roll-up of every system. For a
            ``calibration`` view, the calibration kind it shows over the set.
    """

    view: ViewKind
    kind: ThingKind
    id: str
    by: str | None = None
    system: str | None = None


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
        unit: The channel's unit, shown beside its value; ``None`` with none.
        readback_unit: The readback's unit, or ``None``.
    """

    label: str
    channel: str
    readback: str | None
    tolerance: tuple[str, float] | None
    writable: bool
    drawn_as: str | None = None
    inferred: bool = False
    unit: str | None = None
    readback_unit: str | None = None


@dataclass(frozen=True)
class Light:
    """A health light for one rule-bearing channel.

    Attributes:
        channel: The channel whose state the light shows.
        inferred: Every health row of the channel is inferred.
        disagree: The channel's rows give one stated value more than one state
            (:func:`~osprey.facility.panels.health.disagrees`).
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
        calibration: In a calibration view, what the member's records say of
            the view's kind; ``None`` in a set overview.
    """

    member: Link | Fact
    channels: tuple[str, ...]
    pair: ValueRow | None = None
    calibration: CalibrationCell | None = None


@dataclass(frozen=True)
class CalibrationCell:
    """One member in a calibration view: what its records say of the view's kind.

    Attributes:
        text: The deviation from the design value (``+0.21 %``; ``a / b`` when
            two claims disagree), the value when there is no design value,
            ``design`` when the design value is in force, or ``no value``.
        state: ``contested``, a basis, or ``none``.
        channels: The channels that hold the value live, in channel order, for a
            kind held in the control system; empty otherwise.
    """

    text: str
    state: str
    channels: tuple[str, ...] = ()


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
        kind: In a calibration view, the kind's title; ``None`` in a set overview.
    """

    signal: str | None
    rows: tuple[GridRow, ...]
    kind: str | None = None


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
        short: The short name the data states for it, its label on the strip;
            ``None`` with none.
        annotations: The channels whose values the strip shows with it, each with
            its unit (``None`` with none stated).
    """

    thing: str
    cls: str | None
    s_m: float
    symbol: str
    lane: str
    link: Link | None
    channels: tuple[str, ...] = ()
    short: str | None = None
    annotations: tuple[tuple[str, str | None], ...] = ()


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
        tray: The devices with none whose class has a symbol (they belong on
            the beamline), in the place's order.
        extent_m: The least and greatest ``s_m``; ``None`` with no element.
        others: The place's card, labelled with how many other devices the
            place holds, or ``None`` when there are none.
    """

    elements: tuple[StripElement, ...]
    tray: tuple[TrayEntry, ...]
    extent_m: tuple[float, float] | None
    others: Link | None = None


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
class ProbePart:
    """One element of a probe picture with the channels it shows.

    Attributes:
        element: The ``probe_pictures.csv`` row.
        channel: The channel stating the row's ``signal`` (:meth:`PanelGraph.probe_channel`),
            or ``None``: none states it, or the row names none.
        channel_y: The same for ``signal_y`` (the marker's y).
    """

    element: ProbeElement
    channel: str | None = None
    channel_y: str | None = None


@dataclass(frozen=True)
class Probe:
    """A pickup's cross-section, looking downstream (:meth:`PanelGraph.probe`).

    Attributes:
        device_class: The device's class.
        parts: Its elements, in table order.
        inferred: A button's placement is inferred, not stated (an anchor row's
            ``state`` says ``inferred``).
    """

    device_class: str
    parts: tuple[ProbePart, ...]
    inferred: bool = False


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
        unit_text: For an unpositioned device, the unit its claim states
            when that is no integer (``A5``); ``None`` otherwise.
    """

    position: int | None
    link: Link | Fact
    lights: tuple[Light, ...]
    size_u: int | float | None = None
    standing: str | None = None
    conflicting: bool = False
    citations: tuple[Citation, ...] = ()
    unit_text: str | None = None


@dataclass(frozen=True)
class Elevation:
    """A rack's devices.

    Attributes:
        units: The devices whose place relation names a unit, in *order*
            (ties by id).
        unpositioned: The devices built in the rack that no claim gives a unit,
            and the devices a claim puts in it with a unit that is no
            integer (its text on the entry), by id.
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
        thing: The device it is on; ``None`` for a channel on no thing.
        on_device: It is on a device.
        typed: ``record_types.csv`` types it as other than ``untyped``.
        has_rule: It has a health row.
        has_source: A rule-table row for it (record type, pair, health) names
            its source.
    """

    channel: str
    thing: str | None
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


@dataclass(frozen=True)
class Tile:
    """One key number: a row of ``metrics.csv`` and what it selects in the place.

    Attributes:
        title: What it is.
        measure: What it computes (``value``, ``count_true``, ``count_false``,
            ``count_in_tolerance``, ``rms``, ``max`` or ``min``).
        unit: The row's unit, or ``None``.
        caption: The row's one line under the number, or ``None``.
        channels: The channels in the place stating the row's signal (on a
            thing of the row's class, when it names one), by name.
        pairs: For ``count_in_tolerance``: per setpoint stating the signal
            that has a readback and a tolerance, ``(setpoint, readback,
            term, amount)``, by setpoint.
    """

    title: str
    measure: str
    unit: str | None
    caption: str | None
    channels: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str, str, float], ...] = ()


@dataclass(frozen=True)
class Band:
    """The key numbers at the top of a place view or a roll-up root, in order."""

    tiles: tuple[Tile, ...]


@dataclass(frozen=True)
class Group:
    """Things of one kind shown together, each with its health.

    Attributes:
        title: What they are (``racks``, a place level, ``other``, ``by system``).
        form: ``chips`` (names that wrap across the page), ``rows`` (a
            summary line each) or ``ring`` (an arc each, in beam order).
        nodes: One per thing: its name, its screen, the rule-bearing channels
            its light reads.
        more: Where the things not listed are, labelled with how many, or ``None``.
    """

    title: str
    form: Literal["chips", "rows", "ring"]
    nodes: tuple[Node, ...]
    more: Link | None = None


@dataclass(frozen=True)
class CalibrationRow:
    """One calibration kind of a device, as its card shows it. Read-only.

    Attributes:
        title: The kind's title.
        conditions: What the value in force is valid for, or ``None``.
        shown: The value in force as the records state it, with its unit (text, never
            a live reading); ``None``: no value yet.
        note: A second line under the value (the deviation from the design
            value, a curve's range, a disagreeing claim), or ``None``.
        state: ``contested``, a basis, or ``none``.
        chip: The basis chip's text.
        below_wanted: The value in force ranks under the wanted basis; the chip
            is dashed.
        date: The claim's date, or ``None``.
        source: The claim's document and version, or ``None``.
        history: What the claim replaced, or that it is the first value.
        channels: The channels that hold the value in force live (those of the
            device stating the kind's ``held_in``), in channel order; empty when
            the records hold it.
        link: The calibration view of the kind over a set the device is in, or ``None``.
    """

    title: str
    conditions: str | None
    shown: str | None
    note: str | None
    state: str
    chip: str
    below_wanted: bool
    date: str | None
    source: str | None
    history: str | None
    channels: tuple[str, ...] = ()
    link: Link | None = None


@dataclass(frozen=True)
class Calibrations:
    """A device's calibration block: one row per kind of its class."""

    rows: tuple[CalibrationRow, ...]


Item = (
    ValueRow
    | Light
    | Link
    | Fact
    | Placement
    | Grid
    | Strip
    | Picture
    | Probe
    | Elevation
    | Node
    | Group
    | Band
    | CheckRow
    | Totals
    | Calibrations
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
        alias: The short name the data states for the thing, shown beside
            its id; ``None`` with none.
    """

    id: Target
    title: str
    crumbs: tuple[Link, ...]
    sections: tuple[Section, ...]
    alias: str | None = None
