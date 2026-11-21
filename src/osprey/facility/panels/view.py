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
    "BlockRow",
    "CardFoot",
    "CardHead",
    "CardStrip",
    "StripMember",
    "Chain",
    "ChainBlock",
    "ChannelCount",
    "ChannelGrid",
    "Connector",
    "Control",
    "CurvePlot",
    "ControlGroup",
    "KeyTile",
    "Operate",
    "Tile",
    "CheckRow",
    "Elevation",
    "Fact",
    "FarEnd",
    "Front",
    "FrontGroup",
    "Grid",
    "GridCell",
    "GridRow",
    "Group",
    "Item",
    "Light",
    "LimitLadder",
    "Link",
    "Node",
    "PartFacts",
    "Picture",
    "Probe",
    "RingMember",
    "RingStrip",
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

ViewKind = Literal[
    "home", "card", "channels", "set", "calibration", "place", "rack", "rollup", "checkout"
]

#: Every view kind, in the order screens of one thing are listed.
VIEW_KINDS: tuple[ViewKind, ...] = (
    "home",
    "card",
    "channels",
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
            For a device's roll-up (``by`` ``device``) or a ``set`` view, its page
            past the first (``"2"``, ``"3"``, ...); ``None`` for the first.
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
        short_name: The member's stated short name (``names.csv``), or ``None``.
        tile: For a member with no light and no pair, the first key tile of its
            card that shows a channel: its value stands in the cell; else ``None``.
    """

    member: Link | Fact
    channels: tuple[str, ...]
    pair: ValueRow | None = None
    calibration: CalibrationCell | None = None
    short_name: str | None = None
    tile: KeyTile | None = None


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
        parts: The ``pictures.csv`` rows of that view, in file order, each with
            the channel its ``signal`` resolves to on the device.
        symbol: The class's strip symbol (the drawing settings' ``symbols``),
            whose colour the body is drawn in; ``None`` when it has none.
        focus: The parts outlined in the focus colour: on a part's card, the
            part itself on its owner's drawing.
    """

    device_class: str
    view: str
    parts: tuple[PicturePart, ...]
    symbol: str | None = None
    focus: tuple[str, ...] = ()


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
        side: A half-width unit's half, ``L`` or ``R``, from the claim's ``side``
            attribute; ``None``: full width.
    """

    position: int | None
    link: Link | Fact
    lights: tuple[Light, ...]
    size_u: int | float | None = None
    standing: str | None = None
    conflicting: bool = False
    citations: tuple[Citation, ...] = ()
    unit_text: str | None = None
    side: str | None = None


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
        precision: The decimals the row states for the number, or ``None``.
    """

    title: str
    measure: str
    unit: str | None
    caption: str | None
    channels: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str, str, float], ...] = ()
    precision: int | None = None


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


@dataclass(frozen=True)
class CalibrationTally:
    """How far one calibration kind is under a place, summed over its classes.

    Attributes:
        title: The kind's short title (``short``, else ``title``).
        good: The devices whose value in force is at their wanted basis, uncontested.
        total: The devices the kind applies to under the place.
        state: The worst present: ``contested`` > ``none`` > ``below_wanted`` >
            ``at_wanted``.
        basis: The wanted basis (an ``at_wanted`` chip's tint).
        link: The calibration view over the set holding most of the devices, or ``None``.
    """

    title: str
    good: int
    total: int
    state: str
    basis: str
    link: Link | None


@dataclass(frozen=True)
class CalibrationTallies:
    """A place's calibration chips: one per kind title with a value somewhere under
    it, in kind order; how many kinds have none yet; and where "more" leads (the
    place's roll-up; drawn only when the chips overflow their line), ``None`` on the
    roll-up itself."""

    tallies: tuple[CalibrationTally, ...]
    not_started: int
    more: Link | None


@dataclass(frozen=True)
class CardFoot:
    """A drawn card's last muted lines: where the device is, the sets it is in (when
    any), and what it is read from, each one line; the full provenance on hover.

    Attributes:
        where: The where line in pieces, each its text and the screen it opens (or
            ``None``): the places with their units, the whole, the positions.
        sources: The count of statements and the documents they cite.
        provenance: One line per statement, shown on hover.
        claims: What the where line is made of: the place, each ``located_in``
            claim (:class:`Placement`), each pose (:class:`Fact`).
        citations: The statements the sources line counts.
        sets: The set views of the sets the device is a member of, one link each,
            drawn on a line of their own between the two.
    """

    where: tuple[tuple[str, Link | None], ...]
    sources: str
    provenance: tuple[str, ...]
    claims: tuple[Item, ...] = ()
    citations: tuple[Citation, ...] = ()
    sets: tuple[Link, ...] = ()


@dataclass(frozen=True)
class CardHead:
    """A drawn card's title line under the bar: the device's class and recipe in
    muted words, and the recipe's sentence (``None`` with none).

    Attributes:
        kind: What it is: ``"<class> · <recipe> recipe"`` or ``"<class> · default
            recipe"``.
        description: The recipe's sentence, its relation slots filled.
        default: The card is drawn with the default recipe (its class has no row).
        nowhere: The recipe draws a strip, and where the device sits is not stated.
        by_whole: ``(place, whole)`` when the device has no place of its own and
            stands where its whole is (an accepted ``part_of``); else ``None``.
    """

    kind: str
    description: str | None
    default: bool = True
    nowhere: bool = False
    by_whole: tuple[str, str] | None = None


@dataclass(frozen=True)
class KeyTile:
    """One tile of a drawn card's top row: one value.

    Attributes:
        title: Its head.
        channel: The channel it shows, or ``None``: not served yet.
        shown_as: ``value`` (a number with its unit), ``word`` (the state word),
            ``lamp`` (a light and the state word), or a measure over *channels*:
            ``count_true`` ("k of N"), ``max`` or ``sum``.
        unit: The unit printed after the number, or ``None``.
        scale: The divisor the raw value is shown through, or ``None`` (1).
        precision: The decimals of the number, or ``None`` (significant digits).
        words: The stated word per value, ``(value, word)``.
        channels: For ``count_true``, ``max`` or ``sum``: every channel of the
            role on the device, the figure computed over them all.
        names: Each of *channels*' name short of the device's id, in the same
            order, where the tile names the channel that holds the figure; empty
            where it names none.
    """

    title: str
    channel: str | None
    shown_as: str = "value"
    unit: str | None = None
    scale: float | None = None
    precision: int | None = None
    words: tuple[tuple[str, str], ...] = ()
    channels: tuple[str, ...] = ()
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class BlockRow:
    """One line inside a chain block: a muted label and what it shows.

    Attributes:
        label: The line's label (a role or a relation word).
        channel: A live channel, or ``None``.
        text: A stated text (a port, a class, ``not stated``), when no channel.
        shown_as: ``value``, ``word``, ``lamp`` or ``text``.
        unit, scale, words: As :class:`KeyTile`.
    """

    label: str
    channel: str | None = None
    text: str | None = None
    shown_as: str = "text"
    unit: str | None = None
    scale: float | None = None
    words: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ChainBlock:
    """One block of a chain: a thing, its lines, the arrow to the next block.

    Attributes:
        title: Its title (the thing's class word and id), or ``not stated``.
        link: Its card, or ``None``.
        rows: Its lines.
        focus: It is the card's own device, drawn in the focus colour.
        dashed: Nothing is stated for this hop.
        arrow: The word on the arrow into the next block, or ``None``.
        standing: ``contested`` when sources disagree on the hop, else ``None``.
        stack: More blocks under it in its column: the other things one hop
            reaches (all of them, or the answers of a contested hop).
        arrow_in: The word on the arrow into it (its hop's port), or ``None``.
        inferred: For a hop on an inferred relation, the rule it rests on (shown
            on hover beside the word "inferred"); ``None`` for a stated one.
    """

    title: str
    link: Link | None
    rows: tuple[BlockRow, ...] = ()
    focus: bool = False
    dashed: bool = False
    arrow: str | None = None
    standing: str | None = None
    stack: tuple[ChainBlock, ...] = ()
    arrow_in: str | None = None
    inferred: str | None = None


@dataclass(frozen=True)
class Chain:
    """Blocks left to right, arrows between them, a muted line under them."""

    blocks: tuple[ChainBlock, ...]
    foot: str | None = None


@dataclass(frozen=True)
class StripMember:
    """One thing on a card's strip.

    Attributes:
        thing: Its id.
        label: What is written under it (its short name, else its id).
        s_m: Where it is along the beam, m.
        symbol: The strip symbol ``symbols.csv`` names (``pump``, ``valve``, ...).
        lane: ``below``, ``beam`` or ``above`` the pipe.
        link: Its card, or ``None`` (the card's own device has none).
        channel: The channel of the section's role on it, or ``None``.
        unit: That channel's unit, or ``None``.
        focus: It is the card's own device.
    """

    thing: str
    label: str
    s_m: float
    symbol: str
    lane: str
    link: Link | None = None
    channel: str | None = None
    unit: str | None = None
    focus: bool = False


@dataclass(frozen=True)
class CardStrip:
    """A card's strip: the things of the stated classes within a window of the
    device along the beam, to scale.

    Attributes:
        window: The stretch drawn, ``(from, to)`` in m.
        members: The things on it, by ``s_m``.
        bracket: Where the device is when only a place around it is placed:
            ``(from, to, label)`` in m; ``None`` when the device is placed.
        caption: The line under it, or ``None``.
    """

    window: tuple[float, float]
    members: tuple[StripMember, ...]
    bracket: tuple[float, float, str] | None = None
    caption: str | None = None


#: Entries of one role past this many are laid out as a grid under the role's title,
#: and their line names no channel (each on hover).
ENTRIES_IN_ONE_COLUMN = 6


@dataclass(frozen=True)
class Control:
    """One press or entry of a drawn card's operate block.

    Attributes:
        label: The word on it (an entry's: what it sets).
        channel: The channel it writes.
        press_value: What a press writes; ``None`` for an entry, or a press whose
            value no source states (drawn, not pressable).
        status: The channel whose value shades a choice's segment, or ``None``.
        inferred: The press value is inferred, not stated by the channel's own
            database; its words say so wherever they name it.
    """

    label: str
    channel: str
    press_value: float | None = None
    status: str | None = None
    inferred: bool = False


@dataclass(frozen=True)
class ControlGroup:
    """Controls of one kind that belong together, and the line saying what they write.

    Attributes:
        kind: ``action``, ``choice``, ``entry``, or ``withheld``: the presses of
            a channel the card reaches only through an inferred join, shown read
            only where they would have stood.
        title: What the group is (a role), or ``None``.
        controls: Its controls.
        writes: The muted line under it: the channels it writes and the values.
    """

    kind: str
    title: str | None
    controls: tuple[Control, ...]
    writes: str
    rule: str | None = None  # a withheld group's join rule, on hover


@dataclass(frozen=True)
class Operate:
    """A drawn card's operate block; with no group, *note* says why."""

    groups: tuple[ControlGroup, ...]
    note: str | None = None


@dataclass(frozen=True)
class ChannelGrid:
    """Every channel of the device not drawn above, compact: (short name, channel);
    *hidden* driver-internal records are left out and counted under it, with *full*
    the link to the ``channels`` view that lists them, read only."""

    rows: tuple[tuple[str, str], ...]
    columns: int = 4
    hidden: int = 0
    full: Link | None = None


@dataclass(frozen=True)
class ChannelCount:
    """In place of the grid, for a device with more channels than a card lists: how
    many there are by owner, and the link to its ``channels`` view.

    Attributes:
        total: The channels the full list holds.
        parts: The counts by owner: the device's own, each device a role walks to,
            the driver-internal records.
        link: The ``channels`` view.
    """

    total: int
    parts: tuple[str, ...]
    link: Link


@dataclass(frozen=True)
class FarEnd:
    """A thing on a connector of a front.

    Attributes:
        thing: Its id.
        link: Its card, or ``None`` when it has none.
        port: Its own port on the cable, when stated.
        inferred: For an end on an inferred relation, the rule it rests on.
    """

    thing: str
    link: Link | None = None
    port: str | None = None
    inferred: str | None = None


@dataclass(frozen=True)
class Connector:
    """One connector of a front: an outlet, a port, a socket, a channel.

    Attributes:
        label: Its word under the socket (``Outlet 7``, ``CH1``).
        kind: ``socket``, ``hv-out``, ``eth``, ``power``, ``io`` or ``slot``.
        lamp: Its state channel, or ``None``.
        sides: What is on it, per side (one side; a socket of its own has two):
            each side the things found, none (nothing stated) or two and more
            (contested, all drawn).
        rows: Values of the section's roles for this connector or what is on it.
        controls: Presses the section's control roles offer on this connector.
    """

    label: str
    kind: str
    lamp: str | None = None
    sides: tuple[tuple[FarEnd, ...], ...] = ((),)
    rows: tuple[BlockRow, ...] = ()
    controls: tuple[Control, ...] = ()


@dataclass(frozen=True)
class FrontGroup:
    """The connectors of one group of a face, in the unit's own order.

    ``empty`` is the line drawn when the group has no connector stated (a switch's
    ports, whose count and assignment no source states)."""

    face: str
    title: str
    connectors: tuple[Connector, ...]
    empty: str | None = None


@dataclass(frozen=True)
class Front:
    """A device's faces with their connectors and what is on each.

    Attributes:
        groups: The groups, by ``order``.
        loose: Things the face's relations name on no stated port, by class:
            ``(class, things)``; each thing with its rows.
        loose_note: The line over them.
        writes: The line of what the connectors' presses write, or ``None``.
        caption: The line under it, or ``None``.
        entries: The typed values of the section's roles per connector, one group
            per role, each entry named by its connector: drawn under the front.
    """

    groups: tuple[FrontGroup, ...]
    loose: tuple[tuple[str, tuple[tuple[FarEnd, tuple[BlockRow, ...]], ...]], ...] = ()
    loose_note: str | None = None
    writes: str | None = None
    caption: str | None = None
    entries: tuple[ControlGroup, ...] = ()


@dataclass(frozen=True)
class RingMember:
    """One member of a ring-strip: a column with its name and values.

    Attributes:
        thing: Its id.
        label: The name its column shows.
        link: Its card, or ``None`` for the card's own device.
        rows: The section's roles it plays, each a value or a lamp.
        focus: Whether it is the card's own device.
        inferred: For a member on an inferred relation, the rule it rests on.
        stated: Marked "stated": in a section most of whose members are inferred
            (its caption says so), the member that rests on a stated relation.
    """

    thing: str
    label: str
    link: Link | None
    rows: tuple[BlockRow, ...]
    focus: bool = False
    inferred: str | None = None
    stated: bool = False


@dataclass(frozen=True)
class RingStrip:
    """The devices of a group, one column each, in ring order.

    Attributes:
        members: In order along the ring, else in number order.
        ordered: ``s`` (by ring position) or ``number`` (no position stated for one).
        caption: The line under it, or ``None``.
        places: When every member has a place: the runs of *members* by place, in
            ring order, ``(place, how many)``; empty otherwise.
        counted: The words over the count of members whose lamp is ok now
            ("giving permit now"), or ``None``.
    """

    members: tuple[RingMember, ...]
    ordered: str
    caption: str | None = None
    places: tuple[tuple[str, int], ...] = ()
    counted: str | None = None


@dataclass(frozen=True)
class LimitLadder:
    """A device's value against its set points, in their stated order.

    Attributes:
        value: The value's channel, or ``None`` when it is not served.
        unit: The value's unit, or ``None``.
        marks: Each set point in order: its title and channel.
        left_out: The set points of the section no channel serves (not drawn).
        caption: The line under it, or ``None``.
    """

    value: str | None
    unit: str | None
    marks: tuple[tuple[str, str], ...]
    left_out: tuple[str, ...] = ()
    caption: str | None = None


@dataclass(frozen=True)
class PartFacts:
    """Beside a part's drawing: what is stated about it, and what is not.

    Attributes:
        stated: Its relations, each a link to the other thing (or a fact naming it).
        not_stated: Each fact the card needs that no source states.
    """

    stated: tuple[Link | Fact, ...]
    not_stated: tuple[str, ...]


@dataclass(frozen=True)
class CurvePlot:
    """A curve the records state for a device, as x/y points.

    Attributes:
        title: The calibration kind's title.
        points: ``(x, y)`` in order of x.
        x_unit, y_unit: The units of the axes, or ``None``.
        source: The document and version of the claim in force, or ``None``.
        note: What is not drawn and why (no curve stated, no operating point).
    """

    title: str
    points: tuple[tuple[float, float], ...]
    x_unit: str | None = None
    y_unit: str | None = None
    source: str | None = None
    note: str | None = None


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
    | CalibrationTallies
    | CardHead
    | KeyTile
    | Chain
    | CardStrip
    | Operate
    | ChannelGrid
    | ChannelCount
    | CardFoot
    | Front
    | RingStrip
    | LimitLadder
    | PartFacts
    | CurvePlot
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
