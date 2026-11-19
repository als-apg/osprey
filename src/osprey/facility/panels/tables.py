"""The rule tables a facility repository writes beside its facility file.

A panels directory holds the CSV files below; :data:`COLUMNS` is each one's header
line, the schema this module holds them to:

- ``positions.csv``: where a thing sits, one row per thing and frame;
- ``record_types.csv``: each channel's record type and how a screen draws it,
  one row per channel;
- ``health.csv``: the state a channel's value means, one row per stated value
  and source, so two sources may state two states for one value;
- ``excluded.csv``: things, channels and documents no screen shows;
- ``pictures.csv``: the drawing of a device class, seen from the end or the
  side: its body (yoke, poles, coils, corrector windings, bore) and its parts
  at their positions (see :class:`PicturePart`);
- ``names.csv``: a thing's short name; ``metrics.csv``: the key numbers;
- ``calibration_kinds.csv``: the calibration quantities a device class has;
- ``calibrations.csv``: the calibration claims, one value of one kind for one
  subject each, with basis, date, source and standing;
- ``calibration_curves.csv``: the points of a claim whose value is a curve;
- ``state_labels.csv``: the word a discrete channel's value means, one row per
  channel and value, as its record states it;
- ``units.csv``: a channel's engineering unit and the divisor its raw value is
  shown through (``0.1A``: unit ``A``, scale 10);
- ``channel_roles.csv``: what a channel is to a device card, by class and name
  pattern (see :class:`ChannelRole`);
- ``card_recipes.csv``: a class's card recipe and its title-line sentence;
- ``card_sections.csv``: a recipe's drawn sections in order;
- ``chains.csv``: the hops of a recipe's chains;
- ``symbols.csv``: how a card's strip draws a device of a class: its symbol and lane.

Every file may be absent, which reads as empty: what the facility file states
(labels, roles by signal) is the default; a setpoint's
readback is the facility file's alone (its channel ``pair``). Across
the calibration files, a claim's kind must be a stated kind, a claim's
``supersedes`` a stated claim, and a curve's ``id`` a claim whose value is a
curve. A header must name exactly the file's columns (in
any order). A row must have one field per column; a field the schema requires
must be non-empty, a field with a vocabulary must hold one of its words, a
number must parse, and a key the file is unique on must not repeat. An empty
field that is not required reads as ``None``. Every refusal is a
:class:`PanelTableError` naming the file and the 1-based line the row starts
on.
"""

from __future__ import annotations

import csv
import dataclasses
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

__all__ = [
    "BASES",
    "COLUMNS",
    "DRAWN_AS",
    "EXCLUDED_KINDS",
    "HEALTH_STATES",
    "OPTIONAL_COLUMNS",
    "CARD_MEASURES",
    "OPTIONAL_FILES",
    "TOLERANCE_TERMS",
    "PICTURE_SHAPES",
    "PICTURE_VIEWS",
    "PROBE_NEEDS",
    "PROBE_SHAPES",
    "ProbeElement",
    "TABLE_FILES",
    "Exclusion",
    "HealthRule",
    "PanelTableError",
    "PanelTables",
    "PicturePart",
    "ShortName",
    "StateLabel",
    "ChannelRole",
    "ChannelUnit",
    "CardRecipe",
    "CardSection",
    "ChainStep",
    "FrontRow",
    "CardRule",
    "CARD_ACTIONS",
    "CARD_CONDITIONS",
    "WALKED_RELATIONS",
    "StripSymbol",
    "Metric",
    "CalibrationKind",
    "Calibration",
    "CurvePoint",
    "CALIBRATION_BASES",
    "CALIBRATION_SHAPES",
    "STANDINGS",
    "Pose",
    "RecordType",
    "read_tables",
]

POSITIONS = "positions.csv"
RECORD_TYPES = "record_types.csv"
HEALTH = "health.csv"
EXCLUDED = "excluded.csv"
PICTURES = "pictures.csv"
NAMES = "names.csv"
METRICS = "metrics.csv"
CALIBRATION_KINDS = "calibration_kinds.csv"
CALIBRATIONS = "calibrations.csv"
CALIBRATION_CURVES = "calibration_curves.csv"
PROBE_PICTURES = "probe_pictures.csv"
STATE_LABELS = "state_labels.csv"
UNITS = "units.csv"
CHANNEL_ROLES = "channel_roles.csv"
CARD_RECIPES = "card_recipes.csv"
CARD_SECTIONS = "card_sections.csv"
CHAINS = "chains.csv"
STRIP_SYMBOLS = "symbols.csv"
FRONTS = "fronts.csv"
CARDS = "cards.csv"
#: What a ``cards.csv`` row does to the devices it selects: give each a card the
#: default would not, or take the card the default would give.
CARD_ACTIONS: tuple[str, ...] = ("keep", "drop")
#: The rule tables, by file name.
TABLE_FILES: tuple[str, ...] = (
    POSITIONS,
    RECORD_TYPES,
    HEALTH,
    EXCLUDED,
    PICTURES,
    NAMES,
    METRICS,
    CALIBRATION_KINDS,
    CALIBRATIONS,
    CALIBRATION_CURVES,
    PROBE_PICTURES,
    STATE_LABELS,
    UNITS,
    CHANNEL_ROLES,
    CARD_RECIPES,
    CARD_SECTIONS,
    CHAINS,
    STRIP_SYMBOLS,
    FRONTS,
    CARDS,
)

#: Each file's columns, in the order the facility repository writes them.
COLUMNS: Mapping[str, list[str]] = MappingProxyType(
    {
        POSITIONS: [
            "thing",
            "frame",
            "x_mm",
            "y_mm",
            "z_mm",
            "ax_rad",
            "ay_rad",
            "az_rad",
            "s_m",
            "side_mm",
            "height_mm",
            "derived_assumes",
            "box_mm",
            "basis",
            "document",
            "version",
            "state",
            "locator",
        ],
        RECORD_TYPES: [
            "channel",
            "record_type",
            "drawn_as",
            "basis",
            "source",
            "revision",
            "locator",
        ],
        HEALTH: ["channel", "value", "state", "evidence", "basis", "source", "revision", "locator"],
        EXCLUDED: ["kind", "id", "reason", "source", "revision", "locator"],
        PICTURES: [
            "device_class",
            "view",
            "part",
            "kind",
            "x_mm",
            "y_mm",
            "z_mm",
            "document",
            "version",
            "state",
            "locator",
            "w_mm",
            "h_mm",
            "angle_deg",
            "signal",
            "under",
            "devices",
        ],
        STATE_LABELS: ["channel", "value", "label", "source", "revision", "locator"],
        UNITS: ["channel", "unit", "scale", "label", "source", "revision", "locator"],
        CHANNEL_ROLES: [
            "class",
            "role",
            "signal",
            "scope",
            "shown_as",
            "unit",
            "scale",
            "press_value",
            "label",
            "basis",
            "source",
            "revision",
            "locator",
        ],
        CARD_SECTIONS: [
            "recipe",
            "order",
            "kind",
            "title",
            "focus",
            "members",
            "window_m",
            "roles",
            "caption",
            "source",
        ],
        CHAINS: [
            "recipe",
            "chain",
            "step",
            "from_step",
            "relation",
            "direction",
            "to",
            "many",
            "port_from",
            "title",
            "roles",
            "source",
        ],
        STRIP_SYMBOLS: ["class", "devices", "symbol", "lane", "source"],
        CARDS: [
            "action",
            "class",
            "under",
            "devices",
            "attribute",
            "when",
            "owner_relation",
            "owner_direction",
            "owner_to",
            "reason",
            "source",
        ],
        FRONTS: [
            "recipe",
            "face",
            "group",
            "order",
            "connector",
            "kind",
            "count",
            "channel",
            "far_relation",
            "far_direction",
            "far_port",
            "label",
            "source",
            "roles",
            "apart",
        ],
        CARD_RECIPES: [
            "class",
            "under",
            "devices",
            "recipe",
            "description",
            "channels_cols",
            "source",
        ],
        NAMES: [
            "thing",
            "short_name",
            "scheme",
            "basis",
            "document",
            "version",
            "state",
            "locator",
        ],
        METRICS: [
            "view",
            "level",
            "order",
            "title",
            "measure",
            "signal",
            "class",
            "unit",
            "caption",
            "source",
            "precision",
        ],
        CALIBRATION_KINDS: [
            "kind",
            "title",
            "class",
            "shape",
            "unit",
            "wanted_basis",
            "tolerance",
            "tolerance_source",
            "held_in",
            "order",
            "under",
            "short",
        ],
        CALIBRATIONS: [
            "id",
            "kind",
            "subject",
            "value",
            "unit",
            "conditions",
            "basis",
            "method",
            "date",
            "document",
            "version",
            "state",
            "supersedes",
            "standing",
        ],
        CALIBRATION_CURVES: ["id", "x", "y"],
        PROBE_PICTURES: [
            "class",
            "element",
            "shape",
            "label",
            "angle_deg",
            "radius_mm",
            "size_mm",
            "signal",
            "signal_y",
            "min",
            "max",
            "range_mm",
            "count",
            "document",
            "version",
            "state",
            "locator",
            "under",
        ],
    }
)

#: The columns a file's header may leave out; an absent one reads as empty.
OPTIONAL_COLUMNS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        CALIBRATION_KINDS: frozenset({"under", "short"}),
        # a file of the first eleven columns (parts at points, no signals) still reads
        PICTURES: frozenset({"w_mm", "h_mm", "angle_deg", "signal", "under", "devices"}),
        PROBE_PICTURES: frozenset({"under"}),
        METRICS: frozenset({"precision"}),
        CARD_RECIPES: frozenset({"devices"}),
        STRIP_SYMBOLS: frozenset({"devices"}),
        FRONTS: frozenset({"roles", "apart"}),
    }
)

#: The files a panels directory may leave out; an absent one reads as empty.
OPTIONAL_FILES = frozenset(TABLE_FILES)

Basis = Literal["stated", "inferred"]
DrawnAs = Literal["value", "setpoint", "state", "command", "text", "waveform", "untyped"]
HealthState = Literal["normal", "warn", "bad"]
ExcludedKind = Literal["thing", "channel", "document"]

BASES: tuple[str, ...] = ("stated", "inferred")
DRAWN_AS: tuple[str, ...] = ("value", "setpoint", "state", "command", "text", "waveform", "untyped")
HEALTH_STATES: tuple[str, ...] = ("normal", "warn", "bad")
EXCLUDED_KINDS: tuple[str, ...] = ("thing", "channel", "document")
PICTURE_VIEWS: tuple[str, ...] = ("end", "side")
#: The kinds of a ``pictures.csv`` row that draw the body of the class, not a part
#: (in a file that has the size columns; in one without, every row is a part).
PICTURE_SHAPES: tuple[str, ...] = ("yoke", "pole", "coil", "corrector_winding", "bore", "gap")
METRIC_VIEWS: tuple[str, ...] = ("place", "rollup", "card")
#: What a key number computes over the channels its row selects.
MEASURES: tuple[str, ...] = (
    "value",
    "word",
    "count_true",
    "count_false",
    "count_in_tolerance",
    "rms",
    "max",
    "min",
    "sum",
)
#: What a card's key number may compute: one value or word, or over every channel
#: of its role on the device a count of the true ones, the largest or the sum.
CARD_MEASURES: tuple[str, ...] = ("value", "word", "count_true", "max", "sum")

#: How a calibration was found, in the order the screens rank it: a display order
#: only (which basis to show as further along), not a rule of which value counts.
CALIBRATION_BASES: tuple[str, ...] = ("design", "bench", "survey", "beam")
#: What a calibration value is: a number, numbers separated by ``;``, or a curve.
CALIBRATION_SHAPES: tuple[str, ...] = ("scalar", "tuple", "curve")
#: What a row of ``probe_pictures.csv`` draws: the bore, a button, the beam dot,
#: a curve of a plot, or a value in the line of readouts.
PROBE_SHAPES: tuple[str, ...] = ("bore", "anchor", "marker", "plot", "readout")
#: Per probe shape, the columns its rows must fill.
PROBE_NEEDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "bore": ("radius_mm",),
        "anchor": ("angle_deg", "radius_mm", "size_mm"),
        "marker": ("signal", "signal_y", "range_mm"),
        "plot": ("signal",),
        "readout": ("label", "signal"),
    }
)


def _walked_relations() -> tuple[str, ...]:
    """The vocabulary's relation types that run from a device to a device, sorted."""
    table = resources.files("osprey.facility.schema._generated") / "vocabulary.json"
    rows = json.loads(table.read_text(encoding="utf-8"))["relations"]
    return tuple(
        sorted(
            row["name"] for row in rows if "device" in row["subject"] and "device" in row["object"]
        )
    )


#: Where a ``channel_roles.csv`` row looks for its channels: the device's own, its
#: parts', or those of the device a relation of the device names (any relation
#: type of the vocabulary from a device to a device: what reads it out, controls
#: it, interlocks it, ...).
WALKED_RELATIONS: tuple[str, ...] = _walked_relations()
ROLE_SCOPES: tuple[str, ...] = ("device", "parts", *WALKED_RELATIONS)
#: How a card shows a channel of a role: a number, its state word, a lamp, or a
#: control (a typed entry, one segment of a choice, an action button).
SHOWN_AS: tuple[str, ...] = ("value", "word", "lamp", "entry", "choice", "action")
#: The kinds of a drawn card's section a ``card_sections.csv`` row names.
SECTION_KINDS: tuple[str, ...] = (
    "strip",
    "chain",
    "front",
    "section",
    "ring-strip",
    "calibration-strip",
    "curve",
    "ladder",
)
#: The lanes a strip puts a symbol in (a place's strip has one lane off the beam:
#: ``above`` and ``below`` both draw there).
STRIP_LANES: tuple[str, ...] = ("below", "beam", "above")
#: Which way a chain hop walks its relation: from the subject (``out``) or to it.
DIRECTIONS: tuple[str, ...] = ("out", "in")
#: How a ``fronts.csv`` row finds what is on a connector: a thing whose relation to
#: the device names the connector (``in``), the device's own relation from it
#: (``out``), or the connector as a thing of its own, both directions (``socket``).
FAR_DIRECTIONS: tuple[str, ...] = ("in", "out", "socket")
#: The kinds of connector a front draws.
CONNECTOR_KINDS: tuple[str, ...] = ("socket", "hv-out", "eth", "power", "io", "slot")
#: The ways of showing that are a control a press or a typed value writes.
CONTROLS: tuple[str, ...] = ("entry", "choice", "action")
#: A claim's standing, as relation claims have it.
STANDINGS: tuple[str, ...] = ("accepted", "contested", "superseded")

# Per file: the columns that must be non-empty, the columns with a vocabulary,
# the number columns (required or not by the first set) and the unique key.
_REQUIRED: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        POSITIONS: ("thing", "frame", "basis"),
        RECORD_TYPES: ("channel", "record_type", "basis"),
        HEALTH: ("channel", "value", "state", "basis"),
        EXCLUDED: ("kind", "id"),
        PICTURES: ("device_class", "view", "part", "kind"),
        NAMES: ("thing", "short_name", "basis"),
        METRICS: ("view", "order", "title", "measure", "signal"),
        CALIBRATION_KINDS: ("kind", "title", "class", "shape", "wanted_basis"),
        CALIBRATIONS: ("id", "kind", "subject", "value", "basis", "standing"),
        CALIBRATION_CURVES: ("id", "x", "y"),
        PROBE_PICTURES: ("class", "element", "shape"),
        STATE_LABELS: ("channel", "value", "label"),
        UNITS: ("channel",),
        CHANNEL_ROLES: ("class", "role", "signal", "scope", "shown_as", "basis"),
        CARD_RECIPES: ("class", "recipe"),
        CARD_SECTIONS: ("recipe", "order", "kind", "title"),
        CHAINS: ("recipe", "chain", "step", "relation", "direction", "title"),
        STRIP_SYMBOLS: ("class", "symbol", "lane"),
        CARDS: ("action", "class"),
        FRONTS: (
            "recipe",
            "face",
            "group",
            "order",
            "connector",
            "kind",
        ),
    }
)
_VOCABULARY: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        POSITIONS: {"basis": BASES},
        RECORD_TYPES: {"drawn_as": DRAWN_AS, "basis": BASES},
        HEALTH: {"state": HEALTH_STATES, "basis": BASES},
        EXCLUDED: {"kind": EXCLUDED_KINDS},
        # a part's kind is the facility's word: any text
        PICTURES: {"view": PICTURE_VIEWS},
        NAMES: {"basis": BASES},
        METRICS: {"view": METRIC_VIEWS, "measure": MEASURES},
        CALIBRATION_KINDS: {"shape": CALIBRATION_SHAPES, "wanted_basis": CALIBRATION_BASES},
        CALIBRATIONS: {"basis": CALIBRATION_BASES, "standing": STANDINGS},
        CALIBRATION_CURVES: {},
        PROBE_PICTURES: {"shape": PROBE_SHAPES},
        STATE_LABELS: {},
        UNITS: {},
        CHANNEL_ROLES: {"scope": ROLE_SCOPES, "shown_as": SHOWN_AS, "basis": BASES},
        CARD_RECIPES: {},
        CARD_SECTIONS: {"kind": SECTION_KINDS},
        CHAINS: {"direction": DIRECTIONS, "many": ("yes", "1", "0")},
        STRIP_SYMBOLS: {"lane": STRIP_LANES},
        FRONTS: {"kind": CONNECTOR_KINDS, "far_direction": FAR_DIRECTIONS, "apart": ("no",)},
        CARDS: {"action": CARD_ACTIONS, "owner_direction": DIRECTIONS},
    }
)
_NUMBERS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        POSITIONS: (
            "x_mm",
            "y_mm",
            "z_mm",
            "ax_rad",
            "ay_rad",
            "az_rad",
            "s_m",
            "side_mm",
            "height_mm",
        ),
        PICTURES: ("x_mm", "y_mm", "z_mm", "w_mm", "h_mm", "angle_deg"),
        METRICS: ("order",),
        CALIBRATION_KINDS: ("order",),
        CALIBRATION_CURVES: ("x", "y"),
        PROBE_PICTURES: ("angle_deg", "radius_mm", "size_mm", "min", "max", "range_mm", "count"),
        UNITS: ("scale",),
        CHANNEL_ROLES: ("scale", "press_value"),
        CARD_RECIPES: ("channels_cols",),
        CARD_SECTIONS: ("order", "window_m"),
        CHAINS: ("step", "from_step"),
        FRONTS: ("order", "count"),
    }
)
_UNIQUE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        POSITIONS: ("thing", "frame"),
        RECORD_TYPES: ("channel",),
        NAMES: ("thing",),
        METRICS: ("view", "level", "class", "order"),
        CALIBRATION_KINDS: ("kind", "class"),
        CALIBRATIONS: ("id",),
        STATE_LABELS: ("channel", "value"),
        UNITS: ("channel",),
        CARD_RECIPES: ("class", "under", "devices"),
        CARD_SECTIONS: ("recipe", "order"),
        CHAINS: ("recipe", "chain", "step"),
        STRIP_SYMBOLS: ("class", "devices"),
        FRONTS: ("recipe", "face", "group", "connector"),
        CARDS: ("class", "under", "devices", "attribute", "when"),
    }
)


class PanelTableError(ValueError):
    """A rule table the panels cannot read.

    Attributes:
        file: The table's file name, ``health.csv`` for example.
        line: The 1-based line the refused row starts on (1 for the header),
            or ``None`` when the file itself is missing.
    """

    def __init__(self, file: str, line: int | None, detail: str) -> None:
        self.file = file
        self.line = line
        where = file if line is None else f"{file} line {line}"
        super().__init__(f"{where}: {detail}")


@dataclass(frozen=True)
class Pose:
    """One row of ``positions.csv``: where a thing sits in one frame.

    Numbers are ``None`` where the row leaves them empty.
    ``on_beam`` is set by the graph, which knows each model's beam frame.
    """

    thing: str
    frame: str
    x_mm: float | None
    y_mm: float | None
    z_mm: float | None
    ax_rad: float | None
    ay_rad: float | None
    az_rad: float | None
    s_m: float | None
    side_mm: float | None
    height_mm: float | None
    derived_assumes: str | None
    box_mm: str | None
    basis: Basis
    document: str | None
    version: str | None
    state: str | None
    locator: str | None
    on_beam: bool = False

    @property
    def is_beam_position(self) -> bool:
        """Whether ``s_m`` is a position along a beam path: stated, in the thing's beam
        frame, nothing assumed."""
        return self.on_beam and self.s_m is not None and not self.derived_assumes


@dataclass(frozen=True)
class RecordType:
    """One row of ``record_types.csv``; an empty ``drawn_as`` is the drawing of the
    channel's role (a setpoint as an entry, a readback as a value)."""

    channel: str
    record_type: str
    drawn_as: DrawnAs | None
    basis: Basis
    source: str | None
    revision: str | None
    locator: str | None


@dataclass(frozen=True)
class HealthRule:
    """One row of ``health.csv``: the state one value of a channel means to one source."""

    channel: str
    value: str
    state: HealthState
    evidence: str | None
    basis: Basis
    source: str | None
    revision: str | None
    locator: str | None


@dataclass(frozen=True)
class Exclusion:
    """One row of ``excluded.csv``."""

    kind: ExcludedKind
    id: str
    reason: str | None
    source: str | None
    revision: str | None
    locator: str | None


@dataclass(frozen=True)
class PicturePart:
    """One row of ``pictures.csv``: an element of a device class's drawing.

    The drawing is in the plane of ``view``: ``end`` draws ``x_mm`` across and
    ``y_mm`` up, looking along the beam; ``side`` draws ``z_mm`` across and
    ``y_mm`` up. The origin is the beam axis. ``document``, ``version``,
    ``state`` and ``locator`` cite where the row is stated.

    By ``kind`` (:data:`PICTURE_SHAPES`; any other word is a part):

    - ``yoke``: the iron, an ellipse ``w_mm`` across and ``h_mm`` high (a
      circle when ``h_mm`` is empty) centred at ``x_mm``, ``y_mm``;
    - ``outline``: one point of the yoke as a polygon (a hexagon, a C or an H),
      the class's rows of this kind joined in file order;
    - ``pole``: a rectangle ``w_mm`` wide and ``h_mm`` long, its length along
      ``angle_deg`` (0 is +x, counter-clockwise), centred at ``x_mm``, ``y_mm``;
    - ``coil``: the coil round the pole at the same ``angle_deg``: two sides,
      each ``w_mm`` wide and ``h_mm`` long, one at each flank of that pole;
      with no such pole, one block where the row puts it; its number in
      ``part`` is drawn on the yoke ring at its angle (``coil 6``: 6); a
      ``signal`` with a health row sets the outline of its blocks by state;
    - ``corrector_winding``: a thin band outside each block of the coil at its
      ``angle_deg``; a second row at the same angle is a second band, stacked
      outside the first and drawn in a second colour;
    - ``bore``: the aperture, an ellipse as the yoke's, with the beam at its
      centre;
    - ``gap``: instead of a bore, the gap between two pole faces: ``w_mm``
      is its full height; ``angle_deg`` tilts the faces (a gradient); the beam
      at ``x_mm``, ``y_mm``;
    - any other word: a part, named by ``part``.

    A class of two poles is drawn as a rectangular yoke with a pole above and below
    the gap; of more, as a ring yoke (or the outline) with the poles at their
    angles. A part sits by its kind: a water part (a kind naming a manifold, water
    or cooling) or a terminal block is a block in a column beside the yoke, right of
    it unless its row places it left: a position left of the beam, or with no
    position an ``angle_deg`` pointing left (a source that states only the side,
    "on the +x side", is ``angle_deg`` 0, and 180 for -x); with neither, a dashed
    block in the row of parts not placed (it claims no side); any other part
    is a sensor: with an ``angle_deg`` and no position, a dot on the coil at that
    angle (two on one coil side by side); with a position, a dot on the coil nearest
    its position's angle (a drawing with no coil: at its position); with neither, it
    is not placed: a hollow ring in a row of its kind beside the drawing, and the
    legend says "not placed".

    ``x_mm``, ``y_mm`` and ``z_mm`` are optional on every row: a facility leaves
    empty what no source states. A yoke, a bore or a gap with no position is
    centred on the beam.

    ``under`` (optional) is a place id: the row applies only to the devices of the
    class in that place or a place within it (:meth:`PanelGraph.contents`, the walk
    the place views use). Empty, it applies to every device of the class. Two
    machines with devices of one class name (two rings' ``QF``) each state their
    own rows under their own place; a device under neither gets no drawing, and
    ``check`` notes nothing for it.

    ``devices`` (optional) is an id pattern, as on ``card_recipes.csv``: the row
    applies only to the devices of the class whose id it matches; with ``!``
    first, to every device of the class but those, so a table can leave one
    device out of a drawing. Empty, it applies to every device ``under`` takes.

    A shape row may leave its size empty, the normal case: a pole, a coil or a
    corrector winding is then placed by its ``angle_deg`` alone (its ``x_mm``,
    ``y_mm`` are not read), at the default proportions of
    :mod:`osprey.facility.panels.picture`: a broad pole from the yoke's cavity to a
    rounded tip short of the bore, the coil blocks along its flanks. The proportions
    are scaled to a stated bore (or gap), else to a stated yoke or outline, else to
    :data:`osprey.facility.panels.picture.DEFAULT_BORE_MM`. The caption under the
    drawing's scale bar names the row the drawing is scaled from and the kinds drawn
    at the default proportion ("scaled from the yoke; pole, coil at default
    proportions"), or says "size not stated" and "schematic, not to scale" when no
    row states a size (a row with state ``schematic`` adds the second too); a yoke no
    row sizes has a dashed outline.

    ``signal`` (any row) names what the element shows live: a signal name
    stated on a channel, or a channel pattern with ``*``, ``?`` and ``[``
    matched against channel names; looked for on the device, then on its
    parts (the devices ``part_of`` it), then where it is read out
    (:meth:`PanelGraph.picture_matches`).
    A part whose signal has a health row shows its state where it sits; a value
    its channel prints stands beside it; a part with no signal is drawn plain. No part is named on the drawing: its tooltip and the
    legend under the drawing name it.

    The last four columns are optional: a file of the first eleven reads as
    before, every row a part at a point, a shape kind included. ``w_mm`` and
    ``h_mm``, when stated, are above 0.
    A class with shape or outline rows is drawn; one with parts only has them in
    a dashed frame that says the outline is not stated; one with no rows has no
    picture. An invented four-pole class (one pole, its coil, the bore, two
    parts shown; ``x_mm``/``y_mm`` of a pole are its centre)::

        device_class,view,part,kind,x_mm,y_mm,z_mm,document,version,state,locator,w_mm,h_mm,angle_deg,signal
        Quad4,end,yoke,yoke,0,0,0,DWG-100,B,released,sheet 1,260,,,
        Quad4,end,pole 1,pole,35.4,35.4,0,DWG-100,B,released,sheet 1,24,70,45,
        Quad4,end,coil 1,coil,45.3,45.3,0,DWG-100,B,released,sheet 1,10,36,45,
        Quad4,end,bore,bore,0,0,0,DWG-100,B,released,sheet 1,30,,,
        Quad4,end,switch 1,thermal_switch,70,95,0,DWG-100,B,released,sheet 3,,,,switch_1_status
        Quad4,end,water in,manifold,150,-40,0,DWG-100,B,released,sheet 3,,,,
    """

    device_class: str
    view: str
    part: str
    kind: str
    x_mm: float | None
    y_mm: float | None
    z_mm: float | None
    document: str | None
    version: str | None
    state: str | None
    locator: str | None
    w_mm: float | None = None
    h_mm: float | None = None
    angle_deg: float | None = None
    signal: str | None = None
    under: str | None = None
    devices: str | None = None
    #: The channel ``signal`` resolves to on one device: set when a card is
    #: described (:func:`osprey.facility.panels.describe.device_card`), ``None``
    #: as the table is read.
    channel: str | None = None
    #: Read from a file without the size columns: the row is a part at a point,
    #: whatever its kind.
    as_point: bool = False

    @property
    def shape(self) -> bool:
        """Whether the row draws the body: a shape kind, from a file with the size columns."""
        return self.kind in PICTURE_SHAPES and not self.as_point

    @property
    def sized(self) -> bool:
        """Whether the row states its size (its width)."""
        return self.w_mm is not None

    @property
    def positioned(self) -> bool:
        """Whether the row states its position in its view's plane."""
        across = self.z_mm if self.view == "side" else self.x_mm
        return across is not None and self.y_mm is not None

    @property
    def placed(self) -> bool:
        """Whether the row places its part: a position, or an angle (its coil's)."""
        return self.positioned or self.angle_deg is not None


@dataclass(frozen=True)
class StateLabel:
    """One row of ``state_labels.csv``: the word a discrete channel's *value* means,
    as its record states it (kept as stated, even where it reads inverted: the
    colour of a state comes from ``health.csv``, never from the word)."""

    channel: str
    value: str
    label: str
    source: str | None
    revision: str | None
    locator: str | None


@dataclass(frozen=True)
class ChannelUnit:
    """One row of ``units.csv``: a channel's unit as its record states it
    (``label``), cut to the unit (``unit``, ``None`` when the label is no unit) and
    the divisor (``scale``) its raw value is shown through."""

    channel: str
    unit: str | None
    scale: float | None
    label: str | None
    source: str | None
    revision: str | None
    locator: str | None


@dataclass(frozen=True)
class ChannelRole:
    """One row of ``channel_roles.csv``: on a device of class ``cls``, the channels
    whose name matches ``signal`` (a pattern with ``*``, ``?`` and ``[``, matched
    against the whole name), looked for where ``scope`` says, play ``role`` and are
    shown as ``shown_as``. In a relation scope (``read_out_by``, ``interlocked_by``,
    ...) the pattern may hold ``{port}`` (the relation's object port) and ``{n}`` (that
    port's trailing number). A ``choice`` or ``action`` row is one press:
    ``label`` is the word on it, ``press_value`` what it writes (``None``: the
    value whose state word is ``label``). On any other row a stated ``label`` is
    what the card calls the channel, on its tile, its value row and over its
    group, and stays in front of what tells two channels of the role apart; with
    none, the role's name. ``unit`` and ``scale`` (a divisor) win
    over ``units.csv``. The role words are the facility's vocabulary; the engine
    reads them as text."""

    cls: str
    role: str
    signal: str
    scope: str
    shown_as: str
    unit: str | None
    scale: float | None
    press_value: float | None
    label: str | None
    basis: Basis
    source: str | None
    revision: str | None
    locator: str | None
    #: False for the role a channel's own ``signal`` gives it when no row names it
    #: (:func:`osprey.facility.panels.model.signal_role`): it names what the channel
    #: is, never how the card presses it.
    stated: bool = True
    #: The ``title`` the signal named ``role`` states (in the vocabulary or
    #: ``signals.yaml``): what the card calls the channel when ``label`` is empty.
    title: str | None = None


def _channel_role(**columns: Any) -> ChannelRole:
    return ChannelRole(cls=columns.pop("class"), **columns)


@dataclass(frozen=True)
class CardRecipe:
    """One row of ``card_recipes.csv``: the card recipe of the devices of ``cls``
    (under the place ``under``, a part by its whole's place, and whose id matches
    the pattern ``devices``, when given). ``recipe`` names it: the key the card's key numbers (``metrics.csv``
    rows of view ``card`` whose ``level`` is the recipe) attach to. Its
    ``description`` is the title line's sentence, ``{relation}`` slots filled from
    the device's claims (``on {located_in}``); ``channels_cols`` the columns of its
    all-channels grid."""

    cls: str
    under: str | None
    recipe: str
    description: str | None
    channels_cols: float | None
    source: str | None
    devices: str | None = None


def _card_recipe(**columns: Any) -> CardRecipe:
    return CardRecipe(cls=columns.pop("class"), **columns)


@dataclass(frozen=True)
class CardSection:
    """One row of ``card_sections.csv``: a drawn section of the cards of ``recipe``,
    in ``order``. ``kind`` says how it is drawn; ``members`` what it draws (for a
    chain: the ``chain`` of ``chains.csv`` it draws); ``roles`` the roles it shows
    (``;``-separated); ``caption`` the line under it."""

    recipe: str
    order: float
    kind: str
    title: str
    focus: str | None
    members: str | None
    window_m: float | None
    roles: str | None
    caption: str | None
    source: str | None


@dataclass(frozen=True)
class ChainStep:
    """One row of ``chains.csv``: hop ``step`` of the chain ``chain`` of ``recipe``.

    It starts at the thing step ``from_step`` reached (``0``: the device; empty: the
    step before), walks ``relation`` from its subject (``out``) or to it (``in``),
    and keeps the things ``to`` takes: ``device``, ``place``, a class, or
    ``class:pattern`` on the thing's id (empty: any). Two distinct things after that filter make the hop
    contested (a claim several layers state is one thing), unless ``many`` is
    ``yes`` or ``1``: then every thing is a block (empty or ``0``: one expected). ``port_from`` labels the arrow: ``object_port``, ``subject_port`` or
    a word of its own. ``roles`` are shown in the block (``;``-separated)."""

    recipe: str
    chain: str
    step: float
    from_step: float | None
    relation: str
    direction: str
    to: str | None
    many: str | None
    port_from: str | None
    title: str
    roles: str | None
    source: str | None


@dataclass(frozen=True)
class StripSymbol:
    """One row of ``symbols.csv``: every strip (a card's and a place's) draws the
    devices of ``cls`` (whose id matches the pattern ``devices``, when given) as
    ``symbol``, one of the drawing's symbol kinds, in ``lane``; the first row that
    takes a device wins. A class no row names is not drawn on a strip."""

    cls: str
    devices: str | None
    symbol: str
    lane: str
    source: str | None


def _check_symbol(row: StripSymbol, line: int) -> None:
    """A ``symbols.csv`` row names a symbol the drawing knows."""
    from osprey.facility.panels.symbols import KINDS  # the drawing's, read at run time

    if row.symbol not in KINDS:
        raise PanelTableError(
            STRIP_SYMBOLS, line, f"symbol is {row.symbol!r}, not one of {', '.join(KINDS)}"
        )


def _strip_symbol(**columns: Any) -> StripSymbol:
    return StripSymbol(cls=columns.pop("class"), **columns)


#: The conditions a ``cards.csv`` row's ``when`` names, joined by ``;``: the device
#: has no channel a card shows (:meth:`~osprey.facility.panels.model.PanelGraph.card_channels`);
#: the device has no drawing (no card recipe, no picture).
CARD_CONDITIONS: tuple[str, ...] = ("no-channel", "no-drawing")


@dataclass(frozen=True)
class CardRule:
    """One row of ``cards.csv``, an exception to the card default (a device gets a
    card when it has a channel or a drawing): ``keep`` gives the selected devices a
    card, ``drop`` takes it. A row selects the devices of ``cls`` (``*``: of any
    class) under the place ``under`` (a part by its whole's place), whose id
    matches ``devices``, that carry the
    attribute ``attribute`` (a name and a value) and meet every condition of
    ``when``; every selector a row states must hold, and the first row that
    selects a device decides. A dropped device is drawn inside its owner, the thing
    ``owner_relation`` names in ``owner_direction`` that ``owner_to`` takes (as a
    chain hop's ``to``); with no owner it has no screen and no index entry, and a
    related card names it without a link."""

    action: str
    cls: str
    under: str | None
    devices: str | None
    attribute: tuple[str, str] | None
    when: tuple[str, ...] | None
    owner_relation: str | None
    owner_direction: str | None
    owner_to: str | None
    reason: str | None
    source: str | None


def _conditions(text: str) -> tuple[str, ...] | None:
    """``no-channel``, ``no-drawing`` or both joined by ``;``; ``None`` when not."""
    words = tuple(word.strip() for word in text.split(";"))
    if not all(word in CARD_CONDITIONS for word in words) or len(set(words)) != len(words):
        return None
    return words


def _name_value(text: str) -> tuple[str, str] | None:
    """``name=value`` as a pair (the value may be empty); ``None`` when there is no
    ``=`` or no name."""
    name, sep, value = text.partition("=")
    if not sep or not name.strip():
        return None
    return name.strip(), value.strip()


def _check_card_rule(row: CardRule, line: int) -> None:
    """An owner is named on a ``drop`` row only, by a relation and a direction
    together."""
    if (row.owner_relation is None) != (row.owner_direction is None):
        raise PanelTableError(
            CARDS, line, "owner_relation and owner_direction are given together or not at all"
        )
    if row.owner_to is not None and row.owner_relation is None:
        raise PanelTableError(CARDS, line, "owner_to needs owner_relation")
    if row.action == "keep" and row.owner_relation is not None:
        raise PanelTableError(CARDS, line, "a keep row names no owner: the device has its card")


def _card_rule(**columns: Any) -> CardRule:
    return CardRule(cls=columns.pop("class"), **columns)


@dataclass(frozen=True)
class FrontRow:
    """One row of ``fronts.csv``: a connector, or a numbered range of them, on the
    ``face`` of the devices of ``recipe``, in ``group``, by ``order``.

    ``connector`` names it; ``{n}`` makes it a range, numbered by ``count`` when
    stated, else by the device's served channels that match ``channel``, else by
    the socket things that exist (``far_direction`` ``socket``), else by the ports
    the relation states. ``channel`` is the connector's state channel. What is on it
    is found by ``far_relation`` in ``far_direction`` on the port ``far_port`` (empty:
    any port); a row with neither relation nor direction is a reading, with nothing
    on it; ``label`` is its word
    under the socket. ``roles`` (optional,
    ``;``-separated) are the section roles the row's connectors carry: empty, every
    section role; a role binds to a connector only through a row that lists it.
    A ``{n}`` row with no ``channel`` and with ``roles`` is numbered by the
    device's channels of its first role. ``apart`` (optional) ``no``: only the
    clients on a stated port are drawn, never the group of those on no stated
    port; empty, they are drawn apart by class."""

    recipe: str
    face: str
    group: str
    order: float
    connector: str
    kind: str
    count: float | None
    channel: str | None
    far_relation: str | None
    far_direction: str | None
    far_port: str | None
    label: str | None
    source: str | None
    roles: str | None = None
    apart: str | None = None


@dataclass(frozen=True)
class ShortName:
    """One row of ``names.csv``: the short name a strip shows for a thing.

    ``scheme`` says which stated name it is cut from; the last four columns
    cite that name's source.
    """

    thing: str
    short_name: str
    scheme: str | None
    basis: Basis
    document: str | None
    version: str | None
    state: str | None
    locator: str | None


@dataclass(frozen=True)
class Metric:
    """One row of ``metrics.csv``: a key number of a place view or a roll-up root.

    It applies to the views of ``view`` whose place has ``level`` (any place
    when ``None``), in ``order``; it computes ``measure`` over the channels in
    the place that state ``signal``, on things of class ``cls`` when given;
    ``precision`` (optional) is the decimals its number prints with.
    """

    view: str
    level: str | None
    order: float
    title: str
    measure: str
    signal: str
    cls: str | None
    unit: str | None
    caption: str | None
    source: str | None
    precision: int | None = None


def _metric(**columns: Any) -> Metric:
    return Metric(cls=columns.pop("class"), **columns)


@dataclass(frozen=True)
class CalibrationKind:
    """One row of ``calibration_kinds.csv``: a quantity the devices of ``cls`` have.

    The key is ``kind`` and ``cls``: one kind id may name a kind of several
    classes. ``under`` limits the row to the devices of ``cls`` that belong to
    that model or lie under that place; ``None``: every device of ``cls``.
    ``tolerance`` is ``("rel", fraction)`` or ``("abs", amount in unit)``: two
    claims further apart disagree; ``None``: claims are never compared.
    A curve's ``unit`` is ``x;y``. ``held_in`` names the signal of the subject's
    channels that hold the value in force in the control system, or is ``None``
    when the records hold it. ``short`` is the kind's name on a place's chip.
    """

    kind: str
    title: str
    cls: str
    shape: str
    unit: str | None
    wanted_basis: str
    tolerance: tuple[str, float] | None
    tolerance_source: str | None
    held_in: str | None
    order: float | None
    under: str | None = None
    short: str | None = None

    @property
    def tally_title(self) -> str:
        """What a place's chip calls the kind: ``short``, else ``title``."""
        return self.short or self.title


def _calibration_kind(**columns: Any) -> CalibrationKind:
    return CalibrationKind(cls=columns.pop("class"), **columns)


@dataclass(frozen=True)
class Calibration:
    """One row of ``calibrations.csv``: one claim of a value of a kind for a subject.

    ``value`` is the text as stated: a number, numbers separated by ``;`` (a
    tuple), or ``curve:<n> points`` with the points in ``calibration_curves.csv``.
    """

    id: str
    kind: str
    subject: str
    value: str
    unit: str | None
    conditions: str | None
    basis: str
    method: str | None
    date: str | None
    document: str | None
    version: str | None
    state: str | None
    supersedes: str | None
    standing: str


@dataclass(frozen=True)
class CurvePoint:
    """One row of ``calibration_curves.csv``: a point of a claim's curve."""

    id: str
    x: float
    y: float


@dataclass(frozen=True)
class ProbeElement:
    """One row of ``probe_pictures.csv``: an element of a pickup's cross-section,
    looking downstream, shared by the devices of ``cls`` (under ``under``, as in
    :class:`CalibrationKind`).

    By ``shape``: ``bore``, a circle of ``radius_mm`` (an ellipse of full height
    ``size_mm`` when given); ``anchor``, a button of diameter ``size_mm`` at
    ``angle_deg`` (0 is +x, counter-clockwise) on ``radius_mm``, lettered by
    ``label``, its ``signal`` a bar from ``min`` to ``max``; ``marker``, the beam
    at ``signal`` (x) and ``signal_y`` (y) in a box of ``range_mm`` about the
    centre; ``plot``, a curve named ``label`` of the plot named ``element``, of
    ``count`` points; ``readout``, ``signal``'s value named ``label``. The last
    four columns before ``under`` cite the geometry.
    """

    cls: str
    element: str
    shape: str
    label: str | None
    angle_deg: float | None
    radius_mm: float | None
    size_mm: float | None
    signal: str | None
    signal_y: str | None
    min: float | None
    max: float | None
    range_mm: float | None
    count: float | None
    document: str | None
    version: str | None
    state: str | None
    locator: str | None
    under: str | None = None


def _probe_element(**columns: Any) -> ProbeElement:
    return ProbeElement(cls=columns.pop("class"), **columns)


@dataclass(frozen=True)
class PanelTables:
    """The rule tables, each in file order."""

    positions: tuple[Pose, ...]
    record_types: tuple[RecordType, ...]
    health: tuple[HealthRule, ...]
    excluded: tuple[Exclusion, ...]
    pictures: tuple[PicturePart, ...]
    names: tuple[ShortName, ...] = ()
    metrics: tuple[Metric, ...] = ()
    calibration_kinds: tuple[CalibrationKind, ...] = ()
    calibrations: tuple[Calibration, ...] = ()
    calibration_curves: tuple[CurvePoint, ...] = ()
    probe_pictures: tuple[ProbeElement, ...] = ()
    state_labels: tuple[StateLabel, ...] = ()
    units: tuple[ChannelUnit, ...] = ()
    channel_roles: tuple[ChannelRole, ...] = ()
    card_recipes: tuple[CardRecipe, ...] = ()
    card_sections: tuple[CardSection, ...] = ()
    chains: tuple[ChainStep, ...] = ()
    strip_symbols: tuple[StripSymbol, ...] = ()
    fronts: tuple[FrontRow, ...] = ()
    cards: tuple[CardRule, ...] = ()


_ROWS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {
        POSITIONS: Pose,
        RECORD_TYPES: RecordType,
        HEALTH: HealthRule,
        EXCLUDED: Exclusion,
        PICTURES: PicturePart,
        NAMES: ShortName,
        METRICS: _metric,
        CALIBRATION_KINDS: _calibration_kind,
        CALIBRATIONS: Calibration,
        CALIBRATION_CURVES: CurvePoint,
        PROBE_PICTURES: _probe_element,
        STATE_LABELS: StateLabel,
        UNITS: ChannelUnit,
        CHANNEL_ROLES: _channel_role,
        CARD_RECIPES: _card_recipe,
        CARD_SECTIONS: CardSection,
        CHAINS: ChainStep,
        STRIP_SYMBOLS: _strip_symbol,
        FRONTS: FrontRow,
        CARDS: _card_rule,
    }
)


def _records(path: Path) -> Iterator[tuple[int, list[str]]]:
    """Each record of a CSV file with the 1-based line it starts on."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        start = 1
        for record in reader:
            yield start, record
            start = reader.line_num + 1


def _header(name: str, record: list[str]) -> list[str]:
    expected = COLUMNS[name]
    optional = OPTIONAL_COLUMNS.get(name, frozenset())
    missing = [column for column in expected if column not in record and column not in optional]
    extra = [column for column in record if column not in expected]
    if missing or extra or len(set(record)) != len(record):
        parts = []
        if missing:
            parts.append(f"missing column {', '.join(missing)}")
        if extra:
            parts.append(f"extra column {', '.join(extra)}")
        if not parts:
            parts.append("a column is named twice")
        raise PanelTableError(name, 1, f"{'; '.join(parts)}; the header is {','.join(expected)}")
    return record


#: The most decimals a metric's ``precision`` may state.
MOST_DECIMALS = 10


def _decimals(text: str) -> int | None:
    """``0`` to :data:`MOST_DECIMALS` as an int; ``None`` when it is not one."""
    return int(text) if text.isdigit() and int(text) <= MOST_DECIMALS else None


#: The terms a calibration kind's ``tolerance`` is stated in.
TOLERANCE_TERMS: tuple[str, ...] = ("rel", "abs")


def _tolerance_term(text: str) -> tuple[str, float] | None:
    """``rel:<fraction>`` or ``abs:<number>`` as ``(term, amount)``; ``None`` when it is neither."""
    term, _, amount = text.partition(":")
    try:
        number = float(amount)
    except ValueError:
        return None
    if term not in TOLERANCE_TERMS or not number >= 0 or number == float("inf"):
        return None
    return term, number


#: Per file, the columns read by a parser of their own: it gives ``None`` for text
#: that does not hold, and the message says what does.
_PARSED: Mapping[str, Mapping[str, tuple[Callable[[str], Any], str]]] = MappingProxyType(
    {
        CALIBRATION_KINDS: {"tolerance": (_tolerance_term, "rel:<fraction> or abs:<number>")},
        METRICS: {"precision": (_decimals, f"a whole number of decimals 0 to {MOST_DECIMALS}")},
        CARDS: {
            "when": (_conditions, f"{' or '.join(CARD_CONDITIONS)}, joined by ;"),
            "attribute": (_name_value, "name=value"),
        },
    }
)


def _field(name: str, line: int, column: str, text: str) -> Any:
    if not text:
        if column in _REQUIRED[name]:
            raise PanelTableError(name, line, f"{column} is empty")
        return None
    parser = _PARSED.get(name, {}).get(column)
    if parser is not None:
        parse, form = parser
        found = parse(text)
        if found is None:
            raise PanelTableError(name, line, f"{column} is {text!r}, not {form}")
        return found
    words = _VOCABULARY[name].get(column)
    if words is not None and text not in words:
        raise PanelTableError(name, line, f"{column} is {text!r}, not one of {', '.join(words)}")
    if column in _NUMBERS.get(name, ()):
        try:
            return float(text)
        except ValueError:
            raise PanelTableError(name, line, f"{column} is {text!r}, not a number") from None
    return text


def _read(directory: Path, name: str, lines: list[int] | None = None) -> tuple[Any, ...]:
    path = directory / name
    if not path.is_file():
        if name in OPTIONAL_FILES:
            return ()
        raise PanelTableError(name, None, f"the file is missing from {directory}")
    records = _records(path)
    first = next(records, None)
    if first is None:
        raise PanelTableError(
            name, 1, f"the file is empty; the header is {','.join(COLUMNS[name])}"
        )
    header = _header(name, first[1])
    make = _ROWS[name]
    unique = _UNIQUE.get(name)
    seen: dict[tuple[str, ...], int] = {}
    seen_probe: dict[tuple[str | None, ...], int] = {}
    seen_metric: dict[tuple[str | float | None, ...], int] = {}
    rows = []
    for line, record in records:
        if not record:
            continue
        if len(record) != len(header):
            raise PanelTableError(
                name, line, f"the row has {len(record)} fields, the header {len(header)}"
            )
        texts = dict(zip(header, record, strict=True))
        if unique is not None:
            key = tuple(texts.get(column, "") for column in unique)  # an optional one: empty
            if key in seen:
                raise PanelTableError(
                    name,
                    line,
                    f"{', '.join(unique)} {' '.join(key)!r} already has a row at line {seen[key]}",
                )
            seen[key] = line
        rows.append(
            make(**{column: _field(name, line, column, texts[column]) for column in header})
        )
        if name == CALIBRATION_KINDS:
            _check_kind_unit(rows[-1], line)
        if name == CHANNEL_ROLES:
            _check_role(rows[-1], line)
        if name == METRICS:
            _check_metric(rows[-1], line, seen_metric)
        if name == FRONTS:
            _check_front_row(rows[-1], line)
        if name == STRIP_SYMBOLS:
            _check_symbol(rows[-1], line)
        if name == CARDS:
            _check_card_rule(rows[-1], line)
        if name == PROBE_PICTURES:
            _check_probe_row(rows[-1], line, seen_probe)
        if name == PICTURES:
            if "w_mm" in header:
                _check_picture_row(rows[-1], line)
            else:
                rows[-1] = dataclasses.replace(rows[-1], as_point=True)
        if lines is not None:
            lines.append(line)
    return tuple(rows)


def _check_metric(
    metric: Metric, line: int, seen: dict[tuple[str | float | None, ...], int]
) -> None:
    """A card's key number names its class and shows a value or a word; a place's or
    a roll-up's is one per view, level and order, whatever its class."""
    if metric.view == "card":
        if metric.cls is None:
            raise PanelTableError(METRICS, line, "a card's key number names its class")
        if metric.measure not in CARD_MEASURES:
            raise PanelTableError(
                METRICS,
                line,
                f"a card's key number is one of {', '.join(CARD_MEASURES)}, not {metric.measure!r}",
            )
        return
    if metric.measure == "word":
        raise PanelTableError(METRICS, line, "only a card's key number is a word")
    key = (metric.view, metric.level, metric.order)
    if key in seen:
        raise PanelTableError(
            METRICS, line, f"view, level, order {key!r} already has a row at line {seen[key]}"
        )
    seen[key] = line


def _check_front_row(row: FrontRow, line: int) -> None:
    """A front row names its far relation and its direction together, or neither
    (a reading: nothing is on it)."""
    if (row.far_relation is None) != (row.far_direction is None):
        raise PanelTableError(
            FRONTS, line, "far_relation and far_direction are both given or both empty"
        )


def _check_role(role: ChannelRole, line: int) -> None:
    """A press names its word; a relation's port fills a pattern only where a
    relation is walked; a divisor is not 0."""
    if role.shown_as in ("choice", "action") and not role.label:
        raise PanelTableError(CHANNEL_ROLES, line, f"a {role.shown_as} names its press in label")
    if role.scope in ("device", "parts") and ("{port}" in role.signal or "{n}" in role.signal):
        raise PanelTableError(
            CHANNEL_ROLES,
            line,
            "{port} and {n} come from a walked relation, not scope " + role.scope,
        )
    if role.scale is not None and role.scale == 0:
        raise PanelTableError(CHANNEL_ROLES, line, "scale is 0; it divides the raw value")


def _check_kind_unit(kind: CalibrationKind, line: int) -> None:
    """A curve's unit is ``x;y``; only a curve's unit has a ``;``."""
    if kind.unit is None:
        return
    parts = kind.unit.split(";")
    if kind.shape == "curve" and (len(parts) != 2 or not all(p.strip() for p in parts)):
        raise PanelTableError(
            CALIBRATION_KINDS, line, f"unit is {kind.unit!r}; a curve's unit is x;y, as A;T"
        )
    if kind.shape != "curve" and len(parts) > 1:
        raise PanelTableError(
            CALIBRATION_KINDS, line, f"unit is {kind.unit!r}; only a curve's unit has a ';'"
        )


def _check_probe_row(row: ProbeElement, line: int, seen: dict[tuple[str | None, ...], int]) -> None:
    """A probe row fills the columns its shape needs; one row per element of a class
    under one ``under`` (per curve, for a plot); one bore and one marker each."""
    for column in PROBE_NEEDS[row.shape]:
        if getattr(row, column) is None:
            raise PanelTableError(PROBE_PICTURES, line, f"a {row.shape} needs {column}")
    keys = [
        (row.cls, row.under, "element", row.element, row.label if row.shape == "plot" else None)
    ]
    if row.shape in ("bore", "marker"):
        keys.append((row.cls, row.under, "shape", row.shape, None))
    for key in keys:
        if key in seen:
            what = f"one {row.shape}" if key[2] == "shape" else f"element {row.element!r}"
            under = f" under {row.under}" if row.under else ""
            raise PanelTableError(
                PROBE_PICTURES,
                line,
                f"class {row.cls}{under} already has {what} at line {seen[key]}",
            )
        seen[key] = line


def _check_picture_row(row: PicturePart, line: int) -> None:
    """A size a pictures row states is above 0; a row may state none."""
    for column in ("w_mm", "h_mm"):
        value = getattr(row, column)
        if value is not None and not value > 0:
            raise PanelTableError(PICTURES, line, f"{column} is {value:g}, not above 0")


def _check_calibrations(
    kinds: tuple[CalibrationKind, ...],
    claims: tuple[Calibration, ...],
    claim_lines: list[int],
    points: tuple[CurvePoint, ...],
    point_lines: list[int],
) -> None:
    """Hold the calibration files to each other: a claim's kind is stated, its
    ``supersedes`` is a claim, and a curve point's ``id`` is a claim of a curve."""
    known = {kind.kind for kind in kinds}
    ids = {claim.id for claim in claims}
    for claim, line in zip(claims, claim_lines, strict=True):
        if claim.kind not in known:
            raise PanelTableError(CALIBRATIONS, line, f"kind {claim.kind!r} is no stated kind")
        if claim.supersedes is not None and claim.supersedes not in ids:
            raise PanelTableError(
                CALIBRATIONS, line, f"supersedes {claim.supersedes!r}, which is no claim"
            )
    curves = {claim.id for claim in claims if claim.value.startswith(CURVE_PREFIX)}
    for point, line in zip(points, point_lines, strict=True):
        if point.id not in curves:
            raise PanelTableError(
                CALIBRATION_CURVES, line, f"id {point.id!r} is no claim whose value is a curve"
            )


#: The start of a claim's value that holds its points in ``calibration_curves.csv``.
CURVE_PREFIX = "curve:"


def read_tables(panels_dir: Path) -> PanelTables:
    """Read and check the rule tables of a panels directory.

    Args:
        panels_dir: The directory holding the CSV files.

    Returns:
        The tables.

    Raises:
        PanelTableError: A required file is missing, or a header or row breaks
            the schema; the message names the file and line.
    """
    directory = Path(panels_dir)
    return PanelTables(
        positions=_read(directory, POSITIONS),
        record_types=_read(directory, RECORD_TYPES),
        health=_read(directory, HEALTH),
        excluded=_read(directory, EXCLUDED),
        pictures=_read(directory, PICTURES),
        names=_read(directory, NAMES),
        metrics=_read(directory, METRICS),
        **_read_calibrations(directory),
        probe_pictures=_read(directory, PROBE_PICTURES),
        state_labels=_read(directory, STATE_LABELS),
        units=_read(directory, UNITS),
        channel_roles=_read(directory, CHANNEL_ROLES),
        card_recipes=_read(directory, CARD_RECIPES),
        card_sections=_read(directory, CARD_SECTIONS),
        chains=_read(directory, CHAINS),
        strip_symbols=_read(directory, STRIP_SYMBOLS),
        fronts=_read(directory, FRONTS),
        cards=_read(directory, CARDS),
    )


def _read_calibrations(directory: Path) -> dict[str, tuple[Any, ...]]:
    claim_lines: list[int] = []
    point_lines: list[int] = []
    kinds = _read(directory, CALIBRATION_KINDS)
    claims = _read(directory, CALIBRATIONS, claim_lines)
    points = _read(directory, CALIBRATION_CURVES, point_lines)
    _check_calibrations(kinds, claims, claim_lines, points, point_lines)
    return {"calibration_kinds": kinds, "calibrations": claims, "calibration_curves": points}
