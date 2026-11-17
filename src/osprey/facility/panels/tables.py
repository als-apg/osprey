"""The rule tables a facility repository writes beside its facility file.

A panels directory holds six CSV files; :data:`COLUMNS` is each one's header
line, the schema this module holds them to:

- ``positions.csv``: where a thing sits, one row per thing and frame;
- ``record_types.csv``: each channel's record type and how a screen draws it,
  one row per channel;
- ``pairs.csv``: the readback a setpoint is shown beside, one row per setpoint;
- ``health.csv``: the state a channel's value means, one row per stated value
  and source, so two sources may state two states for one value;
- ``excluded.csv``: things, channels and documents no screen shows;
- ``pictures.csv``: the parts drawn on a device class's picture;
- ``names.csv``: a thing's short name; ``metrics.csv``: the key numbers;
- ``calibration_kinds.csv``: the calibration quantities a device class has;
- ``calibrations.csv``: the calibration claims, one value of one kind for one
  subject each, with basis, date, source and standing;
- ``calibration_curves.csv``: the points of a claim whose value is a curve.

``positions.csv``, ``record_types.csv``, ``pairs.csv`` and ``health.csv``
must be there; every other file may be absent, which reads as empty. Across
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
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
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
    "OPTIONAL_FILES",
    "TOLERANCE_TERMS",
    "PICTURE_VIEWS",
    "PROBE_NEEDS",
    "PROBE_SHAPES",
    "ProbeElement",
    "TABLE_FILES",
    "Exclusion",
    "HealthRule",
    "Pair",
    "PanelTableError",
    "PanelTables",
    "PicturePart",
    "ShortName",
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
PAIRS = "pairs.csv"
HEALTH = "health.csv"
EXCLUDED = "excluded.csv"
PICTURES = "pictures.csv"
NAMES = "names.csv"
METRICS = "metrics.csv"
CALIBRATION_KINDS = "calibration_kinds.csv"
CALIBRATIONS = "calibrations.csv"
CALIBRATION_CURVES = "calibration_curves.csv"
PROBE_PICTURES = "probe_pictures.csv"
#: The rule tables, by file name.
TABLE_FILES: tuple[str, ...] = (
    POSITIONS,
    RECORD_TYPES,
    PAIRS,
    HEALTH,
    EXCLUDED,
    PICTURES,
    NAMES,
    METRICS,
    CALIBRATION_KINDS,
    CALIBRATIONS,
    CALIBRATION_CURVES,
    PROBE_PICTURES,
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
        PAIRS: ["setpoint", "readback", "evidence", "basis", "source", "revision", "locator"],
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
    {CALIBRATION_KINDS: frozenset({"under"}), PROBE_PICTURES: frozenset({"under"})}
)

#: The files a panels directory may leave out; an absent one reads as empty.
OPTIONAL_FILES = frozenset(
    {
        PICTURES,
        EXCLUDED,
        NAMES,
        METRICS,
        CALIBRATION_KINDS,
        CALIBRATIONS,
        CALIBRATION_CURVES,
        PROBE_PICTURES,
    }
)

Basis = Literal["stated", "inferred"]
DrawnAs = Literal["value", "setpoint", "state", "command", "text", "waveform", "untyped"]
HealthState = Literal["normal", "warn", "bad"]
ExcludedKind = Literal["thing", "channel", "document"]

BASES: tuple[str, ...] = ("stated", "inferred")
DRAWN_AS: tuple[str, ...] = ("value", "setpoint", "state", "command", "text", "waveform", "untyped")
HEALTH_STATES: tuple[str, ...] = ("normal", "warn", "bad")
EXCLUDED_KINDS: tuple[str, ...] = ("thing", "channel", "document")
PICTURE_VIEWS: tuple[str, ...] = ("end", "side")
METRIC_VIEWS: tuple[str, ...] = ("place", "rollup")
#: What a key number computes over the channels its row selects.
MEASURES: tuple[str, ...] = (
    "value",
    "count_true",
    "count_false",
    "count_in_tolerance",
    "rms",
    "max",
    "min",
)

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
#: A claim's standing, as relation claims have it.
STANDINGS: tuple[str, ...] = ("accepted", "contested", "superseded")

# Per file: the columns that must be non-empty, the columns with a vocabulary,
# the number columns (required or not by the first set) and the unique key.
_REQUIRED: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        POSITIONS: ("thing", "frame", "basis"),
        RECORD_TYPES: ("channel", "record_type", "drawn_as", "basis"),
        PAIRS: ("setpoint", "readback", "basis"),
        HEALTH: ("channel", "value", "state", "basis"),
        EXCLUDED: ("kind", "id"),
        PICTURES: ("device_class", "view", "part", "kind", "x_mm", "y_mm", "z_mm"),
        NAMES: ("thing", "short_name", "basis"),
        METRICS: ("view", "order", "title", "measure", "signal"),
        CALIBRATION_KINDS: ("kind", "title", "class", "shape", "wanted_basis"),
        CALIBRATIONS: ("id", "kind", "subject", "value", "basis", "standing"),
        CALIBRATION_CURVES: ("id", "x", "y"),
        PROBE_PICTURES: ("class", "element", "shape"),
    }
)
_VOCABULARY: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        POSITIONS: {"basis": BASES},
        RECORD_TYPES: {"drawn_as": DRAWN_AS, "basis": BASES},
        PAIRS: {"basis": BASES},
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
        PICTURES: ("x_mm", "y_mm", "z_mm"),
        METRICS: ("order",),
        CALIBRATION_KINDS: ("order",),
        CALIBRATION_CURVES: ("x", "y"),
        PROBE_PICTURES: ("angle_deg", "radius_mm", "size_mm", "min", "max", "range_mm", "count"),
    }
)
_UNIQUE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        POSITIONS: ("thing", "frame"),
        RECORD_TYPES: ("channel",),
        PAIRS: ("setpoint",),
        NAMES: ("thing",),
        METRICS: ("view", "level", "order"),
        CALIBRATION_KINDS: ("kind", "class"),
        CALIBRATIONS: ("id",),
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
    ``in_ring_frame`` is set by the graph, which knows the ring frame.
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
    in_ring_frame: bool = False

    @property
    def is_ring_position(self) -> bool:
        """Whether ``s_m`` is a true ring position: stated, in the ring frame, nothing assumed."""
        return self.in_ring_frame and self.s_m is not None and not self.derived_assumes


@dataclass(frozen=True)
class RecordType:
    """One row of ``record_types.csv``."""

    channel: str
    record_type: str
    drawn_as: DrawnAs
    basis: Basis
    source: str | None
    revision: str | None
    locator: str | None


@dataclass(frozen=True)
class Pair:
    """The readback a setpoint is shown beside, and where that is stated."""

    setpoint: str
    readback: str
    evidence: str | None
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
    """One row of ``pictures.csv``: a part drawn on a device class's picture."""

    device_class: str
    view: str
    part: str
    kind: str
    x_mm: float
    y_mm: float
    z_mm: float
    document: str | None
    version: str | None
    state: str | None
    locator: str | None
    #: The channel the part reads, for a light on the picture. A hook: the
    #: table has no column for it yet, so every row read today gives ``None``.
    channel: str | None = None


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
    the place that state ``signal``, on things of class ``cls`` when given.
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
    when the records hold it.
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
    pairs: tuple[Pair, ...]
    health: tuple[HealthRule, ...]
    excluded: tuple[Exclusion, ...]
    pictures: tuple[PicturePart, ...]
    names: tuple[ShortName, ...] = ()
    metrics: tuple[Metric, ...] = ()
    calibration_kinds: tuple[CalibrationKind, ...] = ()
    calibrations: tuple[Calibration, ...] = ()
    calibration_curves: tuple[CurvePoint, ...] = ()
    probe_pictures: tuple[ProbeElement, ...] = ()


_ROWS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {
        POSITIONS: Pose,
        RECORD_TYPES: RecordType,
        PAIRS: Pair,
        HEALTH: HealthRule,
        EXCLUDED: Exclusion,
        PICTURES: PicturePart,
        NAMES: ShortName,
        METRICS: _metric,
        CALIBRATION_KINDS: _calibration_kind,
        CALIBRATIONS: Calibration,
        CALIBRATION_CURVES: CurvePoint,
        PROBE_PICTURES: _probe_element,
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
    {CALIBRATION_KINDS: {"tolerance": (_tolerance_term, "rel:<fraction> or abs:<number>")}}
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
            key = tuple(texts[column] for column in unique)
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
        if name == PROBE_PICTURES:
            _check_probe_row(rows[-1], line, seen_probe)
        if lines is not None:
            lines.append(line)
    return tuple(rows)


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
        pairs=_read(directory, PAIRS),
        health=_read(directory, HEALTH),
        excluded=_read(directory, EXCLUDED),
        pictures=_read(directory, PICTURES),
        names=_read(directory, NAMES),
        metrics=_read(directory, METRICS),
        **_read_calibrations(directory),
        probe_pictures=_read(directory, PROBE_PICTURES),
    )


def _read_calibrations(directory: Path) -> dict[str, tuple[Any, ...]]:
    claim_lines: list[int] = []
    point_lines: list[int] = []
    kinds = _read(directory, CALIBRATION_KINDS)
    claims = _read(directory, CALIBRATIONS, claim_lines)
    points = _read(directory, CALIBRATION_CURVES, point_lines)
    _check_calibrations(kinds, claims, claim_lines, points, point_lines)
    return {"calibration_kinds": kinds, "calibrations": claims, "calibration_curves": points}
