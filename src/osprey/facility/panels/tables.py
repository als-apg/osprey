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
- ``pictures.csv``: the parts drawn on a device class's picture.

``pictures.csv`` and ``excluded.csv`` may be absent, which reads as empty; the
other four must be there. A header must name exactly the file's columns (in
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
    "OPTIONAL_FILES",
    "PICTURE_KINDS",
    "PICTURE_VIEWS",
    "Exclusion",
    "HealthRule",
    "Pair",
    "PanelTableError",
    "PanelTables",
    "PicturePart",
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
    }
)

#: The files a panels directory may leave out; an absent one reads as empty.
OPTIONAL_FILES = frozenset({PICTURES, EXCLUDED})

Basis = Literal["stated", "inferred"]
DrawnAs = Literal["value", "setpoint", "state", "command", "text", "waveform", "untyped"]
HealthState = Literal["normal", "warn", "bad"]
ExcludedKind = Literal["thing", "channel", "document"]

BASES: tuple[str, ...] = ("stated", "inferred")
DRAWN_AS: tuple[str, ...] = ("value", "setpoint", "state", "command", "text", "waveform", "untyped")
HEALTH_STATES: tuple[str, ...] = ("normal", "warn", "bad")
EXCLUDED_KINDS: tuple[str, ...] = ("thing", "channel", "document")
PICTURE_VIEWS: tuple[str, ...] = ("end", "side")
PICTURE_KINDS: tuple[str, ...] = (
    "pole",
    "coil",
    "thermal_switch",
    "thermocouple",
    "manifold",
    "terminal",
)

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
    }
)
_VOCABULARY: Mapping[str, Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        POSITIONS: {"basis": BASES},
        RECORD_TYPES: {"drawn_as": DRAWN_AS, "basis": BASES},
        PAIRS: {"basis": BASES},
        HEALTH: {"state": HEALTH_STATES, "basis": BASES},
        EXCLUDED: {"kind": EXCLUDED_KINDS},
        PICTURES: {"view": PICTURE_VIEWS, "kind": PICTURE_KINDS},
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
    }
)
_UNIQUE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {POSITIONS: ("thing", "frame"), RECORD_TYPES: ("channel",), PAIRS: ("setpoint",)}
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
class PanelTables:
    """The six rule tables, each in file order."""

    positions: tuple[Pose, ...]
    record_types: tuple[RecordType, ...]
    pairs: tuple[Pair, ...]
    health: tuple[HealthRule, ...]
    excluded: tuple[Exclusion, ...]
    pictures: tuple[PicturePart, ...]


_ROWS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {
        POSITIONS: Pose,
        RECORD_TYPES: RecordType,
        PAIRS: Pair,
        HEALTH: HealthRule,
        EXCLUDED: Exclusion,
        PICTURES: PicturePart,
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
    missing = [column for column in expected if column not in record]
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


def _field(name: str, line: int, column: str, text: str) -> Any:
    if not text:
        if column in _REQUIRED[name]:
            raise PanelTableError(name, line, f"{column} is empty")
        return None
    words = _VOCABULARY[name].get(column)
    if words is not None and text not in words:
        raise PanelTableError(name, line, f"{column} is {text!r}, not one of {', '.join(words)}")
    if column in _NUMBERS.get(name, ()):
        try:
            return float(text)
        except ValueError:
            raise PanelTableError(name, line, f"{column} is {text!r}, not a number") from None
    return text


def _read(directory: Path, name: str) -> tuple[Any, ...]:
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
    return tuple(rows)


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
    )
