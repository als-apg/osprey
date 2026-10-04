"""The response check: each kept response export held against its model.

An mml import keeps a model's exported orbit-response matrix as
``imported/mml/<model>.response.json``. :func:`check_responses` re-measures
every block of each kept export on the model the facility file describes --
the deck the model names, driven through its wiring records -- and returns one
:class:`ModelCheck` per model, whose :attr:`~ModelCheck.line` is what
``osprey facility validate`` prints. Nothing is written.

**The comparison.** Both sides are in the physics units the export states. A
corrector is swept by the hardware width the export states for it, half either
side of the record's ``default``, and the response is the change of each
monitor's reading over the physics distance between the two arms. Each entry
is then held to a band that is a share of its own size, with a floor that is
a share of the whole export's scale::

    |model - file| <= 0.05 * max(|file|, 0.1 * rms(every compared entry))

**Which blocks are judged.** A block pairs a monitor family with a corrector
family. The mapping's wiring block states the engine words of each family, and
the plane follows from them: a monitor reads its ``axis``, a kick drives the
plane of its ``index``. A block whose two planes are the same is judged; any
other block is compared and judged by nothing. An unjudged block's sign
agreement is not counted: its model entries there are the deck's coupling, at
the noise level, so whether their sign matches the export's says nothing about
the model.

**Rows are matched by device.** A response export states a ``DeviceList`` per
side; each row is matched to the device carrying that ``DeviceList`` and to
that device's wiring record of the family's engine words. A row the export
marks ``Status`` 0, a row no wired device answers to, a column with no finite
width and a column whose sweep leaves the deck without a solve are left out of
the comparison. Each corrector is swept on its own through the engine's
response matrix, rows matched by the readback addresses it returns, and its
hardware entries are carried into the export's physics units through the two
records' calibrations, so one unsolvable corrector leaves its column out and
the rest are still judged.

**Reversed columns are set aside.** Inside a judged block, a corrector whose
entries above the floor disagree in sign on more than half of its compared
entries is left out of what the block counts.

**The bar is the block's origin.** A block whose export says ``model`` was
computed from a deck, so the model has to reproduce it: at least 0.99 of its
counted entries inside the band. Any other block was measured on a machine:
the median size of model over file, above the floor, sits in 0.8 to 1.25 and
the sign agrees on at least 0.95 of the entries above the floor. Either way a
judged block needs one entry above the floor. A model passes when every judged
block of its export passes, and the line names the block nearest its bar.

The engine is imported inside the function that solves.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import IO, Any

__all__ = [
    "FLOOR_FRACTION",
    "MEASURED_RATIO",
    "MEASURED_SIGN",
    "MODEL_INSIDE_BAND",
    "MODEL_ORIGIN",
    "POLARITY_SHARE",
    "RESPONSE_SUFFIX",
    "TOLERANCE_FRACTION",
    "Block",
    "Entry",
    "Figures",
    "ModelCheck",
    "Verdict",
    "banded",
    "check_responses",
    "compare",
    "figures",
    "judge",
    "report",
]

#: The share of an entry's own scale the model may differ by.
TOLERANCE_FRACTION = 0.05

#: The share of the whole export's rms below which an entry is held to size
#: alone and its sign is not asserted.
FLOOR_FRACTION = 0.1

#: The share of a column's compared entries that has to disagree in sign
#: before the column is set aside as reversed.
POLARITY_SHARE = 0.5

#: The origin word of a block computed from a deck.
MODEL_ORIGIN = "model"

#: The share of a model-derived block's counted entries inside the band.
MODEL_INSIDE_BAND = 0.99

#: The band a measured block's median model-over-file size sits in.
MEASURED_RATIO = (0.8, 1.25)

#: The share of a measured block's entries above the floor agreeing in sign.
MEASURED_SIGN = 0.95

#: File-name suffix of a model's kept response export under the layer.
RESPONSE_SUFFIX = ".response.json"

#: Where the mml layer's sources live, relative to ``data/facility/``.
_LAYER_DIR = "imported/mml"

#: The two transverse planes and the orbit coordinate each one is read at.
_COORDINATE = {"x": 0, "y": 2}

_KICK = "KickAngle"


# -- the figures -------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """One matrix entry, as the export states it and as the model measured it.

    Attributes:
        monitor_address: The channel the row was read on.
        actuator_address: The channel the column was driven through.
        monitor_device: The 1-based row of the entry in the export's block.
        actuator_device: The 1-based column of the entry in the export's block.
        file_value: The exported entry, in the export's physics units.
        model_value: What the model did, in the same units.
        tolerance: The band this entry has to fall inside.
        floor: The floor this entry is weighed against.
    """

    monitor_address: str
    actuator_address: str
    monitor_device: int
    actuator_device: int
    file_value: float
    model_value: float
    tolerance: float = 0.0
    floor: float = 0.0

    @property
    def deviation(self) -> float:
        """How far the model landed from the export."""
        return abs(self.model_value - self.file_value)

    @property
    def passed(self) -> bool:
        """Whether the entry is inside its band."""
        return self.deviation <= self.tolerance

    @property
    def above_floor(self) -> bool:
        """Whether the export states an entry large enough for its sign to count."""
        return abs(self.file_value) > self.floor

    @property
    def sign_agrees(self) -> bool | None:
        """Whether both sides move the beam the same way; ``None`` below the floor."""
        if not self.above_floor:
            return None
        return self.model_value * self.file_value > 0.0

    @property
    def ratio(self) -> float:
        """The model over the export; ``nan`` where the export states zero."""
        if self.file_value == 0.0:
            return math.nan
        return self.model_value / self.file_value


@dataclass(frozen=True)
class Figures:
    """What one set of compared entries says about the model.

    Attributes:
        compared: How many entries were held against the export.
        passed: How many of them are inside their band.
        checked: How many sit above the floor.
        agreed: How many of those agree in sign; ``None`` for an unjudged
            block, whose sign is not asserted.
        median_ratio: The median model-over-file size above the floor;
            ``nan`` when no entry is above it.
    """

    compared: int
    passed: int
    checked: int
    agreed: int | None
    median_ratio: float

    @property
    def pass_ratio(self) -> float:
        """The share inside the band; ``nan`` when nothing was compared."""
        return self.passed / self.compared if self.compared else math.nan

    @property
    def sign_ratio(self) -> float:
        """The share above the floor agreeing in sign.

        ``nan`` when none is above it or when the sign is not counted.
        """
        if self.agreed is None or not self.checked:
            return math.nan
        return self.agreed / self.checked


def figures(entries: Iterable[Entry], *, judged: bool = True) -> Figures:
    """Weigh a set of banded entries.

    Args:
        entries: The entries, each carrying its band and floor.
        judged: Whether the entries belong to a judged block; an unjudged
            block counts no sign agreement.

    Returns:
        The counts and the median size ratio, which is taken above the floor
        alone.
    """
    weighed = tuple(entries)
    checked = [entry for entry in weighed if entry.above_floor]
    return Figures(
        compared=len(weighed),
        passed=sum(1 for entry in weighed if entry.passed),
        checked=len(checked),
        agreed=sum(1 for entry in checked if entry.sign_agrees) if judged else None,
        median_ratio=_median(
            sorted(abs(entry.ratio) for entry in checked if math.isfinite(entry.ratio))
        ),
    )


@dataclass(frozen=True)
class Block:
    """One monitor family against one corrector family.

    Attributes:
        monitor_family: The family the rows were read on, as the export names it.
        actuator_family: The family the columns were driven through.
        origin: Where the export says the block came from.
        entries: Every compared entry.
        judged: Whether the monitors read the plane the correctors drive.
        reversed_columns: The columns of a judged block set aside as reversed,
            by ``actuator_device``.
    """

    monitor_family: str
    actuator_family: str
    origin: str
    entries: tuple[Entry, ...]
    judged: bool
    reversed_columns: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        """The block as a line names it: ``<monitor family>/<corrector family>``."""
        return f"{self.monitor_family}/{self.actuator_family}"

    @property
    def counted(self) -> tuple[Entry, ...]:
        """Every entry outside the reversed columns."""
        if not self.reversed_columns:
            return self.entries
        aside = set(self.reversed_columns)
        return tuple(entry for entry in self.entries if entry.actuator_device not in aside)

    @property
    def counts(self) -> Figures:
        """The figures the block is judged on, reversed columns left out."""
        return figures(self.counted, judged=self.judged)

    @property
    def full(self) -> Figures:
        """The figures of every entry of the block."""
        return figures(self.entries, judged=self.judged)


def banded(block: Block, floor: float) -> Block:
    """Hold every entry of one block to the export's floor.

    Args:
        block: A block whose entries carry no band yet.
        floor: ``FLOOR_FRACTION`` of the rms of every compared entry of the
            export.

    Returns:
        The block with each entry's ``floor`` and ``tolerance`` set and, where
        it is judged, its reversed columns named.
    """
    entries = tuple(
        replace(
            entry,
            floor=floor,
            tolerance=TOLERANCE_FRACTION * max(abs(entry.file_value), floor),
        )
        for entry in block.entries
    )
    return replace(
        block,
        entries=entries,
        reversed_columns=_reversed(entries) if block.judged else (),
    )


def _reversed(entries: Sequence[Entry]) -> tuple[int, ...]:
    """The columns whose entries above the floor mostly disagree in sign.

    The flipped entries are counted against every compared entry of the
    column, so a column has to be mostly sizable as well as mostly flipped.
    """
    columns: dict[int, list[Entry]] = {}
    for entry in entries:
        columns.setdefault(entry.actuator_device, []).append(entry)
    found: list[int] = []
    for device, column in sorted(columns.items()):
        flipped = sum(1 for entry in column if entry.sign_agrees is False)
        if flipped > POLARITY_SHARE * len(column):
            found.append(device)
    return tuple(found)


def _rms(values: Iterable[float]) -> float:
    squares = [value * value for value in values]
    return math.sqrt(sum(squares) / len(squares)) if squares else 0.0


def _median(values: Sequence[float]) -> float:
    """The median of an ordered sequence; ``nan`` when it is empty."""
    if not values:
        return math.nan
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


# -- the verdict -------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One figure of one judged block against its bar.

    Attributes:
        block: The block's name, ``-`` when the export holds no judged block.
        origin: The block's origin word.
        figure: The figure's name and value, as the line prints them.
        bar: The bar the figure is held to, as the line prints it.
        passed: Whether the figure meets the bar.
        margin: How far inside the bar the figure sits, as a share of the
            distance from the bar to the figure's best value; negative
            outside it.
    """

    block: str
    origin: str
    figure: str
    bar: str
    passed: bool
    margin: float


def _share(value: float) -> str:
    return f"{value:.3f}"


def _verdicts(block: Block) -> list[Verdict]:
    """Every figure one judged block is held to, each against its bar."""
    counts = block.counts
    if not counts.checked:
        return [Verdict(block.name, block.origin, "above floor 0", "1", False, -math.inf)]
    if block.origin == MODEL_ORIGIN:
        share = counts.pass_ratio
        return [
            Verdict(
                block.name,
                block.origin,
                f"inside band {_share(share)}",
                f"{MODEL_INSIDE_BAND:g}",
                share >= MODEL_INSIDE_BAND,
                (share - MODEL_INSIDE_BAND) / (1.0 - MODEL_INSIDE_BAND),
            )
        ]
    low, high = MEASURED_RATIO
    ratio = counts.median_ratio
    inside = math.isfinite(ratio) and low <= ratio <= high
    reach = (
        math.log(min(ratio / low, high / ratio)) / math.log(math.sqrt(high / low))
        if math.isfinite(ratio) and ratio > 0.0
        else -math.inf
    )
    sign = counts.sign_ratio
    return [
        Verdict(
            block.name,
            block.origin,
            f"median ratio {_share(ratio)}",
            f"{low:g} to {high:g}",
            inside,
            reach,
        ),
        Verdict(
            block.name,
            block.origin,
            f"sign agreement {_share(sign)}",
            f"{MEASURED_SIGN:g}",
            sign >= MEASURED_SIGN,
            (sign - MEASURED_SIGN) / (1.0 - MEASURED_SIGN),
        ),
    ]


def judge(blocks: Iterable[Block], origin: str = "") -> Verdict:
    """Return the figure of the judged blocks that sits nearest its bar.

    A failing figure comes before a passing one, so the verdict passes only
    when every judged block passes every figure it is held to.

    Args:
        blocks: The banded blocks of one export.
        origin: The origin word to state when no block is judged.

    Returns:
        The worst figure; a passing ``judged blocks 0`` when no block is
        judged, because nothing was held to a bar.
    """
    found = [verdict for block in blocks if block.judged for verdict in _verdicts(block)]
    if not found:
        return Verdict("-", origin, "judged blocks 0", "0", True, math.inf)
    return min(found, key=lambda verdict: (verdict.passed, verdict.margin))


@dataclass(frozen=True)
class ModelCheck:
    """One model's response check.

    Attributes:
        model: The model's name.
        verdict: The figure nearest its bar.
    """

    model: str
    verdict: Verdict

    @property
    def passed(self) -> bool:
        """Whether every judged block of the model's export passes."""
        return self.verdict.passed

    @property
    def line(self) -> str:
        """The one line ``osprey facility validate`` prints for the model."""
        verdict = self.verdict
        word = "pass" if verdict.passed else "fail"
        return (
            f"response check {self.model}: {verdict.origin} {verdict.block} "
            f"{verdict.figure} ({word} at {verdict.bar})"
        )


def report(checks: Iterable[ModelCheck], file: IO[Any] | None = None) -> bool:
    """Print each check's line and say whether every one passed.

    Args:
        checks: The checks, in the order they are printed.
        file: Where the lines go; standard error when omitted.

    Returns:
        ``True`` when no check failed.
    """
    import sys

    stream = sys.stderr if file is None else file
    passed = True
    for check in checks:
        print(check.line, file=stream)
        passed = passed and check.passed
    return passed


# -- the exported document ---------------------------------------------------


def _word(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    """A finite number, or ``None`` for anything else."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _first(value: Any) -> Any:
    """A one-entry list unwrapped, the way an exported column vector writes a scalar."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _side(block: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    body = block.get(side)
    return body if isinstance(body, Mapping) else {}


def _list(body: Mapping[str, Any], key: str) -> list[Any]:
    value = body.get(key)
    return list(value) if isinstance(value, list) else []


def _row_key(row: Any) -> tuple[int, ...] | None:
    """One ``DeviceList`` row as the key both sides are matched on."""
    values = row if isinstance(row, list | tuple) else [row]
    key: list[int] = []
    for value in values:
        number = _number(value)
        if number is None or number != int(number):
            return None
        key.append(int(number))
    return tuple(key) if key else None


def _widths(block: Mapping[str, Any]) -> list[float | None]:
    """The hardware sweep width of each corrector row; ``None`` where none is stated.

    A block stating one number sweeps every row by it; a list of another
    length than the corrector rows states no width for any of them.
    """
    rows = len(_list(_side(block, "actuator"), "device_list"))
    stated = block.get("actuator_delta")
    values = list(stated) if isinstance(stated, list) else [stated] * rows
    if len(values) != rows:
        return [None] * rows
    widths = [_number(_first(value)) for value in values]
    return [width if width else None for width in widths]


def _kept_rows(block: Mapping[str, Any], side: str, records: Mapping[tuple[int, ...], Any]) -> dict:
    """The rows of one side that reach a wiring record, keyed by position in the export."""
    body = _side(block, side)
    statuses = _list(body, "status")
    kept: dict[int, Any] = {}
    for index, row in enumerate(_list(body, "device_list")):
        status = _number(_first(statuses[index])) if index < len(statuses) else None
        if status is not None and status == 0.0:
            continue
        key = _row_key(row)
        record = records.get(key) if key is not None else None
        if record is not None:
            kept[index] = record
    return kept


def _matrix(data: Any, row: int, column: int) -> float | None:
    if not isinstance(data, list) or row >= len(data):
        return None
    line = data[row]
    if not isinstance(line, list) or column >= len(line):
        return None
    return _number(line[column])


def _well_shaped(block: Mapping[str, Any]) -> bool:
    """Whether the matrix is one row per monitor row and one column per corrector row."""
    monitors = len(_list(_side(block, "monitor"), "device_list"))
    actuators = len(_list(_side(block, "actuator"), "device_list"))
    data = block.get("data")
    rows = data if isinstance(data, list) else []
    widths = {len(row) if isinstance(row, list) else 0 for row in rows}
    return bool(rows) and bool(monitors) and len(rows) == monitors and widths <= {actuators}


# -- the facility file -------------------------------------------------------


def _family_records(
    document: Mapping[str, Any],
    model: Mapping[str, Any],
    group_id: str,
    engine: Any,
    direction: str,
) -> dict[tuple[int, ...], Mapping[str, Any]]:
    """One family's wiring records in a model, keyed by their device's ``DeviceList``.

    A record belongs to the family when its address sits on one device of the
    family's group and its engine block is the one the mapping wires the
    family through. A device carrying no such record, or several, and a
    ``DeviceList`` several wired devices state, binds nothing.
    """
    members: set[str] = set()
    for group in document.get("groups", []):
        if group.get("id") == group_id:
            members = set(group.get("members", []))
    rows = {
        device["id"]: key
        for device in document.get("devices", [])
        if device["id"] in members
        and (key := _row_key((device.get("attributes") or {}).get("DeviceList"))) is not None
    }
    on_device = {
        channel["id"]: (channel.get("on") or {}).get("device")
        for channel in document.get("channels", [])
    }
    words = {
        key: getattr(engine, key)
        for key in ("attribute", "index", "axis")
        if getattr(engine, key) is not None
    }
    found: dict[tuple[int, ...], list[Mapping[str, Any]]] = {}
    for record in model.get("wiring", []):
        device = on_device.get(record.get("address"))
        if device not in rows or record.get("direction") != direction:
            continue
        if dict(record.get("engine") or {}) != words:
            continue
        found.setdefault(rows[device], []).append(record)
    return {key: records[0] for key, records in found.items() if len(records) == 1}


def _plane(engine: Any) -> str | None:
    """The transverse plane a family's engine words work in."""
    if engine is None:
        return None
    if engine.axis in _COORDINATE:
        return str(engine.axis)
    if engine.attribute == _KICK:
        return {0: "x", 1: "y"}.get(engine.index)
    return None


# -- the model ---------------------------------------------------------------


def _physics_span(record: Mapping[str, Any], held: float, width: float) -> float:
    """The physics distance between ``held + width/2`` and ``held - width/2``."""
    from osprey.simulation.engines.calibration import curve_from_record, to_physics

    curve = curve_from_record((record.get("calibration") or {}).get("curve"))
    return to_physics(curve, held + 0.5 * width) - to_physics(curve, held - 0.5 * width)


def _monitor_gain(record: Mapping[str, Any]) -> float:
    """A monitor's physics reading per hardware unit, where a centred beam reads.

    Exact for a linear calibration, whose slope is the same everywhere.
    """
    from osprey.simulation.engines.calibration import curve_from_record, to_hardware

    calibration = record.get("calibration") or {}
    curve = curve_from_record(calibration.get("curve"))
    if curve is None:
        return 1.0
    centred = to_hardware(curve, curve_from_record(calibration.get("inverse")), 0.0)
    step = 1.0e-6 * max(1.0, abs(centred))
    return _physics_span(record, centred, step) / step


def _response(
    deck: Path,
    wiring: Sequence[Mapping[str, Any]],
    settings: Any,
    address: str,
    width: float,
    span: float,
) -> dict[str, float] | None:
    """One corrector's response on every wired monitor, by monitor address.

    The engine's response matrix states hardware readback per hardware
    setpoint; each entry is carried into physics reading per physics unit of
    the corrector through the monitor's calibration and the corrector's
    physics ``span`` over its hardware ``width``.

    Returns:
        The response, or ``None`` when the sweep leaves the deck without a
        solve.
    """
    from lume_pyat.exceptions import OrbitSolveError

    from osprey.simulation.engines.pyat import response_matrix

    try:
        rows, matrix = response_matrix(deck, wiring, settings, [address], {address: width})
    except OrbitSolveError:
        return None
    per_hardware = width / span
    monitors = {str(record["address"]): record for record in wiring}
    return {
        row: float(matrix[index, 0]) * _monitor_gain(monitors[row]) * per_hardware
        for index, row in enumerate(rows)
    }


# -- the comparison ----------------------------------------------------------


def compare(
    facility_dir: Path, document: Mapping[str, Any], model: Mapping[str, Any]
) -> tuple[Block, ...]:
    """Hold every block of one model's kept response export against the model.

    Args:
        facility_dir: The ``data/facility`` directory.
        document: The facility file the build made of it.
        model: The model's record in ``document``, carrying its ``deck`` and
            filled ``wiring``.

    Returns:
        The banded blocks, in the order the export writes them; empty when the
        export states none.
    """
    from osprey.facility.layers.mml.mapping import MAPPING_FILE, read_mapping

    name = model["name"]
    layer = facility_dir / _LAYER_DIR
    response = json.loads((layer / f"{name}{RESPONSE_SUFFIX}").read_text(encoding="utf-8"))
    stated = response.get("blocks") if isinstance(response, dict) else None
    blocks = [block for block in stated or [] if isinstance(block, dict)]

    mapping = read_mapping(facility_dir / MAPPING_FILE)
    wiring = next(
        (entry.wiring for entry in mapping.models.values() if entry.name == name),
        {},
    )

    def records(family: str, direction: str) -> dict[tuple[int, ...], Mapping[str, Any]]:
        wired = wiring.get(family)
        if wired is None or wired.engine is None or family not in mapping.families:
            return {}
        return _family_records(document, model, mapping.mapped(family), wired.engine, direction)

    def plane(family: str) -> str | None:
        wired = wiring.get(family)
        return None if wired is None else _plane(wired.engine)

    swept: dict[tuple[str, float], dict[str, float] | None] = {}

    def sweep(address: str, width: float, span: float) -> dict[str, float] | None:
        key = (address, width)
        if key not in swept:
            swept[key] = _response(
                facility_dir / model["deck"],
                list(model.get("wiring", [])),
                model.get("settings"),
                address,
                width,
                span,
            )
        return swept[key]

    drafts: list[Block] = []
    for block in blocks:
        monitor_family = _word(_side(block, "monitor").get("family"))
        actuator_family = _word(_side(block, "actuator").get("family"))
        monitor_plane, actuator_plane = plane(monitor_family), plane(actuator_family)
        entries: list[Entry] = []
        rows = _kept_rows(block, "monitor", records(monitor_family, "read"))
        columns = _kept_rows(block, "actuator", records(actuator_family, "write"))
        if rows and columns and "deck" in model and _well_shaped(block):
            widths = _widths(block)
            for column, actuator in columns.items():
                width = widths[column] if column < len(widths) else None
                # A filled setpoint's default is the deck's own value, the one
                # response_matrix steps about.
                held = float(actuator.get("default") or 0.0)
                span = math.nan if width is None else _physics_span(actuator, held, width)
                if width is None or not math.isfinite(span) or span == 0.0:
                    continue
                measured = sweep(str(actuator["address"]), width, span)
                if measured is None:
                    continue
                for row, monitor in rows.items():
                    file_value = _matrix(block.get("data"), row, column)
                    model_value = measured.get(monitor["address"])
                    if file_value is None or model_value is None:
                        continue
                    entries.append(
                        Entry(
                            monitor_address=monitor["address"],
                            actuator_address=actuator["address"],
                            monitor_device=row + 1,
                            actuator_device=column + 1,
                            file_value=file_value,
                            model_value=model_value,
                        )
                    )
        drafts.append(
            Block(
                monitor_family=monitor_family,
                actuator_family=actuator_family,
                origin=_word(block.get("origin")) or "unstated",
                entries=tuple(entries),
                judged=monitor_plane is not None and monitor_plane == actuator_plane,
            )
        )

    floor = FLOOR_FRACTION * _rms(entry.file_value for draft in drafts for entry in draft.entries)
    return tuple(banded(draft, floor) for draft in drafts)


def check_responses(facility_dir: Path, document: Mapping[str, Any]) -> list[ModelCheck]:
    """Check every model of the facility file that has a kept response export.

    Args:
        facility_dir: The ``data/facility`` directory.
        document: The facility file the build made of it.

    Returns:
        One check per model with ``imported/mml/<model>.response.json``,
        sorted by model name; empty when no export is kept.
    """
    layer = facility_dir / _LAYER_DIR
    checks: list[ModelCheck] = []
    for model in sorted(document.get("models", []), key=lambda entry: str(entry.get("name"))):
        if not (layer / f"{model.get('name')}{RESPONSE_SUFFIX}").is_file():
            continue
        blocks = compare(facility_dir, document, model)
        origin = next((block.origin for block in blocks), "unstated")
        checks.append(ModelCheck(str(model["name"]), judge(blocks, origin)))
    return checks
