"""The exported response matrix, re-measured on the model the tree serves.

``osprey mml verify`` is the step between ``emit`` and ``osprey build``: it
boots the emitted tree in-process and asks the model the same question the
Middle Layer asked the machine -- move one corrector, what did the monitors do
-- then holds the answer against the matrix the export carries. A wrong
calibration, a binding on the wrong element, an energy knob rescaling the
wrong families: each of those boots happily and is visible only here, as a
column that disagrees with the file.

**The file is the oracle, and it is never adjusted.** Both sides are in the
physics units the export states, because
:func:`~osprey.services.virtual_accelerator.lattice.response.orbit_response`
drives the write path in hardware units and converts through the same
calibrations the bindings carry. What is left between the two is physics
rather than bookkeeping, so the comparison is a plain per-entry band:

    ``|R_model - R_file| <= 0.05 * max(|R_file|, 0.1 * rms(column))``

The ``rms`` term is the floor. Without it every near-zero entry -- the
cross-plane blocks of a model-derived export are exactly zero -- would be held
to a tolerance of zero and fail on the last bit of a solve. With it, an entry
far below its own column's scale is asked only to stay small, and its sign is
asserted only above that floor, where a sign means something.

**Alignment is by device row, never by position.** A response file states a
``DeviceList`` per side and the judged family states its own; a row the
reviewer dropped is gone from the second and still present in the first.
Pairing the two positionally would shift a whole column onto the wrong magnet
and still pass wherever the ring is periodic. So every row is matched by its
``[sector, device]`` key, and a row with no match is reported rather than
compared.

**Two ``Status`` fields, one of which is applied.** The response file's own
per-side ``Status`` says which rows were in the measurement, and a zero row is
dropped -- its matrix entries are ``NaN`` and comparing them means nothing.
The ``Status`` the AO carries is a different statement, which device the
control system considers in service today, and it is reported beside the
comparison and never applied: a magnet switched off this morning does not
change what a matrix measured years ago says.

The report this writes, ``data/mml/VA-REPORT.md``, is a reviewer's document:
where the matrix came from, how much of it agrees, and every entry that does
not -- plus the two things the emit lane had to decide quietly, the bands a
nominal widened and the nominals the model only seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.lattice.response import orbit_response
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.manifest.classify import (
    PARTITION_PYAT_COUPLED,
    READBACK_SUBFIELD,
    SETPOINT_SUBFIELD,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from osprey.services.mml.emit.va import ChannelBand, NominalSeed
    from osprey.services.mml.family import FamilyView, FieldView
    from osprey.services.mml.mapping.schema import VAFamily
    from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument
    from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

__all__ = [
    "ACTUATOR_KIND",
    "FLOOR_FRACTION",
    "REPORT_FILENAME",
    "TOLERANCE_FRACTION",
    "BlockReport",
    "Dropped",
    "Entry",
    "VerifyError",
    "VerifyReport",
    "model_channels",
    "render_report",
    "verify",
]

#: Where the report lands, under ``data/mml/``.
REPORT_FILENAME = "VA-REPORT.md"

#: The share of an entry's own scale the model may differ by.
TOLERANCE_FRACTION = 0.05

#: The share of a column's rms below which an entry is held to size alone and
#: its sign is not asserted.
FLOOR_FRACTION = 0.1

#: How many disagreeing entries the report lists per block, worst first.
OUTLIERS_LISTED = 12

#: The binding kind a response file's actuator side is driven through.
ACTUATOR_KIND = "kick"

#: The two transverse planes, in the order an orbit-response entry pairs them.
_PLANES = ("x", "y")


class VerifyError(ValueError):
    """The tree cannot be verified, for a reason a reviewer can act on."""


@dataclass(frozen=True)
class Entry:
    """One matrix entry, as the file states it and as the model measured it.

    Attributes:
        monitor_address: The channel the row was read on.
        actuator_address: The channel the column was driven through.
        monitor_device: The 1-based device position of the row.
        actuator_device: The 1-based device position of the column.
        file_value: The exported entry, in the export's physics units.
        model_value: What the served model did, in the same units.
        tolerance: The band this entry had to fall inside.
        floor: The column floor this entry was weighed against.
    """

    monitor_address: str
    actuator_address: str
    monitor_device: int
    actuator_device: int
    file_value: float
    model_value: float
    tolerance: float
    floor: float

    @property
    def deviation(self) -> float:
        """How far the model landed from the file."""
        return abs(self.model_value - self.file_value)

    @property
    def passed(self) -> bool:
        """Whether the entry is inside its band."""
        return self.deviation <= self.tolerance

    @property
    def above_floor(self) -> bool:
        """Whether the file states an entry large enough for its sign to mean something."""
        return abs(self.file_value) > self.floor

    @property
    def sign_agrees(self) -> bool | None:
        """Whether both sides move the beam the same way; ``None`` below the floor."""
        if not self.above_floor:
            return None
        return self.model_value * self.file_value > 0.0

    @property
    def ratio(self) -> float:
        """The model over the file -- what a measured matrix is read through."""
        if self.file_value == 0.0:
            return math.nan
        return self.model_value / self.file_value


@dataclass(frozen=True)
class Dropped:
    """One device row that was not compared, and why.

    Attributes:
        side: ``monitor`` or ``actuator``.
        family: The family the row belongs to.
        row: The ``DeviceList`` row, as the file writes it.
        reason: What stopped it being compared.
    """

    side: str
    family: str
    row: str
    reason: str


@dataclass(frozen=True)
class BlockReport:
    """One monitor family against one actuator family, compared entry by entry.

    Attributes:
        monitor_family: The family the rows were read on.
        actuator_family: The family the columns were driven through.
        monitor_mode: The ``Mode`` the monitor side was measured in.
        actuator_mode: The ``Mode`` the actuator side was driven in.
        origin: Where the matrix came from -- measured, or a model.
        timestamp: When the export says it was measured.
        gev: The beam energy the file states it was measured at.
        units: The units word the file states.
        units_string: What one entry is worth, as the file spells it.
        modulation_method: How the file says the actuator was modulated.
        actuator_delta: The hardware sweep the model repeated.
        entries: Every compared entry.
        dropped: Every row that was not compared.
    """

    monitor_family: str
    actuator_family: str
    monitor_mode: str
    actuator_mode: str
    origin: str
    timestamp: str
    gev: float | None
    units: str
    units_string: str
    modulation_method: str
    actuator_delta: float
    entries: tuple[Entry, ...]
    dropped: tuple[Dropped, ...]

    @property
    def compared(self) -> int:
        """How many entries were held against the file."""
        return len(self.entries)

    @property
    def passed(self) -> int:
        """How many of those are inside their band."""
        return sum(1 for entry in self.entries if entry.passed)

    @property
    def pass_ratio(self) -> float:
        """The share inside the band; ``nan`` when nothing was compared."""
        return self.passed / self.compared if self.entries else math.nan

    @property
    def signed(self) -> tuple[int, int]:
        """How many entries sit above the floor, and how many of those agree in sign."""
        checked = [entry for entry in self.entries if entry.above_floor]
        return len(checked), sum(1 for entry in checked if entry.sign_agrees)

    @property
    def median_ratio(self) -> float:
        """The median model-over-file magnitude ratio above the floor.

        The number a measured matrix is read through: an export taken on the
        machine differs from any model entry by entry, and what says the two
        describe one accelerator is that the bulk of the ratios sit near one.
        """
        return _median(
            sorted(
                abs(entry.ratio)
                for entry in self.entries
                if entry.above_floor and math.isfinite(entry.ratio)
            )
        )

    @property
    def outliers(self) -> tuple[Entry, ...]:
        """The entries outside their band, worst first."""
        failed = [entry for entry in self.entries if not entry.passed]
        failed.sort(key=lambda entry: entry.deviation / entry.tolerance, reverse=True)
        return tuple(failed)


@dataclass(frozen=True)
class VerifyReport:
    """Everything ``VA-REPORT.md`` states, as data.

    Attributes:
        system: The system the virtual accelerator serves.
        machine: The machine the export names.
        deck_energy_gev: The energy the emitted deck is built at.
        energy_at_nominal: The energy the export's own table reads at the
            energy family's nominal, or ``None`` where it states none.
        blocks: One report per block of the response document.
        bands: Every coupled setpoint's band, for the widened ones.
        seeds: Every nominal the machine document seeded or refused.
        ao_status_off: One line per family whose AO marks a device out of
            service -- reported, never applied.
        response_timestamp: When the response document was exported.
        response_file: The file the exported matrix was read from, empty where
            no file answered and the model was measured instead.
    """

    system: str
    machine: str
    deck_energy_gev: float
    energy_at_nominal: float | None
    blocks: tuple[BlockReport, ...]
    bands: tuple[ChannelBand, ...]
    seeds: tuple[NominalSeed, ...]
    ao_status_off: tuple[str, ...]
    response_timestamp: str
    response_file: str

    @property
    def compared(self) -> int:
        """Every entry held against the file, across every block."""
        return sum(block.compared for block in self.blocks)

    @property
    def passed(self) -> int:
        """Every entry inside its band, across every block."""
        return sum(block.passed for block in self.blocks)

    @property
    def pass_ratio(self) -> float:
        """The share inside the band; ``nan`` when nothing was compared."""
        return self.passed / self.compared if self.compared else math.nan

    @property
    def signed(self) -> tuple[int, int]:
        """Entries above the floor, and those of them that agree in sign."""
        return (
            sum(block.signed[0] for block in self.blocks),
            sum(block.signed[1] for block in self.blocks),
        )

    @property
    def widened(self) -> tuple[ChannelBand, ...]:
        """The bands a nominal outside its ``Range`` pushed open."""
        return tuple(band for band in self.bands if band.widened)

    @property
    def seed_only(self) -> tuple[NominalSeed, ...]:
        """The nominals the model starts a channel at and never maintains."""
        return tuple(seed for seed in self.seeds if seed.seed_only and seed.refused is None)

    @property
    def refused_nominals(self) -> tuple[NominalSeed, ...]:
        """The nominals nothing was seeded from."""
        return tuple(seed for seed in self.seeds if seed.refused is not None)


def model_channels(document: BindingsDocument) -> list[dict[str, str]]:
    """The channel list a model over an emitted tree is served on.

    An emitted tree carries no channel manifest -- the served namespace is
    derived at build time from the facility's own databases -- so the set a
    verification model is built on is the bindings themselves, which is
    exactly the pyat-coupled partition a build would put in that manifest.
    Only the three keys the variable catalog reads are stated: this list is
    served to nothing, so a record type, a noise flag and a hierarchy path
    have no reader here.

    The subfields follow the manifest's own vocabulary, because they are what
    decides whether an address is written: a written binding states its
    setpoint and, where it serves one, its readback; a monitor states the axis
    it reads, which is neither and so is read only.

    Args:
        document: The served ``va_bindings.json``, already parsed.

    Returns:
        One entry per address the bindings claim, in document order.
    """
    channels: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(address: str, subfield: str) -> None:
        if address and address not in seen:
            seen.add(address)
            channels.append(
                {"address": address, "partition": PARTITION_PYAT_COUPLED, "subfield": subfield}
            )

    for binding in document.bindings:
        if binding.kind == "monitor":
            add(binding.setpoint_address, (binding.attribute or "").upper())
            continue
        add(binding.setpoint_address, SETPOINT_SUBFIELD)
        if binding.readback_address:
            add(binding.readback_address, READBACK_SUBFIELD)
    return channels


def verify(
    data_dir: Path,
    *,
    system: str,
    response: dict,
    views: Iterable[FamilyView],
    verdicts: Mapping[tuple[str, str], VAFamily],
    judged_va: Mapping[tuple[str, str], dict],
    seeds: Sequence[NominalSeed],
    bands: Sequence[ChannelBand],
) -> VerifyReport:
    """Re-measure every exported response block on the model the tree serves.

    One solve pass per actuator device, whatever the file states about it: the
    blocks that share an actuator family and a sweep width are driven together
    and every monitor they name is read off the same two arms, so a matrix of
    ``n`` correctors costs ``3n`` solves rather than three per block.

    The blocks are reported in the order the document writes them, whether or
    not each one could be re-measured, so a reader holding the report beside
    the export reads the two in one order.

    Args:
        data_dir: The deployment's ``data/`` directory -- the emitted tree the
            model is built over.
        system: The raw system token the virtual accelerator serves.
        response: That system's block of the canonical ``response.json``.
        views: The judged family views of the system, whose ``DeviceList``
            rows the file's rows are aligned against.
        verdicts: What the virtual accelerator does with each family, keyed by
            ``(raw system, raw family)``.
        judged_va: Each family's ``va.json`` block in the judged device order,
            keyed the same way; the energy family's table is read off it.
        seeds: What the machine emitter read, for the report to list.
        bands: What the limits emitter derived, for the widened ones.

    Returns:
        Everything the report states.

    Raises:
        VerifyError: The response document states no block at all.
        FileNotFoundError: The tree carries no emitted model; the message
            names the file that is missing.
    """
    from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

    blocks = _blocks(response)
    if not blocks:
        raise VerifyError(
            "the response document states no block for this system, so there is no exported "
            "matrix to hold the model against; re-export with mml_export 2.0"
        )

    document = load_bindings(ManifestPaths(data_root=data_dir).va_bindings)
    grain = {view.raw_name: view for view in views if view.system == system}
    by_family = _bindings_by_device(document, grain, verdicts, system)
    model = PyATRingModel(data_dir, model_channels(document))

    swept = {
        key: _sweep(model, by_family, key, group, grain, judged_va)
        for key, group in _grouped(blocks, verdicts, by_family, system).items()
    }

    reports: list[BlockReport] = []
    for block in blocks:
        refusal = _undrivable(block, verdicts, by_family, system)
        measured, unsolved = ({}, {}) if refusal is not None else swept[_group_key(block)]
        reports.append(
            _block_report(block, measured, unsolved, grain, judged_va, by_family, refusal)
        )

    export = response.get("_export")
    export = export if isinstance(export, dict) else {}
    return VerifyReport(
        system=system,
        machine=_word(export.get("machine")),
        deck_energy_gev=document.energy_gev,
        energy_at_nominal=_energy_at_nominal(judged_va, verdicts, system),
        blocks=tuple(reports),
        bands=tuple(bands),
        seeds=tuple(seeds),
        ao_status_off=_ao_status_off(grain),
        response_timestamp=_word(export.get("timestamp")),
        response_file=_word(response.get("file")),
    )


# --- the exported document -------------------------------------------------


def _blocks(response: dict) -> list[dict]:
    """The response document's blocks, each of which pairs two families."""
    blocks = response.get("blocks")
    if not isinstance(blocks, list):
        return []
    return [block for block in blocks if isinstance(block, dict)]


def _side(block: dict, side: str) -> dict:
    """One side of a block -- its family, its rows, its ``Status`` and its mode."""
    body = block.get(side)
    return body if isinstance(body, dict) else {}


def _family(block: dict, side: str) -> str:
    return _word(_side(block, side).get("family"))


def _rows(block: dict, side: str) -> list[Any]:
    rows = _side(block, side).get("device_list")
    return list(rows) if isinstance(rows, list) else []


def _statuses(block: dict, side: str) -> list[Any]:
    status = _side(block, side).get("status")
    return list(status) if isinstance(status, list) else []


def _delta(block: dict) -> float:
    """The hardware sweep the file was measured with."""
    delta = _number(block.get("actuator_delta"))
    return delta if delta is not None else math.nan


# --- what the model can be asked ------------------------------------------


def _undrivable(
    block: dict,
    verdicts: Mapping[tuple[str, str], VAFamily],
    by_family: Mapping[str, dict[int, Binding]],
    system: str,
) -> str | None:
    """Why this block cannot be re-measured, or ``None`` when it can be."""
    family = _family(block, "actuator")
    verdict = verdicts.get((system, family))
    kind = None if verdict is None else str(verdict.kind)
    if kind != ACTUATOR_KIND:
        stated = "is not coupled" if kind is None else f"couples as {kind}"
        return (
            f"the actuator family {family!r} {stated}, and an orbit response is driven "
            f"through a coupled {ACTUATOR_KIND} corrector"
        )
    if not by_family.get(family):
        return f"the emitted bindings carry no device of the actuator family {family!r}"
    if not math.isfinite(_delta(block)):
        return f"the block states no finite actuator_delta for {family!r} to be swept by"
    return None


def _grouped(
    blocks: Iterable[dict],
    verdicts: Mapping[tuple[str, str], VAFamily],
    by_family: Mapping[str, dict[int, Binding]],
    system: str,
) -> dict[tuple[str, float], list[dict]]:
    """The drivable blocks, grouped by the sweep one solve pass can serve.

    Two blocks share a pass when they drive the same family through the same
    hardware delta: the arms are then the same two lattice states, and every
    monitor either block names is read off them.
    """
    groups: dict[tuple[str, float], list[dict]] = {}
    for block in blocks:
        if _undrivable(block, verdicts, by_family, system) is not None:
            continue
        groups.setdefault(_group_key(block), []).append(block)
    return groups


def _group_key(block: dict) -> tuple[str, float]:
    """The sweep one drivable block is served by: its actuator family and delta."""
    return (_family(block, "actuator"), _delta(block))


def _sweep(
    model: PyATRingModel,
    by_family: Mapping[str, dict[int, Binding]],
    key: tuple[str, float],
    group: Sequence[dict],
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[int, str]]:
    """Drive one actuator family through one sweep width, device by device.

    A corrector whose arms leave the lattice without a stable closed orbit is
    a fact about that column, not a reason to abandon the rest of the matrix:
    the solver's refusal is kept against the device and every block that names
    it reports the column among the rows that were not compared.

    Args:
        model: The model the tree serves, driven and left where it was found.
        by_family: The emitted bindings, keyed by family and device position.
        key: The actuator family and the hardware delta the group is swept by.
        group: The blocks this one pass serves.
        grain: The judged family views, for the rows the file keeps.
        judged_va: Each family's judged ``va.json`` block.

    Returns:
        What each swept device's monitors read, and, for each device the
        solver refused, the reason the report states in its place.
    """
    family, delta = key
    monitors = _monitors(group, by_family)
    bindings = by_family[family]
    measured: dict[int, dict[str, tuple[float, float]]] = {}
    unsolved: dict[int, str] = {}
    for device in sorted(_driven(group, grain, judged_va, by_family)):
        binding = bindings[device]
        try:
            measured[device] = orbit_response(model, binding, delta, monitors=monitors)
        except OrbitSolveError as exc:
            unsolved[device] = (
                f"sweeping {binding.setpoint_address} by {_figure(delta)} leaves the lattice "
                f"without a stable closed orbit ({exc})"
            )
    return measured, unsolved


def _driven(
    group: Iterable[dict],
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
    by_family: Mapping[str, dict[int, Binding]],
) -> set[int]:
    """The actuator devices some block of this group needs swept."""
    devices: set[int] = set()
    for block in group:
        family = _family(block, "actuator")
        bindings = by_family.get(family, {})
        for device in _kept(block, "actuator", grain, judged_va):
            if device in bindings:
                devices.add(device)
    return devices


def _monitors(group: Iterable[dict], by_family: Mapping[str, dict[int, Binding]]) -> list[Binding]:
    """Every monitor binding the group's blocks read, each named once."""
    monitors: dict[str, Binding] = {}
    for block in group:
        for binding in by_family.get(_family(block, "monitor"), {}).values():
            monitors.setdefault(binding.setpoint_address, binding)
    return list(monitors.values())


# --- alignment -------------------------------------------------------------


def _kept(
    block: dict,
    side: str,
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
) -> dict[int, int]:
    """The file's rows of one side that survive alignment and ``Status``.

    Returns:
        The judged 0-based device position of each kept row, keyed by the
        row's own position in the file.
    """
    return {
        index: device
        for index, (device, reason) in _alignment(block, side, grain, judged_va).items()
        if reason is None and device is not None
    }


def _alignment(
    block: dict,
    side: str,
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
) -> dict[int, tuple[int | None, str | None]]:
    """Every row of one side, as a judged device position or a reason it is not one."""
    family = _family(block, side)
    view = grain.get(family)
    judged = _judged_rows(judged_va, family, view)
    statuses = _statuses(block, side)
    aligned: dict[int, tuple[int | None, str | None]] = {}
    for index, row in enumerate(_rows(block, side)):
        status = _number(_first(statuses[index])) if index < len(statuses) else None
        if status is not None and status == 0.0:
            aligned[index] = (None, "the response file marks the row Status 0")
            continue
        key = _row_key(row)
        device = judged.get(key) if key is not None else None
        if device is None:
            aligned[index] = (
                None,
                f"no device of the judged {family} sits at this row",
            )
            continue
        aligned[index] = (device, None)
    return aligned


def _judged_rows(
    judged_va: Mapping[tuple[str, str], dict], family: str, view: FamilyView | None
) -> dict[tuple[int, ...], int]:
    """The judged ``DeviceList`` of one family, as a row key to device position."""
    if view is None:
        return {}
    block = judged_va.get((view.system, family))
    rows = block.get("device_list") if isinstance(block, dict) else None
    if not isinstance(rows, list):
        return {}
    found: dict[tuple[int, ...], int] = {}
    for index, row in enumerate(rows):
        key = _row_key(row)
        if key is not None:
            found.setdefault(key, index)
    return found


def _row_key(row: Any) -> tuple[int, ...] | None:
    """One ``DeviceList`` row as the key both lists are matched on."""
    values = row if isinstance(row, (list, tuple)) else [row]
    key: list[int] = []
    for value in values:
        number = _number(value)
        if number is None or number != int(number):
            return None
        key.append(int(number))
    return tuple(key) if key else None


def _bindings_by_device(
    document: BindingsDocument,
    grain: Mapping[str, FamilyView],
    verdicts: Mapping[tuple[str, str], VAFamily],
    system: str,
) -> dict[str, dict[int, Binding]]:
    """The emitted bindings, keyed by family and judged device position.

    A binding carries the address it claims and not the device it was written
    for, and the emit lane skips a device whose element or address the export
    does not state -- so the bindings of a family are not one per device and
    cannot be indexed by position. The field the family is keyed by states
    which device each address sits at, and that is the one lookup that holds
    whatever the export left out.
    """
    by_family: dict[str, dict[int, Binding]] = {}
    for binding in document.bindings:
        by_family.setdefault(binding.family, {})
    for family, bindings in by_family.items():
        verdict = verdicts.get((system, family))
        written = (
            "Monitor" if verdict is not None and str(verdict.kind) == "monitor" else "Setpoint"
        )
        view = grain.get(family)
        devices = _device_by_address(None if view is None else view.fields.get(written))
        for binding in document.bindings:
            if binding.family != family:
                continue
            device = devices.get(binding.setpoint_address)
            if device is not None:
                bindings[device] = binding
    return by_family


def _device_by_address(field_view: FieldView | None) -> dict[str, int]:
    """The 0-based device position every address of one field sits at."""
    if field_view is None:
        return {}
    found: dict[str, int] = {}
    for key in field_view.keys:
        for index, slot in enumerate(field_view.slots(key)):
            if isinstance(slot, str) and slot.strip():
                found.setdefault(slot.strip(), index)
    return found


# --- the comparison --------------------------------------------------------


def _block_report(
    block: dict,
    measured: Mapping[int, dict[str, tuple[float, float]]],
    unsolved: Mapping[int, str],
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
    by_family: Mapping[str, dict[int, Binding]],
    refusal: str | None,
) -> BlockReport:
    """Hold one block's matrix against what the model did, entry by entry."""
    monitor_family = _family(block, "monitor")
    actuator_family = _family(block, "actuator")
    monitors = by_family.get(monitor_family, {})
    actuators = by_family.get(actuator_family, {})
    data = block.get("data")
    data = data if isinstance(data, list) else []

    dropped: list[Dropped] = []
    rows = _usable(
        block, "monitor", grain, judged_va, monitors, dropped, None, "the bindings read no"
    )
    columns = _usable(
        block, "actuator", grain, judged_va, actuators, dropped, refusal, "the bindings drive no"
    )

    entries: list[Entry] = []
    unread: dict[int, str] = {}
    for column, actuator_device in columns.items():
        readings = measured.get(actuator_device)
        if readings is None:
            dropped.append(
                Dropped(
                    side="actuator",
                    family=actuator_family,
                    row=_row_text(_rows(block, "actuator"), column),
                    reason=unsolved.get(
                        actuator_device, "the model measured no response for this device"
                    ),
                )
            )
            continue
        stated = {
            row: value
            for row, device in rows.items()
            if (value := _matrix(data, row, column)) is not None
        }
        floor = FLOOR_FRACTION * _rms(stated.values())
        for row, file_value in stated.items():
            monitor = monitors[rows[row]]
            model_value = _reading(readings, monitor)
            if model_value is None:
                unread[row] = monitor.setpoint_address
                continue
            entries.append(
                Entry(
                    monitor_address=monitor.setpoint_address,
                    actuator_address=actuators[actuator_device].setpoint_address,
                    monitor_device=rows[row] + 1,
                    actuator_device=actuator_device + 1,
                    file_value=file_value,
                    model_value=model_value,
                    tolerance=TOLERANCE_FRACTION * max(abs(file_value), floor),
                    floor=floor,
                )
            )

    # A monitor the model publishes nothing on is one unreadable row, not one
    # per column it would have been compared against.
    dropped += [
        Dropped(
            side="monitor",
            family=monitor_family,
            row=_row_text(_rows(block, "monitor"), row),
            reason=f"the model published nothing on {address}",
        )
        for row, address in sorted(unread.items())
    ]

    return BlockReport(
        monitor_family=monitor_family,
        actuator_family=actuator_family,
        monitor_mode=_word(_side(block, "monitor").get("mode")),
        actuator_mode=_word(_side(block, "actuator").get("mode")),
        origin=_word(block.get("origin")),
        timestamp=_word(block.get("timestamp")),
        gev=_number(block.get("gev")),
        units=_word(block.get("units")),
        units_string=_word(block.get("units_string")),
        modulation_method=_word(block.get("modulation_method")),
        actuator_delta=_delta(block),
        entries=tuple(entries),
        dropped=tuple(dropped),
    )


def _usable(
    block: dict,
    side: str,
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
    bindings: Mapping[int, Binding],
    dropped: list[Dropped],
    refusal: str | None,
    unbound: str,
) -> dict[int, int]:
    """One side's rows that reach a binding, recording why each other one did not."""
    family = _family(block, side)
    rows = _rows(block, side)
    usable: dict[int, int] = {}
    for index, (device, reason) in _alignment(block, side, grain, judged_va).items():
        if refusal is not None:
            reason = refusal
        elif reason is None and device not in bindings:
            reason = f"{unbound} device at this row"
        if reason is not None:
            dropped.append(
                Dropped(side=side, family=family, row=_row_text(rows, index), reason=reason)
            )
            continue
        usable[index] = device  # type: ignore[assignment]
    return usable


def _matrix(data: list, row: int, column: int) -> float | None:
    """One entry of the exported matrix, or ``None`` where it states no number."""
    if row >= len(data):
        return None
    line = data[row]
    if not isinstance(line, list) or column >= len(line):
        return None
    return _number(line[column])


def _reading(readings: Mapping[str, tuple[float, float]], monitor: Binding) -> float | None:
    """What the model's monitor read, in the plane its binding names."""
    element = readings.get(str(monitor.element))
    plane = (monitor.attribute or "").lower()
    if element is None or plane not in _PLANES:
        return None
    value = element[_PLANES.index(plane)]
    return value if math.isfinite(value) else None


def _rms(values: Iterable[float]) -> float:
    """The root mean square of a column, which sets its floor."""
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


# --- the facts beside the comparison ---------------------------------------


def _energy_at_nominal(
    judged_va: Mapping[tuple[str, str], dict],
    verdicts: Mapping[tuple[str, str], VAFamily],
    system: str,
) -> float | None:
    """The energy the export's own table reads at the energy family's nominal."""
    for (block_system, family), verdict in verdicts.items():
        if block_system != system or str(verdict.kind) != "energy":
            continue
        block = judged_va.get((system, family))
        table = block.get("energy_table") if isinstance(block, dict) else None
        if isinstance(table, dict):
            return _number(table.get("energy_at_nominal"))
    return None


def _ao_status_off(grain: Mapping[str, FamilyView]) -> tuple[str, ...]:
    """One line per family whose AO marks a device out of service.

    Reported and never applied: the AO says what the control system thinks of
    a magnet today, and the matrix was measured whenever it was measured.
    """
    lines: list[str] = []
    for family in sorted(grain):
        view = grain[family]
        status = view.body.get("Status")
        if not isinstance(status, list):
            continue
        off = sum(1 for row in status if (value := _number(_first(row))) is not None and value == 0)
        if off:
            lines.append(f"{family}: {off} of {len(status)} devices")
    return tuple(lines)


def _first(row: Any) -> Any:
    """A ``Status`` row, which a facility may state as a scalar or a one-column row."""
    if isinstance(row, (list, tuple)):
        return row[0] if row else None
    return row


def _row_text(rows: Sequence[Any], index: int) -> str:
    """One ``DeviceList`` row as the report writes it."""
    if index >= len(rows):
        return f"row {index + 1}"
    row = rows[index]
    if isinstance(row, (list, tuple)):
        return "[" + ", ".join(_figure(_number(value)) for value in row) + "]"
    return _figure(_number(row))


def _or_unstated(value: Any) -> str:
    """A string field of one block, or the word for a block that states none."""
    return str(value) if value else "unstated"


def _word(value: Any) -> str:
    """A string field of the export, or an empty string where it states none."""
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    """A finite number as a float, or ``None``.

    Normalisation spells a non-finite number as a string (``"NaN"``, ``"Inf"``),
    so a string is never a value here.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# --- the report ------------------------------------------------------------


def render_report(report: VerifyReport, *, provenance: str) -> str:
    """Render ``VA-REPORT.md``: what a reviewer reads before ``osprey build``.

    Args:
        report: What the comparison found.
        provenance: The emit run's provenance string, so the report names the
            export it was written from.

    Returns:
        The document text, ending in a newline.
    """
    lines: list[str] = [
        f"# Virtual accelerator — {report.machine or 'export'} {report.system}".rstrip(),
        "",
        f"Written by `osprey mml verify` from {provenance}.",
        "Read it before `osprey build`: it says whether the deck this deployment "
        "will serve answers like the machine the export came from.",
        "",
    ]
    lines += _headline(report)
    lines += _energy_section(report)
    lines += _response_section(report)
    lines += _dropped_section(report)
    lines += _bands_section(report)
    lines += _nominals_section(report)
    return "\n".join(lines).rstrip() + "\n"


def _headline(report: VerifyReport) -> list[str]:
    """The one paragraph that says how the comparison went."""
    checked, agreed = report.signed
    lines = ["## Verdict", ""]
    if not report.compared:
        lines += [
            "Not one entry of the exported response matrix was compared. The sections "
            "below say why; until they are answered this deployment has no evidence "
            "that its virtual accelerator answers like the machine.",
            "",
        ]
        return lines
    lines += [
        f"{report.passed} of {report.compared} entries "
        f"({_percent(report.pass_ratio)}) are inside the band "
        f"`|R_model - R_file| <= {TOLERANCE_FRACTION} * "
        f"max(|R_file|, {FLOOR_FRACTION} * rms(column))`.",
        "",
        "`rms(column)` is the root mean square of the file's own entries in that "
        "actuator's column, taken over the rows that were compared -- so the floor "
        "is the column's scale as the export states it, and the model has no say in "
        "the tolerance it is held to.",
        "",
        f"Sign agrees on {agreed} of the {checked} entries above the floor."
        if checked
        else "No entry sits above its column's floor, so no sign was asserted.",
        "",
    ]
    return lines


def _block_name(block: BlockReport) -> str:
    """One block as every table of the report names it."""
    return f"{block.monitor_family} ← {block.actuator_family}"


def _stated(
    blocks: Sequence[BlockReport], read: Callable[[BlockReport], Any]
) -> list[tuple[Any, list[str]]]:
    """Each distinct value one per-block field takes, with the blocks stating it.

    In the document's own order, so a field the blocks disagree on is read in
    the order the export writes them.
    """
    found: list[tuple[Any, list[str]]] = []
    for block in blocks:
        value = read(block)
        for stated, names in found:
            if stated == value:
                names.append(_block_name(block))
                break
        else:
            found.append((value, [_block_name(block)]))
    return found


def _provenance(
    label: str,
    blocks: Sequence[BlockReport],
    read: Callable[[BlockReport], Any],
    spell: Callable[[Any], str],
) -> str:
    """One row of the ``Export`` table, which is a per-block fact.

    Where every block says the same thing the row says it once. Where they do
    not, it says each thing and which blocks say it, rather than lending one
    block's provenance to entries measured under another's.
    """
    stated = _stated(blocks, read)
    if len(stated) == 1:
        return f"| {label} | {spell(stated[0][0])} |"
    said = "; ".join(f"{spell(value)} ({', '.join(names)})" for value, names in stated)
    return f"| {label} | {said} |"


def _energy_section(report: VerifyReport) -> list[str]:
    """Where the matrix came from, and at which energy.

    Origin, timestamp, units, modulation and energy are stated per block, and
    an export may carry a measured block beside a model-derived one or two
    taken at different energies. Every row here therefore reads all the blocks
    and names them wherever they disagree.
    """
    lines = ["## Export", "", "| what | value |", "| --- | --- |"]
    blocks = report.blocks
    if blocks:
        lines += [
            _provenance("origin", blocks, lambda block: block.origin, _or_unstated),
            _provenance("measured", blocks, lambda block: block.timestamp, _or_unstated),
            _provenance(
                "units",
                blocks,
                lambda block: (block.units, block.units_string),
                lambda value: f"{value[0]} ({value[1]})",
            ),
            _provenance("modulation", blocks, lambda block: block.modulation_method, _or_unstated),
            _provenance(
                "file energy",
                blocks,
                lambda block: block.gev,
                lambda value: f"{_figure(value)} GeV",
            ),
        ]
    lines += [
        f"| deck energy | {_figure(report.deck_energy_gev)} GeV |",
        f"| energy at nominal | {_figure(report.energy_at_nominal)} GeV |",
        f"| read from | {report.response_file or 'no file: the model was measured'} |",
        f"| exported | {report.response_timestamp or 'unstated'} |",
        "",
    ]
    lines += _model_derived_lines(blocks)
    energies = [
        (gev, names) for gev, names in _stated(blocks, lambda block: block.gev) if gev is not None
    ]
    for gev, names in energies:
        subject = (
            "The matrix was measured"
            if len(energies) == 1
            else f"{', '.join(names)} {'were' if len(names) > 1 else 'was'} measured"
        )
        lines += [_energy_line(subject, gev, report.deck_energy_gev), ""]
    if report.energy_at_nominal is not None:
        lines += [
            _energy_line(
                "The export's own table reads the ring",
                report.energy_at_nominal,
                report.deck_energy_gev,
            ),
            "",
        ]
    return lines


def _model_derived_lines(blocks: Sequence[BlockReport]) -> list[str]:
    """What a matrix nobody measured can and cannot settle.

    A model-derived block was computed from a model rather than measured on a
    machine, so both sides of the comparison are physics and the only thing
    under test is the chain between them. On the page that reads exactly like a
    machine the model agrees with, which is the one conclusion it does not
    support -- so the page says so where the origin is stated.

    Which model it came from is a second question the origin does not answer. A
    facility may keep a computed matrix in a file like any other, and the row
    naming the file is what separates a matrix this deck answered a moment ago
    from one an older deck answered years back.
    """
    modelled = [
        names for origin, names in _stated(blocks, lambda block: block.origin) if origin == "model"
    ]
    if not modelled:
        return []
    named = [name for names in modelled for name in names]
    subject = (
        "The matrix was"
        if len(named) == len(blocks)
        else f"{', '.join(named)} {'were' if len(named) > 1 else 'was'}"
    )
    return [
        f"{subject} computed from a model rather than measured on the machine, "
        "so the comparison above holds a deck against a deck. Agreement there is "
        "the chain doing its job -- the bindings, the device order, the units, the "
        "conversions -- and is not the machine agreeing with the model. The row "
        "above says whether the matrix came from a file or was measured here.",
        "",
    ]


#: How far two stated energies may sit apart before the report calls it a gap.
_ENERGY_AGREEMENT = 1e-3


def _energy_line(subject: str, stated: float, deck: float) -> str:
    """One energy against the deck's, said plainly whether or not they agree.

    A response scales with the rigidity ratio, so an energy the export and the
    deck disagree on is a scale factor sitting under every entry above -- and
    a large enough one is not a tolerance question but a different ring.
    """
    if deck and abs(stated - deck) <= _ENERGY_AGREEMENT * abs(deck):
        return f"{subject} at {_figure(stated)} GeV, which is the energy the deck is built for."
    return (
        f"{subject} at {_figure(stated)} GeV and the deck is built for "
        f"{_figure(deck)} GeV ({_percent(stated / deck) if deck else '—'} of it). "
        "That ratio sits under every entry it covers: a response scales with the "
        "rigidity, so check the export before reading the comparison as a verdict on "
        "the bindings."
    )


def _response_section(report: VerifyReport) -> list[str]:
    """One row per block, then the entries that fell outside their band."""
    lines = [
        "## Orbit response",
        "",
        "| monitors | actuators | compared | inside the band | sign above floor | median ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for block in report.blocks:
        checked, agreed = block.signed
        lines.append(
            f"| {block.monitor_family} | {block.actuator_family} | {block.compared} | "
            f"{block.passed} ({_percent(block.pass_ratio)}) | {agreed}/{checked} | "
            f"{_figure(block.median_ratio)} |"
        )
    lines.append("")
    for block in report.blocks:
        outliers = block.outliers
        if not outliers:
            continue
        lines += [
            f"### {block.monitor_family} outliers against {block.actuator_family}",
            "",
            "| monitor | actuator | file | model | allowed | off by |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for entry in outliers[:OUTLIERS_LISTED]:
            lines.append(
                f"| {entry.monitor_address} | {entry.actuator_address} | "
                f"{_figure(entry.file_value)} | {_figure(entry.model_value)} | "
                f"{_figure(entry.tolerance)} | {_figure(entry.deviation)} |"
            )
        if len(outliers) > OUTLIERS_LISTED:
            lines.append(f"| … | and {len(outliers) - OUTLIERS_LISTED} more | | | | |")
        lines.append("")
    return lines


def _dropped_section(report: VerifyReport) -> list[str]:
    """Every row the comparison left out, and the status it did not act on."""
    lines = ["## Rows not compared", ""]
    dropped = [(block, row) for block in report.blocks for row in block.dropped]
    if dropped:
        lines += [
            "| monitors ← actuators | side | family | row | why |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {block.monitor_family} ← {block.actuator_family} | {row.side} | "
            f"{row.family} | {row.row} | {row.reason} |"
            for block, row in dropped
        ]
    else:
        lines.append("Every row of every block reached a binding and was compared.")
    lines.append("")
    if report.ao_status_off:
        lines += [
            "The AO marks these devices out of service. That is read for information "
            "and never applied: the matrix was measured when it was measured, and a "
            "device switched off today does not change what it says.",
            "",
        ]
        lines += [f"- {line}" for line in report.ao_status_off]
        lines.append("")
    return lines


def _bands_section(report: VerifyReport) -> list[str]:
    """The write bands a nominal pushed open."""
    lines = ["## Widened bands", ""]
    widened = report.widened
    if not widened:
        lines += ["Every nominal sits inside the `Range` its family exports.", ""]
        return lines
    lines += [
        "These nominals sit outside the `Range` their family exports. The model "
        "refuses to boot on a nominal outside its band, so the band gave way -- "
        "check the export before the deployment writes into the widened part.",
        "",
        "| address | family | band | nominal |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {band.address} | {band.family} | "
        f"{_figure(band.min_value)} to {_figure(band.max_value)} | {_figure(band.nominal)} |"
        for band in widened
    ]
    lines.append("")
    return lines


def _nominals_section(report: VerifyReport) -> list[str]:
    """The nominals the model only seeds, and the ones nothing was seeded from."""
    lines = ["## Nominals the model does not maintain", ""]
    families: dict[tuple[str, str, str], int] = {}
    for seed in report.seed_only:
        families[(seed.family, seed.field, seed.at_type)] = (
            families.get((seed.family, seed.field, seed.at_type), 0) + 1
        )
    if families:
        lines += [
            "These channels start at the exported value and stay there: the virtual "
            "accelerator drives nothing behind them.",
            "",
            "| family | field | AT field | channels |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {family} | {field} | {at_type or 'none'} | {count} |"
            for (family, field, at_type), count in sorted(families.items())
        ]
        lines.append("")
    else:
        lines += ["Every nominal the export states seeds a channel the model drives.", ""]
    refused = report.refused_nominals
    if refused:
        lines += [
            "Nothing was seeded from these, so their channels start wherever the "
            "deployment's own files put them.",
            "",
            "| family | field | address | why |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {seed.family} | {seed.field} | {seed.address or 'the whole nominal'} | "
            f"{seed.refused} |"
            for seed in refused
        ]
        lines.append("")
    return lines


def _figure(value: float | None) -> str:
    """A number as the report spells it, or a dash where there is none."""
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.6g}"


def _percent(ratio: float) -> str:
    """A ratio as a percentage, or a dash where there is none."""
    if not math.isfinite(ratio):
        return "—"
    return f"{ratio * 100:.1f} %"
