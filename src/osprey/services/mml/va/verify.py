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

    ``|R_model - R_file| <= 0.05 * max(|R_file|, 0.1 * rms(matrix))``

The ``rms`` term is the floor, and it is the whole file's: the root mean
square of every entry of the export that was compared, in every block. Without
a floor each near-zero entry -- the cross-plane blocks of a model-derived
export are exactly zero -- would be held to a tolerance of zero and fail on the
last bit of a solve. Drawn per column instead, a column the file states as zero
throughout would set its own floor to zero and fail the same way, so the scale
an entry is asked to stay small against is the matrix it belongs to.

**A block is judged where the two planes meet.** An orbit response pairs a
monitor family with a corrector family, and the emitted bindings say which
transverse plane each of them works in: a monitor states the axis it reads, a
kick states the component of its attribute it writes. Where those planes are
the same the block is judged, and its entries are what the verdict is made of.
Where they differ the block is compared and printed with the same numbers and
judged by nothing -- what a deck couples the two planes by is the deck's own,
and a measured file's cross-plane entries are its noise. No family name decides
any of this: a facility calls its horizontal correctors whatever it likes.
Sign is asserted above the floor and inside a judged block, where it means the
model pushes the beam the way the machine did.

**A column that runs backwards is named, not absorbed.** Inside a judged
block, one corrector whose entries above the floor disagree in sign on more
than half of themselves is the file and the model disagreeing about that
device rather than about the ring: a reversed cable, at the size the rest of
the matrix is worth, which nothing in a lattice or a calibration reproduces.
Such a column is reported on its own and left out of what the block counts, so
the comparison is a statement about the correctors whose direction the two
sides agree on, and the reversed ones are a short list somebody can walk down.

**Alignment is by device row, never by position.** A response file states a
``DeviceList`` per side and the judged family states its own; a row the
reviewer dropped is gone from the second and still present in the first.
Pairing the two positionally would shift a whole column onto the wrong magnet
and still pass wherever the ring is periodic. So every row is matched by its
``[sector, device]`` key, and a row with no match is reported rather than
compared.

**The sweep is one width per corrector.** A response file states its
``ActuatorDelta`` per device of the actuator side, because a machine trims the
kick it measures each corrector with, and a file stating one number swept
every device by it. So each column is re-measured by the width its own device
was measured with, and a column the file states no width for is reported
rather than swept by another device's.

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
from dataclasses import dataclass, replace
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

    from osprey.services.mml.emit.va import (
        CalibrationTrim,
        ChannelBand,
        NominalSeed,
        SeriesSupply,
    )
    from osprey.services.mml.family import FamilyView, FieldView
    from osprey.services.mml.mapping.schema import VAFamily
    from osprey.services.mml.va.elements import ServedMarker
    from osprey.services.mml.va.verdicts import BuiltCavity
    from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument
    from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

__all__ = [
    "ACTUATOR_KIND",
    "FLOOR_FRACTION",
    "POLARITY_SHARE",
    "REPORT_FILENAME",
    "TOLERANCE_FRACTION",
    "BlockReport",
    "Dropped",
    "Entry",
    "Figures",
    "PolarityColumn",
    "VerifyError",
    "VerifyReport",
    "figures",
    "model_channels",
    "render_report",
    "verify",
]

#: Where the report lands, under ``data/mml/``.
REPORT_FILENAME = "VA-REPORT.md"

#: The share of an entry's own scale the model may differ by.
TOLERANCE_FRACTION = 0.05

#: The share of the whole matrix's rms below which an entry is held to size
#: alone and its sign is not asserted.
FLOOR_FRACTION = 0.1

#: How many disagreeing entries the report lists per block, worst first.
OUTLIERS_LISTED = 12

#: The share of a column's compared entries that has to disagree in sign
#: before the column is the two sides disagreeing about the device rather than
#: a scatter of misses. More than half, so a tie is not a polarity outlier,
#: and a column has to be mostly sizable as well as mostly flipped.
POLARITY_SHARE = 0.5

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
        floor: The matrix floor this entry was weighed against.
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
        """Whether the file states an entry large enough for its sign to mean something.

        Large against the matrix's own scale: a corrector that barely moves a
        monitor is the file saying so, not a direction it has an opinion on.
        """
        return abs(self.file_value) > self.floor

    @property
    def sign_agrees(self) -> bool | None:
        """Whether both sides move the beam the same way; ``None`` below the floor."""
        if not self.above_floor:
            return None
        return self.model_value * self.file_value > 0.0

    @property
    def excess(self) -> float:
        """How many times its own band the entry missed by.

        A file that states zero everywhere it was compared gives the matrix no
        scale, so its floor and every band in it are zero and anything the
        model does is outside by every measure: the excess is infinite rather
        than a division nobody can take.
        """
        if self.tolerance == 0.0:
            return math.inf if self.deviation else 0.0
        return self.deviation / self.tolerance

    @property
    def ratio(self) -> float:
        """The model over the file -- what a measured matrix is read through."""
        if self.file_value == 0.0:
            return math.nan
        return self.model_value / self.file_value


@dataclass(frozen=True)
class Figures:
    """What one set of compared entries says about the model.

    The same five numbers describe a block, a block with its polarity columns
    left out, and the whole verdict, so they are computed once and read the
    same way wherever they appear.

    Attributes:
        compared: How many entries were held against the file.
        passed: How many of them are inside their band.
        checked: How many sit above the floor, where a sign means something.
        agreed: How many of those push the beam the way the file says.
        median_ratio: The median model-over-file magnitude, above the floor.
    """

    compared: int
    passed: int
    checked: int
    agreed: int
    median_ratio: float

    @property
    def pass_ratio(self) -> float:
        """The share inside the band; ``nan`` when nothing was compared."""
        return self.passed / self.compared if self.compared else math.nan

    @property
    def signed(self) -> tuple[int, int]:
        """How many entries sit above the floor, and how many of those agree."""
        return (self.checked, self.agreed)


def figures(entries: Iterable[Entry]) -> Figures:
    """Weigh a set of entries.

    The median magnitude ratio is taken above the floor alone: below it the
    file states a number the measurement had no resolution for, and dividing
    two of those describes the noise rather than the model.
    """
    weighed = tuple(entries)
    checked = [entry for entry in weighed if entry.above_floor]
    return Figures(
        compared=len(weighed),
        passed=sum(1 for entry in weighed if entry.passed),
        checked=len(checked),
        agreed=sum(1 for entry in checked if entry.sign_agrees),
        median_ratio=_median(
            sorted(abs(entry.ratio) for entry in checked if math.isfinite(entry.ratio))
        ),
    )


@dataclass(frozen=True)
class PolarityColumn:
    """One corrector whose measured column pushes the beam the other way.

    Attributes:
        family: The actuator family the device belongs to.
        address: The channel the column was driven through.
        device: The 1-based device position of the column.
        flipped: How many entries of the column sit above the floor and
            disagree in sign.
        checked: How many entries of the column were compared, which is what
            ``flipped`` is a share of.
        median_ratio: The median magnitude ratio of the entries above the
            floor, which is what says the column is the right size and the
            wrong way round.
    """

    family: str
    address: str
    device: int
    flipped: int
    checked: int
    median_ratio: float


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
        actuator_deltas: The hardware sweep the model repeated, one width per
            column it compared.
        entries: Every compared entry.
        dropped: Every row that was not compared.
        monitor_plane: The transverse plane the monitor family reads, as the
            emitted bindings state it; empty where they state none.
        actuator_plane: The plane the actuator family kicks, the same way.
        unjudged: Why the block is reported rather than judged, in the words
            the report's table states; ``None`` where it is judged.
        polarity: The columns whose measured sign is the model's reversed,
            which a judged block states and leaves out of what it counts.
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
    actuator_deltas: tuple[float, ...]
    entries: tuple[Entry, ...]
    dropped: tuple[Dropped, ...]
    monitor_plane: str = ""
    actuator_plane: str = ""
    unjudged: str | None = None
    polarity: tuple[PolarityColumn, ...] = ()

    @property
    def judged(self) -> bool:
        """Whether this block's entries are what the verdict is made of.

        A block is judged where its monitors read the plane its correctors
        kick. Everything else is compared, printed and pooled into nothing.
        """
        return self.unjudged is None

    @property
    def counted(self) -> tuple[Entry, ...]:
        """Every entry bar the polarity columns: what a judged block contributes.

        A reversed cable is a fact about the machine the file was measured on,
        and no lattice, binding or calibration reproduces one. Left in, three
        such columns decide whether a whole plane passes its bar; named and
        left out, the bar is about the correctors whose polarity the two sides
        agree on, and the reversed ones are a list somebody can act on.
        """
        if not self.polarity:
            return self.entries
        aside = {column.device for column in self.polarity}
        return tuple(entry for entry in self.entries if entry.actuator_device not in aside)

    @property
    def counts(self) -> Figures:
        """What this block contributes to the verdict, polarity columns out."""
        return figures(self.counted)

    @property
    def full(self) -> Figures:
        """What the block holds, every compared entry of it."""
        return figures(self.entries)

    @property
    def delta_range(self) -> tuple[float, float] | None:
        """The smallest and largest width this block's columns were swept by.

        ``None`` where no column was swept, which is every block nothing was
        compared in: what the file states there was never asked of the model.
        """
        if not self.actuator_deltas:
            return None
        return (min(self.actuator_deltas), max(self.actuator_deltas))

    @property
    def compared(self) -> int:
        """How many entries were held against the file."""
        return len(self.entries)

    @property
    def passed(self) -> int:
        """How many of those are inside their band."""
        return self.full.passed

    @property
    def pass_ratio(self) -> float:
        """The share inside the band; ``nan`` when nothing was compared."""
        return self.full.pass_ratio

    @property
    def signed(self) -> tuple[int, int]:
        """How many entries sit above the floor, and how many of those agree in sign."""
        return self.full.signed

    @property
    def median_ratio(self) -> float:
        """The median model-over-file magnitude ratio above the floor.

        The number a measured matrix is read through: an export taken on the
        machine differs from any model entry by entry, and what says the two
        describe one accelerator is that the bulk of the ratios sit near one.
        """
        return self.full.median_ratio

    @property
    def outliers(self) -> tuple[Entry, ...]:
        """The entries outside their band, worst first."""
        failed = [entry for entry in self.entries if not entry.passed]
        failed.sort(key=lambda entry: entry.excess, reverse=True)
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
        trims: Every sampled conversion the emitter cut back to the stretch
            holding its device's operating point.
        supplies: Every supply the emitter found feeding a string of magnets.
        markers: Every repeated monitor name no family reads, which the served
            deck carries as plain markers.
        monitors: How many monitor-type elements the served deck is left with
            once those conversions are done.
        cavity: The cavity the emitter built into a deck that carried none,
            or ``None`` where the deck brought its own or needed none.
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
    trims: tuple[CalibrationTrim, ...] = ()
    supplies: tuple[SeriesSupply, ...] = ()
    markers: tuple[ServedMarker, ...] = ()
    monitors: int = 0
    cavity: BuiltCavity | None = None

    @property
    def judged(self) -> tuple[BlockReport, ...]:
        """The blocks whose monitors read the plane their correctors kick.

        The verdict is made of these alone. A cross-plane block is compared
        and printed like any other, and what it holds -- a deck's own coupling
        against a measured file's noise -- is not a statement about the served
        model that anything here could weigh.
        """
        return tuple(block for block in self.blocks if block.judged)

    @property
    def reported(self) -> tuple[BlockReport, ...]:
        """The blocks that were compared and pooled into nothing."""
        return tuple(block for block in self.blocks if not block.judged)

    @property
    def counted(self) -> tuple[Entry, ...]:
        """Every entry the verdict is made of: the judged blocks, less their
        polarity columns."""
        return tuple(entry for block in self.judged for entry in block.counted)

    @property
    def counts(self) -> Figures:
        """The verdict, as the five numbers every other table states."""
        return figures(self.counted)

    @property
    def polarity(self) -> tuple[tuple[BlockReport, PolarityColumn], ...]:
        """Every column set aside, with the block it was measured in."""
        return tuple((block, column) for block in self.judged for column in block.polarity)

    @property
    def compared(self) -> int:
        """Every entry the verdict counts, across every judged block."""
        return self.counts.compared

    @property
    def passed(self) -> int:
        """Every counted entry inside its band."""
        return self.counts.passed

    @property
    def pass_ratio(self) -> float:
        """The share inside the band; ``nan`` when nothing was compared."""
        return self.counts.pass_ratio

    @property
    def signed(self) -> tuple[int, int]:
        """Counted entries above the floor, and those of them that agree in sign."""
        return self.counts.signed

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
    trims: Sequence[CalibrationTrim] = (),
    supplies: Sequence[SeriesSupply] = (),
    markers: Sequence[ServedMarker] = (),
    monitors: int = 0,
    cavity: BuiltCavity | None = None,
) -> VerifyReport:
    """Re-measure every exported response block on the model the tree serves.

    One solve pass per actuator device, whatever the file states about it: the
    blocks that share an actuator family and a sweep width are driven together
    and every monitor they name is read off the same two arms, so a matrix of
    ``n`` correctors costs ``3n`` solves rather than three per block.

    The blocks are reported in the order the document writes them, whether or
    not each one could be re-measured, so a reader holding the report beside
    the export reads the two in one order. Every one of them is compared; the
    verdict is pooled over those whose monitors read the plane their
    correctors kick, and the band each entry is held to is a share of the
    scale of the whole file, so it is drawn once every block has been read.

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
        trims: What the bindings emitter cut back, for the report to name.
        supplies: The supplies feeding a string of magnets, for the same.
        markers: What the addressing pass served as plain markers, for the
            same.
        monitors: How many monitor-type elements the served deck is left with,
            so the report can say where the conversion took the last one.
        cavity: The cavity the emitter built into a cavity-less deck, for the
            report's Export section to say the model solves on one that the
            facility's own deck does not carry.

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
    by_family, shared = _bindings_by_device(document, grain, verdicts, system)
    model = PyATRingModel(data_dir, model_channels(document))

    swept = {
        key: _sweep(model, by_family, key, group, grain, judged_va)
        for key, group in _grouped(blocks, verdicts, by_family, system).items()
    }

    drafts: list[_Draft] = []
    for block in blocks:
        refusal = _undrivable(block, verdicts, by_family, system)
        measured, unsolved = ({}, {}) if refusal is not None else swept[_group_key(block)]
        drafts.append(
            _block_draft(block, measured, unsolved, grain, judged_va, by_family, shared, refusal)
        )

    # The band is a share of the matrix's own scale, so it can only be drawn
    # once every block of the file has been read.
    floor = FLOOR_FRACTION * _rms(entry.file_value for draft in drafts for entry in draft.entries)
    reports = [_banded(draft, floor) for draft in drafts]

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
        trims=tuple(trims),
        supplies=tuple(supplies),
        markers=tuple(markers),
        monitors=monitors,
        cavity=cavity,
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


def _deltas(block: dict) -> tuple[float, ...] | None:
    """The hardware sweep the file was measured with, one width per actuator device.

    A block stating one number swept every device of its actuator side by it.
    A block stating a vector swept each device by its own width, and the
    vector runs with the actuator's ``DeviceList``, so a vector of another
    length describes rows this cannot pair and is no sweep at all.

    Each entry is unwrapped the way a ``Status`` entry is: a facility
    exporting a column vector writes one-column rows, and a width read
    straight out of ``[4.5e-05]`` is no number at all -- which would leave the
    block stating no finite width and drop every column of it.

    Returns:
        One width per row of the actuator side, ``nan`` where the file states
        no number -- or ``None`` where a stated vector is the wrong length.
    """
    rows = len(_rows(block, "actuator"))
    stated = block.get("actuator_delta")
    values = list(stated) if isinstance(stated, list) else [stated] * rows
    if len(values) != rows:
        return None
    return tuple(
        math.nan if (width := _number(_first(value))) is None else width for value in values
    )


def _unswept(block: dict) -> dict[int, str]:
    """The actuator rows the file states no width for, and why each is out.

    The export fills the width of a device its own file does not hold with a
    value that is not a number, and usually marks the same row out in its
    ``Status``. Either way it is one column the file cannot be asked about,
    not a reason to refuse the rest of the block.
    """
    statuses = _statuses(block, "actuator")
    reasons: dict[int, str] = {}
    for index, width in enumerate(_deltas(block) or ()):
        if math.isfinite(width):
            continue
        status = _number(_first(statuses[index])) if index < len(statuses) else None
        flag = "" if status is None else f", which the file marks Status {_figure(status)}"
        reasons[index] = f"the block states no actuator_delta for this device{flag}"
    return reasons


# --- what the model can be asked ------------------------------------------


def _undrivable(
    block: dict,
    verdicts: Mapping[tuple[str, str], VAFamily],
    by_family: Mapping[str, dict[int, Binding]],
    system: str,
) -> str | None:
    """Why this block cannot be re-measured and compared, or ``None`` when it can be."""
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
    deltas = _deltas(block)
    if deltas is None:
        stated = block.get("actuator_delta")
        return (
            f"the block states {len(stated)} actuator_delta values for the "
            f"{len(_rows(block, 'actuator'))} devices of {family!r}, and each device is "
            "swept by its own"
        )
    if not any(math.isfinite(delta) for delta in deltas):
        return f"the block states no finite actuator_delta for {family!r} to be swept by"
    return _misshapen(block)


def _misshapen(block: dict) -> str | None:
    """Why the exported matrix cannot be read against the two device lists.

    An entry is read at ``data[monitor row][actuator row]``, so a matrix
    written against another pair of lists -- transposed where the two sides
    differ in length, or a row or a column short -- carries entries no device
    pair can be found for. Nothing downstream says so: an index past the data
    states no number, so the block would quietly compare the part that
    overlaps and report the rest as rows the bindings never reached.

    A block with no monitor row and no matrix is that same statement made of
    nothing, and it is named here rather than passed as well shaped: zero rows
    against zero devices agree, and a reader would otherwise have to work out
    from an empty comparison that there was never anything to compare.
    """
    monitors = len(_rows(block, "monitor"))
    actuators = len(_rows(block, "actuator"))
    data = block.get("data")
    rows = data if isinstance(data, list) else []
    widths = {len(row) if isinstance(row, list) else 0 for row in rows}
    if rows and monitors and len(rows) == monitors and widths <= {actuators}:
        return None
    if not rows:
        stated = "no matrix"
    elif len(widths) == 1:
        stated = f"a {len(rows)} by {next(iter(widths))} matrix"
    else:
        stated = f"{len(rows)} matrix rows of uneven length"
    return (
        f"the block states {stated} for the {monitors} devices of "
        f"{_family(block, 'monitor')!r} and the {actuators} of "
        f"{_family(block, 'actuator')!r}"
    )


def _grouped(
    blocks: Iterable[dict],
    verdicts: Mapping[tuple[str, str], VAFamily],
    by_family: Mapping[str, dict[int, Binding]],
    system: str,
) -> dict[tuple[str, tuple[float | None, ...]], list[dict]]:
    """The drivable blocks, grouped by the sweep one solve pass can serve.

    Two blocks share a pass when they drive the same family through the same
    hardware widths, device by device: the arms are then the same lattice
    states, and every monitor either block names is read off them. That is
    what lets one pass serve both the horizontal and the vertical block of one
    corrector family.
    """
    groups: dict[tuple[str, tuple[float | None, ...]], list[dict]] = {}
    for block in blocks:
        if _undrivable(block, verdicts, by_family, system) is not None:
            continue
        groups.setdefault(_group_key(block), []).append(block)
    return groups


def _group_key(block: dict) -> tuple[str, tuple[float | None, ...]]:
    """The sweep one drivable block is served by: its actuator family and widths."""
    deltas = _deltas(block) or ()
    return (
        _family(block, "actuator"),
        tuple(delta if math.isfinite(delta) else None for delta in deltas),
    )


def _sweep(
    model: PyATRingModel,
    by_family: Mapping[str, dict[int, Binding]],
    key: tuple[str, tuple[float | None, ...]],
    group: Sequence[dict],
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
) -> tuple[dict[int, dict[str, tuple[float, float]]], dict[int, str]]:
    """Drive one actuator family device by device, each by its own sweep width.

    A corrector whose arms leave the lattice without a stable closed orbit is
    a fact about that column, not a reason to abandon the rest of the matrix:
    the solver's refusal is kept against the device and every block that names
    it reports the column among the rows that were not compared.

    Args:
        model: The model the tree serves, driven and left where it was found.
        by_family: The emitted bindings, keyed by family and device position.
        key: The actuator family and the hardware width each of its devices is
            swept by.
        group: The blocks this one pass serves.
        grain: The judged family views, for the rows the file keeps.
        judged_va: Each family's judged ``va.json`` block.

    Returns:
        What each swept device's monitors read, and, for each device the
        solver refused, the reason the report states in its place.
    """
    family, _widths = key
    monitors = _monitors(group, by_family)
    bindings = by_family[family]
    measured: dict[int, dict[str, tuple[float, float]]] = {}
    unsolved: dict[int, str] = {}
    for device, delta in sorted(_driven(group, grain, judged_va, by_family).items()):
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
) -> dict[int, float]:
    """The actuator devices some block of this group needs swept, each with its width."""
    devices: dict[int, float] = {}
    for block in group:
        family = _family(block, "actuator")
        bindings = by_family.get(family, {})
        deltas = _deltas(block) or ()
        for row, device in _kept(block, "actuator", grain, judged_va).items():
            if device in bindings and row < len(deltas) and math.isfinite(deltas[row]):
                devices.setdefault(device, deltas[row])
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
) -> tuple[dict[str, dict[int, Binding]], dict[str, dict[int, str]]]:
    """The emitted bindings, keyed by family and judged device position.

    A binding carries the address it claims and not the device it was written
    for, and the emit lane skips a device whose element or address the export
    does not state -- so the bindings of a family are not one per device and
    cannot be indexed by position. The field the family is keyed by states
    which device each address sits at, and that is the one lookup that holds
    whatever the export left out.

    **An address several devices share binds none of them.** A supply feeding
    magnets in series is one knob that moves the whole string, while a
    response file's column is one magnet trimmed on its own: holding the two
    against each other says the model moved every magnet of the string where
    the machine moved one. Taking the first device would do exactly that and
    say nothing, while the rest of the string reported that the bindings drive
    nothing at those rows. So every device of a shared address is a row that
    was not compared, carrying the reason it was not.

    Returns:
        The bindings by family and device, and per family the devices no
        binding was taken for, each with the reason the report states.
    """
    by_family: dict[str, dict[int, Binding]] = {}
    shared: dict[str, dict[int, str]] = {}
    for binding in document.bindings:
        by_family.setdefault(binding.family, {})
    for family, bindings in by_family.items():
        verdict = verdicts.get((system, family))
        written = (
            "Monitor" if verdict is not None and str(verdict.kind) == "monitor" else "Setpoint"
        )
        view = grain.get(family)
        devices = _devices_by_address(None if view is None else view.fields.get(written))
        for binding in document.bindings:
            if binding.family != family:
                continue
            at = devices.get(binding.setpoint_address, ())
            if len(at) == 1:
                bindings[at[0]] = binding
                continue
            for device in at:
                shared.setdefault(family, {})[device] = _shared_reason(binding, len(at))
    return by_family, shared


def _shared_reason(binding: Binding, devices: int) -> str:
    """Why one device of an address several of them share was not compared."""
    if binding.is_writable:
        others = devices - 1
        return (
            f"{binding.setpoint_address} feeds this magnet in series with {others} "
            f"other{'' if others == 1 else 's'} of {binding.family!r}: the model moves "
            "the whole string at once and the file measured one magnet"
        )
    return (
        f"{binding.setpoint_address} is read by {devices} devices of "
        f"{binding.family!r}: the model publishes one reading and the file measured "
        "one row"
    )


def _devices_by_address(field_view: FieldView | None) -> dict[str, tuple[int, ...]]:
    """Every 0-based device position each address of one field sits at."""
    if field_view is None:
        return {}
    found: dict[str, list[int]] = {}
    for key in field_view.keys:
        for index, slot in enumerate(field_view.slots(key)):
            if isinstance(slot, str) and slot.strip():
                at = found.setdefault(slot.strip(), [])
                if index not in at:
                    at.append(index)
    return {address: tuple(at) for address, at in found.items()}


# --- the comparison --------------------------------------------------------


@dataclass(frozen=True)
class _Draft:
    """One block, paired but not yet banded.

    Every entry is held to a share of the whole matrix's scale, so no band can
    be drawn until each block of the file has been read. A draft carries what
    one block found, and :func:`_banded` turns it into the block the report
    states once that scale is known.

    Attributes:
        block: The exported block it was read from.
        entries: Every pair the comparison made, banded against nothing yet.
        dropped: Every row that was not compared.
        swept: The hardware widths the compared columns were driven by.
        monitor_plane: The plane the monitor family reads, empty where the
            bindings place it in none.
        actuator_plane: The plane the actuator family kicks, the same.
        unjudged: Why the block is reported rather than judged, ``None`` where
            it is judged.
    """

    block: dict
    entries: tuple[Entry, ...]
    dropped: tuple[Dropped, ...]
    swept: tuple[float, ...]
    monitor_plane: str
    actuator_plane: str
    unjudged: str | None


def _banded(draft: _Draft, floor: float) -> BlockReport:
    """One drafted block, with every entry held to the matrix's own floor.

    The polarity columns are found here rather than in the draft because a
    sign only means something above the floor, and the floor is the whole
    file's. Only a judged block has them: a cross-plane column has no polarity
    to be wrong about.
    """
    block = draft.block
    entries = tuple(
        replace(
            entry,
            floor=floor,
            tolerance=TOLERANCE_FRACTION * max(abs(entry.file_value), floor),
        )
        for entry in draft.entries
    )
    return BlockReport(
        monitor_family=_family(block, "monitor"),
        actuator_family=_family(block, "actuator"),
        monitor_mode=_word(_side(block, "monitor").get("mode")),
        actuator_mode=_word(_side(block, "actuator").get("mode")),
        origin=_word(block.get("origin")),
        timestamp=_word(block.get("timestamp")),
        gev=_number(block.get("gev")),
        units=_word(block.get("units")),
        units_string=_word(block.get("units_string")),
        modulation_method=_word(block.get("modulation_method")),
        actuator_deltas=draft.swept,
        entries=entries,
        dropped=draft.dropped,
        monitor_plane=draft.monitor_plane,
        actuator_plane=draft.actuator_plane,
        unjudged=draft.unjudged,
        polarity=()
        if draft.unjudged is not None
        else _polarity(entries, _family(block, "actuator")),
    )


def _polarity(entries: Sequence[Entry], family: str) -> tuple[PolarityColumn, ...]:
    """The columns the file and the model disagree with about which way is which.

    One corrector at a time. A sign only means something above the floor, so
    only entries above it are counted flipped; they are counted against the
    whole column, every entry that was compared, because a device the two
    sides disagree about pushes the beam the wrong way all down its column.
    Counting them against the entries above the floor alone would let a column
    holding one sizable entry retire itself and its small entries with it: a
    scatter of flipped entries is the comparison finding something in the
    lattice, and one flipped entry is no scatter at all.
    """
    columns: dict[int, list[Entry]] = {}
    for entry in entries:
        columns.setdefault(entry.actuator_device, []).append(entry)
    found: list[PolarityColumn] = []
    for device, column in sorted(columns.items()):
        above = [entry for entry in column if entry.above_floor]
        flipped = sum(1 for entry in above if entry.sign_agrees is False)
        if flipped <= POLARITY_SHARE * len(column):
            continue
        found.append(
            PolarityColumn(
                family=family,
                address=column[0].actuator_address,
                device=device,
                flipped=flipped,
                checked=len(column),
                median_ratio=figures(above).median_ratio,
            )
        )
    return tuple(found)


def _block_draft(
    block: dict,
    measured: Mapping[int, dict[str, tuple[float, float]]],
    unsolved: Mapping[int, str],
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
    by_family: Mapping[str, dict[int, Binding]],
    shared: Mapping[str, dict[int, str]],
    refusal: str | None,
) -> _Draft:
    """Hold one block's matrix against what the model did, entry by entry."""
    monitor_family = _family(block, "monitor")
    actuator_family = _family(block, "actuator")
    monitors = by_family.get(monitor_family, {})
    actuators = by_family.get(actuator_family, {})
    data = block.get("data")
    data = data if isinstance(data, list) else []

    dropped: list[Dropped] = []
    rows = _usable(
        block,
        "monitor",
        grain,
        judged_va,
        monitors,
        dropped,
        None,
        "the bindings read no",
        shared=shared.get(monitor_family, {}),
    )
    columns = _usable(
        block,
        "actuator",
        grain,
        judged_va,
        actuators,
        dropped,
        refusal,
        "the bindings drive no",
        unswept=_unswept(block),
        shared=shared.get(actuator_family, {}),
    )

    deltas = _deltas(block) or ()
    swept: list[float] = []
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
        if column < len(deltas):
            swept.append(deltas[column])
        stated = {
            row: value
            for row, device in rows.items()
            if (value := _matrix(data, row, column)) is not None
        }
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
                    tolerance=0.0,
                    floor=0.0,
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

    monitor_plane, monitor_reason = _family_plane(monitor_family, monitors)
    actuator_plane, actuator_reason = _family_plane(actuator_family, actuators)
    return _Draft(
        block=block,
        entries=tuple(entries),
        dropped=tuple(dropped),
        swept=tuple(swept),
        monitor_plane=monitor_plane,
        actuator_plane=actuator_plane,
        unjudged=monitor_reason
        or actuator_reason
        or (None if monitor_plane == actuator_plane else "cross-plane"),
    )


def _plane(binding: Binding) -> str | None:
    """The transverse plane one binding works in, or ``None`` where it has none.

    A monitor states the axis it reads as its attribute, and a kick states the
    component of its attribute it writes -- the two components being the two
    planes, in the order a reading pairs them. Both are the bindings' own
    vocabulary, so the plane of a block is read off the emitted file and never
    off what either family is called: a facility names its horizontal
    correctors whatever it likes, and a rule keyed on the name would judge the
    next machine's blocks against the wrong half of its matrix.
    """
    if binding.kind == "monitor":
        axis = (binding.attribute or "").lower()
        return axis if axis in _PLANES else None
    index = binding.index
    if binding.kind == ACTUATOR_KIND and isinstance(index, int) and not isinstance(index, bool):
        return _PLANES[index] if 0 <= index < len(_PLANES) else None
    return None


def _family_plane(family: str, bindings: Mapping[int, Binding]) -> tuple[str, str | None]:
    """The plane one family works in, or why the bindings place it in none.

    A family whose bindings name no plane, or more than one, is a block this
    cannot put on either side of the comparison, so it is reported with that
    reason rather than judged as though its plane were the other side's.
    """
    planes = sorted({plane for binding in bindings.values() if (plane := _plane(binding))})
    if len(planes) == 1:
        return planes[0], None
    if not planes:
        return "", f"the bindings place {family!r} in no plane"
    return "", f"the bindings place {family!r} in {' and '.join(planes)}"


def _usable(
    block: dict,
    side: str,
    grain: Mapping[str, FamilyView],
    judged_va: Mapping[tuple[str, str], dict],
    bindings: Mapping[int, Binding],
    dropped: list[Dropped],
    refusal: str | None,
    unbound: str,
    *,
    unswept: Mapping[int, str] | None = None,
    shared: Mapping[int, str] | None = None,
) -> dict[int, int]:
    """One side's rows that reach a binding, recording why each other one did not.

    A device whose address several of them share has a reason of its own, and
    it is stated ahead of the plain unbound line: "the bindings drive no
    device at this row" is true of it and says the wrong thing, because what
    happened is that the binding drives several rows at once.
    """
    family = _family(block, side)
    rows = _rows(block, side)
    usable: dict[int, int] = {}
    for index, (device, reason) in _alignment(block, side, grain, judged_va).items():
        if refusal is not None:
            reason = refusal
        elif reason is None and device not in bindings:
            reason = (shared or {}).get(device) or f"{unbound} device at this row"
        elif reason is None and unswept is not None:
            reason = unswept.get(index)
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
    """The root mean square of every compared entry, which sets the floor."""
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
    lines += _polarity_section(report)
    lines += _dropped_section(report)
    lines += _bands_section(report)
    lines += _trims_section(report)
    lines += _supplies_section(report)
    lines += _markers_section(report)
    lines += _nominals_section(report)
    return "\n".join(lines).rstrip() + "\n"


def _headline(report: VerifyReport) -> list[str]:
    """The one paragraph that says how the comparison went."""
    checked, agreed = report.signed
    lines = ["## Verdict", ""]
    if not report.compared:
        lines += _nothing_judged(report)
        return lines
    lines += [
        f"{report.passed} of {report.compared} entries "
        f"({_percent(report.pass_ratio)}) are inside the band "
        f"`|R_model - R_file| <= {TOLERANCE_FRACTION} * "
        f"max(|R_file|, {FLOOR_FRACTION} * rms(matrix))`.",
        "",
        "`rms(matrix)` is the root mean square of every entry of the file that was "
        "compared, in every block -- so the floor is the scale of the whole matrix "
        "as the export states it, and the model has no say in the tolerance it is "
        "held to. Drawn per column instead, a column the file states as zero "
        "throughout would set its own floor to zero and hold the model to a band of "
        "nothing; against the matrix it is asked to stay small, which is what such "
        "a column says.",
        "",
        _judged_line(report),
        "",
        *_polarity_line(report),
        f"Sign agrees on {agreed} of the {checked} entries above the floor, "
        "counted in the judged blocks alone."
        if checked
        else "No judged entry sits above the matrix floor, so no sign was asserted.",
        "",
    ]
    return lines


def _nothing_judged(report: VerifyReport) -> list[str]:
    """The headline of a run with no judged entry to pool.

    Two different things end here. A run that compared nothing at all has the
    sections below to explain itself. A run that compared whole blocks and
    judged none of them has a page full of numbers that answer a different
    question, and saying "nothing was compared" over the top of them would
    read as a contradiction rather than a verdict.
    """
    entries = sum(block.compared for block in report.reported)
    if not entries:
        return [
            "Not one entry of the exported response matrix was compared. The sections "
            "below say why; until they are answered this deployment has no evidence "
            "that its virtual accelerator answers like the machine.",
            "",
        ]
    return [
        f"No block of the exported response matrix is judged, so none of the "
        f"{entries} entries below carries a verdict: in every block the monitors "
        "read a plane its correctors do not kick, and what such a block holds is "
        "the deck's own coupling rather than a statement about the served model. "
        "Until a block pairs one plane with itself this deployment has no evidence "
        "that its virtual accelerator answers like the machine.",
        "",
    ]


def _polarity_line(report: VerifyReport) -> list[str]:
    """What the counts leave out, where a corrector's column runs backwards."""
    columns = report.polarity
    if not columns:
        return []
    entries = sum(block.compared - block.counts.compared for block in report.judged)
    subject = "corrector" if len(columns) == 1 else "correctors"
    return [
        f"{len(columns)} {subject} whose column the file and the model disagree "
        f"about the direction of are named under Polarity outliers, and their "
        f"{entries} entries are not in the counts above.",
        "",
    ]


def _judged_line(report: VerifyReport) -> str:
    """Which blocks the counts above are made of, and which are only printed."""
    reported = report.reported
    if not reported:
        return (
            "Every block is judged: in each one the monitors read the plane their "
            "correctors kick, which the emitted bindings state and no family name "
            "decides."
        )
    named = ", ".join(f"{_block_name(block)} ({block.unjudged})" for block in reported)
    return (
        "The counts above are the judged blocks alone -- those whose monitors read "
        "the plane their correctors kick, which the emitted bindings state and no "
        f"family name decides. {named} {'are' if len(reported) > 1 else 'is'} "
        "compared with the same numbers in the table below and pooled into nothing."
    )


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
        *_cavity_row(report.cavity),
        f"| energy at nominal | {_figure(report.energy_at_nominal)} GeV |",
        f"| read from | {report.response_file or 'no file: the model was measured'} |",
        f"| exported | {report.response_timestamp or 'unstated'} |",
        "",
    ]
    lines += _built_cavity_lines(report.cavity)
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


def _cavity_row(cavity: BuiltCavity | None) -> list[str]:
    """The ``Export`` row for a cavity the facility's deck does not carry.

    Three numbers, because the two frequencies are not the same statement: the
    cavity is built at the frequency a whole number of waves fits around this
    deck, while the export states the one the facility quotes its machine at,
    and a reader comparing them is reading how far the deck sits from the
    ring. They are printed to the hertz, as the emit line prints them: the gap
    between them is the last few figures, and rounded to the report's usual
    six they would read as one number twice over. The voltage is the
    reviewer's answer and moves no orbit.
    """
    if cavity is None:
        return []
    built = (
        f"{_hertz(cavity.frequency_hz)} Hz on harmonic {cavity.harmonic}"
        if cavity.frequency_hz is not None
        else f"harmonic {cavity.harmonic}, not yet built against a deck"
    )
    volts = "an unanswered voltage" if cavity.voltage is None else f"{_figure(cavity.voltage)} V"
    return [
        f"| built cavity | {cavity.family} at {built}, {volts}; "
        f"the export states {_hertz(cavity.nominal_hz)} Hz |"
    ]


def _built_cavity_lines(cavity: BuiltCavity | None) -> list[str]:
    """Why the served ring holds an element the facility's deck does not.

    Without a cavity the model solves at a fixed energy, and a machine that
    holds its radio frequency moves in dispersion in a way only a cavity makes
    a model move -- which is most of the entries above.
    """
    if cavity is None:
        return []
    return [
        f"The deck carries no cavity of its own, so one was built onto it for the "
        f"{cavity.family} family to drive, and the model solves on it. Without a "
        "cavity the model holds the energy fixed and answers nothing about what a "
        "held frequency does to the orbit wherever there is dispersion.",
        "",
    ]


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
        "| monitors | actuators | compared | inside the band | sign above floor | "
        "median ratio | swept by | judged |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for block in report.blocks:
        checked, agreed = block.signed
        lines.append(
            f"| {block.monitor_family} | {block.actuator_family} | {block.compared} | "
            f"{block.passed} ({_percent(block.pass_ratio)}) | {agreed}/{checked} | "
            f"{_figure(block.median_ratio)} | {_swept(block)} | {_judged(block)} |"
        )
    lines.append("")
    if report.reported:
        lines += [
            "Every block is compared and printed the same way; the judged ones are "
            "those whose monitors read the plane their correctors kick, which the "
            "emitted bindings state. A cross-plane block holds the deck's own "
            "coupling against whatever the file has there -- exactly zero in a "
            "model-derived export, its own noise in a measured one -- so it is "
            "reported and pooled into nothing.",
            "",
        ]
    lines += _below_floor_lines(report)
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


def _below_floor_lines(report: VerifyReport) -> list[str]:
    """One sentence per judged block holding no entry above the matrix floor.

    The floor is the whole file's, so a judged block whose plane is an order
    of magnitude smaller than the rest of the matrix can hold nothing above
    it. Every entry of such a block is then inside a band that is the floor,
    which says the model is small there and nothing at all about its size or
    its direction -- and the row states that as a ``0/0`` and a dash, which
    are easy to read as "fine". So it is said in words.
    """
    named = [block for block in report.judged if not block.counts.checked]
    if not named:
        return []
    return [
        *(
            f"No entry of {_block_name(block)} sits above the matrix floor, so the "
            "block is held to the band alone: its sign and its median ratio state "
            "nothing, and being inside a band that is the floor is being small "
            "rather than being right."
            for block in named
        ),
        "",
    ]


def _judged(block: BlockReport) -> str:
    """Whether one block's numbers carry a verdict, and why where they do not.

    The row's numbers are the whole block's either way. Where a column was set
    aside for its polarity the cell says how many, and the section below
    states what the block is judged on without them.
    """
    if not block.judged:
        return f"reported only, {block.unjudged}"
    aside = len(block.polarity)
    if not aside:
        return "yes"
    return f"yes, less {aside} polarity column{'' if aside == 1 else 's'}"


def _swept(block: BlockReport) -> str:
    """The hardware widths one block's columns were driven by.

    A file states a width per corrector, so a block is read by the range its
    columns cover: one figure where every column was swept by the same width,
    and the two ends where they differ.
    """
    span = block.delta_range
    if span is None:
        return "—"
    low, high = span
    return _figure(low) if low == high else f"{_figure(low)} to {_figure(high)}"


#: How many polarity columns the report lists before it counts the rest.
POLARITY_LISTED = 12


def _polarity_section(report: VerifyReport) -> list[str]:
    """The correctors the file and the model disagree about the direction of.

    Named rather than absorbed: the counts above are about a deck, and a
    reversed cable is about the machine. The section carries what a person
    needs to go and look -- which channel, how much of its column, and that
    the size is right -- and then restates each judged block without them, so
    the bar and the list are read off one page.
    """
    lines = ["## Polarity outliers", ""]
    columns = report.polarity
    if not columns:
        lines += ["Every corrector moves the beam the way the file says it did.", ""]
        return lines
    lines += [
        "The file has these correctors moving the beam the opposite way to the "
        "model on more than half of the entries of their column, counting only "
        "the entries big enough for a sign to mean anything, and at the size "
        "the rest of the matrix is worth. Nothing in a lattice, a binding or a "
        "calibration "
        "reproduces a reversed device, so each column is named here and left out "
        "of the counts above. The served model keeps the facility's own "
        "conversion, which is what states the direction the model drives them in.",
        "",
        "| monitors ← actuators | family | address | flipped | median ratio |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {_block_name(block)} | {column.family} | {column.address} | "
        f"{column.flipped}/{column.checked} | {_figure(column.median_ratio)} |"
        for block, column in columns[:POLARITY_LISTED]
    ]
    if len(columns) > POLARITY_LISTED:
        lines.append(f"| … | and {len(columns) - POLARITY_LISTED} more | | | |")
    lines += [
        "",
        "Left out, the judged blocks stand at:",
        "",
        "| monitors | actuators | compared | inside the band | sign above floor | median ratio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for block in report.judged:
        counts = block.counts
        checked, agreed = counts.signed
        lines.append(
            f"| {block.monitor_family} | {block.actuator_family} | {counts.compared} | "
            f"{counts.passed} ({_percent(counts.pass_ratio)}) | {agreed}/{checked} | "
            f"{_figure(counts.median_ratio)} |"
        )
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


#: How many trimmed conversions the report lists before it counts the rest.
TRIMS_LISTED = 12


def _trims_section(report: VerifyReport) -> list[str]:
    """The sampled conversions that kept one stretch of themselves."""
    lines = ["## Trimmed conversions", ""]
    trims = report.trims
    if not trims:
        lines += ["Every sampled conversion converts one way over its whole grid.", ""]
        return lines
    lines += [
        "The facility's own function turns back inside the band these were sampled "
        "over, so one reading would answer two hardware values. Each kept the "
        "stretch holding its device's operating point; outside that stretch the "
        "model continues along the end segment. Check the operating point against "
        "the span before the deployment writes near either end.",
        "",
        "| address | family | conversion | kept | operating point | points dropped |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {trim.address} | {trim.family} | {trim.curve} | "
        f"{_figure(trim.kept[0])} to {_figure(trim.kept[1])} | "
        f"{_figure(trim.working)} | {trim.dropped} |"
        for trim in trims[:TRIMS_LISTED]
    ]
    if len(trims) > TRIMS_LISTED:
        lines.append(f"| … | and {len(trims) - TRIMS_LISTED} more | | | | |")
    lines.append("")
    return lines


#: How many series supplies the report lists before it counts the rest.
SUPPLIES_LISTED = 12


def _supplies_section(report: VerifyReport) -> list[str]:
    """The supplies that feed several magnets, and how far apart they sit."""
    lines = ["## Series supplies", ""]
    supplies = report.supplies
    if not supplies:
        lines += ["Every supply feeds one magnet.", ""]
        return lines
    lines += [
        "These supplies feed their magnets in series, so the model drives each "
        "string with one knob and every magnet on it carries a fixed share. The "
        "shares are chosen at the starting current, where each magnet sits "
        "exactly where the deck puts it; the wider the spread, the further the "
        "string drifts from the deck as the knob is moved away from that current.",
        "",
        "| address | family | magnets | start | spread | of start |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {supply.address} | {supply.family} | {supply.magnets} | "
        f"{_figure(supply.start)} | {_figure(supply.spread)} | "
        f"{_percent(supply.relative)} |"
        for supply in supplies[:SUPPLIES_LISTED]
    ]
    if len(supplies) > SUPPLIES_LISTED:
        lines.append(f"| … | and {len(supplies) - SUPPLIES_LISTED} more | | | | |")
    lines.append("")
    return lines


#: How many marker names the report lists before it counts the rest.
MARKERS_LISTED = 12


def _markers_section(report: VerifyReport) -> list[str]:
    """The repeated monitor names the served deck carries as plain markers.

    Where the conversion takes the last monitor the section says so: the
    counts alone do not, since reading that off them means knowing how many
    monitors the deck held.
    """
    lines = ["## Monitors served as markers", ""]
    markers = report.markers
    if not markers:
        lines += ["Every monitor-type element on the deck is read or uniquely named.", ""]
        return lines
    lines += [
        "The deck gives each of these names to several monitor-type elements and "
        "no family reads them, so the model carries them as plain markers of the "
        "same name in the same place -- the position is still there, with nothing "
        "to read off it. Bind a family to one of these names and its elements come "
        "back as monitors, named for their devices.",
        "",
        "| name | elements |",
        "| --- | --- |",
    ]
    lines += [f"| {marker.name} | {marker.elements} |" for marker in markers[:MARKERS_LISTED]]
    if len(markers) > MARKERS_LISTED:
        lines.append(f"| … | and {len(markers) - MARKERS_LISTED} more |")
    lines.append("")
    if not report.monitors:
        lines += [
            f"The conversion leaves {report.system} with no monitor-type element at "
            "all, so the served system reads no beam position anywhere and every "
            "orbit answer below is measured on a model that publishes none.",
            "",
        ]
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


def _hertz(value: float) -> str:
    """A frequency to the hertz, which is where two of them start to differ."""
    return f"{value:.0f}"


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
