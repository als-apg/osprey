"""The pyat engine's LUME model: one deck, the variables its wiring builds.

Everything generic about serving a pyAT lattice through the LUME contract --
owning one persistent lattice, atomic multi-variable writes, one solve per
batch, rollback on a lost closed orbit, retained inputs and cached outputs --
belongs to :class:`~lume_pyat.model.LUMEPyATModel` and is inherited. What
:class:`PyATLatticeModel` adds is read off the channel variables it is handed,
each built from one wiring record by
:mod:`~osprey.simulation.engines.pyat_variables`:

- the fault state every wired monitor and magnet can carry,
- the energy knob's coupling to the rigidity-scaled setpoints,
- the optics arrays of the solved lattice, computed on read,
- and the wired readbacks it serves from those: a :class:`ReadbackVariable`
  reads a setpoint back, or one plane of the tunes or the chromaticity.

**Faults are variables, held as state on the element.** Each is named
``<address>/<field>`` after the wired address it perturbs, so a fault name is
never a channel address and the two rosters cannot collide. A monitor readback
carries ``offset``, ``gain``, ``noise`` and ``polarity`` on its own axis, and a
monitor carries one ``roll``, on its x-axis readback or its only readback. A
setpoint on ``PolynomB`` or ``KickAngle`` carries ``cal_factor`` and
``cal_offset``. Each fault is bound to an element attribute no pyAT pass
method reads (``readout_<field>_<axis>``, ``readout_roll``, and a setpoint's
:func:`~osprey.simulation.engines.pyat_faults.supply_attribute` pair), so
seeding or writing a fault changes what the element carries and never the
orbit; applying it to a
reading or to a commanded value is the readout's work. A seed is its
variable's ``default_value`` and is written onto the element from that same
value, so the two cannot disagree, and :meth:`reset` returns a fault to its
seed rather than to identity. Seeds are checked by name, then by bound, before
the lattice is touched.

**Bounds say what a device can be.** A gain outside its window, a roll beyond
its angle, a calibration factor beyond its multiple names no instrument the
model could stand in for; a polarity is a sign and lands on one of its two
values; a noise width is a standard deviation and is never negative. A
displacement, a noise width and a calibration offset are magnitudes asked for
in the facility's own unit, and carry no other bound.

**Each setpoint carries its own calibration.** A calibration is the supply's,
so it sits on the first element its setpoint binds, under attributes named
for that setpoint: a family supply and a trim on one of its magnets, or the
two planes of one corrector, are each miscalibrated alone, and a fault on one
never rescales what another delivers.

**A model with an injection tolerates a lost closed orbit.** A ``periodic``
model handed an ``injection`` is solved by
:class:`~osprey.simulation.engines.pyat_tolerant.TolerantPeriodicSimulator`:
a write that leaves the lattice without a closed orbit commits, every monitor
reading, optics array and chromaticity reads NaN, and
:attr:`PyATLatticeModel.no_closed_orbit` is true until a later solve finds a
closed orbit again. A ``periodic`` model without one refuses that write and
rolls it back; a ``single_pass`` model refuses a lost particle either way.

**Optics are read-only arrays, computed on read.** ``tunes`` (the fractional
tunes: the two transverse ones, and the synchrotron tune third on a deck with
longitudinal motion), ``chromaticity`` (one per tune), ``beta_at_monitors``
and ``orbit_at_monitors`` (one ``(x, y)`` row per monitor element, in lattice
order) sit beside the per-monitor readings. A ``single_pass`` model solves a
line by one tracked pass from its ``twiss_in``
(:class:`~osprey.simulation.engines.pyat_single_pass.SinglePassSimulator`); a
line has no tunes, so it declares neither ``tunes`` nor ``chromaticity``, and
its orbit and beta functions are the ones that pass solved. The base class re-reads every
read-only output after every solve, so a setpoint write would pay for an
optics pass nobody asked for; these are left out of that re-read. The linear
ones are computed together the first time one is read after a solve, then
served from memory until the next. The chromaticity needs a chromatic solve,
far dearer than the linear pass, so it has a memo of its own and is solved
only for a read that names it.

**A shot is a command, and commits with its batch.** A model wires the shot
as records naming no element: any number of ``arm`` commands, which all set
one armed flag; ``inject`` requests; ``armed`` readings of that flag; and
turn series, each one coordinate of one monitor per tracked sample. A write
of 1 to an ``arm`` command arms. A write that raises a request above the
last value the model saw is one shot: the shot count advances, and if the
flag is set the shot is tracked from the injection through the lattice
as the batch's setpoints left it, every series is refilled, and the flag is
cleared; unarmed, every series keeps what it held. The track runs after the
batch's solve, and the series, the flag, the count and the last request
seen change only when the batch commits, so a refused batch leaves all four
alone. A write of a request's default is never a shot, and neither is
:meth:`PyATLatticeModel.reset`, which returns all four to rest.

**A shot is one particle, or a bunch.** Without a bunch a shot tracks one
particle from the launch. A scenario that states ``bunch_particles``
declares a bunch: every shot tracks that many particles in one pass, the
launch plus a fixed Gaussian shape -- the scenario's emittances, momentum
spread and length, matched to the Twiss parameters and dispersion of a 4D
copy of the deck at the injection marker, solved once when the model is
built -- plus a centroid jitter in ``x`` and ``y``. The jitter is the only
randomness: a keyed draw of the model's id and the shot count, so two
models built alike, a restart and a reset replay the same shots. A sample's
physics sum is then the share of the bunch alive there and its positions
the survivors' centroid. A one-particle bunch with no jitter is the single
particle.

**A bunch's shots accumulate a stored charge.** A bunch injects the
scenario's ``bunch_charge`` (1 when unstated), and the model holds a stored
charge, the scenario's ``stored_charge`` (0 when unstated). Every shot,
tracked or not, is one injection step. With a closed orbit the particles
alive at the end of the tracked window are captured: their share of the
bunch charge adds to the stored charge. With no closed orbit nothing is
captured. When every cavity is off, each shot leaves the surviving fraction
of the stored charge, and a charge at or below the floor is gone; both are
data on the model's stored-current reading, which loses everything to one
shot when the model wires none. A batch that loses the closed orbit with RF
on loses the stored charge at once. A ``clear_stored_charge`` command
written anything but its default empties it, and :meth:`PyATLatticeModel.reset`
restores the scenario's. A stored-current reading serves the charge times
the deck's revolution frequency. While the lattice has a closed orbit the
monitors see the stored beam: a sample's physics sum is ``f + Q_s/Q_i``, the
share of the bunch alive there plus the stored charge over the bunch charge,
and its positions are the charge-weighted centroid of the survivors and of
the stored beam on the closed orbit as solved, before any readout fault, so
the monitor applies its faults once. With no closed orbit the stored beam
adds nothing to a series. The single particle has no stored charge in its
sums.

**A series holds one launch, and every sample is a reading.** A periodic
shot tracks the model's *turns tracked*, one model-wide global setting: the
turns after the trigger sample that carry beam, whole turns from 1 to the
samples a series holds after the trigger. A model that wires no setting for
it, or whose setting states no default, tracks
:data:`~osprey.simulation.engines.pyat.TURNS_TRACKED_DEFAULT` turns up to that
bound; a single-pass model tracks one pass. A tracked
sample is lost when its ``x`` is NaN, and a bunch's sample when every
particle is. The samples before the trigger, every sample past the tracked
turns, and every lost one carry no injected beam; with no stored beam either
their physics sum is 0, which the monitor reads at the floor
(:data:`~osprey.simulation.engines.pyat_faults.SUM_FLOOR`), and their
positions read as a beam on axis under the floor's noise. A sample the
single particle passed has a physics sum of 1. No sample is NaN; every one passes its
record's calibration.

**A series is read by its monitor.** A tracked position takes the readout
faults its monitor's element carries, the ones the monitor's orbit reading
takes, at the same physical size whatever unit the series publishes
(:func:`~osprey.simulation.engines.pyat_faults.shot_read`), and the
electronics' readout attributes the element carries -- a position noise over
the sum, a tone -- act beneath them. ``sum`` takes no readout fault: only the
floor, the tone, the signal gain and the sum scale
(:func:`~osprey.simulation.engines.pyat_faults.shot_sum`). The noise is
drawn once, when the shot is tracked, per series, shot count and sample, so a
series reads the same until the next shot refills it and a fault written
afterwards reaches the next shot only.

Declared defaults are recorded as the retained input values at construction
but are not written to the lattice, so the lattice boots exactly as handed
over; :meth:`reset` writes them back through the calibrations. This module
never touches a control system and never raises ``SystemExit``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import at
import numpy as np
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ConfigEnum, ScalarVariable
from lume_pyat.actions import ElementBinding, PyATWritableScalarVariable, SnapshotTarget
from lume_pyat.exceptions import UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator
from pydantic import ConfigDict, model_validator

from osprey.simulation.engines.calibration import (
    Calibration,
    check_inverse,
    curve_from_record,
)
from osprey.simulation.engines.calibration import field as slot
from osprey.simulation.engines.pyat import SOLVES, turns_tracked_default
from osprey.simulation.engines.pyat_faults import (
    BUNCH_CHARGE,
    BUNCH_PARTICLES,
    BUNCH_SHAPE_FAULTS,
    CALIBRATED_ATTRIBUTES,
    CENTROID_JITTER_FAULTS,
    FAULT_BOUNDS,
    FAULT_SEPARATOR,
    POLARITY_OPTIONS,
    READOUT_ATTRIBUTE_IDENTITY,
    READOUT_IDENTITY,
    ROLL,
    ROLL_IDENTITY,
    STORED_CHARGE,
    SUPPLY_IDENTITY,
    bunch_key,
    samples_to_hardware,
    shot_read,
    shot_sum,
    supply_attribute,
)
from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator
from osprey.simulation.engines.pyat_tolerant import TolerantPeriodicSimulator
from osprey.simulation.engines.pyat_variables import (
    CalibratedSetpoint,
    EnergyVariable,
    MonitorVariable,
    PyATReadOnlyNDVariable,
    PyATWritableEnumVariable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.actions import ActionVariable
    from lume.variables import Variable

__all__ = [
    "BETA_AT_MONITORS",
    "CHROMATICITY",
    "FAULT_SEPARATOR",
    "OPTICS_NAMES",
    "ORBIT_AT_MONITORS",
    "TUNES",
    "ArmedVariable",
    "ArrayReadbackVariable",
    "CommandVariable",
    "PyATLatticeModel",
    "ReadbackVariable",
    "SeriesVariable",
    "StoredCurrentVariable",
    "TurnsTrackedVariable",
]

# Element-attribute prefix of a readout fault. No pyAT pass method reads an
# attribute under it, which is what keeps a fault off the orbit.
_READOUT_PREFIX = "readout_"

_ROLL_UNIT = "rad"

#: The readout fields whose magnitude is in the monitor reading's own unit.
_FIELDS_IN_READING_UNIT = frozenset({"offset", "noise"})

# The optics arrays, by name. No separator and no colon, so none of them
# parses as a fault name or as an address.
TUNES = "tunes"
CHROMATICITY = "chromaticity"
BETA_AT_MONITORS = "beta_at_monitors"
ORBIT_AT_MONITORS = "orbit_at_monitors"
#: The optics one linear pass computes together.
LINEAR_OPTICS: frozenset[str] = frozenset({TUNES, BETA_AT_MONITORS, ORBIT_AT_MONITORS})
OPTICS_NAMES: frozenset[str] = LINEAR_OPTICS | {CHROMATICITY}
#: The optics arrays a readback reads one plane of.
PLANE_OPTICS: frozenset[str] = frozenset({TUNES, CHROMATICITY})

#: The two solves: a closed orbit, and one tracked pass through a line.
PERIODIC, SINGLE_PASS = SOLVES

#: The two commands of a shot: ``arm`` sets the armed flag, ``inject``
#: requests a launch.
ARM, INJECT = "arm", "inject"

#: The command that empties the charge stored in the model: a write of any
#: value but its default. It takes no shot and changes no series.
CLEAR = "clear_stored_charge"

#: Every command a model takes.
COMMANDS: tuple[str, ...] = (ARM, INJECT, CLEAR)

#: What a turn series holds of its monitor per tracked sample: a transverse
#: position, or the share of the launched charge that passed.
SERIES_AXES: tuple[str, ...] = ("x", "y", "sum")

#: The value an ``arm`` command arms with.
_ARMING = 1.0

FaultVariable = PyATWritableScalarVariable | PyATWritableEnumVariable

# A solve (the simulator's last solution object), and the optics computed for
# it.
_OpticsMemo = tuple[Any, dict[str, np.ndarray]]
_ChromaticityMemo = tuple[Any, np.ndarray]

#: The prime bases of the six low-discrepancy sequences a bunch's shape is
#: drawn from, one per pyAT coordinate.
_SHAPE_BASES: tuple[int, ...] = (2, 3, 5, 7, 11, 13)

#: The most floats one tracking call may return. A bunch whose track would
#: return more is tracked in particle chunks, which track exactly as one
#: call does: particles do not interact.
_TRACK_FLOATS = 1 << 24


def tune_planes(lattice: at.Lattice) -> int:
    """How many tunes a lattice has: three with longitudinal motion, else two."""
    return 3 if lattice.is_6d else 2


def _fault_name(address: str, field: str) -> str:
    return f"{address}{FAULT_SEPARATOR}{field}"


def _monitor_faults(
    channels: Iterable[Variable], seeds: Mapping[str, float]
) -> tuple[list[FaultVariable], list[str]]:
    """Declare every monitor readback's readout faults, and list the monitors.

    The first readback on an element's axis carries that axis's fields; the
    roll sits on the element's x-axis readback, else on its only one.

    Returns:
        ``(faults, monitors)``: the fault variables, and every monitor element
        in the order its first readback was given.
    """
    by_element: dict[str, dict[str, MonitorVariable]] = {}
    for variable in channels:
        if isinstance(variable, MonitorVariable):
            axes = by_element.setdefault(variable.element_name, {})
            axes.setdefault(variable.axis, variable)

    faults: list[FaultVariable] = []
    for element, axes in by_element.items():
        for axis, reading in axes.items():
            for field, identity in READOUT_IDENTITY.items():
                name = _fault_name(reading.name, field)
                binding: dict[str, Any] = {
                    "name": name,
                    "element_name": element,
                    "attribute": f"{_READOUT_PREFIX}{field}_{axis}",
                    "default_value": seeds.get(name, identity),
                    "default_validation_config": ConfigEnum.ERROR,
                }
                if field == "polarity":
                    faults.append(
                        PyATWritableEnumVariable(**binding, options=list(POLARITY_OPTIONS))
                    )
                    continue
                unit = reading.unit if field in _FIELDS_IN_READING_UNIT else None
                faults.append(
                    PyATWritableScalarVariable(
                        **binding, value_range=FAULT_BOUNDS.get(field), unit=unit
                    )
                )
        rolled = axes.get("x") or next(iter(axes.values()))
        name = _fault_name(rolled.name, ROLL)
        faults.append(
            PyATWritableScalarVariable(
                name=name,
                bindings=[ElementBinding(element_name=element, attribute=_READOUT_PREFIX + ROLL)],
                default_value=seeds.get(name, ROLL_IDENTITY),
                value_range=FAULT_BOUNDS[ROLL],
                unit=_ROLL_UNIT,
                default_validation_config=ConfigEnum.ERROR,
            )
        )
    return faults, list(by_element)


def _calibration_faults(
    channels: Iterable[Variable], seeds: Mapping[str, float]
) -> list[FaultVariable]:
    """Declare each magnet setpoint's own calibration, on its first bound element."""
    faults: list[FaultVariable] = []
    for variable in channels:
        if not (
            isinstance(variable, CalibratedSetpoint)
            and variable.bindings[0].attribute in CALIBRATED_ATTRIBUTES
        ):
            continue
        for field, identity in SUPPLY_IDENTITY.items():
            name = _fault_name(variable.name, field)
            faults.append(
                PyATWritableScalarVariable(
                    name=name,
                    bindings=[
                        ElementBinding(
                            element_name=variable.bindings[0].element_name,
                            attribute=supply_attribute(variable.name, field),
                        )
                    ],
                    default_value=seeds.get(name, identity),
                    value_range=FAULT_BOUNDS.get(field),
                    unit=variable.unit if field == "cal_offset" else None,
                    default_validation_config=ConfigEnum.ERROR,
                )
            )
    return faults


def _optics_variables(monitor_count: int, planes: int, solve: str) -> list[PyATReadOnlyNDVariable]:
    """Declare the optics arrays.

    The tunes and the chromaticity hold one value per plane, and only a
    periodic model declares them; per-monitor arrays hold one ``(x, y)`` row
    each.
    """
    per_monitor = (monitor_count, 2)
    planar = (
        [
            PyATReadOnlyNDVariable(name=TUNES, shape=(planes,)),
            PyATReadOnlyNDVariable(name=CHROMATICITY, shape=(planes,)),
        ]
        if solve == PERIODIC
        else []
    )
    return [
        *planar,
        PyATReadOnlyNDVariable(name=BETA_AT_MONITORS, shape=per_monitor, unit="m"),
        PyATReadOnlyNDVariable(name=ORBIT_AT_MONITORS, shape=per_monitor, unit="m"),
    ]


class ReadbackVariable(ReadOnlyActionMixin[PyATSimulator], ScalarVariable):
    """A wired readback the model serves from another of its variables.

    It binds no element: the model computes it on read from ``source``, so it
    follows every write of that source and every :meth:`reset`.

    Attributes:
        source: A setpoint's address, read back through the setpoint's own
            calibration at the hardware value it holds (a deck property's
            setpoint, such as the energy knob, at that value itself); ``tunes`` or
            ``chromaticity``, read at ``component``; or ``None`` for a
            readback with nothing to follow, which reads ``default_value``.
        component: The plane of ``tunes`` or ``chromaticity``; ``None``
            otherwise.
        element_name: Always ``None``, so a model's per-element binding check
            has nothing to resolve.
    """

    model_config = ConfigDict(extra="forbid")

    source: str | None = None
    component: int | None = None
    element_name: None = None
    read_only: bool = True

    def _get(self, simulator: PyATSimulator) -> float:
        """Refuse. The owning model computes the value, not the variable.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"{self.name!r} is read from {self.source!r}; read it through the model that "
            "declares it"
        )


class ArrayReadbackVariable(PyATReadOnlyNDVariable):
    """A wired readback serving the whole of ``tunes`` or ``chromaticity``.

    Attributes:
        source: ``tunes`` or ``chromaticity``.
    """

    source: str


class CommandVariable(WritableActionMixin[PyATSimulator], ScalarVariable):
    """A command of the shot: a write the model acts on, bound to no element.

    Nothing of the lattice changes when it is written, so it has nothing to
    snapshot and nothing to roll back; the model reads what the write means
    off the batch. A read answers the value last written.

    Attributes:
        command: ``arm``, ``inject`` or ``clear_stored_charge``.
        element_name: Always ``None``, so a model's per-element binding check
            has nothing to resolve.
    """

    model_config = ConfigDict(extra="forbid")

    command: str
    element_name: None = None

    @model_validator(mode="after")
    def _check_the_command(self) -> CommandVariable:
        """Refuse a command the shot does not have, or one with no rest value."""
        if self.command not in COMMANDS:
            raise ValueError(f"command {self.name} is {self.command!r}; use {', '.join(COMMANDS)}")
        if self.default_value is None:
            raise ValueError(
                f"command {self.name} needs a default_value: it is what reset() writes back"
            )
        return self

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:  # noqa: ARG002
        """Nothing: a command writes no attribute of the lattice."""
        return []

    def _get(self, simulator: PyATSimulator) -> float:
        """Refuse. The owning model answers with the value last written.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"{self.name!r} is a command; read it through the model that declares it"
        )

    def _set(self, simulator: PyATSimulator, value: Any) -> None:
        """Write nothing: the owning model acts on the batch that carries the command."""


class TurnsTrackedVariable(WritableActionMixin[PyATSimulator], ScalarVariable):
    """The model's turns tracked: the turns after the trigger sample a shot tracks.

    A global setting, bound to no element: nothing of the lattice changes
    when it is written, so it has nothing to snapshot and nothing to roll
    back, and the model reads it off the batch for the next shot. A value is
    a whole number of turns, at least 1; the model bounds it by the samples
    its turn series hold after the trigger.

    Attributes:
        element_name: Always ``None``, so a model's per-element binding check
            has nothing to resolve.
    """

    model_config = ConfigDict(extra="forbid")

    element_name: None = None
    default_validation_config: ConfigEnum = ConfigEnum.ERROR

    @model_validator(mode="after")
    def _check_the_turns(self) -> TurnsTrackedVariable:
        """Refuse a setting with no rest value, or a range admitting no turn."""
        if self.default_value is None:
            raise ValueError(
                f"turns tracked {self.name} needs a default_value: it is what reset() writes back"
            )
        if self.value_range is None or self.value_range[0] < 1:
            raise ValueError(
                f"turns tracked {self.name} ranges over {self.value_range}; give a value_range "
                "of whole turns from 1"
            )
        return self

    def validate_value(self, value: Any, config: ConfigEnum | None = None) -> None:
        """Check ``value`` as any scalar is checked, and that it is a whole number.

        Raises:
            ValueError: ``value`` is outside the range or not a whole number.
        """
        super().validate_value(value, config)
        if not float(value).is_integer():
            raise ValueError(f"turns tracked is {value}; a shot tracks a whole number of turns")

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:  # noqa: ARG002
        """Nothing: the setting writes no attribute of the lattice."""
        return []

    def _get(self, simulator: PyATSimulator) -> float:
        """Refuse. The owning model answers with the value last written.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"{self.name!r} is the turns tracked; read it through the model that declares it"
        )

    def _set(self, simulator: PyATSimulator, value: Any) -> None:
        """Write nothing: the owning model reads the value off the batch for the next shot."""


class ArmedVariable(ReadOnlyActionMixin[PyATSimulator], ScalarVariable):
    """A reading of the model's armed flag: 1 while a shot will fill the series, else 0.

    Attributes:
        element_name: Always ``None``, so a model's per-element binding check
            has nothing to resolve.
    """

    model_config = ConfigDict(extra="forbid")

    element_name: None = None
    read_only: bool = True

    def _get(self, simulator: PyATSimulator) -> float:
        """Refuse. The owning model holds the flag, not the variable.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"{self.name!r} reads the armed flag; read it through the model that declares it"
        )


class StoredCurrentVariable(ReadOnlyActionMixin[PyATSimulator], ScalarVariable):
    """A reading of the charge stored in the model, served as a current.

    The model serves it: the stored charge times the revolution frequency of
    the deck, in amperes for a charge in coulombs, published through the
    record's calibration. It carries what the model needs to know of a stored
    beam with no closed orbit to circulate on.

    Attributes:
        surviving_fraction: The share of the stored charge left after each
            shot taken while the lattice has no closed orbit for want of RF,
            from 0 (gone in one shot) to 1.
        charge_floor: The stored charge, in the unit of ``stored_charge``,
            at or below which a decaying stored charge is gone.
        calibration: Hardware to physics; ``None`` when the published unit is
            amperes.
        inverse: Physics back to hardware, or ``None`` where the record states
            none.
        element_name: Always ``None``, so a model's per-element binding check
            has nothing to resolve.
    """

    model_config = ConfigDict(extra="forbid")

    surviving_fraction: float = 0.0
    charge_floor: float = 0.0
    calibration: Calibration | None = None
    inverse: Calibration | None = None
    element_name: None = None
    read_only: bool = True

    @model_validator(mode="after")
    def _check_the_decay(self) -> StoredCurrentVariable:
        """Refuse a surviving fraction outside 0 to 1, a negative floor, or no way back."""
        if not 0.0 <= self.surviving_fraction <= 1.0:
            raise ValueError(
                f"stored current {self.name} keeps {self.surviving_fraction} of its charge per "
                "shot; give a surviving_fraction from 0 to 1"
            )
        if not 0.0 <= self.charge_floor < math.inf:
            raise ValueError(
                f"stored current {self.name} has charge floor {self.charge_floor}; give a "
                "finite charge_floor of 0 or more"
            )
        check_inverse(self.calibration, self.inverse)
        return self

    @classmethod
    def from_wiring(cls, record: Any) -> StoredCurrentVariable:
        """Build the reading one wiring record describes.

        Args:
            record: The wiring record: ``address``, ``unit``, the engine
                block's ``surviving_fraction`` and ``charge_floor`` (each 0
                when unstated) and ``calibration`` (``curve``, ``inverse``).

        Raises:
            ValueError: the decay data are out of range, or the calibration
                has no way back to hardware units.
        """
        block = slot(record, "engine")
        calibration = slot(record, "calibration")
        fraction = slot(block, "surviving_fraction")
        floor = slot(block, "charge_floor")
        return cls(
            name=str(slot(record, "address")),
            unit=slot(record, "unit"),
            surviving_fraction=0.0 if fraction is None else float(fraction),
            charge_floor=0.0 if floor is None else float(floor),
            calibration=curve_from_record(slot(calibration, "curve")),
            inverse=curve_from_record(slot(calibration, "inverse")),
        )

    def _get(self, simulator: PyATSimulator) -> float:
        """Refuse. The owning model holds the stored charge, not the variable.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"{self.name!r} reads the stored charge; read it through the model that declares it"
        )


class SeriesVariable(PyATReadOnlyNDVariable):
    """A turn series: one coordinate of one monitor per tracked sample.

    The model that declares it fills it with a shot and serves it on read. A
    position is tracked in metres and published through the record's
    calibration, by the rule a monitor reading follows.

    Attributes:
        monitor: The monitor element the samples are read at.
        axis: ``x`` or ``y``, the position; or ``sum``, 1 for a sample the
            particle reached.
        series: What one sample is taken per: ``turn``.
        calibration: Hardware to physics; ``None`` when the published unit is
            the tracked one.
        inverse: Physics back to hardware, or ``None`` where the record states
            none.
    """

    model_config = ConfigDict(extra="forbid")

    monitor: str
    axis: str
    series: str
    calibration: Calibration | None = None
    inverse: Calibration | None = None

    @model_validator(mode="after")
    def _check_the_series(self) -> SeriesVariable:
        """Refuse an unknown axis, a shape of more than one length, or no way back."""
        if self.axis not in SERIES_AXES:
            raise ValueError(
                f"turn series {self.name} names axis {self.axis!r}; use one of {list(SERIES_AXES)}"
            )
        if len(self.shape) != 1:
            raise ValueError(
                f"turn series {self.name} has shape {tuple(self.shape)}; a series is one length"
            )
        check_inverse(self.calibration, self.inverse)
        return self

    @classmethod
    def from_wiring(cls, record: Any, samples: int) -> SeriesVariable:
        """Build the series one wiring record describes.

        Args:
            record: The wiring record: ``address``, ``element`` (or the first
                of its ``slices``), the engine block's ``axis`` and
                ``series``, ``unit`` and ``calibration`` (``curve``,
                ``inverse``).
            samples: The samples the series holds.

        Returns:
            The variable.

        Raises:
            ValueError: the record names no element or an axis that is none of
                :data:`SERIES_AXES`, or its calibration has no way back to
                hardware units.
        """
        address = str(slot(record, "address"))
        slices = slot(record, "slices")
        element = slot(slices[0], "element") if slices else slot(record, "element")
        if element is None:
            raise ValueError(f"turn series {address} names no monitor element")
        block = slot(record, "engine")
        calibration = slot(record, "calibration")
        return cls(
            name=address,
            shape=(samples,),
            monitor=str(element),
            axis=str(slot(block, "axis")),
            series=str(slot(block, "series")),
            unit=slot(record, "unit"),
            calibration=curve_from_record(slot(calibration, "curve")),
            inverse=curve_from_record(slot(calibration, "inverse")),
        )


@dataclass
class _Shot:
    """What a batch leaves of the shot state once it commits.

    Attributes:
        armed: The armed flag.
        count: The shots taken since the model was last at rest.
        seen: Each request's last written value.
        buffers: Every series' samples, or ``None`` to keep the ones held.
        track: Whether the batch tracks a shot, which then fills ``buffers``.
        time: When the shot that filled ``buffers`` was taken, in epoch
            seconds; ``None`` for buffers no shot filled. Read only with
            ``buffers``.
        turns: The turns tracked the batch leaves, which a tracked shot
            fills after the trigger sample.
        stored: The charge stored in the model once the batch's shots are
            taken, in the unit of ``stored_charge``.
        fired: The shots the batch takes, tracked or not.
    """

    armed: bool
    count: int
    seen: dict[str, float]
    buffers: dict[str, np.ndarray] | None = None
    track: bool = False
    time: float | None = None
    turns: int = 0
    stored: float = 0.0
    fired: int = 0


def _check_series(channels: Iterable[Variable], injection: Any) -> int:
    """The samples every turn series among ``channels`` holds; zero with none.

    Raises:
        ValueError: a model with a turn series has no injection, its series
            differ in length, or one cannot hold a sample after the
            injection's pre-trigger samples.
    """
    series = [variable for variable in channels if isinstance(variable, SeriesVariable)]
    if not series:
        return 0
    if injection is None:
        raise ValueError(
            f"turn series {series[0].name} is on a model with no injection; a series is "
            "filled by a particle launched at one"
        )
    lengths = {int(variable.shape[0]) for variable in series}
    if len(lengths) > 1:
        raise ValueError(
            f"the model's turn series hold {sorted(lengths)} samples; every turn series of "
            "a model holds the same number"
        )
    samples = lengths.pop()
    if samples <= injection.trigger_sample:
        raise ValueError(
            f"the model's turn series hold {samples} samples; a series holds the "
            f"{injection.trigger_sample} samples before the launch and at least one after"
        )
    return samples


def _check_turns_tracked(
    channels: Iterable[Variable], samples: int, injection: Any
) -> TurnsTrackedVariable | None:
    """The model's turns-tracked setting among ``channels``; ``None`` with none.

    Raises:
        ValueError: the model wires more than one, wires one with no turn
            series, or ranges it past the samples its series hold after the
            trigger.
    """
    settings = [variable for variable in channels if isinstance(variable, TurnsTrackedVariable)]
    if not settings:
        return None
    if len(settings) > 1:
        raise ValueError(
            f"the model wires turns tracked as {[variable.name for variable in settings]}; a "
            "model tracks one number of turns, so it wires one setting for it"
        )
    setting = settings[0]
    if not samples:
        raise ValueError(
            f"turns tracked {setting.name} is on a model with no turn series; it sizes the "
            "turns a shot fills them for"
        )
    bound = samples - int(injection.trigger_sample)
    if setting.value_range is not None and setting.value_range[1] > bound:
        raise ValueError(
            f"turns tracked {setting.name} ranges up to {setting.value_range[1]}; the model's "
            f"turn series hold {bound} turns after the trigger sample"
        )
    return setting


def _check_stored_current(channels: Iterable[Variable]) -> tuple[float, float]:
    """The decay data the model's stored-current readings state: fraction, floor.

    A model that wires no stored-current reading loses its whole stored
    charge to the first shot with no RF, with no floor.

    Raises:
        ValueError: two readings state different decay data.
    """
    readings = [variable for variable in channels if isinstance(variable, StoredCurrentVariable)]
    stated = {(reading.surviving_fraction, reading.charge_floor) for reading in readings}
    if len(stated) > 1:
        raise ValueError(
            f"the model's stored-current readings {[reading.name for reading in readings]} "
            f"state the decay data {sorted(stated)}; a model holds one stored charge, so its "
            "readings state one surviving_fraction and charge_floor"
        )
    return stated.pop() if stated else (0.0, 0.0)


def _check_readbacks(channels: Iterable[Variable], planes: int, solve: str) -> None:
    """Refuse a readback whose source the model does not serve.

    Raises:
        ValueError: a readback names a source that is neither a setpoint
            among ``channels`` nor ``tunes``/``chromaticity``, an optics
            plane the deck does not have, or ``tunes``/``chromaticity`` on a
            single-pass model.
    """
    setpoints = {
        variable.name
        for variable in channels
        if isinstance(variable, (CalibratedSetpoint, EnergyVariable, TurnsTrackedVariable))
    }
    for variable in channels:
        if isinstance(variable, ArrayReadbackVariable):
            if solve != PERIODIC or variable.source not in PLANE_OPTICS:
                raise ValueError(
                    f"readback {variable.name} reads all of {variable.source}; only a periodic "
                    f"model serves {sorted(PLANE_OPTICS)}"
                )
            if tuple(variable.shape) != (planes,):
                raise ValueError(
                    f"readback {variable.name} has shape {tuple(variable.shape)}; "
                    f"{variable.source} holds {planes} planes"
                )
            continue
        if not isinstance(variable, ReadbackVariable) or variable.source is None:
            continue
        if variable.source in PLANE_OPTICS:
            if solve != PERIODIC:
                raise ValueError(
                    f"readback {variable.name} reads {variable.source}; a single_pass model "
                    "serves no tunes or chromaticity"
                )
            if variable.component is None or not 0 <= variable.component < planes:
                raise ValueError(
                    f"readback {variable.name} reads {variable.source} plane "
                    f"{variable.component}; the deck has planes 0 to {planes - 1}"
                )
        elif variable.source not in setpoints:
            raise ValueError(
                f"readback {variable.name} reads {variable.source}, which is no setpoint "
                f"of the model and none of {sorted(PLANE_OPTICS)}"
            )


def _refuse_name_collisions(channels: Iterable[Variable], declared: Iterable[Variable]) -> None:
    """Refuse a declared name a channel variable already carries.

    Raises:
        ValueError: a fault or optics name is already a channel address.
    """
    wired = {variable.name for variable in channels}
    if clashes := sorted({variable.name for variable in declared} & wired):
        raise ValueError(
            f"the model declares {clashes}, which the wiring already carries as channel "
            f"addresses; a model-only variable is told from a wired one by its name, so the "
            f"two rosters cannot share one"
        )


def _seed_fault_attributes(lattice: at.Lattice, faults: Iterable[FaultVariable]) -> None:
    """Write each fault variable's default onto every element it binds.

    Every element is resolved before any is written, so a fault bound to an
    element the lattice lacks leaves the lattice untouched.

    Raises:
        UnknownElementError: a fault binds an element the lattice does not have.
    """
    elements = {element.FamName: element for element in lattice}
    bound = [
        (binding, float(fault.default_value))  # type: ignore[arg-type]
        for fault in faults
        for binding in fault.bindings
    ]
    if missing := sorted({binding.element_name for binding, _ in bound} - set(elements)):
        raise UnknownElementError(f"the lattice has no element for faults on {missing}")
    for binding, default in bound:
        setattr(elements[binding.element_name], binding.attribute, default)


def _seed_readout_attributes(lattice: at.Lattice, channels: Iterable[Variable]) -> None:
    """Give every element a setting drives a readout attribute of its identity, where absent.

    A facility's setting record may bind a readout attribute the deck never
    states; the element then carries its identity, so a read before the
    first write answers it and the monitor reads exactly.
    """
    elements = {element.FamName: element for element in lattice}
    for variable in channels:
        for binding in getattr(variable, "bindings", None) or ():
            attribute = getattr(binding, "attribute", None)
            element = elements.get(getattr(binding, "element_name", None))
            if attribute not in READOUT_ATTRIBUTE_IDENTITY or element is None:
                continue
            if not hasattr(element, attribute):
                setattr(element, attribute, READOUT_ATTRIBUTE_IDENTITY[attribute])


@dataclass(frozen=True)
class _Bunch:
    """The bunch a scenario declares, fixed when the model is built.

    Attributes:
        key: The model's id, which keys every shot's jitter.
        shape: The particles' offsets from the launch, ``(6, particles)``,
            centred and matched to the deck's optics at the injection.
        jitter: The centroid jitter's sigma in ``x`` and ``y``, in metres.
        charge: The charge one shot injects, in the unit of ``stored_charge``;
            a sum of 1 is this charge passing a monitor.
    """

    key: str
    shape: np.ndarray
    jitter: tuple[float, float]
    charge: float = 1.0


def _radical_inverse(indexes: np.ndarray, base: int) -> np.ndarray:
    """The van der Corput sequence in ``base`` at ``indexes``."""
    value = np.zeros(indexes.shape, dtype=np.float64)
    scale = 1.0 / base
    remaining = indexes.copy()
    while remaining.any():
        value += (remaining % base) * scale
        remaining //= base
        scale /= base
    return value


def _normal_quantiles(particles: int, base: int) -> np.ndarray:
    """``particles`` centred standard-normal quantiles, ordered by the sequence in ``base``.

    Each particle takes one stratum ``(rank + 0.5) / particles`` of the normal
    distribution, its rank the order of its van der Corput value in
    ``base``: every coordinate holds the same quantiles, and the distinct
    bases leave them uncorrelated across coordinates. One particle sits at 0.
    """
    from scipy.special import ndtri

    ranks = np.argsort(np.argsort(_radical_inverse(np.arange(particles), base), kind="stable"))
    quantiles = np.asarray(ndtri((ranks + 0.5) / particles), dtype=np.float64)
    centred: np.ndarray = quantiles - quantiles.mean()
    return centred


def _bunch_shape(
    optics: Mapping[str, np.ndarray], particles: int, physics: Mapping[str, float]
) -> np.ndarray:
    """The particles' offsets from the launch, matched to ``optics``, ``(6, particles)``.

    Each transverse plane is a Gaussian of its emittance in the Twiss
    ``beta`` and ``alpha`` at the injection, displaced by the dispersion
    times the momentum offset; the momentum offset and the path length are
    Gaussians of ``bunch_sigma_delta`` and ``bunch_sigma_z``. An unstated
    size is 0.
    """
    normal = np.vstack([_normal_quantiles(particles, base) for base in _SHAPE_BASES])
    emittance_x, emittance_y, sigma_delta, sigma_z = (
        float(physics.get(name, 0.0)) for name in BUNCH_SHAPE_FAULTS
    )
    beta = np.asarray(optics["beta"], dtype=np.float64)
    alpha = np.asarray(optics["alpha"], dtype=np.float64)
    dispersion = np.asarray(optics["dispersion"], dtype=np.float64)
    shape = np.zeros((6, particles), dtype=np.float64)
    shape[4] = sigma_delta * normal[4]
    shape[5] = sigma_z * normal[5]
    for plane, emittance in enumerate((emittance_x, emittance_y)):
        a, b = normal[2 * plane], normal[2 * plane + 1]
        shape[2 * plane] = math.sqrt(emittance * beta[plane]) * a
        shape[2 * plane + 1] = math.sqrt(emittance / beta[plane]) * (b - alpha[plane] * a)
        shape[2 * plane] += dispersion[2 * plane] * shape[4]
        shape[2 * plane + 1] += dispersion[2 * plane + 1] * shape[4]
    return shape


def _declared_bunch(
    lattice: at.Lattice,
    solve: str,
    injection: Any,
    name: str | None,
    physics: Mapping[str, float],
) -> _Bunch | None:
    """The bunch ``physics`` declares, or ``None`` when it states no particle count.

    The bunch is matched to the optics of a 4D copy of ``lattice`` at the
    injection marker, solved here, once. It injects ``bunch_charge``, 1 when
    unstated.

    Raises:
        ValueError: the model launches at no injection marker, states no id
            to key the jitter by, a bunch charge that is not a finite number
            above 0, or its deck has no periodic optics at the marker to
            match the bunch to.
    """
    if BUNCH_PARTICLES not in physics:
        return None
    if solve != PERIODIC or injection is None or getattr(injection, "marker", None) is None:
        raise ValueError(
            f"model-wide fault {BUNCH_PARTICLES} is set on a model that launches at no "
            "injection marker; state the bunch on a periodic model whose settings state "
            "pyat.injection"
        )
    if not name:
        raise ValueError(f"model-wide fault {BUNCH_PARTICLES} needs the model's id")
    charge = float(physics.get(BUNCH_CHARGE, 1.0))
    if not 0.0 < charge < math.inf:
        raise ValueError(
            f"model-wide fault {BUNCH_CHARGE} is {charge:g} on model {name}; a bunch injects a "
            "finite charge above 0, the unit its sum and stored charge are counted in"
        )
    particles = int(physics[BUNCH_PARTICLES])
    index = int(injection.index)
    try:
        _, _, optics = lattice.disable_6d(copy=True).get_optics(refpts=[index])
    except Exception as error:
        raise ValueError(
            f"model-wide fault {BUNCH_PARTICLES} is set on model {name}, whose deck has no "
            f"periodic optics at its injection marker to match the bunch to: {error}"
        ) from error
    row = {key: optics[key][0] for key in ("beta", "alpha", "dispersion")}
    jitter_x, jitter_y = (float(physics.get(key, 0.0)) for key in CENTROID_JITTER_FAULTS)
    return _Bunch(
        key=name,
        shape=_bunch_shape(row, particles, physics),
        jitter=(jitter_x, jitter_y),
        charge=charge,
    )


def _bunch_reading(
    lattice: at.Lattice, particles: np.ndarray, refpts: np.ndarray, turns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track ``particles`` and reduce them per monitor and turn, each ``(monitors, turns)``.

    A particle is alive where neither tracked transverse coordinate is NaN.

    Returns:
        The share of ``particles`` alive, and the survivors' centroid in
        ``x`` and ``y``, NaN where none is alive.
    """
    count = particles.shape[1]
    alive = np.zeros((refpts.size, turns), dtype=np.int64)
    sum_x = np.zeros((refpts.size, turns), dtype=np.float64)
    sum_y = np.zeros((refpts.size, turns), dtype=np.float64)
    chunk = max(1, _TRACK_FLOATS // (6 * max(1, refpts.size) * max(1, turns)))
    for begin in range(0, count, chunk):
        tracked, _, _ = at.lattice_track(
            lattice,
            np.ascontiguousarray(particles[:, begin : begin + chunk]),
            nturns=turns,
            refpts=refpts,
            losses=True,
        )
        x, y = tracked[0], tracked[2]
        live = ~(np.isnan(x) | np.isnan(y))
        alive += live.sum(axis=0)
        sum_x += np.where(live, x, 0.0).sum(axis=0)
        sum_y += np.where(live, y, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid_x = np.where(alive > 0, sum_x / alive, np.nan)
        centroid_y = np.where(alive > 0, sum_y / alive, np.nan)
    return alive / count, centroid_x, centroid_y


def _with_stored_beam(
    x: np.ndarray, y: np.ndarray, sums: np.ndarray, share: float, orbit: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The centroid of the injected beam and a stored beam on ``orbit``, sample by sample.

    ``x`` and ``y`` are the injected survivors' centroid, NaN where none is
    alive, and ``sums`` their share of the bunch; the stored beam is ``share``
    bunches on the closed orbit ``orbit``. Each sample weighs the two by
    their charge, so a sample no injected particle reaches reads the orbit.
    """
    alive = (sums > 0.0) & ~(np.isnan(x) | np.isnan(y))
    weight = np.where(alive, sums, 0.0)
    total = weight + share
    mixed_x = (weight * np.where(alive, x, 0.0) + share * orbit[0]) / total
    mixed_y = (weight * np.where(alive, y, 0.0) + share * orbit[1]) / total
    return mixed_x, mixed_y


class PyATLatticeModel(LUMEPyATModel):
    """A LUME model over one persistent ``at.Lattice`` and its wired variables.

    Hardware setpoints in, monitor readings out, both in the units the wiring's
    calibrations state -- the conversions live on the variables. Beside them
    stand the settable fault state of every wired monitor and magnet, and the
    read-only optics arrays computed on read. One instance owns one lattice for
    its whole lifetime, so sequential writes compose exactly like their
    physical counterparts would, and a seeded fault survives every later write
    and every :meth:`reset`.
    """

    def __init__(
        self,
        lattice: at.Lattice,
        channels: Iterable[Variable],
        *,
        faults: Mapping[str, float] | None = None,
        element_misalignments: Mapping[str, Mapping[str, float]] | None = None,
        solve: str = PERIODIC,
        twiss_in: Mapping[str, np.ndarray] | None = None,
        injection: Any = None,
        name: str | None = None,
        bunch: Mapping[str, float] | None = None,
    ) -> None:
        """Adopt the wired variables, declare their faults, and solve once.

        Args:
            lattice: The deck to drive, already loaded. It is mutated in place
                for the model's whole lifetime, so a caller that keeps its own
                copy hands over a fresh one.
            channels: One variable per wired address, as
                :func:`~osprey.simulation.engines.pyat_variables.variable_from_wiring`
                builds them. An energy knob among them adopts every
                rigidity-scaled setpoint among them.
            faults: Fault name (``<address>/<field>``) -> seed. Each seeds its
                variable's default and its element attribute; every fault not
                named is identity.
            element_misalignments: Element ``FamName`` -> kwargs for
                :func:`~lume_pyat.utils.apply_misalignment` (``dx``/``dy``/
                ``roll``, all optional), applied once before the boot solve.
            solve: ``periodic`` solves the closed orbit; ``single_pass``
                tracks one pass through a line from ``twiss_in``.
            twiss_in: pyAT's initial conditions, as ``prepare`` normalises
                them; read by a ``single_pass`` model only.
            injection: Where a tracked particle is launched, as ``prepare``
                normalises it, or ``None``. A ``periodic`` model that has one
                tolerates a lost closed orbit: its solve never refuses, and
                :attr:`no_closed_orbit` says when the lattice has none.
            name: The model's id, which keys a bunch's centroid jitter.
            bunch: The scenario's model-wide physics. A stated
                ``bunch_particles`` declares a bunch, shaped by the bunch
                shape and centroid jitter faults among them and injecting
                ``bunch_charge``; without it every shot tracks one particle.
                A stated ``stored_charge`` is the charge the model holds at
                rest.

        Raises:
            ValueError: a seed names a fault the model does not declare, a
                fault or optics name is already a channel address, ``solve``
                is neither of its two values, a ``single_pass`` model has no
                ``twiss_in``, a single-pass readback reads the tunes or
                the chromaticity, a turn series has no ``injection`` to be
                filled from or a length unlike the model's other series, or a
                bunch is declared on a model with no injection marker, with no
                ``name``, with a bunch charge that is no finite number above 0,
                or on a deck with no periodic optics at its marker, or two
                stored-current readings state different decay data.
            pydantic.ValidationError: a seed lies outside its field's bounds,
                or a polarity seed is neither of its two values.
            UnknownElementError: a variable binds an element the lattice does
                not have.
            OrbitSolveError: the lattice as handed over, with its
                misalignments, has no stable closed orbit and the model has
                no ``injection``, or loses the single-pass particle.
        """
        if solve not in SOLVES:
            raise ValueError(f"solve is {solve!r}; use {' or '.join(SOLVES)}")
        if solve == SINGLE_PASS and twiss_in is None:
            raise ValueError("a single_pass model needs twiss_in")
        channels = list(channels)
        seeds = dict(faults or {})
        for variable in channels:
            if isinstance(variable, EnergyVariable):
                variable.couple(
                    other for other in channels if isinstance(other, PyATWritableScalarVariable)
                )

        monitor_faults, monitors = _monitor_faults(channels, seeds)
        calibration_faults = _calibration_faults(channels, seeds)
        declared = [*monitor_faults, *calibration_faults]
        if unknown := sorted(set(seeds) - {fault.name for fault in declared}):
            raise ValueError(
                f"fault seeds name {unknown}, which the model does not declare; a monitor "
                f"readback takes {sorted(READOUT_IDENTITY)} (and {ROLL!r} on its x-axis or "
                f"only readback), a PolynomB or KickAngle setpoint takes "
                f"{sorted(SUPPLY_IDENTITY)}"
            )
        planes = tune_planes(lattice)
        _check_readbacks(channels, planes, solve)
        samples = _check_series(channels, injection)
        turns_setting = _check_turns_tracked(channels, samples, injection)
        decay = _check_stored_current(channels)
        optics = _optics_variables(len(monitors), planes, solve)
        _refuse_name_collisions(channels, [*declared, *optics])
        declared_bunch = _declared_bunch(lattice, solve, injection, name, bunch or {})
        _seed_fault_attributes(lattice, declared)
        _seed_readout_attributes(lattice, channels)

        # lume's ActionVariable is a union of hinting stubs in lume/actions.py
        # that no variable class inherits.
        variables: list[ActionVariable[PyATSimulator]] = [*channels, *declared, *optics]  # type: ignore[list-item]
        misalignments = (
            None
            if element_misalignments is None
            else {name: dict(kwargs) for name, kwargs in element_misalignments.items()}
        )
        simulator: PyATSimulator
        if solve == SINGLE_PASS:
            simulator = SinglePassSimulator(
                lattice,
                twiss_in=twiss_in,  # type: ignore[arg-type]
                element_misalignments=misalignments,
            )
        elif injection is not None:
            simulator = TolerantPeriodicSimulator(lattice, element_misalignments=misalignments)
        else:
            simulator = PyATSimulator(lattice, element_misalignments=misalignments)
        # Read by _read_outputs, which the base constructor already calls.
        self._pending_shot: _Shot | None = None
        self._restoring = False
        super().__init__(simulator=simulator, action_variables=variables)

        #: The variables this model declares that no channel addresses: the
        #: faults and the optics.
        self.derived_names: frozenset[str] = frozenset(
            variable.name for variable in (*declared, *optics)
        )
        self._fault_names: frozenset[str] = frozenset(fault.name for fault in declared)
        #: ``periodic`` or ``single_pass``.
        self.solve: str = solve
        #: Where a tracked particle is launched, or ``None``.
        self.injection: Any = injection
        self._bunch: _Bunch | None = declared_bunch
        self._optics_names: frozenset[str] = frozenset(variable.name for variable in optics)

        # Sorted by lattice index, so row i of every per-monitor optics array
        # is the i-th monitor along the lattice.
        self._monitor_order: list[str] = sorted(monitors, key=self.element_index)
        self._monitor_refpts = np.array(
            [self.element_index(element) for element in self._monitor_order], dtype=np.uint32
        )
        self._planes = planes
        self._readbacks: dict[str, ReadbackVariable | ArrayReadbackVariable] = {
            variable.name: variable
            for variable in channels
            if isinstance(variable, (ReadbackVariable, ArrayReadbackVariable))
        }
        self._setpoints: dict[str, CalibratedSetpoint] = {
            variable.name: variable
            for variable in channels
            if isinstance(variable, CalibratedSetpoint)
        }
        self._knobs: frozenset[str] = frozenset(
            variable.name for variable in channels if isinstance(variable, EnergyVariable)
        )
        self._optics_memo: _OpticsMemo | None = None
        self._chromaticity_memo: _ChromaticityMemo | None = None

        self._commands: dict[str, str] = {
            variable.name: variable.command
            for variable in channels
            if isinstance(variable, CommandVariable)
        }
        #: Each request's default, the value it has been seen at while at rest.
        self._requests: dict[str, float] = {
            variable.name: float(variable.default_value)  # type: ignore[arg-type]
            for variable in channels
            if isinstance(variable, CommandVariable) and variable.command == INJECT
        }
        self._armed_names: frozenset[str] = frozenset(
            variable.name for variable in channels if isinstance(variable, ArmedVariable)
        )
        self._series: dict[str, SeriesVariable] = {
            variable.name: variable for variable in channels if isinstance(variable, SeriesVariable)
        }
        self._samples = samples
        #: The turns tracked setting's address, or ``None`` when the model
        #: wires none and tracks its default.
        self._turns_name: str | None = None if turns_setting is None else turns_setting.name
        self._turns_default: int = (
            int(turns_tracked_default(None, samples - int(injection.trigger_sample)))
            if samples
            else 0
        )
        #: Each series monitor's index in the lattice.
        self._series_index: dict[str, int] = {
            monitor: self.element_index(monitor)
            for monitor in sorted({variable.monitor for variable in self._series.values()})
        }
        #: Each clearing command's default, which a write must differ from to clear.
        self._clears: dict[str, float] = {
            variable.name: float(variable.default_value)  # type: ignore[arg-type]
            for variable in channels
            if isinstance(variable, CommandVariable) and variable.command == CLEAR
        }
        self._stored_currents: dict[str, StoredCurrentVariable] = {
            variable.name: variable
            for variable in channels
            if isinstance(variable, StoredCurrentVariable)
        }
        #: The share of the stored charge a shot with no RF leaves, and the
        #: charge at or below which it is gone.
        self._decay: tuple[float, float] = decay
        #: The charge the model holds at rest: the scenario's.
        self._stored_rest: float = float((bunch or {}).get(STORED_CHARGE, 0.0))
        #: Turns per second: a stored charge times this is its current.
        self._revolution: float = float(at.constants.clight / lattice.circumference)
        self._cavities: tuple[int, ...] = tuple(
            index for index, element in enumerate(lattice) if isinstance(element, at.RFCavity)
        )
        rest = self._shot_at_rest()
        self._stored: float = rest.stored
        self._armed: bool = rest.armed
        self._shot_count: int = rest.count
        self._seen: dict[str, float] = rest.seen
        self._buffers: dict[str, np.ndarray] = rest.buffers or {}
        self._shot_time: float | None = rest.time

    @property
    def armed(self) -> bool:
        """Whether the next shot fills the turn series."""
        return self._armed

    @property
    def shot_count(self) -> int:
        """The shots taken since the model was last at rest, tracked or not."""
        return self._shot_count

    @property
    def shot_times(self) -> dict[str, float]:
        """When each turn series was filled, in epoch seconds, by address.

        Every series is filled by the same shot, so each carries that shot's
        time. A model whose series no shot has filled since it was last at
        rest -- built, reset or rebuilt -- names none. Each tracked shot is
        later than the one before it, even on a clock that has not moved.
        """
        if self._shot_time is None:
            return {}
        return dict.fromkeys(self._series, self._shot_time)

    @property
    def turns_tracked(self) -> int:
        """The turns after the trigger sample the next periodic shot tracks.

        The value of the model's turns-tracked setting when it wires one,
        else the default up to the samples its series hold after the
        trigger; 0 for a model with no turn series. A single-pass model
        tracks one pass whatever it says.
        """
        if self._turns_name is None:
            return self._turns_default
        return int(self._inputs[self._turns_name])

    @property
    def no_closed_orbit(self) -> bool:
        """Whether the lattice as it stands has no closed orbit.

        Only a ``periodic`` model with an ``injection`` can be in that state;
        every other model refuses the write that would lead to it. While it
        holds, every monitor reading, the optics arrays and the chromaticity
        are NaN. It ends with the first solve that finds a closed orbit.
        """
        return bool(getattr(self.simulator, "no_closed_orbit", False))

    def _set(self, values: dict[str, Any]) -> None:
        """Apply a batch atomically, its faults before its channel variables.

        A setpoint is converted through the calibration its element holds when
        it is written, so a batch that names a setpoint and its calibration --
        and :meth:`reset`, which names every writable -- writes the
        calibration first, whatever order the mapping gives. Each part keeps
        its own order.

        A batch that raises a request is one shot, taken after the batch's
        solve from the lattice the batch left. What the batch makes of the
        armed flag, the shot count, the last request seen, the series and the
        time of the shot that filled them is worked out beside the write and adopted only once the write has
        committed, so a refused batch changes none of them.

        A batch of commands alone, with or without the turns tracked, writes
        nothing to the lattice, so the lattice and its solve stand as they
        are and are not solved again: only the shot is worked out, from the
        lattice as the last solve left it.
        """
        held = self._commands.keys() | ({self._turns_name} if self._turns_name else set())
        if values and values.keys() <= held and not self._restoring:
            self._set_commands(values)
            return
        faults = {name: value for name, value in values.items() if name in self._fault_names}
        rest = {name: value for name, value in values.items() if name not in self._fault_names}
        self._pending_shot = self._shot_after(values)
        try:
            super()._set({**faults, **rest})
            shot = self._pending_shot
        finally:
            self._pending_shot = None
        self._adopt(shot)

    def _set_commands(self, values: dict[str, Any]) -> None:
        """Apply a batch of commands alone: work out its shot, and track it when armed.

        Raises:
            Whatever pyAT raises for a lattice it cannot track; nothing of
            the batch is adopted then.
        """
        shot = self._shot_after(values)
        self._fire(shot)
        self._inputs.update(values)
        self._adopt(shot)

    def _adopt(self, shot: _Shot) -> None:
        """Make ``shot`` the model's shot state, once its batch has committed."""
        self._armed, self._shot_count, self._seen = shot.armed, shot.count, shot.seen
        self._stored = shot.stored
        if shot.buffers is not None:
            self._buffers, self._shot_time = shot.buffers, shot.time

    def reset(self) -> None:
        """Write every writable's default back, and return the shot to rest.

        One atomic batch, as the base class describes. No shot is taken: the
        armed flag is cleared, the shot count is zero, every series is zero,
        each request has been seen at its default and the stored charge is
        the scenario's again.

        Raises:
            OrbitSolveError: the default configuration has no stable solve.
                The lattice and the shot state are left as they were.
        """
        self._restoring = True
        try:
            super().reset()
        finally:
            self._restoring = False

    # -- the shot: commands, the armed flag and the turn series -------------

    def _shot_at_rest(self) -> _Shot:
        """The shot state of a model no command has reached."""
        return _Shot(
            armed=False,
            count=0,
            seen=dict(self._requests),
            buffers={name: np.zeros(self._samples, dtype=np.float64) for name in self._series},
            stored=self._stored_rest,
        )

    def _shot_after(self, values: Mapping[str, Any]) -> _Shot:
        """The shot state ``values`` leads to, before any of it is tracked.

        An ``arm`` command written 1 arms, whatever its place in the batch. A
        request written above the value last seen, and not at its default, is
        one shot; every written request is seen. A ``clear_stored_charge``
        command written anything but its default empties the stored charge
        before the batch's shots are taken; it is no shot itself.
        """
        if self._restoring:
            return self._shot_at_rest()
        shot = _Shot(
            armed=self._armed, count=self._shot_count, seen=dict(self._seen), stored=self._stored
        )
        shot.turns = (
            int(float(values.get(self._turns_name, self.turns_tracked)))
            if self._turns_name is not None
            else self._turns_default
        )
        fired = 0
        for name, value in values.items():
            command = self._commands.get(name)
            if command == ARM and float(value) == _ARMING:
                shot.armed = True
            elif command == INJECT:
                if float(value) > shot.seen[name] and float(value) != self._requests[name]:
                    fired += 1
                shot.seen[name] = float(value)
            elif command == CLEAR and float(value) != self._clears[name]:
                shot.stored = 0.0
        if fired:
            shot.fired = fired
            shot.count += fired
            shot.track = shot.armed
            shot.armed = False
        return shot

    def _fire(self, shot: _Shot) -> None:
        """Take the shots ``shot`` counts, from the lattice as the last solve left it.

        Each shot is one injection step, in order. The last one is tracked
        when the batch armed it, which fills ``shot.buffers`` with the
        stored charge held before it; the stored charge left after every
        step lands in ``shot.stored``.

        Raises:
            Whatever pyAT raises for a lattice it cannot track.
        """
        if not shot.fired:
            return
        for count in range(shot.count - shot.fired + 1, shot.count):
            shot.stored = self._injection_step(shot.stored, count, shot.turns, None)
        captured: float | None = None
        if shot.track:
            shot.buffers, captured = self._track_shot(shot.count, shot.turns, shot.stored)
            shot.time = self._next_shot_time()
        shot.stored = self._injection_step(shot.stored, shot.count, shot.turns, captured)

    def _injection_step(
        self, stored: float, count: int, turns: int, captured: float | None
    ) -> float:
        """The stored charge shot ``count`` leaves of ``stored``.

        Only a bunch steps the stored charge. With a closed orbit the share of
        the bunch alive at the end of the tracked window -- ``captured``, or
        tracked here when ``None`` -- is captured, adding that share of the
        bunch charge. With no closed orbit nothing is captured: when every
        cavity is off the stored charge keeps its surviving fraction, and is
        gone at or below the floor; when the orbit is lost with RF on it is
        gone.
        """
        if self._bunch is None:
            return stored
        if self.no_closed_orbit:
            if not self._rf_off():
                return 0.0
            fraction, floor = self._decay
            kept = stored * fraction
            return kept if kept > floor else 0.0
        if captured is None:
            captured = self._captured(count, turns)
        return stored + captured * self._bunch.charge

    def _rf_off(self) -> bool:
        """Whether the lattice has cavities and every one of them is at zero voltage."""
        return bool(self._cavities) and all(
            float(self.lattice[index].Voltage) == 0.0 for index in self._cavities
        )

    def _rotated(self) -> at.Lattice:
        """The lattice rotated to start at the injection marker."""
        return self.lattice.rotate(int(self.injection.index))

    def _captured(self, count: int, turns: int) -> float:
        """The share of shot ``count``'s bunch alive at the end of ``turns`` turns."""
        lattice = self._rotated()
        end = np.array([len(lattice)], dtype=np.uint32)
        fraction, _, _ = _bunch_reading(lattice, self._bunch_at(count), end, turns)
        return float(fraction[0, -1])

    def _bunch_at(self, count: int) -> np.ndarray:
        """The particles shot ``count`` launches, ``(6, particles)``.

        The launch, plus the bunch's fixed shape, plus a centroid jitter in
        ``x`` and ``y``: the jitter sigmas times the first two normals of the
        keyed stream of the model and the shot. Nothing else in a bunch
        varies from shot to shot.
        """
        from osprey_connectors.simulation.series import keyed_normals

        bunch = self._bunch
        assert bunch is not None
        launch = np.array(self.injection.launch, dtype=np.float64).reshape(6, 1)
        jitter = np.zeros((6, 1), dtype=np.float64)
        if bunch.jitter != (0.0, 0.0):
            normals = keyed_normals(bunch_key(bunch.key, count), np.arange(2))
            jitter[0, 0] = bunch.jitter[0] * normals[0]
            jitter[2, 0] = bunch.jitter[1] * normals[1]
        return np.asarray(launch + jitter + bunch.shape, dtype=np.float64)

    def _track_shot(
        self, count: int, turns: int, stored: float = 0.0
    ) -> tuple[dict[str, np.ndarray], float | None]:
        """Every series, filled by the shot tracked from the injection.

        A periodic model tracks the lattice from its marker for ``turns``
        turns, one sample per turn from the trigger sample; a single-pass
        model tracks its line once, for one sample. Without a bunch one
        particle is tracked, and a sample's physics sum is 1 where it passed.
        With one, every particle of shot ``count`` is tracked in the same
        pass: a sample's physics sum is the share ``f`` of the bunch alive
        there, and its positions are the survivors' centroid. The samples
        before the trigger, every one past the tracked window, and every one
        no particle reaches carry no injected beam.

        A bunch's monitors also see the ``stored`` charge, while the lattice
        has a closed orbit: every sample's physics sum is ``f + Q_s/Q_i``, the
        stored charge over the bunch charge, and its positions the
        charge-weighted centroid of the survivors and of the stored beam on
        the closed orbit, as the solve found it before any readout fault. A
        sample with neither reads the no-beam reading: a physics sum of 0. No
        sample is left at zero.

        A position is read by its monitor before its record publishes it, with
        the readout faults and attributes the monitor's element carries as the
        batch left them, once; the noise is drawn for shot ``count``.

        Returns:
            The series by address, and the share of a bunch alive at the end
            of the tracked window (``None`` without a bunch).

        Raises:
            Whatever pyAT raises for a lattice it cannot track.
        """
        buffers: dict[str, np.ndarray] = {}
        if not self._series:
            return buffers, None if self._bunch is None else self._captured(count, turns)
        first = int(self.injection.trigger_sample)
        order = sorted(self._series_index, key=self._series_index.__getitem__)
        if self.solve == SINGLE_PASS:
            lattice, turns = self.lattice, 1
            indexes = [self._series_index[monitor] for monitor in order]
        else:
            start = int(self.injection.index)
            lattice = self._rotated()
            order.sort(key=lambda monitor: (self._series_index[monitor] - start) % len(lattice))
            indexes = [(self._series_index[monitor] - start) % len(lattice) for monitor in order]
        fraction: np.ndarray | None = None
        captured: float | None = None
        if self._bunch is None:
            refpts = np.array(indexes, dtype=np.uint32)
            launch = np.array(self.injection.launch, dtype=np.float64).reshape(6, 1)
            tracked, _, _ = at.lattice_track(
                lattice, launch, nturns=turns, refpts=refpts, losses=True
            )
            track_x, track_y = tracked[0, 0], tracked[2, 0]
        else:
            # One more reading, at the end of each turn, judges the capture.
            refpts = np.array([*indexes, len(lattice)], dtype=np.uint32)
            fraction, track_x, track_y = _bunch_reading(
                lattice, self._bunch_at(count), refpts, turns
            )
            captured = float(fraction[-1, -1])
        stored_share = (
            stored / self._bunch.charge
            if self._bunch is not None and stored > 0.0 and not self.no_closed_orbit
            else 0.0
        )
        orbit = self.simulator.last_solution if stored_share else {}
        column = {monitor: place for place, monitor in enumerate(order)}
        window = slice(first, first + turns)
        for name, variable in self._series.items():
            x = np.full(self._samples, np.nan, dtype=np.float64)
            y = np.full(self._samples, np.nan, dtype=np.float64)
            x[window] = track_x[column[variable.monitor], :]
            y[window] = track_y[column[variable.monitor], :]
            if fraction is None:
                sums = np.where(np.isnan(x), 0.0, 1.0)
            else:
                sums = np.zeros(self._samples, dtype=np.float64)
                sums[window] = fraction[column[variable.monitor], :]
            if stored_share:
                x, y = _with_stored_beam(x, y, sums, stored_share, orbit[variable.monitor])
                sums = sums + stored_share
            if variable.axis == "sum":
                physics = shot_sum(self, variable.monitor, sums)
            else:
                physics = shot_read(
                    self,
                    variable.monitor,
                    variable.axis,
                    x,
                    y,
                    address=name,
                    shot=count,
                    first=0,
                    sums=sums,
                )
            buffers[name] = samples_to_hardware(variable.calibration, variable.inverse, physics)
        return buffers, captured

    # -- the optics arrays, which bind no element and are read on demand ----

    def _validate_binding(self, name: str, variable: Variable) -> None:
        """Check a variable's binding; an optics array, a readback and the shot's bind none."""
        if isinstance(
            variable,
            (
                PyATReadOnlyNDVariable,
                ReadbackVariable,
                CommandVariable,
                ArmedVariable,
                TurnsTrackedVariable,
                StoredCurrentVariable,
            ),
        ):
            return
        super()._validate_binding(name, variable)

    def _read_outputs(self) -> dict[str, float]:
        """Every per-monitor reading off the last solve, never the optics or a readback.

        This runs after every solve, so leaving the optics out is what keeps a
        setpoint write from paying for them; :meth:`_get` computes them, and
        every readback, when one is read.

        A batch that takes a shot takes it here: after the batch's solve, and
        inside the write the base class rolls back on any failure. A batch
        that leaves a bunch's lattice with no closed orbit while RF is on
        loses the stored charge with it.
        """
        outputs = {
            name: variable._get(self.simulator)
            for name, variable in self.supported_variables.items()
            if not isinstance(
                variable,
                (
                    WritableActionMixin,
                    PyATReadOnlyNDVariable,
                    ReadbackVariable,
                    ArmedVariable,
                    StoredCurrentVariable,
                ),
            )
        }
        shot = self._pending_shot
        if shot is not None and not self._restoring:
            if self._bunch is not None and self.no_closed_orbit and not self._rf_off():
                shot.stored = 0.0
            self._fire(shot)
        return outputs

    def _next_shot_time(self) -> float:
        """The time a shot taken now is stamped with: now, or just after the last shot."""
        now = time.time()
        last = self._shot_time
        return now if last is None or now > last else math.nextafter(last, math.inf)

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return one value per name, in the order asked.

        Optics arrays come from :meth:`_optics` and :meth:`_chromaticity`, as
        copies, so a caller that edits one cannot alter what the next read
        returns; the chromatic solve runs only when ``names`` holds
        ``chromaticity`` or a readback of it. A readback is computed from its
        source. A turn series is a copy of what the last tracked shot left,
        an armed reading is the flag, 1 or 0, and a stored-current reading is
        the stored charge times the deck's revolution frequency, through its
        record's calibration. Everything else is the base class's cached
        answer.

        Raises:
            UnknownElementError: a name is not a variable of this model.
        """
        held = self._series.keys() | self._armed_names | self._stored_currents.keys()
        if any(name in held for name in names):
            others = self._get([name for name in names if name not in held])
            return {
                name: self._buffers[name].copy()
                if name in self._series
                else float(self._armed)
                if name in self._armed_names
                else self._stored_current(name)
                if name in self._stored_currents
                else others[name]
                for name in names
            }
        readbacks = {name: self._readbacks[name] for name in names if name in self._readbacks}
        wanted = {name for name in names if name in self._optics_names}
        wanted.update(
            readback.source for readback in readbacks.values() if readback.source in PLANE_OPTICS
        )
        if not wanted and not readbacks:
            return super()._get(names)
        arrays: dict[str, np.ndarray] = {}
        if not LINEAR_OPTICS.isdisjoint(wanted):
            arrays.update(self._optics())
        if CHROMATICITY in wanted:
            arrays[CHROMATICITY] = self._chromaticity()
        cached = super()._get(
            [name for name in names if name not in wanted and name not in readbacks]
        )
        values: dict[str, Any] = {}
        for name in names:
            if name in wanted:
                values[name] = arrays[name].copy()
            elif name in readbacks:
                values[name] = self._readback(readbacks[name], arrays)
            else:
                values[name] = cached[name]
        return values

    @property
    def stored_charge(self) -> float:
        """The charge stored in the model, in the unit of ``stored_charge``."""
        return self._stored

    def _stored_current(self, name: str) -> float:
        """One stored-current reading: the stored charge's current, as its record publishes it."""
        reading = self._stored_currents[name]
        current = np.array([self._stored * self._revolution], dtype=np.float64)
        return float(samples_to_hardware(reading.calibration, reading.inverse, current)[0])

    def _readback(
        self, readback: ReadbackVariable | ArrayReadbackVariable, arrays: Mapping[str, np.ndarray]
    ) -> Any:
        """One readback's value, from the optics in ``arrays`` or the held setpoints."""
        if isinstance(readback, ArrayReadbackVariable):
            return arrays[readback.source].copy()
        if readback.source is None:
            return float(readback.default_value or 0.0)
        if readback.source in PLANE_OPTICS:
            return float(arrays[readback.source][readback.component])
        if readback.source in self._knobs or readback.source == self._turns_name:
            return float(self._inputs[readback.source])
        setpoint = self._setpoints[readback.source]
        return setpoint.readback(self._inputs[readback.source])

    def _optics(self) -> dict[str, np.ndarray]:
        """The optics arrays of the last solve, computed at most once per solve.

        The memo is keyed on the identity of the simulator's last solution.
        Every successful solve makes a new solution object, and a rolled-back
        write puts the prior object back along with the prior lattice, so an
        identity match means the same lattice solve. The memo holds that
        object, not merely its ``id``, so it cannot be freed and its id handed
        to a later solve.

        A single-pass model's beta functions are the ones its solve
        propagated from ``twiss_in``, and it has no tunes. A lattice with no
        closed orbit has no optics either: every array is NaN.

        Raises:
            OrbitSolveError: no solve has succeeded yet.
        """
        solution = self.simulator.last_solution
        if self._optics_memo is not None and self._optics_memo[0] is solution:
            return self._optics_memo[1]
        orbit = np.array([solution[element] for element in self._monitor_order], dtype=np.float64)
        if self.no_closed_orbit:
            # pyAT raises on a lattice with no closed orbit, so it is not asked.
            optics = {
                TUNES: np.full((self._planes,), np.nan, dtype=np.float64),
                BETA_AT_MONITORS: np.full((len(self._monitor_order), 2), np.nan, dtype=np.float64),
                ORBIT_AT_MONITORS: orbit.reshape(len(self._monitor_order), 2),
            }
            self._optics_memo = (solution, optics)
            return optics
        if self.solve == SINGLE_PASS:
            optics = {
                BETA_AT_MONITORS: np.array(
                    [solution.beta[element] for element in self._monitor_order],  # type: ignore[attr-defined]
                    dtype=np.float64,
                ),
                ORBIT_AT_MONITORS: orbit,
            }
            self._optics_memo = (solution, optics)
            return optics
        _, lattice_data, element_data = at.get_optics(self.lattice, refpts=self._monitor_refpts)
        optics = {
            TUNES: np.array(lattice_data.tune[: self._planes], dtype=np.float64),
            BETA_AT_MONITORS: np.array(element_data.beta, dtype=np.float64),
            ORBIT_AT_MONITORS: orbit,
        }
        self._optics_memo = (solution, optics)
        return optics

    def _chromaticity(self) -> np.ndarray:
        """The chromaticity of the last solve, solved at most once per solve.

        Keyed on the simulator's last solution exactly as :meth:`_optics` is,
        and kept apart from it, so a read of the linear optics never pays for
        the chromatic solve. A lattice with no closed orbit has no
        chromaticity: both planes are NaN.

        Raises:
            OrbitSolveError: no solve has succeeded yet.
        """
        solution = self.simulator.last_solution
        if self._chromaticity_memo is not None and self._chromaticity_memo[0] is solution:
            return self._chromaticity_memo[1]
        if self.no_closed_orbit:
            chromaticity = np.full((self._planes,), np.nan, dtype=np.float64)
            self._chromaticity_memo = (solution, chromaticity)
            return chromaticity
        _, lattice_data, _ = at.get_optics(self.lattice, get_chrom=True)
        chromaticity = np.array(lattice_data.chromaticity[: self._planes], dtype=np.float64)
        self._chromaticity_memo = (solution, chromaticity)
        return chromaticity
