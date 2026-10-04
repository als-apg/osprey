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

**Optics are read-only arrays, computed on read.** ``tunes`` (the fractional
tunes: the two transverse ones, and the synchrotron tune third on a deck with
longitudinal motion), ``chromaticity`` (one per tune), ``beta_at_monitors``
and ``orbit_at_monitors`` (one ``(x, y)`` row per monitor element, in lattice
order) sit beside the per-monitor readings. The base class re-reads every
read-only output after every solve, so a setpoint write would pay for an
optics pass nobody asked for; these are left out of that re-read. The linear
ones are computed together the first time one is read after a solve, then
served from memory until the next. The chromaticity needs a chromatic solve,
far dearer than the linear pass, so it has a memo of its own and is solved
only for a read that names it.

Declared defaults are recorded as the retained input values at construction
but are not written to the lattice, so the lattice boots exactly as handed
over; :meth:`reset` writes them back through the calibrations. This module
never touches a control system and never raises ``SystemExit``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import at
import numpy as np
from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ConfigEnum, ScalarVariable
from lume_pyat.actions import ElementBinding, PyATWritableScalarVariable
from lume_pyat.exceptions import UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator
from pydantic import ConfigDict

from osprey.simulation.engines.pyat_faults import SUPPLY_IDENTITY, supply_attribute
from osprey.simulation.engines.pyat_variables import (
    KICK_ATTRIBUTE,
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
    "PyATLatticeModel",
    "ReadbackVariable",
]

#: What separates a wired address from a field in a fault name.
FAULT_SEPARATOR = "/"

#: The engine attributes whose setpoints carry a calibration fault.
CALIBRATED_ATTRIBUTES: frozenset[str] = frozenset({"PolynomB", KICK_ATTRIBUTE})

# Element-attribute prefix of a readout fault. No pyAT pass method reads an
# attribute under it, which is what keeps a fault off the orbit.
_READOUT_PREFIX = "readout_"

#: The readout fields every monitor readback carries on its own axis, each at
#: its identity: a monitor that reports the true orbit position exactly.
READOUT_IDENTITY: dict[str, float] = {
    "offset": 0.0,
    "gain": 1.0,
    "noise": 0.0,
    "polarity": 1.0,
}

#: The roll a monitor carries once, at its identity.
ROLL = "roll"
ROLL_IDENTITY = 0.0
_ROLL_UNIT = "rad"

#: The calibration fields a magnet setpoint carries, each at its identity: a
#: magnet that delivers exactly what it was commanded.
CALIBRATION_IDENTITY: dict[str, float] = SUPPLY_IDENTITY

MIN_MONITOR_GAIN = 0.1
MAX_MONITOR_GAIN = 10.0
MAX_MONITOR_ROLL_RAD = 0.1
#: A factor of -1 is a polarity flip; a magnitude beyond 5x is never a real
#: calibration error.
MAX_CALIBRATION_FACTOR = 5.0

#: Fault field -> its inclusive (min, max). A field absent here is bounded by
#: well-formedness alone.
FAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "gain": (MIN_MONITOR_GAIN, MAX_MONITOR_GAIN),
    "noise": (0.0, math.inf),
    ROLL: (-MAX_MONITOR_ROLL_RAD, MAX_MONITOR_ROLL_RAD),
    "cal_factor": (-MAX_CALIBRATION_FACTOR, MAX_CALIBRATION_FACTOR),
}

#: A polarity is a direction: it lands exactly on one of these two values.
POLARITY_OPTIONS: tuple[float, float] = (-1.0, 1.0)

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

FaultVariable = PyATWritableScalarVariable | PyATWritableEnumVariable

# A solve (the simulator's last solution object), and the optics computed for
# it.
_OpticsMemo = tuple[Any, dict[str, np.ndarray]]
_ChromaticityMemo = tuple[Any, np.ndarray]


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
        for field, identity in CALIBRATION_IDENTITY.items():
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


def _optics_variables(monitor_count: int, planes: int) -> list[PyATReadOnlyNDVariable]:
    """Declare the optics arrays.

    The tunes and the chromaticity hold one value per plane; per-monitor
    arrays hold one ``(x, y)`` row each.
    """
    per_monitor = (monitor_count, 2)
    return [
        PyATReadOnlyNDVariable(name=TUNES, shape=(planes,)),
        PyATReadOnlyNDVariable(name=CHROMATICITY, shape=(planes,)),
        PyATReadOnlyNDVariable(name=BETA_AT_MONITORS, shape=per_monitor, unit="m"),
        PyATReadOnlyNDVariable(name=ORBIT_AT_MONITORS, shape=per_monitor, unit="m"),
    ]


class ReadbackVariable(ReadOnlyActionMixin[PyATSimulator], ScalarVariable):
    """A wired readback the model serves from another of its variables.

    It binds no element: the model computes it on read from ``source``, so it
    follows every write of that source and every :meth:`reset`.

    Attributes:
        source: A setpoint's address, read back through the setpoint's own
            calibration at the hardware value it holds; ``tunes`` or
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


def _check_readbacks(channels: Iterable[Variable], planes: int) -> None:
    """Refuse a readback whose source the model does not serve.

    Raises:
        ValueError: a readback names a source that is neither a setpoint
            among ``channels`` nor ``tunes``/``chromaticity``, or an optics
            plane the deck does not have.
    """
    setpoints = {variable.name for variable in channels if isinstance(variable, CalibratedSetpoint)}
    for variable in channels:
        if not isinstance(variable, ReadbackVariable) or variable.source is None:
            continue
        if variable.source in PLANE_OPTICS:
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

        Raises:
            ValueError: a seed names a fault the model does not declare, or a
                fault or optics name is already a channel address.
            pydantic.ValidationError: a seed lies outside its field's bounds,
                or a polarity seed is neither of its two values.
            UnknownElementError: a variable binds an element the lattice does
                not have.
            OrbitSolveError: the lattice as handed over, with its
                misalignments, has no stable closed orbit.
        """
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
                f"{sorted(CALIBRATION_IDENTITY)}"
            )
        planes = tune_planes(lattice)
        _check_readbacks(channels, planes)
        optics = _optics_variables(len(monitors), planes)
        _refuse_name_collisions(channels, [*declared, *optics])
        _seed_fault_attributes(lattice, declared)

        # lume's ActionVariable is a union of hinting stubs in lume/actions.py
        # that no variable class inherits.
        variables: list[ActionVariable[PyATSimulator]] = [*channels, *declared, *optics]  # type: ignore[list-item]
        super().__init__(
            simulator=PyATSimulator(
                lattice,
                element_misalignments=(
                    None
                    if element_misalignments is None
                    else {name: dict(kwargs) for name, kwargs in element_misalignments.items()}
                ),
            ),
            action_variables=variables,
        )

        #: The variables this model declares that no channel addresses: the
        #: faults and the optics.
        self.derived_names: frozenset[str] = frozenset(
            variable.name for variable in (*declared, *optics)
        )
        self._fault_names: frozenset[str] = frozenset(fault.name for fault in declared)

        # Sorted by lattice index, so row i of every per-monitor optics array
        # is the i-th monitor along the lattice.
        self._monitor_order: list[str] = sorted(monitors, key=self.element_index)
        self._monitor_refpts = np.array(
            [self.element_index(element) for element in self._monitor_order], dtype=np.uint32
        )
        self._planes = planes
        self._readbacks: dict[str, ReadbackVariable] = {
            variable.name: variable
            for variable in channels
            if isinstance(variable, ReadbackVariable)
        }
        self._setpoints: dict[str, CalibratedSetpoint] = {
            variable.name: variable
            for variable in channels
            if isinstance(variable, CalibratedSetpoint)
        }
        self._optics_memo: _OpticsMemo | None = None
        self._chromaticity_memo: _ChromaticityMemo | None = None

    def _set(self, values: dict[str, Any]) -> None:
        """Apply a batch atomically, its faults before its channel variables.

        A setpoint is converted through the calibration its element holds when
        it is written, so a batch that names a setpoint and its calibration --
        and :meth:`reset`, which names every writable -- writes the
        calibration first, whatever order the mapping gives. Each part keeps
        its own order.
        """
        faults = {name: value for name, value in values.items() if name in self._fault_names}
        rest = {name: value for name, value in values.items() if name not in self._fault_names}
        super()._set({**faults, **rest})

    # -- the optics arrays, which bind no element and are read on demand ----

    def _validate_binding(self, name: str, variable: Variable) -> None:
        """Check a variable's binding; an optics array or a readback binds none."""
        if isinstance(variable, (PyATReadOnlyNDVariable, ReadbackVariable)):
            return
        super()._validate_binding(name, variable)

    def _read_outputs(self) -> dict[str, float]:
        """Every per-monitor reading off the last solve, never the optics or a readback.

        This runs after every solve, so leaving the optics out is what keeps a
        setpoint write from paying for them; :meth:`_get` computes them, and
        every readback, when one is read.
        """
        return {
            name: variable._get(self.simulator)
            for name, variable in self.supported_variables.items()
            if not isinstance(
                variable, (WritableActionMixin, PyATReadOnlyNDVariable, ReadbackVariable)
            )
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return one value per name, in the order asked.

        Optics arrays come from :meth:`_optics` and :meth:`_chromaticity`, as
        copies, so a caller that edits one cannot alter what the next read
        returns; the chromatic solve runs only when ``names`` holds
        ``chromaticity`` or a readback of it. A readback is computed from its
        source. Everything else is the base class's cached answer.

        Raises:
            UnknownElementError: a name is not a variable of this model.
        """
        readbacks = {name: self._readbacks[name] for name in names if name in self._readbacks}
        wanted = {name for name in names if name in OPTICS_NAMES}
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
            [name for name in names if name not in OPTICS_NAMES and name not in readbacks]
        )
        values: dict[str, Any] = {}
        for name in names:
            if name in OPTICS_NAMES:
                values[name] = arrays[name].copy()
            elif name in readbacks:
                values[name] = self._readback(readbacks[name], arrays)
            else:
                values[name] = cached[name]
        return values

    def _readback(self, readback: ReadbackVariable, arrays: Mapping[str, np.ndarray]) -> float:
        """One readback's value, from the optics in ``arrays`` or the held setpoints."""
        if readback.source is None:
            return float(readback.default_value or 0.0)
        if readback.source in PLANE_OPTICS:
            return float(arrays[readback.source][readback.component])
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

        Raises:
            OrbitSolveError: no solve has succeeded yet.
        """
        solution = self.simulator.last_solution
        if self._optics_memo is not None and self._optics_memo[0] is solution:
            return self._optics_memo[1]
        _, lattice_data, element_data = at.get_optics(self.lattice, refpts=self._monitor_refpts)
        optics = {
            TUNES: np.array(lattice_data.tune[: self._planes], dtype=np.float64),
            BETA_AT_MONITORS: np.array(element_data.beta, dtype=np.float64),
            ORBIT_AT_MONITORS: np.array(
                [solution[element] for element in self._monitor_order], dtype=np.float64
            ),
        }
        self._optics_memo = (solution, optics)
        return optics

    def _chromaticity(self) -> np.ndarray:
        """The chromaticity of the last solve, solved at most once per solve.

        Keyed on the simulator's last solution exactly as :meth:`_optics` is,
        and kept apart from it, so a read of the linear optics never pays for
        the chromatic solve.

        Raises:
            OrbitSolveError: no solve has succeeded yet.
        """
        solution = self.simulator.last_solution
        if self._chromaticity_memo is not None and self._chromaticity_memo[0] is solution:
            return self._chromaticity_memo[1]
        _, lattice_data, _ = at.get_optics(self.lattice, get_chrom=True)
        chromaticity = np.array(lattice_data.chromaticity[: self._planes], dtype=np.float64)
        self._chromaticity_memo = (solution, chromaticity)
        return chromaticity
