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
- and three optics arrays of the solved lattice, computed on read.

**Faults are variables, held as state on the element.** Each is named
``<address>/<field>`` after the wired address it perturbs, so a fault name is
never a channel address and the two rosters cannot collide. A monitor readback
carries ``offset``, ``gain``, ``noise`` and ``polarity`` on its own axis, and a
monitor carries one ``roll``, on its x-axis readback or its only readback. A
setpoint on ``PolynomB`` or ``KickAngle`` carries ``cal_factor`` and
``cal_offset``. Each fault is bound to an element attribute no pyAT pass
method reads (``readout_<field>_<axis>``, ``readout_roll``,
``supply_cal_factor``, ``supply_cal_offset``), so seeding or writing a fault
changes what the element carries and never the orbit; applying it to a
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

**One element carries one calibration.** Setpoints that share an element,
directly or through another setpoint, carry one calibration over all their
elements, named after the first of them in the order given, and it scales
them all, which is what a miscalibrated magnet does; an offset that shifts
them all can only be in one unit, so they must agree on it.

**Optics are read-only arrays, computed on read.** ``tunes`` (the two
fractional transverse tunes), ``beta_at_monitors`` and ``orbit_at_monitors``
(one ``(x, y)`` row per monitor element, in lattice order) sit beside the
per-monitor readings. The base class re-reads every read-only output after
every solve, so a setpoint write would pay for a linear optics pass nobody
asked for; these three are left out of that re-read and computed together the
first time one is read after a solve, then served from memory until the next.

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
from lume.actions import WritableActionMixin
from lume.variables import ConfigEnum
from lume_pyat.actions import ElementBinding, PyATWritableScalarVariable
from lume_pyat.exceptions import UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator

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
    "FAULT_SEPARATOR",
    "OPTICS_NAMES",
    "ORBIT_AT_MONITORS",
    "TUNES",
    "PyATLatticeModel",
]

#: What separates a wired address from a field in a fault name.
FAULT_SEPARATOR = "/"

#: The engine attributes whose setpoints carry a calibration fault.
CALIBRATED_ATTRIBUTES: frozenset[str] = frozenset({"PolynomB", KICK_ATTRIBUTE})

# Element-attribute prefix per fault kind. No pyAT pass method reads an
# attribute under either, which is what keeps a fault off the orbit.
_READOUT_PREFIX = "readout_"
_SUPPLY_PREFIX = "supply_"

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
CALIBRATION_IDENTITY: dict[str, float] = {"cal_factor": 1.0, "cal_offset": 0.0}

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
BETA_AT_MONITORS = "beta_at_monitors"
ORBIT_AT_MONITORS = "orbit_at_monitors"
OPTICS_NAMES: frozenset[str] = frozenset({TUNES, BETA_AT_MONITORS, ORBIT_AT_MONITORS})

FaultVariable = PyATWritableScalarVariable | PyATWritableEnumVariable

# A solve (the simulator's last solution object), and the optics computed for
# it.
_OpticsMemo = tuple[Any, dict[str, np.ndarray]]


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
    """Declare one calibration per group of magnet setpoints sharing elements.

    Setpoints that share any element, directly or through another setpoint,
    form one group. The group's first setpoint in the order given names its
    calibration, which binds every element of the group.

    Raises:
        ValueError: the setpoints of one group are commanded in different
            units.
    """
    calibrated = [
        variable
        for variable in channels
        if isinstance(variable, CalibratedSetpoint)
        and variable.bindings[0].attribute in CALIBRATED_ATTRIBUTES
    ]
    # Each group is the positions of its setpoints, in the order given, and
    # its elements, in the order first bound.
    groups: list[tuple[list[int], dict[str, None]]] = []
    for position, variable in enumerate(calibrated):
        members = [position]
        elements = dict.fromkeys(binding.element_name for binding in variable.bindings)
        apart = []
        for group in groups:
            if elements.keys().isdisjoint(group[1]):
                apart.append(group)
            else:
                members = [*group[0], *members]
                elements = {**group[1], **elements}
        groups = [*apart, (sorted(members), elements)]
    groups.sort(key=lambda group: group[0][0])

    faults: list[FaultVariable] = []
    for members, elements in groups:
        setpoints = [calibrated[position] for position in members]
        units = {setpoint.unit for setpoint in setpoints}
        if len(units) > 1:
            raise ValueError(
                f"the setpoints {sorted(setpoint.name for setpoint in setpoints)} share "
                f"elements of {sorted(elements)} but are commanded in "
                f"{sorted(unit or '<none>' for unit in units)}; one element carries one "
                f"calibration, and an offset that shifts them all can only be in one unit"
            )
        owner = setpoints[0]
        for field, identity in CALIBRATION_IDENTITY.items():
            name = _fault_name(owner.name, field)
            faults.append(
                PyATWritableScalarVariable(
                    name=name,
                    bindings=[
                        ElementBinding(element_name=element, attribute=_SUPPLY_PREFIX + field)
                        for element in elements
                    ],
                    default_value=seeds.get(name, identity),
                    value_range=FAULT_BOUNDS.get(field),
                    unit=owner.unit if field == "cal_offset" else None,
                    default_validation_config=ConfigEnum.ERROR,
                )
            )
    return faults


def _optics_variables(monitor_count: int) -> list[PyATReadOnlyNDVariable]:
    """Declare the optics arrays. Per-monitor arrays hold one ``(x, y)`` row each."""
    per_monitor = (monitor_count, 2)
    return [
        PyATReadOnlyNDVariable(name=TUNES, shape=(2,)),
        PyATReadOnlyNDVariable(name=BETA_AT_MONITORS, shape=per_monitor, unit="m"),
        PyATReadOnlyNDVariable(name=ORBIT_AT_MONITORS, shape=per_monitor, unit="m"),
    ]


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
    stand the settable fault state of every wired monitor and magnet, and three
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
            ValueError: a seed names a fault the model does not declare, a
                fault or optics name is already a channel address, or the
                setpoints sharing a calibration disagree on their unit.
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
        optics = _optics_variables(len(monitors))
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

        # Sorted by lattice index, so row i of every per-monitor optics array
        # is the i-th monitor along the lattice.
        self._monitor_order: list[str] = sorted(monitors, key=self.element_index)
        self._monitor_refpts = np.array(
            [self.element_index(element) for element in self._monitor_order], dtype=np.uint32
        )
        self._optics_memo: _OpticsMemo | None = None

    # -- the optics arrays, which bind no element and are read on demand ----

    def _validate_binding(self, name: str, variable: Variable) -> None:
        """Check a variable's binding; an optics array binds none to check."""
        if isinstance(variable, PyATReadOnlyNDVariable):
            return
        super()._validate_binding(name, variable)

    def _read_outputs(self) -> dict[str, float]:
        """Every per-monitor reading off the last solve, and never the optics.

        This runs after every solve, so leaving the optics out is what keeps a
        setpoint write from paying for them; :meth:`_get` computes them when
        one is read.
        """
        return {
            name: variable._get(self.simulator)
            for name, variable in self.supported_variables.items()
            if not isinstance(variable, (WritableActionMixin, PyATReadOnlyNDVariable))
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return one value per name, in the order asked.

        Optics arrays come from :meth:`_optics`, as copies, so a caller that
        edits one cannot alter what the next read returns. Everything else is
        the base class's cached answer.

        Raises:
            UnknownElementError: a name is not a variable of this model.
        """
        if OPTICS_NAMES.isdisjoint(names):
            return super()._get(names)
        optics = self._optics()
        cached = super()._get([name for name in names if name not in OPTICS_NAMES])
        return {
            name: optics[name].copy() if name in OPTICS_NAMES else cached[name] for name in names
        }

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
            # The two transverse tunes. A lattice solved with longitudinal
            # motion on reports a third, and the synchrotron tune is not a
            # betatron one -- the array says which two it holds.
            TUNES: np.array(lattice_data.tune[:2], dtype=np.float64),
            BETA_AT_MONITORS: np.array(element_data.beta, dtype=np.float64),
            ORBIT_AT_MONITORS: np.array(
                [solution[element] for element in self._monitor_order], dtype=np.float64
            ),
        }
        self._optics_memo = (solution, optics)
        return optics
