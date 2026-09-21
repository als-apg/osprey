"""The served ring as a ``lume-pyat`` model.

:class:`PyATRingModel` is the facility adapter, and only that. Everything
generic about serving a pyAT ring through the LUME contract -- owning one
persistent lattice, atomic multi-variable writes, one solve per batch,
rollback on a lost closed orbit, retained inputs and cached outputs --
belongs to :class:`~lume_pyat.model.LUMEPyATModel` and is inherited rather
than restated here. What is left is the four facility-specific facts that
class cannot know, and all of them are read out of the tree this model is
served:

- which lattice to drive, and whether it is still the one the bindings were
  derived against
  (:func:`~osprey.services.virtual_accelerator.lattice.build_ring`),
- which variables exist, what each is bound to and how a commanded hardware
  value becomes a physics one
  (:func:`~osprey.services.virtual_accelerator.model.bindings.build_action_variables`
  over the served ``va_bindings.json``, through
  :func:`~osprey.services.virtual_accelerator.model.catalog.build_variable_catalog`),
- which faults every monitor and magnet the tree serves can carry, and what
  each was seeded with,
- and how a boot failure should read.

**One served tree, named once.** ``data_dir`` is a facility's ``data/``
directory and every file this model reads is resolved against it through
:class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`:
the lattice, the bindings, the ``machine.json`` nominals and the
``channel_limits.json`` bands. There is no default and no fallback -- a
standalone demo names the packaged tree explicitly, like any other -- because
the alternative is serving the framework's own demo nominals and bands behind
a facility's addresses, and the band a nominal is weighed against is the one
thing a facility must recognise as its own.

**The channel list is passed in, not resolved here.** It is the set of
channels the deployment resolved and the IOC is serving on, and the model has
to be on exactly that namespace: resolving a second one would let the two
drift apart silently. The tree is not asked for it either, because an emitted
tree carries no manifest at all -- the channel set is derived at build time
from the facility's databases.

**Faults are variables, held as state on the element.** Every monitor the
document publishes a reading for carries nine reading-error fields, and every
magnet it drives a calibration factor and offset: one writable each, named in
dot grammar (``<element>.offset_x``, ``<element>.cal_factor``) so no fault
name is a channel address and the two rosters cannot collide. Each is bound
to an element attribute no pyAT pass method reads, so seeding or writing a
fault changes what the element carries and never the orbit; applying it to a
reading or to a commanded value is the serving layer's work. A seed is its
variable's ``default_value`` and is written onto the element from that same
value, so the two cannot disagree, and :meth:`reset` returns a fault to its
seed rather than to identity. Seeds are checked in full -- device, then
field, then range -- before the ring is touched.

The roster is the served one: a device carries a fault when the document
binds it *and* this deployment's channel set carries the address, so every
fault stands behind something a client can observe. A magnet is a device the
document drives a strength or a kick on; a cavity is neither, and the ring
energy binds no element at all. Where one element backs two setpoints, one
calibration scales both, which is what a miscalibrated magnet does.

**Optics are read-only arrays, computed on read.** Three quantities of the
solved ring sit beside the per-monitor readings: ``tunes`` (the two fractional
betatron tunes), ``beta_at_monitors`` and ``orbit_at_monitors`` (one ``(x, y)``
row per monitor element, in ring order). The orbit is the true, un-faulted
position the reading variables are solved from. The base class re-reads every
read-only output after every solve, so a setpoint write would pay for a linear
optics pass nobody asked for; these three are left out of that re-read and
computed together the first time one is read after a solve, then served from
memory until the next solve.

**The energy knob has to be coupled, and that happens here.** The catalog
hands out one variable per address, so nothing inside it can see the finished
set; :func:`~osprey.services.virtual_accelerator.model.bindings.couple_energy_knob`
adopts every rigidity-scaled setpoint into the knob once they all exist,
which is after the catalog is built and before the model takes the variables
over. Skipping it would leave a knob that writes electron-volts and rescales
nothing, with the fields a rescale touches outside the model's rollback.

Declared defaults are recorded as the retained input values at construction
but are not written to the lattice, so the ring boots in exactly the state
``build_ring()`` produced -- the served ``machine.json`` nominals the catalog
carries as ``default_value`` *are* that state. :meth:`reset` writes them back
through the calibrations, which is what makes a reset undo writes without
disturbing construction-time faults.

This module imports nothing from ``ioc`` and never touches EPICS, and -- like
the base class -- never raises ``SystemExit``: killing the host process is
the serving layer's decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import at
import numpy as np
from lume.actions import WritableActionMixin
from lume_pyat.actions import ElementBinding, PyATWritableScalarVariable
from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.bindings import (
    build_action_variables,
    couple_energy_knob,
)
from osprey.services.virtual_accelerator.model.catalog import build_variable_catalog
from osprey.services.virtual_accelerator.model.fault_bounds import (
    BPM_ERROR_FIELD_BOUNDS,
    BPM_ERROR_FIELDS,
    BPM_ERROR_IDENTITY,
    BPM_NOISE_BOUNDS,
    BPM_NOISE_FIELDS,
    BPM_POLARITY_FIELDS,
    BPM_POLARITY_OPTIONS,
    CORRECTOR_GAIN_FIELDS,
    MAGNET_CAL_BOUNDS,
    MAGNET_CAL_FIELDS,
    MAGNET_CAL_IDENTITY,
)
from osprey.services.virtual_accelerator.model.variables import (
    PyATReadOnlyNDVariable,
    PyATWritableEnumVariable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping
    from pathlib import Path

    from lume.variables import ScalarVariable, Variable

    from osprey.services.virtual_accelerator.bindings import BindingsDocument

#: The binding kinds a calibration fault belongs to: a device commanded in
#: hardware units whose delivered value the calibration scales and shifts. A
#: cavity is driven the same way but is no magnet, and the ring energy binds
#: no element at all.
MAGNET_KINDS: frozenset[str] = frozenset({"strength", "kick"})
MONITOR_KIND = "monitor"

#: What separates an element from a field in a fault name. An address is the
#: facility's, whatever its grammar; a fault name is an element and a field
#: with this between them, and a collision between the two rosters is refused
#: rather than resolved.
FAULT_SEPARATOR = "."

# Element-attribute prefix per fault kind. No pyAT pass method reads an
# attribute under either, which is what keeps a fault off the orbit.
_MONITOR_ATTRIBUTE_PREFIX = "readout_"
_MAGNET_ATTRIBUTE_PREFIX = "supply_"

#: The unit of a roll. Every other dimensioned fault takes the unit of the
#: variable it perturbs, because that unit is the facility's own.
_ROLL_UNIT = "rad"

#: The reading-error fields whose magnitude is in the monitor's own unit, and
#: the axis each is read on.
_AXIS_OF_FIELD: dict[str, str] = {
    "offset_x": "x",
    "offset_y": "y",
    "noise_x": "x",
    "noise_y": "y",
}

# The optics arrays, by name. No separator and no colon, so none of them
# parses as a fault name or as an address.
_TUNES = "tunes"
_BETA_AT_MONITORS = "beta_at_monitors"
_ORBIT_AT_MONITORS = "orbit_at_monitors"
OPTICS_NAMES: frozenset[str] = frozenset({_TUNES, _BETA_AT_MONITORS, _ORBIT_AT_MONITORS})

FaultVariable = PyATWritableScalarVariable | PyATWritableEnumVariable

# A solve (the simulator's last solution object), and the optics computed for
# it.
_OpticsMemo = tuple[Any, dict[str, np.ndarray]]

# An alias, never a subclass. The backend raises ``UnknownElementError``
# itself, from every element lookup it performs -- when it adopts the
# variables, when a misalignment names an element, and on the write path --
# so a distinct class here could only ever cover failures raised on this side
# of the boundary. Aliasing is what keeps an existing ``except
# UnknownDeviceError`` catching every lookup failure the model can produce,
# wherever in the stack it was raised.
UnknownDeviceError = UnknownElementError


def _fault_devices(
    document: BindingsDocument, catalog: Mapping[str, ScalarVariable]
) -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    """The devices the served tree carries a fault for, keyed by element.

    A binding is counted when this deployment serves its address, so a fault
    always stands behind something a client can observe -- and the model-only
    roster stays the faults plus nothing else.

    Args:
        document: the served bindings document, already parsed.
        catalog: the built variable catalog, keyed by address.

    Returns:
        ``(monitors, magnets)``: the monitors as element -> axis -> address,
        one entry per plane the document publishes at that element, and the
        magnets as element -> every address driving it, in document order.
        Both in document order, so a roster reads the same on every boot.
    """
    monitors: dict[str, dict[str, str]] = {}
    magnets: dict[str, list[str]] = {}
    for binding in document.bindings:
        element = binding.element
        if element is None or binding.setpoint_address not in catalog:
            continue
        if binding.kind == MONITOR_KIND:
            axes = monitors.setdefault(element, {})
            axes.setdefault(str(binding.attribute), binding.setpoint_address)
        elif binding.kind in MAGNET_KINDS:
            magnets.setdefault(element, []).append(binding.setpoint_address)
    return monitors, magnets


def _check_fault_seeds(
    bpm_errors: Mapping[str, Mapping[str, float]],
    corrector_gains: Mapping[str, Mapping[str, float]],
    monitors: Mapping[str, Any],
    magnets: Mapping[str, Any],
) -> None:
    """Refuse a seed naming a device or a field the served tree has no fault for.

    Raises:
        UnknownDeviceError: a reading-error seed names an element no monitor
            sits at, or a calibration seed names an element no magnet is
            driven at. One error names every offender, sorted.
        ValueError: a seed names a field no fault variable carries.
    """
    unknown: list[str] = []
    if absent := sorted(set(bpm_errors) - set(monitors)):
        unknown.append(f"reading errors name {absent}, which the tree publishes no monitor at")
    if absent := sorted(set(corrector_gains) - set(magnets)):
        unknown.append(f"calibrations name {absent}, which the tree drives no magnet at")
    if unknown:
        raise UnknownDeviceError("; ".join(unknown))

    bad_fields = sorted(
        {
            f"{device}{FAULT_SEPARATOR}{field}"
            for device, fields in bpm_errors.items()
            for field in fields
            if field not in BPM_ERROR_FIELDS
        }
        | {
            f"{device}{FAULT_SEPARATOR}{field}"
            for device, fields in corrector_gains.items()
            for field in fields
            if field not in CORRECTOR_GAIN_FIELDS
        }
    )
    if bad_fields:
        raise ValueError(
            f"fault seeds name unknown fields {bad_fields}: a monitor takes "
            f"{sorted(BPM_ERROR_FIELDS)}, a magnet takes {sorted(CORRECTOR_GAIN_FIELDS)}"
        )


def _unit_of(catalog: Mapping[str, ScalarVariable], address: str | None) -> str | None:
    """The unit the served catalog states for ``address``, if it states one."""
    if address is None:
        return None
    return getattr(catalog[address], "unit", None)


def _monitor_bound(field: str) -> tuple[float, float] | None:
    """The range one reading-error field is held inside, or ``None`` for none.

    Two kinds of bound, from one table: a window a device's own property has to
    fall in, and the half-open floor under a noise width, which is a standard
    deviation and describes no distribution below zero. A magnitude an operator
    asked for is held by neither.
    """
    if bound := BPM_ERROR_FIELD_BOUNDS.get(field):
        return bound
    return BPM_NOISE_BOUNDS if field in BPM_NOISE_FIELDS else None


def _monitor_fault_variables(
    element: str,
    axes: Mapping[str, str],
    catalog: Mapping[str, ScalarVariable],
    seeds: Mapping[str, float],
) -> list[FaultVariable]:
    """Declare one monitor's nine reading-error faults, each at its seed or identity.

    A displacement and a noise amplitude are in the unit that monitor's own
    reading is published in, taken from the variable they perturb; a roll is
    radians and the rest are dimensionless. A plane the document does not
    publish has no unit to state, and its fields carry none.
    """
    seeded = {**BPM_ERROR_IDENTITY, **seeds}
    variables: list[FaultVariable] = []
    for field in BPM_ERROR_FIELDS:
        binding: dict[str, Any] = {
            "name": f"{element}{FAULT_SEPARATOR}{field}",
            "element_name": element,
            "attribute": _MONITOR_ATTRIBUTE_PREFIX + field,
            "default_value": seeded[field],
            "default_validation_config": "error",
        }
        if field in BPM_POLARITY_FIELDS:
            variables.append(
                PyATWritableEnumVariable(**binding, options=list(BPM_POLARITY_OPTIONS))
            )
            continue
        unit = _ROLL_UNIT if field == "roll" else None
        if (axis := _AXIS_OF_FIELD.get(field)) is not None:
            unit = _unit_of(catalog, axes.get(axis))
        variables.append(
            PyATWritableScalarVariable(**binding, value_range=_monitor_bound(field), unit=unit)
        )
    return variables


def _commanded_unit(
    element: str, addresses: list[str], catalog: Mapping[str, ScalarVariable]
) -> str | None:
    """The unit every setpoint driving ``element`` is commanded in.

    One element carries one calibration, and its offset shifts whatever
    arrives on any address bound to it. A factor is dimensionless and scales
    them all; an offset is a magnitude and can only be one of them, so two
    setpoints commanded in different units leave it undefined.

    Raises:
        ValueError: the addresses driving ``element`` disagree on their unit.
    """
    units = {_unit_of(catalog, address) for address in addresses}
    if len(units) > 1:
        raise ValueError(
            f"the setpoints {sorted(addresses)} all drive element {element!r} but are "
            f"commanded in {sorted(unit or '<none>' for unit in units)}; one element carries "
            f"one calibration, and an offset that shifts them all can only be in one unit"
        )
    return units.pop()


def _magnet_fault_variables(
    element: str,
    addresses: list[str],
    catalog: Mapping[str, ScalarVariable],
    seeds: Mapping[str, float],
) -> list[FaultVariable]:
    """Declare one magnet's calibration faults, each at its seed or identity.

    The offset shifts the value commanded, so it is in the unit that magnet is
    commanded in; the factor scales it and is dimensionless.

    Raises:
        ValueError: the setpoints driving ``element`` disagree on their unit.
    """
    seeded = {
        **MAGNET_CAL_IDENTITY,
        **{CORRECTOR_GAIN_FIELDS[field]: value for field, value in seeds.items()},
    }
    unit = _commanded_unit(element, addresses, catalog)
    return [
        PyATWritableScalarVariable(
            name=f"{element}{FAULT_SEPARATOR}{field}",
            bindings=[
                ElementBinding(element_name=element, attribute=_MAGNET_ATTRIBUTE_PREFIX + field)
            ],
            default_value=seeded[field],
            value_range=MAGNET_CAL_BOUNDS.get(field),
            unit=unit if field == "cal_offset" else None,
            default_validation_config="error",
        )
        for field in MAGNET_CAL_FIELDS
    ]


def _fault_variables(
    monitors: Mapping[str, Mapping[str, str]],
    magnets: Mapping[str, list[str]],
    catalog: Mapping[str, ScalarVariable],
    bpm_errors: Mapping[str, Mapping[str, float]],
    corrector_gains: Mapping[str, Mapping[str, float]],
) -> list[FaultVariable]:
    """Declare every fault writable the served roster carries.

    Expects seeds :func:`_check_fault_seeds` has passed. Every variable
    refuses an out-of-bounds write; a polarity is an enum on its two values.

    Raises:
        pydantic.ValidationError: a seed lies outside its field's bounds, or
            a polarity seed is neither of its two values.
    """
    variables: list[FaultVariable] = []
    for element, axes in monitors.items():
        variables.extend(
            _monitor_fault_variables(element, axes, catalog, bpm_errors.get(element, {}))
        )
    for element, addresses in magnets.items():
        variables.extend(
            _magnet_fault_variables(element, addresses, catalog, corrector_gains.get(element, {}))
        )
    return variables


def _refuse_name_collisions(channels: list[dict], declared: list[Variable]) -> None:
    """Refuse a declared name the served namespace already claims.

    The served / model-only partition is drawn by name alone, against the
    whole channel set rather than against the catalog -- so a fault or an
    optics array spelled like any served address would be served instead of
    held back, and a client would write physics state through a channel.

    Raises:
        ValueError: a declared name is already a served address.
    """
    served = {channel["address"] for channel in channels}
    if clashes := sorted({variable.name for variable in declared} & served):
        raise ValueError(
            f"the model declares {clashes}, which the served namespace already carries as "
            f"channel addresses; a model-only variable is told from a served one by its "
            f"name, so the two rosters cannot share one"
        )


def _optics_variables(monitor_count: int) -> list[PyATReadOnlyNDVariable]:
    """Declare the optics arrays. Per-monitor arrays hold one ``(x, y)`` row each."""
    per_monitor = (monitor_count, 2)
    return [
        PyATReadOnlyNDVariable(name=_TUNES, shape=(2,)),
        PyATReadOnlyNDVariable(name=_BETA_AT_MONITORS, shape=per_monitor, unit="m"),
        PyATReadOnlyNDVariable(name=_ORBIT_AT_MONITORS, shape=per_monitor, unit="m"),
    ]


def _seed_fault_attributes(ring: at.Lattice, variables: list[FaultVariable]) -> None:
    """Write each fault variable's default onto its element's attribute.

    Every element is resolved before any is written, so a fault device the
    ring lacks leaves the ring untouched.

    Raises:
        UnknownDeviceError: a fault device has no element in ``ring``.
    """
    elements = {element.FamName: element for element in ring}
    bound = [(variable.bindings[0], variable.default_value) for variable in variables]
    if missing := sorted({binding.element_name for binding, _ in bound} - set(elements)):
        raise UnknownDeviceError(f"the ring has no element for fault devices {missing}")
    for binding, default in bound:
        setattr(elements[binding.element_name], binding.attribute, float(default))


class PyATRingModel(LUMEPyATModel):
    """A LUME model over the single persistent ``at.Lattice`` a tree serves.

    Hardware setpoints in, monitor readings out, both in the units the
    facility's own calibrations state -- the conversions live on the variables
    (:mod:`~osprey.services.virtual_accelerator.model.variables`), which read
    them from the served bindings document. Beside them stand the settable
    fault state of every monitor and magnet the tree serves, and three
    read-only optics arrays computed on read. One instance owns one lattice for
    its whole lifetime -- every write mutates that same lattice in place -- so
    sequential writes compose exactly like their physical counterparts would,
    and a fault seeded at construction survives every later write and every
    :meth:`reset`.
    """

    def __init__(
        self,
        data_dir: Path,
        channels: list[dict],
        *,
        element_misalignments: dict[str, dict[str, float]] | None = None,
        bpm_errors: dict[str, dict[str, float]] | None = None,
        corrector_gains: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Build the ring from a served tree and adopt the variables it binds.

        Args:
            data_dir: The facility ``data/`` directory to serve: the lattice
                and the bindings under its ``simulation/``, the scenario seed
                and the write bands where
                :class:`~osprey.services.virtual_accelerator.manifest.paths.ManifestPaths`
                resolves them. Required, and never defaulted.
            channels: The served manifest's channel list, exactly as
                :func:`~osprey.services.virtual_accelerator.manifest.loaders.load_manifest_file`
                returns it -- the namespace the deployment resolved, which the
                IOC beside this model is serving on.
            element_misalignments: Element ``FamName`` -> kwargs for
                :func:`~lume_pyat.utils.apply_misalignment` (``dx``/``dy``/
                ``roll``, all optional). Applied once, here, after the ring is
                built and before the nominal orbit is solved -- an element
                absent from this dict keeps AT's default (unmisaligned)
                T1/T2/R1/R2. Every name is validated against the ring before
                any element is mutated.
            bpm_errors: Monitor element name, as the deck spells it and as the
                bindings document carries it -> a partial map of reading-error
                fields (the names of
                :data:`~osprey.services.virtual_accelerator.model.fault_bounds.BPM_ERROR_FIELDS`).
                Each seeds the ``<element>.<field>`` variable; every field not
                named, on every monitor, is identity.
            corrector_gains: Magnet element name, spelled the same way -> a
                partial map of ``factor`` and ``offset``, seeding
                ``<element>.cal_factor`` and ``<element>.cal_offset``; every
                field not named, on every magnet, is identity.

        Raises:
            FileNotFoundError: The tree carries no lattice, no bindings, no
                ``machine.json`` or no ``channel_limits.json``; the message
                names the file that is missing.
            BindingsError: The served bindings document is refused by its own
                schema, the lattice is not the file those bindings were
                derived against, or a bound element name is not carried by
                exactly one element.
            ValueError: A served ``machine.json`` nominal falls outside its
                ``channel_limits.json`` band -- the boot refusal that makes
                the model's declared state the facility's own -- or a fault
                seed names a field no fault variable carries, or a fault name
                collides with a served address.
            UnknownDeviceError: A misalignment names an element the ring does
                not have, or a variable binds one, or a fault seed names an
                element the tree publishes no monitor at (for ``bpm_errors``)
                or drives no magnet at (for ``corrector_gains``). A fault seed
                is refused before the ring is touched, naming every offender.
            pydantic.ValidationError: A fault seed lies outside its field's
                bounds, or a polarity seed is neither of its two values.
            OrbitSolveError: The seeded misalignments leave the ring without a
                stable closed orbit -- the message names the seeded elements
                and their magnitudes so an otherwise opaque boot failure is
                diagnosable. Deliberately *not* ``SystemExit``: whether an
                unusable model should end the process is the caller's call.
        """
        bpm_errors = bpm_errors or {}
        corrector_gains = corrector_gains or {}

        paths = ManifestPaths(data_root=data_dir)
        # Read for the bindings alone. ``build_ring`` reads the same file and
        # owns every check that the two files still describe one accelerator,
        # so what comes back here needs no validation of its own -- and a
        # document this read accepts is the document that ring was checked
        # against.
        document = load_bindings(paths.va_bindings)
        ring = build_ring(paths)
        catalog = build_variable_catalog(paths, channels, build_action_variables(document))
        # Before the model takes the variables over, and before the faults are
        # declared: the knob adopts the rigidity-scaled setpoints of the
        # catalog, and a fault scales nothing when the energy moves.
        couple_energy_knob(catalog)

        monitors, magnets = _fault_devices(document, catalog)
        _check_fault_seeds(bpm_errors, corrector_gains, monitors, magnets)
        faults = _fault_variables(monitors, magnets, catalog, bpm_errors, corrector_gains)
        optics = _optics_variables(len(monitors))
        _refuse_name_collisions(channels, [*faults, *optics])
        _seed_fault_attributes(ring, faults)

        try:
            super().__init__(
                simulator=PyATSimulator(ring, element_misalignments=element_misalignments),
                action_variables=[*catalog.values(), *faults, *optics],
            )
        except OrbitSolveError as exc:
            raise OrbitSolveError(
                f"seeded misalignments {element_misalignments!r} left the lattice "
                f"{paths.lattice_json} without a stable closed orbit at boot ({exc}); "
                "reduce the misalignment magnitude or remove the fault"
            ) from exc

        #: The variables this model declares that no channel addresses: the
        #: faults and the optics. They are named for elements rather than for
        #: channels, and a collision with a served address is refused above,
        #: so the rest of the roster is exactly the addresses the tree binds.
        self.derived_names: frozenset[str] = frozenset(
            variable.name for variable in (*faults, *optics)
        )

        # Sorted by lattice index rather than taken in document order, so row
        # i of every per-monitor optics array is the i-th monitor around the
        # ring.
        self._monitor_order: list[str] = sorted(monitors, key=self.element_index)
        self._monitor_refpts = np.array(
            [self.element_index(element) for element in self._monitor_order]
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
        _, ringdata, elemdata = at.get_optics(self.lattice, refpts=self._monitor_refpts)
        optics = {
            # The two transverse tunes. A ring solved with longitudinal
            # motion on reports a third, and the synchrotron tune is not a
            # betatron one -- the array says which two it holds.
            _TUNES: np.array(ringdata.tune[:2], dtype=np.float64),
            _BETA_AT_MONITORS: np.array(elemdata.beta, dtype=np.float64),
            _ORBIT_AT_MONITORS: np.array(
                [solution[element] for element in self._monitor_order], dtype=np.float64
            ),
        }
        self._optics_memo = (solution, optics)
        return optics


__all__ = [
    "MAGNET_KINDS",
    "MONITOR_KIND",
    "OPTICS_NAMES",
    "PyATRingModel",
    "UnknownDeviceError",
    "OrbitSolveError",
]
