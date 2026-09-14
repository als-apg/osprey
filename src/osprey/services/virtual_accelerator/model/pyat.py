"""The ALS-U AR ring as a ``lume-pyat`` model.

:class:`PyATRingModel` is the facility adapter, and only that. Everything
generic about serving a pyAT ring through the LUME contract -- owning one
persistent lattice, atomic multi-variable writes, one solve per batch,
rollback on a lost closed orbit, retained inputs and cached outputs --
belongs to :class:`~lume_pyat.model.LUMEPyATModel` and is inherited rather
than restated here. What is left is the five facility-specific facts that
class cannot know:

- which lattice to drive (:func:`~osprey.services.virtual_accelerator.lattice.build_ring`),
- how a commanded current becomes a strength
  (:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`),
- which variables exist and what each is bound to
  (:func:`~osprey.services.virtual_accelerator.model.bindings.build_action_variables`),
- which faults every BPM and magnet can carry, and what each was seeded with,
- and how a boot failure should read.

The ring, the strength map and the variables are built as one set, in that
order: the map bakes its strength baseline from the very lattice the
simulator goes on to mutate, so a variable's conversion is always relative
to the ring it writes into. Building the map from a second ``build_ring()``
would give a baseline that merely *happens* to match.

**Faults are variables, held as state on the element.** Every BPM carries
nine reading-error fields and every magnet, correctors included, a
calibration factor and offset: one writable each, 1,344 on the bundled demo
ring. Their names use dot grammar (``BPM01.offset_x``, ``QF07.cal_factor``),
so no fault name is a channel address. Each is bound to an element
attribute (``bpm_offset_x``, ``mag_cal_factor``) that no pyAT pass method
reads. Seeding or writing a fault therefore changes what the element
carries and never the orbit; applying it to a reading or to a commanded
current is the serving layer's work. A seed is its variable's
``default_value`` and is written onto the element from that same value, so
the two cannot disagree, and :meth:`reset` returns a fault to its seed
rather than to identity. Seeds are checked in full (device, then field,
then range) before the ring is touched.

**Optics are read-only arrays, computed on read.** Three quantities of the
solved ring sit beside the per-BPM readings: ``tunes`` (the two fractional
betatron tunes), ``beta_at_bpms`` and ``orbit_at_bpms`` (one ``(x, y)`` row
per BPM, in ring order). The orbit is the true, un-faulted position, the
same values the BPM output variables carry. The base class re-reads every
read-only output after every solve, so a setpoint write would pay for a
linear-optics pass nobody asked for. These three are left out of that
re-read and computed together the first time one is read after a solve,
then served from memory until the next solve.

Declared defaults are recorded as the retained input values at construction
but are not written to the lattice, so the ring boots in exactly the state
``build_ring()`` produced -- the ``machine.json`` nominals the catalog
carries as ``default_value`` *are* that state. :meth:`reset` writes them
back through the transformer, which is what makes a reset undo writes
without disturbing construction-time faults.

This module imports nothing from ``ioc`` and never touches EPICS, and -- like
the base class -- never raises ``SystemExit``: killing the host process is
the serving layer's decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import at
import numpy as np
from lume.actions import WritableActionMixin
from lume_pyat.actions import PyATWritableScalarVariable
from lume_pyat.exceptions import OrbitSolveError, UnknownElementError
from lume_pyat.model import LUMEPyATModel
from lume_pyat.simulator import PyATSimulator

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.strengths import StrengthMap
from osprey.services.virtual_accelerator.model.bindings import build_action_variables
from osprey.services.virtual_accelerator.model.fault_bounds import (
    BPM_ERROR_FIELD_BOUNDS,
    BPM_POLARITY_FIELDS,
    MAGNET_CAL_BOUNDS,
)
from osprey.services.virtual_accelerator.model.variables import (
    PyATReadOnlyNDVariable,
    PyATWritableEnumVariable,
)
from osprey.simulation.facility_spec import ALS_U_AR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lume.variables import Variable

# An alias, never a subclass. ``CurrentSetpointVariable`` raises
# ``UnknownElementError`` from inside the write path, and importing this
# module from there to raise a distinct class would close an import cycle
# (pyat -> bindings -> variables -> pyat). Aliasing is what keeps an existing
# ``except UnknownDeviceError`` catching every lookup failure the model can
# produce, wherever in the stack it was raised.
UnknownDeviceError = UnknownElementError

FaultVariable = PyATWritableScalarVariable | PyATWritableEnumVariable

# What each fault field reads on a device nobody seeded: a BPM that reports
# the true orbit position, a magnet that delivers exactly the current
# commanded.
_BPM_ERROR_IDENTITY: dict[str, float] = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
}
_MAGNET_CAL_IDENTITY: dict[str, float] = {"cal_factor": 1.0, "cal_offset": 0.0}

# ``corrector_gains`` field -> the magnet calibration field it seeds.
_CORRECTOR_GAIN_FIELDS: dict[str, str] = {"factor": "cal_factor", "offset": "cal_offset"}

# Element-attribute prefix per fault kind. No pyAT pass method reads an
# attribute under either, which is what keeps a fault off the orbit.
_BPM_ATTRIBUTE_PREFIX = "bpm_"
_MAGNET_ATTRIBUTE_PREFIX = "mag_"

# Unit of each dimensioned fault field. Gains, factors and polarities are
# dimensionless and carry none.
_FAULT_UNITS: dict[str, str] = {
    "offset_x": "m",
    "offset_y": "m",
    "roll": "rad",
    "noise_x": "m",
    "noise_y": "m",
    "cal_offset": "A",
}


def _device_names(*kinds: str) -> tuple[str, ...]:
    """Every device of the facility-spec families of ``kinds``, in spec order."""
    return tuple(
        ALS_U_AR.device_name("", family.name, ident)
        for family in ALS_U_AR.families
        if family.kind in kinds
        for ident in range(1, family.count + 1)
    )


# The fault-carrying devices. Correctors count as magnets: a calibration
# error scales a commanded current whatever the family.
_BPM_DEVICES = _device_names("monitor")
_MAGNET_DEVICES = _device_names("magnet", "corrector")

# The optics arrays, by name. No colon, so none parses as a channel address.
_TUNES = "tunes"
_BETA_AT_BPMS = "beta_at_bpms"
_ORBIT_AT_BPMS = "orbit_at_bpms"
_OPTICS_NAMES = frozenset({_TUNES, _BETA_AT_BPMS, _ORBIT_AT_BPMS})

# A solve (the simulator's last solution object), and the optics computed
# for it.
_OpticsMemo = tuple[dict[str, tuple[float, float]], dict[str, np.ndarray]]


def _optics_variables() -> list[PyATReadOnlyNDVariable]:
    """Declare the optics arrays. Per-BPM arrays hold one ``(x, y)`` row per BPM."""
    per_bpm = (len(_BPM_DEVICES), 2)
    return [
        PyATReadOnlyNDVariable(name=_TUNES, shape=(2,)),
        PyATReadOnlyNDVariable(name=_BETA_AT_BPMS, shape=per_bpm, unit="m"),
        PyATReadOnlyNDVariable(name=_ORBIT_AT_BPMS, shape=per_bpm, unit="m"),
    ]


def _check_fault_seeds(
    bpm_errors: dict[str, dict[str, float]],
    corrector_gains: dict[str, dict[str, float]],
) -> None:
    """Refuse a seed that names a device or field the model has no fault for.

    Raises:
        UnknownDeviceError: a ``bpm_errors`` key is not a BPM, or a
            ``corrector_gains`` key is not a magnet. One error names every
            offender, sorted.
        ValueError: a seed names a field no fault variable carries.
    """
    unknown: list[str] = []
    if bad_bpms := sorted(set(bpm_errors) - set(_BPM_DEVICES)):
        unknown.append(f"bpm_errors names {bad_bpms}, which the ring has no BPM for")
    if bad_magnets := sorted(set(corrector_gains) - set(_MAGNET_DEVICES)):
        unknown.append(f"corrector_gains names {bad_magnets}, which the ring has no magnet for")
    if unknown:
        raise UnknownDeviceError("; ".join(unknown))

    bad_fields = sorted(
        {
            f"{device}.{field}"
            for device, fields in bpm_errors.items()
            for field in fields
            if field not in BPM_ERROR_FIELD_BOUNDS
        }
        | {
            f"{device}.{field}"
            for device, fields in corrector_gains.items()
            for field in fields
            if field not in _CORRECTOR_GAIN_FIELDS
        }
    )
    if bad_fields:
        raise ValueError(
            f"fault seeds name unknown fields {bad_fields}: a BPM takes "
            f"{sorted(BPM_ERROR_FIELD_BOUNDS)}, a magnet takes {sorted(_CORRECTOR_GAIN_FIELDS)}"
        )


def _fault_variables(
    bpm_errors: dict[str, dict[str, float]],
    corrector_gains: dict[str, dict[str, float]],
) -> list[FaultVariable]:
    """Declare every fault writable, each defaulting to its seed or identity.

    Expects seeds :func:`_check_fault_seeds` has passed. Every variable
    refuses an out-of-bounds write; a polarity is an enum on its two bounds.

    Raises:
        pydantic.ValidationError: a seed lies outside its field's bounds, or
            a polarity seed is not one of them.
    """
    variables: list[FaultVariable] = []
    for bpm in _BPM_DEVICES:
        seeds = {**_BPM_ERROR_IDENTITY, **bpm_errors.get(bpm, {})}
        for field, bounds in BPM_ERROR_FIELD_BOUNDS.items():
            binding = {
                "name": f"{bpm}.{field}",
                "element_name": bpm,
                "attribute": _BPM_ATTRIBUTE_PREFIX + field,
                "default_value": seeds[field],
                "default_validation_config": "error",
            }
            if field in BPM_POLARITY_FIELDS:
                variables.append(PyATWritableEnumVariable(**binding, options=list(bounds)))
            else:
                variables.append(
                    PyATWritableScalarVariable(
                        **binding, value_range=bounds, unit=_FAULT_UNITS.get(field)
                    )
                )
    for magnet in _MAGNET_DEVICES:
        seeds = {
            **_MAGNET_CAL_IDENTITY,
            **{
                _CORRECTOR_GAIN_FIELDS[field]: value
                for field, value in corrector_gains.get(magnet, {}).items()
            },
        }
        for field, bounds in MAGNET_CAL_BOUNDS.items():
            variables.append(
                PyATWritableScalarVariable(
                    name=f"{magnet}.{field}",
                    element_name=magnet,
                    attribute=_MAGNET_ATTRIBUTE_PREFIX + field,
                    default_value=seeds[field],
                    value_range=bounds,
                    unit=_FAULT_UNITS.get(field),
                    default_validation_config="error",
                )
            )
    return variables


def _seed_fault_attributes(ring: at.Lattice, variables: list[FaultVariable]) -> None:
    """Write each fault variable's default onto its element's attribute.

    Every element is resolved before any is written, so a fault device the
    ring lacks leaves the ring untouched.

    Raises:
        UnknownDeviceError: a fault device has no element in ``ring``.
    """
    elements = {element.FamName: element for element in ring}
    if missing := sorted({variable.element_name for variable in variables} - set(elements)):
        raise UnknownDeviceError(f"the ring has no element for fault devices {missing}")
    for variable in variables:
        setattr(elements[variable.element_name], variable.attribute, float(variable.default_value))


class PyATRingModel(LUMEPyATModel):
    """LUME model over a single persistent ALS-U AR ``at.Lattice``.

    Magnet and corrector setpoints in (Amps, absolute and post-calibration),
    BPM positions out (meters), and beside them the settable fault state of
    every BPM and magnet and three read-only optics arrays computed on
    read. One instance owns one lattice for its whole
    lifetime -- every write mutates that same lattice in place -- so
    sequential writes compose exactly like their physical counterparts
    would, and a fault seeded at construction survives every later write and
    every :meth:`reset`.
    """

    def __init__(
        self,
        *,
        element_misalignments: dict[str, dict[str, float]] | None = None,
        bpm_errors: dict[str, dict[str, float]] | None = None,
        corrector_gains: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Build the ring, seed optional faults, and solve the nominal orbit.

        Args:
            element_misalignments: fam_name (e.g. ``"QF07"``, ``"DIPOLE03"``)
                -> kwargs for
                :func:`~lume_pyat.utils.apply_misalignment` (``dx``/``dy``/
                ``roll``, all optional). Applied once, here, after the ring is
                built and before the nominal orbit is solved -- an element
                absent from this dict keeps AT's default (unmisaligned)
                T1/T2/R1/R2. Every fam_name is validated against the ring
                before any element is mutated.
            bpm_errors: BPM fam_name (e.g. ``"BPM01"``) -> a partial map of
                reading-error fields (``offset_x``, ``gain_y``,
                ``polarity_x``, ...; the keys of
                :data:`~osprey.services.virtual_accelerator.model.fault_bounds.BPM_ERROR_FIELD_BOUNDS`).
                Each seeds the ``BPMnn.<field>`` variable; every field not
                named, on every BPM, is identity.
            corrector_gains: magnet fam_name (any family, e.g. ``"HCM01"``,
                ``"QF07"``) -> a partial map of ``factor`` and ``offset``,
                seeding ``<fam>.cal_factor`` and ``<fam>.cal_offset``; every
                field not named, on every magnet, is identity.

        Raises:
            UnknownDeviceError: a misalignment names an element the ring does
                not have, a fault seed names a device that is not a BPM (for
                ``bpm_errors``) or a magnet (for ``corrector_gains``), or a
                variable binds an element the ring lacks. A fault seed is
                refused before the ring is touched, naming every offender.
            ValueError: a fault seed names a field no fault variable carries.
            pydantic.ValidationError: a fault seed lies outside its field's
                bounds, or a polarity seed is neither ``-1`` nor ``1``.
            OrbitSolveError: the seeded misalignments leave the ring without a
                stable closed orbit -- the message names the seeded elements
                and their magnitudes so an otherwise opaque boot failure is
                diagnosable. Deliberately *not* ``SystemExit``: whether an
                unusable model should end the process is the caller's call.
        """
        bpm_errors = bpm_errors or {}
        corrector_gains = corrector_gains or {}

        ring = build_ring()
        _check_fault_seeds(bpm_errors, corrector_gains)
        fault_variables = _fault_variables(bpm_errors, corrector_gains)
        _seed_fault_attributes(ring, fault_variables)
        strength_map = StrengthMap(ring)
        variables = build_action_variables(strength_map=strength_map)

        try:
            super().__init__(
                simulator=PyATSimulator(ring, element_misalignments=element_misalignments),
                action_variables=[*variables.values(), *fault_variables, *_optics_variables()],
            )
        except OrbitSolveError as exc:
            raise OrbitSolveError(
                f"seeded misalignments {element_misalignments!r} left the SR "
                f"lattice without a stable closed orbit at boot ({exc}); reduce the "
                "misalignment magnitude or remove the fault"
            ) from exc

        # Sorted by lattice index rather than trusted to be in spec order, so
        # row i of every per-BPM optics array is the i-th BPM around the ring.
        self._bpm_order: list[str] = sorted(_BPM_DEVICES, key=self.element_index)
        self._bpm_refpts = np.array([self.element_index(bpm) for bpm in self._bpm_order])
        self._optics_memo: _OpticsMemo | None = None

    # -- LUMEPyATModel overrides: the optics arrays ------------------------

    def _validate_binding(self, name: str, variable: Variable) -> None:
        """Check a variable's binding; an optics array binds none to check."""
        if isinstance(variable, PyATReadOnlyNDVariable):
            return
        super()._validate_binding(name, variable)

    def _read_outputs(self) -> dict[str, float]:
        """Every per-BPM reading off the last solve, and never the optics.

        This runs after every solve, so leaving the optics out is what keeps
        a setpoint write from paying for them; :meth:`_get` computes them
        when one is read.
        """
        return {
            name: variable._get(self.simulator)
            for name, variable in self.supported_variables.items()
            if not isinstance(variable, (WritableActionMixin, PyATReadOnlyNDVariable))
        }

    def _get(self, names: list[str]) -> dict[str, Any]:
        """Return one value per name, in the order asked.

        Optics arrays come from :meth:`_optics`, as copies, so a caller that
        edits one cannot alter what the next read returns. Everything else
        is the base class's cached answer.

        Raises:
            UnknownElementError: a name is not a variable of this model.
        """
        if _OPTICS_NAMES.isdisjoint(names):
            return super()._get(names)
        optics = self._optics()
        cached = super()._get([name for name in names if name not in _OPTICS_NAMES])
        return {
            name: optics[name].copy() if name in _OPTICS_NAMES else cached[name] for name in names
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
        _, ringdata, elemdata = at.get_optics(self.lattice, refpts=self._bpm_refpts)
        optics = {
            _TUNES: np.array(ringdata.tune, dtype=np.float64),
            _BETA_AT_BPMS: np.array(elemdata.beta, dtype=np.float64),
            _ORBIT_AT_BPMS: np.array([solution[bpm] for bpm in self._bpm_order], dtype=np.float64),
        }
        self._optics_memo = (solution, optics)
        return optics


__all__ = [
    "PyATRingModel",
    "UnknownDeviceError",
    "OrbitSolveError",
]
