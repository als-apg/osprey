"""What a channel's value does to the lattice, one class per binding kind.

``lume_pyat`` knows only native pyAT quantities: a
:class:`~lume_pyat.actions.PyATWritableScalarVariable` puts the number it is
handed onto the element attributes its bindings name, and the lattice-level
kind writes electron-volts. Everything that makes a number mean *what the
control system set* -- a magnet current, a cavity frequency, a millimetre of
orbit -- lives here, on OSPREY's side of that boundary, and every bit of it is
read from the bindings document rather than from a family name:

* :class:`StrengthVariable` -- a hardware setpoint through the family's
  calibration onto one polynomial coefficient, in full on every slice of a
  split device.
* :class:`KickVariable` -- the same conversion, shared out instead: each of
  the ``n`` pieces of a corrector takes ``1/n`` of the kick, so the first
  slice reads back as the whole of it.
* :class:`RFVariable` -- a frequency, written to every cavity.
* :class:`MonitorVariable` -- a solved orbit reading, in the hardware units
  the facility publishes it in.
* :class:`EnergyVariable` -- the ring energy, driven by the bend's own
  hardware setpoint through its energy table.

Each of those slice shares is one reading of the same arithmetic: a write puts
the physics value times the slice's weight on the element, and a read divides
the first slice's reading by the first weight. A supply feeding several
magnets in series is the third reading of it, each magnet weighing the fixed
factor its own strength stands in to the string's.

Two kinds sit beside them, bound to no channel and declared by the model
rather than by a binding. :class:`PyATWritableEnumVariable` is the enum twin
of :class:`~lume_pyat.actions.PyATWritableScalarVariable`: the same binding
and the same raw, unconverted read and write, whose value is checked against
a list of options instead of a range -- because a sign is not a range, and
nothing between its two values means a smaller sign.
:class:`PyATReadOnlyNDVariable` declares a quantity of the whole solved
lattice, which belongs to no one element and comes out as an array; it
declares the value and leaves computing it to the model that owns it.

**Beam rigidity is the coupling between them.** A calibration states its
physics value at the energy the lattice deck was built for. A family the
control system scales with the rigidity (``energy_scaling: brho``) is worth
:func:`~osprey.services.virtual_accelerator.lattice.calibration.energy_factor`
of that at any other energy, so two writes have to agree: a setpoint write
applies the factor for the energy the ring is at *now*, and an energy write
rescales every rigidity-scaled field already on the lattice. The ring then
ends in the same state whichever order the two arrive in -- which is what the
serving path needs, because a client writes the two independently and nothing
orders them.

**The lattice is the only record of that state.** No variable here retains
the hardware value it last wrote, and the energy rescale is a factor applied
to what an element already holds rather than a fresh conversion of a
remembered setpoint. So a batch the model rolls back leaves no stale copy
behind to be rescaled later, and repeated energy writes stay exact for the
setpoint that is actually on the machine.

**Back to hardware units is never an inversion.** The control system samples
hardware -> physics and physics -> hardware along two independent paths, so a
readback is read off the exported ``monitor_inverse`` and a calibration is
never inverted to stand in for it (see
:mod:`~osprey.services.virtual_accelerator.lattice.calibration`). Where an
export carries no inverse -- an ``identity`` readback, and the energy knob,
whose table the exporter records in one direction only -- there is no path
back, and asking for one raises rather than fabricating a curve.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import EnumVariable, NDVariable
from lume_pyat.actions import (
    ElementBinding,
    PyATLatticeScalarVariable,
    PyATReadOnlyScalarVariable,
    PyATWritableScalarVariable,
)
from lume_pyat.simulator import PyATSimulator
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from osprey.services.virtual_accelerator.bindings import Calibration, Table
from osprey.services.virtual_accelerator.lattice.calibration import (
    energy_factor,
    to_hardware,
    to_physics,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

    import at
    from lume_pyat.actions import SnapshotTarget

#: pyAT holds the ring energy in electron-volts; a calibration's energies, and
#: the rigidity expression, are in GeV.
EV_PER_GEV: float = 1.0e9


class _CalibratedSetpoint(PyATWritableScalarVariable):
    """A hardware setpoint written onto lattice elements through a calibration.

    The shared half of the three writable kinds. What separates them is the
    attribute their bindings name; the conversion, the rigidity factor and the
    way back to hardware units are the same for all three.

    **What a slice weight means.** A write puts the setpoint's physics value
    times the slice's own weight on each bound element, and a read divides the
    first slice's reading by that first weight. A weight is therefore any
    finite non-zero number, which the bindings document is the one guard of:
    one for a piece that carries the whole value, ``1/n`` for a piece that
    takes an equal share of a divisible one, and the fixed factor its own
    strength stands in to the string's for a magnet a supply feeds in series.
    Nothing here narrows that further, because the three meanings are the same
    arithmetic and only the emitter knows which of them it wrote.

    Attributes:
        calibration: Hardware to physics, as the facility sampled it at the
            deck energy.
        monitor_inverse: Physics back to hardware, sampled along the
            facility's own reverse path. ``None`` where the export carries
            none, which is what an ``identity`` readback means: what is read
            back is the value that was written.
        energy_scaling: ``"brho"`` where the physics value moves with the beam
            rigidity, ``"none"`` where the control system's own conversion
            leaves it fixed. The vocabulary is the document's
            (:data:`~osprey.services.virtual_accelerator.bindings.ENERGY_SCALINGS`).
        deck_energy_gev: The beam energy the calibration was sampled at -- the
            document's ``energy_gev``, and the energy the lattice file is built
            for.
    """

    # A misspelled field is a mistake, not an extra to ignore: a dropped
    # `monitor_inverse=` would quietly turn a computed readback into an echo of
    # the setpoint, and a dropped `energy_scaling=` would uncouple a family
    # from the energy knob.
    model_config = ConfigDict(extra="forbid")

    calibration: Calibration
    monitor_inverse: Calibration | None = None
    energy_scaling: Literal["brho", "none"] = "none"
    deck_energy_gev: float

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Convert ``value`` to physics and write it to every bound slice.

        The rigidity factor is taken from the energy the ring is at when the
        write happens, not from the deck energy, so a setpoint written after
        an energy move lands where that move left the family.
        """
        super()._set(simulator, self._physics(simulator, value))

    def _get(self, simulator: PyATSimulator) -> float:
        """Return the hardware value the lattice is presently holding.

        The inverse of :meth:`_set`, to the precision the facility's two
        samplings share. Off the serving path -- ``LUMEPyATModel`` answers a
        read of a writable from the value retained at its last successful
        write -- but it is the honest answer to "what is this device sitting
        at", and it is what a caller driving a simulator directly gets.

        Raises:
            NotImplementedError: the binding carries no ``monitor_inverse``,
                so there is no exported path from physics back to hardware.
        """
        return self._hardware(super()._get(simulator) / self._rigidity_factor(simulator))

    def readback(self, value: float) -> float:
        """What the control system reads back once ``value`` has been written.

        ``monitor_inverse(calibration(value))``: the setpoint's own physics
        value mapped back along the facility's reverse path, which is how a
        coupled readback is computed rather than by inverting anything. Both
        directions scale with the rigidity the same way, so the factors cancel
        and the readback is the same at every ring energy.

        Args:
            value: the hardware value written.

        Returns:
            The hardware value to serve on the readback address. With no
            exported inverse this is ``value`` itself, which is the whole of
            an ``identity`` readback.
        """
        if self.monitor_inverse is None:
            return float(value)
        return self._hardware(float(to_physics(self.calibration, value)))

    def rescale(self, elements: Mapping[str, Any], factor: float) -> None:
        """Multiply every bound field by ``factor``, at unchanged hardware value.

        What an energy write does to a rigidity-scaled family: the setpoint has
        not moved, so its physics value moves by the ratio of the two
        rigidities. The factor applies to whatever the element holds, which is
        what keeps a kick's slices sharing it in the same proportions and what
        makes successive energy writes exact for the setpoint on the machine.

        Args:
            elements: the ring's elements, keyed by ``FamName``.
            factor: ``brho(E_before) / brho(E_after)``.
        """
        for binding in self.bindings:
            element = elements[binding.element_name]
            held = getattr(element, binding.attribute)
            if binding.index is None:
                setattr(element, binding.attribute, held * factor)
            else:
                held[binding.index] = held[binding.index] * factor

    def _physics(self, simulator: PyATSimulator, value: float) -> float:
        """The physics value ``value`` is worth at the ring's present energy."""
        physics = float(to_physics(self.calibration, value))
        return physics * self._rigidity_factor(simulator)

    def _rigidity_factor(self, simulator: PyATSimulator) -> float:
        """What a rigidity-scaled value is worth at the ring's present energy."""
        if self.energy_scaling != "brho":
            return 1.0
        energy_gev = float(simulator.lattice.energy) / EV_PER_GEV
        return float(energy_factor(energy_gev, self.deck_energy_gev))

    def _hardware(self, physics: float) -> float:
        """Map a deck-energy physics value back to hardware units.

        Raises:
            NotImplementedError: no ``monitor_inverse`` was exported for this
                binding.
        """
        if self.monitor_inverse is None:
            raise NotImplementedError(
                f"variable {self.name!r} carries no monitor_inverse, so its physics value "
                "has no exported path back to hardware units; the readback of such a "
                "binding is the value that was written"
            )
        return float(to_hardware(self.monitor_inverse, physics))


class StrengthVariable(_CalibratedSetpoint):
    """One magnet setpoint, onto a polynomial coefficient of every slice.

    The slices of a strength are the pieces a split magnet is modelled as and
    the magnets a supply feeds in series, and each carries the setpoint's
    physics value times its own weight. A split magnet's pieces each carry the
    whole strength, because the control system sets a strength rather than a
    strength to divide up; a series magnet carries the fixed factor its own
    strength stands in to the string's.
    """


class KickVariable(_CalibratedSetpoint):
    """One corrector setpoint, over the slices it is bound to.

    A kick *is* divisible: a corrector modelled as ``n`` pieces bends the beam
    by the sum of what its pieces do, so each piece takes ``1/n`` of the kick
    and reading the first slice back multiplies by ``n`` again, which is the
    value the control system reads. Where one supply bends several correctors
    in series, that share is multiplied by the magnet's own fixed factor.
    """


class RFVariable(_CalibratedSetpoint):
    """The cavity frequency, written to every cavity in the ring.

    One setpoint over every cavity, each of them carrying the whole frequency:
    a ring's cavities run at one frequency, so the slices replicate it exactly
    as a split magnet's pieces replicate a strength.
    """


class MonitorVariable(PyATReadOnlyScalarVariable):
    """One transverse orbit reading, in the hardware units it is published in.

    The solved orbit is in metres -- pyAT's unit, and the physics unit the
    control system records for a position monitor -- and ``monitor_inverse``
    is the facility's sampled physics-to-hardware curve, which is where the
    metre-to-millimetre factor lives (an NSLS-II BPM exports a gain of 1000).
    So nothing here scales the reading: applying the exported inverse is the
    whole conversion, which is also why a facility publishing microns, counts
    or anything else needs no case of its own.

    Attributes:
        monitor_inverse: Physics back to hardware for this monitor. Required,
            not optional as on a setpoint: a monitor has no written value to
            fall back on, so the inverse is the only thing that makes its
            reading publishable at all.
    """

    model_config = ConfigDict(extra="forbid")

    monitor_inverse: Calibration

    def _get(self, simulator: PyATSimulator) -> float:
        """Read this monitor's coordinate off the last solve, in hardware units.

        Raises:
            OrbitSolveError: no solve has succeeded yet.
            UnknownElementError: the bound element is not a monitor of this
                lattice.
        """
        return float(to_hardware(self.monitor_inverse, super()._get(simulator)))


class EnergyVariable(PyATLatticeScalarVariable):
    """The ring energy, driven by the bending magnet's own hardware setpoint.

    ``E(I) = E_deck * table(I) / table(I_nom)``: the energy table is the
    facility's own setpoint-to-energy curve, and the deck energy is what the
    lattice was built at, so the ring sits at exactly its deck energy while
    the bend is at its nominal setpoint -- whatever the table's absolute scale
    says. The ratio is what the table is trusted for; its absolute value is
    not, because the lattice file and the control system need not agree on it.

    A write lands on the ring energy and every cavity's, and then rescales
    every rigidity-scaled binding this knob has adopted through
    :meth:`couple`. That rescale runs in ``_after_write``, inside the model's
    batch, so the whole move -- energy and strengths together -- costs one
    orbit solve and rolls back as one.

    Attributes:
        energy_table: The sampled setpoint-to-energy curve, in GeV.
        nominal: The bend's nominal hardware setpoint, at which the ring is at
            its deck energy.
        deck_energy_gev: The energy the lattice file is built at.
    """

    model_config = ConfigDict(extra="forbid")

    energy_table: Table
    nominal: float
    deck_energy_gev: float

    # The variables this knob rescales, and the ring energy it is about to
    # move away from. Both are infrastructure rather than part of the
    # variable's declared shape: the first is a ring-sized object graph that
    # has no business in a model_dump, the second lives only for the duration
    # of one write.
    _scaled: tuple[_CalibratedSetpoint, ...] = PrivateAttr(default=())
    _energy_before_write: float | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_the_energy_reference(self) -> EnergyVariable:
        """Refuse a deck energy or a nominal that leaves ``E(I)`` undefined."""
        if not math.isfinite(self.deck_energy_gev) or self.deck_energy_gev <= 0.0:
            raise ValueError(
                f"the deck energy must be a positive number of GeV, got {self.deck_energy_gev}"
            )
        if float(to_physics(self.energy_table, self.nominal)) == 0.0:
            raise ValueError(
                f"the energy table maps the nominal setpoint {self.nominal} to zero, "
                "which leaves every energy this knob writes undefined"
            )
        return self

    def couple(self, variables: Iterable[PyATWritableScalarVariable]) -> tuple[str, ...]:
        """Adopt the rigidity-scaled variables this knob rescales, and name them.

        Separate from construction because the energy knob is one binding
        among a facility's thousands, built in the order the manifest lists
        them: nothing can hand a constructor the whole set while it is still
        being built. Call it once, before the model adopts the variables --
        :meth:`snapshot_targets` reports what they touch, and the model reads
        that both when it validates the variable set and when it snapshots a
        batch.

        Args:
            variables: every writable of the catalog. The rigidity-scaled ones
                are selected here, so a caller hands over the whole set and
                does not have to know which families move with the energy.

        Returns:
            The names adopted, in the order they were given.
        """
        self._scaled = tuple(
            variable
            for variable in variables
            if isinstance(variable, _CalibratedSetpoint) and variable.energy_scaling == "brho"
        )
        return tuple(variable.name for variable in self._scaled)

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:
        """The ring energy, every cavity's, and every field the rescale touches.

        The adopted variables' own targets, exactly as they declare them: a
        write of this knob reaches their elements too, and what the model does
        not know about it cannot roll back.
        """
        targets = super().snapshot_targets(simulator)
        for variable in self._scaled:
            targets.extend(variable.snapshot_targets(simulator))
        return targets

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Write the beam energy that ``value`` on the bend implies.

        Raises:
            ValueError: the table maps ``value`` to an energy that is not
                positive and finite, which is not an energy a ring can be run
                at. Nothing has been written.
        """
        energy_gev = self._energy_gev(value)
        if not math.isfinite(energy_gev) or energy_gev <= 0.0:
            raise ValueError(
                f"setpoint {value} maps to a beam energy of {energy_gev} GeV; "
                "a ring's energy is positive and finite"
            )
        # Read before the write, for _after_write: it runs once the new energy
        # is on the ring, by which time the energy it moved from is gone.
        self._energy_before_write = float(simulator.lattice.energy)
        super()._set(simulator, energy_gev * EV_PER_GEV)

    def _get(self, simulator: PyATSimulator) -> float:
        """Always raises: the ring's energy does not say what setpoint made it.

        Recovering the bend's hardware value from the energy would mean
        inverting the energy table, and the export carries that curve in one
        direction only. The model serves a read of this knob from the value
        retained at its last write, which is the answer a client gets.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"variable {self.name!r} drives the ring energy through an energy table that is "
            "exported in one direction only, so the setpoint behind the present energy "
            "cannot be recovered from the lattice"
        )

    def _after_write(self, ring: at.Lattice, value: float) -> None:
        """Rescale every adopted binding to the energy just written.

        Args:
            ring: the live lattice, already carrying the new energy.
            value: the energy just written, in eV.
        """
        before = self._energy_before_write
        self._energy_before_write = None
        # The ring's own number rather than the one just handed to it: pyAT
        # re-derives the energy it is given, and a setpoint written afterwards
        # takes its factor from what the ring holds, so the rescale has to use
        # the same reading for the two paths to agree.
        after = float(ring.energy)
        if before is None or before == after or not self._scaled:
            return
        factor = float(energy_factor(after / EV_PER_GEV, before / EV_PER_GEV))
        # One pass over the ring, keyed by the name the bindings use. Every
        # bound name reaches exactly one element -- the model checks that when
        # it adopts the variables -- and an energy write is rare enough that
        # the pass costs less than holding indices a rebuilt ring would stale.
        elements = {element.FamName: element for element in ring}
        for variable in self._scaled:
            variable.rescale(elements, factor)

    def _energy_gev(self, value: float) -> float:
        """The beam energy ``value`` on the bend implies, in GeV."""
        reference = float(to_physics(self.energy_table, self.nominal))
        return self.deck_energy_gev * float(to_physics(self.energy_table, value)) / reference


class PyATWritableEnumVariable(WritableActionMixin[PyATSimulator], EnumVariable):
    """A settable discrete value bound to one attribute of one lattice element.

    Binds, reads, writes and snapshots exactly as
    :class:`~lume_pyat.actions.PyATWritableScalarVariable` does, and takes the
    same single-element shorthand; only the validation differs, and that is
    inherited from ``EnumVariable``: a value not in ``options`` is refused by
    ``LUMEModel.set`` before any write, and a ``default_value`` not in
    ``options`` fails at definition.

    Options must be numeric. ``LUMEPyATModel`` retains every writable's
    default as ``float(default_value)``, so a non-numeric option could be
    declared but never booted.

    Attributes:
        bindings: The elements this variable drives, in order. At least one;
            the first is the one a read comes from. A weight scales what the
            element receives, exactly as it does for the scalar kind.
    """

    bindings: list[ElementBinding] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _expand_the_single_element_form(cls, data: Any) -> Any:
        """Turn ``element_name``/``attribute``/``index`` into one binding."""
        if not isinstance(data, dict) or "element_name" not in data:
            return data
        if "bindings" in data:
            raise ValueError(
                "give either bindings= or the single-element form "
                "(element_name=, attribute=, index=), not both"
            )
        data = dict(data)
        binding = {
            "element_name": data.pop("element_name"),
            "attribute": data.pop("attribute", None),
            "index": data.pop("index", None),
        }
        data["bindings"] = [ElementBinding.model_validate(binding)]
        return data

    @model_validator(mode="after")
    def _require_a_default_value(self) -> PyATWritableEnumVariable:
        """Reject a writable with no default, at definition time.

        ``reset()`` writes every writable's ``default_value`` back to the
        lattice, so a variable without one would push ``None`` there.
        """
        if self.default_value is None:
            raise ValueError(
                f"writable variable {self.name!r} needs a default_value: "
                "it is what reset() writes back to the lattice"
            )
        return self

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:
        """Every ``(element index, attribute)`` a write of this variable touches.

        Raises:
            UnknownElementError: a binding names an element the lattice does
                not have.
        """
        return [
            (simulator.element_index(binding.element_name), binding.attribute)
            for binding in self.bindings
        ]

    def _get(self, simulator: PyATSimulator) -> float:
        """Read the first binding's element, unconverted.

        Raises:
            UnknownElementError: the lattice has no such element.
        """
        first = self.bindings[0]
        value = getattr(simulator.element(first.element_name), first.attribute)
        if first.index is not None:
            value = value[first.index]
        return float(value) / first.weight

    def _set(self, simulator: PyATSimulator, value: Any) -> None:
        """Write ``value`` to every bound element, unconverted.

        Every element is checked before any is written, so a bad binding
        leaves the lattice untouched.

        Raises:
            UnknownElementError: the lattice has no such element.
            AttributeError: an element has no such attribute. pyAT elements
                accept arbitrary attribute assignment, so the write would
                otherwise land on a dead attribute and read back intact.
                Through ``LUMEPyATModel`` this cannot be reached: the model
                checks the same condition when it adopts the variable.
        """
        elements = [simulator.element(binding.element_name) for binding in self.bindings]
        for binding, element in zip(self.bindings, elements, strict=True):
            if not hasattr(element, binding.attribute):
                raise AttributeError(
                    f"element {binding.element_name!r} has no attribute "
                    f"{binding.attribute!r} to write"
                )
        for binding, element in zip(self.bindings, elements, strict=True):
            scaled = value * binding.weight
            if binding.index is None:
                setattr(element, binding.attribute, scaled)
            else:
                getattr(element, binding.attribute)[binding.index] = scaled


class PyATReadOnlyNDVariable(ReadOnlyActionMixin[PyATSimulator], NDVariable):
    """A read-only array derived from the whole solved lattice.

    Declares a name, a shape and a ``float64`` dtype, which
    ``LUMEModel.get`` checks every returned value against exactly, and
    nothing more. The value is computed by the model that declares the
    variable, never by the variable: what makes such a quantity affordable is
    *when* it is computed (on read, at most once per solve), and that is model
    state rather than variable state.

    Attributes:
        element_name: Always ``None``. No single element carries the value,
            so a model's per-element binding check has nothing to resolve.
    """

    element_name: None = None
    read_only: bool = True

    def _get(self, simulator: PyATSimulator) -> Any:
        """Refuse. The owning model computes the value, not the variable.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"{self.name!r} is derived from the whole solved lattice; "
            "read it through the model that declares it"
        )


__all__ = [
    "EV_PER_GEV",
    "EnergyVariable",
    "KickVariable",
    "MonitorVariable",
    "PyATReadOnlyNDVariable",
    "PyATWritableEnumVariable",
    "RFVariable",
    "StrengthVariable",
]
