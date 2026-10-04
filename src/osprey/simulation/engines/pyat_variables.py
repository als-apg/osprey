"""What a wired channel's value does to the deck, one class per wiring shape.

``lume_pyat`` knows only native pyAT quantities: a
:class:`~lume_pyat.actions.PyATWritableScalarVariable` puts the number it is
handed onto the element attributes its bindings name, and the lattice-level
kind writes electron-volts. Everything that makes a number mean *what the
control system set* -- a magnet current, a cavity frequency, a millimetre of
orbit -- lives here, and every bit of it is read from the wiring record (its
``element`` or ``slices``, its engine block in pyAT's words, its
``calibration``) rather than from a family name:

* :class:`StrengthVariable` -- a hardware setpoint through the record's
  calibration onto one polynomial coefficient, in full on every slice of a
  split device.
* :class:`KickVariable` -- the same conversion onto ``KickAngle``, shared out
  over the slices by their weights.
* :class:`RFVariable` -- a frequency, written to every cavity the record
  names.
* :class:`MonitorVariable` -- a solved orbit reading on the record's
  ``axis``, in the hardware units the facility publishes it in.
* :class:`EnergyVariable` -- the deck energy (``attribute: energy``, no
  element), driven by the bend's own hardware setpoint through its curve.

:func:`variable_from_wiring` picks the class from the engine block.

Each slice share is one reading of the same arithmetic: a write puts the
physics value times the slice's weight on the element, and a read divides the
first slice's reading by the first weight. A supply feeding several magnets in
series is the third reading of it, each magnet weighing the fixed factor its
own strength stands in to the string's.

Two kinds sit beside them, bound to no channel and declared by the model
rather than by a wiring record. :class:`PyATWritableEnumVariable` is the enum
twin of :class:`~lume_pyat.actions.PyATWritableScalarVariable`: the same
binding and the same raw, unconverted read and write, whose value is checked
against a list of options instead of a range -- because a sign is not a range,
and nothing between its two values means a smaller sign.
:class:`PyATReadOnlyNDVariable` declares a quantity of the whole solved
lattice, which belongs to no one element and comes out as an array; it
declares the value and leaves computing it to the model that owns it.

**Beam rigidity is the coupling between them.** A calibration states its
physics value at the energy the deck was built for. A record the control
system scales with the rigidity (``energy_scaling: brho``) is worth
:func:`~osprey.simulation.engines.calibration.energy_factor` of that at any
other energy, so two writes have to agree: a setpoint write applies the factor
for the energy the lattice is at *now*, and an energy write rescales every
rigidity-scaled field already on the lattice. The lattice then ends in the
same state whichever order the two arrive in -- which is what the serving path
needs, because a client writes the two independently and nothing orders them.

**The lattice is the only record of that state.** No variable here retains the
hardware value it last wrote, and the energy rescale is a factor applied to
what an element already holds rather than a fresh conversion of a remembered
setpoint. So a batch the model rolls back leaves no stale copy behind to be
rescaled later, and repeated energy writes stay exact for the setpoint that is
actually on the machine.

**Back to hardware units is one rule.** The record's ``calibration.inverse``
when it states one, else the algebraic inverse of a ``linear`` curve
(:func:`~osprey.simulation.engines.calibration.to_hardware`). A ``table``
without an ``inverse``, or a linear gain of 0, has no way back, and a variable
built on one is refused at construction.
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

from osprey.simulation.engines.calibration import (
    Calibration,
    check_inverse,
    curve_from_record,
    energy_factor,
    field,
    to_hardware,
    to_physics,
)
from osprey.simulation.engines.pyat_faults import magnet_cal, supply_calibration

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable, Mapping

    import at
    from lume.variables import ScalarVariable
    from lume_pyat.actions import SnapshotTarget

#: pyAT holds the deck energy in electron-volts; the rigidity expression is in
#: GeV.
EV_PER_GEV: float = 1.0e9

#: The engine-block attribute of the deck energy: a property of the whole
#: deck, read and written with no element.
ENERGY_ATTRIBUTE = "energy"

#: The engine-block attributes a setpoint class writes other than a
#: strength's polynomial coefficient.
KICK_ATTRIBUTE = "KickAngle"
FREQUENCY_ATTRIBUTE = "Frequency"

EnergyScaling = Literal["brho", "none"]


def _record_id(record: Any) -> str:
    """The name a refusal gives a record: its ``id``, else its ``address``."""
    return str(field(record, "id") or field(record, "address"))


def _curves(record: Any) -> tuple[Calibration | None, Calibration | None]:
    """A record's ``calibration.curve`` and ``calibration.inverse``."""
    calibration = field(record, "calibration")
    return (
        curve_from_record(field(calibration, "curve")),
        curve_from_record(field(calibration, "inverse")),
    )


def _energy_scaling(record: Any) -> EnergyScaling:
    """A record's ``calibration.energy_scaling``; ``none`` when it states none."""
    word = field(field(record, "calibration"), "energy_scaling") or "none"
    if word == "brho":
        return "brho"
    if word == "none":
        return "none"
    raise ValueError(f"wiring {_record_id(record)}: energy_scaling is {word!r}; use brho or none")


def _element_bindings(record: Any) -> list[ElementBinding]:
    """One element binding per slice of a record, on its engine attribute.

    A record naming ``element`` is one slice of weight 1; a slice stating no
    weight weighs 1. The first slice is the one a read comes from.
    """
    block = field(record, "engine")
    attribute = field(block, "attribute")
    if attribute is None:
        raise ValueError(f"wiring {_record_id(record)}: the engine block names no attribute")
    index = field(block, "index")
    slices = field(record, "slices")
    if slices:
        pieces = [(str(field(piece, "element")), field(piece, "weight")) for piece in slices]
    else:
        element = field(record, "element")
        if element is None:
            raise ValueError(f"wiring {_record_id(record)}: names no element and no slices")
        pieces = [(str(element), None)]
    return [
        ElementBinding(
            element_name=element,
            attribute=str(attribute),
            index=None if index is None else int(index),
            weight=1.0 if weight is None else float(weight),
        )
        for element, weight in pieces
    ]


def _scalar_fields(record: Any, overrides: Mapping[str, Any], *, writable: bool) -> dict[str, Any]:
    """The ``ScalarVariable`` fields a record states, under the caller's own.

    ``name`` is the record's address, ``unit`` its computed unit and, for a
    writable, ``default_value`` its computed default; a field the caller
    passes wins.
    """
    stated: dict[str, Any] = {"name": str(field(record, "address"))}
    unit = field(record, "unit")
    if unit is not None:
        stated["unit"] = str(unit)
    default = field(record, "default")
    if writable and default is not None:
        stated["default_value"] = float(default)
    return {**stated, **overrides}


class CalibratedSetpoint(PyATWritableScalarVariable):
    """A hardware setpoint written onto deck elements through a calibration.

    The shared half of the three writable kinds. What separates them is the
    attribute their bindings name; the conversion, the rigidity factor and the
    way back to hardware units are the same for all three.

    **What a slice weight means.** A write puts the setpoint's physics value
    times the slice's own weight on each bound element, and a read divides the
    first slice's reading by that first weight. A weight is therefore any
    finite non-zero number: one for a piece that carries the whole value,
    ``1/n`` for a piece that takes an equal share of a divisible one, and the
    fixed factor its own strength stands in to the string's for a magnet a
    supply feeds in series.

    Attributes:
        calibration: Hardware to physics, as the facility states it at the
            deck energy; ``None`` when the hardware value is the physics
            value.
        inverse: Physics back to hardware, the record's
            ``calibration.inverse``; ``None`` where the record states none,
            and the way back is then the algebraic inverse of a linear
            ``calibration``.
        energy_scaling: ``"brho"`` where the physics value moves with the beam
            rigidity, ``"none"`` where the control system's own conversion
            leaves it fixed.
        deck_energy_gev: The beam energy the calibration is stated at, which
            is the energy the deck is built for.
    """

    # A misspelled field is a mistake, not an extra to ignore: a dropped
    # `inverse=` would quietly serve the algebraic inverse instead of the
    # facility's own reverse path, and a dropped `energy_scaling=` would
    # uncouple a record from the energy knob.
    model_config = ConfigDict(extra="forbid")

    calibration: Calibration | None = None
    inverse: Calibration | None = None
    energy_scaling: EnergyScaling = "none"
    deck_energy_gev: float

    @model_validator(mode="after")
    def _require_a_way_back(self) -> CalibratedSetpoint:
        """Refuse a calibration with no way back to hardware units."""
        check_inverse(self.calibration, self.inverse)
        return self

    @classmethod
    def from_wiring(
        cls, record: Any, *, deck_energy_gev: float, **scalar_fields: Any
    ) -> CalibratedSetpoint:
        """Build the setpoint one wiring record describes.

        Args:
            record: The wiring record, read by key or by attribute: ``address``,
                ``element`` or ``slices`` (each ``element``, ``weight``), the
                engine block's ``attribute`` and ``index``, ``calibration``
                (``curve``, ``inverse``, ``energy_scaling``) and the computed
                ``default`` and ``unit``.
            deck_energy_gev: The energy the deck is built for.
            **scalar_fields: ``ScalarVariable`` fields, each winning over the
                one the record states.

        Returns:
            The variable, of the class this method is called on.

        Raises:
            ValueError: the record names no element, no slices or no engine
                attribute, states an unknown ``energy_scaling``, or its
                calibration has no way back to hardware units.
        """
        curve, inverse = _curves(record)
        return cls(
            bindings=_element_bindings(record),
            calibration=curve,
            inverse=inverse,
            energy_scaling=_energy_scaling(record),
            deck_energy_gev=deck_energy_gev,
            **_scalar_fields(record, scalar_fields, writable=True),
        )

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Convert ``value`` to physics and write it to every bound slice.

        The rigidity factor is taken from the energy the lattice is at when the
        write happens, not from the deck energy, so a setpoint written after
        an energy move lands where that move left the record.
        """
        super()._set(simulator, self._physics(simulator, value))

    def _get(self, simulator: PyATSimulator) -> float:
        """Return the hardware value the lattice is presently holding.

        The inverse of :meth:`_set`, to the precision the calibration's two
        directions share.
        """
        return self._hardware(super()._get(simulator) / self._rigidity_factor(simulator))

    def readback(self, value: float) -> float:
        """What the control system reads back once ``value`` has been written.

        ``inverse(calibration(value))``: the setpoint's own physics value
        mapped back along the facility's reverse path. Both directions scale
        with the rigidity the same way, so the factors cancel and the readback
        is the same at every deck energy.

        Args:
            value: the hardware value written.

        Returns:
            The hardware value to serve on the readback address. With no
            ``inverse`` the way back is the calibration's own algebraic
            inverse, so this is ``value`` itself.
        """
        if self.inverse is None:
            return float(value)
        return self._hardware(to_physics(self.calibration, value))

    def rescale(self, elements: Mapping[str, Any], factor: float) -> None:
        """Multiply every bound field by ``factor``, at unchanged hardware value.

        What an energy write does to a rigidity-scaled record: the setpoint has
        not moved, so its physics value moves by the ratio of the two
        rigidities. The factor applies to whatever the element holds, which is
        what keeps a kick's slices sharing it in the same proportions and what
        makes successive energy writes exact for the setpoint on the machine.

        Args:
            elements: the deck's elements, keyed by ``FamName``.
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
        """The physics value ``value`` is worth at the lattice's present energy.

        This setpoint's own supply calibration, held on its first bound
        element, acts on the commanded value first: a miscalibrated supply
        delivers a different hardware value, which the record's own
        calibration then converts. A setpoint whose element carries none
        delivers what was commanded.
        """
        element = simulator.element(self.bindings[0].element_name)
        delivered = magnet_cal(value, **supply_calibration(element, self.name))
        return to_physics(self.calibration, delivered) * self._rigidity_factor(simulator)

    def _rigidity_factor(self, simulator: PyATSimulator) -> float:
        """What a rigidity-scaled value is worth at the lattice's present energy."""
        if self.energy_scaling != "brho":
            return 1.0
        energy_gev = float(simulator.lattice.energy) / EV_PER_GEV
        return energy_factor(energy_gev, self.deck_energy_gev)

    def _hardware(self, physics: float) -> float:
        """Map a deck-energy physics value back to hardware units."""
        return to_hardware(self.calibration, self.inverse, physics)


class StrengthVariable(CalibratedSetpoint):
    """One magnet setpoint, onto a polynomial coefficient of every slice.

    The slices of a strength are the pieces a split magnet is modelled as and
    the magnets a supply feeds in series, and each carries the setpoint's
    physics value times its own weight. A split magnet's pieces each carry the
    whole strength, because the control system sets a strength rather than a
    strength to divide up; a series magnet carries the fixed factor its own
    strength stands in to the string's.
    """


class KickVariable(CalibratedSetpoint):
    """One corrector setpoint, over the slices it is bound to.

    A kick *is* divisible: a corrector modelled as ``n`` pieces bends the beam
    by the sum of what its pieces do, so each piece takes ``1/n`` of the kick
    and reading the first slice back multiplies by ``n`` again, which is the
    value the control system reads. Where one supply bends several correctors
    in series, that share is multiplied by the magnet's own fixed factor.
    """


class RFVariable(CalibratedSetpoint):
    """The cavity frequency, written to every cavity the record names.

    One setpoint over every cavity, each of them carrying the whole frequency:
    the cavities of one lattice run at one frequency, so the slices replicate
    it exactly as a split magnet's pieces replicate a strength.
    """


class MonitorVariable(PyATReadOnlyScalarVariable):
    """One transverse orbit reading, in the hardware units it is published in.

    The solved orbit is in metres -- pyAT's unit, and the physics unit of a
    position monitor -- and the record's calibration carries whatever factor
    turns that into the published unit. The reading goes back through the
    one rule: ``inverse`` when the record states it, else the algebraic
    inverse of a linear ``calibration``; never an identity on a table.

    Attributes:
        calibration: Hardware to physics; ``None`` when the published unit is
            metres.
        inverse: Physics back to hardware, or ``None`` where the record states
            none.
    """

    model_config = ConfigDict(extra="forbid")

    calibration: Calibration | None = None
    inverse: Calibration | None = None

    @model_validator(mode="after")
    def _require_a_way_back(self) -> MonitorVariable:
        """Refuse a calibration with no way back to hardware units."""
        check_inverse(self.calibration, self.inverse)
        return self

    @classmethod
    def from_wiring(cls, record: Any, **scalar_fields: Any) -> MonitorVariable:
        """Build the reading one wiring record describes.

        Args:
            record: The wiring record: ``address``, ``element`` (or the first
                of its ``slices``), the engine block's ``axis`` and
                ``calibration`` (``curve``, ``inverse``).
            **scalar_fields: ``ScalarVariable`` fields, each winning over the
                one the record states.

        Returns:
            The variable.

        Raises:
            ValueError: the record names no element or an axis other than
                ``x`` or ``y``, or its calibration has no way back to hardware
                units.
        """
        slices = field(record, "slices")
        element = field(slices[0], "element") if slices else field(record, "element")
        if element is None:
            raise ValueError(f"wiring {_record_id(record)}: names no monitor element")
        axis = field(field(record, "engine"), "axis")
        if axis not in ("x", "y"):
            raise ValueError(f"wiring {_record_id(record)}: axis is {axis!r}; use x or y")
        curve, inverse = _curves(record)
        return cls(
            element_name=str(element),
            axis=axis,
            calibration=curve,
            inverse=inverse,
            read_only=True,
            **_scalar_fields(record, scalar_fields, writable=False),
        )

    def _get(self, simulator: PyATSimulator) -> float:
        """Read this monitor's coordinate off the last solve, in hardware units.

        Raises:
            OrbitSolveError: no solve has succeeded yet.
            UnknownElementError: the bound element is not a monitor of this
                lattice.
        """
        return to_hardware(self.calibration, self.inverse, super()._get(simulator))


class EnergyVariable(PyATLatticeScalarVariable):
    """The deck energy, driven by the bending magnet's own hardware setpoint.

    ``E(I) = E_deck * curve(I) / curve(I_nom)``: the curve is the facility's
    own setpoint-to-energy curve, and the deck energy is what the lattice was
    built at, so the lattice sits at exactly its deck energy while the bend is
    at its nominal setpoint -- whatever the curve's absolute scale says. The
    ratio is what the curve is trusted for; its absolute value is not, because
    the deck and the control system need not agree on it.

    A write lands on the lattice energy and every cavity's, and then rescales
    every rigidity-scaled setpoint this knob has adopted through
    :meth:`couple`. That rescale runs in ``_after_write``, inside the model's
    batch, so the whole move -- energy and strengths together -- costs one
    orbit solve and rolls back as one.

    Attributes:
        calibration: The setpoint-to-energy curve.
        nominal: The bend's nominal hardware setpoint, at which the lattice is
            at its deck energy.
        deck_energy_gev: The energy the deck is built at.
    """

    model_config = ConfigDict(extra="forbid")

    calibration: Calibration
    nominal: float
    deck_energy_gev: float

    # The variables this knob rescales, and the lattice energy it is about to
    # move away from. Both are infrastructure rather than part of the
    # variable's declared shape: the first is a lattice-sized object graph that
    # has no business in a model_dump, the second lives only for the duration
    # of one write.
    _scaled: tuple[CalibratedSetpoint, ...] = PrivateAttr(default=())
    _energy_before_write: float | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_the_energy_reference(self) -> EnergyVariable:
        """Refuse a deck energy or a nominal that leaves ``E(I)`` undefined."""
        if not math.isfinite(self.deck_energy_gev) or self.deck_energy_gev <= 0.0:
            raise ValueError(
                f"the deck energy must be a positive number of GeV, got {self.deck_energy_gev}"
            )
        if to_physics(self.calibration, self.nominal) == 0.0:
            raise ValueError(
                f"the energy curve maps the nominal setpoint {self.nominal} to zero, "
                "which leaves every energy this knob writes undefined"
            )
        return self

    @classmethod
    def from_wiring(
        cls, record: Any, *, deck_energy_gev: float, **scalar_fields: Any
    ) -> EnergyVariable:
        """Build the energy knob one wiring record describes.

        Args:
            record: The wiring record: ``address``, the engine block's
                ``attribute: energy`` with no element, ``calibration.curve``
                and the computed ``default``, which is the nominal setpoint.
            deck_energy_gev: The energy the deck is built for.
            **scalar_fields: ``ScalarVariable`` fields, each winning over the
                one the record states.

        Returns:
            The variable.

        Raises:
            ValueError: the record states no curve or no default.
        """
        curve, _inverse = _curves(record)
        if curve is None:
            raise ValueError(f"wiring {_record_id(record)}: the energy knob states no curve")
        fields = _scalar_fields(record, scalar_fields, writable=True)
        nominal = field(record, "default")
        if nominal is None:
            nominal = fields.get("default_value")
        if nominal is None:
            raise ValueError(f"wiring {_record_id(record)}: the energy knob states no default")
        return cls(
            calibration=curve,
            nominal=float(nominal),
            deck_energy_gev=deck_energy_gev,
            **fields,
        )

    def couple(self, variables: Iterable[PyATWritableScalarVariable]) -> tuple[str, ...]:
        """Adopt the rigidity-scaled variables this knob rescales, and name them.

        Separate from construction because the energy knob is one record among
        a facility's thousands: nothing can hand a constructor the whole set
        while it is still being built. Call it once, before the model adopts
        the variables -- :meth:`snapshot_targets` reports what they touch, and
        the model reads that both when it validates the variable set and when
        it snapshots a batch.

        Args:
            variables: every writable of the model. The rigidity-scaled ones
                are selected here, so a caller hands over the whole set.

        Returns:
            The names adopted, in the order they were given.
        """
        self._scaled = tuple(
            variable
            for variable in variables
            if isinstance(variable, CalibratedSetpoint) and variable.energy_scaling == "brho"
        )
        return tuple(variable.name for variable in self._scaled)

    def snapshot_targets(self, simulator: PyATSimulator) -> list[SnapshotTarget]:
        """The lattice energy, every cavity's, and every field the rescale touches.

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
            ValueError: the curve maps ``value`` to an energy that is not
                positive and finite. Nothing has been written.
        """
        energy_gev = self._energy_gev(value)
        if not math.isfinite(energy_gev) or energy_gev <= 0.0:
            raise ValueError(
                f"setpoint {value} maps to a beam energy of {energy_gev} GeV; "
                "a beam energy is positive and finite"
            )
        # Read before the write, for _after_write: it runs once the new energy
        # is on the lattice, by which time the energy it moved from is gone.
        self._energy_before_write = float(simulator.lattice.energy)
        super()._set(simulator, energy_gev * EV_PER_GEV)

    def _get(self, simulator: PyATSimulator) -> float:
        """Always raises: the lattice energy does not say what setpoint made it.

        The model serves a read of this knob from the value retained at its
        last write, which is the answer a client gets.

        Raises:
            NotImplementedError: always.
        """
        raise NotImplementedError(
            f"variable {self.name!r} drives the deck energy through the bend's curve; "
            "the setpoint behind the present energy is the value last written"
        )

    def _after_write(self, lattice: at.Lattice, _value: float) -> None:
        """Rescale every adopted setpoint to the energy just written.

        Args:
            lattice: the live lattice, already carrying the new energy.
            _value: the energy just written, in eV.
        """
        before = self._energy_before_write
        self._energy_before_write = None
        # The lattice's own number rather than the one just handed to it:
        # pyAT re-derives the energy it is given, and a setpoint written
        # afterwards takes its factor from what the lattice holds, so the
        # rescale has to use the same reading for the two paths to agree.
        after = float(lattice.energy)
        if before is None or before == after or not self._scaled:
            return
        factor = energy_factor(after / EV_PER_GEV, before / EV_PER_GEV)
        # One pass over the lattice, keyed by the name the bindings use. Every
        # bound name reaches exactly one element -- the model checks that when
        # it adopts the variables.
        elements = {element.FamName: element for element in lattice}
        for variable in self._scaled:
            variable.rescale(elements, factor)

    def _energy_gev(self, value: float) -> float:
        """The beam energy ``value`` on the bend implies, in GeV."""
        reference = to_physics(self.calibration, self.nominal)
        return self.deck_energy_gev * to_physics(self.calibration, value) / reference


def variable_from_wiring(
    record: Any, *, deck_energy_gev: float, **scalar_fields: Any
) -> ScalarVariable | None:
    """Build the variable a wiring record's engine block describes.

    * ``axis`` and no ``attribute``: a :class:`MonitorVariable`;
    * ``attribute: energy`` and no element: the :class:`EnergyVariable`;
    * ``attribute: KickAngle``: a :class:`KickVariable`;
    * ``attribute: Frequency``: an :class:`RFVariable`;
    * any other attribute on an element or slices: a :class:`StrengthVariable`.

    A record naming no element, no slices and no deck property (an optics
    output such as ``attribute: tune``) is not an element variable: the model
    serves it from its own solve, and ``None`` is returned.

    Args:
        record: The wiring record.
        deck_energy_gev: The energy the deck is built for.
        **scalar_fields: ``ScalarVariable`` fields, each winning over the one
            the record states.

    Returns:
        The variable, or ``None`` for a record that is not an element variable.

    Raises:
        ValueError: the record cannot be built (see each ``from_wiring``).
    """
    block = field(record, "engine")
    attribute = field(block, "attribute")
    has_element = field(record, "element") is not None or bool(field(record, "slices"))
    if attribute is None and field(block, "axis") is not None:
        return MonitorVariable.from_wiring(record, **scalar_fields)
    if not has_element:
        if attribute == ENERGY_ATTRIBUTE:
            return EnergyVariable.from_wiring(
                record, deck_energy_gev=deck_energy_gev, **scalar_fields
            )
        return None
    kind: type[CalibratedSetpoint] = StrengthVariable
    if attribute == KICK_ATTRIBUTE:
        kind = KickVariable
    elif attribute == FREQUENCY_ATTRIBUTE:
        kind = RFVariable
    return kind.from_wiring(record, deck_energy_gev=deck_energy_gev, **scalar_fields)


class PyATWritableEnumVariable(WritableActionMixin[PyATSimulator], EnumVariable):
    """A settable discrete value bound to one attribute of one deck element.

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
    "ENERGY_ATTRIBUTE",
    "EV_PER_GEV",
    "CalibratedSetpoint",
    "EnergyVariable",
    "KickVariable",
    "MonitorVariable",
    "PyATReadOnlyNDVariable",
    "PyATWritableEnumVariable",
    "RFVariable",
    "StrengthVariable",
    "variable_from_wiring",
]
