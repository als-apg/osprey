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
* the RF-on switch (``attribute: Voltage`` on a cavity), built by
  :func:`rf_switch_from_wiring` -- a :class:`PyATWritableEnumVariable` over
  the indices of the record's two labels, off and on, each binding weighted
  by the voltage its cavity carries in the deck.
* a labelled readout setting (a readout attribute on a monitor element, its
  engine block marked ``discrete``), built by
  :func:`readout_setting_from_wiring` -- a :class:`ReadoutSettingVariable`
  over the indices of the record's labels, each index written as the value
  its calibration table states for it; its readback,
  :class:`ReadoutSettingReadback`, answers the index whose table value the
  element carries.

:func:`variable_from_wiring` picks the class from the engine block.

Each slice share is one reading of the same arithmetic: a write puts the
physics value times the slice's weight on the element, and a read divides the
first slice's reading by the first weight. A supply feeding several magnets in
series is the third reading of it: every magnet converts the one current the
supply delivers through its own curve, the slice's ``curve`` where it states
one, and its weight is the split share and the polarity.

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
from lume.variables import ConfigEnum, EnumVariable, NDVariable
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
    Table,
    check_inverse,
    curve_from_record,
    energy_factor,
    field,
    to_hardware,
    to_physics,
)
from osprey.simulation.engines.pyat_faults import (
    READOUT_ATTRIBUTE_IDENTITY,
    is_series_record,
    magnet_cal,
    supply_calibration,
)

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

#: The engine-block attribute of the RF-on switch: a cavity's voltage, which
#: the switch takes off and puts back whole.
VOLTAGE_ATTRIBUTE = "Voltage"

#: The value types whose channels name their states by labels.
_LABELLED_TYPES = ("bool", "enum")

#: The labels of a bool channel that states none, off first.
_BOOL_LABELS = ("FALSE", "TRUE")

#: The index of the off label and of the on label of the RF-on switch.
RF_OFF_INDEX = 0
RF_ON_INDEX = 1

#: The engine-block key marking a readout setting whose channel names its
#: states by labels: the value written is a label's index, not a reading.
DISCRETE_KEY = "discrete"

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


def _slice_curves(record: Any) -> tuple[Calibration | None, ...]:
    """Each slice's own ``curve``, aligned with the bindings; empty where none states one.

    Raises:
        ValueError: a slice states a curve that is neither ``linear`` nor
            ``table``.
    """
    slices = field(record, "slices") or ()
    curves: list[Calibration | None] = []
    for piece in slices:
        stated = field(piece, "curve")
        curve = curve_from_record(stated)
        if stated is not None and curve is None:
            raise ValueError(
                f"wiring {_record_id(record)}: slice {field(piece, 'element')} states a curve "
                "that is neither linear nor table"
            )
        curves.append(curve)
    if all(curve is None for curve in curves):
        return ()
    return tuple(curves)


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
    ``1/n`` for a piece that takes an equal share of a divisible one, and
    negative for a magnet wound the other way.

    **What a slice curve means.** A magnet a supply feeds in series converts
    the current the supply delivers through its own curve, which its slice
    states; a slice stating none converts through ``calibration``. The first
    slice states none: it is the one a read comes back from, through
    ``inverse``.

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
        slice_curves: Each binding's own hardware-to-physics conversion, in
            binding order, ``None`` for one converting through
            ``calibration``; empty when every binding does.
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
    slice_curves: tuple[Calibration | None, ...] = ()

    @model_validator(mode="after")
    def _require_a_way_back(self) -> CalibratedSetpoint:
        """Refuse a calibration with no way back to hardware units."""
        check_inverse(self.calibration, self.inverse)
        return self

    @model_validator(mode="after")
    def _read_back_through_the_record(self) -> CalibratedSetpoint:
        """Refuse slice curves that do not line up with the bindings or convert the read."""
        if not self.slice_curves:
            return self
        if len(self.slice_curves) != len(self.bindings):
            raise ValueError(
                f"{self.name}: {len(self.slice_curves)} slice curves for "
                f"{len(self.bindings)} bindings"
            )
        if self.slice_curves[0] is not None:
            raise ValueError(
                f"{self.name}: the first slice converts through the record's calibration"
            )
        return self

    @classmethod
    def from_wiring(
        cls, record: Any, *, deck_energy_gev: float, **scalar_fields: Any
    ) -> CalibratedSetpoint:
        """Build the setpoint one wiring record describes.

        Args:
            record: The wiring record, read by key or by attribute: ``address``,
                ``element`` or ``slices`` (each ``element``, ``weight``,
                ``curve``), the
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
                attribute, states an unknown ``energy_scaling``, a slice
                states a curve of neither shape or the first slice states one,
                or its calibration has no way back to hardware units.
        """
        curve, inverse = _curves(record)
        return cls(
            bindings=_element_bindings(record),
            calibration=curve,
            inverse=inverse,
            energy_scaling=_energy_scaling(record),
            deck_energy_gev=deck_energy_gev,
            slice_curves=_slice_curves(record),
            **_scalar_fields(record, scalar_fields, writable=True),
        )

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Convert ``value`` to physics and write it to every bound slice.

        The rigidity factor is taken from the energy the lattice is at when the
        write happens, not from the deck energy, so a setpoint written after
        an energy move lands where that move left the record. Every slice
        converts the one delivered value through its own curve; every element
        is checked before any is written, so a bad binding leaves the lattice
        untouched.

        Raises:
            AttributeError: an element has no such attribute.
        """
        if not self.slice_curves:
            super()._set(simulator, self._physics(simulator, value))
            return
        delivered = self._delivered(simulator, value)
        factor = self._rigidity_factor(simulator)
        elements = [simulator.element(binding.element_name) for binding in self.bindings]
        for binding, element in zip(self.bindings, elements, strict=True):
            if not hasattr(element, binding.attribute):
                raise AttributeError(
                    f"element {binding.element_name!r} has no attribute "
                    f"{binding.attribute!r} to write"
                )
        for binding, element, own in zip(self.bindings, elements, self.slice_curves, strict=True):
            curve = self.calibration if own is None else own
            weighted = to_physics(curve, delivered) * factor * binding.weight
            if binding.index is None:
                setattr(element, binding.attribute, weighted)
            else:
                getattr(element, binding.attribute)[binding.index] = weighted

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
        delivered = self._delivered(simulator, value)
        return to_physics(self.calibration, delivered) * self._rigidity_factor(simulator)

    def _delivered(self, simulator: PyATSimulator, value: float) -> float:
        """The hardware value the supply delivers when ``value`` is commanded."""
        element = simulator.element(self.bindings[0].element_name)
        return magnet_cal(value, **supply_calibration(element, self.name))

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
    the magnets a supply feeds in series. A split magnet's pieces each carry
    the whole strength, because the control system sets a strength rather than
    a strength to divide up; a series magnet converts the supply's current
    through its own curve, its weight carrying only its polarity.
    """


class KickVariable(CalibratedSetpoint):
    """One corrector setpoint, over the slices it is bound to.

    A kick *is* divisible: a corrector modelled as ``n`` pieces bends the beam
    by the sum of what its pieces do, so each piece takes ``1/n`` of the kick
    and reading the first slice back multiplies by ``n`` again, which is the
    value the control system reads. Where one supply bends several correctors
    in series, each converts the supply's current through its own curve and
    takes that share of it.
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
    serves it from its own solve, and ``None`` is returned. So is a record
    whose engine block carries ``series``, whatever else it names: a buffer of
    tracked samples the model fills, never a monitor.

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
    if is_series_record(record):
        return None
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


def is_rf_switch_record(record: Any) -> bool:
    """Whether a wiring record is the RF-on switch: a cavity's ``Voltage``.

    The record names an element or slices, and its engine block names the
    attribute :data:`VOLTAGE_ATTRIBUTE`.
    """
    has_element = field(record, "element") is not None or bool(field(record, "slices"))
    return has_element and field(field(record, "engine"), "attribute") == VOLTAGE_ATTRIBUTE


def rf_switch_labels(record: Any) -> list[str]:
    """The labels a record declares for the RF-on switch, off first.

    The record's ``options`` when it states them, else the two labels of a
    bool channel.

    Raises:
        ValueError: the record declares other than two labels, or a value
            type that carries no labels.
    """
    value_type = field(record, "value_type")
    if value_type is not None and value_type not in _LABELLED_TYPES:
        raise ValueError(
            f"wiring {_record_id(record)}: the RF-on switch is a discrete setting; "
            f"its value_type is {value_type!r}, not bool or enum"
        )
    labels = [str(label) for label in (field(record, "options") or _BOOL_LABELS)]
    if len(labels) != 2:
        raise ValueError(
            f"wiring {_record_id(record)}: the RF-on switch takes two labels, off and on; "
            f"it declares {len(labels)}: {labels}"
        )
    return labels


def _rf_switch_bindings(record: Any, voltages: Mapping[str, float]) -> list[ElementBinding]:
    """The RF-on switch's bindings: each cavity on ``Voltage``, weighted by its deck voltage.

    Raises:
        ValueError: the labels are not two (:func:`rf_switch_labels`), or a
            bound cavity carries no voltage, or zero, in the deck.
    """
    rf_switch_labels(record)
    bindings = []
    for binding in _element_bindings(record):
        voltage = voltages.get(binding.element_name)
        if voltage is None or not math.isfinite(voltage) or voltage == 0.0:
            raise ValueError(
                f"wiring {_record_id(record)}: cavity {binding.element_name} carries "
                f"Voltage {voltage} in the deck; the RF-on switch restores that voltage, "
                "so it must be finite and non-zero"
            )
        bindings.append(binding.model_copy(update={"weight": float(voltage)}))
    return bindings


def rf_switch_from_wiring(record: Any, voltages: Mapping[str, float]) -> PyATWritableEnumVariable:
    """Build the RF-on switch a cavity-voltage record describes.

    The options are the indices of the record's two labels
    (:func:`rf_switch_labels`): :data:`RF_OFF_INDEX` and :data:`RF_ON_INDEX`.
    Each binding is the record's cavity (each slice's element) on its
    ``Voltage``, weighted by the voltage that cavity carries in the deck, so a
    write of the on index puts the deck voltage back and the off index puts
    zero there. The default is on.

    Args:
        record: The wiring record.
        voltages: Each bound cavity's voltage in the deck, keyed by element
            name.

    Returns:
        The switch.

    Raises:
        ValueError: the labels are not two (:func:`rf_switch_labels`), or a
            bound cavity carries no voltage, or zero, in the deck.
    """
    return PyATWritableEnumVariable(
        name=str(field(record, "address")),
        bindings=_rf_switch_bindings(record, voltages),
        options=[RF_OFF_INDEX, RF_ON_INDEX],
        default_value=RF_ON_INDEX,
        default_validation_config=ConfigEnum.ERROR,
    )


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


def rf_switch_readback_from_wiring(
    record: Any, voltages: Mapping[str, float]
) -> PyATReadOnlyEnumVariable:
    """Build the readback of the RF-on switch a cavity-voltage read record describes.

    It reads the first bound cavity's voltage as a share of its deck voltage,
    so it answers the off or on index from the lattice itself, whichever
    writer last set the cavity.

    Args:
        record: The wiring record.
        voltages: Each bound cavity's voltage in the deck, keyed by element
            name.

    Returns:
        The readback.

    Raises:
        ValueError: as :func:`rf_switch_from_wiring`.
    """
    return PyATReadOnlyEnumVariable(
        name=str(field(record, "address")),
        bindings=_rf_switch_bindings(record, voltages),
        options=[RF_OFF_INDEX, RF_ON_INDEX],
        default_validation_config=ConfigEnum.ERROR,
    )


class PyATReadOnlyEnumVariable(ReadOnlyActionMixin[PyATSimulator], EnumVariable):
    """A discrete reading of one attribute of one deck element.

    Reads exactly as :class:`PyATWritableEnumVariable` does -- the first
    binding's attribute divided by its weight -- and is never written.

    Attributes:
        bindings: The elements the value is read from; the first is read.
    """

    bindings: list[ElementBinding] = Field(min_length=1)
    read_only: bool = True

    def _get(self, simulator: PyATSimulator) -> float:
        """Read the first binding's element, divided by its weight.

        Raises:
            UnknownElementError: the lattice has no such element.
        """
        first = self.bindings[0]
        value = getattr(simulator.element(first.element_name), first.attribute)
        if first.index is not None:
            value = value[first.index]
        return float(value) / first.weight


def is_readout_setting_record(record: Any) -> bool:
    """Whether a wiring record is a labelled readout setting.

    The record names an element or slices, its engine block names one of the
    readout attributes of
    :data:`~osprey.simulation.engines.pyat_faults.READOUT_ATTRIBUTE_IDENTITY`
    and carries :data:`DISCRETE_KEY` set true.
    """
    has_element = field(record, "element") is not None or bool(field(record, "slices"))
    block = field(record, "engine")
    return (
        has_element
        and field(block, "attribute") in READOUT_ATTRIBUTE_IDENTITY
        and field(block, DISCRETE_KEY) is True
    )


def readout_setting_levels(record: Any) -> list[float]:
    """The value a labelled readout setting writes for each label's index.

    The record's ``calibration.curve`` is a ``table`` whose grid is the label
    indices ``0, 1, ...`` in order; its values are what each index puts on
    the readout attribute.

    Raises:
        ValueError: the record states no table, or one whose grid is not the
            indices 0 to its length less one, or that has fewer than two
            entries or a value that is not finite.
    """
    curve, _ = _curves(record)
    if not isinstance(curve, Table):
        raise ValueError(
            f"wiring {_record_id(record)}: a discrete readout setting maps each label "
            "index through a calibration table; it states none"
        )
    indices = tuple(float(index) for index in range(len(curve.grid)))
    if len(curve.grid) < 2 or curve.grid != indices:
        raise ValueError(
            f"wiring {_record_id(record)}: a discrete readout setting's table grid is the "
            f"label indices 0, 1, ... in order; it states {list(curve.grid)}"
        )
    if not all(math.isfinite(value) for value in curve.values):
        raise ValueError(
            f"wiring {_record_id(record)}: a discrete readout setting's table values are "
            f"finite; it states {list(curve.values)}"
        )
    return list(curve.values)


def _nearest_index(levels: list[float], value: float) -> int:
    """The index whose level lies nearest ``value``; the first on a tie."""
    return min(range(len(levels)), key=lambda index: (abs(levels[index] - value), index))


def readout_setting_from_wiring(record: Any, default: Any) -> ReadoutSettingVariable:
    """Build the labelled readout setting a discrete readout record describes.

    The options are the label indices of the record's table
    (:func:`readout_setting_levels`); the bindings are the record's element
    or slices on its readout attribute.

    Args:
        record: The wiring record.
        default: The index the setting boots at, and returns to on reset.

    Returns:
        The setting.

    Raises:
        ValueError: the table is refused (:func:`readout_setting_levels`), or
            ``default`` is not one of its indices.
    """
    levels = readout_setting_levels(record)
    return ReadoutSettingVariable(
        name=str(field(record, "address")),
        bindings=_element_bindings(record),
        options=list(range(len(levels))),
        levels=levels,
        default_value=_index(record, default, len(levels)),
        default_validation_config=ConfigEnum.ERROR,
    )


def readout_setting_readback_from_wiring(record: Any) -> ReadoutSettingReadback:
    """Build the readback of a labelled readout setting a discrete read record describes.

    It reads the first bound element's readout attribute and answers the
    index whose table value lies nearest, so it follows whichever writer last
    set the element.

    Raises:
        ValueError: as :func:`readout_setting_levels`.
    """
    levels = readout_setting_levels(record)
    return ReadoutSettingReadback(
        name=str(field(record, "address")),
        bindings=_element_bindings(record),
        options=list(range(len(levels))),
        levels=levels,
        default_validation_config=ConfigEnum.ERROR,
    )


def _index(record: Any, value: Any, count: int) -> int:
    """``value`` as one of ``count`` label indices.

    Raises:
        ValueError: ``value`` is not a whole number from 0 to ``count`` less one.
    """
    number = float(value)
    if not number.is_integer() or not 0 <= number < count:
        raise ValueError(
            f"wiring {_record_id(record)}: {value!r} is no label index; "
            f"the setting takes 0 to {count - 1}"
        )
    return int(number)


class ReadoutSettingVariable(PyATWritableEnumVariable):
    """A labelled setting of a monitor's readout attribute.

    Its value is the index of one of its channel's labels; a write puts the
    level its table states for that index on every bound element, times the
    binding's weight, and a read answers the index whose level the first
    element carries.

    Attributes:
        levels: The readout attribute's value for each index, in index order.
    """

    levels: list[float] = Field(min_length=2)

    def _get(self, simulator: PyATSimulator) -> int:
        """The index whose level the first bound element carries, nearest first."""
        return _nearest_index(self.levels, super()._get(simulator))

    def _set(self, simulator: PyATSimulator, value: Any) -> None:
        """Write the level of index ``value`` to every bound element."""
        super()._set(simulator, self.levels[int(round(float(value)))])


class ReadoutSettingReadback(PyATReadOnlyEnumVariable):
    """The reading of a labelled readout setting, from the element itself.

    Attributes:
        levels: The readout attribute's value for each index, in index order.
    """

    levels: list[float] = Field(min_length=2)

    def _get(self, simulator: PyATSimulator) -> int:
        """The index whose level the first bound element carries, nearest first."""
        return _nearest_index(self.levels, super()._get(simulator))


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
    "DISCRETE_KEY",
    "CalibratedSetpoint",
    "EnergyVariable",
    "KickVariable",
    "MonitorVariable",
    "PyATReadOnlyEnumVariable",
    "PyATReadOnlyNDVariable",
    "PyATWritableEnumVariable",
    "RF_OFF_INDEX",
    "RF_ON_INDEX",
    "VOLTAGE_ATTRIBUTE",
    "RFVariable",
    "ReadoutSettingReadback",
    "ReadoutSettingVariable",
    "StrengthVariable",
    "is_readout_setting_record",
    "is_rf_switch_record",
    "rf_switch_from_wiring",
    "rf_switch_labels",
    "readout_setting_from_wiring",
    "readout_setting_levels",
    "readout_setting_readback_from_wiring",
    "rf_switch_readback_from_wiring",
    "variable_from_wiring",
]
