"""The ALS-U AR current->strength transformer, as a pyAT action variable.

``lume_pyat`` knows only about native pyAT quantities: its
:class:`~lume_pyat.actions.PyATWritableScalarVariable` binds a name to one
attribute of one element and writes the value it is handed, unconverted.
Everything that makes a number mean *Amps of magnet current in the ALS-U
accumulator ring* lives here, on this side of the package boundary --
:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`,
the per-family formulas it owns, and ``AMPS_PER_RADIAN_KICK`` stay in
OSPREY and are called, unchanged, from :meth:`CurrentSetpointVariable._set`.
That is the whole of the facility adapter's write path: the same
``StrengthMap.apply`` call the model has always made, now reached through a
variable instead of a method.

**The declared attribute is what rollback snapshots.**
``LUMEPyATModel._set`` captures ``getattr(element, variable.attribute)``
before a batch and restores it if anything fails, so the attribute a
variable declares has to cover every field the write actually touches --
not merely the one it nominally names. :data:`DECLARED_ATTRIBUTE` is
derived from ``StrengthMap``'s own family sets and records that write
footprint per family:

- ``HCM``/``VCM`` -> ``KickAngle`` (the write is to one plane; the snapshot
  is the whole two-element sequence).
- every magnet family -> ``PolynomB``. Quadrupoles are written through the
  ``K`` alias, which pyAT routes onto ``PolynomB[1]``, so declaring
  ``PolynomB`` snapshots the storage the write lands in; sextupoles write
  index 2; dipoles write index 0, which has no scalar alias at all.
  Declaring the whole array is what makes one rule cover all three.

Errors keep the model layer's vocabulary. ``StrengthMap.apply`` signals an
unrecognized family or a missing element with a bare :class:`ValueError`;
both ``_set`` and ``_get`` convert it to
:class:`~lume_pyat.exceptions.UnknownElementError`, which is what
``model.pyat``'s ``UnknownDeviceError`` names -- so a caller catching
``UnknownDeviceError`` catches these unchanged.

**A discrete value gets its own writable.** ``lume_pyat`` binds only
scalars, and a scalar's range admits every value between its bounds. A BPM
polarity is a sign: ``-1`` or ``1``, with nothing in between that means a
smaller polarity. :class:`PyATWritableEnumVariable` is the enum twin of
:class:`~lume_pyat.actions.PyATWritableScalarVariable` -- the same binding
and the same raw, unconverted read and write -- whose value is checked
against a list of options instead of a range.

**A quantity of the whole lattice binds no element.** Tunes, or beta at
every BPM, belong to the solved ring rather than to any one element, and
come out as arrays. :class:`PyATReadOnlyNDVariable` declares one -- a name,
a shape, ``float64`` -- with ``element_name`` fixed at ``None``. It only
declares the value. Computing it is the owning model's job, because what
makes such a quantity affordable is *when* it is computed (on read, at most
once per solve), and that is model state, not variable state.
"""

from __future__ import annotations

from typing import Any

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import EnumVariable, NDVariable
from lume_pyat.actions import PyATWritableScalarVariable
from lume_pyat.exceptions import UnknownElementError
from lume_pyat.simulator import PyATSimulator
from pydantic import PrivateAttr, model_validator

from osprey.services.virtual_accelerator.lattice.calibration import AMPS_PER_RADIAN_KICK
from osprey.services.virtual_accelerator.lattice.strengths import (
    # Which KickAngle index each corrector family writes. Imported rather
    # than restated: it is the same fact StrengthMap.apply writes through,
    # and a second copy would be a second thing to keep in step.
    _CORRECTOR_PLANE,
    CORRECTOR_FAMILIES,
    DIPOLE_FAMILY,
    QUADRUPOLE_FAMILIES,
    SEXTUPOLE_FAMILIES,
    StrengthMap,
)

# Element attribute each family's write lands in -- see the module docstring
# for why the whole array is declared rather than the index. Built from
# StrengthMap's own family sets so a family added there cannot quietly go
# missing here.
DECLARED_ATTRIBUTE: dict[str, str] = {
    **dict.fromkeys(CORRECTOR_FAMILIES, "KickAngle"),
    **dict.fromkeys(QUADRUPOLE_FAMILIES, "PolynomB"),
    **dict.fromkeys(SEXTUPOLE_FAMILIES, "PolynomB"),
    DIPOLE_FAMILY: "PolynomB",
}


class CurrentSetpointVariable(PyATWritableScalarVariable):
    """One magnet or corrector current setpoint, in Amps.

    Writes go through ``StrengthMap.apply``, which converts the current to
    the family's strength and mutates the element -- so the value this
    variable carries is a *current*, while the lattice underneath holds a
    strength. The ``element_name``/``attribute`` binding it inherits is not
    used to perform the write (``StrengthMap`` finds its own element by
    ``FamName``); it is what tells ``LUMEPyATModel`` which element to
    validate at construction and which attribute to snapshot for rollback.

    The current is post-calibration and absolute, not a delta: a seeded
    magnet calibration error acts on the commanded current before it reaches
    this variable.

    Attributes:
        family: Family token, e.g. ``"QF"``, ``"DIPOLE"``, ``"HCM"``.
        device_id: Zero-padded family-scoped device id, e.g. ``"01"``.
    """

    family: str
    device_id: str

    # A pydantic field would need `arbitrary_types_allowed` and would drag a
    # ring-sized object into every model_dump; the map is shared by all 348
    # setpoints and is infrastructure, not part of the variable's declared
    # shape.
    _strength_map: StrengthMap = PrivateAttr()

    def __init__(self, *, strength_map: StrengthMap, **data: Any) -> None:
        """Build the variable and attach the shared strength map.

        ``strength_map`` is keyword-only and separate from the validated
        fields because pydantic private attributes cannot be populated
        through validation.

        Args:
            strength_map: The map to convert currents through. Shared, not
                copied -- every setpoint of one ring uses the same instance,
                and it is the ring's baked-strength baseline.
            **data: The declared fields. ``element_name`` and ``attribute``
                are derived from ``family``/``device_id`` when omitted.

        Raises:
            pydantic.ValidationError: a field is missing or invalid,
                including a ``family`` no ``StrengthMap`` formula covers and
                an ``element_name``/``attribute`` that contradicts the
                derivation. Both are mistakes in the *definition* of a
                variable, so they surface here rather than at the first
                write.
        """
        super().__init__(**data)
        self._strength_map = strength_map

    @model_validator(mode="before")
    @classmethod
    def _derive_binding(cls, data: Any) -> Any:
        """Fill in ``element_name``/``attribute`` from family and device id.

        A caller may still pass them -- the bindings layer derives the
        element name for the read-only monitors anyway, so it has one to
        hand -- but only the values this would have derived. Silently
        accepting a different binding would leave the variable writing one
        element through ``StrengthMap`` and snapshotting another.
        """
        if not isinstance(data, dict):
            return data
        family = data.get("family")
        device_id = data.get("device_id")
        if not isinstance(family, str) or not isinstance(device_id, str):
            return data  # let field validation report the missing/ill-typed field

        attribute = DECLARED_ATTRIBUTE.get(family)
        if attribute is None:
            raise UnknownElementError(
                f"family {family!r} is not pyat-coupled: no StrengthMap formula "
                f"covers it (known families: {', '.join(sorted(DECLARED_ATTRIBUTE))})"
            )

        derived = {"element_name": f"{family}{device_id}", "attribute": attribute}
        for field, value in derived.items():
            declared = data.get(field)
            if declared is not None and declared != value:
                raise ValueError(
                    f"{field}={declared!r} contradicts family={family!r} "
                    f"device_id={device_id!r}, which binds {field}={value!r}"
                )
        return {**data, **derived}

    def _set(self, simulator: PyATSimulator, value: float) -> None:
        """Convert ``value`` (Amps) to strength and write it onto the lattice.

        Raises:
            UnknownElementError: the family is not pyat-coupled, or the
                lattice has no element for it. Nothing has been written.
        """
        try:
            self._strength_map.apply(simulator.lattice, self.family, self.device_id, value)
        except ValueError as exc:
            # StrengthMap.apply signals an unrecognized family or a missing
            # element with a bare ValueError; name it for what it is.
            raise UnknownElementError(
                f"family {self.family!r} (device {self.element_name!r}) is not pyat-coupled: {exc}"
            ) from exc

    def _get(self, simulator: PyATSimulator) -> float:
        """Return the current the element's present strength implies, in Amps.

        The inverse of :meth:`_set`, per family. Off the serving path --
        ``LUMEPyATModel`` answers reads of a writable from the value retained
        at its last successful write, which is bit-exact where this is a
        round trip through the strength formula. It is what a caller driving
        a simulator directly gets, and the honest answer to "what current is
        this element sitting at".

        Raises:
            UnknownElementError: the family is not pyat-coupled, or the
                lattice has no element for it.
            ZeroDivisionError: the family scales a baked strength that is
                zero for this device, leaving no current to recover.
        """
        if self.family not in DECLARED_ATTRIBUTE:
            raise UnknownElementError(f"family {self.family!r} is not pyat-coupled")
        element = simulator.element(self.element_name)

        if self.family in CORRECTOR_FAMILIES:
            kick = float(element.KickAngle[_CORRECTOR_PLANE[self.family]])
            return kick * AMPS_PER_RADIAN_KICK

        if self.family in QUADRUPOLE_FAMILIES:
            fraction = float(element.K) / self._strength_map.baked(self.element_name)
        elif self.family == DIPOLE_FAMILY:
            field_error = float(element.PolynomB[0])
            fraction = field_error * float(element.Length) / float(element.BendingAngle) + 1.0
        else:
            fraction = float(element.PolynomB[2]) / self._strength_map.baked(self.element_name)
        i_nom = self._strength_map.i_nom_for(self.family, self.device_id)
        return i_nom * fraction


class PyATWritableEnumVariable(WritableActionMixin[PyATSimulator], EnumVariable):
    """A settable discrete value bound to one attribute of one lattice element.

    Binds, reads and writes exactly as
    :class:`~lume_pyat.actions.PyATWritableScalarVariable` does; only the
    validation differs, and that is inherited from ``EnumVariable``: a value
    not in ``options`` is refused by ``LUMEModel.set`` before any write, and
    a ``default_value`` not in ``options`` fails at definition.

    Options must be numeric. ``LUMEPyATModel`` retains every writable's
    default as ``float(default_value)``, so a non-numeric option could be
    declared but never booted.

    Attributes:
        element_name: ``FamName`` of the target element.
        attribute: The element attribute to read and write.
        index: Position within ``attribute`` when it holds a sequence;
            ``None`` for a scalar attribute.
    """

    element_name: str
    attribute: str
    index: int | None = None

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

    def _get(self, simulator: PyATSimulator) -> float:
        """Read the bound attribute, unconverted.

        Raises:
            UnknownElementError: the lattice has no element ``element_name``.
        """
        value = getattr(simulator.element(self.element_name), self.attribute)
        if self.index is not None:
            value = value[self.index]
        return float(value)

    def _set(self, simulator: PyATSimulator, value: Any) -> None:
        """Write the bound attribute, unconverted.

        Raises:
            UnknownElementError: the lattice has no element ``element_name``.
            AttributeError: the element has no such attribute. pyAT elements
                accept arbitrary attribute assignment, so the write would
                otherwise land on a dead attribute and read back intact.
                Through ``LUMEPyATModel`` this cannot be reached: the model
                checks the same condition when it adopts the variable.
        """
        element = simulator.element(self.element_name)
        if not hasattr(element, self.attribute):
            raise AttributeError(
                f"element {self.element_name!r} has no attribute {self.attribute!r} to write"
            )
        if self.index is None:
            setattr(element, self.attribute, value)
        else:
            getattr(element, self.attribute)[self.index] = value


class PyATReadOnlyNDVariable(ReadOnlyActionMixin[PyATSimulator], NDVariable):
    """A read-only array derived from the whole solved lattice.

    Declares a name, a shape and a ``float64`` dtype, which
    ``LUMEModel.get`` checks every returned value against exactly, and
    nothing more. The value is computed by the model that declares the
    variable, never by the variable: see the module docstring.

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
    "DECLARED_ATTRIBUTE",
    "CurrentSetpointVariable",
    "PyATReadOnlyNDVariable",
    "PyATWritableEnumVariable",
]
