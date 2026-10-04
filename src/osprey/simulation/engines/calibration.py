"""Wiring calibrations: the curve between a hardware value and a physics value.

A curve maps a channel's hardware value to the engine's physics value; its
inverse maps back. Either direction is a straight line (:class:`Linear`) or a
sampled table (:class:`Table`). A wiring record states its curves in the
facility file's shape, ``{linear: {gain, offset}}`` or ``{table: {grid,
values}}``, read by attribute or by key; :func:`curve_from_record` turns one
into these dataclasses and :func:`evaluate` applies it.

The way back from a physics value to hardware units is one rule,
:func:`to_hardware`: the record's ``inverse`` when it states one, else the
algebraic inverse of a ``linear`` curve. A ``table`` without an ``inverse``,
and a linear gain of 0, have no way back (:class:`NoInverse`).

A physics value stated at the deck energy is worth :func:`energy_factor` of
itself at another energy when it scales with the beam rigidity
(``energy_scaling: brho``). Pure stdlib.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "REST_MASS_GEV",
    "Calibration",
    "Linear",
    "NoInverse",
    "Table",
    "brho",
    "check_inverse",
    "curve_from_record",
    "energy_factor",
    "evaluate",
    "field",
    "to_hardware",
    "to_physics",
]

#: Electron rest mass in GeV. A deck's energy is kinetic, so the rest mass is
#: what turns it into a momentum.
REST_MASS_GEV: float = 0.51099906e-3

#: Tesla-metres of rigidity per GeV/c of momentum: 1e9 divided by the speed of
#: light in metres per second.
RIGIDITY_PER_MOMENTUM: float = 10.0 / 2.99792458


@dataclass(frozen=True)
class Linear:
    """A straight-line conversion ``y = gain * x + offset``."""

    gain: float
    offset: float


@dataclass(frozen=True)
class Table:
    """A sampled conversion: piecewise linear through ``(grid, values)``.

    ``grid`` is strictly monotonic, so the curve is well defined everywhere;
    beyond either end the consumer extrapolates along the last segment.
    """

    grid: tuple[float, ...]
    values: tuple[float, ...]


#: Either shape a conversion may take.
Calibration = Linear | Table


def field(record: Any, name: str) -> Any:
    """Return one field of a record read by key or by attribute.

    Args:
        record: A mapping, an object with attributes, or ``None``.
        name: The field name.

    Returns:
        The field's value, or ``None`` when the record does not carry it.
    """
    if record is None:
        return None
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def curve_from_record(curve: Any) -> Calibration | None:
    """Turn a facility-file curve into a :data:`Calibration`.

    Args:
        curve: ``{linear: {gain, offset}}`` or ``{table: {grid, values}}``,
            as a mapping or an object with those attributes; ``None`` for no
            curve.

    Returns:
        The curve as :class:`Linear` or :class:`Table`, or ``None`` when
        ``curve`` is ``None`` or states neither shape.
    """
    linear = field(curve, "linear")
    if linear is not None:
        return Linear(gain=float(field(linear, "gain")), offset=float(field(linear, "offset")))
    table = field(curve, "table")
    if table is not None:
        return Table(
            grid=tuple(float(x) for x in field(table, "grid")),
            values=tuple(float(y) for y in field(table, "values")),
        )
    return None


def evaluate(curve: Calibration, x: float) -> float:
    """Apply a curve to one value.

    A table interpolates linearly between its samples and extrapolates along
    its first or last segment beyond the grid.

    Args:
        curve: The conversion to apply.
        x: The value on the curve's input side.

    Returns:
        The value on the curve's output side.
    """
    if isinstance(curve, Linear):
        return curve.gain * x + curve.offset
    grid, values = curve.grid, curve.values
    if grid[0] > grid[-1]:
        grid, values = grid[::-1], values[::-1]
    segment = len(grid) - 2
    for index in range(1, len(grid) - 1):
        if x < grid[index]:
            segment = index - 1
            break
    x0, x1 = grid[segment], grid[segment + 1]
    y0, y1 = values[segment], values[segment + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class NoInverse(ValueError):
    """A calibration with no way back from a physics value to hardware units.

    Attributes:
        reason: ``table`` for a sampled curve stating no ``inverse``,
            ``zero-gain`` for a linear curve of gain 0.
    """

    def __init__(self, reason: Literal["table", "zero-gain"]) -> None:
        if reason == "table":
            message = "a table calibration has no inverse"
        else:
            message = "the linear calibration's gain is 0, so it has no inverse"
        super().__init__(message)
        self.reason = reason


def check_inverse(curve: Calibration | None, inverse: Calibration | None) -> None:
    """Refuse a calibration :func:`to_hardware` cannot map back.

    Args:
        curve: Hardware to physics; ``None`` for no calibration.
        inverse: Physics to hardware as the record states it, or ``None``.

    Raises:
        NoInverse: ``inverse`` is ``None`` and ``curve`` is a table or a
            linear curve of gain 0.
    """
    if inverse is not None or curve is None:
        return
    if isinstance(curve, Table):
        raise NoInverse("table")
    if curve.gain == 0.0:
        raise NoInverse("zero-gain")


def to_physics(curve: Calibration | None, hardware: float) -> float:
    """Map a hardware value to the engine's physics value.

    Args:
        curve: Hardware to physics; ``None`` for no calibration, under which
            the hardware value is the physics value.
        hardware: The value in the channel's unit.

    Returns:
        The physics value.
    """
    if curve is None:
        return float(hardware)
    return evaluate(curve, hardware)


def to_hardware(curve: Calibration | None, inverse: Calibration | None, physics: float) -> float:
    """Map a physics value back to hardware units.

    ``inverse`` when present, else the algebraic inverse of a linear
    ``curve``; with neither, the physics value is the hardware value.

    Args:
        curve: Hardware to physics; ``None`` for no calibration.
        inverse: Physics to hardware as the record states it, or ``None``.
        physics: The engine's value.

    Returns:
        The value in the channel's unit.

    Raises:
        NoInverse: ``inverse`` is ``None`` and ``curve`` is a table or a
            linear curve of gain 0.
    """
    check_inverse(curve, inverse)
    if inverse is not None:
        return evaluate(inverse, physics)
    if isinstance(curve, Linear):
        return (physics - curve.offset) / curve.gain
    return float(physics)


def brho(energy_gev: float) -> float:
    """Return the beam rigidity in tesla-metres at a kinetic energy.

    The rest mass enters the momentum twice, so the massless form ``E / c``
    is not used: at a few GeV it is high by parts in ten thousand.

    Args:
        energy_gev: The kinetic energy in GeV.

    Returns:
        The rigidity in tesla-metres.
    """
    total = float(energy_gev) + REST_MASS_GEV
    return RIGIDITY_PER_MOMENTUM * math.sqrt(total**2 - REST_MASS_GEV**2)


def energy_factor(energy_gev: float, deck_energy_gev: float) -> float:
    """Return what a rigidity-scaled physics value is worth at ``energy_gev``.

    A calibration states its physics value at the deck energy. The same
    hardware value bends a stiffer beam less, in the ratio of the two
    rigidities, and the factor is one at the deck energy.

    Args:
        energy_gev: The energy the beam is at, in GeV.
        deck_energy_gev: The energy the calibration was stated at, in GeV.

    Returns:
        ``brho(deck_energy_gev) / brho(energy_gev)``.
    """
    return brho(deck_energy_gev) / brho(energy_gev)
