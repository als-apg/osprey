"""Wiring calibrations: the curve between a hardware value and a physics value.

A curve maps a channel's hardware value to the engine's physics value; its
inverse maps back. Either direction is a straight line (:class:`Linear`) or a
sampled table (:class:`Table`). A wiring record states its curves in the
facility file's shape, ``{linear: {gain, offset}}`` or ``{table: {grid,
values}}``, read by attribute or by key; :func:`curve_from_record` turns one
into these dataclasses and :func:`evaluate` applies it. Pure stdlib.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["Calibration", "Linear", "Table", "curve_from_record", "evaluate", "field"]


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
