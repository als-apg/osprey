"""A button pickup: what each button of a monitor reads of a beam at (x, y).

A turn-by-turn monitor may state a *pickup*: its buttons on the chamber wall
and the way its electronics form positions from them. With one stated, the
turn read-out no longer takes the beam position as given. It computes each
button's signal at the beam position, and forms the sum and the two positions
from the button signals, as the electronics do, so both carry the pickup's
own non-linearity.

Two shapes, one interface (:class:`Pickup`):

* ``round`` -- buttons on a round pipe of radius ``a``. A button centred at
  angle ``phi`` with angular half-width ``delta`` reads, of a relativistic
  beam at ``(r, theta)``, the wall image charge averaged over its arc::

      1 + 2 * sum_n (r/a)^n * cos(n (theta - phi)) * sin(n delta) / (n delta)

  The series is the Poisson kernel integrated over the arc, which has a
  closed form (:func:`round_response`); the engine evaluates that, so there
  is no truncation and no loss of accuracy near the wall. A beam at or
  beyond :data:`WALL_SHARE` of the radius is read at that share of it, in
  its own direction.
* ``table`` -- each button's relative response on a regular grid in x and y,
  computed outside the engine for a chamber that is not round. Between the
  grid points it is interpolated, bilinearly or bicubically
  (:data:`INTERPOLATIONS`); outside the grid the beam position is clamped to
  the grid's edge, plane by plane.

Either way the responses are normalised so that the buttons of a centred beam
sum to 1 (:attr:`Pickup.centre`). A position is the electronics' difference
over sum times a scale: the buttons of the plane's ``plus`` list minus those
of its ``minus`` list, over all buttons, times ``scale``. A plane that states
no scale takes the one that makes the reading exact to first order at the
centre (:func:`first_order_scale`).

:func:`parse` reads one pickup statement of a model's turn-by-turn block.
Everything here takes plain numbers and arrays; nothing reads a lattice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "INTERPOLATIONS",
    "PLANE_KEYS",
    "ROUND_KEYS",
    "SHAPES",
    "TABLE_KEYS",
    "WALL_SHARE",
    "Pickup",
    "RoundPickup",
    "TablePickup",
    "first_order_scale",
    "parse",
    "round_response",
]

#: The pickup shapes.
SHAPES: tuple[str, ...] = ("round", "table")
#: The ways a table is read between its grid points, the first the default.
INTERPOLATIONS: tuple[str, ...] = ("bilinear", "bicubic")
#: The share of a round pipe's radius a beam is read at, at most.
WALL_SHARE = 0.999

#: The keys of a plane's difference-over-sum statement.
PLANE_KEYS: tuple[str, ...] = ("plus", "minus", "scale")
#: The keys a round pickup may state, beside ``elements`` on a list item.
ROUND_KEYS: tuple[str, ...] = ("shape", "radius", "buttons", "half_width", "diameter", "x", "y")
#: The keys a tabulated pickup may state, beside ``elements`` on a list item.
TABLE_KEYS: tuple[str, ...] = ("shape", "table", "buttons", "interpolation", "x", "y")

#: The step of the central difference that finds a table's slope at the
#: centre, as a share of the grid's spacing.
_SLOPE_STEP = 1.0e-3


#: What a shape's statement gives: the button count, each button's response
#: to a centred beam, its slope along x and along y there, and the builder of
#: the pickup from the two planes' signs and scales.
_Stated = tuple[
    int,
    np.ndarray,
    tuple[np.ndarray, np.ndarray],
    Callable[[np.ndarray, np.ndarray, float, float], "Pickup"],
]


def round_response(x: Any, y: Any, radius: float, angles: Any, half_width: float) -> np.ndarray:
    """Each button's reading of a beam at ``(x, y)`` in a round pipe, 1 for a centred beam.

    The wall image charge of a line charge at ``(r, theta)`` in a round pipe
    of radius ``a`` is the Poisson kernel; its mean over a button's arc
    ``[phi - delta, phi + delta]`` is, with ``F(u) = atan2(rho sin u, 1 - rho
    cos u)`` and ``rho = r / a``::

        1 + (F(phi + delta - theta) - F(phi - delta - theta)) / delta

    Args:
        x: Horizontal beam positions, metres; a float or an array.
        y: Vertical beam positions, shaped like ``x``.
        radius: The pipe's radius, metres.
        angles: The buttons' centre angles, radians, counterclockwise from
            the positive x axis.
        half_width: The buttons' angular half-width, radians.

    Returns:
        An array of shape ``(buttons, *x.shape)``.
    """
    u = np.asarray(x, dtype=np.float64) / radius
    v = np.asarray(y, dtype=np.float64) / radius
    rho = np.hypot(u, v)
    shrink = np.where(rho > WALL_SHARE, WALL_SHARE / np.where(rho > 0.0, rho, 1.0), 1.0)
    u, v = u * shrink, v * shrink
    phi = np.asarray(angles, dtype=np.float64).reshape((-1,) + (1,) * u.ndim)

    def primitive(psi: np.ndarray) -> np.ndarray:
        cos, sin = np.cos(psi), np.sin(psi)
        angle: np.ndarray = np.arctan2(u * sin - v * cos, 1.0 - (u * cos + v * sin))
        return angle

    response: np.ndarray = (
        1.0 + (primitive(phi + half_width) - primitive(phi - half_width)) / half_width
    )
    return response


@dataclass(frozen=True, eq=False)
class Pickup:
    """A monitor's buttons and its electronics' difference over sum.

    Attributes:
        buttons: How many buttons the monitor has.
        signs_x: Per button, ``+1`` where it adds to the horizontal
            difference, ``-1`` where it subtracts, ``0`` where it takes no
            part.
        signs_y: The same for the vertical difference.
        scale_x: The horizontal reading per unit of difference over sum,
            metres.
        scale_y: The vertical one.
        centre: Each button's response to a centred beam; they sum to 1.
    """

    buttons: int
    signs_x: np.ndarray
    signs_y: np.ndarray
    scale_x: float
    scale_y: float
    centre: np.ndarray

    def response(self, x: Any, y: Any) -> np.ndarray:
        """Each button's response to a beam at ``(x, y)``, shape ``(buttons, *x.shape)``.

        Normalised so the buttons of a centred beam sum to 1.
        """
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class RoundPickup(Pickup):
    """Buttons on a round pipe.

    Attributes:
        radius: The pipe's radius, metres.
        angles: The buttons' centre angles, radians.
        half_width: The buttons' angular half-width, radians.
    """

    radius: float = 0.0
    angles: np.ndarray | None = None
    half_width: float = 0.0

    def response(self, x: Any, y: Any) -> np.ndarray:
        """Each button's response, from the closed form of the wall image charge."""
        return round_response(x, y, self.radius, self.angles, self.half_width) / self.buttons


@dataclass(frozen=True, eq=False)
class TablePickup(Pickup):
    """Buttons read off a table on a regular grid.

    Attributes:
        grid_x: The grid's x values, metres, increasing in equal steps.
        grid_y: The grid's y values.
        values: The normalised responses, shape ``(buttons, len(grid_y),
            len(grid_x))``.
        interpolation: One of :data:`INTERPOLATIONS`.
    """

    grid_x: np.ndarray | None = None
    grid_y: np.ndarray | None = None
    values: np.ndarray | None = None
    interpolation: str = INTERPOLATIONS[0]

    def response(self, x: Any, y: Any) -> np.ndarray:
        """Each button's response, interpolated; a position off the grid reads at its edge."""
        assert self.grid_x is not None and self.grid_y is not None and self.values is not None
        return _interpolate(self.values, self.grid_x, self.grid_y, x, y, self.interpolation)


def _cell(grid: np.ndarray, position: Any) -> tuple[np.ndarray, np.ndarray]:
    """The index of the grid point at or below each position, and the share of the step past it."""
    last = grid.shape[0] - 1
    step = (grid[-1] - grid[0]) / last
    place = (np.clip(np.asarray(position, dtype=np.float64), grid[0], grid[-1]) - grid[0]) / step
    index = np.clip(np.floor(place).astype(np.intp), 0, last - 1)
    return index, place - index


def _cubic_weights(share: np.ndarray) -> tuple[np.ndarray, ...]:
    """The Catmull-Rom weights of the four points around a position ``share`` into its step."""
    t, t2, t3 = share, share * share, share * share * share
    return (
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    )


def _interpolate(
    values: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, x: Any, y: Any, how: str
) -> np.ndarray:
    """``values`` (``(buttons, ny, nx)``) at ``(x, y)``, clamped to the grid.

    ``bilinear`` reads the four points of the cell. ``bicubic`` is the
    Catmull-Rom spline through the sixteen points around it; a point the
    grid does not have is the grid's edge point.
    """
    column, share_x = _cell(grid_x, x)
    row, share_y = _cell(grid_y, y)
    if how == "bilinear":
        low = values[:, row, column] * (1.0 - share_x) + values[:, row, column + 1] * share_x
        high = (
            values[:, row + 1, column] * (1.0 - share_x) + values[:, row + 1, column + 1] * share_x
        )
        mixed: np.ndarray = low * (1.0 - share_y) + high * share_y
        return mixed
    reach = np.arange(-1, 3).reshape((4,) + (1,) * np.ndim(share_x))
    rows = np.clip(row + reach, 0, grid_y.shape[0] - 1)
    columns = np.clip(column + reach, 0, grid_x.shape[0] - 1)
    around = values[:, rows[:, None], columns[None, :]]
    weights = np.stack(_cubic_weights(share_y))[:, None] * np.stack(_cubic_weights(share_x))[None]
    total: np.ndarray = (around * weights).sum(axis=(1, 2))
    return total


def first_order_scale(signs: np.ndarray, centre: np.ndarray, slope: np.ndarray) -> float | None:
    """The scale that makes a plane's reading exact to first order at the centre.

    The reading is ``scale * D / S`` with ``D`` the signed and ``S`` the
    plain sum of the button responses. Its slope along the plane at the
    centre is ``scale * (D' - D S')`` there, since the responses sum to 1.

    Args:
        signs: The plane's sign per button.
        centre: Each button's response to a centred beam.
        slope: Each button's slope along the plane at the centre, per metre.

    Returns:
        The scale in metres, or ``None`` where the plane's difference over
        sum does not move with the beam to first order.
    """
    rate = float(signs @ slope - (signs @ centre) * slope.sum())
    if not math.isfinite(rate) or abs(rate) < 1.0e-12:
        return None
    return 1.0 / rate


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def parse(
    stated: Any,
    name: str,
    directory: Path,
    stop: Callable[[str, str], Exception],
    *,
    extra_keys: tuple[str, ...] = (),
) -> Pickup:
    """Check one pickup statement and build its :class:`Pickup`.

    Args:
        stated: The statement, a map.
        name: What a stop calls the statement, such as
            ``settings key pyat.turn_by_turn.pickup``.
        directory: The directory a ``table`` path is relative to.
        stop: Builds the error to raise from a detail and a remedy.
        extra_keys: Keys the statement may carry beside the pickup's own.

    Returns:
        The pickup.

    Raises:
        Exception: whatever ``stop`` builds, for a statement that is not a
            map, names an unknown shape or key, misstates a number, a button
            list or a plane, or names a table that is missing or malformed.
    """
    if not isinstance(stated, Mapping):
        raise stop(f"{name} is not a map", f"state it as a map with shape {' or '.join(SHAPES)}")
    shape = stated.get("shape")
    if shape not in SHAPES:
        raise stop(f"{name}.shape is {shape!r}", f"set shape to {' or '.join(SHAPES)}")
    keys = ROUND_KEYS if shape == "round" else TABLE_KEYS
    unknown = sorted(str(key) for key in stated if key not in (*keys, *extra_keys))
    if unknown:
        raise stop(
            f"{name} carries unknown keys {', '.join(unknown)}",
            f"a {shape} pickup states only {', '.join(keys)}",
        )
    if shape == "round":
        count, centre, slopes, build = _round(stated, name, stop)
    else:
        count, centre, slopes, build = _table(stated, name, directory, stop)
    signs, scales = [], []
    for plane, slope in zip(("x", "y"), slopes, strict=True):
        sign, scale = _plane(stated.get(plane), f"{name}.{plane}", count, stop)
        if scale is None:
            scale = first_order_scale(sign, centre, slope)
            if scale is None:
                raise stop(
                    f"{name}.{plane}: the difference over sum does not move with the beam "
                    f"along {plane} at the centre, so it has no first-order scale",
                    f"put buttons on both sides of the {plane} plane under plus and minus, "
                    f"or state {plane}.scale",
                )
        signs.append(sign)
        scales.append(scale)
    return build(signs[0], signs[1], scales[0], scales[1])


def _plane(
    stated: Any, name: str, count: int, stop: Callable[[str, str], Exception]
) -> tuple[np.ndarray, float | None]:
    """A plane's sign per button and its stated scale, ``None`` for the first-order one."""
    remedy = (
        f"state plus and minus, each a list of button numbers from 1 to {count}, "
        "and optionally scale"
    )
    if not isinstance(stated, Mapping):
        raise stop(f"{name} is {stated!r}, not a map", remedy)
    unknown = sorted(str(key) for key in stated if key not in PLANE_KEYS)
    if unknown:
        raise stop(f"{name} carries unknown keys {', '.join(unknown)}", remedy)
    signs = np.zeros(count, dtype=np.float64)
    for key, sign in (("plus", 1.0), ("minus", -1.0)):
        listed = stated.get(key)
        if not isinstance(listed, list | tuple) or not listed:
            raise stop(f"{name}.{key} is {listed!r}", remedy)
        for button in listed:
            if isinstance(button, bool) or not isinstance(button, int) or not 1 <= button <= count:
                raise stop(
                    f"{name}.{key} names button {button!r}; the pickup has buttons 1 to {count}",
                    remedy,
                )
            if signs[button - 1] != 0.0:
                raise stop(f"{name} names button {button} twice", remedy)
            signs[button - 1] = sign
    scale = stated.get("scale")
    if scale is None:
        return signs, None
    number = _number(scale)
    if number is None or number == 0.0:
        raise stop(
            f"{name}.scale is {scale!r}", "give scale as a finite number of metres other than 0"
        )
    return signs, number


def _round(stated: Mapping[str, Any], name: str, stop: Callable[[str, str], Exception]) -> _Stated:
    radius = _number(stated.get("radius"))
    if radius is None or radius <= 0.0:
        raise stop(
            f"{name}.radius is {stated.get('radius')!r}", "give the pipe's radius in metres above 0"
        )
    listed = stated.get("buttons")
    degrees = [_number(angle) for angle in listed] if isinstance(listed, list | tuple) else []
    if len(degrees) < 2 or None in degrees:
        raise stop(
            f"{name}.buttons is {listed!r}",
            "list the centre angle of each button in degrees, two buttons or more",
        )
    if ("half_width" in stated) == ("diameter" in stated):
        raise stop(
            f"{name} states {'both' if 'half_width' in stated else 'neither'} of half_width "
            "and diameter",
            "state the buttons' angular half_width in degrees, or their diameter in metres",
        )
    if "half_width" in stated:
        width = _number(stated["half_width"])
        if width is None or not 0.0 < width <= 90.0:
            raise stop(
                f"{name}.half_width is {stated['half_width']!r}",
                "give half_width in degrees, above 0 and at most 90",
            )
        half_width = math.radians(width)
    else:
        diameter = _number(stated["diameter"])
        if diameter is None or not 0.0 < diameter <= math.pi * radius:
            raise stop(
                f"{name}.diameter is {stated['diameter']!r}",
                "give diameter in metres, above 0 and at most pi times the radius",
            )
        # An arc of the button's width along the wall.
        half_width = diameter / (2.0 * radius)
    angles = np.radians(np.asarray(degrees, dtype=np.float64))
    count = len(degrees)
    centre = np.full(count, 1.0 / count)
    # The first-order term of the series: 2 (r/a) cos(theta - phi) sin(delta)/delta.
    rate = 2.0 * math.sin(half_width) / half_width / radius / count
    slopes = (rate * np.cos(angles), rate * np.sin(angles))

    def build(signs_x: np.ndarray, signs_y: np.ndarray, scale_x: float, scale_y: float) -> Pickup:
        return RoundPickup(
            buttons=count,
            signs_x=signs_x,
            signs_y=signs_y,
            scale_x=scale_x,
            scale_y=scale_y,
            centre=centre,
            radius=radius,
            angles=angles,
            half_width=half_width,
        )

    return count, centre, slopes, build


def _grid(document: Any, axis: str, where: str, stop: Callable[[str, str], Exception]) -> Any:
    remedy = (
        f"give {axis} as two or more values in metres, increasing in equal steps, "
        "with 0 inside their range"
    )
    listed = document.get(axis)
    values = [_number(value) for value in listed] if isinstance(listed, list) else []
    if len(values) < 2 or None in values:
        raise stop(f"{where}: {axis} is not a list of two or more numbers", remedy)
    grid = np.asarray(values, dtype=np.float64)
    steps = np.diff(grid)
    if not np.all(steps > 0.0):
        raise stop(f"{where}: {axis} is not strictly increasing", remedy)
    if not np.allclose(steps, steps[0], rtol=1.0e-6, atol=0.0):
        raise stop(f"{where}: {axis} is not a regular grid; its steps differ", remedy)
    if not grid[0] <= 0.0 <= grid[-1]:
        raise stop(
            f"{where}: {axis} runs from {grid[0]:g} to {grid[-1]:g} and does not hold 0", remedy
        )
    return grid


def _table(
    stated: Mapping[str, Any], name: str, directory: Path, stop: Callable[[str, str], Exception]
) -> _Stated:
    count = stated.get("buttons")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise stop(
            f"{name}.buttons is {count!r}",
            "give the number of buttons the table holds, two or more",
        )
    how = stated.get("interpolation", INTERPOLATIONS[0])
    if how not in INTERPOLATIONS:
        raise stop(
            f"{name}.interpolation is {how!r}",
            f"set interpolation to {' or '.join(INTERPOLATIONS)}",
        )
    stated_path = stated.get("table")
    if not isinstance(stated_path, str) or not stated_path:
        raise stop(
            f"{name}.table is {stated_path!r}",
            "name the table's JSON file, relative to the directory of the model's deck",
        )
    path = directory / stated_path
    where = f"{name}.table {stated_path} ({path})"
    shape_remedy = (
        "write the table as a JSON map of x, y and buttons: one array per button, each a "
        "list of len(y) rows of len(x) relative responses"
    )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise stop(
            f"{where} cannot be read: {error.strerror or error}",
            "put the table beside the model's deck, or correct the path",
        ) from error
    except ValueError as error:
        raise stop(f"{where} is not JSON: {error}", shape_remedy) from error
    if not isinstance(document, Mapping):
        raise stop(f"{where} is not a JSON map", shape_remedy)
    grid_x = _grid(document, "x", where, stop)
    grid_y = _grid(document, "y", where, stop)
    try:
        values = np.asarray(document.get("buttons"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise stop(f"{where}: buttons is not an array of numbers", shape_remedy) from error
    expected = (count, grid_y.shape[0], grid_x.shape[0])
    if values.shape != expected:
        raise stop(
            f"{where}: buttons has shape {list(values.shape)}; {count} buttons on a grid of "
            f"{expected[1]} y by {expected[2]} x values have shape {list(expected)}",
            shape_remedy,
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise stop(f"{where}: buttons holds a value that is negative or not finite", shape_remedy)
    raw_centre = _interpolate(values, grid_x, grid_y, 0.0, 0.0, how)
    total = float(raw_centre.sum())
    if not total > 0.0:
        raise stop(
            f"{where}: the responses sum to {total:g} at the centre",
            "give each button's response to a centred beam, above 0 in sum",
        )
    values = values / total
    centre = raw_centre / total
    step_x = _SLOPE_STEP * float(grid_x[1] - grid_x[0])
    step_y = _SLOPE_STEP * float(grid_y[1] - grid_y[0])
    ends = np.array([-1.0, 1.0])
    along_x = _interpolate(values, grid_x, grid_y, ends * step_x, np.zeros(2), how)
    along_y = _interpolate(values, grid_x, grid_y, np.zeros(2), ends * step_y, how)
    slopes = (
        (along_x[:, 1] - along_x[:, 0]) / (2.0 * step_x),
        (along_y[:, 1] - along_y[:, 0]) / (2.0 * step_y),
    )

    def build(signs_x: np.ndarray, signs_y: np.ndarray, scale_x: float, scale_y: float) -> Pickup:
        return TablePickup(
            buttons=count,
            signs_x=signs_x,
            signs_y=signs_y,
            scale_x=scale_x,
            scale_y=scale_y,
            centre=centre,
            grid_x=grid_x,
            grid_y=grid_y,
            values=values,
            interpolation=how,
        )

    return count, centre, slopes, build
