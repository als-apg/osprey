"""Resolve a variable's optimization bounds from a ``VariableOverride``."""

from __future__ import annotations

import math

from osprey_optimizer.models.api import ResolvedBoundsModel
from osprey_optimizer.models.config import VariableOverride
from osprey_optimizer.models.types import require_finite_positive

_STEP_GRID_EPS = 1e-9
"""Tolerance for the step-grid alignment check, in units of *step_size*.

Lets a value that lies on a grid point up to floating-point error count as
on-grid (e.g. ``0.1 + 0.2 == 0.30000000000000004``) instead of being snapped
to the next/previous point.
"""


def effective_step_size(override: VariableOverride | None, hardware_step: float) -> float:
    """The step grid a variable is quantized to: the override's step when set, else the hardware step.

    Shared by the optimization run and the cockpit bounds preview so the two
    never diverge — both snap every variable to the same grid. A variable is
    always gridded; only the granularity (override vs hardware) varies.
    """
    if override is not None and override.step_size is not None:
        return override.step_size
    return hardware_step


def snap_value_to_grid(value: float, *, vmin: float, vmax: float, step: float) -> float:
    """Round *value* to the step grid anchored at *vmin*, clamped to ``[vmin, vmax]``.

    The grid is ``vmin + k*step``. Used to quantize optimizer candidates at the
    evaluation seam, so a variable with a step only ever evaluates (and records)
    grid points. Idempotent: snapping a grid point returns it.
    """
    snapped = vmin + round((value - vmin) / step) * step
    return min(max(snapped, vmin), vmax)


def resolve_bounds(
    override: VariableOverride | None,
    current_value: float,
    safety_min: float,
    safety_max: float,
    step_size: float | None = None,
) -> ResolvedBoundsModel:
    """Return the optimization bounds for one variable.

    Three input modes:

    * ``override is None`` -> the plugin's full safety range.
    * Absolute             -> ``override.min`` / ``override.max`` (each falls
      back to the matching safety bound when ``None``).
    * Window               -> symmetric window around ``current_value``. Width
      is ``window_fraction`` (as a fraction of the safety range, so it is
      independent of ``current_value``'s magnitude) or ``window_abs`` (in
      absolute units). The resolved range is then clamped to the safety range.

    When ``step_size`` is provided, the clamped range is snapped to the step
    grid anchored at ``safety_min``: ``min`` is rounded up and ``max`` is
    rounded down, so the resolved interval is a subset of the requested one
    and every endpoint sits on a grid point.

    If clamping or snapping would collapse the interval (no overlap with the
    safety range, or no grid point inside the clamped range), the result
    collapses to a single grid point: the safety endpoint nearest the
    requested interval, snapped to the grid.

    Raises:
        ValueError: If the safety range is non-finite or inverted; if a
            window mode is requested with a non-finite ``current_value``; or
            if ``step_size`` is provided but is not a finite positive number.
    """
    if not (math.isfinite(safety_min) and math.isfinite(safety_max)):
        raise ValueError(f"Safety range [{safety_min}, {safety_max}] is non-finite")
    if safety_min > safety_max:
        raise ValueError(f"Safety range [{safety_min}, {safety_max}] is inverted")
    if step_size is not None:
        require_finite_positive(step_size, label="step_size")

    if override is None:
        snapped_min, snapped_max = _snap_to_grid(
            safety_min, safety_max, safety_min, safety_max, step_size
        )
        return ResolvedBoundsModel(
            min=snapped_min,
            max=snapped_max,
            requested_min=safety_min,
            requested_max=safety_max,
            was_clamped=snapped_min != safety_min or snapped_max != safety_max,
        )

    in_window_mode = override.window_fraction is not None or override.window_abs is not None
    if in_window_mode and not math.isfinite(current_value):
        raise ValueError(
            f"Window-mode override requires a finite current_value; got {current_value}"
        )

    if override.window_fraction is not None:
        half_width = override.window_fraction * (safety_max - safety_min) / 2.0
        requested_min = current_value - half_width
        requested_max = current_value + half_width
    elif override.window_abs is not None:
        width = override.window_abs
        requested_min = current_value - width
        requested_max = current_value + width
    else:
        requested_min = override.min if override.min is not None else safety_min
        requested_max = override.max if override.max is not None else safety_max

    clamped_min = max(requested_min, safety_min)
    clamped_max = min(requested_max, safety_max)

    if clamped_min > clamped_max:
        # Requested interval is entirely outside safety: collapse to the nearest safety endpoint.
        collapse = safety_min if requested_max < safety_min else safety_max
        clamped_min = clamped_max = collapse

    snapped_min, snapped_max = _snap_to_grid(
        clamped_min, clamped_max, safety_min, safety_max, step_size
    )

    was_clamped = snapped_min != requested_min or snapped_max != requested_max
    return ResolvedBoundsModel(
        min=snapped_min,
        max=snapped_max,
        requested_min=requested_min,
        requested_max=requested_max,
        was_clamped=was_clamped,
    )


def _snap_to_grid(
    lo: float,
    hi: float,
    safety_min: float,
    safety_max: float,
    step_size: float | None,
) -> tuple[float, float]:
    """Snap ``[lo, hi]`` to the step grid anchored at ``safety_min``.

    ``lo`` is rounded up and ``hi`` is rounded down so the result is a subset
    of the input. If no grid point lies in ``[lo, hi]``, both endpoints
    collapse to the grid point nearest the midpoint (clamped to safety).

    ``step_size`` must be ``None`` or a finite positive number; the public
    ``resolve_bounds`` validates this before delegating here.
    """
    if step_size is None:
        return lo, hi

    n_lo = math.ceil((lo - safety_min) / step_size - _STEP_GRID_EPS)
    n_hi = math.floor((hi - safety_min) / step_size + _STEP_GRID_EPS)
    snapped_lo = safety_min + n_lo * step_size
    snapped_hi = safety_min + n_hi * step_size

    if snapped_lo > snapped_hi:
        mid = (lo + hi) / 2.0
        n_mid = round((mid - safety_min) / step_size)
        grid_point = safety_min + n_mid * step_size
        grid_point = min(max(grid_point, safety_min), safety_max)
        return grid_point, grid_point

    return snapped_lo, snapped_hi
