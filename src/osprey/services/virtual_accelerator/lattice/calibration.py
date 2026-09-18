"""Unit conversions between a device's hardware units and the lattice's.

A binding carries two curves and they are independent data: ``calibration``
maps the hardware value a control system writes onto the physics value the
lattice element takes, and ``monitor_inverse`` maps a physics reading back to
hardware units. Neither is derived from the other -- the control system samples
each along its own path, and the two agree only to the precision of those
samples -- so a readback is read off the exported inverse and a calibration is
never inverted here.

A curve is either a straight line or a sampled table. A table is piecewise
linear through its points and continues along its end segments beyond them, so
it is defined for every value a device can be asked for, including one outside
the range that was sampled. Its grid runs strictly one way but either way: a
curve with negative gain is sampled onto a falling grid.

The other conversion is the ring's. A calibration states a physics value at the
energy its lattice was built for, and a quantity that scales with beam rigidity
is worth :func:`energy_factor` of that at any other energy.

Everything is vectorised: an array of per-device values goes in and an array of
the same shape comes out, one curve applying to all of them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from osprey.services.virtual_accelerator.bindings import Calibration, Linear, Table

__all__ = [
    "brho",
    "energy_factor",
    "to_hardware",
    "to_physics",
]

#: Electron rest mass in GeV. The energy a lattice carries is kinetic, so the
#: rest mass is what turns it into a momentum.
REST_MASS_GEV: float = 0.51099906e-3

#: Tesla-metres of rigidity per GeV/c of momentum: 1e9 divided by the speed of
#: light in metres per second.
RIGIDITY_PER_MOMENTUM: float = 10.0 / 2.99792458


def to_physics(calibration: Calibration, hardware: ArrayLike) -> NDArray[np.float64]:
    """Convert hardware-unit values to the physics units the lattice takes."""
    return _evaluate(calibration, hardware)


def to_hardware(monitor_inverse: Calibration, physics: ArrayLike) -> NDArray[np.float64]:
    """Convert physics-unit readings to hardware units through the exported inverse.

    The inverse is the only path back: it is sampled data in its own right, not
    the calibration read the other way, and applying it to a value that came
    from the calibration returns the original only to the precision the two
    samplings share.
    """
    return _evaluate(monitor_inverse, physics)


def brho(energy_gev: ArrayLike) -> NDArray[np.float64]:
    """Return the beam rigidity in tesla-metres at ``energy_gev``.

    The energy is kinetic, so the rest mass enters the momentum twice and the
    massless form ``E / c`` is not a usable shortcut for it: at a few GeV that
    form is high by parts in ten thousand, and it misstates the ratio between
    two nearby energies by several parts in a million -- coarser than the
    agreement a rigidity-scaled family is recognised by.
    """
    total = np.asarray(energy_gev, dtype=float) + REST_MASS_GEV
    return RIGIDITY_PER_MOMENTUM * np.sqrt(total**2 - REST_MASS_GEV**2)


def energy_factor(energy_gev: ArrayLike, deck_energy_gev: ArrayLike) -> NDArray[np.float64]:
    """Return what a rigidity-scaled physics value is worth at ``energy_gev``.

    A calibration states its physics value at ``deck_energy_gev``, the energy
    the lattice was built for. The same hardware value bends a stiffer beam
    less, in the ratio of the two rigidities, and the factor is one wherever
    the ring is at the energy its calibrations were sampled at.
    """
    return brho(deck_energy_gev) / brho(energy_gev)


def _evaluate(curve: Calibration, values: ArrayLike) -> NDArray[np.float64]:
    """Apply one curve to an array of values, whichever shape the curve has."""
    points = np.asarray(values, dtype=float)
    if isinstance(curve, Linear):
        return curve.gain * points + curve.offset
    return _interpolate(curve, points)


def _interpolate(table: Table, points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Read a table at ``points``, continuing along its end segments beyond it."""
    grid = np.asarray(table.grid, dtype=float)
    values = np.asarray(table.values, dtype=float)
    if grid[-1] < grid[0]:
        grid = grid[::-1]
        values = values[::-1]

    low_slope = float(values[1] - values[0]) / float(grid[1] - grid[0])
    high_slope = float(values[-1] - values[-2]) / float(grid[-1] - grid[-2])
    read: NDArray[np.float64] = np.interp(np.clip(points, grid[0], grid[-1]), grid, values)
    below: NDArray[np.float64] = np.minimum(points - float(grid[0]), 0.0)
    above: NDArray[np.float64] = np.maximum(points - float(grid[-1]), 0.0)
    return read + low_slope * below + high_slope * above
