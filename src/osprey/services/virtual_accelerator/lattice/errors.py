"""Error-model formulas for the VA lattice, ported from pySC (Python Simulated
Commissioning, https://github.com/kparasch/pySC -- LBNL's `accelerator-commissioning`
toolkit): a BPM reading formula and a linear magnet calibration. Every formula
here is a straight arithmetic port of the corresponding pySC routine, not a
reimplementation from first principles, so its sign/roll conventions match
pySC's. Pure numpy, no new dependencies.

Provenance:
  - `bpm_read` ports pySC's `pySC.core.bpm_system.BPMSystem.capture_orbit`
    (the calibration/roll/noise/gain chain applied to a single BPM reading;
    the transmission/BBA/dead-BPM/reference-subtraction machinery around it
    is out of scope here).
  - `magnet_cal` ports pySC's `pySC.core.control.LinearConv.transform`.

`apply_misalignment` (pySC's `sc_tools.update_transformation`, restricted to
the dx/dy/roll degrees of freedom) is geometry on an AT element rather than a
readout model, so it lives in :mod:`lume_pyat.utils` and is re-exported here
alongside the two error formulas that stayed.
"""

from __future__ import annotations

import numpy as np
from lume_pyat.utils import apply_misalignment

__all__ = [
    "apply_misalignment",
    "bpm_read",
    "magnet_cal",
]


def bpm_read(
    x: float,
    y: float,
    *,
    offset_x: float,
    offset_y: float,
    gain_x: float,
    gain_y: float,
    polarity_x: float,
    polarity_y: float,
    roll: float,
    cal_x: float,
    cal_y: float,
    noise_x: float,
    noise_y: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Simulate one BPM reading from a true (x, y) closed-orbit position.

    Ports pySC's ``BPMSystem.capture_orbit`` per-BPM chain: roll-mix the true
    position, subtract the offset, apply calibration error and polarity, add
    noise, then apply gain -- in that order (noise is added *before* the gain
    multiply, matching pySC).

    Args:
        x: True horizontal closed-orbit position at the BPM, in meters.
        y: True vertical closed-orbit position at the BPM, in meters.
        offset_x: BPM horizontal offset, in meters.
        offset_y: BPM vertical offset, in meters.
        gain_x: Horizontal gain correction (multiplicative, applied last).
        gain_y: Vertical gain correction (multiplicative, applied last).
        polarity_x: Horizontal polarity, +1.0 or -1.0.
        polarity_y: Vertical polarity, +1.0 or -1.0.
        roll: BPM roll about the beam axis, in radians. Positive roll rotates
            the true (x, y) position counterclockwise before it is read out
            (pySC's `_rotation_matrix`: `[[cos, -sin], [sin, cos]]`), so e.g.
            roll = pi/2 reads a purely horizontal true position as purely
            vertical.
        cal_x: Horizontal calibration error (fractional, applied as `1 + cal_x`).
        cal_y: Vertical calibration error (fractional, applied as `1 + cal_y`).
        noise_x: Standard deviation of horizontal readout noise, in meters.
        noise_y: Standard deviation of vertical readout noise, in meters.
        rng: Seeded `numpy.random.Generator` noise is drawn from.

    Returns:
        (reading_x, reading_y) in meters.
    """
    rotated_x = np.cos(roll) * x - np.sin(roll) * y
    rotated_y = np.sin(roll) * x + np.cos(roll) * y

    drawn_noise_x = rng.normal(scale=noise_x)
    drawn_noise_y = rng.normal(scale=noise_y)

    reading_x = (rotated_x - offset_x) * (1.0 + cal_x) * polarity_x + drawn_noise_x
    reading_y = (rotated_y - offset_y) * (1.0 + cal_y) * polarity_y + drawn_noise_y

    reading_x *= gain_x
    reading_y *= gain_y

    return float(reading_x), float(reading_y)


def magnet_cal(setpoint: float, *, factor: float = 1.0, offset: float = 0.0) -> float:
    """Apply a linear magnet calibration, ported from pySC's `LinearConv.transform`.

    Args:
        setpoint: Commanded value (e.g. a corrector current in Amps).
        factor: Multiplicative calibration error. `factor = -1.0` is a
            polarity flip; `factor = 1.3` is a 30% calibration error.
        offset: Additive calibration error, in the same units as `setpoint`.

    Returns:
        `setpoint * factor + offset`.
    """
    return setpoint * factor + offset
