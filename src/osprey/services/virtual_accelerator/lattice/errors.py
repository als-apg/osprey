"""Error-model formulas for the VA lattice, ported from pySC (Python Simulated
Commissioning, https://github.com/kparasch/pySC -- LBNL's `accelerator-commissioning`
toolkit): a BPM reading formula, a magnet-misalignment transform, and a linear
magnet calibration. Every formula here is a straight arithmetic/geometric port
of the corresponding pySC routine, not a reimplementation from first
principles, so its sign/roll conventions match pySC's (and, by construction,
AT's -- pySC's `update_transformation` builds AT `T1`/`T2`/`R1`/`R2`
directly). Pure numpy; operates on `at` lattice elements by attribute access
without importing `at`. No new dependencies.

Provenance:
  - `bpm_read` ports pySC's `pySC.core.bpm_system.BPMSystem.capture_orbit`
    (the calibration/roll/noise/gain chain applied to a single BPM reading;
    the transmission/BBA/dead-BPM/reference-subtraction machinery around it
    is out of scope here).
  - `apply_misalignment` ports pySC's `pySC.utils.sc_tools.update_transformation`,
    restricted to the dx/dy/roll degrees of freedom (no dz/yaw/pitch).
  - `magnet_cal` ports pySC's `pySC.core.control.LinearConv.transform`.
"""

from __future__ import annotations

import numpy as np


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


def _rotation_matrix_3d(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """3D extrinsic Z-Y-X rotation (Roll, Yaw, Pitch), ported from pySC's
    `sc_tools.rotation`."""
    ax, ay, az = pitch, yaw, roll
    return np.array(
        [
            [np.cos(ay) * np.cos(az), -np.cos(ay) * np.sin(az), np.sin(ay)],
            [
                np.cos(az) * np.sin(ax) * np.sin(ay) + np.cos(ax) * np.sin(az),
                np.cos(ax) * np.cos(az) - np.sin(ax) * np.sin(ay) * np.sin(az),
                -np.cos(ay) * np.sin(ax),
            ],
            [
                -np.cos(ax) * np.cos(az) * np.sin(ay) + np.sin(ax) * np.sin(az),
                np.cos(az) * np.sin(ax) + np.cos(ax) * np.sin(ay) * np.sin(az),
                np.cos(ax) * np.cos(ay),
            ],
        ]
    )


def _translation_vector(
    ld: float, r3d: np.ndarray, xaxis_xyz: np.ndarray, yaxis_xyz: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Ported from pySC's `sc_tools._translation_vector`."""
    t_d0 = np.array([-np.dot(offsets, xaxis_xyz), 0.0, -np.dot(offsets, yaxis_xyz), 0.0, 0.0, 0.0])
    t_0 = np.array(
        [
            ld * r3d[2, 0] / r3d[2, 2],
            r3d[2, 0],
            ld * r3d[2, 1] / r3d[2, 2],
            r3d[2, 1],
            0.0,
            ld / r3d[2, 2],
        ]
    )
    return t_0 + t_d0


def _r_matrix(ld: float, r3d: np.ndarray) -> np.ndarray:
    """Ported from pySC's `sc_tools._r_matrix`."""
    return np.array(
        [
            [
                r3d[1, 1] / r3d[2, 2],
                ld * r3d[1, 1] / r3d[2, 2] ** 2,
                -r3d[0, 1] / r3d[2, 2],
                -ld * r3d[0, 1] / r3d[2, 2] ** 2,
                0.0,
                0.0,
            ],
            [0.0, r3d[0, 0], 0.0, r3d[1, 0], r3d[2, 0], 0.0],
            [
                -r3d[1, 0] / r3d[2, 2],
                -ld * r3d[1, 0] / r3d[2, 2] ** 2,
                r3d[0, 0] / r3d[2, 2],
                ld * r3d[0, 0] / r3d[2, 2] ** 2,
                0.0,
                0.0,
            ],
            [0.0, r3d[0, 1], 0.0, r3d[1, 1], r3d[2, 1], 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [
                -r3d[0, 2] / r3d[2, 2],
                -ld * r3d[0, 2] / r3d[2, 2] ** 2,
                -r3d[1, 2] / r3d[2, 2],
                -ld * r3d[1, 2] / r3d[2, 2] ** 2,
                0.0,
                1.0,
            ],
        ]
    )


def apply_misalignment(element, *, dx: float = 0.0, dy: float = 0.0, roll: float = 0.0) -> None:
    """Set an AT element's `T1`/`T2`/`R1`/`R2` for a dx/dy/roll misalignment.

    Ports pySC's `sc_tools.update_transformation`, restricted to the dx/dy/roll
    degrees of freedom (dz = yaw = pitch = 0.0 always) -- FR3's error model has
    no use for longitudinal shift or out-of-plane tilt. Bend-aware: for an
    element with a nonzero `BendingAngle`, the exit transform (`T2`/`R2`)
    accounts for the magnet's curved geometry via its `Length` and
    `BendingAngle`, matching pySC exactly for a straight element
    (`BendingAngle == 0`).

    ABSOLUTE, not additive: each call fully replaces any prior T1/T2/R1/R2
    misalignment on `element` -- it does not compose with a previous call.
    Pass the complete desired dx/dy/roll together in one call; calling this
    twice with, say, dx then dy separately does not accumulate both -- the
    second call overwrites the first's transform entirely.

    Args:
        element: An `at` lattice element (mutated in place).
        dx: Horizontal offset, in meters.
        dy: Vertical offset, in meters.
        roll: Roll about the beam axis, in radians. Positive roll matches
            `bpm_read`'s convention (see there) -- both are the same
            `sc_tools.rotation`-derived z-axis rotation.

    Raises:
        ZeroDivisionError: never for a valid AT element -- pySC's own formulas
            divide by `r3d[2, 2]`, which is 1.0 for any dx/dy/roll-only
            (dz = yaw = pitch = 0) misalignment.
    """
    mag_length = getattr(element, "Length", 0.0)
    mag_theta = getattr(element, "BendingAngle", 0.0)

    offsets = np.array([dx, dy, 0.0])
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    # Entrance transform.
    r_3d = _rotation_matrix_3d(0.0, 0.0, roll)
    ld = np.dot(np.dot(r_3d, z_axis), offsets)

    t_entrance = _translation_vector(ld, r_3d, np.dot(r_3d, x_axis), np.dot(r_3d, y_axis), offsets)
    element.R1 = _r_matrix(ld, r_3d)
    element.T1 = np.dot(np.linalg.inv(element.R1), t_entrance)

    # Exit transform -- bend-aware: undoes the entrance rotation in the frame
    # of the magnet's own curvature (RB), so a straight element (mag_theta ==
    # 0) reduces to the mirror image of the entrance transform.
    rx = r_3d
    rb = _rotation_matrix_3d(0.0, -mag_theta, 0.0)
    r_3d_exit = np.dot(rb.T, np.dot(rx.T, rb))
    op_p = np.array(
        [
            mag_length * (np.cos(mag_theta) - 1.0) / mag_theta if mag_theta else 0.0,
            0.0,
            mag_length * (np.sin(mag_theta) / mag_theta if mag_theta else 1.0),
        ]
    )
    op_p_prime = op_p - np.dot(rx, op_p) - offsets
    ld_exit = np.dot(np.dot(rb, z_axis), op_p_prime)

    element.T2 = _translation_vector(
        ld_exit, r_3d_exit, np.dot(rb, x_axis), np.dot(rb, y_axis), op_p_prime
    )
    element.R2 = _r_matrix(ld_exit, r_3d_exit)


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
