"""What a faulty monitor reads and what a miscalibrated supply delivers.

Two formulas, ported from pySC (Python Simulated Commissioning,
https://github.com/kparasch/pySC) as straight arithmetic, so their sign and
roll conventions are pySC's:

* :func:`bpm_read` -- pySC's ``BPMSystem.capture_orbit`` chain for one
  monitor: roll, offset, calibration error, polarity, noise, gain.
* :func:`magnet_cal` -- pySC's ``LinearConv.transform``: the value a supply
  delivers against the one commanded.

**The faults live on the deck elements.** A monitor element carries
``readout_<field>_x`` and ``readout_<field>_y`` for ``offset``, ``gain``,
``noise`` and ``polarity``, and one ``readout_roll``. A magnet setpoint's
supply calibration sits on the first element it binds, under attributes named
for the setpoint (:func:`supply_attribute`), so setpoints that share an
element each keep their own. An attribute the element does not carry is its
identity value, so an unseeded deck reads and writes exactly.

**The reading is split from the solve.** A monitor variable returns the solved
truth; :func:`readout` turns truth -- plus whatever beam motion the caller has
added to it -- into what each monitor reports. It runs once per monitor
element on its (x, y) pair, because a roll mixes the two planes, and its noise
is a pure function of the reading's address and the time it is read at.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

#: What each readout attribute reads on an element that does not carry it.
READOUT_IDENTITY: dict[str, float] = {
    "offset": 0.0,
    "gain": 1.0,
    "polarity": 1.0,
    "noise": 0.0,
}

#: The supply calibration fields a magnet setpoint carries, each at its
#: identity: a supply that delivers exactly what it was commanded.
SUPPLY_IDENTITY: dict[str, float] = {"cal_factor": 1.0, "cal_offset": 0.0}

_SUPPLY_PREFIX = "supply_"

#: The subkey that makes a reading's noise stream its own.
NOISE_SUBKEY = b":readout_noise"

_AXES = ("x", "y")


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
    normal_x: float,
    normal_y: float,
) -> tuple[float, float]:
    """One monitor's reading of a true (x, y) position.

    Roll-mix the true position, subtract the offset, apply the calibration
    error and the polarity, add noise, then apply the gain -- in that order,
    so the noise is added before the gain multiply, as pySC does.

    Every length -- the true position, the offsets, the noise widths and the
    reading -- is in the unit the monitor publishes. The formula is
    scale-free, so one unit throughout is the whole requirement.

    Args:
        x: True horizontal position at the monitor.
        y: True vertical position at the monitor.
        offset_x: Horizontal offset.
        offset_y: Vertical offset.
        gain_x: Horizontal gain, multiplied in last.
        gain_y: Vertical gain, multiplied in last.
        polarity_x: Horizontal polarity, +1.0 or -1.0.
        polarity_y: Vertical polarity, +1.0 or -1.0.
        roll: Roll about the beam axis, in radians. A positive roll rotates
            the true position counterclockwise before it is read
            (``[[cos, -sin], [sin, cos]]``), so a roll of pi/2 reads a purely
            horizontal position as purely vertical.
        cal_x: Horizontal calibration error, applied as ``1 + cal_x``.
        cal_y: Vertical calibration error, applied as ``1 + cal_y``.
        noise_x: Standard deviation of the horizontal noise.
        noise_y: Standard deviation of the vertical noise.
        normal_x: The standard normal draw the horizontal noise scales.
        normal_y: The standard normal draw the vertical noise scales.

    Returns:
        ``(reading_x, reading_y)`` in the unit the monitor publishes.
    """
    rotated_x = math.cos(roll) * x - math.sin(roll) * y
    rotated_y = math.sin(roll) * x + math.cos(roll) * y

    reading_x = (rotated_x - offset_x) * (1.0 + cal_x) * polarity_x + noise_x * normal_x
    reading_y = (rotated_y - offset_y) * (1.0 + cal_y) * polarity_y + noise_y * normal_y

    reading_x *= gain_x
    reading_y *= gain_y

    return float(reading_x), float(reading_y)


def magnet_cal(setpoint: float, *, factor: float = 1.0, offset: float = 0.0) -> float:
    """The value a supply delivers when ``setpoint`` is commanded.

    Args:
        setpoint: The commanded value, in the hardware unit.
        factor: Multiplicative calibration error; ``-1.0`` is a polarity flip,
            ``1.3`` a 30% error.
        offset: Additive calibration error, in the unit of ``setpoint``.

    Returns:
        ``setpoint * factor + offset``.
    """
    return setpoint * factor + offset


def supply_attribute(setpoint: str, field: str) -> str:
    """The element attribute that holds one field of a setpoint's supply calibration.

    No pyAT pass method reads it, so a calibration on the element never moves
    the orbit by itself.

    Args:
        setpoint: The setpoint's address.
        field: ``cal_factor`` or ``cal_offset``.

    Returns:
        ``supply_<field>[<setpoint>]``.
    """
    return f"{_SUPPLY_PREFIX}{field}[{setpoint}]"


def supply_calibration(element: Any, setpoint: str) -> dict[str, float]:
    """:func:`magnet_cal`'s ``factor`` and ``offset`` for one setpoint, as ``element`` holds them.

    Args:
        element: The deck element the setpoint's calibration sits on.
        setpoint: The setpoint's address.

    Returns:
        The two keywords, each at identity where the element carries none.
    """
    return {
        keyword: float(getattr(element, supply_attribute(setpoint, field), SUPPLY_IDENTITY[field]))
        for keyword, field in (("factor", "cal_factor"), ("offset", "cal_offset"))
    }


def _monitors(model: Any) -> dict[str, dict[str, str]]:
    """Each monitor element's readings, axis -> address, from the model's variables."""
    monitors: dict[str, dict[str, str]] = {}
    for address, variable in model.supported_variables.items():
        axis = getattr(variable, "axis", None)
        element = getattr(variable, "element_name", None)
        if not getattr(variable, "read_only", False) or axis not in _AXES or element is None:
            continue
        monitors.setdefault(str(element), {})[axis] = address
    return monitors


def _readout_faults(element: Any) -> dict[str, float]:
    """:func:`bpm_read`'s fault keywords as ``element`` holds them, identity where absent."""
    faults = {
        f"{field}_{axis}": float(getattr(element, f"readout_{field}_{axis}", identity))
        for field, identity in READOUT_IDENTITY.items()
        for axis in _AXES
    }
    faults["roll"] = float(getattr(element, "readout_roll", 0.0))
    return faults


def _normal(address: str | None, width: float, t_ms: int | float) -> float:
    """The standard normal draw of ``address``'s noise at ``t_ms``.

    0 where there is no address or no noise to scale, which leaves the
    reading exact.
    """
    if address is None or width == 0.0:
        return 0.0
    import numpy as np

    from osprey_connectors.simulation.series import channel_key_bytes, keyed_normals

    key = channel_key_bytes(address) + NOISE_SUBKEY
    return float(keyed_normals(key, np.asarray([t_ms]))[0])


def readout(model: Any, values: Mapping[str, float], t_ms: int | float) -> dict[str, float]:
    """What each monitor reports for the positions in ``values``.

    Every monitor element with a reading in ``values`` is read once, on its
    (x, y) pair: :func:`bpm_read` with the element's readout attributes, no
    calibration error, and noise drawn per address at ``t_ms``. A monitor with
    one reading only reads its other plane as exactly on axis.

    Args:
        model: The model the readings belong to: its ``supported_variables``
            say which addresses are a monitor's (a read-only variable with an
            ``element_name`` and an ``axis``), and its ``simulator`` holds the
            elements the faults are read from.
        values: Positions by address, in the unit each monitor publishes --
            the solved truth plus any beam motion.
        t_ms: The time of the reading, in epoch milliseconds; the noise is a
            pure function of it.

    Returns:
        One reading per address in ``values``; an address that is no monitor's
        passes through unchanged.

    Raises:
        ValueError: ``values`` holds one reading of a monitor that has two;
            a roll mixes the planes, so the other cannot be guessed.
    """
    read = dict(values)
    for element_name, axes in sorted(_monitors(model).items()):
        missing = sorted(address for address in axes.values() if address not in values)
        if len(missing) == len(axes):
            continue
        if missing:
            raise ValueError(
                f"monitor {element_name!r} is read on {sorted(axes.values())}; "
                f"values hold no reading of {missing}"
            )
        x_address = axes.get("x")
        y_address = axes.get("y")
        faults = _readout_faults(model.simulator.element(element_name))
        reading_x, reading_y = bpm_read(
            0.0 if x_address is None else float(values[x_address]),
            0.0 if y_address is None else float(values[y_address]),
            **faults,
            cal_x=0.0,
            cal_y=0.0,
            normal_x=_normal(x_address, faults["noise_x"], t_ms),
            normal_y=_normal(y_address, faults["noise_y"], t_ms),
        )
        if x_address is not None:
            read[x_address] = reading_x
        if y_address is not None:
            read[y_address] = reading_y
    return read


__all__ = [
    "READOUT_IDENTITY",
    "SUPPLY_IDENTITY",
    "bpm_read",
    "magnet_cal",
    "readout",
    "supply_attribute",
    "supply_calibration",
]
