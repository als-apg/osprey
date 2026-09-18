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

Both formulas are per device and stateless: the caller holds the per-device
parameters and keys them by the monitor the bindings document names. A monitor
has two spellings there -- the element it sits at (the deck's own `FamName`,
which is what the serving bridge and a facility's device database use) and the
address each of its planes is published on (what an operator seeding
`VA_BPM_ERRORS` reads off the control system) -- and `resolve_bpm_errors` is
the one place the second becomes the first. It refuses a spelling the document
knows neither way, so a typo'd device is a boot refusal rather than a machine
that serves unperturbed while looking configured.

Nothing here parses either spelling: an address is an opaque token the document
supplied, and no family, subfield or unit is read out of it. Units are the
monitor's own throughout (see `bpm_read`), and a seeded value passes through
resolution unchanged -- it is bounded where it is parsed, never here and never
by clamping.

`apply_misalignment` (pySC's `sc_tools.update_transformation`, restricted to
the dx/dy/roll degrees of freedom) is geometry on an AT element rather than a
readout model, so it lives in :mod:`lume_pyat.utils` and is re-exported here
alongside the two error formulas that stayed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lume_pyat.utils import apply_misalignment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

__all__ = [
    "BpmErrorSeedError",
    "apply_misalignment",
    "bpm_read",
    "magnet_cal",
    "resolve_bpm_errors",
]


class BpmErrorSeedError(ValueError):
    """Raised when seeded readout errors name a monitor the document has not.

    Either the device is not one the bindings publish a reading for, or two
    spellings of one device seed the same field. Both are configuration a
    caller can only refuse: the first perturbs nothing, and the second has no
    answer that is not a guess.
    """


def resolve_bpm_errors(
    seeded: Mapping[str, Mapping[str, float]],
    monitors: Mapping[str, str],
) -> dict[str, dict[str, float]]:
    """Key seeded readout errors by the element their monitor sits at.

    The lookup the `VA_BPM_ERRORS` grammar needs and the serving bridge does
    not do: its own state is keyed by element, because a reading is a pair --
    `bpm_read` mixes the two planes through the monitor's roll -- while the
    grammar's `DEV` token is whatever the person seeding the fault knows the
    device by. Both spellings the document carries are accepted, and the two
    addresses of one monitor therefore seed one error model between them.

    Args:
        seeded: Device token -> the `bpm_read` fields seeded on it, as
            `entrypoint._parse_bpm_errors` returns them (already bounded).
            A token is either a published monitor address or the element name
            a monitor binding states; the address is looked up first.
        monitors: Published monitor address -> the element that monitor sits
            at, one entry per monitor binding of the served document.

    Returns:
        Element name -> the seeded fields, merged across every token that
        named that element. A partial override of `bpm_read`'s keyword
        arguments, so a field nobody seeded is absent rather than restated at
        identity, and a fresh mapping the caller may keep.

    Raises:
        BpmErrorSeedError: a token is neither a published monitor address nor
            an element a monitor sits at, or two tokens seed one field on one
            monitor.
    """
    elements = frozenset(monitors.values())
    resolved: dict[str, dict[str, float]] = {}
    claimed_by: dict[tuple[str, str], str] = {}

    for token, fields in seeded.items():
        element = monitors.get(token)
        if element is None:
            if token not in elements:
                raise BpmErrorSeedError(
                    f"the seeded readout errors name {token!r}, which is neither an "
                    f"address the served bindings publish a monitor reading on nor an "
                    f"element one sits at; the document binds {len(monitors)} monitor "
                    f"readings at {len(elements)} elements, and a device is named by "
                    f"one of those two spellings exactly"
                )
            element = token

        target = resolved.setdefault(element, {})
        for field, value in fields.items():
            claimant = claimed_by.get((element, field))
            if claimant is not None:
                raise BpmErrorSeedError(
                    f"{claimant!r} and {token!r} are both the monitor at element "
                    f"{element!r} and both seed {field!r}; one device's readout error "
                    f"is one value, and which of the two was meant could only be guessed"
                )
            claimed_by[element, field] = token
            target[field] = float(value)

    return resolved


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

    **Units are the monitor's, not the lattice's.** Every length here -- the
    true position, the offsets, the noise widths and the reading returned --
    is in the unit the monitor's own binding publishes: the caller maps the
    solved orbit through that binding's ``monitor_inverse`` first (which is
    where a facility's metre-to-millimetre step lives), so the offsets a
    device database states in millimetres are passed in millimetres. The
    formula is scale-free, so one unit throughout is the whole requirement.

    Args:
        x: True horizontal closed-orbit position at the monitor, in the unit
            that monitor publishes.
        y: True vertical closed-orbit position at the monitor, in the same unit.
        offset_x: Monitor horizontal offset, in the same unit.
        offset_y: Monitor vertical offset, in the same unit.
        gain_x: Horizontal gain correction (multiplicative, applied last).
        gain_y: Vertical gain correction (multiplicative, applied last).
        polarity_x: Horizontal polarity, +1.0 or -1.0.
        polarity_y: Vertical polarity, +1.0 or -1.0.
        roll: Monitor roll about the beam axis, in radians. Positive roll rotates
            the true (x, y) position counterclockwise before it is read out
            (pySC's `_rotation_matrix`: `[[cos, -sin], [sin, cos]]`), so e.g.
            roll = pi/2 reads a purely horizontal true position as purely
            vertical.
        cal_x: Horizontal calibration error (fractional, applied as `1 + cal_x`).
        cal_y: Vertical calibration error (fractional, applied as `1 + cal_y`).
        noise_x: Standard deviation of horizontal readout noise, in the
            monitor's own unit.
        noise_y: Standard deviation of vertical readout noise, in the same unit.
        rng: Seeded `numpy.random.Generator` noise is drawn from.

    Returns:
        (reading_x, reading_y) in the unit the monitor publishes.
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
