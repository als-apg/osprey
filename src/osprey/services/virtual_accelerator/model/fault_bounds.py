"""Bounds every fault seed is checked against before anything is built from it.

A fault seed -- a monitor reading error, a magnet calibration error -- is
operator input, typed into an environment variable or a scenario file. It is
checked here, and only here, so the boot-time parser and the model that holds
the fault as a variable weigh a value against one table: a seed that boots is
a value the model accepts.

**What is bounded is what a device can BE.** A gain outside its window, a roll
beyond its angle, a calibration factor beyond its multiple: each names no
instrument the model could stand in for, whoever typed it. A polarity is a
sign rather than a range and must land exactly on one of its two bounds.

**What is not bounded is what was asked for.** A seeded displacement, a noise
amplitude and a calibration offset are magnitudes an operator deliberately
asked the simulator for, in whatever unit the facility publishes that device
in -- so a number that would be absurd in one unit is ordinary in another, and
there is no size at which one of them stops meaning what it says. What still
refuses them is well-formedness: a value that is not a finite number names no
magnitude, and a negative noise amplitude names no distribution, since it is a
standard deviation.

Every bound is inclusive at both ends, and a bound may be half-open where only
one end means anything. The module is pure constants over the standard library
alone, so a caller validates a seed without reaching the lattice,
``lume_pyat``, ``at`` or ``lume``.
"""

from __future__ import annotations

import math

MIN_BPM_GAIN = 0.1
MAX_BPM_GAIN = 10.0
MAX_BPM_ROLL_RAD = 0.1
# A factor of -1 is a polarity flip; a magnitude beyond 5x is never a real
# calibration error.
MAX_CORR_GAIN_FACTOR = 5.0

#: Every reading-error field a monitor carries, in the order a rendered field
#: list spells them. A field named in a seed and missing here is unknown.
BPM_ERROR_FIELDS: tuple[str, ...] = (
    "offset_x",
    "offset_y",
    "gain_x",
    "gain_y",
    "polarity_x",
    "polarity_y",
    "roll",
    "noise_x",
    "noise_y",
)

#: Reading-error field -> (min, max). Only the fields describing what a monitor
#: *is* appear; a field absent from this map is bounded by well-formedness
#: alone (see the module docstring).
BPM_ERROR_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "gain_x": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "gain_y": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "roll": (-MAX_BPM_ROLL_RAD, MAX_BPM_ROLL_RAD),
}

#: A polarity is a direction: it lands exactly on one of these two values.
BPM_POLARITY_OPTIONS: tuple[float, float] = (-1.0, 1.0)
BPM_POLARITY_FIELDS = frozenset({"polarity_x", "polarity_y"})

#: A noise amplitude is a standard deviation: non-negative, unbounded above.
#: Half-open rather than absent, because a negative width describes no
#: distribution at all -- the draw it produces raises rather than returning a
#: number, and a model holding one refuses every reading it is asked for
#: afterwards.
BPM_NOISE_FIELDS = frozenset({"noise_x", "noise_y"})
BPM_NOISE_BOUNDS: tuple[float, float] = (0.0, math.inf)

#: What each reading-error field reads on a monitor nobody seeded: one that
#: reports the true orbit position exactly.
BPM_ERROR_IDENTITY: dict[str, float] = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
}

#: Magnet calibration field -> (min, max): the factor that scales the value a
#: magnet delivers against the one commanded. The offset that shifts it is a
#: magnitude in the facility's own hardware unit and carries no bound.
MAGNET_CAL_FIELDS: tuple[str, ...] = ("cal_factor", "cal_offset")
MAGNET_CAL_BOUNDS: dict[str, tuple[float, float]] = {
    "cal_factor": (-MAX_CORR_GAIN_FACTOR, MAX_CORR_GAIN_FACTOR),
}

#: What each calibration field reads on a magnet nobody seeded: one that
#: delivers exactly what it was commanded.
MAGNET_CAL_IDENTITY: dict[str, float] = {"cal_factor": 1.0, "cal_offset": 0.0}

#: The seed grammar's own field names -> the calibration field each seeds.
CORRECTOR_GAIN_FIELDS: dict[str, str] = {"factor": "cal_factor", "offset": "cal_offset"}

__all__ = [
    "BPM_ERROR_FIELDS",
    "BPM_ERROR_FIELD_BOUNDS",
    "BPM_ERROR_IDENTITY",
    "BPM_NOISE_BOUNDS",
    "BPM_NOISE_FIELDS",
    "BPM_POLARITY_FIELDS",
    "BPM_POLARITY_OPTIONS",
    "CORRECTOR_GAIN_FIELDS",
    "MAGNET_CAL_BOUNDS",
    "MAGNET_CAL_FIELDS",
    "MAGNET_CAL_IDENTITY",
    "MAX_BPM_GAIN",
    "MAX_BPM_ROLL_RAD",
    "MAX_CORR_GAIN_FACTOR",
    "MIN_BPM_GAIN",
]
