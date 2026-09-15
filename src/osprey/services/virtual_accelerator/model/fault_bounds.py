"""Bounds every fault seed is checked against before anything is built from it.

A fault seed -- a BPM reading error, a magnet calibration error -- is
operator input, typed into an environment variable or a scenario file. Each
magnitude is bounded here: generous against plausible commissioning-error
magnitudes, so a real fault always passes, yet tight enough to reject a
fat-fingered or unit-confused entry (millimeters typed where meters were
meant) before it reaches the physics.

Every bound is inclusive at both ends. The module is pure constants and
imports nothing, so every place that validates a fault seed checks this one
table, and the boot-time env parser does so without reaching the lattice,
``lume_pyat``, ``at`` or ``lume``.
"""

from __future__ import annotations

MAX_BPM_OFFSET_M = 1e-2
MIN_BPM_GAIN = 0.1
MAX_BPM_GAIN = 10.0
MAX_BPM_ROLL_RAD = 0.1
MAX_BPM_NOISE_M = 1e-2
# A factor of -1 is a polarity flip; a magnitude beyond 5x is never a real
# calibration error.
MAX_CORR_GAIN_FACTOR = 5.0
MAX_MAGNET_CAL_OFFSET_A = 10.0

# BPM error field -> (min, max). The polarity fields must additionally land
# exactly on a bound: a polarity is a sign, never a scale.
BPM_ERROR_FIELD_BOUNDS: dict[str, tuple[float, float]] = {
    "offset_x": (-MAX_BPM_OFFSET_M, MAX_BPM_OFFSET_M),
    "offset_y": (-MAX_BPM_OFFSET_M, MAX_BPM_OFFSET_M),
    "gain_x": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "gain_y": (MIN_BPM_GAIN, MAX_BPM_GAIN),
    "polarity_x": (-1.0, 1.0),
    "polarity_y": (-1.0, 1.0),
    "roll": (-MAX_BPM_ROLL_RAD, MAX_BPM_ROLL_RAD),
    "noise_x": (0.0, MAX_BPM_NOISE_M),
    "noise_y": (0.0, MAX_BPM_NOISE_M),
}
BPM_POLARITY_FIELDS = frozenset({"polarity_x", "polarity_y"})

# Magnet calibration field -> (min, max): the factor that scales, and the
# offset in amps that shifts, the current a magnet delivers against the one
# commanded. The factor shares the corrector-gain bound, so a seed that parses
# at boot is a value the model accepts.
MAGNET_CAL_BOUNDS: dict[str, tuple[float, float]] = {
    "cal_factor": (-MAX_CORR_GAIN_FACTOR, MAX_CORR_GAIN_FACTOR),
    "cal_offset": (-MAX_MAGNET_CAL_OFFSET_A, MAX_MAGNET_CAL_OFFSET_A),
}

__all__ = [
    "BPM_ERROR_FIELD_BOUNDS",
    "BPM_POLARITY_FIELDS",
    "MAGNET_CAL_BOUNDS",
    "MAX_BPM_GAIN",
    "MAX_BPM_NOISE_M",
    "MAX_BPM_OFFSET_M",
    "MAX_BPM_ROLL_RAD",
    "MAX_CORR_GAIN_FACTOR",
    "MAX_MAGNET_CAL_OFFSET_A",
    "MIN_BPM_GAIN",
]
