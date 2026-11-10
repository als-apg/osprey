"""The pulsed-septum current trace: one pure function of address, time and drive.

A pulsed supply fires about once a second; each firing is captured as a fixed
window of 500 samples. The pulse has a fast rise, a flat top
that droops slightly, and a slower exponential fall. The flat top rings: a few
damped sinusoids whose amplitudes and phases are drawn fresh for every shot,
together with a small per-shot jitter of the rise edge. On top of that sit a
slow amplitude drift and a slow timing drift (periods of minutes) and a small
sample noise, all keyed by the channel address and the epoch millisecond, the
same way the texture engine keys its motion. That is what lets the live server
and the seeded archive produce the identical trace for the same instant.

The trace is linear in the driver level (the supply setpoint): every term,
noise included, is a multiple of it, so the peak ratio of two traces at the
same instant equals their driver ratio. A disabled supply reads flat zero.

The shape constants live here, not in seeds. :data:`SHAPE_VERSION` is a digest
of them; the archive seed fingerprint consumes it, so changing the shape
re-seeds the stored history.
"""

from __future__ import annotations

import hashlib

import numpy as np

from osprey_connectors.simulation import series

__all__ = [
    "SHAPE_VERSION",
    "trace",
]

#: Samples per captured trace.
_TRACE_POINTS = 500
#: Sample index (at zero timing drift) where the pulse starts to rise.
_RISE_START_SAMPLE = 60.0
#: Length of the half-cosine rise, in samples.
_RISE_SAMPLES = 12.0
#: Length of the flat top, in samples.
_FLAT_TOP_SAMPLES = 200.0
#: Fractional droop across the flat top (the top ends at ``1 - _FLAT_TOP_DROOP``).
_FLAT_TOP_DROOP = 0.02
#: Time constant of the exponential fall, in samples.
_FALL_TAU_SAMPLES = 40.0
#: Bound on the amplitude drift, as a fraction of the driver level. The drift
#: is a band-limited wander bounded by this value, so peak-to-peak it stays
#: within twice this fraction.
_AMPLITUDE_DRIFT_FRACTION = 0.012
#: Bound on how far timing drift moves the rise edge, in samples, peak to peak.
_MAX_TIMING_DRIFT_SAMPLES = 8
#: Slowest period of the amplitude and timing drift, in seconds.
_DRIFT_PERIOD_S = 300.0
#: Standard deviation of the sample noise, as a fraction of the driver level.
_NOISE_FRACTION = 0.002
#: Number of damped sinusoids in the flat-top ringing.
_RINGING_COMPONENTS = 3
#: Period of the slowest ringing component, in samples.
_RINGING_LONGEST_PERIOD_SAMPLES = 64.0
#: Ratio between the periods of successive ringing components.
_RINGING_PERIOD_RATIO = 0.6
#: Bound on each ringing component's amplitude, as a fraction of the flat-top
#: level; each shot draws every amplitude uniformly below it.
_RINGING_AMPLITUDE_FRACTION = 0.02
#: Time constant of the ringing's decay from the rise edge, in samples.
_RINGING_DAMPING_SAMPLES = 80.0
#: Bound on the per-shot jitter of the rise edge, in samples either way.
_EDGE_JITTER_SAMPLES = 1.5

#: The constants :data:`SHAPE_VERSION` digests, in a fixed order. Each is
#: digested under its name without the leading underscore, the spelling it had
#: while it was public, so making them private left the digest unchanged.
_SHAPE_CONSTANT_NAMES: tuple[str, ...] = (
    "_TRACE_POINTS",
    "_RISE_START_SAMPLE",
    "_RISE_SAMPLES",
    "_FLAT_TOP_SAMPLES",
    "_FLAT_TOP_DROOP",
    "_FALL_TAU_SAMPLES",
    "_AMPLITUDE_DRIFT_FRACTION",
    "_MAX_TIMING_DRIFT_SAMPLES",
    "_DRIFT_PERIOD_S",
    "_NOISE_FRACTION",
    "_RINGING_COMPONENTS",
    "_RINGING_LONGEST_PERIOD_SAMPLES",
    "_RINGING_PERIOD_RATIO",
    "_RINGING_AMPLITUDE_FRACTION",
    "_RINGING_DAMPING_SAMPLES",
    "_EDGE_JITTER_SAMPLES",
)

_AMPLITUDE_SUBKEY = b":pulse_amplitude"
_TIMING_SUBKEY = b":pulse_timing"
_NOISE_SUBKEY = b":pulse_noise"
_RINGING_SUBKEY = b":pulse_ringing"
_JITTER_SUBKEY = b":pulse_jitter"
_MS_PER_S = 1000.0


def _shape_version() -> str:
    """Digest of the current shape constants (see :data:`_SHAPE_CONSTANT_NAMES`)."""
    module = globals()
    text = ";".join(f"{name.lstrip('_')}={module[name]!r}" for name in _SHAPE_CONSTANT_NAMES)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


#: Digest of the shape constants; the archive seed fingerprint consumes it.
SHAPE_VERSION = _shape_version()


def _unit_shape(x: np.ndarray) -> np.ndarray:
    """The noise-free pulse of unit height at sample offset ``x`` from its start."""
    flat_end = _RISE_SAMPLES + _FLAT_TOP_SAMPLES
    rise = 0.5 * (1.0 - np.cos(np.pi * np.clip(x, 0.0, _RISE_SAMPLES) / _RISE_SAMPLES))
    top = 1.0 - _FLAT_TOP_DROOP * (np.clip(x, _RISE_SAMPLES, flat_end) - _RISE_SAMPLES) / (
        _FLAT_TOP_SAMPLES
    )
    fall = (1.0 - _FLAT_TOP_DROOP) * np.exp(-np.clip(x - flat_end, 0.0, None) / _FALL_TAU_SAMPLES)
    return np.where(
        x < 0.0, 0.0, np.where(x < _RISE_SAMPLES, rise, np.where(x < flat_end, top, fall))
    )


def _edge_jitter(key: bytes, t_ms: float) -> float:
    """The shot's extra shift of the rise edge, within :data:`_EDGE_JITTER_SAMPLES` either way."""
    words, _ = series._keyed_words(key + _JITTER_SUBKEY, np.array([t_ms], dtype=np.int64))
    return _EDGE_JITTER_SAMPLES * (2.0 * float(series._unit_interval(words)[0]) - 1.0)


def _ringing(key: bytes, t_ms: float, x: np.ndarray) -> np.ndarray:
    """The shot's flat-top ringing, relative to the pulse, at sample offset ``x``.

    Each component's amplitude and phase are drawn from the shot's own counters,
    so a shot is reproducible and consecutive shots differ in their ringing. The sum is
    damped from the rise edge and windowed by the unit pulse, so it is zero before
    the rise and fades into the fall; its magnitude stays below
    ``_RINGING_COMPONENTS * _RINGING_AMPLITUDE_FRACTION``.
    """
    counters = np.int64(t_ms) * _RINGING_COMPONENTS + np.arange(_RINGING_COMPONENTS, dtype=np.int64)
    amplitude_words, phase_words = series._keyed_words(key + _RINGING_SUBKEY, counters)
    amplitudes = _RINGING_AMPLITUDE_FRACTION * series._unit_interval(amplitude_words)
    phases = 2.0 * np.pi * series._unit_interval(phase_words)
    periods = _RINGING_LONGEST_PERIOD_SAMPLES * _RINGING_PERIOD_RATIO ** np.arange(
        _RINGING_COMPONENTS
    )
    since_rise = np.clip(x, 0.0, None)
    oscillation = np.sin(2.0 * np.pi * since_rise[:, None] / periods + phases) @ amplitudes
    return np.asarray(_unit_shape(x) * np.exp(-since_rise / _RINGING_DAMPING_SAMPLES) * oscillation)


def _clean(key: bytes, t_ms: float) -> np.ndarray:
    """The noise-free trace of unit driver level for the shot at epoch millisecond ``t_ms``."""
    t_abs = np.asarray(t_ms / _MS_PER_S, dtype=np.float64)
    gain = 1.0 + float(
        series.wander(key + _AMPLITUDE_SUBKEY, t_abs, _AMPLITUDE_DRIFT_FRACTION, _DRIFT_PERIOD_S)
    )
    shift = float(
        series.wander(key + _TIMING_SUBKEY, t_abs, _MAX_TIMING_DRIFT_SAMPLES / 2.0, _DRIFT_PERIOD_S)
    )
    x = np.arange(_TRACE_POINTS, dtype=np.float64) - (
        _RISE_START_SAMPLE + shift + _edge_jitter(key, t_ms)
    )
    return np.asarray(gain * (_unit_shape(x) + _ringing(key, t_ms, x)))


def trace(address: str, t_s: float, driver: float, enable: bool) -> np.ndarray:
    """The captured pulse of ``address`` at epoch second ``t_s``.

    Pure: the result depends only on the arguments. Time is taken at whole
    epoch milliseconds, matching how the texture engine keys its motion.

    Args:
        address: Channel address; keys the drift, ringing and noise streams.
        t_s: Absolute epoch seconds.
        driver: Driver level (the supply setpoint); the trace scales linearly with it.
        enable: Supply on-state; ``False`` gives a flat zero trace.

    Returns:
        A float64 array of shape ``(_TRACE_POINTS,)``.
    """
    if not enable:
        return np.zeros(_TRACE_POINTS, dtype=np.float64)
    t_ms = float(np.rint(float(t_s) * _MS_PER_S))
    key = series.channel_key_bytes(address)
    counters = np.int64(t_ms) * _TRACE_POINTS + np.arange(_TRACE_POINTS, dtype=np.int64)
    noise = _NOISE_FRACTION * series.keyed_normals(key + _NOISE_SUBKEY, counters)
    return float(driver) * (_clean(key, t_ms) + noise)
