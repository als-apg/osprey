"""Pure time-series synthesis primitives for the simulation engine.

These functions are the stateless computational core behind
:meth:`SimulationEngine.synthesize_series`: they turn a baseline series plus a
scenario's archiver event scripts (step/ramp/spike, positioned by window
fraction, wall-clock offset, or daily time-of-day) into a concrete value series.
They hold no engine state — the engine supplies the channel values and events
and these functions do the math — which keeps the synthesis logic unit-testable
in isolation (see ``tests/simulation/test_series_primitives.py``; the
engine-driven behavior is covered in ``tests/simulation/test_series.py``).
"""

import hashlib
from collections.abc import Sequence
from datetime import datetime, timedelta
from datetime import time as dtime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from osprey.simulation.expressions import ExpressionError
from osprey.utils.logger import get_logger

logger = get_logger("simulation_series")

# splitmix64 mixing constants: an odd increment (the golden-ratio word) and the
# two finalizer multipliers. The finalizer is a bijection on uint64, so distinct
# counters can never collide onto the same word for a given key.
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)
_SHIFT_11 = np.uint64(11)
_ONE = np.uint64(1)
# 53 significand bits scaled into the unit interval.
_TWO_POW_M53 = 2.0**-53
_TWO_PI = 2.0 * np.pi

# Texture: a stack of sinusoids log-spaced downward from the declared slowest
# period. The per-channel weight jitter is applied in log space and is narrower
# than half the decay step, so the slowest component dominates for every
# channel rather than only on average.
_WANDER_SUBKEY = b":wander"
_WANDER_COMPONENTS = 5
_WANDER_OCTAVE_SPAN = 3.5
_WANDER_DECAY_PER_OCTAVE = 0.5
_WANDER_JITTER_OCTAVES = 0.15


def pv_key_bytes(pv_name: str) -> bytes:
    """Per-channel key for the counter-based draw primitives.

    The key is the SHA-256 digest of the UTF-8 channel name. The builtin
    ``hash()`` is deliberately not used: it is salted per process, so draws
    keyed on it would differ between runs of the same scenario.

    Callers derive independent streams for the same channel by appending a
    subkey (e.g. ``pv_key_bytes(pv) + b":noise_abs"``).

    Args:
        pv_name: Channel name.

    Returns:
        A 32-byte digest suitable as the ``pv_key`` argument of
        :func:`keyed_normals`.
    """
    return hashlib.sha256(pv_name.encode("utf-8")).digest()


def _key_words(pv_key: bytes) -> tuple[np.uint64, np.uint64]:
    """Two uint64 lane keys derived from an arbitrary-length key."""
    digest = hashlib.sha256(pv_key).digest()
    return (
        np.uint64(int.from_bytes(digest[0:8], "big")),
        np.uint64(int.from_bytes(digest[8:16], "big")),
    )


def _mix64(z: "np.ndarray") -> "np.ndarray":
    """splitmix64 avalanche finalizer over a uint64 array (wrapping arithmetic)."""
    z = (z ^ (z >> _SHIFT_30)) * _MIX_A
    z = (z ^ (z >> _SHIFT_27)) * _MIX_B
    return np.asarray(z ^ (z >> _SHIFT_31))


def _as_counter_array(counters: "np.ndarray") -> "np.ndarray":
    """Normalize counters to uint64, rounding floats to the nearest integer.

    Epoch-millisecond counters usually arrive as float64 (``epoch_s * 1000``);
    signed integers (sample-index fallback, pre-epoch timestamps) wrap into
    uint64 modularly, which keys deterministically like any other value.
    """
    array = np.asarray(counters)
    if array.dtype.kind == "f":
        array = np.rint(array).astype(np.int64)
    if array.dtype != np.uint64:
        array = array.astype(np.uint64)
    return array


def _keyed_words(pv_key: bytes, counters_ms: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """The two uint64 words each sample consumes, as a pair of arrays.

    Exactly two words per sample, addressed by that sample's own counter — no
    stream position is involved, which is what makes the draws pointwise
    reproducible across differently-shaped windows.
    """
    counters = _as_counter_array(counters_ms)
    k0, k1 = _key_words(pv_key)
    base = counters * _GAMMA
    return _mix64(base ^ k0), _mix64(base ^ k1)


def keyed_normals(pv_key: bytes, counters_ms: "np.ndarray") -> "np.ndarray":
    """Standard normal draws addressed by ``(channel key, counter)``.

    Each sample's draw is a pure function of its own counter, so two windows
    that share a timestamp get the same value, in any order, in any batch size,
    from any process. This is the pinned construction for all synthesis noise:

    1. a uint64 avalanche mix of the counter against two key-derived lane words
       (exactly two words per sample, fixed consumption by construction);
    2. an **open**-interval mapping of the log argument,
       ``u1 = ((w >> 11) | 1) * 2**-53``, so ``u1 > 0`` always and ``log(u1)``
       can never be ``-inf`` — a frozen calendar timestamp that produced ``-inf``
       would fail permanently rather than transiently;
    3. the Box-Muller cosine branch.

    The ziggurat rejection sampler behind numpy's batch normal draws consumes a
    *variable* number of words per sample, so a batch drawn from a window-start
    counter does not equal the corresponding per-sample draws — it is forbidden
    on this path, and ``tests/simulation/test_series_primitives.py`` pins the
    equivalence against a per-sample reference implementation.

    Args:
        pv_key: Channel key, typically :func:`pv_key_bytes` output optionally
            suffixed with a subkey to get an independent stream.
        counters_ms: Per-sample counters, conventionally epoch milliseconds.
            Floats are rounded to the nearest integer; any shape is accepted.

    Returns:
        A float64 array of standard normal draws with the shape of
        ``counters_ms``.
    """
    w0, w1 = _keyed_words(pv_key, counters_ms)
    u1 = ((w0 >> _SHIFT_11) | _ONE).astype(np.float64) * _TWO_POW_M53
    u2 = (w1 >> _SHIFT_11).astype(np.float64) * _TWO_POW_M53
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(_TWO_PI * u2)


def _unit_interval(words: "np.ndarray") -> "np.ndarray":
    """Map uint64 words onto float64 in the half-open interval ``[0, 1)``."""
    return np.asarray((words >> _SHIFT_11).astype(np.float64) * _TWO_POW_M53)


def _wander_components(
    pv_key: bytes, period_s: float
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Periods, phases and normalized weights of one channel's texture stack.

    Periods are log-spaced downward from ``period_s`` (the slowest) across
    :data:`_WANDER_OCTAVE_SPAN` octaves. Phases and weight jitter come from the
    channel key, so the declared parameters can stay family-uniform while every
    channel still gets its own apparent offset and motion.

    Args:
        pv_key: Channel key, as for :func:`keyed_normals`.
        period_s: Slowest period in seconds.

    Returns:
        ``(periods, phases, weights)``, each of length
        :data:`_WANDER_COMPONENTS`, with ``weights`` summing to 1.
    """
    index = np.arange(_WANDER_COMPONENTS, dtype=np.uint64)
    phase_words, weight_words = _keyed_words(pv_key + _WANDER_SUBKEY, index)
    octaves = np.arange(_WANDER_COMPONENTS, dtype=np.float64) * (
        _WANDER_OCTAVE_SPAN / (_WANDER_COMPONENTS - 1)
    )
    periods = period_s * 2.0**-octaves
    phases = _unit_interval(phase_words) * _TWO_PI
    jitter = (2.0 * _unit_interval(weight_words) - 1.0) * _WANDER_JITTER_OCTAVES
    weights = 2.0 ** (-_WANDER_DECAY_PER_OCTAVE * octaves + jitter)
    return periods, phases, np.asarray(weights / weights.sum())


def wander(pv_key: bytes, t_abs_s: "np.ndarray", amplitude: float, period_s: float) -> "np.ndarray":
    """Band-limited quasi-random baseline texture, evaluated on absolute time.

    A seeded sum of :data:`_WANDER_COMPONENTS` sinusoids whose periods are
    log-spaced downward from ``period_s``. One declaration therefore covers two
    timescales: with ``period_s`` of many hours the slowest terms read as a
    distinct quasi-static offset inside any short query window, while the faster
    octaves supply visible motion — which is what lets a whole channel family
    share one set of declared parameters and still look individual.

    The envelope is bounded structurally rather than by clipping: the weights
    are positive and sum to 1, so ``|wander| <= amplitude`` follows from the
    triangle inequality and the output stays linear in ``amplitude``.

    Being a pure function of absolute epoch seconds is what makes overlapping
    archiver windows agree exactly on their shared timestamps.

    Args:
        pv_key: Channel key, as for :func:`keyed_normals`.
        t_abs_s: Absolute epoch seconds; any shape.
        amplitude: Envelope bound, in the channel's declared units.
        period_s: Slowest period in seconds; must be positive.

    Returns:
        A float64 array of texture offsets with the shape of ``t_abs_s``.

    Raises:
        ValueError: If ``period_s`` is not positive. Machine-file parsing
            already rejects this; the guard keeps a direct caller from turning
            a division by zero into silent ``NaN`` in the series.
    """
    if period_s <= 0.0:
        raise ValueError(f"wander period_s must be > 0, got {period_s!r}")
    times = np.asarray(t_abs_s, dtype=np.float64)
    periods, phases, weights = _wander_components(pv_key, period_s)
    total = np.zeros(times.shape, dtype=np.float64)
    for period, phase, weight in zip(periods, phases, weights, strict=True):
        total += weight * np.sin(_TWO_PI * times / period + phase)
    return amplitude * total


def clamp(value: float, min_value: float | None, max_value: float | None) -> float:
    """Clamp a scalar into ``[min_value, max_value]`` (either bound optional)."""
    if min_value is not None and value < min_value:
        return min_value
    if max_value is not None and value > max_value:
        return max_value
    return value


def ref_value(ref_series: dict[str, "np.ndarray | list[str]"], name: str, index: int) -> float:
    """Resolver for pointwise expression evaluation over referenced series."""
    value = ref_series[name][index]
    if isinstance(value, str):
        raise ExpressionError(
            f"Channel {name!r} holds string values and cannot be used in an expression"
        )
    return float(value)


def epoch_seconds_array(timestamps: "Sequence[Any]") -> "np.ndarray | None":
    """Convert timestamps to epoch seconds, or None when not convertible."""
    values: list[float] = []
    for ts in timestamps:
        if hasattr(ts, "timestamp"):
            values.append(float(ts.timestamp()))
        elif isinstance(ts, (int, float)) and not isinstance(ts, bool):
            values.append(float(ts))
        else:
            return None
    return np.asarray(values, dtype=np.float64)


def event_positions(
    event: dict[str, Any],
    t_frac: "np.ndarray",
    t_abs: "np.ndarray | None",
    anchor: float,
    tz: ZoneInfo | None = None,
) -> "list[tuple[np.ndarray, float]]":
    """Coordinate axis and event position(s) for one event.

    Fraction-positioned (``at``) and offset-anchored (``at_offset``) events
    yield one position, on the normalized window axis and the epoch-seconds
    axis respectively. Daily ``at_time`` events yield one epoch-seconds
    position per calendar date whose time-of-day occurrence falls inside the
    window, placed in the facility timezone (``tz``). Returns an empty list
    when an anchored or time-of-day event cannot be placed because the
    timestamps were not convertible to epoch seconds.
    """
    if "at_offset" in event:
        if t_abs is None:
            logger.debug(
                "Skipping offset-anchored event: timestamps not convertible to epoch seconds"
            )
            return []
        return [(t_abs, anchor + float(event["at_offset"]))]
    if "at_time" in event:
        if t_abs is None:
            logger.debug("Skipping time-of-day event: timestamps not convertible to epoch seconds")
            return []
        return [(t_abs, at) for at in daily_occurrences(str(event["at_time"]), t_abs, tz)]
    return [(t_frac, float(event["at"]))]


def daily_occurrences(at_time: str, t_abs: "np.ndarray", tz: ZoneInfo | None = None) -> list[float]:
    """Epoch-second positions of a daily time-of-day within ``[t_abs[0], t_abs[-1]]``.

    Walks calendar dates from the window start to its end (in timezone ``tz``)
    and keeps every occurrence of ``at_time`` that falls inside the window.
    Placing the time-of-day in an explicit zone — rather than the deploy host's
    local zone — makes the result independent of the box ``$TZ``. ``tz`` defaults
    to UTC; the engine resolves and threads the facility zone (so this module
    stays free of config), so production callers always supply it explicitly.
    Assumes ascending timestamps (the same contract ``apply_events`` relies
    on via ``np.searchsorted``).
    """
    zone = tz if tz is not None else ZoneInfo("UTC")
    tod = dtime.fromisoformat(at_time)
    start = datetime.fromtimestamp(float(t_abs[0]), zone)
    end = datetime.fromtimestamp(float(t_abs[-1]), zone)
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    positions: list[float] = []
    while cursor <= end:
        candidate = cursor.replace(
            hour=tod.hour, minute=tod.minute, second=tod.second, microsecond=tod.microsecond
        )
        if start <= candidate <= end:
            positions.append(candidate.timestamp())
        cursor += timedelta(days=1)
    return positions


def string_series(
    baseline: str,
    events: list[dict[str, Any]],
    n: int,
    t_abs: "np.ndarray | None",
    anchor: float,
    tz: ZoneInfo | None = None,
) -> list[str]:
    """Constant string series; only 'step' events are meaningful for strings."""
    t_frac = np.linspace(0.0, 1.0, n)
    series = [baseline] * n
    for event in events:
        if event["shape"] != "step":
            continue
        for x, at in event_positions(event, t_frac, t_abs, anchor, tz):
            for i in range(n):
                if x[i] >= at:
                    series[i] = str(event["to"])
    return series


def apply_events(
    series: "np.ndarray",
    events: list[dict[str, Any]],
    n: int,
    t_abs: "np.ndarray | None",
    anchor: float,
    tz: ZoneInfo | None = None,
) -> "np.ndarray":
    """Apply step/ramp/spike events in order (window-fraction or wall-clock)."""
    if not events:
        return series
    t_frac = np.linspace(0.0, 1.0, n)
    series = series.copy()
    for event in events:
        shape = event["shape"]
        offset_anchored = "at_offset" in event
        for x, at in event_positions(event, t_frac, t_abs, anchor, tz):
            if shape == "step":
                series[x >= at] = float(event["to"])
            elif shape == "ramp":
                until = (
                    anchor + float(event["until_offset"])
                    if offset_anchored
                    else float(event["until"])
                )
                to = float(event["to"])
                if until <= at:
                    series[x >= at] = to
                    continue
                idx = int(np.searchsorted(x, at))
                start = float(series[min(idx, n - 1)])
                mask = (x >= at) & (x <= until)
                series[mask] = start + (to - start) * (x[mask] - at) / (until - at)
                series[x > until] = to
            else:  # spike (gaussian bump; width as window fraction, or seconds when anchored)
                amplitude = float(event["amplitude"])
                width = float(event["width"])
                series = series + amplitude * np.exp(-((x - at) ** 2) / (2.0 * width**2))
    return series
