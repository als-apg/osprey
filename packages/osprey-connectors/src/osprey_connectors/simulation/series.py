"""Pure time-series synthesis primitives for the simulated archive and texture.

These functions turn a baseline series plus a scenario's archiver events
(step/ramp/spike/burst/loss, positioned by window fraction, wall-clock offset, or daily
time-of-day) into a concrete value series, and supply the keyed draws and the
band-limited texture the texture engine layers on a served value, and say
which stretch of stored history one occurrence of an event changes
(:func:`occurrence_span`), for the archive rewrite. They hold no state: the caller supplies the channel values and events and these functions
do the math (see ``tests/simulation/test_series_primitives.py``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from datetime import time as dtime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from osprey_connectors.logger import get_logger
from osprey_connectors.relative_time import resolve_relative_timestamp
from osprey_connectors.simulation.records import (
    AnchorOffset,
    BurstEvent,
    CalendarDay,
    DailyTime,
    LossEvent,
    RampEvent,
    SpikeEvent,
    StepEvent,
    WindowFraction,
)
from osprey_connectors.simulation.values import _real

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey_connectors.simulation.records import ArchiverEvent, TopoffRecord

__all__ = [
    "channel_key_bytes",
    "clamp",
    "keyed_normals",
    "occurrence_span",
    "occurrences",
]

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

#: How many of a ``burst``'s slowest time constants count as inside its window:
#: ``exp(-10)`` leaves under 0.005% of the burst outside it.
_BURST_WINDOW_TAUS = 10.0
#: How many sigmas either side of a spike count as "inside" its window. A
#: Gaussian is never exactly zero, so the window has to be cut somewhere; four
#: sigmas leaves under 0.01% of the bump outside it, which is far below the
#: channel's own noise and therefore invisible in the restored history.
_SPIKE_WINDOW_SIGMAS = 4.0
#: Injector shots per second while top-off refills a ``loss``.
_REFILL_RATE_HZ = 1.0
#: Charge one refill shot adds, as a fraction of the top-off ``ceiling``
#: (0.75 mA on a 500 mA ring).
_REFILL_SHOT_FRACTION = 0.0015


def channel_key_bytes(channel: str) -> bytes:
    """Per-channel key for the counter-based draw primitives.

    The key is the SHA-256 digest of the UTF-8 channel name. The builtin
    ``hash()`` is deliberately not used: it is salted per process, so draws
    keyed on it would differ between runs of the same scenario.

    Callers derive independent streams for the same channel by appending a
    subkey (e.g. ``channel_key_bytes(channel) + b":noise_abs"``).

    Args:
        channel: Channel name.

    Returns:
        A 32-byte digest suitable as the ``channel_key`` argument of
        :func:`keyed_normals`.
    """
    return hashlib.sha256(channel.encode("utf-8")).digest()


def _driver_key_bytes(driver: str) -> bytes:
    """Key for a scenario's shared latent driver, addressed by the driver's name.

    A driver is the common cause behind several channels, so its draws must not
    depend on which channel asks for them: every channel coupled to the driver
    evaluates :func:`_wander` under this one key and sees the identical value at
    the same instant. The ``driver:`` namespace keeps a driver that happens to
    share a channel's name from replaying that channel's own texture stream.

    Args:
        driver: Driver name, as declared in a scenario's ``drivers`` block.

    Returns:
        A 32-byte digest usable wherever a ``channel_key`` is expected.
    """
    return hashlib.sha256(b"driver:" + driver.encode("utf-8")).digest()


def _key_words(channel_key: bytes) -> tuple[np.uint64, np.uint64]:
    """Two uint64 lane keys derived from an arbitrary-length key."""
    digest = hashlib.sha256(channel_key).digest()
    return (
        np.uint64(int.from_bytes(digest[0:8], "big")),
        np.uint64(int.from_bytes(digest[8:16], "big")),
    )


def _mix64(z: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """splitmix64 avalanche finalizer over a uint64 array (wrapping arithmetic)."""
    z = (z ^ (z >> _SHIFT_30)) * _MIX_A
    z = (z ^ (z >> _SHIFT_27)) * _MIX_B
    return np.asarray(z ^ (z >> _SHIFT_31))


def _as_counter_array(counters: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Normalize counters to uint64, rounding floats to the nearest integer.

    Epoch-millisecond counters usually arrive as float64 (``times_s * 1000``);
    signed integers (sample-index fallback, pre-epoch timestamps) wrap into
    uint64 modularly, which keys deterministically like any other value.
    """
    array = np.asarray(counters)
    if array.dtype.kind == "f":
        array = np.rint(array).astype(np.int64)
    if array.dtype != np.uint64:
        array = array.astype(np.uint64)
    return array


def _keyed_words(
    channel_key: bytes, counters_ms: npt.ArrayLike
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """The two uint64 words each sample consumes, as a pair of arrays.

    Exactly two words per sample, addressed by that sample's own counter — no
    stream position is involved, which is what makes the draws pointwise
    reproducible across differently-shaped windows.
    """
    counters = _as_counter_array(counters_ms)
    k0, k1 = _key_words(channel_key)
    base = counters * _GAMMA
    return _mix64(base ^ k0), _mix64(base ^ k1)


def keyed_normals(channel_key: bytes, counters_ms: npt.ArrayLike) -> npt.NDArray[np.float64]:
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
        channel_key: Channel key, typically :func:`channel_key_bytes` output optionally
            suffixed with a subkey to get an independent stream.
        counters_ms: Per-sample counters, conventionally epoch milliseconds.
            Floats are rounded to the nearest integer; any shape is accepted.

    Returns:
        A float64 array of standard normal draws with the shape of
        ``counters_ms``.
    """
    w0, w1 = _keyed_words(channel_key, counters_ms)
    u1 = ((w0 >> _SHIFT_11) | _ONE).astype(np.float64) * _TWO_POW_M53
    u2 = (w1 >> _SHIFT_11).astype(np.float64) * _TWO_POW_M53
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(_TWO_PI * u2)


def _unit_interval(words: npt.NDArray[np.uint64]) -> npt.NDArray[np.float64]:
    """Map uint64 words onto float64 in the half-open interval ``[0, 1)``."""
    return np.asarray((words >> _SHIFT_11).astype(np.float64) * _TWO_POW_M53)


def _wander_components(
    channel_key: bytes, period_s: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Periods, phases and normalized weights of one channel's texture stack.

    Periods are log-spaced downward from ``period_s`` (the slowest) across
    :data:`_WANDER_OCTAVE_SPAN` octaves. Phases and weight jitter come from the
    channel key, so the declared parameters can stay family-uniform while every
    channel still gets its own apparent offset and motion.

    Args:
        channel_key: Channel key, as for :func:`keyed_normals`.
        period_s: Slowest period in seconds.

    Returns:
        ``(periods, phases, weights)``, each of length
        :data:`_WANDER_COMPONENTS`, with ``weights`` summing to 1.
    """
    index = np.arange(_WANDER_COMPONENTS, dtype=np.uint64)
    phase_words, weight_words = _keyed_words(channel_key + _WANDER_SUBKEY, index)
    octaves = np.arange(_WANDER_COMPONENTS, dtype=np.float64) * (
        _WANDER_OCTAVE_SPAN / (_WANDER_COMPONENTS - 1)
    )
    periods = period_s * 2.0**-octaves
    phases = _unit_interval(phase_words) * _TWO_PI
    jitter = (2.0 * _unit_interval(weight_words) - 1.0) * _WANDER_JITTER_OCTAVES
    weights = 2.0 ** (-_WANDER_DECAY_PER_OCTAVE * octaves + jitter)
    return periods, phases, np.asarray(weights / weights.sum())


def _wander(
    channel_key: bytes, times_s: npt.ArrayLike, amplitude: float, period_s: float
) -> npt.NDArray[np.float64]:
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
        channel_key: Channel key, as for :func:`keyed_normals`.
        times_s: Absolute epoch seconds; any shape.
        amplitude: Envelope bound, in the channel's declared units.
        period_s: Slowest period in seconds; must be positive.

    Returns:
        A float64 array of texture offsets with the shape of ``times_s``.

    Raises:
        ValueError: If ``period_s`` is not positive; the guard keeps a caller
            from turning a division by zero into silent ``NaN`` in the series.
    """
    if period_s <= 0.0:
        raise ValueError(f"wander period_s must be > 0, got {period_s!r}")
    times = np.asarray(times_s, dtype=np.float64)
    periods, phases, weights = _wander_components(channel_key, period_s)
    total = np.zeros(times.shape, dtype=np.float64)
    for period, phase, weight in zip(periods, phases, weights, strict=True):
        total += weight * np.sin(_TWO_PI * times / period + phase)
    return amplitude * total


def clamp(value: float, min_value: float | None, max_value: float | None) -> float:
    """Clamp a scalar into ``[min_value, max_value]`` (either bound optional).

    The result is a float whichever side is hit, because ``clamp`` applies to
    float channels only.
    """
    if min_value is not None and value < min_value:
        return float(min_value)
    if max_value is not None and value > max_value:
        return float(max_value)
    return value


def _epoch_seconds_array(timestamps: Iterable[object]) -> npt.NDArray[np.float64] | None:
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


def _instant(placement: AnchorOffset | CalendarDay, *, anchor_s: float, tz: ZoneInfo) -> float:
    """Epoch seconds of a placement relative to the activation anchor.

    An :class:`AnchorOffset` is seconds from the anchor. A :class:`CalendarDay`
    is the logbook's own ``{days_ago, time}``: the calendar day ``days_ago``
    days before the anchor's date, at ``time``, both read in ``tz`` -- the
    same rule a scenario's logbook entries resolve by, so an event and the
    entry narrating it name one instant.
    """
    match placement:
        case AnchorOffset(offset_s=offset):
            return anchor_s + float(offset)
        case CalendarDay(when=when):
            return resolve_relative_timestamp(
                when, datetime.fromtimestamp(anchor_s, tz)
            ).timestamp()


def _step_end(event: StepEvent, *, anchor_s: float, tz: ZoneInfo) -> float | None:
    """Where a bounded ``step`` ends, on the same axis as its start; ``None`` when unbounded.

    ``until`` is a window fraction, ``until_offset`` seconds from the anchor and
    ``until_when`` a ``{days_ago, time}`` placed exactly as ``at_when`` is.
    """
    match event.end:
        case None:
            return None
        case WindowFraction(at=until):
            return float(until)
        case AnchorOffset() | CalendarDay() as end:
            return _instant(end, anchor_s=anchor_s, tz=tz)


def _ramp_end(event: RampEvent, *, anchor_s: float) -> float:
    """Where a ``ramp`` reaches ``to``: a window fraction, or epoch seconds when anchored."""
    match event.end:
        case WindowFraction(at=until):
            return float(until)
        case AnchorOffset(offset_s=offset):
            return anchor_s + float(offset)


def _event_positions(
    event: ArchiverEvent,
    t_frac: npt.NDArray[np.float64],
    times_s: npt.NDArray[np.float64] | None,
    *,
    anchor_s: float,
    tz: ZoneInfo,
) -> list[tuple[npt.NDArray[np.float64], float]]:
    """Coordinate axis and event position(s) for one event.

    Fraction-positioned (``at``) and anchored (``at_offset``, ``at_when``)
    events yield one position, on the normalized window axis and the
    epoch-seconds axis respectively. Daily ``at_time`` events yield one epoch-seconds
    position per calendar date whose time-of-day occurrence falls inside the
    window, placed in the facility timezone (``tz``). Returns an empty list
    when an anchored or time-of-day event cannot be placed because the
    timestamps were not convertible to epoch seconds.
    """
    match event.placement:
        case WindowFraction(at=at):
            return [(t_frac, float(at))]
        case DailyTime(time=time_of_day):
            if times_s is None:
                logger.debug(
                    "Skipping time-of-day event: timestamps not convertible to epoch seconds"
                )
                return []
            return [(times_s, at) for at in _daily_occurrences(time_of_day, times_s, tz=tz)]
        case AnchorOffset() | CalendarDay() as placement:
            if times_s is None:
                logger.debug("Skipping anchored event: timestamps not convertible to epoch seconds")
                return []
            return [(times_s, _instant(placement, anchor_s=anchor_s, tz=tz))]


def _daily_occurrences(
    time_of_day: dtime, times_s: npt.NDArray[np.float64], *, tz: ZoneInfo
) -> list[float]:
    """Epoch-second positions of a daily time-of-day within ``[times_s[0], times_s[-1]]``.

    Walks calendar dates from the window start to its end (in timezone ``tz``)
    and keeps every occurrence of ``time_of_day`` that falls inside the window.
    Placing the time-of-day in an explicit zone — rather than the deploy host's
    local zone — makes the result independent of the box ``$TZ``. Callers
    resolve and pass the facility zone, so this module stays free of config.
    Assumes ascending timestamps (the same contract ``_apply_events`` relies
    on via ``np.searchsorted``).
    """
    start = datetime.fromtimestamp(float(times_s[0]), tz)
    end = datetime.fromtimestamp(float(times_s[-1]), tz)
    cursor = start.replace(hour=0, minute=0, second=0, microsecond=0)
    positions: list[float] = []
    while cursor <= end:
        candidate = cursor.replace(
            hour=time_of_day.hour,
            minute=time_of_day.minute,
            second=time_of_day.second,
            microsecond=time_of_day.microsecond,
        )
        if start <= candidate <= end:
            positions.append(candidate.timestamp())
        cursor += timedelta(days=1)
    return positions


def occurrences(
    event: ArchiverEvent, *, start_s: float, end_s: float, anchor_s: float, tz: ZoneInfo
) -> list[float]:
    """Every instant an event placed in stored history fires at, in epoch seconds.

    An anchored event (``at_offset``, ``at_when``) fires once, wherever that
    falls; a daily ``at_time`` event fires on every calendar date of ``tz``
    whose occurrence lies inside ``[start_s, end_s]``.

    Args:
        event: An event placed by ``at_offset``, ``at_when`` or ``at_time``.
        start_s: The oldest instant a daily occurrence may fall at.
        end_s: The latest instant a daily occurrence may fall at.
        anchor_s: The activation anchor T0, in epoch seconds.
        tz: The facility timezone anchored days and daily times are placed in.

    Returns:
        The instants, ascending.

    Raises:
        ValueError: The event is placed by window fraction (``at``), which
            names a position in a reader's window and no instant.
    """
    match event.placement:
        case WindowFraction():
            raise ValueError("an event placed by window fraction (`at`) fires at no instant")
        case DailyTime(time=time_of_day):
            return _daily_occurrences(
                time_of_day, np.asarray([start_s, end_s], dtype=np.float64), tz=tz
            )
        case AnchorOffset() | CalendarDay() as placement:
            return [_instant(placement, anchor_s=anchor_s, tz=tz)]


def occurrence_span(
    event: ArchiverEvent,
    at_s: float,
    *,
    anchor_s: float,
    tz: ZoneInfo,
    topoff: TopoffRecord | None,
) -> tuple[float, float]:
    """The stretch one occurrence of an event at ``at_s`` changes, in epoch seconds.

    A spike reaches four sigmas either side of its instant. A burst starts at
    its instant and reaches ten of its slowest time constants after it. A loss
    starts at its instant and lasts until top-off has surely refilled it, or to
    the anchor on a channel no top-off refills. A step runs from its instant to
    its end, or to the anchor; a ramp to the anchor.

    Args:
        event: The event.
        at_s: The instant of this occurrence, in epoch seconds.
        anchor_s: The activation anchor T0, in epoch seconds.
        tz: The facility timezone a step's ``until_when`` is placed in.
        topoff: The channel's ``topoff`` seed record, which alone refills a loss.

    Returns:
        ``(start, end)`` in epoch seconds.
    """
    match event:
        case SpikeEvent(width=width):
            half = float(width) * _SPIKE_WINDOW_SIGMAS
            return at_s - half, at_s + half
        case BurstEvent(decay_s=decay, tail_decay_s=tail):
            slowest = max(float(decay), 0.0 if tail is None else float(tail))
            return at_s, at_s + _BURST_WINDOW_TAUS * slowest
        case LossEvent():
            bound = _loss_refill_bound(event, topoff)
            return at_s, anchor_s if bound is None else at_s + bound
        case StepEvent():
            until = _step_end(event, anchor_s=anchor_s, tz=tz)
            return at_s, anchor_s if until is None else until
        case RampEvent():
            return at_s, anchor_s


def _burst_profile(event: BurstEvent, dt: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A ``burst``'s excursion at ``dt`` seconds after its instant.

    Zero before the instant, ``amplitude`` at it, then a fast exponential
    pump-down with ``decay_s`` plus, when ``tail_share`` and ``tail_decay_s``
    are stated, a slower exponential carrying that share of the amplitude::

        amplitude * ((1 - share) * exp(-dt / decay_s) + share * exp(-dt / tail_decay_s))

    Args:
        event: A ``burst`` event.
        dt: Seconds since the event instant; negative is before it.

    Returns:
        The excursion, same shape as ``dt``.
    """
    amplitude = event.amplitude
    decay = event.decay_s
    share = 0.0 if event.tail_share is None else event.tail_share
    tail = decay if event.tail_decay_s is None else event.tail_decay_s
    after = np.maximum(dt, 0.0)
    shape = (1.0 - share) * np.exp(-after / decay) + share * np.exp(-after / tail)
    return np.where(dt >= 0.0, amplitude * shape, 0.0)


def _loss_deficit(
    event: LossEvent,
    at: float,
    x: npt.NDArray[np.float64],
    topoff: TopoffRecord | None,
) -> npt.NDArray[np.float64]:
    """How far a ``loss`` holds a channel below its level at each instant of ``x``.

    The channel drops by ``amount`` in one step at ``at`` and nothing brings it
    back but top-off: from the first top-off refill instant after ``at`` (an
    epoch-aligned multiple of the seed's ``period_s``) the injector adds one shot
    of ``_REFILL_SHOT_FRACTION * ceiling`` every ``1 / _REFILL_RATE_HZ`` seconds
    until the amount is replaced. A channel without a ``topoff`` seed never
    recovers.

    Args:
        event: A ``loss`` event.
        at: Its instant, in epoch seconds.
        x: Epoch seconds, ascending.
        topoff: The channel's ``topoff`` seed record, or None.

    Returns:
        The deficit, ``amount`` right after ``at`` and 0 before it and once
        refilled.
    """
    deficit = np.where(x >= at, event.amount, 0.0)
    if topoff is None:
        return deficit
    period = topoff.period_s
    shot = _REFILL_SHOT_FRACTION * abs(float(topoff.ceiling))
    refill = (np.floor(at / period) + 1.0) * period
    shots = np.where(x >= refill, np.floor((x - refill) * _REFILL_RATE_HZ) + 1.0, 0.0)
    return np.maximum(deficit - shot * shots, 0.0)


def _loss_refill_bound(event: LossEvent, topoff: TopoffRecord | None) -> float | None:
    """Seconds after its instant by which a ``loss`` is surely refilled; None if never.

    One top-off period to the first refill, wherever in the period the loss
    falls, plus the shots the amount takes.
    """
    if topoff is None:
        return None
    shot = _REFILL_SHOT_FRACTION * abs(float(topoff.ceiling))
    shots = np.ceil(event.amount / shot) if shot > 0 else 0.0
    return float(topoff.period_s) + float(shots) / _REFILL_RATE_HZ


def _apply_events(
    series: npt.NDArray[np.float64],
    events: Sequence[ArchiverEvent],
    n: int,
    times_s: npt.NDArray[np.float64] | None,
    *,
    anchor_s: float,
    tz: ZoneInfo,
    topoff: TopoffRecord | None = None,
) -> npt.NDArray[np.float64]:
    """Apply step/ramp/spike/burst/loss events in order (window-fraction or wall-clock).

    A step with an end (see :func:`_step_end`) holds ``to`` over ``[start, end)``
    and leaves later samples on the baseline; without one it holds to the end.
    A ``burst`` (:func:`_burst_profile`) and a ``loss`` (:func:`_loss_deficit`)
    start at their instant and never move a sample before it; ``topoff`` is the
    channel's ``topoff`` seed record, which alone refills a loss. ``anchor_s`` is
    the activation anchor in epoch seconds and ``tz`` the facility zone anchored
    and daily events are placed in.
    """
    if not events:
        return series
    t_frac = np.linspace(0.0, 1.0, n)
    series = series.copy()
    for event in events:
        # A ``to`` is a number on the float channels whose series these events move.
        for x, at in _event_positions(event, t_frac, times_s, anchor_s=anchor_s, tz=tz):
            match event:
                case StepEvent():
                    end = _step_end(event, anchor_s=anchor_s, tz=tz)
                    inside = x >= at if end is None else (x >= at) & (x < end)
                    series[inside] = _real(event.to)
                case RampEvent():
                    until = _ramp_end(event, anchor_s=anchor_s)
                    to = _real(event.to)
                    if until <= at:
                        series[x >= at] = to
                        continue
                    idx = int(np.searchsorted(x, at))
                    start = float(series[min(idx, n - 1)])
                    mask = (x >= at) & (x <= until)
                    series[mask] = start + (to - start) * (x[mask] - at) / (until - at)
                    series[x > until] = to
                case BurstEvent():
                    series = series + _burst_profile(event, x - at)
                case LossEvent():
                    series = series - _loss_deficit(event, at, x, topoff)
                case SpikeEvent(amplitude=amplitude, width=width):
                    # A gaussian bump; ``width`` is a window fraction, or seconds when anchored.
                    series = series + amplitude * np.exp(-((x - at) ** 2) / (2.0 * width**2))
    return series
