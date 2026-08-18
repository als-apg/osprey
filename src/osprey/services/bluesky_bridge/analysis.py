"""Peak statistics for a settled run: center, width, center of mass.

A run's most-asked question is not "what did every point read" but "where was
the peak, and how wide was it". The stock bluesky statistics callback answers
exactly that, and this module is the bridge's adapter to it: it decides whether
the run *has* a question that statistics can answer, hands the stock code the
pairs it can use, and converts the answer back to plain JSON numbers.

Whether the question exists at all comes from the plan's own declaration, never
from a column-name guess. A run has statistics when its plan declared exactly
one movable channel -- the independent variable -- and at least one readable
channel to fit against it. Everything else is *absent with a reason*, never an
error and never a fabricated number:

- ``run_in_progress`` -- the run is still producing rows. Statistics are
  computed once, when the run settles, because a peak fitted to half a run
  moves as the rest of it arrives.
- ``plan_identity_unavailable`` -- the run carries no usable plan stamp, or the
  stamped name has no current owner in the plan catalog, so nothing declares
  what this run's columns mean. Same word the figure route uses for the same
  situation.
- ``no_declared_movable`` / ``no_declared_readable`` -- the plan declares no
  channel of that role, or its recorded parameters supplied none.
- ``multiple_movable_channels`` -- the plan drove more than one channel, so the
  run has no single independent variable to fit against. This is deliberately
  the serial-sweep case (an orbit response measurement drives every corrector
  in turn); a peak fitted against one of several movables is a number about
  nothing.
- ``movable_column_missing`` -- the declared movable never reached the run's
  data under either spelling `plan_fields.resolve_column` knows.
- ``insufficient_points`` -- the run holds fewer than `MIN_POINTS` rows.
- ``statistics_unavailable`` -- the stock statistics module could not be
  imported, or computing raised. Logged, and reported as an absence like any
  other: this route must not turn a library problem into a failed read.

Per channel, the same discipline with its own words -- ``channel_column_missing``,
``channel_is_the_x_axis`` (a channel declared both movable and readable is its
own axis, not a peak in it), ``insufficient_points`` (fewer than `MIN_POINTS`
pairs survive the finite-number filter -- a channel that never reported, or one
that reads back as text), and ``statistics_unavailable``.

An analyzed channel's three statistics are each independently nullable, and a
null there is a truth about the readings rather than a skip: ``center`` and
``width`` need the channel to cross its own half-maximum, which a monotonic
drift or a flat channel never does. ``center_of_mass`` is null only when it
comes back non-finite (readings summing to zero).

The stock module is imported inside `analyze`, never at module scope, so
`app.py` -- which imports this one -- stays clean of ``bluesky`` in its import
closure (``app._BRIDGE_ONLY_MODULES``, pinned by ``test_app_import_clean.py``).
Same precedent as the lazy ``tiled`` import in the read path.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Sequence
from threading import Lock
from typing import Any

from .plan_fields import resolve_column

logger = logging.getLogger(__name__)

MIN_POINTS = 3
"""Fewest points a channel (or a run) needs before statistics are computed.

Two points have no interior: every statistic over them is a restatement of the
endpoints, and reporting a "center" between two readings would dress an
interpolation up as a measurement.
"""

REASON_RUN_IN_PROGRESS = "run_in_progress"
REASON_PLAN_IDENTITY_UNAVAILABLE = "plan_identity_unavailable"
REASON_NO_DECLARED_MOVABLE = "no_declared_movable"
REASON_NO_DECLARED_READABLE = "no_declared_readable"
REASON_MULTIPLE_MOVABLE_CHANNELS = "multiple_movable_channels"
REASON_MOVABLE_COLUMN_MISSING = "movable_column_missing"
REASON_INSUFFICIENT_POINTS = "insufficient_points"
REASON_STATISTICS_UNAVAILABLE = "statistics_unavailable"

REASON_CHANNEL_COLUMN_MISSING = "channel_column_missing"
REASON_CHANNEL_IS_THE_X_AXIS = "channel_is_the_x_axis"


def absent(reason: str) -> dict[str, Any]:
    """The payload for a run with no statistics, saying which reason applies.

    Same key set as an available result, so a consumer reads ``available``
    first and never has to tell "this run has no statistics" apart from "this
    response shape omits them".
    """
    return {
        "available": False,
        "reason": reason,
        "x_channel": None,
        "x_column": None,
        "points": 0,
        "channels": [],
    }


def _as_finite(value: Any) -> float | None:
    """*value* as a finite float, or ``None`` if it is not a usable number.

    Rows carry whatever a device reported: a missing reading (``None``), a
    string status, a NaN a readback wrote for a bad frame, or a numeric type
    that is not a Python ``float`` (a numpy scalar off a Tiled read). Anything
    that does not convert to a finite float is dropped from the pair, which is
    the pre-filter the stock statistics code assumes has already happened --
    a single NaN in its input poisons every statistic it computes.

    ``bool`` is excluded despite being an ``int``: a boolean reading is a state,
    and averaging states into a center of mass produces a number with no unit.
    """
    if value is None or isinstance(value, (bool, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _plain(value: Any) -> float | None:
    """One statistic as a plain JSON number, or ``None`` when it is not one.

    The stock code answers in numpy scalars, and answers ``None`` for a
    statistic the data does not support. Both reach the wire as JSON: a numpy
    float is not serializable, and a non-finite one serializes to a token no
    JSON reader accepts.
    """
    return _as_finite(value)


def _column_index(columns: Sequence[str], column: str) -> int | None:
    """Where *column* sits in a row, or ``None`` if it is not a column."""
    try:
        return list(columns).index(column)
    except ValueError:
        return None


def _cell(row: Any, index: int) -> Any:
    """One row's value at *index*, or ``None`` for a row that is too short.

    Rows are stored aligned to the column list, but a snapshot is data off a
    store and this module is the last stop before a fit: a ragged row costs a
    point here rather than an ``IndexError`` on a read route.
    """
    try:
        return row[index]
    except (IndexError, KeyError, TypeError):
        return None


def _pairs(rows: Sequence[Any], x_index: int, y_index: int) -> tuple[list[float], list[float]]:
    """The (x, y) pairs both of whose readings are finite numbers, in row order.

    Filtering by PAIR, not by column: a point whose readback reading is missing
    tells nothing about where the peak is, and keeping its x with a substituted
    y would move the center of mass toward a reading that never happened.
    """
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = _as_finite(_cell(row, x_index))
        y = _as_finite(_cell(row, y_index))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _peak_stats(peak_stats_cls: Any, xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    """Run the stock statistics callback over one channel's pairs.

    The callback is driven through its documented document protocol -- one
    ``event`` per point, then ``stop``, which is what triggers its computation
    -- rather than by reaching into its internals. It never sees a ``start``
    document because it does not need one, and the pairs are already filtered,
    so its own ``KeyError`` skip never fires.

    ``cen`` (the midpoint between the outermost half-maximum crossings) is the
    center, ``fwhm`` (the distance between them) the width, and ``com`` the
    center of mass. The first two are ``None`` for a channel that never crosses
    its own half-maximum.
    """
    stats = peak_stats_cls("x", "y")
    for x, y in zip(xs, ys, strict=True):
        stats("event", {"data": {"x": x, "y": y}})
    stats("stop", {})
    return {
        "center": _plain(stats.cen),
        "width": _plain(stats.fwhm),
        "center_of_mass": _plain(stats.com),
    }


def _channel_absent(channel: str, column: str | None, reason: str) -> dict[str, Any]:
    """One readable channel that could not be analyzed, and why."""
    return {
        "channel": channel,
        "column": column,
        "points": 0,
        "center": None,
        "width": None,
        "center_of_mass": None,
        "reason": reason,
    }


def analyze(
    columns: Sequence[str],
    rows: Sequence[Any],
    movable: Sequence[str],
    readable: Sequence[str],
) -> dict[str, Any]:
    """Peak statistics for one settled run's rows. Total -- never raises.

    Args:
        columns: The run's data column names, in row order.
        rows: Every row the run recorded, each aligned to *columns*. The whole
            run, never a window: statistics over a page of a run describe the
            page.
        movable: Channel names the plan's recorded parameters supplied for the
            movable role (``plan_fields.collect_channels``), in declaration
            order.
        readable: The same for the readable role.

    Returns:
        ``{"available", "reason", "x_channel", "x_column", "points",
        "channels"}``. When ``available`` is false, ``reason`` is one of this
        module's run-level reasons and ``channels`` is empty. When it is true,
        ``reason`` is null, ``x_channel``/``x_column`` name the independent
        variable and its data column, ``points`` is how many rows were read,
        and each entry of ``channels`` is ``{"channel", "column", "points",
        "center", "width", "center_of_mass", "reason"}`` -- ``reason`` null for
        an analyzed channel, and one of the per-channel reasons otherwise.

    A run whose plan declares nothing, or drives several channels, comes back
    absent rather than fitted against a guess: see the module docstring for the
    full vocabulary.
    """
    if not movable:
        return absent(REASON_NO_DECLARED_MOVABLE)
    if len(movable) > 1:
        # Stricter than the default figure, which draws against a single
        # RESOLVED column: a plan that declared two movables measured two
        # independent variables whether or not both reached the data, and a
        # peak in one of them is not a peak in the run.
        return absent(REASON_MULTIPLE_MOVABLE_CHANNELS)
    if not readable:
        return absent(REASON_NO_DECLARED_READABLE)

    x_channel = movable[0]
    x_column = resolve_column(x_channel, columns)
    if x_column is None:
        return absent(REASON_MOVABLE_COLUMN_MISSING)
    x_index = _column_index(columns, x_column)
    if x_index is None:  # pragma: no cover - resolve_column answers from `columns`
        return absent(REASON_MOVABLE_COLUMN_MISSING)
    if len(rows) < MIN_POINTS:
        return absent(REASON_INSUFFICIENT_POINTS)

    try:
        from bluesky.callbacks.fitting import PeakStats
    except Exception:
        logger.warning(
            "analysis: the stock statistics module is unavailable; "
            "this run's payload carries no statistics",
            exc_info=True,
        )
        return absent(REASON_STATISTICS_UNAVAILABLE)

    channels: list[dict[str, Any]] = []
    for channel in readable:
        column = resolve_column(channel, columns)
        if column is None:
            channels.append(_channel_absent(channel, None, REASON_CHANNEL_COLUMN_MISSING))
            continue
        if column == x_column:
            channels.append(_channel_absent(channel, column, REASON_CHANNEL_IS_THE_X_AXIS))
            continue
        y_index = _column_index(columns, column)
        if y_index is None:  # pragma: no cover - resolve_column answers from `columns`
            channels.append(_channel_absent(channel, column, REASON_CHANNEL_COLUMN_MISSING))
            continue
        xs, ys = _pairs(rows, x_index, y_index)
        if len(xs) < MIN_POINTS:
            channels.append(_channel_absent(channel, column, REASON_INSUFFICIENT_POINTS))
            continue
        try:
            stats = _peak_stats(PeakStats, xs, ys)
        except Exception:
            logger.warning(
                "analysis: statistics failed for channel %r against %r",
                channel,
                x_column,
                exc_info=True,
            )
            channels.append(_channel_absent(channel, column, REASON_STATISTICS_UNAVAILABLE))
            continue
        channels.append(
            {"channel": channel, "column": column, "points": len(xs), **stats, "reason": None}
        )

    return {
        "available": True,
        "reason": None,
        "x_channel": x_channel,
        "x_column": x_column,
        "points": len(rows),
        "channels": channels,
    }


# ---------------------------------------------------------------------------
# Settled-run cache
# ---------------------------------------------------------------------------
#
# `GET /runs/{id}/data` is polled, and a settled run's statistics are the same
# numbers on every poll. They are computed once and held here, keyed the way
# `figure_cache` keys its settled figures and invalidated by the same signal:
# the per-run generation counter the document plane bumps when a start document
# arrives for a run id that already has entries. A cached entry is served only
# when BOTH that generation and the run's row count still match what it was
# computed from, so a re-run of an interrupted item under the same id can never
# be answered with the previous occupant's peak.
#
# Only settled runs are cached. A run still producing rows has different
# statistics a moment later, which is why it has none at all until it settles.
# ---------------------------------------------------------------------------

_MAX_ENTRIES = 50
"""Settled analyses retained at once, oldest-*used* evicted first. Sized like
`figure_cache._MAX_ENTRIES`, for the same reason: the runs whose rows are still
retained are roughly the runs whose statistics are worth keeping."""

_lock = Lock()
# (run_id, source) -> (generation, total_seen, payload)
_entries: OrderedDict[tuple[str, str], tuple[int, int, dict[str, Any]]] = OrderedDict()


def cached(run_id: str, source: str, generation: int, total_seen: int) -> dict[str, Any] | None:
    """The stored analysis for this exact run occupant, or ``None``.

    Returned by reference and never copied, so a caller must treat the result
    as immutable. A hit refreshes the entry's position, so an analysis that
    keeps being asked for is not the one evicted to make room.
    """
    with _lock:
        entry = _entries.get((run_id, source))
        if entry is None:
            return None
        stored_generation, stored_total, payload = entry
        if stored_generation != generation or stored_total != total_seen:
            return None
        _entries.move_to_end((run_id, source))
        return payload


def store(
    run_id: str, source: str, generation: int, total_seen: int, payload: dict[str, Any]
) -> None:
    """Hold *payload* as this run occupant's settled analysis.

    Unconditional: unlike a figure store, there is no compare-and-set to lose,
    because the generation is part of what a read checks rather than a
    precondition of the write. An eviction landing mid-computation therefore
    leaves an entry that simply never matches again, and falls out by age.
    """
    with _lock:
        _entries[(run_id, source)] = (generation, total_seen, payload)
        _entries.move_to_end((run_id, source))
        while len(_entries) > _MAX_ENTRIES:
            _entries.popitem(last=False)


def _clear() -> None:
    """Drop every cached analysis (test isolation only)."""
    with _lock:
        _entries.clear()
