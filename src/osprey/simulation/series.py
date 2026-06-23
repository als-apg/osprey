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

from collections.abc import Sequence
from datetime import datetime, timedelta
from datetime import time as dtime
from typing import Any

import numpy as np

from osprey.simulation.expressions import ExpressionError
from osprey.utils.logger import get_logger

logger = get_logger("simulation_series")


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
    event: dict[str, Any], t_frac: "np.ndarray", t_abs: "np.ndarray | None", anchor: float
) -> "list[tuple[np.ndarray, float]]":
    """Coordinate axis and event position(s) for one event.

    Fraction-positioned (``at``) and offset-anchored (``at_offset``) events
    yield one position, on the normalized window axis and the epoch-seconds
    axis respectively. Daily ``at_time`` events yield one epoch-seconds
    position per calendar date whose time-of-day occurrence falls inside the
    window. Returns an empty list when an anchored or time-of-day event
    cannot be placed because the timestamps were not convertible to epoch
    seconds.
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
        return [(t_abs, at) for at in daily_occurrences(str(event["at_time"]), t_abs)]
    return [(t_frac, float(event["at"]))]


def daily_occurrences(at_time: str, t_abs: "np.ndarray") -> list[float]:
    """Epoch-second positions of a daily time-of-day within ``[t_abs[0], t_abs[-1]]``.

    Walks calendar dates (local time) from the window start to its end and
    keeps every occurrence of ``at_time`` that falls inside the window.
    Assumes ascending timestamps (the same contract ``apply_events`` relies
    on via ``np.searchsorted``).
    """
    tod = dtime.fromisoformat(at_time)
    start = datetime.fromtimestamp(float(t_abs[0]))
    end = datetime.fromtimestamp(float(t_abs[-1]))
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
) -> list[str]:
    """Constant string series; only 'step' events are meaningful for strings."""
    t_frac = np.linspace(0.0, 1.0, n)
    series = [baseline] * n
    for event in events:
        if event["shape"] != "step":
            continue
        for x, at in event_positions(event, t_frac, t_abs, anchor):
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
) -> "np.ndarray":
    """Apply step/ramp/spike events in order (window-fraction or wall-clock)."""
    if not events:
        return series
    t_frac = np.linspace(0.0, 1.0, n)
    series = series.copy()
    for event in events:
        shape = event["shape"]
        offset_anchored = "at_offset" in event
        for x, at in event_positions(event, t_frac, t_abs, anchor):
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
