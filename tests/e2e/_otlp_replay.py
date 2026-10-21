"""Replaying a captured OTLP session at the present time.

A checked-in capture carries the times it was recorded at, and the gateway
drops anything older than the store's retention window. Replaying it therefore
moves every OTLP time field by one offset, so the newest record lands a few
minutes before *now* and every interval between two records is kept.
"""

from __future__ import annotations

import copy
import datetime
from collections.abc import Iterator
from typing import Any

#: Where the newest replayed record lands, before *now*.
NEWEST_BEFORE_NOW = datetime.timedelta(minutes=5)

#: Every OTLP/JSON field that carries a time: a log record's time and observed
#: time, a span's start and end, a span event's time, and a data point's (or
#: exemplar's) time and start time.
TIME_FIELDS = frozenset(
    {"timeUnixNano", "observedTimeUnixNano", "startTimeUnixNano", "endTimeUnixNano"}
)


def _ns(value: datetime.datetime | int) -> int:
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
        delta = value - epoch
        return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000
    return int(value)


def _time_slots(node: Any) -> Iterator[tuple[dict[str, Any], str]]:
    """Every (mapping, key) pair holding a non-zero OTLP time, depth first."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in TIME_FIELDS and int(value or 0) > 0:
                yield node, key
            else:
                yield from _time_slots(value)
    elif isinstance(node, list):
        for item in node:
            yield from _time_slots(item)


def otlp_times(payload: Any) -> list[int]:
    """Every non-zero OTLP time in *payload*, in nanoseconds since the epoch."""
    return [int(holder[key]) for holder, key in _time_slots(payload)]


def rebase_otlp_times(payload: Any, now: datetime.datetime | int) -> Any:
    """A copy of *payload* with every OTLP time moved by one offset.

    The newest time lands at ``now - 5 min``; every difference between two
    times is unchanged. A field the capture left at zero (unset) stays zero,
    and each value keeps the type it had (OTLP/JSON writes these as strings).

    Args:
        payload: An OTLP/JSON request body, or any structure holding several
            (a capture with a ``requests`` list).
        now: The present, as an aware datetime or nanoseconds since the epoch.

    Raises:
        ValueError: *payload* carries no time at all.
    """
    rebased = copy.deepcopy(payload)
    slots = list(_time_slots(rebased))
    if not slots:
        raise ValueError("the payload carries no OTLP time to rebase")
    newest = max(int(holder[key]) for holder, key in slots)
    lead = NEWEST_BEFORE_NOW // datetime.timedelta(microseconds=1) * 1_000
    offset = _ns(now) - lead - newest
    for holder, key in slots:
        moved = int(holder[key]) + offset
        holder[key] = str(moved) if isinstance(holder[key], str) else moved
    return rebased
