"""Unit tests for the relative-timestamp resolver (pure, no filesystem/DB)."""

from datetime import UTC, datetime
from datetime import time as dtime
from zoneinfo import ZoneInfo

import pytest

from osprey_connectors.relative_time import RelativeTimestamp, resolve_relative_timestamp


@pytest.mark.parametrize(
    ("now", "days_ago", "time", "expected"),
    [
        pytest.param(
            datetime(2026, 6, 13, 9, 41, 7),
            2,
            dtime(3, 15, 0),
            datetime(2026, 6, 11, 3, 15, 0),
            id="days-ago-and-pinned-time-of-day",
        ),
        pytest.param(
            datetime(2026, 6, 13, 23, 59, 59),
            0,
            dtime(14, 32, 0),
            datetime(2026, 6, 13, 14, 32, 0),
            id="days-ago-zero-is-same-date",
        ),
        pytest.param(
            datetime(2026, 3, 2, 12, 0, 0),
            4,
            dtime(8, 0, 0),
            datetime(2026, 2, 26, 8, 0, 0),
            id="shift-crosses-month-boundary",
        ),
    ],
)
def test_resolves_days_ago_and_pins_time_of_day(now, days_ago, time, expected):
    spec = RelativeTimestamp(days_ago=days_ago, time=time)
    assert resolve_relative_timestamp(spec, now) == expected


def test_naive_now_yields_naive_result():
    now = datetime(2026, 6, 13, 9, 0, 0)
    spec = RelativeTimestamp(days_ago=1, time=dtime(10, 0, 0))
    assert resolve_relative_timestamp(spec, now).tzinfo is None


def test_aware_now_preserves_tzinfo():
    now = datetime(2026, 6, 13, 9, 0, 0, tzinfo=UTC)
    spec = RelativeTimestamp(days_ago=1, time=dtime(10, 0, 0))
    resolved = resolve_relative_timestamp(spec, now)
    assert resolved.tzinfo is UTC
    assert resolved == datetime(2026, 6, 12, 10, 0, 0, tzinfo=UTC)


def test_wall_clock_time_is_exact_across_dst_transition():
    # US DST began 2026-03-08. A 3-day shift over that seam must still pin the
    # wall-clock time exactly (whole-day shift, not a 72-hour span).
    tz = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, 3, 10, 12, 0, 0, tzinfo=tz)
    spec = RelativeTimestamp(days_ago=3, time=dtime(2, 30, 0))
    resolved = resolve_relative_timestamp(spec, now)
    assert resolved.date().isoformat() == "2026-03-07"
    assert (resolved.hour, resolved.minute) == (2, 30)
    assert resolved.tzinfo is tz
