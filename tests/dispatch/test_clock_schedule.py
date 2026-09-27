"""Unit tests for the clock schedule of a clock-time cron trigger."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from osprey.dispatch.clock_schedule import next_fire, parse_clock_schedule, previous_fire


def test_times_and_days_parse_into_a_schedule():
    schedule = parse_clock_schedule("t", {"at": ["17:00", "07:45"], "days": ["mon", "fri"]})

    assert schedule is not None
    assert schedule.times == (time(7, 45), time(17, 0))
    assert schedule.days == frozenset({0, 4})


def test_times_without_days_mean_every_day():
    schedule = parse_clock_schedule("t", {"at": ["07:45"]})

    assert schedule is not None
    assert schedule.days is None


@pytest.mark.parametrize("source_config", [{"interval_sec": 60}, {}])
def test_a_trigger_with_neither_key_has_no_clock_schedule(source_config):
    assert parse_clock_schedule("t", source_config) is None


@pytest.mark.parametrize(
    ("source_config", "fragment"),
    [
        pytest.param({"at": ["07:45"], "interval_sec": 60}, "interval_sec", id="at-and-interval"),
        pytest.param({"days": ["mon"]}, "days", id="days-without-at"),
        pytest.param({"at": "07:45"}, "at", id="at-not-a-list"),
        pytest.param({"at": []}, "at", id="at-empty"),
        pytest.param({"at": [465]}, '"07:45"', id="at-unquoted-number"),
        pytest.param({"at": ["7:45"]}, "7:45", id="at-one-digit-hour"),
        pytest.param({"at": ["24:00"]}, "24:00", id="at-hour-out-of-range"),
        pytest.param({"at": ["07:60"]}, "07:60", id="at-minute-out-of-range"),
        pytest.param({"at": ["07:45", "07:45"]}, "07:45", id="at-repeated"),
        pytest.param({"at": ["07:45"], "days": []}, "days", id="days-empty"),
        pytest.param({"at": ["07:45"], "days": ["Mon"]}, "Mon", id="days-capitalised"),
        pytest.param({"at": ["07:45"], "days": ["monday"]}, "monday", id="days-long-name"),
        pytest.param({"at": ["07:45"], "days": ["mon", "mon"]}, "mon", id="days-repeated"),
        pytest.param({"at": ["07:45"], "day": ["mon"]}, "day", id="unknown-key-beside-at"),
    ],
)
def test_an_unreadable_schedule_is_refused(source_config, fragment):
    with pytest.raises(ValueError) as excinfo:
        parse_clock_schedule("morning-report", source_config)

    message = str(excinfo.value)
    assert "morning-report" in message
    assert fragment in message


# ---------------------------------------------------------------------------
# Slot instants
# ---------------------------------------------------------------------------

_LA = ZoneInfo("America/Los_Angeles")


def test_next_fire_is_strictly_after_the_given_instant():
    schedule = parse_clock_schedule("t", {"at": ["07:45", "17:00"]})
    assert schedule is not None
    on_slot = datetime(2026, 9, 28, 7, 45, tzinfo=_LA)

    assert next_fire(schedule, on_slot, _LA) == datetime(2026, 9, 28, 17, 0, tzinfo=_LA)


def test_two_times_inside_one_gap_fire_once():
    schedule = parse_clock_schedule("t", {"at": ["02:15", "02:45", "03:00"]})
    assert schedule is not None
    day_before = datetime(2027, 3, 13, 12, 0, tzinfo=_LA)

    first = next_fire(schedule, day_before, _LA)
    assert first == datetime(2027, 3, 14, 10, 0, tzinfo=UTC)
    assert first.tzinfo is UTC
    assert next_fire(schedule, first, _LA) == datetime(2027, 3, 15, 9, 15, tzinfo=UTC)


def test_previous_fire_is_the_last_slot_at_or_before_an_instant():
    schedule = parse_clock_schedule("t", {"at": ["07:45"], "days": ["mon", "fri"]})
    assert schedule is not None
    wednesday = datetime(2026, 9, 30, 12, 0, tzinfo=_LA)
    monday_slot = datetime(2026, 9, 28, 7, 45, tzinfo=_LA)

    assert previous_fire(schedule, wednesday, _LA) == monday_slot
    assert previous_fire(schedule, monday_slot, _LA) == monday_slot
    assert previous_fire(schedule, monday_slot - timedelta(seconds=1), _LA) == datetime(
        2026, 9, 25, 7, 45, tzinfo=_LA
    )
