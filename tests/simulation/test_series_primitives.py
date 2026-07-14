"""Direct unit tests for the pure synthesis primitives in ``simulation.series``.

These cover the stateless functions in isolation (no engine). The engine-driven
behavior is exercised separately in ``test_series.py``; this file locks the
module's standalone contract so the extracted primitives can be refactored
without going through the full engine each time.
"""

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from osprey.simulation.expressions import ExpressionError
from osprey.simulation.series import (
    apply_events,
    clamp,
    daily_occurrences,
    epoch_seconds_array,
    event_positions,
    ref_value,
    string_series,
)


class TestClamp:
    def test_within_bounds_unchanged(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_clamps_to_min_and_max(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0
        assert clamp(99.0, 0.0, 10.0) == 10.0

    def test_optional_bounds(self):
        assert clamp(-50.0, None, 10.0) == -50.0
        assert clamp(50.0, 0.0, None) == 50.0
        assert clamp(123.0, None, None) == 123.0


class TestEpochSecondsArray:
    def test_datetime_objects(self):
        dts = [datetime(2026, 1, 1, 0, 0, 0), datetime(2026, 1, 1, 1, 0, 0)]
        out = epoch_seconds_array(dts)
        assert out is not None
        np.testing.assert_allclose(out, [dts[0].timestamp(), dts[1].timestamp()])

    def test_epoch_floats_pass_through(self):
        out = epoch_seconds_array([100.0, 200, 300.5])
        np.testing.assert_allclose(out, [100.0, 200.0, 300.5])

    def test_non_convertible_returns_none(self):
        assert epoch_seconds_array(["not-a-time", 1.0]) is None

    def test_bool_is_rejected(self):
        # bool is an int subclass but must not be treated as an epoch second.
        assert epoch_seconds_array([True]) is None


class TestRefValue:
    def test_numeric_lookup(self):
        ref_series = {"PV:A": np.array([1.0, 2.0, 3.0])}
        assert ref_value(ref_series, "PV:A", 2) == 3.0

    def test_string_series_raises(self):
        ref_series = {"PV:STATE": ["OPEN", "CLOSED"]}
        with pytest.raises(ExpressionError, match="cannot be used in an expression"):
            ref_value(ref_series, "PV:STATE", 0)


class TestEventPositions:
    def test_fraction_position(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        positions = event_positions({"shape": "step", "at": 0.5, "to": 1.0}, t_frac, None, 0.0)
        assert len(positions) == 1
        axis, at = positions[0]
        assert at == 0.5
        assert axis is t_frac

    def test_offset_anchored_needs_abs_axis(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        assert event_positions({"at_offset": 10.0}, t_frac, None, 100.0) == []

    def test_offset_anchored_with_abs_axis(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        t_abs = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        positions = event_positions({"at_offset": 10.0}, t_frac, t_abs, 100.0)
        assert len(positions) == 1
        axis, at = positions[0]
        assert at == 110.0
        assert axis is t_abs

    def test_time_of_day_needs_abs_axis(self):
        t_frac = np.linspace(0.0, 1.0, 5)
        assert event_positions({"at_time": "12:00:00"}, t_frac, None, 0.0) == []


class TestDailyOccurrences:
    """``at_time`` occurrences are placed in the facility timezone, not the box's
    local zone. The tests pass an explicit ``tz`` and build expected positions in
    that same zone, then assert the result is independent of the host ``$TZ``."""

    def test_one_per_calendar_date_in_window(self):
        tz = ZoneInfo("UTC")
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        end = datetime(2026, 1, 2, 23, 0, 0, tzinfo=tz)
        t_abs = np.array([start.timestamp(), end.timestamp()])
        occ = daily_occurrences("12:00:00", t_abs, tz=tz)
        assert len(occ) == 2
        assert occ[0] == datetime(2026, 1, 1, 12, 0, 0, tzinfo=tz).timestamp()
        assert occ[1] == datetime(2026, 1, 2, 12, 0, 0, tzinfo=tz).timestamp()

    def test_occurrence_outside_window_excluded(self):
        # Window ends before noon on the only date → no occurrence.
        tz = ZoneInfo("UTC")
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        end = datetime(2026, 1, 1, 6, 0, 0, tzinfo=tz)
        t_abs = np.array([start.timestamp(), end.timestamp()])
        assert daily_occurrences("12:00:00", t_abs, tz=tz) == []

    def test_positions_are_box_tz_independent(self):
        """A UTC ``tz`` yields UTC-noon epochs even when the host ``$TZ`` is a
        wildly different zone — this is the regression that broke the archiver
        ``at_time`` benchmark on non-UTC deploy boxes."""
        tz = ZoneInfo("UTC")
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=tz)
        end = datetime(2026, 1, 1, 23, 0, 0, tzinfo=tz)
        t_abs = np.array([start.timestamp(), end.timestamp()])
        expected = datetime(2026, 1, 1, 12, 0, 0, tzinfo=tz).timestamp()

        old_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "Pacific/Kiritimati"  # UTC+14, maximally far from UTC
            time.tzset()
            occ = daily_occurrences("12:00:00", t_abs, tz=tz)
        finally:
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            time.tzset()

        assert occ == [expected]

    def test_dst_boundary_one_occurrence_per_date(self):
        """Characterization: across a spring-forward DST boundary the daily walk
        still yields exactly one (local-noon) occurrence per calendar date — no
        skipped or duplicated date. Noon is unambiguous (well clear of the 02:00
        gap), so this pins the current contract and would catch a future cursor
        refactor that miscounts dates across the transition."""
        ny = ZoneInfo("America/New_York")
        # 2026-03-08 is the US spring-forward date (02:00 -> 03:00 EST->EDT).
        start = datetime(2026, 3, 7, 0, 0, tzinfo=ny)
        end = datetime(2026, 3, 10, 23, 0, tzinfo=ny)
        t_abs = np.array([start.timestamp(), end.timestamp()])

        occ = daily_occurrences("12:00:00", t_abs, tz=ny)

        assert len(occ) == 4  # Mar 7, 8, 9, 10
        dates = {datetime.fromtimestamp(e, ny).date() for e in occ}
        assert len(dates) == 4  # one per distinct calendar date
        for epoch in occ:
            local = datetime.fromtimestamp(epoch, ny)
            assert (local.hour, local.minute) == (12, 0)


class TestStringSeries:
    def test_step_switches_value(self):
        out = string_series("OPEN", [{"shape": "step", "at": 0.5, "to": "CLOSED"}], 5, None, 0.0)
        # t_frac = [0, .25, .5, .75, 1]; switch at >= 0.5 → last 3 entries.
        assert out == ["OPEN", "OPEN", "CLOSED", "CLOSED", "CLOSED"]

    def test_non_step_events_ignored_for_strings(self):
        out = string_series("OPEN", [{"shape": "spike", "at": 0.5}], 3, None, 0.0)
        assert out == ["OPEN", "OPEN", "OPEN"]


class TestApplyEvents:
    def test_empty_events_returns_input(self):
        series = np.array([1.0, 2.0, 3.0])
        assert apply_events(series, [], 3, None, 0.0) is series

    def test_step(self):
        series = np.zeros(5)
        out = apply_events(series, [{"shape": "step", "at": 0.5, "to": 9.0}], 5, None, 0.0)
        np.testing.assert_allclose(out, [0.0, 0.0, 9.0, 9.0, 9.0])

    def test_ramp_interpolates(self):
        series = np.zeros(5)
        out = apply_events(
            series, [{"shape": "ramp", "at": 0.0, "until": 1.0, "to": 4.0}], 5, None, 0.0
        )
        np.testing.assert_allclose(out, [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_spike_adds_gaussian_bump(self):
        series = np.zeros(11)
        out = apply_events(
            series, [{"shape": "spike", "at": 0.5, "amplitude": 2.0, "width": 0.1}], 11, None, 0.0
        )
        # Peak at the center of the window.
        assert out.argmax() == 5
        assert out[5] == pytest.approx(2.0, rel=1e-6)
        # Input is not mutated (apply_events copies).
        np.testing.assert_array_equal(series, np.zeros(11))
