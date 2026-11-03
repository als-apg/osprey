"""A scenario ``step`` event with an optional end.

Start and end pair ``at`` -> ``until``, ``at_offset`` -> ``until_offset`` and
``at_when`` -> ``until_when``. Samples in ``[start, end)`` read ``to``; later
samples keep the baseline. A step without an end behaves as before. Validation
refuses an end that is not after its start, an end on an ``at_time`` step and
an end on a non-float channel. A bounded step's history window is its own
``(start, end)``, never ``(start, anchor)``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from osprey.facility.scenarios import check_scenario_events
from osprey.simulation.apply import event_subwindows, event_window
from osprey_connectors.simulation.series import apply_events, step_end

UTC_ZONE = ZoneInfo("UTC")
T0 = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
ANCHOR = T0.timestamp()
HORIZON = ANCHOR - 86400 * 7

CHANNELS = [
    {"id": "SR:X", "value_type": "float"},
    {"id": "SR:ON", "value_type": "bool", "options": ["OFF", "ON"]},
    {"id": "SR:N", "value_type": "int"},
]


def _problems(*events: Any, channel: str = "SR:X") -> list[str]:
    document = {
        "channels": CHANNELS,
        "scenarios": [{"name": "warm", "archiver": [{"channel": channel, "events": list(events)}]}],
    }
    return [error.format_message() for error in check_scenario_events(document)]


# ---------------------------------------------------------------- the engine


class TestApplyBoundedStep:
    def test_fraction_step_holds_inside_window_then_returns_to_baseline(self) -> None:
        base = np.full(11, 5.0)
        out = apply_events(
            base, [{"shape": "step", "at": 0.3, "until": 0.6, "to": 0.0}], 11, None, 0.0
        )
        # t_frac = 0.0, 0.1, ... 1.0; window [0.3, 0.6) holds 0.3, 0.4, 0.5.
        expected = np.full(11, 5.0)
        expected[3:6] = 0.0
        np.testing.assert_allclose(out, expected)

    def test_later_samples_keep_a_varying_baseline(self) -> None:
        base = np.linspace(10.0, 20.0, 11)
        out = apply_events(
            base, [{"shape": "step", "at": 0.2, "until": 0.5, "to": 0.0}], 11, None, 0.0
        )
        np.testing.assert_allclose(out[:2], base[:2])
        np.testing.assert_allclose(out[2:5], 0.0)
        np.testing.assert_allclose(out[5:], base[5:])

    def test_step_without_end_is_unchanged(self) -> None:
        out = apply_events(np.zeros(5), [{"shape": "step", "at": 0.5, "to": 9.0}], 5, None, 0.0)
        np.testing.assert_allclose(out, [0.0, 0.0, 9.0, 9.0, 9.0])

    def test_offset_step_is_bounded_on_the_epoch_axis(self) -> None:
        t_abs = ANCHOR + np.arange(-3600.0, 1.0, 600.0)  # -3600 .. 0, seven samples
        base = np.full(t_abs.size, 500.0)
        event = {"shape": "step", "at_offset": -2400, "until_offset": -1200, "to": 440.0}
        out = apply_events(base, [event], t_abs.size, t_abs, ANCHOR)
        # samples at -2400 and -1800 lie in [-2400, -1200); -1200 is the end and is excluded.
        np.testing.assert_allclose(out, [500.0, 500.0, 440.0, 440.0, 500.0, 500.0, 500.0])

    def test_when_step_is_bounded_like_its_logbook_instants(self) -> None:
        event = {
            "shape": "step",
            "at_when": {"days_ago": 4, "time": "03:15:00"},
            "until_when": {"days_ago": 4, "time": "04:30:00"},
            "to": 0.0,
        }
        start = datetime(2026, 6, 6, 3, 15, tzinfo=UTC).timestamp()
        end = datetime(2026, 6, 6, 4, 30, tzinfo=UTC).timestamp()
        assert step_end(event, ANCHOR, UTC_ZONE) == pytest.approx(end)
        t_abs = np.array([start - 60, start, start + 60, end - 1, end, end + 60])
        out = apply_events(np.full(6, 500.0), [event], 6, t_abs, ANCHOR, UTC_ZONE)
        np.testing.assert_allclose(out, [500.0, 0.0, 0.0, 0.0, 500.0, 500.0])

    def test_step_end_is_none_without_an_end(self) -> None:
        assert step_end({"shape": "step", "at_offset": -60, "to": 1.0}, ANCHOR) is None


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "event",
    [
        {"shape": "step", "at": 0.2, "until": 0.4, "to": 1.0},
        {"shape": "step", "at_offset": -3600, "until_offset": -60, "to": 1.0},
        {
            "shape": "step",
            "at_when": {"days_ago": 4, "time": "03:15:00"},
            "until_when": {"days_ago": 4, "time": "04:30:00"},
            "to": 1.0,
        },
        {
            "shape": "step",
            "at_when": {"days_ago": 2, "time": "23:00:00"},
            "until_when": {"days_ago": 1, "time": "01:00:00"},
            "to": 1.0,
        },
        {"shape": "step", "at_offset": -3600, "to": 1.0},
        {"shape": "step", "at_time": "08:00:00", "to": 1.0},
    ],
    ids=["fraction", "offset", "when", "when-across-midnight", "unbounded", "daily-unbounded"],
)
def test_well_formed_steps_pass(event: dict[str, Any]) -> None:
    assert _problems(event) == []


@pytest.mark.parametrize(
    ("event", "fragment"),
    [
        (
            {"shape": "step", "at": 0.4, "until": 0.2, "to": 1.0},
            "ends at `until` 0.2, not after `at` 0.4",
        ),
        (
            {"shape": "step", "at_offset": -60, "until_offset": -60, "to": 1.0},
            "ends at `until_offset` -60, not after `at_offset` -60",
        ),
        (
            {
                "shape": "step",
                "at_when": {"days_ago": 1, "time": "03:00:00"},
                "until_when": {"days_ago": 2, "time": "04:00:00"},
                "to": 1.0,
            },
            "not after `at_when`",
        ),
        (
            {
                "shape": "step",
                "at_when": {"days_ago": 1, "time": "05:00:00"},
                "until_when": {"days_ago": 1, "time": "04:00:00"},
                "to": 1.0,
            },
            "not after `at_when`",
        ),
    ],
    ids=["fraction", "offset-equal", "when-earlier-day", "when-earlier-time"],
)
def test_end_not_after_start_is_refused(event: dict[str, Any], fragment: str) -> None:
    (line,) = _problems(event)
    assert line.startswith("facility: value-invalid: scenario warm")
    assert fragment in line


def test_end_on_at_time_step_is_refused() -> None:
    (line,) = _problems({"shape": "step", "at_time": "08:00:00", "until": 0.5, "to": 1.0})
    assert "ends an `at_time` step with `until`" in line


@pytest.mark.parametrize("channel", ["SR:ON", "SR:N"])
def test_end_on_non_float_channel_is_refused(channel: str) -> None:
    (line,) = _problems(
        {"shape": "step", "at_offset": -60, "until_offset": -30, "to": 1}, channel=channel
    )
    assert f"of channel {channel} ends a `step` on a" in line


def test_unbounded_step_on_non_float_channel_still_passes() -> None:
    assert _problems({"shape": "step", "at_offset": -60, "to": "ON"}, channel="SR:ON") == []


@pytest.mark.parametrize(
    ("event", "fragment"),
    [
        (
            {"shape": "step", "at_offset": -60, "until": 0.5, "to": 1.0},
            "ends a `step` placed by `at_offset` with `until`",
        ),
        (
            {"shape": "step", "at": 0.1, "until": 0.5, "until_offset": 5, "to": 1.0},
            "ends a `step` placed by `at` with `until`, `until_offset`",
        ),
        (
            {"shape": "step", "at_offset": -60, "until_offset": "soon", "to": 1.0},
            "has `until_offset` 'soon', not a number",
        ),
        (
            {"shape": "step", "at": 0.1, "until": 1.5, "to": 1.0},
            "has `until` 1.5, outside 0 to 1",
        ),
        (
            {
                "shape": "step",
                "at_when": {"days_ago": 1, "time": "03:00:00"},
                "until_when": {"days_ago": 1},
                "to": 1.0,
            },
            "has `until_when` {'days_ago': 1}, not {days_ago: <int>, time: HH:MM[:SS]}",
        ),
    ],
    ids=["mismatched-pair", "two-ends", "end-not-number", "end-fraction-range", "end-when-shape"],
)
def test_malformed_end_is_refused(event: dict[str, Any], fragment: str) -> None:
    (line,) = _problems(event)
    assert fragment in line


# ---------------------------------------------------------------- windows


class TestBoundedWindows:
    BOUNDED = {
        "shape": "step",
        "at_offset": -4 * 86400,
        "until_offset": -4 * 86400 + 4500,
        "to": 0.0,
    }

    def test_bounded_step_window_is_its_own_span(self) -> None:
        start, end = event_window([self.BOUNDED], ANCHOR, HORIZON, tz=UTC_ZONE)
        assert start == pytest.approx(ANCHOR - 4 * 86400)
        assert end == pytest.approx(ANCHOR - 4 * 86400 + 4500)

    def test_bounded_step_subwindow_is_its_own_span(self) -> None:
        assert event_subwindows([self.BOUNDED], ANCHOR, HORIZON, tz=UTC_ZONE) == [
            pytest.approx((ANCHOR - 4 * 86400, ANCHOR - 4 * 86400 + 4500))
        ]

    def test_when_bounded_step_windows(self) -> None:
        event = {
            "shape": "step",
            "at_when": {"days_ago": 4, "time": "03:15:00"},
            "until_when": {"days_ago": 4, "time": "04:30:00"},
            "to": 0.0,
        }
        start = datetime(2026, 6, 6, 3, 15, tzinfo=UTC).timestamp()
        end = datetime(2026, 6, 6, 4, 30, tzinfo=UTC).timestamp()
        assert event_window([event], ANCHOR, HORIZON, tz=UTC_ZONE) == pytest.approx((start, end))
        assert event_subwindows([event], ANCHOR, HORIZON, tz=UTC_ZONE) == [
            pytest.approx((start, end))
        ]

    def test_unbounded_step_still_runs_to_the_anchor(self) -> None:
        event = {"shape": "step", "at_offset": -3600, "to": 0.0}
        assert event_window([event], ANCHOR, HORIZON, tz=UTC_ZONE) == pytest.approx(
            (ANCHOR - 3600, ANCHOR)
        )
        assert event_subwindows([event], ANCHOR, HORIZON, tz=UTC_ZONE) == [
            pytest.approx((ANCHOR - 3600, ANCHOR))
        ]

    def test_two_bounded_steps_stay_separate_subwindows(self) -> None:
        later = {"shape": "step", "at_offset": -7200, "until_offset": -3600, "to": 1.0}
        windows = event_subwindows([self.BOUNDED, later], ANCHOR, HORIZON, tz=UTC_ZONE)
        assert windows == [
            pytest.approx((ANCHOR - 4 * 86400, ANCHOR - 4 * 86400 + 4500)),
            pytest.approx((ANCHOR - 7200, ANCHOR - 3600)),
        ]
        assert event_window([self.BOUNDED, later], ANCHOR, HORIZON, tz=UTC_ZONE) == pytest.approx(
            (ANCHOR - 4 * 86400, ANCHOR - 3600)
        )

    def test_bounded_step_beside_a_persistent_step_runs_to_the_anchor(self) -> None:
        persistent = {"shape": "step", "at_offset": -60, "to": 2.0}
        _, end = event_window([persistent, self.BOUNDED], ANCHOR, HORIZON, tz=UTC_ZONE)
        assert end == pytest.approx(ANCHOR)
