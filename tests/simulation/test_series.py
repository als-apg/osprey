"""Tests for synthesize_series(): archiver event shapes and pointwise exprs."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from osprey.simulation import SimulationEngine

QUAD_DRIFT_TRANS = 98.5 - 0.85 * abs(28.4 - 42.0)  # 86.94


def _timestamps(n=200):
    start = datetime(2024, 1, 1)
    return [start + timedelta(seconds=i) for i in range(n)]


class TestBaselineSeries:
    """Baseline-value channels: constant baseline plus per-channel noise."""

    def test_constant_baseline_no_noise(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        series = engine.synthesize_series("T:Q1:CUR:SP", _timestamps())
        assert len(series) == 200
        assert all(v == 42.0 for v in series)

    def test_noisy_baseline(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        series = np.array(engine.synthesize_series("T:NOISY", _timestamps()))
        assert series.std() > 0
        assert series.mean() == pytest.approx(100.0, rel=0.05)

    def test_string_series(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        series = engine.synthesize_series("T:MODE", _timestamps())
        assert series == ["CW"] * 200

    def test_empty_timestamps(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        assert engine.synthesize_series("T:Q1:CUR:SP", []) == []

    def test_unknown_channel_raises(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        with pytest.raises(KeyError):
            engine.synthesize_series("NO:SUCH:PV", _timestamps())


class TestSeriesBounds:
    """``min``/``max`` clamp synthesized series after events and noise."""

    def test_spike_below_min_saturates_at_floor(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:FWD"] = {"value": 450.0, "noise": 0.0, "min": 0.0}
        machine_dict["scenarios"]["quad-drift"]["archiver"] = [
            {
                "channel": "T:FWD",
                "events": [{"shape": "spike", "at": 0.5, "amplitude": -600.0, "width": 0.05}],
            }
        ]
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("quad-drift")
        series = np.array(engine.synthesize_series("T:FWD", _timestamps()))
        assert series.min() == 0.0  # floor is actually reached (saturation)
        assert (series >= 0.0).all()  # never negative

    def test_spike_above_max_saturates_at_ceiling(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:P"] = {"value": 5.0, "noise": 0.0, "max": 50.0}
        machine_dict["scenarios"]["quad-drift"]["archiver"] = [
            {
                "channel": "T:P",
                "events": [{"shape": "spike", "at": 0.5, "amplitude": 200.0, "width": 0.05}],
            }
        ]
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("quad-drift")
        series = np.array(engine.synthesize_series("T:P", _timestamps()))
        assert series.max() == 50.0
        assert (series <= 50.0).all()

    def test_derived_series_sees_clamped_inputs(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:FWD"] = {"value": 450.0, "noise": 0.0, "min": 0.0}
        machine_dict["channels"]["T:REV"] = {"value": 5.0, "noise": 0.0}
        machine_dict["channels"]["T:NET"] = {
            "expr": "ch('T:FWD') - ch('T:REV')",
            "noise": 0.0,
        }
        machine_dict["scenarios"]["quad-drift"]["archiver"] = [
            {
                "channel": "T:FWD",
                "events": [{"shape": "spike", "at": 0.5, "amplitude": -600.0, "width": 0.05}],
            }
        ]
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("quad-drift")
        net = np.array(engine.synthesize_series("T:NET", _timestamps()))
        # NET tracks FWD-REV with FWD floored at 0: minimum is 0 - 5 = -5, not 450-600-5.
        assert net.min() == pytest.approx(-5.0)

    def test_unbounded_channel_unchanged(self, machine_dict, make_machine_file):
        machine_dict["channels"]["T:FREE"] = {"value": 10.0, "noise": 0.0}
        machine_dict["scenarios"]["quad-drift"]["archiver"] = [
            {
                "channel": "T:FREE",
                "events": [{"shape": "spike", "at": 0.5, "amplitude": -600.0, "width": 0.05}],
            }
        ]
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("quad-drift")
        series = np.array(engine.synthesize_series("T:FREE", _timestamps()))
        assert series.min() < -100.0  # no bounds → spike goes deeply negative


class TestEventShapes:
    """Step, ramp, and spike events at window-relative positions."""

    def test_step(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        series = engine.synthesize_series("T:Q1:CUR:SP", _timestamps())
        t = np.linspace(0, 1, 200)
        before = [v for v, ti in zip(series, t, strict=True) if ti < 0.35]
        after = [v for v, ti in zip(series, t, strict=True) if ti >= 0.35]
        assert all(v == 42.0 for v in before)
        assert all(v == 28.4 for v in after)

    def test_ramp_then_step(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        series = np.array(engine.synthesize_series("T:VAC", _timestamps(1000)))
        t = np.linspace(0, 1, 1000)

        # Flat baseline before the ramp starts
        assert series[t < 0.15].max() == pytest.approx(8.0e-9)
        # Monotonic-ish rise toward the ramp target just before the trip
        ramp_end = series[(t > 0.85) & (t < 0.88)]
        assert ramp_end.max() == pytest.approx(9.5e-8, rel=0.05)
        # Clean trip edge: steps to the override-consistent value
        assert all(v == pytest.approx(3.8e-7) for v in series[t >= 0.88])
        # Mid-ramp value sits between baseline and ramp target
        mid = series[(t > 0.5) & (t < 0.55)]
        assert 8.0e-9 < mid.min() and mid.max() < 9.5e-8

    def test_spike(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        series = np.array(engine.synthesize_series("T:RF:STATUS", _timestamps(1000)))
        t = np.linspace(0, 1, 1000)

        # Gaussian dip centered at t=0.5 (amplitude -0.2)
        at_spike = series[np.abs(t - 0.5).argmin()]
        assert at_spike == pytest.approx(0.8, abs=0.01)
        # Flat at 1 away from the spike, before the trip
        assert series[np.abs(t - 0.3).argmin()] == pytest.approx(1.0, abs=0.01)
        # Step to 0 at the trip
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in series[t >= 0.9])

    def test_string_step(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        series = engine.synthesize_series("T:MODE", _timestamps())
        t = np.linspace(0, 1, 200)
        for value, ti in zip(series, t, strict=True):
            assert value == ("FAULT" if ti >= 0.88 else "CW")


class TestPointwiseExpressions:
    """Expr channels evaluate pointwise over referenced series."""

    def test_derived_series_follows_step(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        trans = np.array(engine.synthesize_series("T:TRANS", _timestamps()))
        t = np.linspace(0, 1, 200)
        assert trans[t < 0.35].max() == pytest.approx(98.5)
        assert trans[t >= 0.35].min() == pytest.approx(QUAD_DRIFT_TRANS)
        assert trans[t >= 0.35].max() == pytest.approx(QUAD_DRIFT_TRANS)

    def test_readback_series_follows_setpoint(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        rb = engine.synthesize_series("T:Q1:CUR:RB", _timestamps())
        sp_values = {42.0, 28.4}
        assert set(rb) == sp_values

    def test_derived_series_collapses_on_trip(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("vac-leak")
        trans = np.array(engine.synthesize_series("T:TRANS", _timestamps(1000)))
        t = np.linspace(0, 1, 1000)
        # Transmission follows the RF status step down to zero at the trip
        assert trans[t < 0.4].max() == pytest.approx(98.5, rel=0.01)
        assert all(v == pytest.approx(0.0, abs=1e-9) for v in trans[t >= 0.9])

    def test_nominal_derived_series_flat(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        trans = engine.synthesize_series("T:TRANS", _timestamps(50))
        assert all(v == pytest.approx(98.5) for v in trans)


class TestWallClockAnchoredEvents:
    """Events with at_offset anchor to scenario-activation time, not window fraction."""

    @staticmethod
    def _anchored_machine(machine_dict):
        machine_dict["scenarios"]["anchored"] = {
            "description": "Wall-clock anchored events.",
            "overrides": {"T:Q1:CUR:SP": 28.4},
            "archiver": [
                {
                    "channel": "T:Q1:CUR:SP",
                    "events": [{"shape": "step", "at_offset": -50, "to": 28.4}],
                },
                {
                    "channel": "T:VAC",
                    "events": [
                        {
                            "shape": "ramp",
                            "at_offset": -120,
                            "until_offset": -60,
                            "to": 9.5e-8,
                        }
                    ],
                },
                {
                    "channel": "T:MODE",
                    "events": [{"shape": "step", "at_offset": -50, "to": "FAULT"}],
                },
                {
                    "channel": "T:RF:STATUS",
                    "events": [{"shape": "step", "at_offset": 1000, "to": 0}],
                },
            ],
        }
        return machine_dict

    @staticmethod
    def _window(n=200):
        """1 Hz window covering [now-150s, now+50s)."""
        start = datetime.now() - timedelta(seconds=150)
        return [start + timedelta(seconds=i) for i in range(n)]

    def _engine(self, machine_dict, make_machine_file):
        engine = SimulationEngine.from_file(make_machine_file(self._anchored_machine(machine_dict)))
        engine.set_active_scenario("anchored")  # anchor = state-file mtime = now
        return engine

    def test_step_lands_at_offset_from_activation(self, machine_dict, make_machine_file):
        engine = self._engine(machine_dict, make_machine_file)
        timestamps = self._window()
        series = engine.synthesize_series("T:Q1:CUR:SP", timestamps)
        now = datetime.now()
        for value, ts in zip(series, timestamps, strict=True):
            seconds_from_now = (ts - now).total_seconds()
            if seconds_from_now < -55:
                assert value == 42.0
            elif seconds_from_now > -45:
                assert value == 28.4

    def test_ramp_spans_offset_interval(self, machine_dict, make_machine_file):
        engine = self._engine(machine_dict, make_machine_file)
        timestamps = self._window()
        series = engine.synthesize_series("T:VAC", timestamps)
        now = datetime.now()
        offsets = [(ts - now).total_seconds() for ts in timestamps]
        before = [v for v, off in zip(series, offsets, strict=True) if off < -125]
        after = [v for v, off in zip(series, offsets, strict=True) if off > -55]
        mid = [v for v, off in zip(series, offsets, strict=True) if -100 < off < -80]
        assert max(before) == pytest.approx(8.0e-9)
        assert all(v == pytest.approx(9.5e-8) for v in after)
        assert all(8.0e-9 < v < 9.5e-8 for v in mid)

    def test_string_step_at_offset(self, machine_dict, make_machine_file):
        engine = self._engine(machine_dict, make_machine_file)
        timestamps = self._window()
        series = engine.synthesize_series("T:MODE", timestamps)
        now = datetime.now()
        for value, ts in zip(series, timestamps, strict=True):
            seconds_from_now = (ts - now).total_seconds()
            if seconds_from_now < -55:
                assert value == "CW"
            elif seconds_from_now > -45:
                assert value == "FAULT"

    def test_event_after_window_does_not_appear(self, machine_dict, make_machine_file):
        engine = self._engine(machine_dict, make_machine_file)
        series = engine.synthesize_series("T:RF:STATUS", self._window())
        assert all(v == 1.0 for v in series)

    def test_numeric_epoch_timestamps_supported(self, machine_dict, make_machine_file):
        import time

        engine = self._engine(machine_dict, make_machine_file)
        now = time.time()
        timestamps = [now - 150 + i for i in range(200)]
        series = engine.synthesize_series("T:Q1:CUR:SP", timestamps)
        for value, ts in zip(series, timestamps, strict=True):
            if ts - now < -55:
                assert value == 42.0
            elif ts - now > -45:
                assert value == 28.4

    def test_fraction_events_unaffected(self, machine_file):
        engine = SimulationEngine.from_file(machine_file)
        engine.set_active_scenario("quad-drift")
        series = engine.synthesize_series("T:Q1:CUR:SP", _timestamps())
        t = np.linspace(0, 1, 200)
        assert all(v == 42.0 for v, ti in zip(series, t, strict=True) if ti < 0.35)
        assert all(v == 28.4 for v, ti in zip(series, t, strict=True) if ti >= 0.35)


class TestDailyTimeAnchor:
    """at_time events fire at a wall-clock time-of-day, every date in window."""

    @staticmethod
    def _burst_machine(machine_dict, events, channel="T:VAC"):
        machine_dict["scenarios"]["burst"] = {
            "description": "Daily vacuum burst.",
            "archiver": [{"channel": channel, "events": events}],
        }
        return machine_dict

    def _engine(self, machine_dict, make_machine_file, events, channel="T:VAC"):
        machine = self._burst_machine(machine_dict, events, channel)
        engine = SimulationEngine.from_file(make_machine_file(machine))
        engine.set_active_scenario("burst")
        return engine

    def test_spike_fires_at_time_of_day(self, machine_dict, make_machine_file):
        engine = self._engine(
            machine_dict,
            make_machine_file,
            [{"shape": "spike", "at_time": "14:32:08", "amplitude": 1.5e-7, "width": 15}],
        )
        day = datetime.now().replace(hour=14, minute=30, second=0, microsecond=0)
        ts = [day + timedelta(seconds=i) for i in range(600)]  # 14:30-14:40
        series = engine.synthesize_series("T:VAC", ts)
        peak_idx = max(range(len(series)), key=lambda i: series[i])
        assert abs((ts[peak_idx] - day).total_seconds() - 128) < 5  # peak at 14:32:08
        assert max(series) > 1.0e-7

    def test_no_event_outside_window(self, machine_dict, make_machine_file):
        engine = self._engine(
            machine_dict,
            make_machine_file,
            [{"shape": "spike", "at_time": "14:32:08", "amplitude": 1.5e-7, "width": 15}],
        )
        day = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
        ts = [day + timedelta(seconds=i) for i in range(600)]
        series = engine.synthesize_series("T:VAC", ts)
        assert max(series) < 1.0e-7  # flat baseline

    def test_fires_every_date_in_multiday_window(self, machine_dict, make_machine_file):
        # width 600 s: at a 10-min grid the nearest sample is <=300 s from the
        # peak, so the Gaussian stays above threshold regardless of grid phase.
        engine = self._engine(
            machine_dict,
            make_machine_file,
            [{"shape": "spike", "at_time": "14:32:08", "amplitude": 1.5e-7, "width": 600}],
        )
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ts = [start + timedelta(minutes=10 * i) for i in range(288)]  # 2 days @ 10 min
        series = engine.synthesize_series("T:VAC", ts)
        peaks = [i for i in range(1, 287) if series[i] > 1.0e-7]
        days_hit = {ts[i].date() for i in peaks}
        assert len(days_hit) == 2

    def test_epoch_second_timestamps_supported(self, machine_dict, make_machine_file):
        engine = self._engine(
            machine_dict,
            make_machine_file,
            [{"shape": "spike", "at_time": "14:32:08", "amplitude": 1.5e-7, "width": 15}],
        )
        day = datetime.now().replace(hour=14, minute=30, second=0, microsecond=0)
        ts = [(day + timedelta(seconds=i)).timestamp() for i in range(600)]
        series = engine.synthesize_series("T:VAC", ts)
        assert max(series) > 1.0e-7

    def test_string_step_at_time_of_day(self, machine_dict, make_machine_file):
        engine = self._engine(
            machine_dict,
            make_machine_file,
            [{"shape": "step", "at_time": "14:32:08", "to": "FAULT"}],
            channel="T:MODE",
        )
        day = datetime.now().replace(hour=14, minute=30, second=0, microsecond=0)
        ts = [day + timedelta(seconds=i) for i in range(600)]
        series = engine.synthesize_series("T:MODE", ts)
        for value, t in zip(series, ts, strict=True):
            expected = "FAULT" if (t - day).total_seconds() >= 128 else "CW"
            assert value == expected
