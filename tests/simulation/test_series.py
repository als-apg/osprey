"""Tests for synthesize_series(): archiver event shapes and pointwise exprs."""

import copy
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from osprey.simulation import SimulationEngine

QUAD_DRIFT_TRANS = 98.5 - 0.85 * abs(28.4 - 42.0)  # 86.94


def _timestamps(n=200):
    start = datetime(2024, 1, 1)
    return [start + timedelta(seconds=i) for i in range(n)]


class TestBaselineSeries:
    """Baseline-value channels: constant baseline plus the channel's declared signal model.

    Relative ``noise`` scales the series multiplicatively, ``noise_abs`` adds an
    absolute-sigma term, and both are deterministic keyed draws addressed by
    (channel, absolute timestamp) — not stateful RNG streams — so a channel with
    neither declared stays exactly constant.
    """

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


def _abs_window(offset_s=0, n=600, step_s=1):
    """UTC-aware 1 Hz (by default) timestamps, absolutely positioned."""
    start = datetime(2024, 3, 11, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=offset_s)
    return [start + timedelta(seconds=i * step_s) for i in range(n)]


class TestSignalModelSynthesis:
    """Deterministic keyed noise (relative + absolute) and wander texture.

    Synthesis draws are pure functions of (channel name, absolute timestamp):
    identical or overlapping archiver windows agree pointwise on shared
    timestamps, zero-baseline channels carry additive ``noise_abs``, and
    ``texture`` adds slow baseline motion that rides post-event levels.
    """

    def test_absolute_noise_on_zero_baseline(self, machine_file):
        """FR1: sigma>0 on a 0.0 baseline produces N(0, sigma) samples, never a constant."""
        engine = SimulationEngine.from_file(machine_file)
        series = np.array(engine.synthesize_series("T:ZERO:NOISY", _abs_window(n=500)))
        sigma = 0.02  # T:ZERO:NOISY noise_abs in conftest
        assert len(set(series.tolist())) > 1
        assert 0.5 * sigma < series.std() < 1.5 * sigma
        assert abs(series.mean()) < 0.5 * sigma

    def test_identical_windows_agree_across_engine_instances(self, machine_dict, make_machine_file):
        """FR2: two independently-built engines synthesize identical series."""
        engine_a = SimulationEngine.from_file(make_machine_file(machine_dict, "machine_a.json"))
        engine_b = SimulationEngine.from_file(make_machine_file(machine_dict, "machine_b.json"))
        ts = _abs_window(n=300)
        for pv in ("T:NOISY", "T:ZERO:NOISY", "T:ZERO:TEXTURED"):
            assert engine_a.synthesize_series(pv, ts) == engine_b.synthesize_series(pv, ts)

    def test_overlapping_windows_agree_pointwise(self, machine_file):
        """FR2: overlapping windows agree exactly on shared timestamps.

        Covers texture and both noise kinds; fails if any draw or the texture is
        evaluated on window-relative rather than absolute time.
        """
        engine = SimulationEngine.from_file(machine_file)
        window_a = _abs_window(offset_s=0, n=600)
        window_b = _abs_window(offset_s=300, n=600)
        for pv in ("T:NOISY", "T:ZERO:NOISY", "T:ZERO:TEXTURED"):
            series_a = engine.synthesize_series(pv, window_a)
            series_b = engine.synthesize_series(pv, window_b)
            assert series_a[300:600] == series_b[0:300]

    def test_jittered_windows_agree_on_shared_timestamps(self, machine_file):
        """FR2: window start, length, and sampling stride don't perturb shared samples."""
        engine = SimulationEngine.from_file(machine_file)
        full = engine.synthesize_series("T:ZERO:TEXTURED", _abs_window(n=600))
        shifted = engine.synthesize_series("T:ZERO:TEXTURED", _abs_window(offset_s=37, n=100))
        assert shifted == full[37:137]
        strided = engine.synthesize_series("T:ZERO:TEXTURED", _abs_window(n=100, step_s=5))
        assert strided == full[0:500:5]

    def test_relative_and_absolute_noise_are_independent_streams(
        self, machine_dict, make_machine_file
    ):
        """The two noise draws compose in quadrature, proving distinct subkeys.

        With value=10, noise=0.1, noise_abs=1.0 the independent-stream std is
        sqrt((10*0.1)^2 + 1^2) ~= 1.414; a shared stream would add the sigmas
        coherently and read ~2.0, well outside the asserted band.
        """
        machine_dict["channels"]["T:BOTH"] = {
            "value": 10.0,
            "noise": 0.1,
            "noise_abs": 1.0,
            "description": "Composed relative + absolute noise",
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        series = np.array(engine.synthesize_series("T:BOTH", _abs_window(n=2000)))
        assert series.mean() == pytest.approx(10.0, abs=0.15)
        assert series.std() == pytest.approx(np.sqrt(2.0), rel=0.12)

    def test_texture_moves_the_baseline(self, machine_dict, make_machine_file):
        """Anti-degenerate: texture measurably changes and varies the series, bounded."""
        machine_dict["channels"]["T:TEX"] = {
            "value": 5.0,
            "noise": 0.0,
            "texture": {"kind": "wander", "amplitude": 1.0, "period_s": 3600.0},
            "description": "Pure texture channel",
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        series = np.array(engine.synthesize_series("T:TEX", _abs_window(n=600)))
        offsets = series - 5.0
        assert np.max(np.abs(offsets)) > 0.01  # differs from the texture-free constant
        assert np.ptp(offsets) > 0.05  # and actually moves within the window
        assert np.max(np.abs(offsets)) <= 1.0 + 1e-9  # |wander| <= amplitude

    def test_step_plateau_carries_texture_and_noise(self, machine_dict, make_machine_file):
        """FR6: texture and noise ride the post-step plateau (events applied first).

        The textured/texture-free twin comparison isolates the texture
        contribution (same channel name, hence identical noise draws): applying
        texture before the step would let the overwrite erase it and collapse
        the plateau difference to zero.
        """
        machine_dict["channels"]["T:STEPPED"] = {
            "value": 42.0,
            "noise": 0.0,
            "noise_abs": 0.05,
            "texture": {"kind": "wander", "amplitude": 1.0, "period_s": 3600.0},
            "description": "Stepped channel with texture and absolute noise",
        }
        machine_dict["scenarios"]["stepped"] = {
            "description": "Step down mid-window.",
            "archiver": [
                {
                    "channel": "T:STEPPED",
                    "events": [{"shape": "step", "at": 0.35, "to": 28.4}],
                }
            ],
        }
        textured_file = make_machine_file(machine_dict, "textured.json")
        plain_dict = copy.deepcopy(machine_dict)
        del plain_dict["channels"]["T:STEPPED"]["texture"]
        plain_file = make_machine_file(plain_dict, "plain.json")

        ts = _abs_window(n=600)
        t = np.linspace(0, 1, 600)
        engine = SimulationEngine.from_file(textured_file)
        engine.set_active_scenario("stepped")
        textured = np.array(engine.synthesize_series("T:STEPPED", ts))
        plain_engine = SimulationEngine.from_file(plain_file)
        plain_engine.set_active_scenario("stepped")
        plain = np.array(plain_engine.synthesize_series("T:STEPPED", ts))

        # Noise rides the plateau: the texture-free twin shows the declared
        # noise_abs sigma there — and nothing more (texture-scale motion would
        # blow the upper band).
        plateau_noise = (plain - 28.4)[t >= 0.4]
        assert 0.5 * 0.05 < plateau_noise.std() < 1.5 * 0.05
        # Texture rides the plateau: the twin diff isolates it from the noise.
        texture_contribution = (textured - plain)[t >= 0.4]
        assert np.ptp(texture_contribution) > 0.05  # survives the step overwrite
        assert np.max(np.abs(texture_contribution)) <= 1.0 + 1e-9

    def test_at_time_recurrence_unaffected_by_texture(self, machine_dict, make_machine_file):
        """FR6: daily ``at_time`` events still fire every date on a textured channel."""
        machine_dict["channels"]["T:VAC"]["texture"] = {
            "kind": "wander",
            "amplitude": 1.0e-9,
            "period_s": 3600.0,
        }
        machine_dict["scenarios"]["burst"] = {
            "description": "Daily vacuum burst.",
            "archiver": [
                {
                    "channel": "T:VAC",
                    "events": [
                        {"shape": "spike", "at_time": "14:32:08", "amplitude": 1.5e-7, "width": 600}
                    ],
                }
            ],
        }
        engine = SimulationEngine.from_file(make_machine_file(machine_dict))
        engine.set_active_scenario("burst")
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        ts = [start + timedelta(minutes=10 * i) for i in range(288)]  # 2 days @ 10 min
        series = engine.synthesize_series("T:VAC", ts)
        peaks = [i for i in range(1, 287) if series[i] > 1.0e-7]
        assert {ts[i].date() for i in peaks} == {ts[0].date(), ts[-1].date()}

    def test_non_epoch_timestamps_fall_back_deterministically(
        self, machine_dict, make_machine_file
    ):
        """t_abs=None: noise falls back to sample-index counters; texture is skipped."""
        opaque = ["not-a-timestamp"] * 200
        textured_file = make_machine_file(machine_dict, "textured.json")
        plain_dict = copy.deepcopy(machine_dict)
        del plain_dict["channels"]["T:ZERO:TEXTURED"]["texture"]
        plain_file = make_machine_file(plain_dict, "plain.json")

        engine = SimulationEngine.from_file(textured_file)
        noisy = np.array(engine.synthesize_series("T:ZERO:NOISY", opaque))
        assert noisy.std() > 0  # the fallback still draws real noise
        assert engine.synthesize_series("T:ZERO:NOISY", opaque) == noisy.tolist()

        # Texture needs absolute time; without it the textured channel matches
        # its texture-free twin exactly (identical fallback noise draws).
        textured = engine.synthesize_series("T:ZERO:TEXTURED", opaque)
        plain_engine = SimulationEngine.from_file(plain_file)
        assert textured == plain_engine.synthesize_series("T:ZERO:TEXTURED", opaque)


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
        day = datetime.now(UTC).replace(hour=14, minute=30, second=0, microsecond=0)
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
        day = datetime.now(UTC).replace(hour=9, minute=0, second=0, microsecond=0)
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
        start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
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
        day = datetime.now(UTC).replace(hour=14, minute=30, second=0, microsecond=0)
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
        day = datetime.now(UTC).replace(hour=14, minute=30, second=0, microsecond=0)
        ts = [day + timedelta(seconds=i) for i in range(600)]
        series = engine.synthesize_series("T:MODE", ts)
        for value, t in zip(series, ts, strict=True):
            expected = "FAULT" if (t - day).total_seconds() >= 128 else "CW"
            assert value == expected
