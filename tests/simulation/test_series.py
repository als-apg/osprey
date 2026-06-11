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
