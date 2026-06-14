"""Composition contract: disjoint scenarios stack; overlapping ones are rejected.

Two simultaneously active scenarios must touch disjoint channel sets — archiver
step/ramp events overwrite the synthesized series (and overrides collide on
point reads), so overlapping scenarios would compose order-dependently and
silently wrong. These tests pin both halves: (1) the disjoint target combo
(vacuum-burst + rf-thermal) yields *both* fault signatures at once, and
(2) ``validate_composition`` hard-errors on a hand-built same-channel collision.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from osprey.simulation import SimulationEngine

GAUGE07 = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
C1_TEMP = "SR:RF:CAVITY:01:TEMPERATURE:RB"
RF_SERIES_N = 2016  # 7-day window at 5-minute resolution


def _yesterday_1432_window(minutes: int = 10) -> list[datetime]:
    """Per-second window straddling yesterday 14:32:08 (the vacuum at_time anchor)."""
    center = (datetime.now() - timedelta(days=1)).replace(
        hour=14, minute=32, second=8, microsecond=0
    )
    start = center - timedelta(minutes=minutes / 2)
    return [start + timedelta(seconds=i) for i in range(minutes * 60)]


def _seven_day_window() -> list[datetime]:
    now = datetime.now()
    step = timedelta(days=7) / RF_SERIES_N
    return [now - timedelta(days=7) + step * i for i in range(RF_SERIES_N)]


def _vacuum_spike_present(engine: SimulationEngine) -> bool:
    series = np.array(engine.synthesize_series(GAUGE07, _yesterday_1432_window()))
    # Baseline SR07 pressure is ~5e-8; the burst spikes it well above baseline.
    return series.max() > 2.0 * np.median(series)


def _rf_excursion_present(engine: SimulationEngine) -> bool:
    series = np.array(engine.synthesize_series(C1_TEMP, _seven_day_window()))
    # Nominal cavity body temp ~27 degC; excursions exceed 31 degC.
    return series.max() > 31.0


class TestDisjointComposition:
    def test_nominal_alone_shows_neither_fault(self, engine_factory):
        engine = engine_factory("nominal")
        assert not _vacuum_spike_present(engine)
        assert not _rf_excursion_present(engine)

    def test_composed_pair_shows_both_faults(self, engine_factory):
        engine = engine_factory("vacuum-burst", "rf-thermal")
        assert engine.active_scenarios() == ("nominal", "vacuum-burst", "rf-thermal")
        # Both overlays are live at once on their disjoint channels.
        assert _vacuum_spike_present(engine), "vacuum burst signature missing under composition"
        assert _rf_excursion_present(engine), "rf-thermal signature missing under composition"

    def test_validate_composition_accepts_disjoint_trio(self, engine_factory):
        engine = engine_factory("nominal")
        assert engine.validate_composition(["nominal", "vacuum-burst", "rf-thermal"]) == []


class TestCollisionRejected:
    """A hand-built machine where two scenarios touch one channel must be rejected."""

    COLLIDING = {
        "name": "collide",
        "channels": {"C": {"value": 1.0}, "D": {"value": 2.0}},
        "scenarios": {
            "nominal": {"description": "n"},
            "over-a": {"description": "a", "overrides": {"C": 5.0}},
            "over-b": {"description": "b", "overrides": {"C": 9.0}},
            "arch-c": {
                "description": "c",
                "archiver": [{"channel": "C", "events": [{"shape": "step", "at": 0.5, "to": 7.0}]}],
            },
            "disjoint-d": {"description": "d", "overrides": {"D": 3.0}},
        },
    }

    def _engine(self, make_machine_file) -> SimulationEngine:
        return SimulationEngine.from_file(make_machine_file(self.COLLIDING))

    def test_two_overrides_on_one_channel_collide(self, make_machine_file):
        engine = self._engine(make_machine_file)
        problems = engine.validate_composition(["over-a", "over-b"])
        assert len(problems) == 1
        assert "'C'" in problems[0]
        assert "over-a" in problems[0] and "over-b" in problems[0]

    def test_override_and_archiver_on_one_channel_collide(self, make_machine_file):
        engine = self._engine(make_machine_file)
        assert engine.validate_composition(["over-a", "arch-c"]) != []

    def test_disjoint_scenarios_do_not_collide(self, make_machine_file):
        engine = self._engine(make_machine_file)
        assert engine.validate_composition(["over-a", "disjoint-d"]) == []

    def test_set_active_scenarios_raises_on_collision(self, make_machine_file):
        engine = self._engine(make_machine_file)
        with pytest.raises(ValueError, match="disjoint channel sets"):
            engine.set_active_scenarios(["over-a", "over-b"])

    def test_unknown_scenario_reported(self, make_machine_file):
        engine = self._engine(make_machine_file)
        problems = engine.validate_composition(["no-such-scenario"])
        assert len(problems) == 1
        assert "Unknown scenario" in problems[0]
