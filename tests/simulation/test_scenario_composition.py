"""Composition contract: disjoint scenarios stack; overlapping ones are rejected.

Two simultaneously active scenarios must touch disjoint channel sets — archiver
step/ramp events overwrite the synthesized series (and overrides collide on
point reads), so overlapping scenarios would compose order-dependently and
silently wrong. These tests pin both halves: (1) the disjoint target combo
(vacuum-burst + rf-thermal) yields *both* fault signatures at once, and
(2) ``validate_composition`` hard-errors on a hand-built same-channel collision.
``osprey sim apply`` judges a requested set on the build's simulator view and
stops before any write when two of its scenarios write one target.
"""

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.sim import sim_group
from osprey.simulation import SimulationEngine
from tests._simulator_view import write_scenarios_view
from tests.cli._lifecycle_build import stub_build
from tests.fixtures.lifecycle_repo import build_exemplar_repo

GAUGE07 = "SR:VAC:GAUGE:SR07:PRESSURE:RB"
CAVITY01_TEMP = "SR:RF:CAVITY:01:TEMPERATURE:RB"
RF_SERIES_N = 2016  # 7-day window at 5-minute resolution


def _yesterday_1432_window(minutes: int = 10) -> list[datetime]:
    """Per-second window straddling yesterday 14:32:08 (the vacuum at_time anchor)."""
    center = (datetime.now(UTC) - timedelta(days=1)).replace(
        hour=14, minute=32, second=8, microsecond=0
    )
    start = center - timedelta(minutes=minutes / 2)
    return [start + timedelta(seconds=i) for i in range(minutes * 60)]


def _seven_day_window() -> list[datetime]:
    """A 7-day window ending now.

    rf-thermal's excursions are anchored (``at_when``, days and clock times
    before the scenario-activation anchor T0), not window fractions, so this
    window must *end at now* for them to appear in it: the fixture writes
    ``active_scenarios`` as it builds the engine, which puts T0 within a second
    of now, and the trip four days back always falls inside the week.
    """
    now = datetime.now()
    step = timedelta(days=7) / RF_SERIES_N
    return [now - timedelta(days=7) + step * i for i in range(RF_SERIES_N)]


def _vacuum_spike_present(engine: SimulationEngine) -> bool:
    series = np.array(engine.synthesize_series(GAUGE07, _yesterday_1432_window()))
    # Baseline SR07 pressure is ~5e-8; the burst spikes it well above baseline.
    return series.max() > 2.0 * np.median(series)


def _rf_excursion_present(engine: SimulationEngine) -> bool:
    series = np.array(engine.synthesize_series(CAVITY01_TEMP, _seven_day_window()))
    # Nominal cavity body temp ~27 degC; excursions exceed 31 degC.
    return series.max() > 31.0


class TestDisjointComposition:
    def test_nominal_alone_shows_neither_fault(self, engine_factory):
        engine = engine_factory("nominal")
        assert not _vacuum_spike_present(engine)
        assert not _rf_excursion_present(engine)


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


# -- `osprey sim apply` on the simulator view ----------------------------------

#: The one mock address the override scenarios both write.
MOCK_TARGET = "T:BPM1:X"
#: The one engine key the fault scenarios both write, under ``faults.SR.writes``.
VA_TARGET = "SR:DIAG:BPM:17:POSITION:X"
#: The one channel the archiver scenarios both script.
ARCHIVER_TARGET = "T:GAUGE:P"

#: The scenarios of the staged simulator view, in the view's own shape.
VIEW_SCENARIOS = {
    "over-a": {"description": "a", "overrides": {MOCK_TARGET: 1.0}},
    "over-b": {"description": "b", "overrides": {MOCK_TARGET: 2.0}},
    "fault-a": {"description": "fa", "faults": {"SR": {"writes": {VA_TARGET: {"polarity": -1}}}}},
    "fault-b": {"description": "fb", "faults": {"SR": {"writes": {VA_TARGET: {"polarity": 1}}}}},
    "arch-a": {
        "description": "aa",
        "archiver": [{"channel": ARCHIVER_TARGET, "events": [{"shape": "step", "to": 1.0}]}],
    },
    "arch-b": {
        "description": "ab",
        "archiver": [{"channel": ARCHIVER_TARGET, "events": [{"shape": "step", "to": 2.0}]}],
    },
    "quiet": {"description": "states no overrides and no faults"},
}


@pytest.fixture(autouse=True)
def _contain_env_written_by_the_cli():
    """Keep what ``sim apply`` loads into the environment inside the test that ran it."""
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


def _stage(tmp_path: Path, *, view: bool = True) -> Path:
    """A deployment repo whose render carries :data:`VIEW_SCENARIOS` as its view.

    The machine file lists the same names with no blocks, so only the view can
    tell two scenarios apart, and ``nominal`` so the engine accepts the set.
    """
    repo = build_exemplar_repo(tmp_path / "repo")
    sim_dir = repo / "data" / "simulation"
    shutil.rmtree(sim_dir, ignore_errors=True)
    sim_dir.mkdir(parents=True)
    machine = {"nominal": {}, **{name: {} for name in VIEW_SCENARIOS}}
    (sim_dir / "machine.json").write_text(
        json.dumps({"channels": {MOCK_TARGET: {"value": 0.0}}, "scenarios": machine})
    )
    config = {
        "control_system": {
            "connector": {"mock": {"simulation_file": "data/simulation/machine.json"}}
        },
    }
    build = stub_build(repo, config=yaml.safe_dump(config))
    if view:
        write_scenarios_view(build, VIEW_SCENARIOS)
    return repo


def _apply(repo: Path, *names: str):
    """Run ``sim apply NAMES --no-seed`` with ``apply_scenarios`` stood in."""
    applied = SimpleNamespace(active=["nominal", *names], logbook_seeded=0, archiver=None)
    with patch("osprey.simulation.apply.apply_scenarios", return_value=applied) as apply_mock:
        result = CliRunner().invoke(
            sim_group, ["apply", "--repo", str(repo), *names, "--no-seed", "--yes"]
        )
    return result, apply_mock


def _wrote_nothing(repo: Path, apply_mock) -> bool:
    return not apply_mock.called and not (repo / ".env").is_file()


class TestApplyStopsOnTheView:
    """Two requested scenarios writing one target stop ``sim apply`` before any write."""

    @pytest.mark.parametrize(
        ("first", "second", "target"),
        [
            pytest.param("over-a", "over-b", MOCK_TARGET, id="mock-address"),
            pytest.param("fault-a", "fault-b", VA_TARGET, id="va-fault-key"),
            pytest.param("arch-a", "arch-b", ARCHIVER_TARGET, id="archiver-channel"),
        ],
    )
    def test_a_shared_target_stops_naming_it(self, tmp_path, first, second, target):
        repo = _stage(tmp_path)

        result, apply_mock = _apply(repo, first, second)

        assert result.exit_code == 1, result.output
        assert "Cannot activate these scenarios" in result.output
        assert repr(target) in result.output
        assert repr(first) in result.output and repr(second) in result.output
        assert _wrote_nothing(repo, apply_mock)

    @pytest.mark.parametrize("partner", ["over-a", "fault-a", "arch-a"])
    def test_a_scenario_writing_nothing_composes_with_anything(self, tmp_path, partner):
        repo = _stage(tmp_path)

        result, apply_mock = _apply(repo, "quiet", partner)

        assert result.exit_code == 0, result.output
        assert apply_mock.call_args.args[1] == ["quiet", partner]

    def test_an_unknown_scenario_stops_naming_it(self, tmp_path):
        repo = _stage(tmp_path)

        result, apply_mock = _apply(repo, "no-such-scenario")

        assert result.exit_code == 1, result.output
        assert "Cannot activate these scenarios" in result.output
        assert "no-such-scenario" in result.output
        assert _wrote_nothing(repo, apply_mock)

    def test_nominal_is_accepted_when_the_view_does_not_list_it(self, tmp_path):
        repo = _stage(tmp_path)
        listed = json.loads((repo / "build" / "data" / "simulator" / "scenarios.json").read_text())
        assert "nominal" not in {entry["name"] for entry in listed["scenarios"]}

        result, apply_mock = _apply(repo, "nominal")

        assert result.exit_code == 0, result.output
        apply_mock.assert_called_once()

    def test_a_render_without_a_view_stops_naming_the_build(self, tmp_path):
        repo = _stage(tmp_path, view=False)

        result, apply_mock = _apply(repo, "over-a")

        assert result.exit_code == 1, result.output
        assert "No simulator view in" in result.output
        assert "run 'osprey build' first" in result.output
        assert _wrote_nothing(repo, apply_mock)
