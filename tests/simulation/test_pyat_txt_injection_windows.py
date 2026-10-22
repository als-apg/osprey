"""The trajectory-study injection's calibrated physics, frozen on the demo SR model.

The ``txt-injection`` scenario launches a bunch of N_default particles into the
demo storage ring, which already stores half a bunch of charge. With the RF
off the ring has no closed orbit: the bunch survives the first hundred turns,
is mostly lost by turn 210, and its horizontal centroid decoheres; the stored
beam shows on no monitor and decays shot by shot to nothing. With the RF on
the bunch survives the whole window and the stored beam reads on every sample
from the first, the sum before the trigger standing at the stored share of
the sum after it.

Every threshold is a constant of this module, copied from the calibration
that chose N_default; a window is stated in turns after the trigger (1-based)
and :func:`plotted_index` places a turn on the series a panel plots.

Each RF state is tracked once per run: one session fixture per state takes
shots 1-20 at the window the calibration measured and keeps every shot's
per-monitor survivor share and centroid. The constants above the fixtures
import no simulator, so other suites can read the frozen numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

# --- the frozen calibration --------------------------------------------------
# Copied once from the trajectory-walk calibration (CALIBRATION.json:
# bunch_particles_N_default, windows, results["16"]); a test never reads it.

#: The particles a shot tracks: N_default, the smallest count the windows hold at.
N_DEFAULT = 16
#: The samples a turn series holds before the trigger sample.
TRIGGER_SAMPLE = 10
#: The turns after the trigger the RF-on window tracks: the buffer the study sets.
RF_ON_TURNS = 250
#: The turns the RF-off window tracks: past the last turn it judges.
RF_OFF_TURNS = 230
#: The shots each RF state is judged on.
SHOTS = 20
#: The shots jitter replay and accumulation are judged on.
REPLAY_SHOTS = 3

#: RF off: the survivor share stays above this at the end of turn 110 ...
RF_OFF_HIGH_TURN, RF_OFF_HIGH_SURVIVAL = 110, 0.8
#: ... and has fallen below this by the end of turn 210.
RF_OFF_LOW_TURN, RF_OFF_LOW_SURVIVAL = 210, 0.5
#: The early and late turns the decoherence compares, inclusive.
EARLY_TURNS, LATE_TURNS = (1, 10), (90, 110)
#: RF off: the late envelope of the detrended horizontal survivor centroid over
#: the early one, at the worst monitor, stays at or below this.
DECOHERENCE_MAX = 0.5
#: RF on: the survivor share stays at or above this on every turn of the window.
RF_ON_SURVIVAL_MIN = 0.95

# --- the stored share, frozen here ---------------------------------------------

#: The charge the scenario stores, in bunches (``stored_charge``).
STORED_CHARGE = 0.5
#: The charge of one injected bunch: the scenario states none, so the default.
BUNCH_CHARGE = 1.0
#: The stored share ``Q_s / (Q_i + Q_s)``: the pre-trigger sum over the sum of a
#: sample the whole bunch reaches, on the first shot into the scenario's ring.
STORED_SHARE = STORED_CHARGE / (BUNCH_CHARGE + STORED_CHARGE)
#: The relative margin a pre-trigger/post-injection sum ratio is judged with:
#: the bunch only loses charge (the ratio only rises) and the sum carries no
#: noise, so the margin covers a deployment's sum readout and nothing else.
STORED_SHARE_MARGIN = 0.05
#: The least pre-trigger/post-injection sum ratio a ring storing the scenario's
#: charge reads with the RF on.
STORED_SHARE_THRESHOLD = STORED_SHARE * (1.0 - STORED_SHARE_MARGIN)


def plotted_index(turn: int) -> int:
    """The series sample holding ``turn`` after the trigger (1-based): turn 1 is the trigger sample."""
    return TRIGGER_SAMPLE + turn - 1


# --- the demo model ---------------------------------------------------------------

MODEL = "SR"
INJECTION_FILE = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/facilities/example/scenarios/txt-injection.yaml"
)
STORED_CURRENT = "SR:DIAG:STORED_BEAM:CURRENT:RB"
TURNS_SP = "SR:DIAG:TBT:TURNS:SP"
RF_ON_SP = "SR:RF:ON:SP"
CLEANING = "SR:DIAG:CLEANING:REQUEST"
ARM = "SR:DIAG:BPM:01:TBT:ARM"
REQUEST = "SR:DIAG:INJECTION:REQUEST"
#: The index of each label of the RF switch.
RF_OFF, RF_ON = 0, 1

pytestmark = pytest.mark.xdist_group("txt_injection_windows")


def injection_faults() -> dict[str, Any]:
    return yaml.safe_load(INJECTION_FILE.read_text(encoding="utf-8"))["faults"][MODEL]


@dataclass
class Shot:
    """One tracked shot: its reductions, its served sums and the stored charge around it."""

    #: Survivor share and centroid, each ``(monitors + 1, turns)``: one row per
    #: monitor in tracking order, then the end-of-turn row.
    fraction: np.ndarray
    centroid_x: np.ndarray
    centroid_y: np.ndarray
    #: Every served sum series, by address.
    sums: dict[str, np.ndarray]
    stored_before: float
    stored_after: float
    current_after: float


@dataclass
class Run:
    """Shots 1-n of one RF state into a freshly built model."""

    turns: int
    shots: list[Shot] = field(default_factory=list)


@pytest.fixture(scope="session")
def demo(built_control_assistant: Any) -> dict[str, Any]:
    """The SR model of the built demo: its wiring entries, deck and settings."""
    pytest.importorskip("at")
    from osprey.facility.views.simulator import simulator_wiring

    facility = built_control_assistant.facility
    (model,) = [entry for entry in facility["models"] if entry["name"] == MODEL]
    return {
        "wiring": simulator_wiring(facility, MODEL),
        "deck": built_control_assistant.facility_dir / model["deck"],
        "settings": model.get("settings"),
    }


def build(demo: dict[str, Any], active: dict[str, Any] | None = None) -> Any:
    from osprey.simulation.engines import pyat as engine

    return engine.build(MODEL, demo["wiring"], demo["deck"], demo["settings"], active or {})


def series_names(model: Any, *signals: str) -> list[str]:
    """The model's turn series of ``signals``, by address."""
    ends = tuple(f":TBT:{signal}" for signal in signals)
    return sorted(name for name in model.supported_variables if name.endswith(ends))


def shoot(model: Any, rf: int, turns: int, shots: int) -> Run:
    """Set the RF and the turns tracked, then take ``shots`` armed shots, keeping each one."""
    from osprey.simulation.engines import pyat_model

    model.set({RF_ON_SP: rf, TURNS_SP: turns})
    names = series_names(model, "SUM")
    readings: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    reading = pyat_model._bunch_reading

    def keep(*args: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reduced = reading(*args)
        readings.append(tuple(np.array(part, copy=True) for part in reduced))
        return reduced

    run = Run(turns)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pyat_model, "_bunch_reading", keep)
        for count in range(1, shots + 1):
            before = model.stored_charge
            model.set({ARM: 1, REQUEST: count})
            assert len(readings) == count, "an armed shot tracks its bunch exactly once"
            served = model.get([*names, STORED_CURRENT])
            run.shots.append(
                Shot(
                    *readings[-1],
                    sums={name: np.asarray(served[name], dtype=np.float64) for name in names},
                    stored_before=before,
                    stored_after=model.stored_charge,
                    current_after=float(served[STORED_CURRENT]),
                )
            )
    return run


@pytest.fixture(scope="session")
def rf_off(demo: dict[str, Any]) -> Run:
    """Shots 1-20 with the RF off, each tracked to turn 230."""
    return shoot(build(demo, injection_faults()), RF_OFF, RF_OFF_TURNS, SHOTS)


@pytest.fixture(scope="session")
def rf_on(demo: dict[str, Any]) -> Run:
    """Shots 1-20 with the RF on, each tracked over the full 250-turn window."""
    return shoot(build(demo, injection_faults()), RF_ON, RF_ON_TURNS, SHOTS)


def survival(shot: Shot, turn: int) -> float:
    """The share of the bunch alive at the end of ``turn`` (1-based)."""
    return float(shot.fraction[-1, turn - 1])


def envelope(x: np.ndarray, turns: tuple[int, int]) -> float:
    """The largest distance of ``x`` from its linear fit over ``turns``, inclusive."""
    first, last = turns
    segment = x[first - 1 : last]
    axis = np.arange(segment.size, dtype=np.float64)
    slope, offset = np.polyfit(axis, segment, 1)
    return float(np.max(np.abs(segment - (slope * axis + offset))))


def decoherence(shot: Shot) -> float:
    """The late envelope over the early one of the horizontal survivor centroid, worst monitor."""
    monitors = shot.centroid_x[:-1]
    return max(envelope(row, LATE_TURNS) / envelope(row, EARLY_TURNS) for row in monitors)


class TestTheScenario:
    def test_the_scenario_tracks_n_default_particles_into_a_stored_ring(self):
        faults = injection_faults()
        assert faults["bunch_particles"] == N_DEFAULT
        assert faults["stored_charge"] == STORED_CHARGE
        assert "bunch_charge" not in faults

    def test_a_turn_lands_on_the_sample_the_plotted_index_names(self, rf_off):
        from osprey.simulation.engines.pyat_faults import SUM_FLOOR

        shot = rf_off.shots[0]
        rows = [np.maximum(row, SUM_FLOOR) for row in shot.fraction[:-1]]
        for name, served in shot.sums.items():
            window = served[plotted_index(1) : plotted_index(rf_off.turns) + 1]
            assert any(np.array_equal(window, row) for row in rows), name
            np.testing.assert_allclose(served[: plotted_index(1)], SUM_FLOOR)
            np.testing.assert_allclose(served[plotted_index(rf_off.turns) + 1 :], SUM_FLOOR)


class TestRFOff:
    def test_the_bunch_survives_turn_110(self, rf_off):
        worst = min(survival(shot, RF_OFF_HIGH_TURN) for shot in rf_off.shots)
        assert worst > RF_OFF_HIGH_SURVIVAL

    def test_the_bunch_is_mostly_lost_by_turn_210(self, rf_off):
        worst = max(survival(shot, RF_OFF_LOW_TURN) for shot in rf_off.shots)
        assert worst < RF_OFF_LOW_SURVIVAL

    def test_the_horizontal_centroid_decoheres(self, rf_off):
        worst = max(decoherence(shot) for shot in rf_off.shots)
        assert worst <= DECOHERENCE_MAX

    def test_the_stored_beam_shows_on_no_sample_before_the_trigger(self, rf_off):
        from osprey.simulation.engines.pyat_faults import SUM_FLOOR

        shot = rf_off.shots[0]
        assert shot.stored_before == STORED_CHARGE
        for name, served in shot.sums.items():
            np.testing.assert_allclose(served[:TRIGGER_SAMPLE], SUM_FLOOR, err_msg=name)

    def test_no_shot_is_captured_and_the_stored_charge_decays(self, demo, rf_off):
        fraction = build(demo, injection_faults()).supported_variables[STORED_CURRENT]
        for shot in rf_off.shots:
            assert shot.stored_after == pytest.approx(
                shot.stored_before * fraction.surviving_fraction, rel=1e-12
            )


class TestRFOn:
    def test_the_bunch_survives_the_whole_window(self, rf_on):
        worst = min(float(shot.fraction[-1].min()) for shot in rf_on.shots)
        assert worst >= RF_ON_SURVIVAL_MIN
        assert all(shot.fraction.shape[1] == RF_ON_TURNS for shot in rf_on.shots)

    def test_the_stored_beam_reads_on_every_sample_before_the_trigger(self, rf_on):
        from osprey.simulation.engines.pyat_faults import SUM_FLOOR

        shot = rf_on.shots[0]
        for name, served in shot.sums.items():
            np.testing.assert_allclose(
                served[:TRIGGER_SAMPLE], STORED_CHARGE / BUNCH_CHARGE, err_msg=name
            )
            assert served[:TRIGGER_SAMPLE].min() > SUM_FLOOR

    def test_the_first_shot_reads_the_calibrated_stored_share(self, rf_on):
        shot = rf_on.shots[0]
        for name, served in shot.sums.items():
            ratio = served[:TRIGGER_SAMPLE].mean() / served[plotted_index(1)]
            assert ratio == pytest.approx(STORED_SHARE, rel=1e-9), name

    def test_every_shot_reads_at_least_the_stored_share_with_its_margin(self, rf_on):
        for count, shot in enumerate(rf_on.shots, start=1):
            for name, served in shot.sums.items():
                post = served[plotted_index(1) : plotted_index(rf_on.turns) + 1].max()
                ratio = served[:TRIGGER_SAMPLE].mean() / post
                assert ratio >= STORED_SHARE_THRESHOLD, (count, name, ratio)

    def test_an_empty_ring_reads_well_below_the_stored_share(self, rf_off):
        shot = rf_off.shots[0]
        for name, served in shot.sums.items():
            ratio = served[:TRIGGER_SAMPLE].mean() / served[plotted_index(1)]
            assert ratio < STORED_SHARE_THRESHOLD / 2, name

    def test_each_shot_adds_its_captured_share_to_the_stored_charge(self, rf_on):
        for shot in rf_on.shots[:REPLAY_SHOTS]:
            captured = float(shot.fraction[-1, -1])
            assert shot.stored_after == pytest.approx(
                shot.stored_before + captured * BUNCH_CHARGE, rel=1e-12
            )
        for earlier, later in zip(
            rf_on.shots[: REPLAY_SHOTS - 1], rf_on.shots[1:REPLAY_SHOTS], strict=True
        ):
            assert later.stored_before == earlier.stored_after
            assert later.current_after > earlier.current_after

    def test_the_jitter_replays_shot_for_shot(self, demo, rf_on):
        replay = shoot(build(demo, injection_faults()), RF_ON, RF_ON_TURNS, REPLAY_SHOTS)
        for first, again in zip(rf_on.shots[:REPLAY_SHOTS], replay.shots, strict=True):
            np.testing.assert_array_equal(again.fraction, first.fraction)
            np.testing.assert_array_equal(again.centroid_x, first.centroid_x)
            np.testing.assert_array_equal(again.centroid_y, first.centroid_y)
        one, two = rf_on.shots[0], rf_on.shots[1]
        assert not np.array_equal(one.centroid_x, two.centroid_x)


class TestTheStoredCurrent:
    def test_it_decays_by_the_declared_fraction_to_the_floor_with_the_rf_off(self, demo):
        model = build(demo, injection_faults())
        reading = model.supported_variables[STORED_CURRENT]
        fraction, floor = reading.surviving_fraction, reading.charge_floor
        assert 0.0 < fraction < 1.0
        assert floor > 0.0
        model.set({RF_ON_SP: RF_OFF})
        start = model.get([STORED_CURRENT])[STORED_CURRENT]
        assert start > 0.0
        # The shot whose surviving charge first falls to the floor leaves nothing.
        last = next(count for count in range(1, 1000) if STORED_CHARGE * fraction**count <= floor)
        for count in range(1, last + 3):
            model.set({REQUEST: count})
            current = model.get([STORED_CURRENT])[STORED_CURRENT]
            if count < last:
                assert current == pytest.approx(start * fraction**count, rel=1e-9), count
            else:
                assert current == 0.0, count

    def test_cleaning_empties_it(self, demo):
        model = build(demo, injection_faults())
        assert model.get([STORED_CURRENT])[STORED_CURRENT] > 0.0
        model.set({CLEANING: 1})
        assert model.stored_charge == 0.0
        assert model.get([STORED_CURRENT])[STORED_CURRENT] == 0.0


class TestTheWindow:
    def test_samples_past_a_100_turn_window_read_no_beam(self, demo):
        from osprey.simulation.engines.pyat_faults import SUM_FLOOR

        turns = 100
        model = build(demo, injection_faults())
        model.set({CLEANING: 1})
        run = shoot(model, RF_ON, turns, 1)
        for name, served in run.shots[0].sums.items():
            np.testing.assert_allclose(served[:TRIGGER_SAMPLE], SUM_FLOOR, err_msg=name)
            np.testing.assert_allclose(served[plotted_index(turns) + 1 :], SUM_FLOOR, err_msg=name)
            assert served[plotted_index(1) : plotted_index(turns) + 1].min() > SUM_FLOOR


class TestOneParticle:
    def test_one_particle_with_no_jitter_is_the_single_particle(self, demo):
        faults = injection_faults()
        plain = {
            name: value
            for name, value in faults.items()
            if name.startswith(("launch_", "aperture_"))
        }
        shape = {name: value for name, value in faults.items() if name.startswith("bunch_")}
        served = []
        for active in (plain, {**plain, **shape, "bunch_particles": 1}):
            model = build(demo, active)
            model.set({ARM: 1, REQUEST: 1})
            served.append(model.get(series_names(model, "X", "Y", "SUM")))
        for name, samples in served[0].items():
            np.testing.assert_array_equal(served[1][name], samples, err_msg=name)
