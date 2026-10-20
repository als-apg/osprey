"""The acceptance thresholds land on the sample a panel plots.

``tests/e2e/trajectory_acceptance/sample_index.py`` turns the calibrated
windows -- frozen as turns after the trigger in the injection-window tests --
into the sample index the Raw BPM plots show ("Index of turn"), using the
trigger sample the example facility declares. The walk's rubric and the
deployed e2e both read their thresholds from it, so these tests pin the
conversion to the windows module's own, the trigger sample to the
declaration, and the exported thresholds to the frozen constants.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.trajectory_acceptance import sample_index, shape
from tests.simulation import test_pyat_txt_injection_windows as windows

REPO = Path(__file__).resolve().parents[2]


# --- the conversion ---------------------------------------------------------------


@pytest.mark.parametrize("turn", [1, 2, 110, 210, 230, 250])
def test_a_turn_lands_where_the_windows_module_places_it(turn: int) -> None:
    assert sample_index.plotted_index(turn, windows.TRIGGER_SAMPLE) == windows.plotted_index(turn)


def test_turn_one_is_the_trigger_sample() -> None:
    assert sample_index.plotted_index(1, 7) == 7


@pytest.mark.parametrize("turn", [0, -3])
def test_a_turn_before_the_first_has_no_plotted_index(turn: int) -> None:
    with pytest.raises(ValueError, match="turn"):
        sample_index.plotted_index(turn, 10)


def test_a_negative_trigger_sample_is_refused() -> None:
    with pytest.raises(ValueError, match="trigger"):
        sample_index.plotted_index(1, -1)


# --- the declared trigger sample ---------------------------------------------------


def test_the_example_declares_the_trigger_sample_the_windows_froze() -> None:
    assert sample_index.trigger_sample() == windows.TRIGGER_SAMPLE


def test_an_unknown_model_has_no_trigger_sample() -> None:
    with pytest.raises(KeyError, match="NOPE"):
        sample_index.trigger_sample(model="NOPE")


# --- the thresholds ----------------------------------------------------------------


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return sample_index.thresholds()


def test_the_thresholds_carry_the_declared_trigger_and_the_frozen_counts(thresholds) -> None:
    assert thresholds["model"] == windows.MODEL
    assert thresholds["trigger_sample"] == windows.TRIGGER_SAMPLE
    assert thresholds["n_default"] == windows.N_DEFAULT
    assert thresholds["turns_tracked"] == windows.RF_ON_TURNS
    assert thresholds["n_shots_max"] == shape.N_SHOTS_MAX
    assert thresholds["pre_trigger_indices"] == [0, windows.TRIGGER_SAMPLE - 1]
    assert thresholds["first_turn_index"] == windows.plotted_index(1)
    assert thresholds["last_index"] == windows.plotted_index(windows.RF_ON_TURNS)


def test_the_rf_off_window_is_the_frozen_one_at_its_plotted_index(thresholds) -> None:
    off = thresholds["rf_off"]
    assert off["high"] == {
        "turn": windows.RF_OFF_HIGH_TURN,
        "index": windows.plotted_index(windows.RF_OFF_HIGH_TURN),
        "share_above": windows.RF_OFF_HIGH_SURVIVAL,
    }
    assert off["low"] == {
        "turn": windows.RF_OFF_LOW_TURN,
        "index": windows.plotted_index(windows.RF_OFF_LOW_TURN),
        "share_below": windows.RF_OFF_LOW_SURVIVAL,
    }
    assert off["decoherence"] == {
        "early_turns": list(windows.EARLY_TURNS),
        "late_turns": list(windows.LATE_TURNS),
        "early_indices": [windows.plotted_index(t) for t in windows.EARLY_TURNS],
        "late_indices": [windows.plotted_index(t) for t in windows.LATE_TURNS],
        "ratio_max": windows.DECOHERENCE_MAX,
    }


def test_the_rf_on_window_holds_to_the_last_tracked_turn(thresholds) -> None:
    assert thresholds["rf_on"] == {
        "first_index": windows.plotted_index(1),
        "last_turn": windows.RF_ON_TURNS,
        "last_index": windows.plotted_index(windows.RF_ON_TURNS),
        "share_at_least": windows.RF_ON_SURVIVAL_MIN,
    }


def test_the_stored_share_is_the_calibrated_one_with_its_margin(thresholds) -> None:
    assert thresholds["stored"] == {
        "share": windows.STORED_SHARE,
        "margin": windows.STORED_SHARE_MARGIN,
        "threshold": windows.STORED_SHARE_THRESHOLD,
    }
    assert windows.STORED_SHARE_THRESHOLD < windows.STORED_SHARE


def test_another_trigger_sample_moves_every_index_and_nothing_else(thresholds) -> None:
    moved = sample_index.thresholds(trigger=3)
    shift = 3 - windows.TRIGGER_SAMPLE
    assert moved["trigger_sample"] == 3
    assert moved["rf_off"]["low"]["index"] == thresholds["rf_off"]["low"]["index"] + shift
    assert moved["rf_off"]["high"]["index"] == thresholds["rf_off"]["high"]["index"] + shift
    assert moved["rf_on"]["last_index"] == thresholds["rf_on"]["last_index"] + shift
    assert moved["stored"] == thresholds["stored"]
    assert moved["rf_off"]["low"]["share_below"] == thresholds["rf_off"]["low"]["share_below"]


def test_the_thresholds_are_plain_json(thresholds) -> None:
    assert json.loads(json.dumps(thresholds)) == thresholds


def test_the_module_prints_the_thresholds_as_json(thresholds) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "tests.e2e.trajectory_acceptance.sample_index"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == thresholds


# --- the readings ------------------------------------------------------------------


def _series(pre: float, post: list[float], trigger: int = 4) -> list[float]:
    return [pre] * trigger + post


def test_the_share_at_a_turn_is_over_the_first_turns_reading() -> None:
    series = _series(0.0, [2.0, 2.0, 1.5, 1.0])
    assert sample_index.share_at(series, 4, turn=1) == 1.0
    assert sample_index.share_at(series, 4, turn=3) == 0.75
    assert sample_index.share_at(series, 4, turn=4) == 0.5


def test_the_share_at_a_turn_past_the_series_is_refused() -> None:
    with pytest.raises(IndexError, match="turn 5"):
        sample_index.share_at(_series(0.0, [1.0] * 4), 4, turn=5)


def test_the_stored_ratio_is_the_pre_trigger_mean_over_the_post_injection_peak() -> None:
    series = _series(0.5, [1.5, 1.4, 1.2])
    assert sample_index.stored_ratio(series, 4) == pytest.approx(0.5 / 1.5)


def test_an_empty_ring_reads_a_stored_ratio_far_below_the_threshold() -> None:
    floor = 1e-6
    series = _series(floor, [1.0, 1.0])
    assert sample_index.stored_ratio(series, 4) < windows.STORED_SHARE_THRESHOLD / 2


def test_a_series_without_a_post_injection_sample_has_no_stored_ratio() -> None:
    with pytest.raises(ValueError, match="post-injection"):
        sample_index.stored_ratio([1.0] * 4, 4)
