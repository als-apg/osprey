"""Contiguous first-turn extraction and shot averaging (pure functions)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from osprey.services.trajectory.extraction import (
    PRE_WINDOW_SAMPLES,
    ExtractionFailure,
    ShotExtraction,
    average_shots,
    extract_shot,
)

N_BPM = 40
N_SAMPLES = 60
I0 = 10
CUTOFF = 0.5
THRESHOLD = 0.5
WINDOW_TURNS = 20


def _shot(lost_at: int | None, *, i0: int = I0, garbage: bool = True, seed: int = 0):
    """A VA-like shot: sum 1 from ``i0`` on at every BPM before ``lost_at``.

    Downstream of the loss the sum stays flat (no rise) and, with ``garbage``,
    x and y hold finite nonsense a real BPM would read with no beam.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1e-3, (N_BPM, N_SAMPLES))
    y = rng.normal(0.0, 1e-3, (N_BPM, N_SAMPLES))
    total = np.zeros((N_BPM, N_SAMPLES))
    stop = N_BPM if lost_at is None else lost_at
    total[:stop, i0:] = 1.0
    if garbage:
        x[stop:] = 123.0
        y[stop:] = -456.0
    else:
        x[stop:] = np.nan
        y[stop:] = np.nan
    return x, y, total


def _extract(x, y, total, **overrides):
    kwargs = {
        "loss_sum_cutoff": CUTOFF,
        "shot_sum_threshold": THRESHOLD,
        "raw_window_turns": WINDOW_TURNS,
    }
    kwargs.update(overrides)
    return extract_shot(x, y, total, **kwargs)


# --- one shot --------------------------------------------------------------------------


def test_a_full_turn_has_no_loss_and_reads_sample_i0():
    x, y, total = _shot(None)
    result = _extract(x, y, total)
    assert isinstance(result, ShotExtraction)
    assert result.lost_at is None
    assert result.last_bpm == N_BPM - 1
    assert result.i0 == I0
    assert result.window == (I0 - PRE_WINDOW_SAMPLES, I0 + WINDOW_TURNS)
    np.testing.assert_array_equal(result.x, x[:, I0])
    np.testing.assert_array_equal(result.y, y[:, I0])
    np.testing.assert_array_equal(result.sum, total[:, I0])


def test_finite_garbage_downstream_of_a_loss_comes_out_nan():
    x, y, total = _shot(30, garbage=True)
    assert np.isfinite(x[30:]).all()
    result = _extract(x, y, total)
    assert isinstance(result, ShotExtraction)
    assert result.lost_at == 30
    assert result.last_bpm == 29
    assert np.isnan(result.x[30:]).all()
    assert np.isnan(result.y[30:]).all()
    assert np.isfinite(result.x[:30]).all()
    assert np.isfinite(result.y[:30]).all()
    assert (result.sum[30:] == 0.0).all()


def test_the_loss_mask_is_contiguous_past_a_bpm_that_rises_again():
    x, y, total = _shot(None)
    total[12, :] = 0.0  # BPM 12 never rises; 13.. rise again
    result = _extract(x, y, total)
    assert result.lost_at == 12
    assert np.isnan(result.x[12:]).all()
    assert np.isfinite(result.x[:12]).all()


def test_i0_is_the_median_over_the_risers_before_the_loss():
    x, y, total = _shot(None)
    total[:] = 0.0
    starts = [9, 10, 11, 10, 30]
    for bpm, start in enumerate(starts):
        total[bpm, start:] = 1.0
    # BPM 5 never rises: the late BPM 30 start before it still votes, those after do not.
    result = _extract(x, y, total)
    assert result.lost_at == len(starts)
    assert result.i0 == 10
    np.testing.assert_array_equal(result.x[: len(starts)], x[: len(starts), 10])


def test_a_rise_must_exceed_the_cutoff():
    x, y, total = _shot(None)
    total[3:] = 0.0
    total[3:, I0:] = CUTOFF  # equal to the cutoff: not a rise
    result = _extract(x, y, total)
    assert result.lost_at == 3


def test_nan_sums_never_count_as_a_rise():
    x, y, total = _shot(None)
    total[7, :] = np.nan
    result = _extract(x, y, total)
    assert result.lost_at == 7


def test_the_window_is_clipped_to_the_record():
    x, y, total = _shot(None, i0=3)
    result = _extract(x, y, total, raw_window_turns=500)
    assert result.window == (0, N_SAMPLES)


def test_a_shot_where_no_bpm_rises_fails():
    x, y, total = _shot(None)
    total[:] = 0.0
    result = _extract(x, y, total)
    assert isinstance(result, ExtractionFailure)
    assert "no BPM rose" in result.reason


def test_a_shot_whose_first_bpm_does_not_rise_fails():
    x, y, total = _shot(None)
    total[0, :] = 0.0
    result = _extract(x, y, total)
    assert isinstance(result, ExtractionFailure)
    assert "no BPM rose" in result.reason


def test_a_first_turn_sum_below_the_threshold_fails_with_its_level():
    x, y, total = _shot(None)
    total[:, :] = 0.0
    total[:, I0:] = 0.3  # rises by 0.3 > cutoff 0.2, but below threshold 0.5
    result = _extract(x, y, total, loss_sum_cutoff=0.2)
    assert isinstance(result, ExtractionFailure)
    assert result.sum_level == pytest.approx(0.3)
    assert "shot_sum_threshold" in result.reason
    assert "0.3" in result.reason


def test_the_threshold_reads_the_first_bpm_only():
    x, y, total = _shot(None)
    total[1:, I0:] = 0.6  # downstream lower; still rises and still above the cutoff
    total[0, I0:] = 2.0
    result = _extract(x, y, total, loss_sum_cutoff=0.5, shot_sum_threshold=1.5)
    assert isinstance(result, ShotExtraction)


@pytest.mark.parametrize(
    ("shape_x", "shape_sum"),
    [
        ((N_BPM, N_SAMPLES), (N_BPM, N_SAMPLES - 1)),
        ((N_SAMPLES,), (N_SAMPLES,)),
        ((0, 5), (0, 5)),
        ((3, 1), (3, 1)),
    ],
)
def test_misshapen_records_are_refused(shape_x, shape_sum):
    with pytest.raises(ValueError):
        _extract(np.zeros(shape_x), np.zeros(shape_x), np.zeros(shape_sum))


def test_negative_window_turns_are_refused():
    x, y, total = _shot(None)
    with pytest.raises(ValueError):
        _extract(x, y, total, raw_window_turns=-1)


def test_extraction_does_not_mutate_its_inputs():
    x, y, total = _shot(30)
    copies = (x.copy(), y.copy(), total.copy())
    _extract(x, y, total)
    for before, after in zip(copies, (x, y, total), strict=True):
        np.testing.assert_array_equal(before, after)


# --- averaging -------------------------------------------------------------------------


def _included(lost_at, seed):
    result = _extract(*_shot(lost_at, seed=seed))
    assert isinstance(result, ShotExtraction)
    return result


def test_shots_lost_at_30_and_33_average_to_the_earliest_loss():
    first, second = _included(30, 1), _included(33, 2)
    averaged = average_shots([first, second])
    assert averaged.lost_at == 30
    assert averaged.last_bpm == 29
    assert averaged.per_shot_lost_at == [30, 33]
    assert averaged.shots_included == 2
    assert np.isnan(averaged.x[30:]).all()
    assert np.isnan(averaged.y[30:]).all()
    assert (averaged.sum[30:] == 0.0).all()
    np.testing.assert_allclose(averaged.x[:30], (first.x[:30] + second.x[:30]) / 2)
    np.testing.assert_allclose(averaged.y[:30], (first.y[:30] + second.y[:30]) / 2)


def test_nanmean_still_masks_the_earliest_loss():
    averaged = average_shots([_included(30, 1), _included(33, 2)], average="nanmean")
    assert np.isnan(averaged.x[30:]).all()
    assert np.isfinite(averaged.x[:30]).all()


def test_a_full_turn_shot_keeps_none_as_its_lost_at():
    averaged = average_shots([_included(None, 1), _included(None, 2)])
    assert averaged.lost_at is None
    assert averaged.last_bpm == N_BPM - 1
    assert averaged.per_shot_lost_at == [None, None]
    assert np.isfinite(averaged.x).all()


def test_one_full_turn_and_one_lost_shot_take_the_loss():
    averaged = average_shots([_included(None, 1), _included(35, 2)])
    assert averaged.lost_at == 35
    assert averaged.per_shot_lost_at == [None, 35]


def test_averaged_i0_and_window_come_from_the_median_shot():
    shots = [_extract(*_shot(None, i0=start, seed=start)) for start in (8, 12, 10)]
    averaged = average_shots(shots)
    assert averaged.i0 == 10
    assert averaged.window == (10 - PRE_WINDOW_SAMPLES, 10 + WINDOW_TURNS)


def test_averaging_nothing_is_refused():
    with pytest.raises(ValueError):
        average_shots([])


def test_an_unknown_average_mode_is_refused():
    with pytest.raises(ValueError):
        average_shots([_included(None, 1)], average="median")


def test_shots_of_different_bpm_counts_are_refused():
    short = _extract(*(a[:-1] for a in _shot(None)))
    with pytest.raises(ValueError):
        average_shots([_included(None, 1), short])


def test_a_single_shot_averages_to_itself():
    shot = _included(30, 4)
    averaged = average_shots([shot])
    np.testing.assert_array_equal(averaged.x[:30], shot.x[:30])
    assert math.isnan(averaged.x[30])
