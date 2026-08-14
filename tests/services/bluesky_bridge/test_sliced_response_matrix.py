"""Unit tests for `orm_analysis.sliced_response_matrix`.

The vectorized, index-sliced ORM fit a plan's `render()` runs on every live
tick. Three things are actually load-bearing and get pinned here:

- it agrees with `numpy.polyfit` to 1e-12, so switching a fit onto it is not
  a change in answer, only in cost;
- it fits inputs `build_response_matrix` cannot take at all — a
  `monodirectional` sweep, and a run still in flight — because it slices by
  acquisition index instead of inferring corrector membership from the data;
- it is fast enough for the tick it runs on (a facility-scale shape here,
  well under the route's own budget).

Pure numpy, like the module under test: no bluesky, no bridge, no fixtures
beyond hand-built event rows.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from osprey.services.bluesky_bridge.orm_analysis import (
    DegenerateFitError,
    build_response_matrix,
    sliced_response_matrix,
)


def _sweep_currents(span_a: float, num: int, *, monodirectional: bool = False) -> list[float]:
    """The currents `plans_core/orm.py`'s `build_plan` sweeps, same arithmetic."""
    if monodirectional:
        step = span_a / (num - 1)
        return [i * step for i in range(num)]
    step = (2 * span_a) / (num - 1)
    return [-span_a + i * step for i in range(num)]


def _emit_rows(
    correctors: list[str],
    detectors: list[str],
    truth: np.ndarray,
    currents: list[float],
    *,
    offsets: np.ndarray | None = None,
    stop_after: int | None = None,
) -> list[dict[str, float]]:
    """Rows an `orm` run emits for a machine whose response is *truth*.

    Mirrors the real plan's document shape: one row per (corrector, current)
    point, in corrector-major emission order, every row carrying EVERY
    corrector's current (idle ones read back 0.0) alongside every detector.
    *offsets* gives each detector a nonzero intercept so a fit that ignored
    the intercept would visibly fail.
    """
    if offsets is None:
        offsets = np.zeros(len(detectors))
    rows: list[dict[str, float]] = []
    for j, corrector in enumerate(correctors):
        for current in currents:
            row = dict.fromkeys(correctors, 0.0)
            row[corrector] = current
            orbit = offsets + truth[:, j] * current
            for i, detector in enumerate(detectors):
                row[detector] = float(orbit[i])
            rows.append(row)
            if stop_after is not None and len(rows) >= stop_after:
                return rows
    return rows


def _truth_matrix(n_det: int, n_corr: int, seed: int = 7) -> np.ndarray:
    """A deterministic, non-degenerate `[n_det, n_corr]` response matrix."""
    return np.random.default_rng(seed).normal(size=(n_det, n_corr))


def test_matches_polyfit_within_1e_12_on_a_bidirectional_sweep():
    """The closed-form slope IS polyfit's answer, to 1e-12 relative."""
    correctors = ["CH1", "CH2", "CH3"]
    detectors = ["BPM1", "BPM2", "BPM3", "BPM4"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 7)
    rows = _emit_rows(
        correctors, detectors, truth, currents, offsets=np.array([1.0, -2.0, 0.5, 3.0])
    )

    fit = sliced_response_matrix(rows, correctors, detectors, num=7)

    assert fit.complete == (True, True, True)
    assert fit.fitted_correctors == tuple(correctors)
    assert fit.matrix.shape == (4, 3)

    # Independently polyfit each (detector, corrector) pair over that
    # corrector's own slice -- the reference the vectorized path replaces.
    for j in range(len(correctors)):
        block = rows[j * 7 : (j + 1) * 7]
        x = [row[correctors[j]] for row in block]
        for i, detector in enumerate(detectors):
            y = [row[detector] for row in block]
            expected, _intercept = np.polyfit(x, y, deg=1)
            assert fit.matrix[i, j] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    # ...and that reference is itself the machine we simulated.
    np.testing.assert_allclose(fit.matrix, truth, rtol=1e-9)


def test_fits_a_monodirectional_sweep_that_the_legacy_path_rejects():
    """Index slicing drops the zero-mean-sweep requirement entirely."""
    correctors = ["CH1", "CH2"]
    detectors = ["BPM1", "BPM2", "BPM3"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(1.5, 5, monodirectional=True)
    rows = _emit_rows(correctors, detectors, truth, currents)

    fit = sliced_response_matrix(rows, correctors, detectors, num=5)

    assert fit.complete == (True, True)
    np.testing.assert_allclose(fit.matrix, truth, rtol=1e-9)

    # The per-pair polyfit path cannot take this input at all: a one-sided
    # sweep leaves the idle-corrector rows with real leverage on the fit.
    with pytest.raises(DegenerateFitError, match="not symmetric about 0"):
        build_response_matrix(rows, correctors, detectors)


def test_partial_run_fits_finished_correctors_and_traces_the_one_in_flight():
    """A sweep still in progress costs its own column, not the matrix."""
    correctors = ["CH1", "CH2", "CH3"]
    detectors = ["BPM1", "BPM2"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 6)
    # Two correctors done (12 rows), the third three points into its sweep.
    rows = _emit_rows(correctors, detectors, truth, currents, stop_after=15)

    fit = sliced_response_matrix(rows, correctors, detectors, num=6)

    assert fit.complete == (True, True, False)
    assert fit.fitted_correctors == ("CH1", "CH2")
    assert fit.matrix.shape == (2, 2)
    np.testing.assert_allclose(fit.matrix, truth[:, :2], rtol=1e-9)

    # The in-flight corrector still yields its raw samples for a trace.
    assert fit.currents[2].shape == (3,)
    assert fit.readings[2].shape == (3, 2)
    np.testing.assert_allclose(fit.currents[2], currents[:3])
    # ...and the finished ones carry their full sweeps.
    assert fit.currents[0].shape == (6,)
    assert fit.readings[0].shape == (6, 2)
    np.testing.assert_allclose(fit.currents[0], currents)


def test_no_rows_yet_is_all_incomplete_and_an_empty_matrix():
    """The first tick of a run, before any event arrives, is not an error."""
    fit = sliced_response_matrix([], ["CH1", "CH2"], ["BPM1"], num=5)

    assert fit.complete == (False, False)
    assert fit.fitted_correctors == ()
    assert fit.matrix.shape == (1, 0)
    assert all(current.shape == (0,) for current in fit.currents)
    assert all(reading.shape == (0, 1) for reading in fit.readings)


def test_resolves_prefixed_signal_keys_once_per_run():
    """`name-signal` document keys match, same rules as `_match_value`."""
    correctors = ["CH1", "CH2"]
    detectors = ["BPM1", "BPM2"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(1.0, 4)
    plain = _emit_rows(correctors, detectors, truth, currents)
    # StandardReadable's multi-child spelling, plus unrelated keys that must
    # not be mistaken for a device (no `name-` prefix match).
    rows = [
        {f"{name}-readback": value for name, value in row.items()} | {"seq_num": 1, "CH": 0.0}
        for row in plain
    ]

    fit = sliced_response_matrix(rows, correctors, detectors, num=4)

    assert fit.complete == (True, True)
    np.testing.assert_allclose(fit.matrix, truth, rtol=1e-9)


def test_a_detector_that_never_reported_lands_on_zero_not_nan():
    """One dead BPM costs its own row, and never poisons the matrix."""
    correctors = ["CH1", "CH2"]
    detectors = ["BPM1", "BPM_DEAD", "BPM2"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 5)
    rows = _emit_rows(correctors, detectors, truth, currents)
    for row in rows:
        del row["BPM_DEAD"]

    fit = sliced_response_matrix(rows, correctors, detectors, num=5)

    assert fit.complete == (True, True)
    assert np.all(np.isfinite(fit.matrix))
    np.testing.assert_allclose(fit.matrix[1, :], [0.0, 0.0])
    np.testing.assert_allclose(fit.matrix[[0, 2], :], truth[[0, 2], :], rtol=1e-9)
    # The gap survives where a trace can show it.
    assert np.all(np.isnan(fit.readings[0][:, 1]))


def test_a_detector_dropping_out_mid_slice_costs_only_that_entry():
    """A single missing sample is a `nan` slope; it is clamped to `0.0`."""
    correctors = ["CH1"]
    detectors = ["BPM1", "BPM2"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 5)
    rows = _emit_rows(correctors, detectors, truth, currents)
    del rows[2]["BPM2"]

    fit = sliced_response_matrix(rows, correctors, detectors, num=5)

    assert fit.complete == (True,)
    assert fit.matrix[0, 0] == pytest.approx(truth[0, 0], rel=1e-9)
    assert fit.matrix[1, 0] == 0.0


def test_a_corrector_that_never_moved_yields_no_column():
    """A stuck corrector has no slope; it must not divide by zero."""
    correctors = ["CH1", "CH_STUCK"]
    detectors = ["BPM1", "BPM2"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 5)
    rows = _emit_rows(correctors, detectors, truth, currents)
    for row in rows[5:]:
        row["CH_STUCK"] = 0.0

    fit = sliced_response_matrix(rows, correctors, detectors, num=5)

    assert fit.complete == (True, False)
    assert fit.fitted_correctors == ("CH1",)
    assert fit.matrix.shape == (2, 1)
    assert np.all(np.isfinite(fit.matrix))


def test_a_corrector_missing_its_own_current_yields_no_column():
    """Without the driven current there is no x-axis to fit against."""
    correctors = ["CH1", "CH2"]
    detectors = ["BPM1"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 5)
    rows = _emit_rows(correctors, detectors, truth, currents)
    del rows[7]["CH2"]

    fit = sliced_response_matrix(rows, correctors, detectors, num=5)

    assert fit.complete == (True, False)
    assert fit.fitted_correctors == ("CH1",)


def test_trailing_rows_beyond_the_sweep_are_not_attributed_to_a_corrector():
    """Extra rows belong to no slice and must not shift the ones that do."""
    correctors = ["CH1", "CH2"]
    detectors = ["BPM1", "BPM2"]
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, 4)
    rows = _emit_rows(correctors, detectors, truth, currents)
    rows = rows + [dict(rows[-1]) for _ in range(3)]

    fit = sliced_response_matrix(rows, correctors, detectors, num=4)

    assert fit.complete == (True, True)
    np.testing.assert_allclose(fit.matrix, truth, rtol=1e-9)
    assert all(current.shape == (4,) for current in fit.currents)


def test_num_below_two_is_rejected():
    """One point per corrector defines no slope, and no slicing either."""
    with pytest.raises(ValueError, match="at least 2"):
        sliced_response_matrix([], ["CH1"], ["BPM1"], num=1)


def test_facility_scale_fit_is_fast():
    """Perf sanity at 70 correctors x 100 BPMs x 1470 rows.

    The formal gate (median-of-5) lives with the figure route, which measures
    the whole tick; this only pins that the fit itself is vectorized -- the
    per-pair polyfit path it replaces takes seconds on this shape, so the
    threshold discriminates by orders of magnitude and does not need
    CI-machine tuning.
    """
    correctors = [f"CH{j}" for j in range(70)]
    detectors = [f"BPM{i}" for i in range(100)]
    num = 21
    truth = _truth_matrix(len(detectors), len(correctors))
    currents = _sweep_currents(2.0, num)
    rows = _emit_rows(correctors, detectors, truth, currents)
    assert len(rows) == 1470

    timings = []
    for _ in range(5):
        started = time.perf_counter()
        fit = sliced_response_matrix(rows, correctors, detectors, num=num)
        timings.append(time.perf_counter() - started)

    assert fit.matrix.shape == (100, 70)
    np.testing.assert_allclose(fit.matrix, truth, rtol=1e-9)

    median = sorted(timings)[len(timings) // 2]
    assert median < 0.2, f"fit took {median * 1e3:.1f} ms at 70x100x1470 (timings: {timings})"
