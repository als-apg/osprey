"""Unit tests for `orm_analysis.py` (tasks 3.4-3.7): pure numpy, no bluesky
import anywhere in this file -- keep it that way so it always runs in the
main worktree venv (unlike `test_runengine_integration.py`, which
`importorskip`s bluesky).
"""

from __future__ import annotations

import numpy as np
import pytest

from osprey.services.bluesky_bridge.orm_analysis import (
    DegenerateFitError,
    build_response_matrix,
    column_anomaly,
    localize_kick,
    row_anomaly,
    singular_values,
)

CORRECTORS = ["corr1", "corr2", "corr3"]
BPMS = ["bpm1", "bpm2", "bpm3", "bpm4"]

# A known [n_bpm, n_corr] response matrix the synthetic rows below are built
# to reproduce exactly (no noise).
KNOWN_MATRIX = np.array(
    [
        [0.5, -1.2, 0.3],
        [1.1, 0.4, -0.6],
        [-0.8, 0.9, 1.5],
        [0.2, -0.3, 0.7],
    ]
)


def _rows_for_matrix(matrix: np.ndarray, correctors: list[str], bpms: list[str]) -> list:
    """Synthetic ORM rows: one dict per (corrector, current) point, each
    carrying only the swept corrector's key (others are a different
    corrector's sweep and never appear in this row) plus every BPM reading --
    mirroring the real `orm` plan's per-point event data.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    rows = []
    for j, corrector in enumerate(correctors):
        for current in currents:
            row = {corrector: float(current)}
            for i, bpm in enumerate(bpms):
                row[bpm] = float(matrix[i, j] * current)
            rows.append(row)
    return rows


# =========================================================================
# 3.4 build_response_matrix
# =========================================================================


def test_matrix_recovers_a_known_response_matrix() -> None:
    rows = _rows_for_matrix(KNOWN_MATRIX, CORRECTORS, BPMS)

    result = build_response_matrix(rows, CORRECTORS, BPMS)

    assert result.shape == (len(BPMS), len(CORRECTORS))
    assert np.allclose(result, KNOWN_MATRIX)


def test_matrix_matches_columns_by_device_name_prefix() -> None:
    """ophyd-async may key a hinted signal as `f"{device_name}-{signal}"`
    rather than the bare device name -- the fit must not depend on which.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    rows = []
    for j, corrector in enumerate(CORRECTORS):
        for current in currents:
            row = {f"{corrector}-readback": float(current)}
            for i, bpm in enumerate(BPMS):
                row[f"{bpm}-value"] = float(KNOWN_MATRIX[i, j] * current)
            rows.append(row)

    result = build_response_matrix(rows, CORRECTORS, BPMS)

    assert np.allclose(result, KNOWN_MATRIX)


def test_matrix_prefers_the_exact_column_over_a_prefixed_one() -> None:
    """A run carrying both spellings for a channel is read through the exact one.

    The two spellings come from the two device implementations, and a row can
    legitimately carry both (a device named `bpm1` next to a `bpm1-value` child
    signal of something else). `plan_fields.resolve_column` settles it in one
    place -- exact first -- and this pins that the fit sees that resolution
    rather than whichever key the row happened to list first.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    rows = []
    for current in currents:
        rows.append(
            {
                "corr1-readback": float(current),  # only spelling for the corrector
                "bpm1-value": 99.0,  # decoy: prefixed spelling listed first
                "bpm1": float(2.0 * current),  # the exact column, listed second
            }
        )

    result = build_response_matrix(rows, ["corr1"], ["bpm1"])

    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(2.0)


def test_matrix_skips_a_row_whose_corrector_column_holds_no_value() -> None:
    """A present-but-empty column is not a sample.

    A column resolving to a `None` value is a channel that did not report on
    that point, which is a gap, not a zero -- it must be excluded from the fit
    exactly as an absent column is, not coerced into `float(None)`.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    rows: list[dict] = [{"corr1": float(c), "bpm1": float(2.0 * c)} for c in currents]
    rows.insert(2, {"corr1": None, "bpm1": None})

    result = build_response_matrix(rows, ["corr1"], ["bpm1"])

    assert result[0, 0] == pytest.approx(2.0)


def test_matrix_leaves_an_undersampled_corrector_column_at_zero() -> None:
    """A corrector with a single sample (or none) can't fit a slope; its
    column stays zero rather than raising.
    """
    rows = [{"corr1": 0.5, "bpm1": 0.25}]  # one point only

    result = build_response_matrix(rows, ["corr1"], ["bpm1"])

    assert result.shape == (1, 1)
    assert result[0, 0] == 0.0


def test_matrix_on_empty_rows_is_all_zero() -> None:
    result = build_response_matrix([], CORRECTORS, BPMS)

    assert result.shape == (len(BPMS), len(CORRECTORS))
    assert np.all(result == 0.0)


def _real_shape_rows(
    matrix: np.ndarray, correctors: list[str], bpms: list[str], sweeps: list[np.ndarray]
) -> list[dict]:
    """Rows shaped like a real `orm` plan run: every row carries EVERY
    corrector's key (idle ones at 0.0), not just the one being swept --
    mirroring the bundle `_orm_plan` reads at every point.
    """
    rows = []
    for j, corrector in enumerate(correctors):
        for current in sweeps[j]:
            row = dict.fromkeys(correctors, 0.0)
            row[corrector] = float(current)
            for i, bpm in enumerate(bpms):
                row[bpm] = float(matrix[i, j] * current)
            rows.append(row)
    return rows


def test_guard_is_quiet_on_a_real_shaped_symmetric_sweep() -> None:
    """Every idle-corrector row sits at the fit's x-mean when each
    corrector's own sweep is symmetric about the value it idles at, so it
    carries zero leverage on the slope -- the guard must not fire on this,
    the real plan's shape. Here the idle value is 0.0, the virtual
    accelerator's; see the nonzero-working-point case below for a ring's.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    rows = _real_shape_rows(KNOWN_MATRIX, CORRECTORS, BPMS, [currents] * len(CORRECTORS))

    result = build_response_matrix(rows, CORRECTORS, BPMS)

    assert np.allclose(result, KNOWN_MATRIX)


def _relative_shape_rows(
    matrix: np.ndarray,
    correctors: list[str],
    bpms: list[str],
    kicks: np.ndarray,
    working_points: list[float],
) -> list[dict]:
    """Rows shaped like a real `orm` run on a ring with a corrected orbit.

    The difference from `_real_shape_rows` is the whole point of the relative
    sweep: each corrector holds its own nonzero working point, is swept
    *about* it, and is put back — so an idle corrector's key carries ITS
    working point rather than 0.0. BPM readings respond to the kick away from
    the working point (`matrix @ kick`) on top of the arbitrary corrected
    orbit those working points are already holding, so a fit that mistook the
    absolute setpoint for the kick would land on the wrong slope.
    """
    base_orbit = np.linspace(-3e-4, 3e-4, len(bpms))
    rows = []
    for j, corrector in enumerate(correctors):
        for kick in kicks:
            row = {name: working_points[k] for k, name in enumerate(correctors)}
            row[corrector] = working_points[j] + float(kick)
            for i, bpm in enumerate(bpms):
                row[bpm] = float(base_orbit[i] + matrix[i, j] * kick)
            rows.append(row)
    return rows


def test_guard_is_quiet_on_a_sweep_about_a_nonzero_working_point() -> None:
    """The real plan sweeps each corrector about its own pre-scan working
    point, so an idle corrector reads back that working point, not 0.0.

    The zero-leverage argument survives that intact — a sweep symmetric about
    `w` puts the fit's x-mean exactly at `w`, which is precisely where every
    idle sample sits — so the guard must stay quiet and the fitted slopes
    must still be the truth. The invariant was never "about zero"; zero was
    only the value a machine with no orbit to correct happens to idle at.
    """
    working_points = [2.5, -1.25, 4.0]
    kicks = np.linspace(-1.0, 1.0, 5)
    rows = _relative_shape_rows(KNOWN_MATRIX, CORRECTORS, BPMS, kicks, working_points)

    result = build_response_matrix(rows, CORRECTORS, BPMS)

    assert np.allclose(result, KNOWN_MATRIX)


def test_guard_fires_on_a_sweep_not_centred_on_the_idle_value() -> None:
    """A corrector swept about something other than where it idles is still
    rejected. Widening the invariant to "symmetric about the working point"
    must not widen it to "anything goes": here the corrector idles at 2.5 but
    is swept about 7.5, so the idle rows sit off the fit's x-mean and would
    bias its slope exactly as before.
    """
    working_points = [2.5, -1.25, 4.0]
    kicks = np.linspace(-1.0, 1.0, 5)
    rows = _relative_shape_rows(KNOWN_MATRIX, CORRECTORS, BPMS, kicks, working_points)
    for row in rows[: len(kicks)]:
        row[CORRECTORS[0]] += 5.0  # swept about 7.5 while idling at 2.5

    with pytest.raises(DegenerateFitError, match="symmetric about"):
        build_response_matrix(rows, CORRECTORS, BPMS)


def test_guard_fires_on_a_real_shaped_asymmetric_sweep() -> None:
    """A corrector swept off-center (here [4, 6] while idling at 0.0) breaks
    the invariant `build_response_matrix` depends on -- idle rows from the
    *other* correctors' sweeps no longer sit at this corrector's x-mean, so
    they would silently bias its fitted slope. The guard must raise instead
    of fitting garbage.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    off_center = currents + 5.0  # [4.0, ..., 6.0] -- not centred on the 0.0 idle value
    rows = _real_shape_rows(KNOWN_MATRIX, CORRECTORS, BPMS, [off_center, currents, currents])

    with pytest.raises(DegenerateFitError, match="symmetric about its idle value"):
        build_response_matrix(rows, CORRECTORS, BPMS)


# =========================================================================
# 3.5 localize_kick
# =========================================================================


def test_localize_recovers_the_seeded_corrector_with_high_contrast() -> None:
    # Orthonormal columns so `lstsq` recovers the seeded kick with no
    # crosstalk onto the other correctors -- a clean, deterministic contrast.
    rng = np.random.default_rng(1234)
    q, _ = np.linalg.qr(rng.standard_normal((6, 4)))
    matrix = q[:, :4]

    seeded_index = 2
    kick = np.zeros(4)
    kick[seeded_index] = 3.5
    observed_orbit = matrix @ kick

    index, solution = localize_kick(matrix, observed_orbit)

    assert index == seeded_index
    runner_up = max(abs(solution[k]) for k in range(len(solution)) if k != seeded_index)
    assert abs(solution[seeded_index]) >= 100 * runner_up


def test_localize_raises_on_empty_matrix() -> None:
    with pytest.raises(DegenerateFitError):
        localize_kick(np.zeros((0, 0)), np.zeros(0))


def test_localize_raises_on_zero_correctors() -> None:
    with pytest.raises(DegenerateFitError):
        localize_kick(np.zeros((4, 0)), np.zeros(4))


def test_localize_raises_on_beam_centered_orbit() -> None:
    matrix = np.eye(4)

    with pytest.raises(DegenerateFitError):
        localize_kick(matrix, np.zeros(4))


def test_localize_raises_on_shape_mismatch() -> None:
    matrix = np.eye(4)
    observed_orbit = np.ones(3)

    with pytest.raises(DegenerateFitError):
        localize_kick(matrix, observed_orbit)


# =========================================================================
# 3.6 column_anomaly / row_anomaly
# =========================================================================


def _clean_matrix() -> np.ndarray:
    """A separable, smoothly-varying matrix: every column is a scalar
    multiple of the same BPM shape, and every corrector gain is close to 1 --
    the "nothing is wrong" baseline both detectors should stay quiet on.
    """
    bpm_shape = np.linspace(1.0, 1.4, 5)  # 5 BPMs
    corrector_gains = np.array([1.0, 1.05, 0.95, 1.02])  # 4 correctors
    return np.outer(bpm_shape, corrector_gains)


def test_column_anomaly_detector_is_quiet_on_clean_data() -> None:
    scores = column_anomaly(_clean_matrix())

    assert np.all(scores < 0.5)


def test_column_anomaly_detector_fires_on_an_injected_gain_fault() -> None:
    matrix = _clean_matrix()
    matrix[:, 2] *= 5.0  # corrector-gain fault on column 2

    scores = column_anomaly(matrix)

    assert scores[2] > 2.0
    for j in (0, 1, 3):
        assert scores[j] < 0.5


def test_column_anomaly_detector_fires_on_a_stuck_corrector() -> None:
    matrix = _clean_matrix()
    matrix[:, 1] = 0.0  # stuck corrector

    scores = column_anomaly(matrix)

    assert scores[1] > 0.5
    for j in (0, 2, 3):
        assert scores[j] < 0.5


def test_column_anomaly_detector_is_all_zero_with_fewer_than_two_correctors() -> None:
    scores = column_anomaly(np.ones((4, 1)))

    assert np.all(scores == 0.0)


def test_row_anomaly_detector_is_quiet_on_clean_data() -> None:
    scores = row_anomaly(_clean_matrix())

    assert np.all(scores < 0.5)


def test_row_anomaly_detector_fires_on_an_injected_polarity_fault() -> None:
    matrix = _clean_matrix()
    matrix[2, :] *= -1.0  # BPM polarity flip on row 2

    scores = row_anomaly(matrix)

    assert scores[2] > 0.5
    for i in (0, 1, 3, 4):
        assert scores[i] < 0.5


def test_row_anomaly_detector_is_all_zero_with_fewer_than_two_bpms() -> None:
    scores = row_anomaly(np.ones((1, 4)))

    assert np.all(scores == 0.0)


# =========================================================================
# 3.7 singular_values
# =========================================================================


def test_singular_values_are_returned_largest_first() -> None:
    values = singular_values(_clean_matrix())

    assert values.size > 0
    assert np.all(np.diff(values) <= 0.0)


def test_a_separable_matrix_has_exactly_one_significant_mode() -> None:
    """`_clean_matrix` is a rank-1 outer product, so all the response lives in
    the first singular value -- the spectrum a perfectly-correlated (and so
    uninformative-beyond-one-mode) machine would show."""
    values = singular_values(_clean_matrix())

    assert values[0] > 0.0
    assert np.all(values[1:] < values[0] * 1e-10)


def test_added_independent_structure_raises_the_second_mode() -> None:
    matrix = _clean_matrix()
    matrix[:, 1] += np.array([0.4, -0.4, 0.4, -0.4, 0.4])  # a second direction

    values = singular_values(matrix)

    assert values[1] > values[0] * 1e-3


def test_singular_values_count_is_the_smaller_dimension() -> None:
    assert singular_values(np.ones((7, 3))).size == 3
    assert singular_values(np.ones((3, 7))).size == 3


def test_singular_values_of_an_empty_matrix_is_empty() -> None:
    assert singular_values(np.zeros((0, 0))).size == 0
    assert singular_values(np.zeros((5, 0))).size == 0


def test_singular_values_of_a_non_finite_matrix_is_empty() -> None:
    """`numpy.linalg.svd` raises on NaN rather than returning one, and a
    figure is a view: a partly-fitted matrix must degrade to no spectrum, not
    to an exception that costs the whole figure."""
    matrix = _clean_matrix()
    matrix[1, 1] = np.nan

    assert singular_values(matrix).size == 0
