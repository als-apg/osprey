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
)

CORRECTORS = ["corr1", "corr2", "corr3"]
DETECTORS = ["bpm1", "bpm2", "bpm3", "bpm4"]

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


def _rows_for_matrix(matrix: np.ndarray, correctors: list[str], detectors: list[str]) -> list:
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
            for i, detector in enumerate(detectors):
                row[detector] = float(matrix[i, j] * current)
            rows.append(row)
    return rows


# =========================================================================
# 3.4 build_response_matrix
# =========================================================================


def test_matrix_recovers_a_known_response_matrix() -> None:
    rows = _rows_for_matrix(KNOWN_MATRIX, CORRECTORS, DETECTORS)

    result = build_response_matrix(rows, CORRECTORS, DETECTORS)

    assert result.shape == (len(DETECTORS), len(CORRECTORS))
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
            for i, detector in enumerate(DETECTORS):
                row[f"{detector}-value"] = float(KNOWN_MATRIX[i, j] * current)
            rows.append(row)

    result = build_response_matrix(rows, CORRECTORS, DETECTORS)

    assert np.allclose(result, KNOWN_MATRIX)


def test_matrix_leaves_an_undersampled_corrector_column_at_zero() -> None:
    """A corrector with a single sample (or none) can't fit a slope; its
    column stays zero rather than raising.
    """
    rows = [{"corr1": 0.5, "bpm1": 0.25}]  # one point only

    result = build_response_matrix(rows, ["corr1"], ["bpm1"])

    assert result.shape == (1, 1)
    assert result[0, 0] == 0.0


def test_matrix_on_empty_rows_is_all_zero() -> None:
    result = build_response_matrix([], CORRECTORS, DETECTORS)

    assert result.shape == (len(DETECTORS), len(CORRECTORS))
    assert np.all(result == 0.0)


def _real_shape_rows(
    matrix: np.ndarray, correctors: list[str], detectors: list[str], sweeps: list[np.ndarray]
) -> list[dict]:
    """Rows shaped like a real `orm` plan run: every row carries EVERY
    corrector's key (idle ones at 0.0), not just the one being swept --
    mirroring the bundle `_orm_plan` reads at every point.
    """
    rows = []
    for j, corrector in enumerate(correctors):
        for current in sweeps[j]:
            row = {c: 0.0 for c in correctors}
            row[corrector] = float(current)
            for i, detector in enumerate(detectors):
                row[detector] = float(matrix[i, j] * current)
            rows.append(row)
    return rows


def test_guard_is_quiet_on_a_real_shaped_symmetric_sweep() -> None:
    """Every idle-corrector row (current 0.0) sits at the fit's x-mean when
    each corrector's own sweep is symmetric about 0, so it carries zero
    leverage on the slope -- the guard must not fire on this, the real
    plan's shape.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    rows = _real_shape_rows(KNOWN_MATRIX, CORRECTORS, DETECTORS, [currents] * len(CORRECTORS))

    result = build_response_matrix(rows, CORRECTORS, DETECTORS)

    assert np.allclose(result, KNOWN_MATRIX)


def test_guard_fires_on_a_real_shaped_asymmetric_sweep() -> None:
    """A corrector swept off-center (here [4, 6] instead of [-1, 1]) breaks
    the zero-mean invariant `build_response_matrix` depends on -- idle rows
    from the *other* correctors' symmetric sweeps no longer sit at this
    corrector's x-mean, so they would silently bias its fitted slope. The
    guard must raise instead of fitting garbage.
    """
    currents = np.linspace(-1.0, 1.0, 5)
    off_center = currents + 5.0  # [4.0, ..., 6.0] -- not symmetric about 0
    rows = _real_shape_rows(
        KNOWN_MATRIX, CORRECTORS, DETECTORS, [off_center, currents, currents]
    )

    with pytest.raises(DegenerateFitError, match="symmetric about 0"):
        build_response_matrix(rows, CORRECTORS, DETECTORS)


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
