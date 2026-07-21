"""Unit test for `orm_analysis.py` (task 2.1): dual-fault separability --
pure numpy, no bluesky import anywhere in this file, so it always runs in
the main worktree venv (see `test_orm_analysis.py`, which this file
complements rather than duplicates).

`row_anomaly` and `column_anomaly` (both in `orm_analysis.py`) are derived
from the SAME response matrix's own peer/neighbour structure. The existing
unit tests exercise each detector against a single injected fault at a time;
this module proves the two detectors don't interfere with each other when a
BPM fault and a corrector fault land in the SAME matrix simultaneously --
each detector must still localize its own fault, undistracted by the other.

That separability is not automatic for any pair of fault types -- it holds
for the specific pairing exercised below because of a physics choice:

- The BPM fault is a **polarity flip** (row * -1), which preserves the
  row's L2 norm. `column_anomaly` scores a column by norm-ratio +
  anti-correlation against its peer columns' median *shape*; a
  norm-preserving row corruption barely perturbs that per-column shape
  comparison (only one row's sign disagrees with peers, which
  `row_anomaly` -- not `column_anomaly` -- is built to catch), so it
  doesn't swamp the column detector's ability to also localize an
  independent corrector fault in the same matrix.
- The corrector fault is a **bounded weak-corrector deficit** (column *
  0.5), not a stuck (zeroed) corrector -- a partial gain loss, still
  detectable via `column_anomaly`'s norm-ratio term without relying on
  the degenerate all-zero case `test_orm_analysis.py` already covers.

The guard test at the bottom documents *why* the BPM fault must be
norm-preserving: swap the polarity flip for a large BPM **gain** error
(row * 10, NOT norm-preserving) and the same weak-corrector column no
longer separates cleanly from an innocent peer column -- the inflated
row's own norm-ratio anomaly leaks into every column's peer comparison at
that row, collapsing most of the score margin that made the polarity-flip
case a clean call.
"""

from __future__ import annotations

import numpy as np

from osprey.services.bluesky_bridge.orm_analysis import column_anomaly, row_anomaly

N_BPM, N_CORR = 6, 5
FLIP_ROW = 2  # BPM index carrying the polarity-flip / gain fault
WEAK_COL = 3  # corrector index carrying the bounded weak-corrector deficit


def _baseline_matrix() -> np.ndarray:
    """A smooth, orbit-response-like `[n_bpm, n_corr]` baseline: a slowly
    varying BPM envelope and near-uniform corrector gains (mirroring
    `test_orm_analysis.py`'s `_clean_matrix`), modulated by a small
    phase-advance-like `cos` term so the matrix is genuinely rank > 1 --
    not a bare outer product -- while staying peer-consistent (quiet under
    both detectors) as a "nothing is wrong" baseline.
    """
    bpm_shape = np.linspace(1.0, 1.5, N_BPM)
    corr_gain = np.array([1.0, 1.05, 0.95, 1.02, 0.98])
    s_bpm = np.linspace(0.0, 1.0, N_BPM)
    s_corr = np.linspace(0.0, 1.0, N_CORR)
    phase = 2 * np.pi * (s_bpm[:, None] - s_corr[None, :])
    return np.outer(bpm_shape, corr_gain) * (1 + 0.2 * np.cos(phase))


def _other_max(scores: np.ndarray, excluded: int) -> float:
    """The highest score among all entries other than `excluded`."""
    return max(scores[k] for k in range(len(scores)) if k != excluded)


def test_dual_fault_matrix_localizes_both_faults_simultaneously() -> None:
    """A norm-preserving BPM polarity flip on `FLIP_ROW` and a bounded
    weak-corrector deficit on `WEAK_COL`, injected into the SAME matrix,
    must each be localized by their own detector -- the load-bearing
    separability proof.
    """
    matrix = _baseline_matrix()
    matrix[FLIP_ROW, :] *= -1.0  # BPM polarity flip: norm-preserving
    matrix[:, WEAK_COL] *= 0.5  # weak corrector: bounded, not zeroed

    row_scores = row_anomaly(matrix)
    col_scores = column_anomaly(matrix)

    assert row_scores.argmax() == FLIP_ROW
    assert row_scores[FLIP_ROW] > 2.0 * _other_max(row_scores, FLIP_ROW)

    assert col_scores.argmax() == WEAK_COL
    assert col_scores[WEAK_COL] > 2.5 * _other_max(col_scores, WEAK_COL)


def test_bpm_gain_fault_degrades_column_separability_vs_polarity_flip() -> None:
    """Guard test pinning WHY the polarity-flip choice above is load-bearing:
    replace the norm-preserving flip with a large, non-norm-preserving BPM
    **gain** error (row * 10) and the weak-corrector column's separation
    from an innocent peer column collapses.

    The gain fault still leaves `WEAK_COL` as `column_anomaly`'s argmax in
    this construction (the huge shared row entry inflates every column's
    peer comparison by roughly the same factor, so it doesn't flip the
    winner outright) -- but the MARGIN over the best innocent column, which
    was a clean >2.5x separation under the polarity flip, collapses to a
    near-tie. That collapsed margin -- not requiring a wrong argmax -- is
    what "does not cleanly localize" means here: an analysis step reading
    off "the anomalous column" no longer gets a confident, unambiguous
    answer.
    """
    baseline = _baseline_matrix()

    flip_matrix = baseline.copy()
    flip_matrix[FLIP_ROW, :] *= -1.0  # norm-preserving
    flip_matrix[:, WEAK_COL] *= 0.5

    gain_matrix = baseline.copy()
    gain_matrix[FLIP_ROW, :] *= 10.0  # NOT norm-preserving
    gain_matrix[:, WEAK_COL] *= 0.5

    flip_col_scores = column_anomaly(flip_matrix)
    gain_col_scores = column_anomaly(gain_matrix)

    flip_margin = flip_col_scores[WEAK_COL] / _other_max(flip_col_scores, WEAK_COL)
    gain_margin = gain_col_scores[WEAK_COL] / _other_max(gain_col_scores, WEAK_COL)

    assert flip_margin > 2.5  # polarity flip: clean, unambiguous separation
    assert gain_margin < 1.5  # gain error: separation confused, near-tie
