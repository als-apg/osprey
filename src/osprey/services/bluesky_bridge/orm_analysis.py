"""Orbit-response-matrix (ORM) analysis: pure numpy, bluesky-free.

Consumes the plain `dict` rows an `orm` plan run emits (bluesky's event
document `data`, one dict per point — see `plans.py`'s `_orm_plan`), never a
live-buffer or Tiled-specific shape, so this module has no dependency on
`bluesky`/`ophyd-async`/`tiled` and imports cleanly on the MCP side (where the
`bluesky-bridge` extra is not installed).

Three pieces:

- `build_response_matrix`: per-(BPM, corrector) slope fit from swept rows.
- `localize_kick`: `lstsq` fault localization against a measured orbit.
- `column_anomaly`/`row_anomaly`: model-free structural fault detectors that
  use only the matrix's own peer/neighbour structure — no independent model
  of the machine is required.

`build_response_matrix`'s fit depends on an invariant the real `orm` plan
upholds (see `plans.py`'s `_orm_plan` docstring): every emitted row carries
EVERY corrector's current, not just the one being swept — a non-swept
corrector simply reads back its idle (~0 A) value in that row. The fit still
recovers the right slope only because each corrector's own sweep is
symmetric about 0 and its idle reading is (numerically) exactly 0: together
those put every idle sample at the fit's x-mean, so it carries zero leverage
on the polyfit slope no matter what BPM reading that row actually carries
(driven by whichever OTHER corrector was being swept at the time).
`build_response_matrix` checks this per corrector and raises
`DegenerateFitError` if it's violated — see its docstring below.

Column-key matching (`_match_value`) is deliberately name-fuzzy: ophyd-async's
`StandardReadable` keys a hinted signal's document entry as either the bare
device name or `f"{device_name}-{signal}"` depending on how many readable
children the device exposes (see `devices/mock.py`/`devices/epics.py`,
neither of which this module imports), so device names are matched by exact
key first, then by `f"{name}-"` prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class DegenerateFitError(ValueError):
    """`localize_kick` cannot produce a meaningful localization.

    Raised for an empty/malformed response matrix or observed orbit, a
    zero-corrector matrix, or a beam-centered (all-zero) orbit — the
    argmax of a `lstsq` solution over no correctors or against no signal is
    not a meaningful answer.
    """


def _match_value(row: Mapping[str, Any], device_name: str) -> Any | None:
    """The event-data value for *device_name* in *row*, or ``None`` if absent."""
    if device_name in row:
        return row[device_name]
    prefix = f"{device_name}-"
    for key, value in row.items():
        if key.startswith(prefix):
            return value
    return None


#: Tolerance for the zero-mean-sweep guard in `build_response_matrix`: a
#: corrector's collected currents fail the check when their mean exceeds this
#: fraction of their spread (max - min). Floating-point noise on an exactly
#: symmetric sweep + exactly-zero idle readings sits around 1e-15 relative,
#: so this leaves ~9 orders of magnitude of margin before a genuine
#: asymmetric sweep or biased idle reading is required to trip it.
_SWEEP_SYMMETRY_TOL = 1e-6


def build_response_matrix(
    rows: Sequence[Mapping[str, Any]],
    correctors: Sequence[str],
    detectors: Sequence[str],
) -> np.ndarray:
    """Fit the `[n_bpm, n_corr]` response-slope matrix from emitted ORM rows.

    A real `orm` plan run emits one row per (corrector, current) point, and
    every row carries a value for EVERY corrector in `correctors` — not just
    the one currently being swept (see `plans.py`'s `_orm_plan` docstring) —
    plus a reading for every detector. For each corrector, every row where
    that corrector's key is present (in practice: every row) forms one
    (current, BPM reading) sample; `numpy.polyfit` (degree 1) over those
    samples gives the response slope for each (BPM, corrector) pair.

    This only recovers the correct slope because the real plan's sweep is
    symmetric about 0 and idle correctors read back (numerically) exactly 0:
    together those put the fit's x-mean at 0, so every idle-corrector sample
    sits exactly at that mean and carries zero leverage on the fitted slope —
    regardless of what BPM reading that row actually carries (driven by
    whichever OTHER corrector was being swept at the time). Before fitting,
    each corrector's collected currents are checked against that zero-mean
    invariant (see `_SWEEP_SYMMETRY_TOL`); a sweep that isn't symmetric about
    0, or an idle reading with a non-negligible bias, would silently corrupt
    the slope, so a violation raises `DegenerateFitError` naming the
    corrector instead of fitting garbage.

    A row missing a given corrector's key entirely — not the real plan's
    shape, but a valid input for a caller building rows by hand (e.g. this
    module's own unit tests) — is simply excluded from that corrector's fit.
    The `continue` below is dead for real `orm` plan output, where every row
    carries every corrector's key; it stays live for hand-built or otherwise
    partial rows.

    A corrector with fewer than two samples (or a BPM missing a reading in
    any of them) leaves its column/entry at ``0.0`` rather than raising —
    an incomplete sweep is a data-quality question for the caller, not a
    reason to abort the whole matrix.

    Raises:
        DegenerateFitError: a corrector's collected currents are not
            symmetric about 0 within `_SWEEP_SYMMETRY_TOL` — see above.
    """
    matrix = np.zeros((len(detectors), len(correctors)))

    for j, corrector in enumerate(correctors):
        currents: list[float] = []
        readings: list[list[float]] = [[] for _ in detectors]
        for row in rows:
            current = _match_value(row, corrector)
            if current is None:
                continue  # row built without this corrector's key (see docstring)
            currents.append(float(current))
            for i, detector in enumerate(detectors):
                reading = _match_value(row, detector)
                readings[i].append(np.nan if reading is None else float(reading))

        if len(currents) < 2:
            continue  # not enough points along this corrector to fit a slope

        mean_current = float(np.mean(currents))
        spread = float(np.max(currents) - np.min(currents))
        if spread > 0 and abs(mean_current) > _SWEEP_SYMMETRY_TOL * spread:
            raise DegenerateFitError(
                f"corrector {corrector!r}'s sweep is not symmetric about 0 "
                f"(mean current {mean_current:.6g} over a spread of "
                f"{spread:.6g}): build_response_matrix's polyfit depends on "
                f"a zero-mean sweep so idle-corrector rows carry zero "
                f"leverage on the fitted slope -- an asymmetric sweep or a "
                f"nonzero idle reading would silently bias every slope for "
                f"this corrector"
            )

        for i in range(len(detectors)):
            values = readings[i]
            if any(np.isnan(v) for v in values):
                continue  # this BPM never reported during this corrector's sweep
            slope, _intercept = np.polyfit(currents, values, deg=1)
            matrix[i, j] = slope

    return matrix


def localize_kick(
    response_matrix: np.ndarray, observed_orbit: np.ndarray
) -> tuple[int, np.ndarray]:
    """Localize an unknown kick by solving `response_matrix @ kick = observed_orbit`.

    Returns the index of the corrector whose fitted kick strength has the
    largest magnitude, and the full `numpy.linalg.lstsq` solution vector.

    Raises:
        DegenerateFitError: the matrix or orbit is empty/malformed, the
            matrix has zero correctors, or the observed orbit is all-zero
            (beam-centered — there is no kick signal to localize).
    """
    matrix = np.asarray(response_matrix, dtype=float)
    orbit = np.asarray(observed_orbit, dtype=float)

    if matrix.size == 0 or orbit.size == 0:
        raise DegenerateFitError("response matrix and observed orbit must both be non-empty")
    if matrix.ndim != 2 or orbit.ndim != 1:
        raise DegenerateFitError("response matrix must be 2-D and observed orbit 1-D")
    if matrix.shape[0] != orbit.shape[0]:
        raise DegenerateFitError(
            f"response matrix has {matrix.shape[0]} BPM rows but observed orbit has "
            f"{orbit.shape[0]} entries"
        )
    if matrix.shape[1] == 0:
        raise DegenerateFitError("response matrix has no correctors to localize a kick against")
    if not np.any(orbit):
        raise DegenerateFitError(
            "observed orbit is beam-centered (all-zero) -- no kick to localize"
        )

    solution, *_ = np.linalg.lstsq(matrix, orbit, rcond=None)
    index = int(np.argmax(np.abs(solution)))
    return index, solution


def column_anomaly(matrix: np.ndarray) -> np.ndarray:
    """Per-corrector anomaly score from the matrix's own column structure.

    Model-free: each column is scored against the elementwise *median* of
    every other column (not the mean, so one already-anomalous peer doesn't
    drag down its own reference), combining two signals into one score:

    - norm ratio: this column's L2 norm relative to its peer median's,
      flagging a corrector gain error or a stuck (near-zero) corrector.
    - peer correlation: this column's shape correlated against the peer
      median, flagging a polarity flip a norm ratio alone would miss (a
      sign-flipped column keeps its peers' norm but anti-correlates with
      them).

    Returns one score per corrector (higher = more anomalous, `0.0` for a
    perfectly peer-consistent column); no fixed threshold is imposed here —
    an analysis step picks its own.
    """
    matrix = np.asarray(matrix, dtype=float)
    n_corr = matrix.shape[1]
    scores = np.zeros(n_corr)
    if n_corr < 2:
        return scores

    for j in range(n_corr):
        col = matrix[:, j]
        peer_shape = np.median(np.delete(matrix, j, axis=1), axis=1)
        scores[j] = _peer_score(col, peer_shape)

    return scores


def row_anomaly(matrix: np.ndarray) -> np.ndarray:
    """Per-BPM anomaly score from the matrix's own row structure.

    Model-free, mirroring `column_anomaly` transposed: each row is scored
    against the elementwise median of every *other* row rather than just its
    index-adjacent neighbours -- a literal `i-1`/`i+1` window has only two
    reference points (no robustness at all against the corrupted row itself
    being one of them, and it can coincidentally land close to a smooth
    baseline's local average and hide). Flags a BPM gain error (norm ratio)
    or polarity flip (anti-correlation with its peers).

    Returns one score per BPM (higher = more anomalous, `0.0` for a
    perfectly peer-consistent row).
    """
    matrix = np.asarray(matrix, dtype=float)
    n_bpm = matrix.shape[0]
    scores = np.zeros(n_bpm)
    if n_bpm < 2:
        return scores

    for i in range(n_bpm):
        row = matrix[i, :]
        peer_shape = np.median(np.delete(matrix, i, axis=0), axis=0)
        scores[i] = _peer_score(row, peer_shape)

    return scores


def _peer_score(vector: np.ndarray, reference: np.ndarray) -> float:
    """Combined norm-ratio + anti-correlation score of *vector* vs *reference*."""
    vector_norm = float(np.linalg.norm(vector))
    reference_norm = float(np.linalg.norm(reference))

    norm_ratio = vector_norm / reference_norm if reference_norm > 0 else 0.0
    denom = vector_norm * reference_norm
    correlation = float(np.dot(vector, reference) / denom) if denom > 0 else 0.0

    return abs(norm_ratio - 1.0) + max(0.0, -correlation)
