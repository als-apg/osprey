"""Orbit-response-matrix (ORM) analysis: pure numpy, bluesky-free.

Consumes the plain `dict` rows an `orm` plan run emits (bluesky's event
document `data`, one dict per point — see `plans_core/orm.py`'s `build_plan`), never a
live-buffer or Tiled-specific shape, so this module has no dependency on
`bluesky`/`ophyd-async`/`tiled` and imports cleanly on the MCP side without
loading the bluesky stack (a core dependency, but heavier than this pure-numpy
analysis needs).

Four pieces:

- `build_response_matrix`: per-(BPM, corrector) slope fit from swept rows.
- `sliced_response_matrix`: the same fit, sliced by acquisition index and
  vectorized — the entry point a plan's `render()` uses on every live tick,
  where `build_response_matrix`'s per-pair `polyfit` over per-row dict scans
  is orders of magnitude too slow.
- `localize_kick`: `lstsq` fault localization against a measured orbit.
- `column_anomaly`/`row_anomaly`: model-free structural fault detectors that
  use only the matrix's own peer/neighbour structure — no independent model
  of the machine is required.

`build_response_matrix`'s fit depends on an invariant the real `orm` plan
upholds (see `plans_core/orm.py`'s `build_plan` docstring): every emitted row carries
EVERY corrector's current, not just the one being swept — a non-swept
corrector simply reads back its idle value in that row. The fit still
recovers the right slope only because each corrector's own sweep is
symmetric about that same idle value: together those put every idle sample
at the fit's x-mean, so it carries zero leverage on the polyfit slope no
matter what BPM reading that row actually carries (driven by whichever OTHER
corrector was being swept at the time). The idle value is NOT assumed to be
zero — the plan sweeps each corrector about its own pre-scan working point,
which on a ring running corrected orbit is nonzero, and restores it there.
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
from dataclasses import dataclass
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


#: Tolerance for the centred-sweep guard in `build_response_matrix`: a
#: corrector's collected currents fail the check when their mean differs from
#: their median by more than this fraction of their spread (max - min).
#: Floating-point noise on an exactly symmetric sweep + exactly-equal idle
#: readings sits around 1e-15 relative, so this leaves ~9 orders of magnitude
#: of margin before a genuine off-centre sweep or biased idle reading is
#: required to trip it.
_SWEEP_SYMMETRY_TOL = 1e-6


def build_response_matrix(
    rows: Sequence[Mapping[str, Any]],
    correctors: Sequence[str],
    detectors: Sequence[str],
) -> np.ndarray:
    """Fit the `[n_bpm, n_corr]` response-slope matrix from emitted ORM rows.

    A real `orm` plan run emits one row per (corrector, current) point, and
    every row carries a value for EVERY corrector in `correctors` — not just
    the one currently being swept (see `plans_core/orm.py`'s `build_plan` docstring) —
    plus a reading for every detector. For each corrector, every row where
    that corrector's key is present (in practice: every row) forms one
    (current, BPM reading) sample; `numpy.polyfit` (degree 1) over those
    samples gives the response slope for each (BPM, corrector) pair.

    This only recovers the correct slope because the real plan sweeps each
    corrector symmetrically about the very value that corrector reads back
    while idle — its pre-scan working point. That puts the fit's x-mean
    exactly at the idle value (the sweep's own offsets sum to zero), so every
    idle-corrector sample sits exactly at that mean and carries zero leverage
    on the fitted slope — regardless of what BPM reading that row actually
    carries (driven by whichever OTHER corrector was being swept at the
    time). Note the invariant is "centred on the idle value", not "centred on
    zero": zero is simply where a machine with no orbit to correct happens to
    idle, and a real ring's correctors do not.

    Before fitting, each corrector's collected currents are checked against
    that invariant (see `_SWEEP_SYMMETRY_TOL`) by comparing their mean to
    their median. The median IS the idle value for any run this function is
    meant to take: idle samples are `(n_correctors - 1) / n_correctors` of a
    corrector's rows, so they are at least half of them whenever more than
    one corrector was swept, and for a single corrector every sample is a
    swept one and a symmetric sweep's median is its centre either way. A
    sweep not centred on where the corrector idles — an off-centre sweep, a
    one-sided (`monodirectional`) one, or an idle reading with a bias —
    separates mean from median and would silently corrupt the slope, so it
    raises `DegenerateFitError` naming the corrector instead of fitting
    garbage.

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
            symmetric about that corrector's own idle value within
            `_SWEEP_SYMMETRY_TOL` — see above.
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
        idle_current = float(np.median(currents))
        spread = float(np.max(currents) - np.min(currents))
        if spread > 0 and abs(mean_current - idle_current) > _SWEEP_SYMMETRY_TOL * spread:
            raise DegenerateFitError(
                f"corrector {corrector!r}'s sweep is not symmetric about its "
                f"idle value (mean current {mean_current:.6g} against an idle "
                f"value of {idle_current:.6g}, over a spread of {spread:.6g}): "
                f"build_response_matrix's polyfit depends on idle-corrector "
                f"rows sitting at the fit's x-mean so they carry zero leverage "
                f"on the fitted slope -- an off-centre sweep, or an idle "
                f"reading that is not where the sweep is centred, would "
                f"silently bias every slope for this corrector"
            )

        for i in range(len(detectors)):
            values = readings[i]
            if any(np.isnan(v) for v in values):
                continue  # this BPM never reported during this corrector's sweep
            slope, _intercept = np.polyfit(currents, values, deg=1)
            matrix[i, j] = slope

    return matrix


@dataclass(frozen=True)
class SlicedResponseFit:
    """What `sliced_response_matrix` recovers from one ORM run's rows.

    Two groups of fields, because a plan's `render()` needs both: the raw
    per-corrector sweep it can always draw as traces, and the fitted matrix
    it can only draw once a sweep has actually finished.

    Attributes:
        correctors: The requested corrector names, in the order given —
            `complete`, `currents` and `readings` are all aligned to this.
        detectors: The requested detector names, in the order given — the
            row axis of `matrix` and the column axis of every `readings`
            block.
        complete: One flag per requested corrector: `True` when that
            corrector's slice was fitted into a `matrix` column. `False`
            for a slice that is short (the sweep is still in flight, or the
            row buffer ran out), that is missing a corrector current, or
            whose currents never move (nothing to fit a slope against).
        matrix: `[n_detectors, n_complete]` response slopes. Only complete
            correctors get a column — an in-flight sweep has no slope yet,
            and a zero column would read as a dead corrector. Use
            `fitted_correctors` for its column labels; both are ordered as
            `correctors` is.
        fitted_correctors: The names of the correctors that produced a
            `matrix` column, in column order. A subsequence of `correctors`.
        currents: Per requested corrector, that corrector's own recorded
            currents over its slice, `[k]` with `k <= num` — the x-axis of a
            sweep trace. Present for incomplete correctors too.
        readings: Per requested corrector, the detector block over its
            slice, `[k, n_detectors]` — the y-axes of a sweep trace, one
            column per detector. Present for incomplete correctors too.
            A detector that did not report reads back as `nan` here (in
            `matrix` it lands on `0.0`; see `sliced_response_matrix`).
    """

    correctors: tuple[str, ...]
    detectors: tuple[str, ...]
    complete: tuple[bool, ...]
    matrix: np.ndarray
    fitted_correctors: tuple[str, ...]
    currents: tuple[np.ndarray, ...]
    readings: tuple[np.ndarray, ...]


def _resolve_key(row: Mapping[str, Any], device_name: str) -> str | None:
    """The event-data *key* carrying *device_name* in *row*, or ``None``.

    `_match_value`'s matching rules (exact key, then `f"{name}-"` prefix)
    resolved to the key instead of the value, so a whole run's rows can be
    read through one fixed key list rather than re-matching every row.
    """
    if device_name in row:
        return device_name
    prefix = f"{device_name}-"
    for key in row:
        if key.startswith(prefix):
            return key
    return None


def sliced_response_matrix(
    rows: Sequence[Mapping[str, Any]],
    correctors: Sequence[str],
    detectors: Sequence[str],
    num: int,
) -> SlicedResponseFit:
    """Fit the response matrix by slicing *rows* into one sweep per corrector.

    Same physics as `build_response_matrix`, different — and much stronger —
    assumption about the input, which buys both correctness on inputs that
    function cannot take and a fit fast enough to run on every live tick.

    The `orm` plan sweeps its correctors strictly one at a time, `num`
    points each, in the order they were requested (see `plans_core/orm.py`'s
    `build_plan`), so corrector `j`'s samples are exactly
    `rows[j * num : (j + 1) * num]` — no need to infer which rows belong to
    which corrector from the data. Each corrector's slope is then fitted
    against its OWN currents over its OWN slice only, which drops
    `build_response_matrix`'s centred-sweep requirement entirely: idle
    correctors' rows are never in the slice to begin with, so a
    `monodirectional` sweep over `[0, span_a]` (which that function rejects
    with `DegenerateFitError`, its polyfit having no way to tell a one-sided
    sweep from a biased idle reading) fits correctly here. A slope is
    invariant to a shift in its x axis, so this path needs to know nothing
    about where a corrector's sweep is centred — a sweep about a nonzero
    working point fits exactly as a sweep about zero does.

    The caller owes that slicing invariant: *rows* must be the run's events
    in emission order, un-truncated at the front and with nothing dropped in
    the middle. A truncated-from-the-front buffer silently misattributes
    every sample, so a caller reading a capped live buffer must check its
    completeness before calling (there is nothing in the rows themselves to
    detect it from). Trailing truncation IS safe and expected — that is just
    a run still in progress, and it shows up as trailing `complete=False`.

    Speed comes from doing the two costly things once instead of per pair:
    device names are resolved to event keys once for the whole run (rather
    than `_match_value`-scanning every row for every device), and each
    corrector's slopes are computed for the whole detector block at once
    with the closed-form least-squares slope

        slope = (dx @ (Y - Y_mean)) / (dx @ dx),    dx = x - x_mean

    which is what a degree-1 `numpy.polyfit` solves, without building a
    Vandermonde system per (detector, corrector) pair.

    A slice that is short (the run is still going, or ended early), that is
    missing its corrector's current, or whose currents never move produces
    no matrix column and is flagged `complete=False`; its raw samples are
    still returned for tracing. Inside a complete slice, a detector that
    failed to report leaves its slope at `0.0` rather than `nan`, matching
    `build_response_matrix` and keeping the matrix usable by
    `column_anomaly`/`row_anomaly`; the `nan` survives in `readings` where a
    trace can show the gap.

    Args:
        rows: The run's event `data` dicts in emission order.
        correctors: Corrector device names, in the order the plan swept them.
        detectors: Detector device names; the matrix's row axis.
        num: Points per corrector sweep — the plan's `num` parameter.

    Raises:
        ValueError: *num* is below 2 (no slope is defined over fewer than
            two points, and a non-positive *num* makes the slicing itself
            meaningless).
    """
    if num < 2:
        raise ValueError(f"num must be at least 2 to fit a slope per corrector, got {num}")

    correctors = tuple(correctors)
    detectors = tuple(detectors)
    n_corr = len(correctors)
    n_det = len(detectors)

    # Only the rows the slicing can attribute to a corrector: a run carrying
    # more than the sweep accounts for has extra rows at the END (nothing
    # else preserves the invariant above), so they belong to no slice.
    n_rows = min(len(rows), n_corr * num)
    if n_corr == 0 or n_rows == 0:
        empty = np.zeros((n_det, 0))
        return SlicedResponseFit(
            correctors=correctors,
            detectors=detectors,
            complete=(False,) * n_corr,
            matrix=empty,
            fitted_correctors=(),
            currents=tuple(np.zeros(0) for _ in correctors),
            readings=tuple(np.zeros((0, n_det)) for _ in correctors),
        )

    # Resolve every device to its event key once. The first row normally
    # settles all of them (a bluesky stream's rows share one key set); the
    # loop only runs on to cover a caller's hand-built ragged rows.
    keys: list[str | None] = [None] * (n_corr + n_det)
    unresolved = set(range(n_corr + n_det))
    names = correctors + detectors
    for row in rows[:n_rows]:
        for position in tuple(unresolved):
            key = _resolve_key(row, names[position])
            if key is not None:
                keys[position] = key
                unresolved.discard(position)
        if not unresolved:
            break

    # One pass over the rows through that fixed key list. A device that
    # resolved to no key reads as `nan` everywhere, which is exactly how a
    # missing device should behave downstream.
    nan = float("nan")
    table = np.array(
        [[nan if key is None else row.get(key, nan) for key in keys] for row in rows[:n_rows]],
        dtype=float,
    )
    currents_table = table[:, :n_corr]
    readings_table = table[:, n_corr:]

    complete: list[bool] = []
    slice_currents: list[np.ndarray] = []
    slice_readings: list[np.ndarray] = []
    columns: list[np.ndarray] = []
    fitted: list[str] = []

    for j, corrector in enumerate(correctors):
        start = j * num
        stop = min(start + num, n_rows)
        x = currents_table[start:stop, j]
        block = readings_table[start:stop, :]
        slice_currents.append(x)
        slice_readings.append(block)

        if x.shape[0] < num or not np.all(np.isfinite(x)):
            complete.append(False)
            continue

        dx = x - x.mean()
        denominator = float(dx @ dx)
        if denominator <= 0.0:
            # The corrector never moved over its own sweep: no slope exists,
            # and dividing by zero would fill the column with inf/nan.
            complete.append(False)
            continue

        slopes = (dx @ (block - block.mean(axis=0))) / denominator
        # A detector that dropped out mid-slice poisons only its own entry.
        slopes = np.where(np.isfinite(slopes), slopes, 0.0)
        complete.append(True)
        columns.append(slopes)
        fitted.append(corrector)

    matrix = np.column_stack(columns) if columns else np.zeros((n_det, 0))

    return SlicedResponseFit(
        correctors=correctors,
        detectors=detectors,
        complete=tuple(complete),
        matrix=matrix,
        fitted_correctors=tuple(fitted),
        currents=tuple(slice_currents),
        readings=tuple(slice_readings),
    )


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


def singular_values(matrix: np.ndarray) -> np.ndarray:
    """The response matrix's singular value spectrum, largest first.

    The conventional companion to the matrix itself: how fast the spectrum
    falls says how many independent correction modes the measured machine
    actually supports, and where the tail flattens is the noise floor an
    orbit correction would truncate at. A separable (rank-1) matrix collapses
    to a single significant value; genuine independent structure lifts the
    ones behind it.

    Returns an empty array rather than raising when there is nothing to
    decompose -- no correctors or BPMs yet, or a matrix still carrying
    non-finite cells from a partly-fitted live run. `numpy.linalg.svd` raises
    on both, and a figure is a view: a spectrum that cannot be computed must
    cost its own panel, not the whole figure.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return np.zeros(0)

    return np.linalg.svd(matrix, compute_uv=False)


def _peer_score(vector: np.ndarray, reference: np.ndarray) -> float:
    """Combined norm-ratio + anti-correlation score of *vector* vs *reference*."""
    vector_norm = float(np.linalg.norm(vector))
    reference_norm = float(np.linalg.norm(reference))

    norm_ratio = vector_norm / reference_norm if reference_norm > 0 else 0.0
    denom = vector_norm * reference_norm
    correlation = float(np.dot(vector, reference) / denom) if denom > 0 else 0.0

    return abs(norm_ratio - 1.0) + max(0.0, -correlation)
