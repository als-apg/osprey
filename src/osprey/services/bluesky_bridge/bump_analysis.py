"""Orbit-bump analysis: pure numpy, bluesky-free.

Consumes the plain `dict` rows an `orbit_bump_sweep` plan run emits (bluesky's
event document `data`, one dict per read -- see
`plans_core/orbit_bump_sweep.py`'s `build_plan`) plus plain numpy arrays, never
a live-buffer or Tiled-specific shape, so this module has no dependency on
`bluesky`/`ophyd-async`/`tiled` and imports cleanly on the MCP side without
loading the bluesky stack -- the same split `orm_analysis.py` makes, for the
same reason.

An orbit bump moves the stored beam locally through one element: three or four
correctors kick the beam off its reference orbit inside the bump region and
back onto it outside, so the ring beyond the last corrector never sees the
excursion. `orbit_bump_sweep` builds that bump without a lattice model, by
measuring the machine instead, and the pieces here are the arithmetic of each
measurement step:

- `fit_probe_response`: the corrector -> BPM response, fitted from a two-sided
  probe dither around each corrector's working point.
- `solve_offsets`: the constrained least-squares solve that turns a desired
  orbit into corrector offsets -- the bump itself.
- `demand_is_negligible`: the run-level gate deciding whether anything is
  actually being asked for, before any solve is attempted.
- `band_converged`: whether the machine landed inside the tolerance band.
- `band_violations`: which rows landed outside it -- the leakage check's form.
- `noise_floor_violations`: whether the requested band is measurable at all.

Everything here is expressed *relative to the measured reference orbit*: a
"desired" vector is an orbit change away from where the beam already is, and
"closed" means back inside the band around the reference, never literal zero.
A ring running a corrected orbit idles with nonzero corrector currents and a
nonzero orbit, and neither is knowable in advance -- which is exactly why the
plan measures both instead of assuming them.

BPM-key matching goes through the one shared resolver,
`plan_fields.resolve_column`: ophyd-async's `StandardReadable` keys a hinted
signal's document entry as either the bare channel name or
`f"{name}-{signal}"` depending on how many readable children the device
exposes (see `devices/mock.py`/`devices/epics.py`, neither of which this
module imports), so channel names resolve by exact key first, then by
`f"{name}-"` prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from osprey.services.bluesky_bridge.plan_fields import resolve_column


class DegenerateBumpError(ValueError):
    """The machine as measured cannot answer the bump that was asked for.

    Raised for a probe that read back non-finite or missing values, a probe set
    whose fitted response does not span the correctors being used (two
    correctors at the same phase, or a dead one), and a constraint set the
    response cannot resolve. Deliberately the module's only typed error: all of
    these are the same operational fact -- "this bump is not solvable on this
    machine as measured" -- and the plan reacts to every one of them the same
    way, by aborting before it writes anything and restoring the correctors.

    It is NOT raised for "nothing measurable happened". A response matrix that
    is identically zero says the probe moved no BPM at all, which is a fault
    only when something is actually being demanded; that judgment needs the
    reference orbit and the per-BPM noise, which only `demand_is_negligible`
    sees.
    """


def fit_probe_response(
    probe_rows: Sequence[Mapping[str, Any]],
    correctors: Sequence[str],
    bpm_names: Sequence[str],
    probe_amplitude: float,
) -> np.ndarray:
    """Fit the `[n_bpm, n_corr]` corrector -> BPM response from probe rows.

    *probe_rows* is one pair of rows per corrector, in `correctors` order:
    `probe_rows[2j]` is the orbit read with corrector `j` sitting at
    `working_point + probe_amplitude`, `probe_rows[2j + 1]` the orbit read with
    it at `working_point - probe_amplitude`, every other corrector left at its
    own working point throughout, and the corrector restored before the next
    one is probed. Each column is then the two-point antisymmetric slope

        response[:, j] = (y(+a) - y(-a)) / (2a)

    the centred difference quotient of the orbit with respect to that
    corrector -- the local linearization `solve_offsets` needs.

    Probing both sides is what makes that slope trustworthy, and it is the
    reason the plan spends two reads per corrector rather than one. A one-sided
    dither would measure `(y(+a) - y(0)) / a`, carrying the magnetic hysteresis
    offset between those two settings straight into the fitted slope. Any orbit
    component common to both readings -- the reference orbit itself, a slow
    drift, the remanent field left from however the corrector arrived at its
    working point -- cancels exactly in `y(+a) - y(-a)`. Only the part of the
    orbit that is antisymmetric in the applied kick survives, and that is
    precisely the response. It is also why a corrector that can only move one
    way from its working point is unsuitable for this plan: there is no
    symmetric pair to take.

    The fit divides by the *commanded* `2a` rather than by the correctors'
    recorded currents, so the caller owes the invariant that the pair really
    was applied symmetrically about the working point. A probe write the
    connector's limits check clamped breaks it, which is why the plan guards
    every probe write instead of handing a clamped pair here.

    The fit is a measurement, and it deliberately passes no judgment on the
    matrix it measured. Two correctors that share a response direction -- the
    same phase advance, or one of them dead -- leave it rank-deficient, and a
    bump solved through that would be arbitrary along the null direction; but
    whether a solve happens at all is decided *after* this fit, by
    `demand_is_negligible`, which alone sees the reference orbit and the
    noise. A run that asks for nothing walks its profile without solving, and
    refusing it here would also refuse every physics-free dry run: a mock
    readback that merely drifts between the `+a` and `-a` reads aliases into
    a phantom rank-one "response", the same way it would on any machine whose
    BPMs move for reasons other than the kick. The refusal lives in
    `solve_offsets`, the one place a degenerate response is actually asked to
    answer for a demand -- and every solve, the per-step first pass and every
    trim, goes through it.

    Args:
        probe_rows: `2 * len(correctors)` event `data` dicts, `(+probe,
            -probe)` per corrector in `correctors` order.
        correctors: Corrector device names, in probe order; the matrix's
            column axis.
        bpm_names: BPM device names; the matrix's row axis.
        probe_amplitude: The dither half-amplitude `a`, in corrector units.

    Raises:
        DegenerateBumpError: no correctors or no BPMs were given; the row
            count is not one pair per corrector; the probe amplitude is not
            finite and positive; or a BPM is missing from a probe row or read
            back a non-finite value.
    """
    n_corr = len(correctors)
    n_bpm = len(bpm_names)

    if n_corr == 0:
        raise DegenerateBumpError("no correctors were probed -- there is no response to fit")
    if n_bpm == 0:
        raise DegenerateBumpError("no BPMs were named -- there is nothing to fit a response onto")
    if not np.isfinite(probe_amplitude) or probe_amplitude <= 0.0:
        raise DegenerateBumpError(
            f"probe amplitude must be finite and positive, got {probe_amplitude!r}"
        )
    if len(probe_rows) != 2 * n_corr:
        raise DegenerateBumpError(
            f"expected one (+probe, -probe) row pair per corrector -- {2 * n_corr} rows for "
            f"{n_corr} correctors -- but got {len(probe_rows)}"
        )

    matrix = np.zeros((n_bpm, n_corr))
    for j, corrector in enumerate(correctors):
        plus_row = probe_rows[2 * j]
        minus_row = probe_rows[2 * j + 1]
        for i, bpm in enumerate(bpm_names):
            plus_column = resolve_column(bpm, plus_row)
            minus_column = resolve_column(bpm, minus_row)
            if plus_column is None or minus_column is None:
                raise DegenerateBumpError(
                    f"BPM {bpm!r} did not report in both of corrector {corrector!r}'s probe "
                    f"rows -- a half-measured column cannot be fitted"
                )
            plus_value = float(plus_row[plus_column])
            minus_value = float(minus_row[minus_column])
            if not (np.isfinite(plus_value) and np.isfinite(minus_value)):
                raise DegenerateBumpError(
                    f"BPM {bpm!r} read back a non-finite value during corrector "
                    f"{corrector!r}'s probe"
                )
            matrix[i, j] = (plus_value - minus_value) / (2.0 * probe_amplitude)

    # No rank judgment here -- see the docstring. The demand gate owns the
    # "was anything asked" call, and `solve_offsets` refuses a response that
    # cannot answer what was asked, on every solve including the trims.
    return matrix


def solve_offsets(
    response: np.ndarray,
    desired: np.ndarray,
    *,
    rcond: float = 1e-10,
) -> np.ndarray:
    """Solve `response @ offsets = desired` for the corrector offsets.

    This is the bump itself: *desired* is the orbit change wanted at each
    constrained BPM, relative to the measured reference orbit, and the returned
    `[n_corr]` vector is how far each corrector must move from its working
    point to produce it.

    *response* carries the **constraint rows only** -- the target BPMs and the
    closure BPMs, sliced by the caller out of `fit_probe_response`'s
    `[n_bpm, n_corr]` matrix in the same row order as *desired*. The plan's
    record-only monitors must not appear here. They are watched, not asked for,
    and including them would let the least-squares fit trade accuracy at a
    target the operator specified against a BPM nobody constrained.

    The system is overdetermined by design (more constraint rows than
    correctors), so there is generally no exact answer and `numpy.linalg.lstsq`
    returns the offsets minimising the total squared miss across the constraint
    rows -- spreading the unavoidable residual over the targets and the closure
    BPMs together. That is the right trade for a bump: closure is a constraint
    of the same standing as the target, not an afterthought, because a bump
    that hits its target while leaking outside the corrector span is not a
    local bump at all.

    The opposite case -- fewer independent constraints than correctors -- is
    the dangerous one, and it is prevented upstream rather than papered over
    here. `lstsq` would answer an underdetermined system with the
    minimum-norm solution: one arbitrary pick from an infinite family that all
    satisfy every listed constraint exactly and differ from each other at every
    BPM nobody listed. The run would report "converged" at each target and
    closure BPM while an unconstrained orbit distortion rides along ring-wide.
    So the plan's PARAMS validator rejects `len(targets) +
    len(closure_readbacks) < len(correctors)` at enqueue time, where it is a
    legible 400 against the request instead of a silent success in the middle
    of a run.

    That validator counts constraints; it cannot know their rank. Two closure
    BPMs at the same phase advance see the same orbit and contribute one
    constraint between them, so a nominally sufficient set can still come back
    rank-deficient from the real machine. The singular-value check below
    catches that and raises rather than quietly returning a minimum-norm
    answer, which is why *rcond* is a threshold to refuse at rather than a
    truncation level: no direction is ever silently dropped.

    Args:
        response: `[n_rows, n_corr]` measured response over the constraint
            rows, in the same row order as *desired*.
        desired: `[n_rows]` wanted orbit change, relative to the reference.
        rcond: Singular values at or below `rcond * largest_singular_value`
            count as degenerate directions.

    Raises:
        DegenerateBumpError: the response or demand is empty, mis-shaped,
            non-finite, or does not resolve every corrector independently at
            *rcond*; or *rcond* is not finite and positive.
    """
    matrix = np.asarray(response, dtype=float)
    demand = np.asarray(desired, dtype=float)

    if not np.isfinite(rcond) or rcond <= 0.0:
        raise DegenerateBumpError(f"rcond must be finite and positive, got {rcond!r}")
    if matrix.ndim != 2 or demand.ndim != 1:
        raise DegenerateBumpError("response must be 2-D and the desired orbit 1-D")
    if matrix.size == 0 or demand.size == 0:
        raise DegenerateBumpError("response and desired orbit must both be non-empty")
    if matrix.shape[0] != demand.shape[0]:
        raise DegenerateBumpError(
            f"response has {matrix.shape[0]} constraint rows but the desired orbit has "
            f"{demand.shape[0]} entries"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(demand)):
        raise DegenerateBumpError("response and desired orbit must both be finite")

    n_corr = matrix.shape[1]
    singulars = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singulars[0])
    rank = int(np.count_nonzero(singulars > rcond * largest)) if largest > 0.0 else 0
    if rank < n_corr:
        raise DegenerateBumpError(
            f"the constraint rows resolve only {rank} of {n_corr} correctors at rcond={rcond:g}: "
            f"the remaining direction(s) are unconstrained, and a least-squares answer would "
            f"pick one arbitrary bump out of infinitely many that all satisfy every constrained "
            f"BPM while differing everywhere else in the ring"
        )

    offsets, *_ = np.linalg.lstsq(matrix, demand, rcond=rcond)
    return offsets


def demand_is_negligible(
    desired: np.ndarray,
    response: np.ndarray,
    sigma: np.ndarray,
    tolerance: float,
) -> bool:
    """Is this run asking for an orbit change at all?

    The run-level gate, evaluated once on the unscaled targets -- never per
    amplitude step, where a scale-0 visit would trivially pass it and short out
    the verification the profile exists to do.

    `True` means the run is already converged before it starts: the beam sits
    inside the band at every constrained BPM, nothing needs solving, and the
    plan takes the trivially-converged path. That path is also what keeps a
    physics-free dry run honest. A mock connector's BPMs do not move when a
    corrector is written, so `fit_probe_response` hands back an identically
    zero response; with zero-offset targets nothing is being asked of it, and
    the run walks its full profile without ever calling `solve_offsets` -- the
    one function that would (correctly) refuse a zero response. Demand a real
    orbit change through that same zero response and this returns `False`, the
    solve runs, and the run aborts with `DegenerateBumpError` before writing.

    The band is `max(tolerance, 2σ)` per BPM. Tolerance alone is the operator's
    answer to "how close is close enough"; the `2σ` floor is the machine's, and
    it takes precedence because an orbit change smaller than the measurement
    noise is one nobody can verify was made. In practice tolerance already
    covers `2σ` everywhere by the time this runs -- `noise_floor_violations`
    fails the run fast otherwise, before any sweep write -- so the `maximum` is a
    backstop for a caller that gates differently, not the usual path.

    *response* is deliberately absent from the arithmetic, and cannot be part
    of it: it is measured in BPM units per corrector unit, while *desired*,
    *sigma* and *tolerance* are all in BPM units, and converting between them
    needs a corrector amplitude -- precisely what the solve produces and what
    this gate exists to avoid computing. It is taken and validated here because
    this gate is the last thing between the fit and the solve: `True` tells the
    plan to skip the solve, so a response that is non-finite or structurally
    unusable must never be waved through as trivially converged.

    Anything that makes the judgment unsupportable -- mismatched shapes,
    non-finite values, a non-positive tolerance -- returns `False` rather than
    raising. `False` is the conservative direction: it sends the run down the
    real solve path, which validates hard and raises a typed error naming the
    problem, whereas a claim of trivial convergence resting on data nobody
    could read would pass silently. (A caller with no constrained BPMs at all
    gets `True`, vacuously; PARAMS requires at least one target, so a real run
    never reaches that.)

    Args:
        desired: `[n_rows]` demanded orbit change over the constrained BPMs,
            relative to the measured reference.
        response: The fitted response matrix -- either the full
            `[n_bpm, n_corr]` fit or its constraint-row slice; the shapes are
            not coupled to *desired* because it is only checked for usability.
        sigma: `[n_rows]` per-BPM noise, aligned with *desired*, as measured
            from the baseline reads (ddof=1).
        tolerance: The requested convergence band, in BPM units.
    """
    demand = np.asarray(desired, dtype=float)
    noise = np.asarray(sigma, dtype=float)
    matrix = np.asarray(response, dtype=float)

    if demand.ndim != 1 or noise.shape != demand.shape:
        return False
    if matrix.ndim != 2 or matrix.size == 0:
        return False
    if not (np.all(np.isfinite(demand)) and np.all(np.isfinite(noise))):
        return False
    if not np.all(np.isfinite(matrix)):
        return False
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        return False

    band = np.maximum(tolerance, 2.0 * noise)
    return bool(np.all(np.abs(demand) <= band))


def band_violations(measured: np.ndarray, desired: np.ndarray, tolerance: float) -> list[int]:
    """Indices of rows sitting outside the `±tolerance` band around *desired*.

    The elementwise form of `band_converged`: a row is flagged when
    `|measured - desired| > tolerance`, with both vectors expressed relative
    to the measured reference orbit. `band_converged` answers "is every row
    in?"; this answers "which rows are out?", which is what an error message
    that names the offending BPMs needs. The leakage check runs it with an
    all-zero *desired* -- a monitor BPM is asked for nothing, so any offset
    beyond the band is leakage.

    Equality at the boundary is inside the band, exactly as `band_converged`
    counts it -- the two must agree row by row, since `band_converged` is
    implemented on top of this.

    Anything unreadable -- mismatched shapes, a non-finite reading, a
    non-positive tolerance -- flags *every* row rather than raising or
    returning none. Conservative in the same direction as `band_converged`'s
    `False` and `noise_floor_violations`' all-indices answer: a judgment that
    cannot be established must never read as "all clear".

    Args:
        measured: `[n_rows]` measured offsets, relative to the reference.
        desired: `[n_rows]` wanted offsets, relative to the reference.
        tolerance: The band's half-width, in BPM units.

    Returns:
        The offending indices into *measured*, ascending; empty when every row
        is inside the band.
    """
    got = np.asarray(measured, dtype=float)
    want = np.asarray(desired, dtype=float)

    if got.ndim != 1 or got.shape != want.shape:
        # "Every row" of a mis-shaped input is ill-defined, so flag row 0:
        # the answer must be non-empty for the caller's violation branch to
        # run, and there is no honest list of indices to hand it.
        return [0]
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        # A band nothing can satisfy: every row is out, and even an empty
        # input flags row 0 -- vacuous convergence into an invalid band is
        # still a claim nobody could check.
        return list(range(got.size)) if got.size else [0]

    unreadable = ~(np.isfinite(got) & np.isfinite(want))
    outside = np.zeros(got.shape, dtype=bool)
    finite = ~unreadable
    outside[finite] = np.abs(got[finite] - want[finite]) > tolerance
    return [int(index) for index in np.flatnonzero(unreadable | outside)]


def band_converged(measured: np.ndarray, desired: np.ndarray, tolerance: float) -> bool:
    """Did the machine land inside the tolerance band at every constraint row?

    `True` when `|measured - desired| <= tolerance` at every one of them --
    targets and closure BPMs alike, both expressed relative to the measured
    reference orbit. This is the step-level check the trim loop runs after each
    write: `False` means trim again (up to `max_trim_iterations`), `True` means
    record the step and move on.

    Equality at the boundary counts as converged. A reading exactly `tolerance`
    away is exactly as close as the operator asked for, and rejecting it would
    make the loop chase a band it was never promised.

    The band is exactly `±tolerance`, with no noise term folded in -- unlike
    `demand_is_negligible`'s, which floors at `2σ`. Two reasons it does not
    need one. The run has already passed `noise_floor_violations`, so
    `tolerance >= 2σ` at every constrained BPM: adding σ again would widen the
    band past what was asked for and quietly report convergence the operator
    did not get. And *measured* is the mean of two reads, so the noise on the
    number being tested is σ/√2, further inside the band than the floor
    assumed. Widening here would be double-counting in the one direction that
    matters.

    Anything unreadable -- mismatched shapes, a non-finite reading, a
    non-positive tolerance -- returns `False`. Not converged is the safe answer
    for a step whose outcome cannot be established: it trims, and if trimming
    cannot fix it the run stops or records a best-effort miss, either of which
    is visible. `True` would silently bank an unverified bump.

    Implemented as "no `band_violations`", so the two can never disagree about
    a row.
    """
    got = np.asarray(measured, dtype=float)
    want = np.asarray(desired, dtype=float)
    if got.ndim != 1 or got.shape != want.shape:
        return False

    return not band_violations(measured, desired, tolerance)


def noise_floor_violations(sigma: np.ndarray, tolerance: float) -> list[int]:
    """Indices of BPMs whose noise makes the requested band unverifiable.

    A BPM is flagged when `tolerance < 2σ`, with σ the per-BPM standard
    deviation the caller measured over the baseline reads (`ddof=1` -- these
    are a sample of the machine's noise, not the whole population of it). The
    plan runs this before it writes anything and refuses the run if the list is
    non-empty, because the alternative is worse than a rejection: a band
    narrower than the noise is one a perfectly converged orbit falls outside of
    by chance alone, so the trim loop would burn its iterations chasing noise
    and then report a failure the machine never had. At `tolerance >= 2σ` the
    band covers roughly the 95% interval of a single read, so a converged orbit
    reads as converged.

    **σ = 0 is never a violation.** A BPM whose readback quantizes coarsely
    enough returns the same number for every baseline read, and the sample
    standard deviation of identical values is exactly zero. That is a fact
    about the readback's resolution, not a claim that the BPM is infinitely
    precise, and not evidence that it is broken -- so the floor check skips it
    and its band stays exactly `±tolerance`. Reading σ = 0 as "infinitely
    precise" would wave through any tolerance however absurd; reading it as
    "broken" would fail every run on a facility with coarse BPM readbacks. Both
    are wrong about the same measurement; declining to draw a floor from it is
    the only honest option.

    A non-finite σ *is* flagged: the baseline never produced a noise estimate
    for that BPM, so nothing about its band can be certified. So is every BPM
    when *tolerance* itself is non-finite or non-positive, which no band can
    satisfy.

    Args:
        sigma: `[n_rows]` per-BPM noise over the constrained BPMs.
        tolerance: The requested convergence band, in BPM units.

    Returns:
        The offending indices into *sigma*, ascending; empty when the run can
        proceed.
    """
    noise = np.asarray(sigma, dtype=float)

    if not np.isfinite(tolerance) or tolerance <= 0.0:
        return list(range(noise.size))

    unmeasured = ~np.isfinite(noise)
    below_floor = np.isfinite(noise) & (noise > 0.0) & (tolerance < 2.0 * noise)
    return [int(index) for index in np.flatnonzero(unmeasured | below_floor)]
