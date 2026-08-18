"""Unit tests for `bump_analysis.py` (tasks 1.1-1.3): pure numpy, no bluesky
import anywhere in this file -- keep it that way so it always runs in the main
worktree venv, exactly as `test_orm_analysis.py` does.
"""

from __future__ import annotations

import numpy as np
import pytest

from osprey.services.bluesky_bridge.bump_analysis import (
    DegenerateBumpError,
    band_converged,
    demand_is_negligible,
    fit_probe_response,
    noise_floor_violations,
    solve_offsets,
)

CORRECTORS = ["corr1", "corr2", "corr3"]
BPMS = ["bpm1", "bpm2", "bpm3", "bpm4", "bpm5"]

PROBE_AMPLITUDE = 0.02

# A known [n_bpm, n_corr] response the synthetic probe rows below reproduce
# exactly (no noise). Full column rank, so the fit's degeneracy check stays
# quiet on it.
KNOWN_RESPONSE = np.array(
    [
        [0.5, -1.2, 0.3],
        [1.1, 0.4, -0.6],
        [-0.8, 0.9, 1.5],
        [0.2, -0.3, 0.7],
        [0.9, 1.4, -0.2],
    ]
)

# The orbit the ring is already holding when the probe starts -- nonzero, as a
# corrected orbit always is. Every probe reading sits on top of it.
REFERENCE_ORBIT = np.array([-3e-4, 1e-4, 4e-4, -2e-4, 5e-5])


def _probe_rows(
    response: np.ndarray,
    correctors: list[str],
    bpm_names: list[str],
    amplitude: float,
    *,
    reference: np.ndarray | None = None,
    key: str = "{name}",
) -> list[dict]:
    """Synthetic probe rows: one `(+probe, -probe)` pair per corrector, in
    corrector order, each carrying every BPM's reading -- the shape
    `orbit_bump_sweep`'s probe loop emits.
    """
    base = np.zeros(len(bpm_names)) if reference is None else reference
    rows = []
    for j in range(len(correctors)):
        for sign in (+1.0, -1.0):
            orbit = base + sign * amplitude * response[:, j]
            rows.append({key.format(name=bpm): float(orbit[i]) for i, bpm in enumerate(bpm_names)})
    return rows


# =========================================================================
# 1.1 fit_probe_response
# =========================================================================


def test_fit_recovers_a_known_probe_response() -> None:
    rows = _probe_rows(KNOWN_RESPONSE, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    result = fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    assert result.shape == (len(BPMS), len(CORRECTORS))
    assert np.allclose(result, KNOWN_RESPONSE)


def test_fit_cancels_an_orbit_offset_common_to_both_probe_legs() -> None:
    """The whole point of dithering to both sides: anything the two readings
    share -- the reference orbit the ring is already holding, a drift, the
    remanent field left from arriving at the working point -- subtracts out of
    `y(+a) - y(-a)` exactly, so only the response survives. A one-sided fit
    would carry all of it into the slope.
    """
    rows = _probe_rows(KNOWN_RESPONSE, CORRECTORS, BPMS, PROBE_AMPLITUDE, reference=REFERENCE_ORBIT)

    result = fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    assert np.allclose(result, KNOWN_RESPONSE)


def test_fit_matches_bpm_keys_by_device_name_prefix() -> None:
    """ophyd-async may key a hinted signal as `f"{device_name}-{signal}"`
    rather than the bare device name -- the fit must not depend on which.
    """
    rows = _probe_rows(
        KNOWN_RESPONSE,
        CORRECTORS,
        BPMS,
        PROBE_AMPLITUDE,
        reference=REFERENCE_ORBIT,
        key="{name}-value",
    )

    result = fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    assert np.allclose(result, KNOWN_RESPONSE)


def test_fit_raises_when_two_correctors_share_a_response_direction() -> None:
    """Two correctors at the same phase advance (here literally the same
    column) span one direction, not two: every bump solved through that
    response is arbitrary along the null direction, so the fit refuses it
    rather than handing the solve an unconstrained answer.
    """
    degenerate = KNOWN_RESPONSE.copy()
    degenerate[:, 2] = degenerate[:, 0]

    rows = _probe_rows(degenerate, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    with pytest.raises(DegenerateBumpError, match="independent orbit direction"):
        fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)


def test_fit_of_an_unresponsive_machine_is_all_zero_rather_than_an_error() -> None:
    """A probe that moves no BPM at all -- what a physics-free mock connector
    reads back -- is rank-deficient too, but it is not a degenerate corrector
    set. Whether it matters depends on whether anything is being demanded, and
    only the demand gate sees the reference orbit and the noise, so the fit
    returns the zero response and lets the gate decide.
    """
    rows = _probe_rows(np.zeros_like(KNOWN_RESPONSE), CORRECTORS, BPMS, PROBE_AMPLITUDE)

    result = fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    assert result.shape == (len(BPMS), len(CORRECTORS))
    assert np.all(result == 0.0)


def test_fit_raises_on_a_non_finite_probe_reading() -> None:
    rows = _probe_rows(KNOWN_RESPONSE, CORRECTORS, BPMS, PROBE_AMPLITUDE)
    rows[3]["bpm2"] = float("nan")

    with pytest.raises(DegenerateBumpError, match="non-finite"):
        fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)


def test_fit_raises_when_a_bpm_is_missing_from_one_probe_leg() -> None:
    """Only one side of the pair reported, so there is no difference to take --
    a half-measured column would otherwise fit as whatever the present leg
    happened to read.
    """
    rows = _probe_rows(KNOWN_RESPONSE, CORRECTORS, BPMS, PROBE_AMPLITUDE)
    del rows[1]["bpm4"]

    with pytest.raises(DegenerateBumpError, match="did not report in both"):
        fit_probe_response(rows, CORRECTORS, BPMS, PROBE_AMPLITUDE)


def test_fit_raises_when_rows_are_not_one_pair_per_corrector() -> None:
    rows = _probe_rows(KNOWN_RESPONSE, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    with pytest.raises(DegenerateBumpError, match="row pair per corrector"):
        fit_probe_response(rows[:-1], CORRECTORS, BPMS, PROBE_AMPLITUDE)


@pytest.mark.parametrize("amplitude", [0.0, -0.02, float("nan")])
def test_fit_raises_on_a_probe_amplitude_that_is_not_finite_and_positive(
    amplitude: float,
) -> None:
    rows = _probe_rows(KNOWN_RESPONSE, CORRECTORS, BPMS, PROBE_AMPLITUDE)

    with pytest.raises(DegenerateBumpError, match="finite and positive"):
        fit_probe_response(rows, CORRECTORS, BPMS, amplitude)


def test_fit_raises_without_correctors_or_bpms() -> None:
    with pytest.raises(DegenerateBumpError, match="no correctors"):
        fit_probe_response([], [], BPMS, PROBE_AMPLITUDE)

    with pytest.raises(DegenerateBumpError, match="no BPMs"):
        fit_probe_response([], CORRECTORS, [], PROBE_AMPLITUDE)


# =========================================================================
# 1.2 solve_offsets / demand_is_negligible
# =========================================================================


def _well_conditioned_response(n_rows: int, n_corr: int, seed: int) -> np.ndarray:
    """A `[n_rows, n_corr]` response with orthonormal columns.

    Perfectly conditioned on purpose: these tests measure the solve, not how
    gracefully it degrades on a badly-scaled machine.
    """
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n_rows, n_corr)))
    return q[:, :n_corr]


@pytest.mark.parametrize(
    ("n_corr", "seed"),
    [(3, 4321), (4, 8765)],
    ids=["three-correctors", "four-correctors"],
)
def test_solve_recovers_a_synthetic_bump_to_machine_precision(n_corr: int, seed: int) -> None:
    """Six constraint rows (targets plus closure BPMs) against three or four
    correctors: overdetermined, consistent, so the least-squares answer is the
    exact one that produced the demand.
    """
    response = _well_conditioned_response(6, n_corr, seed)
    true_offsets = np.linspace(0.5, -1.5, n_corr)
    desired = response @ true_offsets

    offsets = solve_offsets(response, desired)

    assert offsets.shape == (n_corr,)
    assert np.allclose(offsets, true_offsets, rtol=0.0, atol=1e-12)


def test_solve_spreads_the_residual_when_the_demand_is_unreachable() -> None:
    """An inconsistent demand -- one no combination of these correctors can
    produce -- has no exact answer, so the solve returns the least-squares
    compromise across every constraint row rather than nailing some rows and
    abandoning others.
    """
    response = _well_conditioned_response(6, 3, 2468)
    desired = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]) * 1e-4

    offsets = solve_offsets(response, desired)
    residual = response @ offsets - desired

    # Orthonormal columns: the least-squares residual is orthogonal to the
    # measured response, which is the defining property of the compromise.
    assert np.allclose(response.T @ residual, 0.0, atol=1e-15)


def test_solve_raises_when_constraints_are_fewer_than_correctors() -> None:
    """Two constraint rows cannot pin four correctors. `lstsq` would happily
    return the minimum-norm member of an infinite family -- every one of them
    exact at both listed BPMs and different everywhere else in the ring. The
    plan's PARAMS validator rejects this shape at enqueue; the solve refuses it
    again here rather than answering it.
    """
    response = _well_conditioned_response(4, 4, 1111)[:2, :]
    desired = np.array([1e-4, -2e-4])

    with pytest.raises(DegenerateBumpError, match="unconstrained"):
        solve_offsets(response, desired)


def test_solve_raises_when_two_constraint_rows_see_the_same_orbit() -> None:
    """Enough rows to pass the PARAMS constraint count, but not enough
    independent ones: duplicated rows carry one constraint between them, so a
    corrector direction is left unresolved.
    """
    response = _well_conditioned_response(4, 3, 3333)
    response[2, :] = response[1, :]
    response[3, :] = response[1, :]
    desired = response @ np.array([0.2, -0.4, 0.6])

    with pytest.raises(DegenerateBumpError, match="resolve only"):
        solve_offsets(response, desired)


def test_solve_raises_on_a_shape_mismatch() -> None:
    response = _well_conditioned_response(6, 3, 5555)

    with pytest.raises(DegenerateBumpError, match="constraint rows"):
        solve_offsets(response, np.zeros(5))


def test_solve_raises_on_non_finite_inputs() -> None:
    response = _well_conditioned_response(6, 3, 6666)
    desired = response @ np.array([0.1, 0.2, 0.3])

    with pytest.raises(DegenerateBumpError, match="finite"):
        solve_offsets(response, np.where(np.arange(6) == 2, np.nan, desired))

    poisoned = response.copy()
    poisoned[0, 0] = np.inf
    with pytest.raises(DegenerateBumpError, match="finite"):
        solve_offsets(poisoned, desired)


def test_solve_gate_calls_a_zero_offset_demand_negligible_with_zero_sigma() -> None:
    """The mock dry-run's shape: zero-offset targets, a BPM quiet enough to
    read back identical values (σ = 0), and a response that measured nothing
    because there is no physics behind the connector. Nothing is being asked,
    so no solve is attempted and the zero response never gets to raise.
    """
    zero_response = np.zeros((5, 3))

    assert demand_is_negligible(np.zeros(5), zero_response, np.zeros(5), tolerance=1e-4)


def test_solve_gate_calls_a_zero_offset_demand_negligible_with_nonzero_sigma() -> None:
    response = _well_conditioned_response(5, 3, 7777)
    sigma = np.array([1e-6, 3e-6, 2e-6, 5e-7, 4e-6])

    assert demand_is_negligible(np.zeros(5), response, sigma, tolerance=1e-4)


def test_solve_gate_calls_a_demand_beyond_the_band_non_negligible() -> None:
    response = _well_conditioned_response(5, 3, 8888)
    desired = np.array([0.0, 0.0, 5e-4, 0.0, 0.0])

    assert not demand_is_negligible(desired, response, np.zeros(5), tolerance=1e-4)


def test_solve_gate_widens_the_band_to_two_sigma() -> None:
    """A demand above the requested tolerance but under the noise the BPM
    actually shows is still not a demand anyone could verify. (The plan's
    fail-fast rejects tolerance < 2σ before any sweep write, so this configuration
    should never reach the gate on a real run -- the floor is a backstop.)
    """
    response = _well_conditioned_response(4, 3, 9999)
    sigma = np.full(4, 1e-4)
    desired = np.full(4, 1.5e-4)

    assert demand_is_negligible(desired, response, sigma, tolerance=1e-5)
    assert not demand_is_negligible(np.full(4, 2.5e-4), response, sigma, tolerance=1e-5)


def test_solve_gate_is_the_only_thing_holding_back_a_zero_response() -> None:
    """The contract the mock dry-run rests on, both halves in one place: the
    same unresponsive machine is fine when nothing is demanded and a typed
    abort when something is.
    """
    zero_response = np.zeros((5, 3))
    sigma = np.zeros(5)

    assert demand_is_negligible(np.zeros(5), zero_response, sigma, tolerance=1e-4)

    real_demand = np.array([2e-4, 0.0, -3e-4, 0.0, 0.0])
    assert not demand_is_negligible(real_demand, zero_response, sigma, tolerance=1e-4)
    with pytest.raises(DegenerateBumpError):
        solve_offsets(zero_response, real_demand)


@pytest.mark.parametrize(
    ("desired", "sigma", "tolerance"),
    [
        (np.array([np.nan, 0.0]), np.zeros(2), 1e-4),
        (np.zeros(2), np.array([np.inf, 0.0]), 1e-4),
        (np.zeros(2), np.zeros(3), 1e-4),
        (np.zeros(2), np.zeros(2), 0.0),
        (np.zeros(2), np.zeros(2), float("nan")),
    ],
    ids=["nan-demand", "inf-sigma", "shape-mismatch", "zero-tolerance", "nan-tolerance"],
)
def test_solve_gate_refuses_to_call_unreadable_inputs_negligible(
    desired: np.ndarray, sigma: np.ndarray, tolerance: float
) -> None:
    """False sends the run into the solve, which validates hard and raises a
    typed error naming the problem -- the loud direction. True would skip it.
    """
    response = _well_conditioned_response(2, 2, 1234)

    assert not demand_is_negligible(desired, response, sigma, tolerance)


def test_solve_gate_refuses_a_non_finite_response() -> None:
    response = _well_conditioned_response(4, 3, 2222)
    response[1, 2] = np.nan

    assert not demand_is_negligible(np.zeros(4), response, np.zeros(4), tolerance=1e-4)


# =========================================================================
# 1.3 band_converged / noise_floor_violations
# =========================================================================

TOLERANCE = 1e-4


def test_band_accepts_an_orbit_sitting_exactly_on_the_boundary() -> None:
    """A reading exactly `tolerance` from its target is exactly as close as
    the operator asked for, so it converges. Desired is zero here to keep the
    difference exact in floating point -- the boundary is the whole point.
    """
    desired = np.zeros(4)
    measured = np.array([TOLERANCE, -TOLERANCE, TOLERANCE, 0.0])

    assert band_converged(measured, desired, TOLERANCE)


def test_band_accepts_an_orbit_inside_it_at_every_constraint_row() -> None:
    desired = np.array([2e-4, 0.0, -3e-4, 1e-4])
    measured = desired + np.array([1e-5, -4e-5, 2e-5, 0.0])

    assert band_converged(measured, desired, TOLERANCE)


def test_band_rejects_an_orbit_outside_it_at_one_row() -> None:
    """Every constraint row has to be in band, not most of them: a closure BPM
    left outside is a bump that is not closed, however good the targets look.
    """
    desired = np.array([2e-4, 0.0, -3e-4, 1e-4])
    measured = desired + np.array([1e-5, -4e-5, 2e-5, 1.5e-4])

    assert not band_converged(measured, desired, TOLERANCE)


@pytest.mark.parametrize(
    ("measured", "desired", "tolerance"),
    [
        (np.array([np.nan, 0.0]), np.zeros(2), TOLERANCE),
        (np.zeros(2), np.array([0.0, np.inf]), TOLERANCE),
        (np.zeros(3), np.zeros(2), TOLERANCE),
        (np.zeros(2), np.zeros(2), 0.0),
        (np.zeros(2), np.zeros(2), float("nan")),
    ],
    ids=["nan-reading", "inf-target", "shape-mismatch", "zero-tolerance", "nan-tolerance"],
)
def test_band_is_not_converged_when_the_outcome_cannot_be_established(
    measured: np.ndarray, desired: np.ndarray, tolerance: float
) -> None:
    assert not band_converged(measured, desired, tolerance)


def test_band_noise_floor_is_empty_when_every_bpm_is_quiet_enough() -> None:
    sigma = np.array([1e-6, 4.9e-5, 0.0, 2e-5])

    assert noise_floor_violations(sigma, TOLERANCE) == []


def test_band_noise_floor_flags_only_the_offending_bpms() -> None:
    """Mixed noise across the constrained BPMs: only the ones whose 2σ exceeds
    the requested tolerance come back, by index, in order.
    """
    sigma = np.array([1e-6, 8e-5, 2e-5, 3e-4, 0.0, 5.1e-5])

    assert noise_floor_violations(sigma, TOLERANCE) == [1, 3, 5]


def test_band_noise_floor_treats_tolerance_equal_to_two_sigma_as_measurable() -> None:
    """The floor is `tolerance < 2σ`, so landing exactly on it passes -- the
    same boundary convention `band_converged` uses.
    """
    sigma = np.full(3, TOLERANCE / 2.0)

    assert noise_floor_violations(sigma, TOLERANCE) == []
    assert noise_floor_violations(sigma * 1.01, TOLERANCE) == [0, 1, 2]


def test_band_noise_floor_never_flags_a_quantized_bpm_reading_zero_sigma() -> None:
    """A BPM whose readback quantizes to identical baseline reads has σ = 0
    from `ddof=1`. That says its resolution is coarse, not that it is perfect
    and not that it is broken, so it is skipped however tight the tolerance --
    its band is simply ±tolerance.
    """
    sigma = np.zeros(4)

    assert noise_floor_violations(sigma, TOLERANCE) == []
    assert noise_floor_violations(sigma, 1e-12) == []


def test_band_noise_floor_flags_a_bpm_with_no_noise_estimate_at_all() -> None:
    """σ = 0 means "measured, and it did not move"; σ = nan means the baseline
    produced no estimate, so nothing about that BPM's band can be certified.
    """
    sigma = np.array([1e-6, np.nan, 2e-5, np.inf])

    assert noise_floor_violations(sigma, TOLERANCE) == [1, 3]


def test_band_noise_floor_flags_every_bpm_for_an_unusable_tolerance() -> None:
    sigma = np.zeros(3)

    assert noise_floor_violations(sigma, 0.0) == [0, 1, 2]
    assert noise_floor_violations(sigma, float("nan")) == [0, 1, 2]
