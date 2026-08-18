"""Cross-check: an orbit bump measured and solved through `PhysicsBridge`
against the independent `lattice/response.py` model oracle (SC4).

Two physics code paths that never call each other meet here, the same way
`test_orm_crosscheck.py` pairs them. The *measured* path drives `PhysicsBridge`
-- the live IOC-facing setpoint -> orbit -> BPM-reading path, read through
`bind()`'s bound records so the seeded-error read pipeline
(`_push_bpm_readbacks`) is the one supplying the numbers -- and hands those
readings to `bump_analysis`, which fits the corrector -> BPM response from a
two-sided probe dither and solves it for the corrector offsets that produce a
requested bump. The *oracle* path evaluates `lattice.response.orbit_response`,
an offline model of the same corrector-kick -> closed-orbit response built and
cached independently in `response.py`'s own module-global ring, once per solved
offset. Agreement between them means the bump `bump_analysis` computed is a real
orbit bump on the modelled machine, not merely an internally consistent piece of
linear algebra.

Corrector and BPM device ids come from the manifest's pyat-coupled inventory
(`lattice.inventory`), never hardcoded preset channel names, per the epic's
VA-safety-test convention. Only the index positions of the chosen devices within
that inventory are fixed here, and only so the bump's geometry is reproducible;
see `CORRECTOR_INDICES` for how they were picked.

**Why the comparison is a bound and not an equality.** The oracle is evaluated
one corrector at a time, so the prediction it yields for a three-corrector bump
is the *linear superposition* of three single-corrector orbits. Superposition is
exact only on a linear ring, and the AR lattice is not one: its strong
low-emittance sextupoles feed down, so a kick set applied together does not
produce quite the sum of the orbits each kick produces alone (see
`calibration.py`'s `AMPS_PER_RADIAN_KICK` comment, and `test_lattice.py`'s
antisymmetry check for the same effect measured directly). The discrepancy grows
with amplitude, which is why every current in this test is held inside the
quasi-linear window that constant was chosen to give: a `PROBE_AMPLITUDE_A` of a
few amps and a solved bump whose offsets come out single-digit. The residual
that remains is then a small fraction of the bump, and `BOUND` is where the two
paths must agree.

This is the opposite tolerance regime from `test_orm_crosscheck`'s `<=1e-9`, and
deliberately so: that test compares two evaluations of the *same* estimator over
the *same* single-corrector sweep, so any disagreement is a code-path bug. Here
the two sides are genuinely different physics -- a simultaneous three-corrector
orbit versus the sum of three separate ones -- so a bound that admits the known
nonlinearity is the honest one. It is still a real check: the residual measured
below sits well inside `BOUND`, so a sign error, a mis-ordered probe pair, or a
wrong-row solve would blow through it by orders of magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from osprey.services.bluesky_bridge.bump_analysis import fit_probe_response, solve_offsets
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.lattice import inventory, orbit_response

# Positions within `inventory.pyat_coupled_device_ids()`'s sorted lists, not
# device names -- the manifest owns which devices exist.
#
# The three correctors are clustered inside one sixth of the ring (roughly
# s = 30 m .. 45 m of a 182 m circumference) with enough betatron phase between
# them to close a bump: the fitted response comes back well conditioned, and the
# solve asks for single-digit amps to make a `TARGET_BUMP_M` displacement. A
# tighter cluster is closer to a single combined kick and needs far more current
# for the same bump, which would push the run out of the quasi-linear window the
# superposition comparison depends on -- `test_solved_bump_stays_quasi_linear`
# is what fails first if a lattice change breaks that.
#
# BPMs and correctors are colocated in this lattice (72 of each, one pair per
# position), so the target BPM sits at the middle corrector, just upstream of
# it, and is the peak of the bump. The closure BPMs are spread around the rest
# of the ring, every one of them at least four positions clear of the corrector
# span.
CORRECTOR_INDICES = (11, 12, 17)
TARGET_BPM_INDEX = 12
CLOSURE_BPM_INDICES = (0, 22, 31, 40, 49, 58)

#: Probe dither half-amplitude, in amps. Large enough that the orbit it moves is
#: far above any readback resolution (tens of microns, per `calibration.py`),
#: small enough to stay quasi-linear.
PROBE_AMPLITUDE_A = 5.0

#: The bump asked for at the target BPM, in meters. Sized from the measured
#: response so the solve lands inside `QUASI_LINEAR_CURRENT_A`.
TARGET_BUMP_M = 20e-6

#: Reads taken with nothing moved, to establish the reference orbit the bump is
#: measured against and the per-BPM noise the bound floors at. Matches the plan
#: schema's `baseline_reads` default.
BASELINE_READS = 5

#: Agreement bound, as a fraction of the peak bump amplitude -- the tolerable
#: share of the bump that sextupole feed-down may account for at these currents.
RELATIVE_BOUND = 1e-2

#: Noise multiplier the bound floors at, so a BPM whose reading scatters is
#: never asked to agree to better than it can be read. The VA's read path is
#: deterministic unless a `noise_x` BPM error is seeded (see
#: `physics_bridge._push_bpm_readbacks`), so on this clean bridge the baseline
#: reads return identical values, sigma measures exactly zero, and the relative
#: term above is the operative bound. The floor is computed anyway because it is
#: the bound the plan itself works to, and it is what would take over on a
#: bridge that does add read noise.
NOISE_SIGMAS = 3.0

#: Current, in amps, above which the quasi-linear premise of this comparison no
#: longer holds -- `calibration.py` sets `AMPS_PER_RADIAN_KICK` so that a
#: corrector's typical +-10 A range stays in the small-signal regime.
QUASI_LINEAR_CURRENT_A = 10.0


class _FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`.

    Mirrors `test_orm_crosscheck.py`'s (and `test_physics_bridge.py`'s) record
    stub rather than importing one, keeping each VA test module's device
    scaffolding self-contained. Reading through `bind()`'s bound records instead
    of `bpm_positions()` (the physics-only truth) is what puts the seeded-error
    read pipeline in the loop, which is the path a real plan's BPM reads take.
    """

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@dataclass(frozen=True)
class _BumpMeasurement:
    """Everything one bump run produced, for the assertions to pick apart.

    `measured` and `predicted` are orbit *changes* away from the reference orbit
    -- the reference is measured, never assumed to be zero, exactly as
    `bump_analysis` documents its inputs.
    """

    bpm_names: list[str]
    offsets: np.ndarray
    measured: np.ndarray
    predicted: np.ndarray
    sigma: np.ndarray
    outside_span: np.ndarray

    @property
    def peak(self) -> float:
        """Largest orbit change the bump made anywhere on the ring."""
        return float(np.max(np.abs(self.measured)))

    @property
    def bound(self) -> np.ndarray:
        """Per-BPM agreement bound: `max(RELATIVE_BOUND * peak, NOISE_SIGMAS * sigma)`."""
        return np.maximum(RELATIVE_BOUND * self.peak, NOISE_SIGMAS * self.sigma)


def _sp_address(corrector: str) -> str:
    family, device = corrector[:3], corrector[3:]
    return f"SR:MAG:{family}:{device}:CURRENT:SP"


def _bpm_x_address(bpm: str) -> str:
    return f"SR:DIAG:BPM:{bpm[len('BPM') :]}:POSITION:X"


def _devices() -> tuple[list[str], list[str], list[int]]:
    """The bump's correctors, every BPM, and the constraint rows into that BPM list.

    Horizontal plane only: the bump is built from HCM correctors and read on the
    BPMs' X axis, so one value per BPM is what the probe rows carry -- the shape
    `fit_probe_response` fits and the shape an `orbit_bump_sweep` run's event
    documents have.
    """
    inv = inventory.pyat_coupled_device_ids()
    correctors = [f"HCM{inv['HCM'][i]}" for i in CORRECTOR_INDICES]
    bpms = [f"BPM{device}" for device in inv["BPM"]]
    constraint_rows = [TARGET_BPM_INDEX, *CLOSURE_BPM_INDICES]
    return correctors, bpms, constraint_rows


def _read_orbit(records: dict[str, _FakeRecord], bpms: list[str]) -> dict[str, float]:
    """One orbit read, shaped like an event document's `data` dict."""
    return {bpm: float(records[bpm].value) for bpm in bpms}


def _measure_bump() -> _BumpMeasurement:
    """Run the whole bump on the bridge and predict it from the oracle.

    Follows the plan's own order of operations: baseline reads first, then a
    two-sided probe of each corrector in turn with every other one left at its
    working point, then the solve, then the bump applied all at once.
    """
    correctors, bpms, constraint_rows = _devices()

    bridge = PhysicsBridge()
    records = {bpm: _FakeRecord() for bpm in bpms}
    bridge.bind({_bpm_x_address(bpm): record for bpm, record in records.items()})

    try:
        baseline = [_read_orbit(records, bpms) for _ in range(BASELINE_READS)]
        reference = np.array([baseline[-1][bpm] for bpm in bpms])
        sigma = np.std(np.array([[row[bpm] for bpm in bpms] for row in baseline]), axis=0, ddof=1)

        # Strictly [corr0+, corr0-, corr1+, corr1-, ...]: `fit_probe_response`
        # reads the pair for column j at rows 2j and 2j+1 and raises on any
        # other row count. Each corrector goes back to its working point (0 A on
        # this VA) before the next is probed, so every pair is a dither about
        # the same orbit.
        probe_rows: list[dict[str, float]] = []
        for corrector in correctors:
            try:
                for sign in (+1.0, -1.0):
                    bridge.on_setpoint(_sp_address(corrector), sign * PROBE_AMPLITUDE_A)
                    probe_rows.append(_read_orbit(records, bpms))
            finally:
                bridge.on_setpoint(_sp_address(corrector), 0.0)

        response = fit_probe_response(probe_rows, correctors, bpms, PROBE_AMPLITUDE_A)

        # Constraint rows only -- the target and the closure BPMs, sliced in the
        # same row order as `desired`. Every other BPM is a monitor here, read
        # and compared but never asked for, which is the split `solve_offsets`
        # requires of its caller.
        desired = np.zeros(len(constraint_rows))
        desired[0] = TARGET_BUMP_M
        offsets = solve_offsets(response[constraint_rows, :], desired)

        try:
            for corrector, offset in zip(correctors, offsets, strict=True):
                bridge.on_setpoint(_sp_address(corrector), float(offset))
            bumped = np.array([_read_orbit(records, bpms)[bpm] for bpm in bpms])
        finally:
            for corrector in correctors:
                bridge.on_setpoint(_sp_address(corrector), 0.0)
    finally:
        for corrector in correctors:
            bridge.on_setpoint(_sp_address(corrector), 0.0)

    # The oracle's own reference orbit: `orbit_response` at zero current is the
    # undisturbed closed orbit, which is not identically zero and must be
    # subtracted from each single-corrector orbit before they are summed.
    oracle_reference = orbit_response(correctors[0], 0.0)
    predicted = np.zeros(len(bpms))
    for corrector, offset in zip(correctors, offsets, strict=True):
        single = orbit_response(corrector, float(offset))
        predicted += np.array([single[bpm][0] - oracle_reference[bpm][0] for bpm in bpms])

    span = set(range(min(CORRECTOR_INDICES), max(CORRECTOR_INDICES) + 1))
    outside_span = np.array([i for i in range(len(bpms)) if i not in span])

    return _BumpMeasurement(
        bpm_names=bpms,
        offsets=offsets,
        measured=bumped - reference,
        predicted=predicted,
        sigma=sigma,
        outside_span=outside_span,
    )


@pytest.fixture(scope="module")
def bump() -> _BumpMeasurement:
    """One bump run, shared by every assertion below.

    Module-scoped because the run is a few dozen closed-orbit solves and the
    bridge is deterministic: re-running it per test would buy nothing but
    seconds. Nothing here mutates the result, and both the bridge and the
    oracle's ring are left with every corrector back at zero.
    """
    return _measure_bump()


def test_solved_bump_stays_quasi_linear(bump: _BumpMeasurement):
    """The solved offsets sit inside the window superposition is valid in.

    Checked first because it is the premise of every other assertion in this
    module rather than a claim about the bump: if a lattice or inventory change
    makes this corrector set need more current for `TARGET_BUMP_M`, the
    single-corrector orbits stop summing and the comparisons below would fail
    for a reason that has nothing to do with `bump_analysis`.
    """
    largest = float(np.max(np.abs(bump.offsets)))
    assert largest <= QUASI_LINEAR_CURRENT_A, (
        f"the bump needs {largest:.2f} A, past the {QUASI_LINEAR_CURRENT_A} A quasi-linear "
        f"window: superposition is no longer a valid prediction, so this crosscheck's "
        f"corrector set or target amplitude needs re-picking, not its bound loosening"
    )


def test_solved_bump_reaches_its_target(bump: _BumpMeasurement):
    """The bump the solve asked for is the bump the machine made.

    Guards the whole module against passing vacuously: a comparison of two
    orbits that both happen to be zero agrees perfectly and proves nothing.
    """
    reached = float(bump.measured[TARGET_BPM_INDEX])
    assert reached == pytest.approx(TARGET_BUMP_M, rel=RELATIVE_BOUND), (
        f"asked for a {TARGET_BUMP_M:.3e} m bump at {bump.bpm_names[TARGET_BPM_INDEX]}, "
        f"reached {reached:.3e} m"
    )
    assert bump.peak == pytest.approx(TARGET_BUMP_M, rel=RELATIVE_BOUND), (
        f"the bump peaks at {bump.peak:.3e} m, not at the target amplitude -- the orbit "
        f"excursion is not where it was asked for"
    )


def test_measured_bump_matches_lattice_superposition(bump: _BumpMeasurement):
    """SC4: the orbit the bridge produced == the oracle's superposition, per BPM."""
    residual = np.abs(bump.measured - bump.predicted)
    bound = bump.bound
    worst = int(np.argmax(residual / bound))
    assert np.all(residual <= bound), (
        f"BPM {bump.bpm_names[worst]} measured {bump.measured[worst]:.3e} m but the lattice "
        f"oracle's superposition predicts {bump.predicted[worst]:.3e} m -- a residual of "
        f"{residual[worst]:.3e} m against a bound of {bound[worst]:.3e} m "
        f"({residual[worst] / bound[worst]:.2f}x)"
    )


def test_bump_closes_outside_the_corrector_span(bump: _BumpMeasurement):
    """SC4: the bump is local -- the ring outside the correctors does not move.

    Closure is asserted over *every* BPM outside the span, not just the six the
    solve constrained. A bump that held its closure BPMs while leaking somewhere
    nobody listed would satisfy the solve and still not be a local bump, and
    that is precisely the failure the monitor BPMs exist to catch.
    """
    leaked = np.abs(bump.measured[bump.outside_span])
    bound = bump.bound[bump.outside_span]
    worst = int(np.argmax(leaked / bound))
    assert np.all(leaked <= bound), (
        f"the bump leaked {leaked[worst]:.3e} m at "
        f"{bump.bpm_names[bump.outside_span[worst]]}, outside the corrector span, against a "
        f"closure bound of {bound[worst]:.3e} m -- it is not a closed local bump"
    )
