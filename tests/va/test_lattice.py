"""Tests for the ALS-U AR PyAT lattice model (virtual accelerator).

Every numeric tolerance below is derived from a value measured against the
real ring while writing this file (see the comment beside each assertion) --
none of it is copied from the retired 24-cell toy-ring test suite.
"""

from __future__ import annotations

import time

import at
import pytest

from osprey.services.virtual_accelerator.lattice import build_ring, orbit_response
from osprey.services.virtual_accelerator.lattice.inventory import pyat_coupled_device_ids
from osprey.simulation.facility_spec import ALS_U_AR

# Measured mean/max solve time for one orbit_response() call on the dev
# machine this suite was written on was ~9.5-10.2 ms (10 warm samples, see
# TestSolveTimeBudget). CI hardware can be substantially slower, so the
# budget below is kept at roughly 10x that measurement rather than tightened
# to match it.
SOLVE_TIME_BUDGET_MS = 100.0

# Measured max|x_bpm + x_bpm(-I)| over all 72 BPMs for a +-5 A HCM01 kick was
# ~1.16e-7 m (sextupole feed-down breaks exact antisymmetry once the orbit
# excursion is large enough to sample the sextupoles' nonlinearity). 2e-6 is
# the required bound per the task spec, ~17x that measured residual.
ANTISYMMETRY_TOLERANCE_M = 2e-6

# Measured relative deviation from perfect linear scaling of the BPM01
# response to an HCM01 kick, comparing I=5 A doubled to I=10 A directly, was
# ~-0.08%; comparing I=1 A scaled x12 to I=12 A was ~-0.18%. 0.5% keeps
# several x margin over both measurements.
LINEARITY_RELATIVE_TOLERANCE = 5e-3

# Measured max cross-plane leakage (the plane a corrector should NOT move):
# HCM kicks (HCM01/10/30/50/70 at 10 A) left every BPM's y reading at exactly
# 0.0 -- with no roll/skew elements in the model, y is algebraically
# independent of an x-only perturbation. VCM kicks (VCM01/10/30/50/70 at
# 10 A) left BPM x readings at up to ~8.1e-8 m, consistent with closed-orbit
# solver noise rather than a real physical coupling. 1e-6 keeps an order of
# magnitude of margin over that VCM-side noise floor.
CROSS_PLANE_LEAKAGE_TOLERANCE_M = 1e-6


@pytest.fixture(scope="module")
def ring() -> at.Lattice:
    return build_ring()


@pytest.fixture(scope="module")
def device_inventory() -> dict[str, list[str]]:
    return pyat_coupled_device_ids()


class TestDeviceInventoryMatchesFacilitySpec:
    """The lattice's device inventory must match the ALS_U_AR facility spec
    exactly -- the spec fixes the lattice (and the manifest it's built
    from), not vice versa (see facility_spec.py's module docstring)."""

    @pytest.mark.parametrize("family", ALS_U_AR.family_names())
    def test_family_counts_match_spec(self, device_inventory, family):
        expected_count = ALS_U_AR.family(family).count
        assert len(device_inventory[family]) == expected_count

    def test_every_device_has_a_lattice_element(self, ring, device_inventory):
        fam_names = {el.FamName for el in ring}
        for family, ids in device_inventory.items():
            for device_id in ids:
                assert f"{family}{device_id}" in fam_names

    def test_element_counts_match_device_counts_exactly(self, ring, device_inventory):
        fam_names = [el.FamName for el in ring]
        for family, ids in device_inventory.items():
            count = sum(
                1 for name in fam_names if name.startswith(family) and name[len(family) :] in ids
            )
            assert count == len(ids), f"{family}: expected {len(ids)} elements, found {count}"


class TestClosedOrbitStability:
    def test_closed_orbit_exists_at_nominal_settings(self, ring):
        # Measured: at.find_orbit4 on the nominal (uncorrected) ring returns
        # exactly [0, 0, 0, 0, 0, 0] -- this is an ideal, unperturbed lattice
        # with no misalignments applied, so the closed orbit sits on-axis to
        # well under 1e-6 m at every BPM.
        orbit0, _ = at.find_orbit4(ring)
        assert orbit0 == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)

    def test_one_turn_matrix_is_stable_in_both_planes(self, ring):
        m44, _ = at.find_m44(ring)
        trace_x = m44[0, 0] + m44[1, 1]
        trace_y = m44[2, 2] + m44[3, 3]
        # |trace| < 2 is the standard stability criterion for a linear
        # transfer matrix (real, bounded betatron tune in that plane).
        # Measured on this ring: trace_x ~= 0.366, trace_y ~= -0.943.
        assert abs(trace_x) < 2.0, f"horizontal plane unstable: trace={trace_x}"
        assert abs(trace_y) < 2.0, f"vertical plane unstable: trace={trace_y}"


class TestOrbitResponse:
    """orbit_response's current->kick calibration (AMPS_PER_RADIAN_KICK,
    see lattice/response.py) is chosen so a corrector's typical +-10 A range
    stays in the small-signal/quasi-linear regime on the real AR lattice
    (tens of microns of orbit shift). What's tested here is the physically
    required shape of that response -- nonzero, linear to a measured
    tolerance, antisymmetric to a measured tolerance, and plane-decoupled --
    not a specific hand-picked number.
    """

    def test_positive_hcm_kick_moves_its_paired_bpm(self):
        # HCM<nn> is co-located immediately upstream of BPM<nn> in the same
        # straight section -- its own BPM is the unambiguous "downstream
        # BPM" for this corrector.
        readings = orbit_response("HCM01", 10.0)
        x, y = readings["BPM01"]
        assert x != 0.0
        assert y == pytest.approx(0.0)

    def test_response_is_antisymmetric_about_zero_current(self):
        # Measured max residual over all 72 BPMs for +-5 A on HCM01 was
        # ~1.16e-7 m; see ANTISYMMETRY_TOLERANCE_M above.
        positive = orbit_response("HCM01", 5.0)
        negative = orbit_response("HCM01", -5.0)
        for bpm_name in positive:
            resid = positive[bpm_name][0] + negative[bpm_name][0]
            assert abs(resid) < ANTISYMMETRY_TOLERANCE_M, (
                f"{bpm_name}: antisymmetry residual {resid} exceeds "
                f"{ANTISYMMETRY_TOLERANCE_M}"
            )

    def test_response_is_linear_in_current(self):
        # Measured: doubling I=5 A -> I=10 A on HCM01/BPM01 deviates from
        # perfect linear scaling by ~-0.08%; see
        # LINEARITY_RELATIVE_TOLERANCE above.
        base = orbit_response("HCM01", 5.0)["BPM01"][0]
        doubled = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert doubled == pytest.approx(2.0 * base, rel=LINEARITY_RELATIVE_TOLERANCE)

    def test_zero_current_gives_zero_response(self):
        x, y = orbit_response("HCM01", 0.0)["BPM01"]
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("corrector_name", ["HCM01", "HCM10", "HCM30", "HCM50", "HCM70"])
    def test_hcm_only_moves_horizontal_plane(self, corrector_name):
        # Measured: every one of these HCM correctors left all 72 BPMs' y
        # readings at exactly 0.0 at 10 A -- with no roll/skew elements in
        # this model, y is algebraically decoupled from an x-only kick.
        readings = orbit_response(corrector_name, 10.0)
        for bpm_name, (_x, y) in readings.items():
            assert abs(y) < CROSS_PLANE_LEAKAGE_TOLERANCE_M, (
                f"{bpm_name} y should be unaffected by an {corrector_name} kick, got {y}"
            )

    @pytest.mark.parametrize("corrector_name", ["VCM01", "VCM10", "VCM30", "VCM50", "VCM70"])
    def test_vcm_only_moves_vertical_plane(self, corrector_name):
        # Measured: these VCM correctors left BPM x readings at up to
        # ~8.1e-8 m (VCM50) at 10 A -- consistent with closed-orbit solver
        # noise, not a real physical coupling; see
        # CROSS_PLANE_LEAKAGE_TOLERANCE_M above.
        readings = orbit_response(corrector_name, 10.0)
        for bpm_name, (x, _y) in readings.items():
            assert abs(x) < CROSS_PLANE_LEAKAGE_TOLERANCE_M, (
                f"{bpm_name} x should be unaffected by a {corrector_name} kick, got {x}"
            )

    def test_vcm_paired_bpm_responds(self):
        readings = orbit_response("VCM05", 10.0)
        _x, y = readings["BPM05"]
        assert y != 0.0

    def test_corrector_kick_resets_between_calls(self):
        # A call with zero current after a nonzero one must not carry over
        # residual state (orbit_response resets KickAngle before returning).
        orbit_response("HCM01", 10.0)
        x, y = orbit_response("HCM01", 0.0)["BPM01"]
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_unknown_corrector_raises(self):
        with pytest.raises(ValueError):
            orbit_response("HCM99", 10.0)

    def test_malformed_corrector_name_raises(self):
        with pytest.raises(ValueError):
            orbit_response("QF01", 10.0)

    def test_returns_every_bpm(self):
        readings = orbit_response("HCM01", 10.0)
        assert len(readings) == ALS_U_AR.family("BPM").count
        assert all(name.startswith("BPM") for name in readings)


class TestSolveTimeBudget:
    def test_solve_time_under_budget(self):
        # Warm up (first call includes one-time import/JIT overhead).
        orbit_response("HCM01", 1.0)

        samples = []
        for i in range(10):
            t0 = time.perf_counter()
            orbit_response("HCM01", float(i))
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000.0)

        assert max(samples) < SOLVE_TIME_BUDGET_MS, samples
