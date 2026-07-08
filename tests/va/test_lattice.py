"""Tests for the generic SR PyAT lattice model."""

from __future__ import annotations

import time

import at
import pytest

from osprey.services.virtual_accelerator.lattice import N_ARC_CELLS, build_ring, orbit_response
from osprey.services.virtual_accelerator.lattice.inventory import pyat_coupled_device_ids

SOLVE_TIME_BUDGET_MS = 100.0  # supports the synchronous FR3 recompute contract


@pytest.fixture(scope="module")
def ring() -> at.Lattice:
    return build_ring()


@pytest.fixture(scope="module")
def device_inventory() -> dict[str, list[str]]:
    return pyat_coupled_device_ids()


class TestDeviceInventoryMatchesManifest:
    """The lattice's device inventory must match manifest partition (a) exactly --
    the DB fixes the lattice, not vice versa."""

    @pytest.mark.parametrize(
        "family,expected_count",
        [
            ("DIPOLE", 24),
            ("QF", 16),
            ("QD", 16),
            ("SF", 12),
            ("SD", 12),
            ("HCM", 20),
            ("VCM", 20),
            ("BPM", 20),
        ],
    )
    def test_family_counts_match_manifest(self, device_inventory, family, expected_count):
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

    def test_dipole_count_matches_arc_cell_count(self, device_inventory):
        assert len(device_inventory["DIPOLE"]) == N_ARC_CELLS


class TestClosedOrbitStability:
    def test_closed_orbit_exists_at_nominal_settings(self, ring):
        orbit0, _ = at.find_orbit4(ring)
        assert orbit0 == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-9)

    def test_one_turn_matrix_is_stable_in_both_planes(self, ring):
        m44, _ = at.find_m44(ring)
        trace_x = m44[0, 0] + m44[1, 1]
        trace_y = m44[2, 2] + m44[3, 3]
        # |trace| < 2 is the standard stability criterion for a linear
        # transfer matrix (real, bounded betatron tune in that plane).
        assert abs(trace_x) < 2.0, f"horizontal plane unstable: trace={trace_x}"
        assert abs(trace_y) < 2.0, f"vertical plane unstable: trace={trace_y}"


class TestOrbitResponse:
    """orbit_response's kick calibration is a documented toy constant (see
    lattice/response.py); what's tested here is the physically-required shape
    of the response, not a specific hand-picked number: nonzero, linear, and
    antisymmetric about zero current, plus plane decoupling (HCM only moves x,
    VCM only moves y).
    """

    def test_positive_hcm_kick_moves_its_paired_bpm(self):
        # HCM<nn> is co-located immediately upstream of BPM<nn> in the same
        # straight section (see ring.py) -- its own BPM is the unambiguous
        # "downstream BPM" for this corrector.
        readings = orbit_response("HCM01", 10.0)
        x, y = readings["BPM01"]
        assert x != 0.0
        assert y == pytest.approx(0.0)

    def test_response_is_antisymmetric_about_zero_current(self):
        positive = orbit_response("HCM01", 10.0)["BPM01"]
        negative = orbit_response("HCM01", -10.0)["BPM01"]
        assert positive[0] == pytest.approx(-negative[0])

    def test_response_is_linear_in_current(self):
        base = orbit_response("HCM01", 10.0)["BPM01"][0]
        doubled = orbit_response("HCM01", 20.0)["BPM01"][0]
        assert doubled == pytest.approx(2.0 * base, rel=1e-6)

    def test_zero_current_gives_zero_response(self):
        x, y = orbit_response("HCM01", 0.0)["BPM01"]
        assert x == pytest.approx(0.0, abs=1e-12)
        assert y == pytest.approx(0.0, abs=1e-12)

    def test_hcm_only_moves_horizontal_plane(self):
        readings = orbit_response("HCM05", 10.0)
        for bpm_name, (_x, y) in readings.items():
            assert y == pytest.approx(0.0), f"{bpm_name} y should be unaffected by an HCM kick"

    def test_vcm_only_moves_vertical_plane(self):
        readings = orbit_response("VCM05", 10.0)
        for bpm_name, (x, _y) in readings.items():
            assert x == pytest.approx(0.0), f"{bpm_name} x should be unaffected by a VCM kick"

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
        assert len(readings) == 20
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
