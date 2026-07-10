"""Tests for the pySC-ported VA error-model formulas (errors.py).

Each formula is checked against a known injected value under a seeded RNG,
with an explicit assertion of the AT sign/roll convention it relies on --
that convention is the ported-math risk this module exists to catch (see
errors.py's provenance docstring).
"""

from __future__ import annotations

import at
import numpy as np
import pytest

from osprey.services.virtual_accelerator.lattice.errors import (
    apply_misalignment,
    bpm_read,
    magnet_cal,
)
from osprey.services.virtual_accelerator.lattice.ring import build_ring


class TestBpmRead:
    """bpm_read ports pySC's BPMSystem.capture_orbit per-BPM chain: roll-mix,
    subtract offset, apply cal + polarity, add noise, then apply gain."""

    def _rng(self, seed: int = 0) -> np.random.Generator:
        return np.random.default_rng(seed)

    def test_recovers_true_minus_offset_times_gain(self):
        # Identity roll/cal/polarity, zero noise -- the formula collapses to
        # exactly (true - offset) * gain.
        rx, ry = bpm_read(
            1.0e-3,
            2.0e-3,
            offset_x=0.1e-3,
            offset_y=0.2e-3,
            gain_x=1.5,
            gain_y=0.8,
            polarity_x=1.0,
            polarity_y=1.0,
            roll=0.0,
            cal_x=0.0,
            cal_y=0.0,
            noise_x=0.0,
            noise_y=0.0,
            rng=self._rng(),
        )
        assert rx == pytest.approx((1.0e-3 - 0.1e-3) * 1.5)
        assert ry == pytest.approx((2.0e-3 - 0.2e-3) * 0.8)

    def test_calibration_error_is_fractional_multiplicative(self):
        rx, _ = bpm_read(
            1.0e-3,
            0.0,
            offset_x=0.0,
            offset_y=0.0,
            gain_x=1.0,
            gain_y=1.0,
            polarity_x=1.0,
            polarity_y=1.0,
            roll=0.0,
            cal_x=0.3,
            cal_y=0.0,
            noise_x=0.0,
            noise_y=0.0,
            rng=self._rng(),
        )
        assert rx == pytest.approx(1.0e-3 * 1.3)

    def test_polarity_flip_negates_the_reading(self):
        rx, ry = bpm_read(
            1.0e-3,
            2.0e-3,
            offset_x=0.0,
            offset_y=0.0,
            gain_x=1.0,
            gain_y=1.0,
            polarity_x=-1.0,
            polarity_y=1.0,
            roll=0.0,
            cal_x=0.0,
            cal_y=0.0,
            noise_x=0.0,
            noise_y=0.0,
            rng=self._rng(),
        )
        assert rx == pytest.approx(-1.0e-3)
        assert ry == pytest.approx(2.0e-3)

    def test_roll_convention_quarter_turn_swaps_axes(self):
        # AT/pySC sign convention: rotated_x = cos(roll)*x - sin(roll)*y,
        # rotated_y = sin(roll)*x + cos(roll)*y (pySC's `_rotation_matrix`,
        # [[cos, -sin], [sin, cos]]). At roll = +pi/2, a purely horizontal
        # true position reads as purely *vertical*, not the other way
        # around -- this is the explicit sign/roll convention assertion.
        rx, ry = bpm_read(
            1.0e-3,
            0.0,
            offset_x=0.0,
            offset_y=0.0,
            gain_x=1.0,
            gain_y=1.0,
            polarity_x=1.0,
            polarity_y=1.0,
            roll=np.pi / 2,
            cal_x=0.0,
            cal_y=0.0,
            noise_x=0.0,
            noise_y=0.0,
            rng=self._rng(),
        )
        assert rx == pytest.approx(0.0, abs=1e-12)
        assert ry == pytest.approx(1.0e-3)

    def test_gain_applies_after_noise_is_added(self):
        # A doubled gain must double the *entire* pre-gain quantity, including
        # any noise already added -- not just the true-position term.
        seed = 123
        drawn = np.random.default_rng(seed).normal(scale=5e-7)
        rx, _ = bpm_read(
            1.0e-3,
            0.0,
            offset_x=0.0,
            offset_y=0.0,
            gain_x=2.0,
            gain_y=1.0,
            polarity_x=1.0,
            polarity_y=1.0,
            roll=0.0,
            cal_x=0.0,
            cal_y=0.0,
            noise_x=5e-7,
            noise_y=0.0,
            rng=np.random.default_rng(seed),
        )
        assert rx == pytest.approx((1.0e-3 + drawn) * 2.0)

    def test_noise_is_drawn_from_the_passed_seeded_rng_deterministically(self):
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        result_a = bpm_read(
            0.0,
            0.0,
            offset_x=0.0,
            offset_y=0.0,
            gain_x=1.0,
            gain_y=1.0,
            polarity_x=1.0,
            polarity_y=1.0,
            roll=0.0,
            cal_x=0.0,
            cal_y=0.0,
            noise_x=1e-6,
            noise_y=1e-6,
            rng=rng_a,
        )
        result_b = bpm_read(
            0.0,
            0.0,
            offset_x=0.0,
            offset_y=0.0,
            gain_x=1.0,
            gain_y=1.0,
            polarity_x=1.0,
            polarity_y=1.0,
            roll=0.0,
            cal_x=0.0,
            cal_y=0.0,
            noise_x=1e-6,
            noise_y=1e-6,
            rng=rng_b,
        )
        assert result_a == result_b
        assert result_a != (0.0, 0.0)


class TestApplyMisalignment:
    """apply_misalignment ports pySC's sc_tools.update_transformation onto AT's
    T1/T2/R1/R2. Verified against a real closed-orbit solve (find_orbit4), not
    just against the transform matrices themselves -- the number that matters
    is the orbit shift it produces.
    """

    @pytest.fixture()
    def ring(self) -> at.Lattice:
        return build_ring()

    @pytest.fixture()
    def quad_index(self, ring: at.Lattice) -> int:
        for i, el in enumerate(ring):
            if el.FamName.startswith("QF"):
                return i
        raise AssertionError("no QF element in the SR lattice")

    @pytest.fixture()
    def dipole_index(self, ring: at.Lattice) -> int:
        for i, el in enumerate(ring):
            if el.FamName.startswith("DIPOLE"):
                return i
        raise AssertionError("no DIPOLE element in the SR lattice")

    def test_zero_misalignment_produces_zero_orbit_shift(self, ring, quad_index):
        apply_misalignment(ring[quad_index], dx=0.0, dy=0.0, roll=0.0)
        orbit0, _ = at.find_orbit4(ring)
        assert orbit0 == pytest.approx([0.0] * 6, abs=1e-12)

    def test_dx_misalignment_on_a_quad_produces_expected_orbit_shift(self, ring, quad_index):
        # Reference point from the plan (Task 2.2): a 300 um quad dx yields
        # ~83 um of peak horizontal orbit shift through find_orbit4 in AT
        # 0.7.1. Measured on this lattice: max|x| across all BPMs is
        # ~82.9 um for the same injected dx -- matches to within the
        # "~83 um" reference's own precision, confirming the ported
        # entrance/exit transform matches AT's sign convention.
        apply_misalignment(ring[quad_index], dx=300e-6)
        try:
            _, orbit_at_bpms = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[quad_index], dx=0.0)

        peak_x_shift = np.max(np.abs(orbit_at_bpms[:, 0]))
        assert peak_x_shift == pytest.approx(83e-6, rel=0.05)

        # A pure horizontal offset must not, by itself, induce any vertical
        # orbit distortion (no x-y coupling without roll).
        assert np.max(np.abs(orbit_at_bpms[:, 2])) == pytest.approx(0.0, abs=1e-12)

    def test_dy_misalignment_shifts_vertical_plane_only(self, ring, quad_index):
        apply_misalignment(ring[quad_index], dy=300e-6)
        try:
            _, orbit_at_bpms = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[quad_index], dy=0.0)

        assert np.max(np.abs(orbit_at_bpms[:, 2])) > 0.0
        assert np.max(np.abs(orbit_at_bpms[:, 0])) == pytest.approx(0.0, abs=1e-12)

    def test_reverting_to_zero_restores_the_original_closed_orbit(self, ring, quad_index):
        baseline, _ = at.find_orbit4(ring, refpts=at.Monitor)
        apply_misalignment(ring[quad_index], dx=200e-6, dy=150e-6, roll=0.02)
        at.find_orbit4(ring, refpts=at.Monitor)  # perturbed solve, discarded
        apply_misalignment(ring[quad_index], dx=0.0, dy=0.0, roll=0.0)
        reverted, _ = at.find_orbit4(ring, refpts=at.Monitor)
        assert reverted == pytest.approx(baseline, abs=1e-9)

    def test_roll_on_a_bend_couples_horizontal_offset_into_vertical_orbit(
        self, ring, dipole_index
    ):
        # sign/roll-convention assertion for the bend-aware exit transform:
        # a rolled dipole with a horizontal offset must couple some of that
        # offset into the vertical plane (x-y coupling via roll), which a
        # roll = 0 misalignment on the same element does not.
        apply_misalignment(ring[dipole_index], dx=100e-6, roll=0.0)
        try:
            _, unrolled = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[dipole_index], dx=0.0, roll=0.0)

        apply_misalignment(ring[dipole_index], dx=100e-6, roll=0.01)
        try:
            _, rolled = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[dipole_index], dx=0.0, roll=0.0)

        assert np.max(np.abs(unrolled[:, 2])) == pytest.approx(0.0, abs=1e-12)
        assert np.max(np.abs(rolled[:, 2])) > 0.0


class TestMagnetCal:
    """magnet_cal ports pySC's LinearConv.transform: setpoint*factor + offset."""

    def test_magnet_cal_identity_by_default(self):
        assert magnet_cal(4.2) == pytest.approx(4.2)

    def test_magnet_cal_matches_factor_times_setpoint_plus_offset(self):
        assert magnet_cal(10.0, factor=1.3, offset=0.5) == pytest.approx(10.0 * 1.3 + 0.5)

    def test_magnet_cal_polarity_flip_is_factor_negative_one(self):
        assert magnet_cal(7.5, factor=-1.0) == pytest.approx(-7.5)

    def test_magnet_cal_thirty_percent_calibration_error(self):
        assert magnet_cal(100.0, factor=1.3) == pytest.approx(130.0)

    def test_magnet_cal_offset_only(self):
        assert magnet_cal(0.0, offset=-0.05) == pytest.approx(-0.05)
