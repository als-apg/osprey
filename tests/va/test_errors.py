"""Tests for the pySC-ported VA error-model formulas (errors.py).

Each formula is checked against a known injected value under a seeded RNG,
with an explicit assertion of the AT sign/roll convention it relies on --
that convention is the ported-math risk this module exists to catch (see
errors.py's provenance docstring).

The lattice these tests solve on is built here, in this module, and is no
facility's: the formulas are pure per-device arithmetic on an AT element, so
what they need is a ring that solves a closed orbit and carries a quadrupole
and a bend, and nothing they assert is a property of any particular
accelerator. A served tree would be the wrong fixture as well as a heavy one:
it is a deployment's artifact, and carrying one here would pin a facility's
optics into a test of two formulas. So every magnitude below is measured on
this ring and recorded per assertion, while the sign and decoupling
assertions -- the part that catches a bad port -- hold for any ring at all.
"""

from __future__ import annotations

import math
from typing import Any

import at
import numpy as np
import pytest

from osprey.services.virtual_accelerator.lattice.errors import (
    DeviceSeedError,
    apply_misalignment,
    bpm_read,
    magnet_cal,
    resolve_device_seeds,
)

#: The fixture ring's beam energy, in GeV, and its focusing strength.
FIXTURE_ENERGY_GEV = 3.0
QUAD_K = 1.1

#: Cells, and the harmonic number the closing cavity is built at.
CELLS = 8
HARMONIC = 88

C_LIGHT = 299792458.0


def _fixture_ring() -> at.Lattice:
    """A small stable FODO ring: quadrupoles, bends, monitors, correctors.

    Eight cells, each with a focusing and a defocusing quadrupole, a monitor,
    a corrector and two bends, closed by one cavity. Returned 4D, which is
    what ``at.find_orbit4`` solves and what a ring saved out of a facility's
    simulator model arrives as. Linear: no sextupoles, so the planes are
    decoupled unless a misalignment couples them -- which is what makes the
    roll assertion below a statement about the transform rather than about
    higher-order ring physics.
    """
    angle = 2 * math.pi / (2 * CELLS)
    elements: list[Any] = []
    for cell in range(1, CELLS + 1):
        elements += [
            at.Quadrupole(f"QF{cell}", 0.3, QUAD_K),
            at.Drift("DR", 1.0),
            at.Dipole(f"BD{cell}A", 1.0, angle),
            at.Monitor(f"MON{cell}"),
            at.Drift("DR", 1.0),
            at.Quadrupole(f"QD{cell}", 0.3, -QUAD_K),
            at.Drift("DR", 1.0),
            at.Corrector(f"HC{cell}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BD{cell}B", 1.0, angle),
            at.Drift("DR", 1.0),
        ]
    ring = at.Lattice(
        elements, name="errors-fixture", energy=FIXTURE_ENERGY_GEV * 1.0e9, periodicity=1
    )
    ring.append(
        at.RFCavity(
            "RFC", 0.0, 1.0e6, HARMONIC * C_LIGHT / ring.circumference, HARMONIC, ring.energy
        )
    )
    ring.disable_6d()
    return ring


class TestBpmRead:
    """bpm_read ports pySC's BPMSystem.capture_orbit per-BPM chain: roll-mix,
    subtract offset, apply cal + polarity, add noise, then apply gain.

    One unit throughout, and it is the monitor's own: the caller maps the
    solved orbit through the binding's ``monitor_inverse`` before it gets
    here, so the numbers below are in whatever the monitor publishes. The
    metre-looking magnitudes are only a scale the assertions happen to use --
    the formula never divides by one.
    """

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
        return _fixture_ring()

    @pytest.fixture()
    def quad_index(self, ring: at.Lattice) -> int:
        """The first quadrupole, found by AT element type rather than by name.

        The transform is geometry on an element, so which element carries it
        is a question about the lattice's physics and never about a family's
        spelling.
        """
        for i, element in enumerate(ring):
            if isinstance(element, at.Quadrupole):
                return i
        raise AssertionError("no quadrupole in the fixture ring")

    @pytest.fixture()
    def dipole_index(self, ring: at.Lattice) -> int:
        """The first bend, found by AT element type."""
        for i, element in enumerate(ring):
            if isinstance(element, at.Dipole):
                return i
        raise AssertionError("no bend in the fixture ring")

    def test_zero_misalignment_produces_zero_orbit_shift(self, ring, quad_index):
        apply_misalignment(ring[quad_index], dx=0.0, dy=0.0, roll=0.0)
        orbit0, _ = at.find_orbit4(ring)
        assert orbit0 == pytest.approx([0.0] * 6, abs=1e-12)

    def test_dx_misalignment_on_a_quad_produces_expected_orbit_shift(self, ring, quad_index):
        # Reference measured on the fixture ring: a 300 um dx on its first
        # quadrupole gives peak|x| across the eight monitors of
        # 2.479707185749027e-4 via find_orbit4, reproduced bit-for-bit across
        # repeated solves and linear in dx (50/100/300/600 um all land on the
        # same 0.8266 ratio), which is what a quadrupole offset must be.
        apply_misalignment(ring[quad_index], dx=300e-6)
        try:
            _, orbit_at_bpms = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[quad_index], dx=0.0)

        peak_x_shift = np.max(np.abs(orbit_at_bpms[:, 0]))
        assert peak_x_shift == pytest.approx(2.4797e-4, rel=0.02)

        # A pure horizontal offset must not, by itself, induce any vertical
        # orbit distortion (no x-y coupling without roll) -- measured exactly
        # 0.0.
        assert np.max(np.abs(orbit_at_bpms[:, 2])) == pytest.approx(0.0, abs=1e-12)

    def test_dy_misalignment_shifts_vertical_plane_only(self, ring, quad_index):
        apply_misalignment(ring[quad_index], dy=300e-6)
        try:
            _, orbit_at_bpms = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[quad_index], dy=0.0)

        peak_y = np.max(np.abs(orbit_at_bpms[:, 2]))
        peak_x = np.max(np.abs(orbit_at_bpms[:, 0]))

        # Measured on the fixture ring: 300 um dy gives
        # peak|y| = 2.595840377503067e-4, linear in dy.
        assert peak_y == pytest.approx(2.5958e-4, rel=0.02)

        # No horizontal orbit at all -- exactly 0.0, because this ring is
        # linear. A ring with sextupoles instead shows a second-order
        # horizontal feed-down off the nonzero vertical orbit (a normal
        # sextupole's kick_x ~ -S*(x^2 - y^2)/2 leaks through the -y^2 term
        # even with x_co = 0): genuine ring physics, and a property of that
        # ring rather than of this transform, which is why the fixture here
        # carries no sextupole.
        assert peak_x == pytest.approx(0.0, abs=1e-12)

    def test_reverting_to_zero_restores_the_original_closed_orbit(self, ring, quad_index):
        baseline, _ = at.find_orbit4(ring, refpts=at.Monitor)
        apply_misalignment(ring[quad_index], dx=200e-6, dy=150e-6, roll=0.02)
        at.find_orbit4(ring, refpts=at.Monitor)  # perturbed solve, discarded
        apply_misalignment(ring[quad_index], dx=0.0, dy=0.0, roll=0.0)
        reverted, _ = at.find_orbit4(ring, refpts=at.Monitor)
        assert reverted == pytest.approx(baseline, abs=1e-9)

    def test_roll_on_a_bend_couples_horizontal_offset_into_vertical_orbit(self, ring, dipole_index):
        # sign/roll-convention assertion for the bend-aware exit transform: a
        # rolled bend steers into the vertical plane -- its bending plane is
        # tilted, so part of its angle becomes a vertical kick -- which the
        # same misalignment at roll = 0 does not. The bend is offset
        # horizontally as well, so the transform is exercised on both degrees
        # of freedom at once, but the vertical orbit is the roll's: measured
        # on this ring, 1.380880e-3 with the 100 um dx and 1.380928e-3
        # without it.
        apply_misalignment(ring[dipole_index], dx=100e-6, roll=0.0)
        try:
            _, unrolled = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[dipole_index], dx=0.0, roll=0.0)

        apply_misalignment(ring[dipole_index], dx=100e-6, roll=0.001)
        try:
            _, rolled_pos = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[dipole_index], dx=0.0, roll=0.0)

        # Explicit sign-convention check: negating roll (same |roll|, same
        # dx) must negate the induced vertical orbit at every monitor.
        # Measured on this lattice: rolled_pos[:, 2] and -rolled_neg[:, 2]
        # agree to ~1e-16 (roll's contribution is an odd function of roll
        # here), which is what confirms the exit transform's roll sign matches
        # bpm_read's documented convention rather than the opposite sign --
        # the thing this test exists to catch, and on a bend, where the exit
        # transform is angle-aware and the sign cannot be assumed to survive
        # from a straight element.
        apply_misalignment(ring[dipole_index], dx=100e-6, roll=-0.001)
        try:
            _, rolled_neg = at.find_orbit4(ring, refpts=at.Monitor)
        finally:
            apply_misalignment(ring[dipole_index], dx=0.0, roll=0.0)

        assert np.max(np.abs(unrolled[:, 2])) == pytest.approx(0.0, abs=1e-12)
        assert np.max(np.abs(rolled_pos[:, 2])) > 0.0
        assert rolled_pos[:, 2] == pytest.approx(-rolled_neg[:, 2], abs=1e-9)


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


def _resolve(
    seeded: dict[str, dict[str, float]], monitors: dict[str, str]
) -> dict[str, dict[str, float]]:
    """The monitor-flavoured resolution, spelled as the boot spells it.

    The resolver serves magnets on the same terms, and the nouns it is given
    are what its refusals read as; the magnet side is exercised where the boot
    wires it, in ``test_serving_entrypoint.py``.
    """
    return resolve_device_seeds(seeded, monitors, device="monitor", seeds="readout errors")


class TestResolveBpmErrors:
    """The resolution turns the seeded ``DEV:field=value`` map into the
    element-keyed one the model holds its faults by, against the monitors the
    bindings document actually publishes.

    Two spellings resolve, because two are in use: an operator seeding a fault
    knows the address the reading is published on, while the deck and the
    bridge know the element the monitor sits at. A spelling that is neither is
    refused here, at boot, rather than serving a perfectly unperturbed machine
    that looks configured.
    """

    #: Two planes of one monitor and one plane of another -- the mapping a
    #: caller builds from the document's monitor bindings, one entry each.
    MONITORS = {
        "SR:DIAG:BPM:12:POSITION:X": "bpm_12_1",
        "SR:DIAG:BPM:12:POSITION:Y": "bpm_12_1",
        "SR:DIAG:BPM:13:POSITION:X": "bpm_13_1",
    }

    def test_an_address_resolves_to_the_element_its_monitor_sits_at(self):
        resolved = _resolve({"SR:DIAG:BPM:12:POSITION:X": {"offset_x": 50e-6}}, self.MONITORS)
        assert resolved == {"bpm_12_1": {"offset_x": pytest.approx(50e-6)}}

    def test_an_element_name_resolves_to_itself(self):
        resolved = _resolve({"bpm_13_1": {"gain_x": 1.05}}, self.MONITORS)
        assert resolved == {"bpm_13_1": {"gain_x": pytest.approx(1.05)}}

    def test_both_planes_of_one_monitor_seed_one_error_model(self):
        # A reading is a pair -- bpm_read mixes the planes through the
        # monitor's roll -- so the two addresses of one device resolve to one
        # entry rather than to two half-configured ones.
        resolved = _resolve(
            {
                "SR:DIAG:BPM:12:POSITION:X": {"offset_x": 50e-6},
                "SR:DIAG:BPM:12:POSITION:Y": {"offset_y": 30e-6, "roll": 0.01},
            },
            self.MONITORS,
        )
        assert resolved == {
            "bpm_12_1": {
                "offset_x": pytest.approx(50e-6),
                "offset_y": pytest.approx(30e-6),
                "roll": pytest.approx(0.01),
            }
        }

    def test_a_field_names_its_own_plane_whichever_address_carried_it(self):
        # The address says which device, never which plane: the plane is the
        # field's own name. Seeding a vertical offset through the horizontal
        # address is therefore the device's vertical offset, not a refusal.
        resolved = _resolve({"SR:DIAG:BPM:12:POSITION:X": {"offset_y": 30e-6}}, self.MONITORS)
        assert resolved == {"bpm_12_1": {"offset_y": pytest.approx(30e-6)}}

    def test_an_unknown_device_is_refused_and_named(self):
        with pytest.raises(DeviceSeedError, match="BPM99"):
            _resolve({"BPM99": {"offset_x": 50e-6}}, self.MONITORS)

    def test_one_field_seeded_through_two_spellings_is_refused(self):
        # The address and the element are one device, so two tokens setting one
        # field is a configuration the operator cannot have meant; picking
        # either by dict order would be a guess.
        with pytest.raises(DeviceSeedError, match="offset_x"):
            _resolve(
                {
                    "SR:DIAG:BPM:12:POSITION:X": {"offset_x": 50e-6},
                    "bpm_12_1": {"offset_x": 10e-6},
                },
                self.MONITORS,
            )

    def test_different_fields_through_two_spellings_merge(self):
        resolved = _resolve(
            {
                "SR:DIAG:BPM:12:POSITION:X": {"offset_x": 50e-6},
                "bpm_12_1": {"gain_y": 0.9},
            },
            self.MONITORS,
        )
        assert resolved == {
            "bpm_12_1": {"offset_x": pytest.approx(50e-6), "gain_y": pytest.approx(0.9)}
        }

    def test_values_are_carried_across_unchanged_and_unbounded(self):
        # Resolution moves the key and nothing else: whatever the parse step
        # accepted arrives here as written, and a value is never converted or
        # clamped on the way through.
        resolved = _resolve({"bpm_13_1": {"offset_x": 1.0e3}}, self.MONITORS)
        assert resolved == {"bpm_13_1": {"offset_x": pytest.approx(1.0e3)}}

    def test_only_the_seeded_fields_are_returned(self):
        # A partial override, not a full error model: the unseeded fields fall
        # back to identity where the model is applied, so nothing here has to
        # restate what an unperturbed monitor reads.
        resolved = _resolve({"bpm_13_1": {"gain_x": 1.05}}, self.MONITORS)
        assert list(resolved["bpm_13_1"]) == ["gain_x"]

    def test_nothing_seeded_resolves_to_nothing(self):
        assert _resolve({}, self.MONITORS) == {}
        assert _resolve({}, {}) == {}

    def test_a_seed_against_a_document_with_no_monitors_is_refused(self):
        with pytest.raises(DeviceSeedError, match="bpm_13_1"):
            _resolve({"bpm_13_1": {"gain_x": 1.05}}, {})

    def test_the_result_does_not_alias_the_seeded_mapping(self):
        seeded = {"bpm_13_1": {"gain_x": 1.05}}
        resolved = _resolve(seeded, self.MONITORS)
        resolved["bpm_13_1"]["gain_x"] = 2.0
        assert seeded == {"bpm_13_1": {"gain_x": 1.05}}
