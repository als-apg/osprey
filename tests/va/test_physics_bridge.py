"""Tests for the SR magnet setpoint -> PyAT orbit-recompute physics bridge.

Unlike test_record_factory.py, this module never imports ioc.records (hence
never touches softioc.builder): PhysicsBridge itself has no softioc
dependency (see physics_bridge.py -- it only imports `at` and `lattice`), so
these tests exercise it directly. The one test that verifies the `bind()`
wiring contract uses a minimal duck-typed fake record (just a `.set()`
method) rather than a real softioc record, sidestepping the process-global
softioc/CA gotchas documented in test_record_factory.py entirely -- the real
end-to-end wiring through live softioc records is orbit-response-e2e's job.

This module targets the real ALS-U AR ring (`lattice.build_ring()`), not a
toy lattice: nominal currents are the per-device values baked into
`machine.json` (there are no `NOMINAL_*_CURRENT_A` module constants to import
-- see `_nominal_current` below), device counts/families come from
`osprey.simulation.facility_spec.ALS_U_AR`, and every "away from nominal"
setpoint used here was probed against the real optics (see each test's
comment) rather than carried over from the old toy-ring numbers, since the
real ring's stability/NaN boundaries sit much closer to nominal than the
toy ring's did.
"""

from __future__ import annotations

from functools import cache

import pytest

from osprey.services.virtual_accelerator.ioc.physics_bridge import (
    OrbitSolveError,
    PhysicsBridge,
    UnknownDeviceError,
)
from osprey.services.virtual_accelerator.lattice import orbit_response
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels
from osprey.simulation.facility_spec import ALS_U_AR


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@cache
def _nominal_current(address: str) -> float:
    """Return the machine.json nominal (baseline) current for a CURRENT:SP address.

    Reads the same scenario-seed file `StrengthMap` calibrates itself from
    (see `strengths.py`), so a test asserting "nominal current reproduces the
    ideal orbit" exercises the real per-device baseline, not a hardcoded
    guess. Cached: `machine.json` is static for the process lifetime and this
    is called from many tests.
    """
    return float(load_machine_json_channels()[address]["value"])


@pytest.fixture
def bridge() -> PhysicsBridge:
    return PhysicsBridge()


class TestNominalState:
    def test_nominal_orbit_is_zero(self, bridge):
        for address, value in bridge.bpm_positions().items():
            assert value == pytest.approx(0.0, abs=1e-9), address

    @pytest.mark.parametrize("family", ["QF", "QD", "QFA", "DIPOLE", "SF", "SD", "SHF", "SHD"])
    def test_writing_nominal_current_reproduces_zero_orbit(self, bridge, family):
        # Measured: writing every one of the 8 magnet families' device 01 at
        # its machine.json nominal current gives an *exactly* zero orbit
        # (this is an ideal, unmisaligned ring -- there is nothing to correct
        # for), not merely a small residual. 1e-9 m matches
        # test_nominal_orbit_is_zero's tolerance and leaves headroom for
        # solver floating-point noise, while still being far tighter than the
        # 1e-6 m "ideal" bar from the task brief.
        address = f"SR:MAG:{family}:01:CURRENT:SP"
        bridge.on_setpoint(address, _nominal_current(address))
        for bpm_address, value in bridge.bpm_positions().items():
            assert value == pytest.approx(0.0, abs=1e-9), bpm_address


class TestSetpointWriteMovesBpm:
    """FR3/SC3: an SP write synchronously updates the BPM RB, in the direction
    the (independently implemented, task 3.2) lattice module predicts."""

    def test_hcm_write_changes_bpm_on_return_matching_lattice_prediction(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        actual = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        assert actual != 0.0
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        # Measured: the bridge and the oracle delegate to the identical
        # StrengthMap.apply + solve_orbit code path, so they agree bit for
        # bit (diff == 0.0 measured); abs=1e-12 keeps a tight but non-zero
        # tolerance rather than asserting exact float equality.
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_vcm_write_changes_bpm_on_return_matching_lattice_prediction(self, bridge):
        bridge.on_setpoint("SR:MAG:VCM:05:CURRENT:SP", 10.0)
        actual = bridge.bpm_positions()["SR:DIAG:BPM:05:POSITION:Y"]

        assert actual != 0.0
        expected = orbit_response("VCM05", 10.0)["BPM05"][1]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_hcm_write_only_moves_x(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:Y"] == pytest.approx(0.0)

    def test_vcm_write_only_moves_y(self, bridge):
        # Measured on the real ring: a VCM kick leaves a tiny (~-7.8e-9 m)
        # horizontal leakage at BPM01 -- real skew coupling the toy ring
        # didn't carry (its HCM->Y leakage measured exactly 0.0, confirming
        # the asymmetry is a real-ring effect, not a bridge bug). abs=1e-7
        # stays two orders of magnitude above that measured leakage while
        # still catching any leakage at the corrector-response (1e-5 m)
        # scale.
        bridge.on_setpoint("SR:MAG:VCM:01:CURRENT:SP", 10.0)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(0.0, abs=1e-7)

    def test_qf_write_away_from_nominal_changes_orbit_response(self, bridge):
        # QF/QD have no independent readback-only "orbit_response" oracle
        # (lattice.orbit_response only covers correctors) -- the meaningful,
        # non-tautological check is that a kicked corrector's downstream
        # response measurably changes when the optics (QF gradient) change,
        # since the closed-orbit response depends on the whole ring's optics.
        #
        # Measured on the real ring: QF01 at 1.2x nominal is the last stable
        # multiplier probed below 1.3x (trace_x jumps to 2.65, unstable);
        # 1.1x is used here to leave comfortable margin below that boundary
        # while still perturbing the gradient enough to move the response
        # (real ring: 1.5x nominal -- the old toy-ring test's multiplier --
        # is already well past the instability boundary, so it is not reused).
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        response_at_nominal_qf = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        qf_nominal = _nominal_current("SR:MAG:QF:01:CURRENT:SP")
        bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", qf_nominal * 1.1)
        response_after_qf_change = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        assert response_after_qf_change != pytest.approx(response_at_nominal_qf)

    def test_dipole_write_away_from_nominal_changes_orbit_response(self, bridge):
        # Measured on the real ring: DIPOLE01's trim model (PolynomB[0] =
        # (I/I_nom - 1) * BendingAngle / Length) is far more sensitive than
        # the toy ring's -- 1.1x nominal already produces a non-finite
        # one-turn matrix (find_m44 NaN), so 1.05x (still stable, orbit shift
        # ~13 mm) is used here instead of the old toy-ring 1.2x multiplier.
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        response_at_nominal_dipole = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        dipole_nominal = _nominal_current("SR:MAG:DIPOLE:01:CURRENT:SP")
        bridge.on_setpoint("SR:MAG:DIPOLE:01:CURRENT:SP", dipole_nominal * 1.05)
        response_after_dipole_change = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        assert response_after_dipole_change != pytest.approx(response_at_nominal_dipole)


class TestWriteComposition:
    """SC3: two rapid sequential writes give the same final state as their composition."""

    def test_overwriting_the_same_device_is_idempotent_not_cumulative(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 8.0)

        actual = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        expected = orbit_response("HCM01", 8.0)["BPM01"][0]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_two_independent_devices_are_order_independent(self):
        forward = PhysicsBridge()
        forward.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        forward.on_setpoint("SR:MAG:VCM:07:CURRENT:SP", -6.0)

        reverse = PhysicsBridge()
        reverse.on_setpoint("SR:MAG:VCM:07:CURRENT:SP", -6.0)
        reverse.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        assert forward.bpm_positions() == reverse.bpm_positions()

    def test_composed_writes_match_writing_final_values_directly(self):
        # Write HCM01 twice (transient 3.0A, then settle at 10.0A) then VCM07
        # once; the final state must equal writing the settled values in one
        # shot each, regardless of the transient in between.
        sequential = PhysicsBridge()
        sequential.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 3.0)
        sequential.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        sequential.on_setpoint("SR:MAG:VCM:07:CURRENT:SP", -6.0)

        direct = PhysicsBridge()
        direct.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        direct.on_setpoint("SR:MAG:VCM:07:CURRENT:SP", -6.0)

        assert sequential.bpm_positions() == direct.bpm_positions()


class TestInstabilityRollback:
    def test_unstable_write_is_rejected_and_state_is_rolled_back(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        before_orbit = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        qf_idx = bridge._element_index("QF01")
        before_k = bridge._ring[qf_idx].K

        # Measured on the real ring: QF01 at 2x nominal is already unstable
        # (|trace_x| = 9.4); 5x nominal (|trace_x| = 34.0) is used here for a
        # robust margin over that boundary while still resolving via the
        # trace-instability guard condition (not the non-finite/NaN guard --
        # see TestNaNWriteRollback below for that distinct failure mode).
        qf_nominal = _nominal_current("SR:MAG:QF:01:CURRENT:SP")
        with pytest.raises(OrbitSolveError):
            bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", qf_nominal * 5.0)

        after_orbit = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert after_orbit == before_orbit

        after_k = bridge._ring[qf_idx].K
        assert after_k == before_k

    def test_bridge_remains_usable_after_a_rejected_write(self, bridge):
        qf_nominal = _nominal_current("SR:MAG:QF:01:CURRENT:SP")
        with pytest.raises(OrbitSolveError):
            bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", qf_nominal * 5.0)

        # A subsequent, valid write must still work normally.
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(
            expected, abs=1e-12
        )


class TestNaNWriteRollback:
    """SC6: a NaN-producing write (a distinct guard condition from a merely
    unstable/high-trace one-turn map -- see solve.py's three-guard docstring)
    must roll back exactly like the trace-instability case, restoring the
    element's PolynomB *elementwise*, even when the orbit was already
    nonzero (a kicked corrector) at the time of the failed write.

    Measured NaN recipe: DIPOLE01's trim model (PolynomB[0] = (I/I_nom - 1) *
    BendingAngle / Length) makes find_m44's one-turn matrix non-finite at
    just 1.1x nominal current already; 2.0x is used here for a clear,
    reliable margin. This is the `find_m44 one-turn matrix has non-finite
    entries` guard (solve.py guard condition 1), confirmed distinct from the
    `|trace| >= 2.0` guard (guard condition 2) that TestInstabilityRollback's
    QF write above trips.
    """

    def test_nan_write_on_kicked_orbit_raises_and_restores_polynomb_elementwise(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        before_orbit = dict(bridge.bpm_positions())

        dipole_idx = bridge._element_index("DIPOLE01")
        before_polynom_b = list(bridge._ring[dipole_idx].PolynomB)

        dipole_nominal = _nominal_current("SR:MAG:DIPOLE:01:CURRENT:SP")
        with pytest.raises(OrbitSolveError, match="non-finite"):
            bridge.on_setpoint("SR:MAG:DIPOLE:01:CURRENT:SP", dipole_nominal * 2.0)

        after_polynom_b = list(bridge._ring[dipole_idx].PolynomB)
        assert len(after_polynom_b) == len(before_polynom_b)
        for before_term, after_term in zip(before_polynom_b, after_polynom_b, strict=True):
            assert after_term == before_term

        assert dict(bridge.bpm_positions()) == before_orbit

    def test_bridge_remains_usable_after_a_nan_producing_write(self, bridge):
        dipole_nominal = _nominal_current("SR:MAG:DIPOLE:01:CURRENT:SP")
        with pytest.raises(OrbitSolveError, match="non-finite"):
            bridge.on_setpoint("SR:MAG:DIPOLE:01:CURRENT:SP", dipole_nominal * 2.0)

        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(
            expected, abs=1e-12
        )


class TestErrorHandling:
    def test_unknown_device_raises(self, bridge):
        with pytest.raises(UnknownDeviceError):
            bridge.on_setpoint("SR:MAG:HCM:99:CURRENT:SP", 1.0)

    def test_non_mag_system_raises(self, bridge):
        with pytest.raises(UnknownDeviceError):
            bridge.on_setpoint("SR:DIAG:BPM:01:POSITION:SP", 1.0)

    def test_malformed_address_raises(self, bridge):
        with pytest.raises(UnknownDeviceError):
            bridge.on_setpoint("not-a-manifest-address", 1.0)


class TestMagnetCalibration:
    """FR3/FR4: a seeded corrector/quad calibration error (errors.magnet_cal)
    perturbs the field the setpoint produces, without touching unseeded
    devices."""

    def test_corrector_gain_error_scales_response(self):
        # Not a "response scales by exactly `factor`" assertion: on the real
        # (nonlinear) ring, a 10A HCM kick's closed-orbit response measurably
        # picks up sextupole feed-down (see response.py's docstring), so
        # doubling the commanded current does not double the BPM reading
        # exactly (measured ratio: 1.9967, not 2.0, at 10A). The correct,
        # exact oracle is the *effective* post-calibration current fed
        # through the same code path magnet_cal + StrengthMap.apply use --
        # i.e. orbit_response at 10.0 * 2.0 = 20.0A -- which matches bit for
        # bit (measured diff == 0.0).
        miscal = PhysicsBridge(corrector_gains={"HCM01": {"factor": 2.0}})
        miscal.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        miscal_x = miscal.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        expected = orbit_response("HCM01", 10.0 * 2.0)["BPM01"][0]
        assert miscal_x == pytest.approx(expected, abs=1e-12)

    def test_corrector_polarity_flip_inverts_response(self):
        # Not `-orbit_response(HCM01, 10.0)`: on the real (nonlinear) ring
        # the +I/-I response is not exactly antisymmetric (response.py's
        # docstring calls this out -- sextupole feed-down breaks the
        # antisymmetry at larger kicks; measured here: -8.9987e-05 vs the
        # naively-negated +10A response of -8.9695e-05, a ~0.3% difference).
        # The exact oracle is the *effective* post-calibration current
        # (magnet_cal(10.0, factor=-1.0) == -10.0) fed through the same code
        # path -- orbit_response(HCM01, -10.0) -- which matches bit for bit.
        flipped = PhysicsBridge(corrector_gains={"HCM01": {"factor": -1.0}})
        flipped.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        actual = flipped.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        expected = orbit_response("HCM01", -10.0)["BPM01"][0]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_corrector_gain_offset_biases_the_commanded_current(self):
        offset_bridge = PhysicsBridge(corrector_gains={"HCM01": {"offset": 5.0}})
        offset_bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 0.0)
        actual = offset_bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        expected = orbit_response("HCM01", 5.0)["BPM01"][0]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_uncalibrated_device_is_unaffected_by_another_devices_cal(self):
        bridge = PhysicsBridge(corrector_gains={"HCM01": {"factor": 3.0}})
        bridge.on_setpoint("SR:MAG:VCM:05:CURRENT:SP", 10.0)

        actual = bridge.bpm_positions()["SR:DIAG:BPM:05:POSITION:Y"]
        expected = orbit_response("VCM05", 10.0)["BPM05"][1]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_default_corrector_cal_state_is_identity(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        actual = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert actual == pytest.approx(expected, abs=1e-12)


class TestElementMisalignment:
    """FR3/FR4/FR12: a seeded element misalignment (errors.apply_misalignment)
    distorts the closed orbit, and an unstable seed fails boot diagnosably."""

    def test_seeded_element_misalignment_induces_nonzero_orbit_shift(self):
        misaligned = PhysicsBridge(element_misalignments={"QF01": {"dx": 300e-6}})
        actual = misaligned.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert actual != pytest.approx(0.0, abs=1e-9)

    def test_aligned_lattice_has_identically_zero_induced_shift(self):
        # An explicit all-zero misalignment must be a no-op, same as omitting
        # the element entirely.
        bridge = PhysicsBridge(element_misalignments={"QF01": {"dx": 0.0, "dy": 0.0, "roll": 0.0}})
        for address, value in bridge.bpm_positions().items():
            assert value == pytest.approx(0.0, abs=1e-9), address

    def test_unknown_misaligned_element_raises(self):
        with pytest.raises(UnknownDeviceError):
            PhysicsBridge(element_misalignments={"QF99": {"dx": 1e-4}})

    def test_destabilizing_misalignment_raises_systemexit_naming_elements(self):
        # Pure dx/dy preserves the one-turn trace (FR3 note); a large-enough
        # roll across the QF/QD families pushes the trace past the |2|
        # stability boundary, so this is a real, not contrived, boot fault.
        #
        # Spec-derived device counts (24 QF + 24 QD on the real ALS-U AR
        # ring, from facility_spec.ALS_U_AR), not the old toy ring's
        # `range(1, 17)` (16 + 16) -- measured: roll=0.6 across all 24+24
        # devices does still destabilize the real ring's one-turn map.
        qf_count = ALS_U_AR.family("QF").count
        qd_count = ALS_U_AR.family("QD").count
        roll_fault = {f"QF{i:02d}": {"roll": 0.6} for i in range(1, qf_count + 1)}
        roll_fault.update({f"QD{i:02d}": {"roll": 0.6} for i in range(1, qd_count + 1)})

        with pytest.raises(SystemExit, match="QF01"):
            PhysicsBridge(element_misalignments=roll_fault)


class TestBpmErrorSignatures:
    """FR3/FR4/FR12: a seeded BPM error (errors.bpm_read) perturbs only the
    IOC-facing reading (_push_bpm_readbacks), never bpm_positions() (the
    physics truth used by the model oracle / ORM cross-check)."""

    def test_bpm_offset_shifts_the_reading_but_not_the_physics_truth(self):
        rec = FakeRecord()
        bridge = PhysicsBridge(bpm_errors={"BPM01": {"offset_x": 50e-6}})
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 5.0)

        true_position = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert rec.value == pytest.approx(true_position - 50e-6, abs=1e-12)

    def test_bpm_offset_leaves_the_response_slope_unchanged(self):
        # A constant additive offset shifts every reading by the same amount,
        # so the *change* in reading between two setpoints (the ORM slope)
        # must be identical with and without the offset.
        clean_rec, offset_rec = FakeRecord(), FakeRecord()
        clean = PhysicsBridge()
        offset = PhysicsBridge(bpm_errors={"BPM01": {"offset_x": 50e-6}})
        clean.bind({"SR:DIAG:BPM:01:POSITION:X": clean_rec})
        offset.bind({"SR:DIAG:BPM:01:POSITION:X": offset_rec})

        clean.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 3.0)
        clean_at_3 = clean_rec.value
        offset.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 3.0)
        offset_at_3 = offset_rec.value

        clean.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 8.0)
        clean_at_8 = clean_rec.value
        offset.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 8.0)
        offset_at_8 = offset_rec.value

        assert (offset_at_8 - offset_at_3) == pytest.approx(clean_at_8 - clean_at_3, abs=1e-12)

    def test_bpm_polarity_flip_anti_correlates_with_the_unflipped_reading(self):
        rec = FakeRecord()
        bridge = PhysicsBridge(bpm_errors={"BPM01": {"polarity_x": -1.0}})
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        expected = -orbit_response("HCM01", 10.0)["BPM01"][0]
        assert rec.value == pytest.approx(expected, abs=1e-12)

    def test_bpm_gain_error_scales_the_reading(self):
        # Unlike the corrector-gain test above, this gain is applied to the
        # already-solved *reading* (bpm_read's trailing `reading_x *=
        # gain_x`), not fed back through the nonlinear orbit solve -- so an
        # exact 2x scaling is the correct expectation here, no oracle
        # re-derivation needed.
        rec = FakeRecord()
        bridge = PhysicsBridge(bpm_errors={"BPM01": {"gain_x": 2.0}})
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        expected = 2.0 * orbit_response("HCM01", 10.0)["BPM01"][0]
        assert rec.value == pytest.approx(expected, abs=1e-12)

    def test_bpm_error_at_one_device_does_not_affect_another(self):
        rec = FakeRecord()
        bridge = PhysicsBridge(bpm_errors={"BPM01": {"gain_x": 5.0}})
        bridge.bind({"SR:DIAG:BPM:05:POSITION:Y": rec})
        bridge.on_setpoint("SR:MAG:VCM:05:CURRENT:SP", 10.0)

        expected = orbit_response("VCM05", 10.0)["BPM05"][1]
        assert rec.value == pytest.approx(expected, abs=1e-12)

    def test_default_bpm_error_state_is_identity(self, bridge):
        rec = FakeRecord()
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        assert rec.value == pytest.approx(0.0, abs=1e-9)


class TestSeededNoise:
    def test_same_seed_gives_reproducible_bpm_noise(self):
        rec_a, rec_b = FakeRecord(), FakeRecord()
        a = PhysicsBridge(bpm_errors={"BPM01": {"noise_x": 1e-6}}, rng_seed=42)
        b = PhysicsBridge(bpm_errors={"BPM01": {"noise_x": 1e-6}}, rng_seed=42)

        a.bind({"SR:DIAG:BPM:01:POSITION:X": rec_a})
        b.bind({"SR:DIAG:BPM:01:POSITION:X": rec_b})

        assert rec_a.value == rec_b.value


class TestSextupoleStrengthWhiteBox:
    """Sextupole current->strength white-box check: PolynomB[2] = h_baked *
    I / I_nom, exactly, for every one of the 4 sextupole families
    (SF/SD/SHF/SHD -- see strengths.py's module docstring). Paired with an
    "echo" into the observable BPM readback path: on this ideal
    (unmisaligned) ring, a pure sextupole strength change alone leaves the
    zero orbit at a fixed point (sextupole feed-down is quadratic in orbit
    position, so 0 stays 0) -- the physically meaningful, non-tautological
    echo is that it measurably perturbs an already-kicked corrector's
    downstream response, exactly like the QF/DIPOLE checks above.
    """

    @pytest.mark.parametrize("family", ["SF", "SD", "SHF", "SHD"])
    def test_polynomb_index2_matches_baked_times_fraction(self, bridge, family):
        fam_name = f"{family}01"
        idx = bridge._element_index(fam_name)
        i_nom = _nominal_current(f"SR:MAG:{family}:01:CURRENT:SP")
        baked = bridge._strength_map.baked(fam_name)

        current = i_nom * 1.3
        bridge.on_setpoint(f"SR:MAG:{family}:01:CURRENT:SP", current)

        expected = baked * current / i_nom
        # Measured: exact to within ~3.6e-15 (float rounding) across all four
        # families -- rel=1e-9 leaves ample margin over that noise floor.
        assert bridge._ring[idx].PolynomB[2] == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("family", ["SF", "SD", "SHF", "SHD"])
    def test_sextupole_write_away_from_nominal_echoes_into_kicked_bpm_response(
        self, bridge, family
    ):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        response_at_nominal = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        # Measured stable for all four families at 1.3x nominal (no
        # OrbitSolveError), each giving a small but nonzero shift in the
        # HCM01->BPM01 response via sextupole feed-down.
        i_nom = _nominal_current(f"SR:MAG:{family}:01:CURRENT:SP")
        bridge.on_setpoint(f"SR:MAG:{family}:01:CURRENT:SP", i_nom * 1.3)
        response_after_change = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        assert response_after_change != pytest.approx(response_at_nominal)


class TestBindWiring:
    def test_bind_pushes_initial_bpm_state_into_records(self, bridge):
        x_rec, y_rec = FakeRecord(), FakeRecord()
        bridge.bind(
            {
                "SR:DIAG:BPM:01:POSITION:X": x_rec,
                "SR:DIAG:BPM:01:POSITION:Y": y_rec,
                "SR:MAG:HCM:01:CURRENT:SP": FakeRecord(),  # non-BPM entry, must be ignored
            }
        )
        assert x_rec.value == pytest.approx(0.0)
        assert y_rec.value == pytest.approx(0.0)

    def test_bind_then_setpoint_pushes_updated_bpm_readings(self, bridge):
        x_rec = FakeRecord()
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": x_rec})

        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert x_rec.value == pytest.approx(expected, abs=1e-12)

    def test_unbound_bpm_records_do_not_prevent_setpoint_writes(self, bridge):
        # No bind() call at all -- on_setpoint must still work (bpm_positions()
        # is the physics-only view, independent of any IOC wiring).
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] != 0.0
