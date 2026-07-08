"""Tests for the SR magnet setpoint -> PyAT orbit-recompute physics bridge.

Unlike test_record_factory.py, this module never imports ioc.records (hence
never touches softioc.builder): PhysicsBridge itself has no softioc
dependency (see physics_bridge.py -- it only imports `at` and `lattice`), so
these tests exercise it directly. The one test that verifies the `bind()`
wiring contract uses a minimal duck-typed fake record (just a `.set()`
method) rather than a real softioc record, sidestepping the process-global
softioc/CA gotchas documented in test_record_factory.py entirely -- the real
end-to-end wiring through live softioc records is orbit-response-e2e's job.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator.ioc.physics_bridge import (
    NOMINAL_DIPOLE_CURRENT_A,
    NOMINAL_QD_CURRENT_A,
    NOMINAL_QF_CURRENT_A,
    OrbitSolveError,
    PhysicsBridge,
    UnknownDeviceError,
)
from osprey.services.virtual_accelerator.lattice import orbit_response


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@pytest.fixture
def bridge() -> PhysicsBridge:
    return PhysicsBridge()


class TestNominalState:
    def test_nominal_orbit_is_zero(self, bridge):
        for address, value in bridge.bpm_positions().items():
            assert value == pytest.approx(0.0, abs=1e-9), address

    def test_writing_nominal_qf_current_reproduces_zero_orbit(self, bridge):
        bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", NOMINAL_QF_CURRENT_A)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(0.0, abs=1e-9)

    def test_writing_nominal_qd_current_reproduces_zero_orbit(self, bridge):
        bridge.on_setpoint("SR:MAG:QD:01:CURRENT:SP", NOMINAL_QD_CURRENT_A)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(0.0, abs=1e-9)

    def test_writing_nominal_dipole_current_reproduces_zero_orbit(self, bridge):
        bridge.on_setpoint("SR:MAG:DIPOLE:01:CURRENT:SP", NOMINAL_DIPOLE_CURRENT_A)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(0.0, abs=1e-9)


class TestSetpointWriteMovesBpm:
    """FR3/SC3: an SP write synchronously updates the BPM RB, in the direction
    the (independently implemented, task 3.2) lattice module predicts."""

    def test_hcm_write_changes_bpm_on_return_matching_lattice_prediction(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        actual = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        assert actual != 0.0
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
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
        bridge.on_setpoint("SR:MAG:VCM:01:CURRENT:SP", 10.0)
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(0.0)

    def test_qf_write_away_from_nominal_changes_orbit_response(self, bridge):
        # QF/QD have no independent readback-only "orbit_response" oracle
        # (lattice.orbit_response only covers correctors) -- the meaningful,
        # non-tautological check is that a kicked corrector's downstream
        # response measurably changes when the optics (QF gradient) change,
        # since the closed-orbit response depends on the whole ring's optics.
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        response_at_nominal_qf = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", NOMINAL_QF_CURRENT_A * 1.5)
        response_after_qf_change = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        assert response_after_qf_change != pytest.approx(response_at_nominal_qf)

    def test_dipole_write_away_from_nominal_changes_orbit_response(self, bridge):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        response_at_nominal_dipole = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        bridge.on_setpoint("SR:MAG:DIPOLE:01:CURRENT:SP", NOMINAL_DIPOLE_CURRENT_A * 1.2)
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
        before = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        with pytest.raises(OrbitSolveError):
            bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", NOMINAL_QF_CURRENT_A * 20.0)

        after = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert after == before

    def test_bridge_remains_usable_after_a_rejected_write(self, bridge):
        with pytest.raises(OrbitSolveError):
            bridge.on_setpoint("SR:MAG:QF:01:CURRENT:SP", NOMINAL_QF_CURRENT_A * 20.0)

        # A subsequent, valid write must still work normally.
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
