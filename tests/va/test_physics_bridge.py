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
comment) rather than taken from a toy-ring model, since the real ring's
stability/NaN boundaries sit much closer to nominal than a toy ring's.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest
from lume.model import LUMEModel
from lume.variables import Variable
from lume.variables.ndvariable import NDVariable

from osprey.services.virtual_accelerator.ioc.physics_bridge import (
    OrbitSolveError,
    PhysicsBridge,
    UnknownDeviceError,
)
from osprey.services.virtual_accelerator.lattice import build_ring, orbit_response
from osprey.services.virtual_accelerator.lattice.errors import bpm_read
from osprey.services.virtual_accelerator.lattice.strengths import StrengthMap
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels
from osprey.services.virtual_accelerator.model.fault_bounds import BPM_ERROR_FIELD_BOUNDS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
from osprey.simulation.facility_spec import ALS_U_AR


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc record: `.set()` and `.get()`.

    `value` is what the record currently holds -- the value it was
    constructed with (a setpoint record's commanded current) or the last
    value pushed into it. `writes` keeps every pushed value, so a test can
    prove a record the bridge only reads was never written at all.
    """

    def __init__(self, value: float | None = None) -> None:
        self.value: float | None = value
        self.writes: list[float] = []

    def set(self, value: float) -> None:
        self.value = value
        self.writes.append(value)

    def get(self) -> float | None:
        return self.value


# Read-only outputs a richer backend carries beside the BPM positions: one
# name outside the six-level manifest grammar, and one six-level name that
# ends in `:X` but belongs to no BPM. Neither may be taken for a BPM reading.
_OPTICS_SHAPE = (8,)
_EXTRA_READ_ONLY_OUTPUTS = ("beta_x", "SR:OPTICS:TWISS:01:BETA:X")


class ModelWithExtraOutputs(LUMEModel):
    """A `PyATRingModel` whose catalog also carries read-only ND optics outputs.

    Everything the ring owns delegates to the wrapped model's public
    `get()`/`set()`; the extra outputs answer zeros of their declared shape.
    `requested` records every name `_get` was asked for, so a test can prove
    the bridge never reads a variable it does not serve; `calls` keeps the
    names of each `_get` call apart, so a test can count the reads; `sets`
    keeps each `_set` batch, so a test can count the writes -- and therefore
    the closed-orbit solves, one per batch.
    """

    def __init__(self, inner: PyATRingModel) -> None:
        self._inner = inner
        self._extra = {
            name: NDVariable(name=name, shape=_OPTICS_SHAPE, read_only=True)
            for name in _EXTRA_READ_ONLY_OUTPUTS
        }
        self._variables: dict[str, Variable] = {**inner.supported_variables, **self._extra}
        self.requested: list[str] = []
        self.calls: list[tuple[str, ...]] = []
        self.sets: list[dict[str, Any]] = []

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.requested.extend(names)
        self.calls.append(tuple(names))
        ring_names = [name for name in names if name not in self._extra]
        values = dict(self._inner.get(ring_names)) if ring_names else {}
        values.update({name: np.zeros(_OPTICS_SHAPE) for name in names if name in self._extra})
        return values

    def _set(self, values: dict[str, Any]) -> None:
        self.sets.append(dict(values))
        self._inner.set(values)

    def reset(self) -> None:
        self._inner.reset()


class ModelWithoutFaultVariables(LUMEModel):
    """A `PyATRingModel` whose catalog hides every fault variable.

    Stands in for a backend with no fault model of its own (a surrogate, a
    Bmad model): setpoints and BPM positions delegate to the wrapped ring,
    and the dot-named fault variables are simply not declared. `requested`
    records every name `_get` was asked for.
    """

    def __init__(self, inner: PyATRingModel) -> None:
        self._inner = inner
        self._variables: dict[str, Variable] = {
            name: variable
            for name, variable in inner.supported_variables.items()
            if "." not in name
        }
        self.requested: list[str] = []

    @property
    def supported_variables(self) -> dict[str, Variable]:
        return self._variables

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.requested.extend(names)
        return dict(self._inner.get(names))

    def _set(self, values: dict[str, Any]) -> None:
        self._inner.set(values)

    def reset(self) -> None:
        self._inner.reset()


def _seeded_bridge(*, rng_seed: int | None = None, **seeds: Any) -> PhysicsBridge:
    """A bridge over a ring whose model carries the fault `seeds`.

    Fault state is model state: `bpm_errors`/`corrector_gains` seed
    `PyATRingModel`, and the bridge reads them back from the model.
    """
    return PhysicsBridge(model=PyATRingModel(**seeds), rng_seed=rng_seed)


# `bpm_read`'s fault keywords at identity. The reference reading below merges
# a seed over these per device, so a field the seed does not name reads as if
# the BPM were perfect.
_REFERENCE_IDENTITY: dict[str, float] = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "cal_x": 0.0,
    "cal_y": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
}


def _reference_readings(
    positions: dict[str, float],
    bpm_errors: dict[str, dict[str, float]],
    rng: np.random.Generator,
) -> dict[str, float]:
    """The readings `bpm_errors`, held as a plain dict, give for `positions`.

    One `bpm_read` per device, in sorted device order, with the seed merged
    over identity -- the draw order a seeded run must keep to reproduce. The
    bridge must serve exactly these values when the same seed lives in the
    model instead.
    """
    readings: dict[str, float] = {}
    for device in sorted({address.split(":")[3] for address in positions}):
        x_address = f"SR:DIAG:BPM:{device}:POSITION:X"
        y_address = f"SR:DIAG:BPM:{device}:POSITION:Y"
        state = {**_REFERENCE_IDENTITY, **bpm_errors.get(f"BPM{device}", {})}
        readings[x_address], readings[y_address] = bpm_read(
            positions[x_address], positions[y_address], rng=rng, **state
        )
    return readings


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
def model() -> PyATRingModel:
    """The backend the `bridge` fixture serves.

    Built here rather than left to `PhysicsBridge()`'s own default so the
    ring-level tests can reach the lattice through the model's public
    `lattice`/`element_index()` surface. Construction is identical to what
    the bridge would have done for itself.
    """
    return PyATRingModel()


@pytest.fixture
def bridge(model) -> PhysicsBridge:
    return PhysicsBridge(model=model)


@pytest.fixture(scope="module")
def strength_map() -> StrengthMap:
    """The current->strength calibration the model bakes at construction.

    Baked here from its own fresh `build_ring()`: the map's nominals come
    from the ring as built, before any write mutates it, so a separately
    baked map carries the same values as the model's.
    """
    return StrengthMap(build_ring())


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
        # (real ring: 1.5x nominal -- a typical toy-ring multiplier --
        # is already well past the instability boundary, so it is not used).
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
        # ~13 mm) is used here instead of a toy-ring 1.2x multiplier.
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
    def test_unstable_write_is_rejected_and_state_is_rolled_back(self, bridge, model):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        before_orbit = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        qf_idx = model.element_index("QF01")
        before_k = model.lattice[qf_idx].K

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

        after_k = model.lattice[qf_idx].K
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

    def test_nan_write_on_kicked_orbit_raises_and_restores_polynomb_elementwise(
        self, bridge, model
    ):
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        before_orbit = dict(bridge.bpm_positions())

        dipole_idx = model.element_index("DIPOLE01")
        before_polynom_b = list(model.lattice[dipole_idx].PolynomB)

        dipole_nominal = _nominal_current("SR:MAG:DIPOLE:01:CURRENT:SP")
        with pytest.raises(OrbitSolveError, match="non-finite"):
            bridge.on_setpoint("SR:MAG:DIPOLE:01:CURRENT:SP", dipole_nominal * 2.0)

        after_polynom_b = list(model.lattice[dipole_idx].PolynomB)
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
        miscal = _seeded_bridge(corrector_gains={"HCM01": {"factor": 2.0}})
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
        flipped = _seeded_bridge(corrector_gains={"HCM01": {"factor": -1.0}})
        flipped.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)
        actual = flipped.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        expected = orbit_response("HCM01", -10.0)["BPM01"][0]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_corrector_gain_offset_biases_the_commanded_current(self):
        offset_bridge = _seeded_bridge(corrector_gains={"HCM01": {"offset": 5.0}})
        offset_bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 0.0)
        actual = offset_bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]

        expected = orbit_response("HCM01", 5.0)["BPM01"][0]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_uncalibrated_device_is_unaffected_by_another_devices_cal(self):
        bridge = _seeded_bridge(corrector_gains={"HCM01": {"factor": 3.0}})
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
        # ring, from facility_spec.ALS_U_AR), not a toy ring's
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
        bridge = _seeded_bridge(bpm_errors={"BPM01": {"offset_x": 50e-6}})
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
        offset = _seeded_bridge(bpm_errors={"BPM01": {"offset_x": 50e-6}})
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
        bridge = _seeded_bridge(bpm_errors={"BPM01": {"polarity_x": -1.0}})
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
        bridge = _seeded_bridge(bpm_errors={"BPM01": {"gain_x": 2.0}})
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        expected = 2.0 * orbit_response("HCM01", 10.0)["BPM01"][0]
        assert rec.value == pytest.approx(expected, abs=1e-12)

    def test_bpm_error_at_one_device_does_not_affect_another(self):
        rec = FakeRecord()
        bridge = _seeded_bridge(bpm_errors={"BPM01": {"gain_x": 5.0}})
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
        a = _seeded_bridge(bpm_errors={"BPM01": {"noise_x": 1e-6}}, rng_seed=42)
        b = _seeded_bridge(bpm_errors={"BPM01": {"noise_x": 1e-6}}, rng_seed=42)

        a.bind({"SR:DIAG:BPM:01:POSITION:X": rec_a})
        b.bind({"SR:DIAG:BPM:01:POSITION:X": rec_b})

        assert rec_a.value == rec_b.value

    # Every fault kind, noise on both axes, and BPMs at both ends of the ring
    # and in the middle, so a reordered draw or a dropped field shows up.
    SEQUENCE_SEEDS = {
        "BPM01": {"offset_x": 50e-6, "noise_x": 1e-6},
        "BPM05": {"gain_y": 1.5, "polarity_y": -1.0, "noise_y": 2e-6},
        "BPM40": {"roll": 0.01, "offset_y": -20e-6, "noise_x": 3e-6, "noise_y": 3e-6},
        "BPM72": {"gain_x": 0.8, "polarity_x": -1.0},
    }

    def test_model_seeds_reproduce_the_dict_seeded_reading_sequence(self):
        # Exact equality, not approx: the model hands back the seeded floats
        # unchanged and the bridge makes the same `bpm_read` calls in the
        # same order off the same generator, so every served value --
        # noise draws included -- must match bit for bit, push after push.
        bridge = _seeded_bridge(bpm_errors=self.SEQUENCE_SEEDS, rng_seed=7)
        records = {address: FakeRecord() for address in bridge.bpm_positions()}
        reference_rng = np.random.default_rng(7)

        def served() -> dict[str, float | None]:
            return {address: rec.value for address, rec in records.items()}

        bridge.bind(records)
        expected = _reference_readings(bridge.bpm_positions(), self.SEQUENCE_SEEDS, reference_rng)
        assert served() == expected

        for address, current in (
            ("SR:MAG:HCM:01:CURRENT:SP", 10.0),
            ("SR:MAG:VCM:05:CURRENT:SP", -4.0),
        ):
            bridge.on_setpoint(address, current)
            expected = _reference_readings(
                bridge.bpm_positions(), self.SEQUENCE_SEEDS, reference_rng
            )
            assert served() == expected, address


class TestFaultsReadFromTheModel:
    """The bridge holds no fault state of its own: every push reads the
    model's BPM fault variables and every write reads the magnet's
    calibration, so a fault written to the model applies at once."""

    BPM_FAULT_NAMES = frozenset(
        f"BPM{index:02d}.{field}"
        for index in range(1, ALS_U_AR.family("BPM").count + 1)
        for field in BPM_ERROR_FIELD_BOUNDS
    )

    def test_a_bpm_fault_written_to_the_model_applies_on_the_next_push(self, bridge, model):
        rec = FakeRecord()
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})

        model.set({"BPM01.gain_x": 2.0})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        expected = 2.0 * orbit_response("HCM01", 10.0)["BPM01"][0]
        assert rec.value == pytest.approx(expected, abs=1e-12)

    def test_a_calibration_written_to_the_model_applies_to_the_next_setpoint(self, bridge, model):
        # Same exact oracle as TestMagnetCalibration: the effective current
        # magnet_cal delivers (10.0 * 2.0), fed through the same code path.
        model.set({"HCM01.cal_factor": 2.0})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        expected = orbit_response("HCM01", 20.0)["BPM01"][0]
        actual = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert actual == pytest.approx(expected, abs=1e-12)

    def test_a_fault_cleared_in_the_model_stops_perturbing_the_reading(self):
        model = PyATRingModel(bpm_errors={"BPM01": {"offset_x": 50e-6}})
        bridge = PhysicsBridge(model=model)
        rec = FakeRecord()
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})

        model.set({"BPM01.offset_x": 0.0})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 5.0)

        true_position = bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"]
        assert rec.value == pytest.approx(true_position, abs=1e-12)

    def test_each_push_reads_every_bpm_fault_in_one_get(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        model.calls.clear()

        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": FakeRecord()})

        fault_reads = [names for names in model.calls if set(names) & self.BPM_FAULT_NAMES]
        assert len(fault_reads) == 1
        assert len(fault_reads[0]) == len(self.BPM_FAULT_NAMES) == 9 * 72
        assert set(fault_reads[0]) == self.BPM_FAULT_NAMES

    def test_a_setpoint_reads_its_calibration_once_and_the_bpm_faults_once(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        model.calls.clear()

        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        cal_names = {"HCM01.cal_factor", "HCM01.cal_offset"}
        assert [set(names) for names in model.calls if set(names) & cal_names] == [cal_names]
        assert len([names for names in model.calls if set(names) & self.BPM_FAULT_NAMES]) == 1

    def test_a_backend_without_fault_variables_serves_the_true_orbit(self):
        # The model's catalog decides which faults exist: faults the backend
        # does not declare are not read, and read as identity -- here even
        # though the hidden ring underneath carries seeded ones.
        model = ModelWithoutFaultVariables(
            PyATRingModel(
                bpm_errors={"BPM01": {"offset_x": 50e-6}},
                corrector_gains={"HCM01": {"factor": 2.0}},
            )
        )
        bridge = PhysicsBridge(model=model)
        rec = FakeRecord()
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": rec})
        bridge.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        assert not [name for name in model.requested if "." in name]
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert rec.value == pytest.approx(expected, abs=1e-12)


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
    def test_polynomb_index2_matches_baked_times_fraction(
        self, bridge, model, strength_map, family
    ):
        fam_name = f"{family}01"
        idx = model.element_index(fam_name)
        i_nom = _nominal_current(f"SR:MAG:{family}:01:CURRENT:SP")
        baked = strength_map.baked(fam_name)

        current = i_nom * 1.3
        bridge.on_setpoint(f"SR:MAG:{family}:01:CURRENT:SP", current)

        expected = baked * current / i_nom
        # Measured: exact to within ~3.6e-15 (float rounding) across all four
        # families -- rel=1e-9 leaves ample margin over that noise floor.
        assert model.lattice[idx].PolynomB[2] == pytest.approx(expected, rel=1e-9)

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


class TestReadingSelection:
    """The bridge serves the model's read-only *scalar* variables as readings.

    Shape, never the address text, tells a reading from a model-only output:
    a backend whose catalog carries further read-only arrays (optics, a Bmad
    model's own diagnostics) binds exactly as the plain ring.
    """

    BPM_COUNT = 2 * ALS_U_AR.family("BPM").count

    def test_reading_selection_ignores_extra_read_only_nd_variables(self, bridge):
        extended = PhysicsBridge(model=ModelWithExtraOutputs(PyATRingModel()))

        served = sorted(extended.bpm_positions())
        assert len(served) == self.BPM_COUNT == 144
        assert served == sorted(bridge.bpm_positions())
        assert not set(_EXTRA_READ_ONLY_OUTPUTS) & set(served)

    def test_reading_selection_never_reads_the_extra_variables(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        extended = PhysicsBridge(model=model)
        x_rec = FakeRecord()
        extended.bind({"SR:DIAG:BPM:01:POSITION:X": x_rec})
        extended.on_setpoint("SR:MAG:HCM:01:CURRENT:SP", 10.0)

        assert model.requested, "the bridge read nothing from the model"
        assert not set(_EXTRA_READ_ONLY_OUTPUTS) & set(model.requested)
        expected = orbit_response("HCM01", 10.0)["BPM01"][0]
        assert x_rec.value == pytest.approx(expected, abs=1e-12)

    def test_reading_selection_keeps_ring_order(self, bridge):
        # Readout noise is drawn device by device in this order, so a seeded
        # run reproduces only if the served order is the sorted address order.
        extended = PhysicsBridge(model=ModelWithExtraOutputs(PyATRingModel()))
        assert list(extended.bpm_positions()) == sorted(bridge.bpm_positions())


class TestBindWiring:
    def test_bind_pushes_initial_bpm_state_into_records(self, bridge):
        x_rec, y_rec, sp_rec = FakeRecord(), FakeRecord(), FakeRecord()
        bridge.bind(
            {
                "SR:DIAG:BPM:01:POSITION:X": x_rec,
                "SR:DIAG:BPM:01:POSITION:Y": y_rec,
                "SR:MAG:HCM:01:CURRENT:SP": sp_rec,  # retained, never pushed into
            },
            physics_setpoints=frozenset({"SR:MAG:HCM:01:CURRENT:SP"}),
        )
        assert x_rec.value == pytest.approx(0.0)
        assert y_rec.value == pytest.approx(0.0)
        assert sp_rec.value is None

    def test_bind_retains_the_magnet_setpoint_records_by_address(self, bridge):
        x_rec, hcm_sp, qf_sp = FakeRecord(), FakeRecord(), FakeRecord()
        bridge.bind(
            {
                "SR:DIAG:BPM:01:POSITION:X": x_rec,
                "SR:MAG:HCM:01:CURRENT:SP": hcm_sp,
                "SR:MAG:QF:07:CURRENT:SP": qf_sp,
            },
            physics_setpoints=frozenset({"SR:MAG:HCM:01:CURRENT:SP", "SR:MAG:QF:07:CURRENT:SP"}),
        )

        assert bridge._setpoint_records == {
            "SR:MAG:HCM:01:CURRENT:SP": hcm_sp,
            "SR:MAG:QF:07:CURRENT:SP": qf_sp,
        }
        assert bridge._setpoint_records["SR:MAG:HCM:01:CURRENT:SP"] is hcm_sp
        # Retaining the setpoints does not displace the first BPM push.
        assert x_rec.value == pytest.approx(0.0, abs=1e-9)
        assert hcm_sp.value is None and qf_sp.value is None

    def test_bind_keeps_only_the_manifest_declared_setpoints(self, bridge):
        # Membership in the manifest's set decides, never the address text:
        # a readback, a BPM record whose subfield happens to read SP, and a
        # record outside the grammar are all ignored, and so is a declared
        # setpoint the caller handed no record for.
        records = {
            "SR:DIAG:BPM:01:POSITION:X": FakeRecord(),
            "SR:DIAG:BPM:01:POSITION:Y": FakeRecord(),
            "SR:MAG:HCM:01:CURRENT:RB": FakeRecord(),
            "SR:DIAG:BPM:01:POSITION:SP": FakeRecord(),
            "not-a-manifest-address": FakeRecord(),
            "SR:MAG:VCM:05:CURRENT:SP": FakeRecord(),
        }
        bridge.bind(
            records,
            physics_setpoints=frozenset({"SR:MAG:VCM:05:CURRENT:SP", "SR:MAG:HCM:09:CURRENT:SP"}),
        )
        assert set(bridge._setpoint_records) == {"SR:MAG:VCM:05:CURRENT:SP"}

    def test_bind_retains_no_setpoint_the_manifest_did_not_declare(self, bridge):
        # The address ends in `:SP`, and that is not what makes it a setpoint.
        bridge.bind({"SR:MAG:HCM:01:CURRENT:SP": FakeRecord()})
        assert bridge._setpoint_records == {}

    def test_bind_retains_a_declared_setpoint_spelled_outside_the_bundled_grammar(self, bridge):
        # A facility's own spelling is retained as declared. Only the
        # calibration re-apply, which maps a magnet by the bundled tree's
        # family and device tokens, has nothing to look it up by.
        rec = FakeRecord()
        bridge.bind(
            {"corrector-one-setpoint": rec}, physics_setpoints=frozenset({"corrector-one-setpoint"})
        )
        assert bridge._setpoint_records == {"corrector-one-setpoint": rec}
        assert bridge._setpoint_addresses == {}

    def test_setpoint_records_are_empty_until_bound(self, bridge):
        assert bridge._setpoint_records == {}

    def test_rebind_replaces_the_setpoint_records(self, bridge):
        hcm = "SR:MAG:HCM:01:CURRENT:SP"
        vcm = "SR:MAG:VCM:05:CURRENT:SP"
        bridge.bind({hcm: FakeRecord()}, physics_setpoints=frozenset({hcm}))
        second = FakeRecord()
        bridge.bind({vcm: second}, physics_setpoints=frozenset({vcm}))
        assert bridge._setpoint_records == {vcm: second}

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


class TestRefreshAfterModelWrite:
    """`refresh(changed)` re-serves the ring after a model-only write.

    A calibration written straight to the model (through the model surface,
    not through a served setpoint) changes what current an already-commanded
    magnet actually delivers, so the commanded currents the served `:SP`
    records still hold have to be pushed through the new calibration again.
    `refresh` does that for every magnet named in `changed`, in a single
    batch -- one closed-orbit solve however many magnets a family-wide
    change touches -- and then refreshes and pushes the BPM readings exactly
    once. Names it serves nothing for (BPM fault fields, model-only
    variables such as the stuck set) cost no write at all.
    """

    CORRECTORS = (
        "SR:MAG:HCM:01:CURRENT:SP",
        "SR:MAG:HCM:02:CURRENT:SP",
        "SR:MAG:HCM:03:CURRENT:SP",
    )
    CAL_FACTORS = ("HCM01.cal_factor", "HCM02.cal_factor", "HCM03.cal_factor")

    def test_a_family_wide_calibration_refresh_applies_in_exactly_one_set(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        records: dict[str, Any] = {address: FakeRecord(2.0) for address in self.CORRECTORS}
        records["SR:DIAG:BPM:01:POSITION:X"] = FakeRecord()
        bridge.bind(records, physics_setpoints=frozenset(self.CORRECTORS))
        model.set(dict.fromkeys(self.CAL_FACTORS, 1.5))
        model.sets.clear()

        bridge.refresh(list(self.CAL_FACTORS))

        # One `set` for the whole family, so one solve: the cost of a
        # family-wide calibration change must not scale with the number of
        # magnets it touches.
        assert len(model.sets) == 1
        # 2.0 A commanded through factor 1.5 is a single multiply, exact in
        # binary floating point; abs=1e-12 only guards the address mapping.
        assert model.sets[0] == {
            address: pytest.approx(3.0, abs=1e-12) for address in self.CORRECTORS
        }

    def test_a_refreshed_calibration_moves_the_orbit_to_the_new_current(self, bridge, model):
        sp_rec, x_rec = FakeRecord(10.0), FakeRecord()
        bridge.bind(
            {
                "SR:MAG:HCM:01:CURRENT:SP": sp_rec,
                "SR:DIAG:BPM:01:POSITION:X": x_rec,
            },
            physics_setpoints=frozenset({"SR:MAG:HCM:01:CURRENT:SP"}),
        )

        model.set({"HCM01.cal_factor": 2.0})
        bridge.refresh(["HCM01.cal_factor"])

        # Same oracle as a calibrated setpoint write: 10 A commanded through
        # factor 2.0 delivers the orbit of a 20 A corrector, whether the
        # calibration arrived before the setpoint or after it.
        expected = orbit_response("HCM01", 20.0)["BPM01"][0]
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(
            expected, abs=1e-12
        )
        assert x_rec.value == pytest.approx(expected, abs=1e-12)

    def test_refresh_reads_the_setpoint_records_but_never_writes_them(self, bridge, model):
        sp_rec = FakeRecord(10.0)
        bridge.bind(
            {"SR:MAG:HCM:01:CURRENT:SP": sp_rec},
            physics_setpoints=frozenset({"SR:MAG:HCM:01:CURRENT:SP"}),
        )

        model.set({"HCM01.cal_factor": 2.0})
        bridge.refresh(["HCM01.cal_factor"])

        # The served setpoint keeps the *commanded* current: the serving
        # write path owns that record, and a calibration change is not an
        # operator moving the magnet. Only the model holds the 20 A the
        # magnet now physically delivers.
        assert sp_rec.writes == []
        assert sp_rec.value == 10.0

    def test_a_bpm_only_refresh_performs_no_set_and_still_pushes(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        sp_rec, x_rec = FakeRecord(10.0), FakeRecord()
        bridge.bind(
            {
                "SR:MAG:HCM:01:CURRENT:SP": sp_rec,
                "SR:DIAG:BPM:01:POSITION:X": x_rec,
            },
            physics_setpoints=frozenset({"SR:MAG:HCM:01:CURRENT:SP"}),
        )

        model.set({"BPM01.offset_x": 50e-6})
        model.sets.clear()

        bridge.refresh(["BPM01.offset_x", "stuck_setpoints"])

        # A readout error changes no magnet, so re-solving the ring for it
        # would be pure cost; the new offset still has to reach the record.
        assert model.sets == []
        assert sp_rec.writes == []
        assert x_rec.value == pytest.approx(-50e-6, abs=1e-12)

    def test_an_empty_refresh_still_pushes_the_current_readings(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        x_rec = FakeRecord()
        bridge.bind({"SR:DIAG:BPM:01:POSITION:X": x_rec})

        model.set({"BPM01.offset_x": 50e-6, "BPM01.gain_x": 2.0})
        model.sets.clear()

        bridge.refresh([])

        # `bpm_read` applies gain last, after the offset: (0 - 50e-6) * 2.
        assert model.sets == []
        assert x_rec.value == pytest.approx(-100e-6, abs=1e-12)

    def test_refresh_rewrites_only_the_magnets_whose_calibration_changed(self):
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        bridge.bind(
            {address: FakeRecord(2.0) for address in self.CORRECTORS},
            physics_setpoints=frozenset(self.CORRECTORS),
        )
        model.set({"HCM02.cal_offset": 1.0})
        model.sets.clear()

        bridge.refresh(["HCM02.cal_offset"])

        assert len(model.sets) == 1
        assert set(model.sets[0]) == {"SR:MAG:HCM:02:CURRENT:SP"}
        assert model.sets[0]["SR:MAG:HCM:02:CURRENT:SP"] == pytest.approx(3.0, abs=1e-12)

    def test_refresh_writes_nothing_for_a_magnet_with_no_bound_setpoint(self):
        # No `bind()` at all: there is no commanded current to re-apply, so
        # the calibration simply waits for the next setpoint write.
        model = ModelWithExtraOutputs(PyATRingModel())
        bridge = PhysicsBridge(model=model)
        model.set({"HCM01.cal_factor": 2.0})
        model.sets.clear()

        bridge.refresh(["HCM01.cal_factor"])

        assert model.sets == []
        assert bridge.bpm_positions()["SR:DIAG:BPM:01:POSITION:X"] == pytest.approx(0.0, abs=1e-9)

    def test_refresh_pushes_the_seeded_readings_exactly_once(self):
        # Each `bpm_read` draws both noise axes whether or not the BPM is
        # noisy, so a refresh that pushed twice would run the generator past
        # the reference stream and every served value would diverge.
        seeds = {"BPM01": {"noise_x": 1e-6}, "BPM40": {"noise_y": 3e-6}}
        bridge = _seeded_bridge(bpm_errors=seeds, rng_seed=7)
        records = {address: FakeRecord() for address in bridge.bpm_positions()}
        reference_rng = np.random.default_rng(7)

        bridge.bind(records)
        expected = _reference_readings(bridge.bpm_positions(), seeds, reference_rng)
        assert {address: rec.value for address, rec in records.items()} == expected

        bridge.refresh(["BPM01.noise_x"])
        expected = _reference_readings(bridge.bpm_positions(), seeds, reference_rng)
        assert {address: rec.value for address, rec in records.items()} == expected
