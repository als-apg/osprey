"""Tests for the virtual accelerator's LUME model layer.

The headline counts below are pinned as literals on purpose: 348 magnet
setpoints plus 144 BPM readings is the contract the model exposes, and a
silent drift in either number is exactly what this suite exists to catch.
Everything else is derived from the same sources the catalog reads.

The ``PyATRingModel`` half of the suite targets the real ALS-U AR ring, so
every "away from nominal" current here is a fraction of the device's own
``machine.json`` nominal rather than an absolute number, and device counts
come from ``osprey.simulation.facility_spec.ALS_U_AR`` -- the databases fix
the model, not vice versa.
"""

from __future__ import annotations

import pytest
from lume.model import ReadOnlyError

from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    build_manifest,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels
from osprey.services.virtual_accelerator.model import (
    PyATRingModel,
    UnknownDeviceError,
    build_variable_catalog,
)
from osprey.services.virtual_accelerator.model.catalog import _load_limit_bands
from osprey.simulation.facility_spec import ALS_U_AR

# 420 pyat-coupled devices minus the 72 BPMs, each contributing one
# :CURRENT:SP; the 72 BPMs contribute one X and one Y reading each.
EXPECTED_INPUTS = 348
EXPECTED_OUTPUTS = 144

A_CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
A_BPM_READING = "SR:DIAG:BPM:01:POSITION:X"


@pytest.fixture(scope="module")
def catalog():
    return build_variable_catalog()


@pytest.fixture(scope="module")
def pyat_coupled_setpoints() -> list[str]:
    """The pyat-coupled ``:SP`` addresses, straight from the manifest."""
    return [
        channel["address"]
        for channel in build_manifest()["channels"]
        if channel["partition"] == PARTITION_PYAT_COUPLED and channel["subfield"] == "SP"
    ]


@pytest.fixture(scope="module")
def bpm_readings(catalog) -> list[str]:
    """Every BPM POSITION output address, derived from the catalog."""
    return [address for address, variable in catalog.items() if variable.read_only]


@pytest.fixture
def model() -> PyATRingModel:
    """A fresh, unfaulted model. Function-scoped: these tests mutate it."""
    return PyATRingModel()


def quadrupole_setpoints(family: str) -> list[str]:
    """Every ``:SP`` address of a quadrupole family, sized from the facility spec."""
    count = ALS_U_AR.family(family).count
    return [f"SR:MAG:{family}:{i:02d}:CURRENT:SP" for i in range(1, count + 1)]


def quad_strengths(model: PyATRingModel, addresses: list[str]) -> list[float]:
    """The live ``K`` of each address's ring element.

    Reaches into the model's lattice on purpose: rollback is a property of
    the *ring*, and a write that was rolled back leaves no trace in the
    value API by design -- so the ring is the only place the evidence lives.
    """
    strengths = []
    for address in addresses:
        _ring, _system, family, device, _field, _subfield = address.split(":")
        strengths.append(float(model._ring[model._index_by_famname[f"{family}{device}"]].K))
    return strengths


class TestCatalogComposition:
    def test_input_output_split(self, catalog):
        inputs = [v for v in catalog.values() if not v.read_only]
        outputs = [v for v in catalog.values() if v.read_only]
        assert len(inputs) == EXPECTED_INPUTS
        assert len(outputs) == EXPECTED_OUTPUTS
        assert len(catalog) == EXPECTED_INPUTS + EXPECTED_OUTPUTS

    def test_setpoint_echo_is_not_a_model_variable(self, catalog):
        """``:RB`` is the IOC's setpoint echo -- a serving-layer concern."""
        assert not [address for address in catalog if address.endswith(":RB")]

    def test_keys_and_names_are_full_addresses(self, catalog):
        """No address translation anywhere: the key is the name is the PV."""
        assert all(variable.name == address for address, variable in catalog.items())
        assert all(len(address.split(":")) == 6 for address in catalog)

    def test_covers_every_pyat_coupled_setpoint(self, catalog, pyat_coupled_setpoints):
        assert set(pyat_coupled_setpoints) <= set(catalog)


class TestInputVariables:
    def test_known_corrector_band_and_unit(self, catalog):
        variable = catalog["SR:MAG:HCM:01:CURRENT:SP"]
        assert variable.read_only is False
        assert variable.value_range == (-12.0, 12.0)
        assert variable.unit == "A"

    def test_every_setpoint_is_writable_and_banded(self, catalog, pyat_coupled_setpoints):
        for address in pyat_coupled_setpoints:
            variable = catalog[address]
            assert variable.read_only is False, address
            assert variable.value_range is not None, address
            assert variable.unit == "A", address

    def test_nominal_default_values(self, catalog, pyat_coupled_setpoints):
        """Defaults come from machine.json's ``value``, not from thin air."""
        machine_channels = load_machine_json_channels()
        for address in pyat_coupled_setpoints:
            assert catalog[address].default_value == machine_channels[address]["value"], address

    def test_set_time_validation_is_off(self, catalog, pyat_coupled_setpoints):
        """Declared ranges are metadata: lume neither rejects nor clamps on
        set(). Enforcement stays with DRVL/DRVH, channel_limits.json and the
        fail-closed orbit solve (see the catalog module docstring)."""
        for address in pyat_coupled_setpoints:
            assert catalog[address].default_validation_config == "none", address


class TestOutputVariables:
    def test_bpm_readings_are_read_only_metres(self, catalog):
        outputs = {address: variable for address, variable in catalog.items() if variable.read_only}
        assert len(outputs) == EXPECTED_OUTPUTS
        for address, variable in outputs.items():
            assert address.split(":")[-1] in ("X", "Y"), address
            assert variable.value_range is None, address
            assert variable.unit == "m", address


class TestNominalsLieInBand:
    """A ``channel_limits.json`` band that excludes its ``machine.json``
    nominal hard-fails catalog construction (ScalarVariable validates
    default_value against value_range unconditionally). This is the CI guard
    that turns a bad ``derive_bands.py`` regeneration into a named test
    failure instead of a VA boot failure."""

    def test_every_setpoint_nominal_is_within_its_band(self, pyat_coupled_setpoints):
        machine_channels = load_machine_json_channels()
        bands = _load_limit_bands()

        missing = [a for a in pyat_coupled_setpoints if a not in bands]
        assert not missing, f"pyat-coupled setpoints with no limit band: {missing}"

        violations = []
        for address in pyat_coupled_setpoints:
            nominal = machine_channels[address]["value"]
            low, high = bands[address]
            if not low <= nominal <= high:
                violations.append((address, nominal, low, high))
        assert not violations, f"nominals outside their channel_limits band: {violations}"


class TestRetainedInputs:
    """The model retains what was written to each setpoint, seeded from the
    machine.json nominals -- state the IOC write path never had."""

    def test_setpoint_reads_its_nominal_before_any_write(self, model, pyat_coupled_setpoints):
        machine_channels = load_machine_json_channels()
        assert model.get(A_CORRECTOR) == machine_channels[A_CORRECTOR]["value"]
        for address in pyat_coupled_setpoints:
            assert model.get(address) == machine_channels[address]["value"], address

    def test_setpoint_reads_back_the_written_value(self, model):
        model.set({A_CORRECTOR: 5.0})
        assert model.get(A_CORRECTOR) == 5.0

    def test_a_write_moves_the_bpm_readings(self, model):
        before = model.get(A_BPM_READING)
        model.set({A_CORRECTOR: 5.0})
        assert model.get(A_BPM_READING) != pytest.approx(before, abs=1e-12)


class TestMultiKeySetAtomicity:
    """``_set`` takes an arbitrary number of setpoints, solves once, and
    commits nothing at all if that solve trips."""

    def test_multi_key_write_retains_every_value(self, model):
        # A quadrupole's stability margin is a fraction of a percent around
        # its nominal, so perturb rather than pick an absolute current.
        values = {address: model.get(address) * 1.001 for address in quadrupole_setpoints("QF")[:3]}
        values[A_CORRECTOR] = 4.0
        model.set(values)
        for address, value in values.items():
            assert model.get(address) == value, address

    def test_destabilizing_combination_rolls_back_every_mutated_element(self, model):
        # 24 QF at 1.2x nominal pushes the one-turn trace past |2| -- measured
        # against the real ring, not carried over from a toy lattice.
        addresses = quadrupole_setpoints("QF")
        model.set({addresses[0]: model.get(addresses[0]) * 1.02})  # a benign prior write

        before_strengths = quad_strengths(model, addresses)
        before_inputs = model.get(addresses)
        before_outputs = model.get([A_BPM_READING])

        fault = {address: model.get(address) * 1.2 for address in addresses}
        with pytest.raises(OrbitSolveError):
            model.set(fault)

        assert quad_strengths(model, addresses) == before_strengths
        assert model.get(addresses) == before_inputs
        assert model.get([A_BPM_READING]) == before_outputs

    def test_a_rejected_write_leaves_the_ring_fit_to_serve(self, model, bpm_readings):
        """The strongest rollback evidence: after a rejected write, the same
        follow-up write reaches the same physics as on a ring that never saw
        the fault."""
        fault = {address: model.get(address) * 1.2 for address in quadrupole_setpoints("QF")}
        with pytest.raises(OrbitSolveError):
            model.set(fault)

        model.set({A_CORRECTOR: 5.0})
        pristine = PyATRingModel()
        pristine.set({A_CORRECTOR: 5.0})
        assert model.get(bpm_readings) == pristine.get(bpm_readings)


class TestReset:
    def test_reset_restores_every_nominal(self, model, pyat_coupled_setpoints):
        machine_channels = load_machine_json_channels()
        qf01 = "SR:MAG:QF:01:CURRENT:SP"
        model.set({A_CORRECTOR: 5.0, qf01: model.get(qf01) * 1.001})
        model.reset()
        for address in pyat_coupled_setpoints:
            assert model.get(address) == machine_channels[address]["value"], address

    def test_reset_returns_an_unfaulted_ring_to_the_nominal_orbit(self, model, bpm_readings):
        model.set({A_CORRECTOR: 5.0})
        model.reset()
        for address, value in model.get(bpm_readings).items():
            assert value == pytest.approx(0.0, abs=1e-12), address

    def test_reset_preserves_construction_time_misalignments(self, bpm_readings):
        """Reset undoes writes, not faults: it re-applies the nominals to the
        existing ring rather than rebuilding it, so a seeded misalignment (and
        the orbit distortion it causes) survives."""
        misaligned = PyATRingModel(element_misalignments={"QF01": {"dx": 300e-6}})
        distorted = misaligned.get(bpm_readings)

        misaligned.set({A_CORRECTOR: 5.0})
        misaligned.reset()

        assert misaligned.get(A_CORRECTOR) == 0.0  # back to the machine.json nominal
        after = misaligned.get(bpm_readings)
        assert any(abs(value) > 1e-9 for value in after.values())
        assert after == distorted


class TestSupportedVariables:
    def test_catalog_is_built_once_and_cached(self, model):
        """LUMEModel.get/set consult this on every call -- rebuilding it per
        access would put a ~70 ms catalog build on the write path."""
        assert model.supported_variables is model.supported_variables

    def test_catalog_matches_a_freshly_built_one(self, model, catalog):
        assert set(model.supported_variables) == set(catalog)


class TestMisalignmentSeeding:
    def test_unknown_element_raises_unknown_device_error(self):
        with pytest.raises(UnknownDeviceError, match="QF99"):
            PyATRingModel(element_misalignments={"QF99": {"dx": 1e-4}})

    def test_destabilizing_seed_raises_orbit_solve_error_naming_the_elements(self):
        # Pure dx/dy preserves the one-turn trace; a large-enough roll across
        # the QF/QD families pushes it past the |2| stability boundary, so
        # this is a real boot fault, not a contrived one.
        roll_fault = {
            f"QF{i:02d}": {"roll": 0.6} for i in range(1, ALS_U_AR.family("QF").count + 1)
        }
        roll_fault.update(
            {f"QD{i:02d}": {"roll": 0.6} for i in range(1, ALS_U_AR.family("QD").count + 1)}
        )

        with pytest.raises(OrbitSolveError, match="QF01") as excinfo:
            PyATRingModel(element_misalignments=roll_fault)
        # Upstream-bound: this class must never take the host process down.
        # Turning this into a SystemExit is the IOC's decision, not the model's.
        assert not isinstance(excinfo.value, SystemExit)


class TestUnknownVariables:
    def test_get_of_an_unknown_address_raises(self, model):
        with pytest.raises(UnknownDeviceError, match="QF:99"):
            model._get(["SR:MAG:QF:99:CURRENT:SP"])

    def test_set_of_an_unknown_address_raises(self, model):
        with pytest.raises(UnknownDeviceError, match="QF:99"):
            model._set({"SR:MAG:QF:99:CURRENT:SP": 1.0})

    def test_set_of_an_output_address_raises(self, model):
        """A BPM reading is a model variable but not a settable input."""
        with pytest.raises(UnknownDeviceError, match="settable input"):
            model._set({A_BPM_READING: 1.0})

    def test_lume_rejects_an_unknown_name_before_reaching_the_model(self, model):
        with pytest.raises(ValueError, match="not supported by the model"):
            model.get(["SR:MAG:QF:99:CURRENT:SP"])


class TestReadOnlyEnforcement:
    def test_writing_a_bpm_reading_raises_readonly(self, model):
        with pytest.raises(ReadOnlyError):
            model.set({A_BPM_READING: 1.0})

    def test_a_rejected_readonly_write_leaves_the_reading_untouched(self, model):
        before = model.get(A_BPM_READING)
        with pytest.raises(ReadOnlyError):
            model.set({A_BPM_READING: 1.0})
        assert model.get(A_BPM_READING) == before
