"""Tests for the virtual accelerator's LUME model layer.

The headline counts below are pinned as literals on purpose: 348 magnet
setpoints plus 144 BPM readings plus 1,344 fault writables plus three
read-only optics arrays is the contract the model exposes, and a silent
drift in any of those numbers is exactly what this suite exists to catch.
Everything else is derived from the same sources the catalog reads.

The ``PyATRingModel`` half of the suite targets the real ALS-U AR ring, so
every "away from nominal" current here is a fraction of the device's own
``machine.json`` nominal rather than an absolute number, and device counts
come from ``osprey.simulation.facility_spec.ALS_U_AR`` -- the databases fix
the model, not vice versa.
"""

from __future__ import annotations

import re

import at
import numpy as np
import pytest
from lume.model import ReadOnlyError
from lume_pyat.actions import PyATWritableScalarVariable
from pydantic import ValidationError

from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    build_manifest,
    setpoint_addresses,
)
from osprey.services.virtual_accelerator.manifest.loaders import load_machine_json_channels
from osprey.services.virtual_accelerator.model import (
    PyATRingModel,
    UnknownDeviceError,
    build_variable_catalog,
)
from osprey.services.virtual_accelerator.model import pyat as pyat_module
from osprey.services.virtual_accelerator.model.catalog import _load_limit_bands
from osprey.services.virtual_accelerator.model.fault_bounds import (
    BPM_ERROR_FIELD_BOUNDS,
    BPM_POLARITY_FIELDS,
    MAGNET_CAL_BOUNDS,
)
from osprey.services.virtual_accelerator.model.variables import (
    PyATReadOnlyNDVariable,
    PyATWritableEnumVariable,
)
from osprey.simulation.facility_spec import ALS_U_AR

# 420 pyat-coupled devices minus the 72 BPMs, each contributing one
# :CURRENT:SP; the 72 BPMs contribute one X and one Y reading each.
EXPECTED_INPUTS = 348
EXPECTED_OUTPUTS = 144
# 72 BPMs x 9 reading-error fields, plus 348 magnets x 2 calibration fields.
EXPECTED_FAULT_VARIABLES = 1344
# The two fractional tunes, and beta and the true orbit at the 72 BPMs.
OPTICS_SHAPES = {"tunes": (2,), "beta_at_bpms": (72, 2), "orbit_at_bpms": (72, 2)}
OPTICS_NAMES = sorted(OPTICS_SHAPES)

A_CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
A_QUADRUPOLE = "SR:MAG:QF:01:CURRENT:SP"
A_BPM_READING = "SR:DIAG:BPM:01:POSITION:X"

BPMS = [f"BPM{i:02d}" for i in range(1, ALS_U_AR.family("BPM").count + 1)]
MAGNETS = [
    f"{family.name}{i:02d}"
    for family in ALS_U_AR.families
    if family.kind != "monitor"
    for i in range(1, family.count + 1)
]
FAULT_NAMES = {f"{bpm}.{field}" for bpm in BPMS for field in BPM_ERROR_FIELD_BOUNDS} | {
    f"{magnet}.{field}" for magnet in MAGNETS for field in MAGNET_CAL_BOUNDS
}
# The value every fault field reads on a device nobody seeded: a BPM that
# reports the true orbit, a magnet that delivers the current commanded.
IDENTITY = {
    "offset_x": 0.0,
    "offset_y": 0.0,
    "gain_x": 1.0,
    "gain_y": 1.0,
    "polarity_x": 1.0,
    "polarity_y": 1.0,
    "roll": 0.0,
    "noise_x": 0.0,
    "noise_y": 0.0,
    "cal_factor": 1.0,
    "cal_offset": 0.0,
}
SEEDED_BPM_ERRORS = {"BPM07": {"offset_x": 5e-5, "gain_y": 1.05, "polarity_x": -1.0}}
SEEDED_CORRECTOR_GAINS = {"QF07": {"factor": 1.02, "offset": 0.5}, "HCM01": {"factor": -1.0}}
# The seeds above, as the model variables they become.
SEEDED_VALUES = {
    "BPM07.offset_x": 5e-5,
    "BPM07.gain_y": 1.05,
    "BPM07.polarity_x": -1.0,
    "QF07.cal_factor": 1.02,
    "QF07.cal_offset": 0.5,
    "HCM01.cal_factor": -1.0,
}


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


@pytest.fixture(scope="module")
def unfaulted_model() -> PyATRingModel:
    """A shared unfaulted model, for tests that only read or are refused."""
    return PyATRingModel()


@pytest.fixture(scope="module")
def seeded_model() -> PyATRingModel:
    """A shared model booted with ``SEEDED_*`` faults, for read-only tests."""
    return PyATRingModel(bpm_errors=SEEDED_BPM_ERRORS, corrector_gains=SEEDED_CORRECTOR_GAINS)


@pytest.fixture(scope="module")
def shared_ring() -> at.Lattice:
    return build_ring()


@pytest.fixture
def refusal_ring(shared_ring, monkeypatch) -> at.Lattice:
    """The ring a refused construction is handed, kept for inspection.

    A refusal must happen before the ring is touched, so one ring serves
    every refusal test -- and each asserts it is still pristine.
    """
    monkeypatch.setattr(pyat_module, "build_ring", lambda: shared_ring)
    assert fault_attributes(shared_ring) == []
    return shared_ring


def fault_attributes(ring: at.Lattice) -> list[tuple[str, str]]:
    """Every ``(FamName, attribute)`` fault attribute any element carries."""
    return [
        (element.FamName, attribute)
        for element in ring
        for attribute in vars(element)
        if attribute.startswith(("bpm_", "mag_cal_"))
    ]


def element_of(model: PyATRingModel, fam_name: str) -> at.Element:
    return model.lattice[model.element_index(fam_name)]


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
        strengths.append(float(model.lattice[model.element_index(f"{family}{device}")].K))
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
        bands = _load_limit_bands(setpoints=setpoint_addresses(build_manifest()["channels"]))

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

    def test_catalog_plus_faults_plus_optics_is_the_whole_roster(self, unfaulted_model, catalog):
        """The address-named catalog, and beside it the dot-named faults and
        the three optics arrays."""
        assert set(unfaulted_model.supported_variables) == (
            set(catalog) | FAULT_NAMES | set(OPTICS_NAMES)
        )
        assert len(unfaulted_model.supported_variables) == (
            EXPECTED_INPUTS + EXPECTED_OUTPUTS + EXPECTED_FAULT_VARIABLES + len(OPTICS_SHAPES)
        )


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


class TestFaultVariableRoster:
    def test_declares_1344_fault_writables(self, unfaulted_model):
        faults = {
            name: variable
            for name, variable in unfaulted_model.supported_variables.items()
            if name in FAULT_NAMES
        }
        assert len(FAULT_NAMES) == EXPECTED_FAULT_VARIABLES
        assert len(faults) == EXPECTED_FAULT_VARIABLES
        assert not [name for name, variable in faults.items() if variable.read_only]

    def test_fault_names_never_parse_as_a_channel_address(self):
        """Dot grammar: ``BPM01.offset_x`` has no colon, so nothing that
        routes by the six-level address can mistake it for a PV."""
        for name in FAULT_NAMES:
            assert ":" not in name, name
            assert len(name.split(".")) == 2, name

    def test_bpm_fields_bind_their_attribute_and_bounds(self, unfaulted_model):
        variables = unfaulted_model.supported_variables
        for bpm in BPMS:
            for field, bounds in BPM_ERROR_FIELD_BOUNDS.items():
                variable = variables[f"{bpm}.{field}"]
                assert variable.element_name == bpm
                assert variable.attribute == f"bpm_{field}"
                assert variable.default_validation_config == "error"
                if field in BPM_POLARITY_FIELDS:
                    # A polarity is a sign, never a scale: an enum on the bounds.
                    assert isinstance(variable, PyATWritableEnumVariable), field
                    assert variable.options == list(bounds)
                else:
                    assert isinstance(variable, PyATWritableScalarVariable), field
                    assert variable.value_range == bounds

    def test_magnet_fields_bind_their_attribute_and_bounds(self, unfaulted_model):
        variables = unfaulted_model.supported_variables
        for magnet in MAGNETS:
            for field, bounds in MAGNET_CAL_BOUNDS.items():
                variable = variables[f"{magnet}.{field}"]
                assert isinstance(variable, PyATWritableScalarVariable)
                assert variable.element_name == magnet
                assert variable.attribute == f"mag_{field}"
                assert variable.value_range == bounds
                assert variable.default_validation_config == "error"


class TestFaultSeeding:
    def test_unseeded_faults_read_identity(self, unfaulted_model):
        for name, value in unfaulted_model.get(sorted(FAULT_NAMES)).items():
            assert value == IDENTITY[name.split(".")[1]], name

    def test_seeds_read_back(self, seeded_model):
        assert seeded_model.get(list(SEEDED_VALUES)) == SEEDED_VALUES

    def test_every_other_fault_of_a_seeded_model_reads_identity(self, seeded_model):
        for name, value in seeded_model.get(sorted(FAULT_NAMES - set(SEEDED_VALUES))).items():
            assert value == IDENTITY[name.split(".")[1]], name

    def test_seeds_land_on_the_element_attributes(self, seeded_model):
        for name, value in SEEDED_VALUES.items():
            fam_name, field = name.split(".")
            prefix = "bpm_" if fam_name in BPMS else "mag_"
            assert getattr(element_of(seeded_model, fam_name), prefix + field) == value, name

    def test_every_fault_device_carries_identity_attributes(self, unfaulted_model):
        """Every BPM and magnet, not only seeded ones: a writable whose
        element lacks the attribute is refused at construction."""
        for bpm in BPMS:
            element = element_of(unfaulted_model, bpm)
            for field in BPM_ERROR_FIELD_BOUNDS:
                assert getattr(element, f"bpm_{field}") == IDENTITY[field], (bpm, field)
        for magnet in MAGNETS:
            element = element_of(unfaulted_model, magnet)
            for field in MAGNET_CAL_BOUNDS:
                assert getattr(element, f"mag_{field}") == IDENTITY[field], (magnet, field)

    def test_seeded_faults_leave_the_orbit_untouched(
        self, seeded_model, unfaulted_model, bpm_readings
    ):
        """A seed is state on the element for the serving layer to apply;
        it never lands on an attribute pyAT tracks through."""
        assert seeded_model.get(bpm_readings) == unfaulted_model.get(bpm_readings)


class TestFaultSeedRefusal:
    def test_unknown_fam_names_are_refused_before_any_mutation(self, refusal_ring):
        with pytest.raises(UnknownDeviceError, match=r"\['BPM98', 'BPM99'\].*\['QF99'\]") as exc:
            PyATRingModel(
                bpm_errors={
                    "BPM99": {"offset_x": 1e-4},
                    "BPM01": {"offset_x": 1e-4},
                    "BPM98": {"gain_x": 1.1},
                },
                corrector_gains={"QF99": {"factor": 1.1}, "QF01": {"factor": 1.1}},
            )
        assert "BPM01" not in str(exc.value)
        assert "'QF01'" not in str(exc.value)
        assert fault_attributes(refusal_ring) == []

    def test_a_bpm_seed_on_a_magnet_is_refused(self, refusal_ring):
        """QF01 is a ring element, but not a BPM."""
        with pytest.raises(UnknownDeviceError, match="QF01"):
            PyATRingModel(bpm_errors={"QF01": {"offset_x": 1e-4}})
        assert fault_attributes(refusal_ring) == []

    def test_a_calibration_seed_on_a_bpm_is_refused(self, refusal_ring):
        with pytest.raises(UnknownDeviceError, match="BPM01"):
            PyATRingModel(corrector_gains={"BPM01": {"factor": 1.1}})
        assert fault_attributes(refusal_ring) == []

    @pytest.mark.parametrize(
        ("seeds", "offender"),
        [
            ({"bpm_errors": {"BPM01": {"cal_x": 1e-4}}}, "BPM01.cal_x"),
            ({"corrector_gains": {"QF01": {"gain": 1.1}}}, "QF01.gain"),
        ],
    )
    def test_an_unknown_field_is_refused_before_any_mutation(self, refusal_ring, seeds, offender):
        """A field the model has no variable for would be a seed silently dropped."""
        with pytest.raises(ValueError, match=re.escape(f"['{offender}']")) as exc:
            PyATRingModel(**seeds)
        assert not isinstance(exc.value, UnknownDeviceError)
        assert fault_attributes(refusal_ring) == []

    @pytest.mark.parametrize(
        "seeds",
        [
            {"bpm_errors": {"BPM01": {"gain_x": 50.0}}},
            {"bpm_errors": {"BPM01": {"polarity_y": 0.5}}},
            {"corrector_gains": {"QF01": {"factor": 6.0}}},
            {"corrector_gains": {"HCM01": {"offset": -11.0}}},
        ],
    )
    def test_an_out_of_range_seed_is_refused_before_any_mutation(self, refusal_ring, seeds):
        with pytest.raises(ValidationError):
            PyATRingModel(**seeds)
        assert fault_attributes(refusal_ring) == []


class TestFaultWrites:
    @pytest.mark.parametrize(
        ("name", "value", "reason"),
        [
            ("BPM01.offset_x", 1.0, "out of valid range"),
            ("BPM01.gain_y", 0.0, "out of valid range"),
            ("BPM01.polarity_x", 0.5, "allowed options"),
            ("QF07.cal_factor", 6.0, "out of valid range"),
            ("HCM01.cal_offset", 11.0, "out of valid range"),
        ],
    )
    def test_an_out_of_range_set_is_refused_by_lume(self, unfaulted_model, name, value, reason):
        with pytest.raises(ValueError, match=f"Validation failed for variable '{name}'.*{reason}"):
            unfaulted_model.set({name: value})
        assert unfaulted_model.get(name) == IDENTITY[name.split(".")[1]]

    def test_an_in_range_write_reads_back_and_lands_on_the_element(self, model):
        writes = {"BPM01.offset_x": 1e-4, "BPM01.polarity_y": -1.0, "QF07.cal_offset": 2.0}
        model.set(writes)
        assert model.get(list(writes)) == writes
        assert element_of(model, "BPM01").bpm_offset_x == 1e-4
        assert element_of(model, "BPM01").bpm_polarity_y == -1.0
        assert element_of(model, "QF07").mag_cal_offset == 2.0

    def test_reset_restores_the_seeds_not_identity(self):
        """Reset undoes writes, not faults: a seed is the fault's default."""
        seeded = PyATRingModel(bpm_errors=SEEDED_BPM_ERRORS, corrector_gains=SEEDED_CORRECTOR_GAINS)
        seeded.set({"BPM07.offset_x": -5e-5, "BPM08.gain_x": 2.0, "QF07.cal_factor": 1.0})
        seeded.reset()
        assert seeded.get(list(SEEDED_VALUES)) == SEEDED_VALUES
        assert seeded.get("BPM08.gain_x") == 1.0
        assert element_of(seeded, "BPM07").bpm_offset_x == 5e-5


@pytest.fixture
def optics_calls(monkeypatch) -> list[tuple]:
    """Record every ``at.get_optics`` call, passing each one through."""
    calls: list[tuple] = []
    real = at.get_optics

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(at, "get_optics", recording)
    return calls


def bpm_positions(model: PyATRingModel) -> np.ndarray:
    """The model's own BPM readings as ``(x, y)`` rows, BPM01 first."""
    addresses = [f"SR:DIAG:BPM:{bpm[3:]}:POSITION:{axis}" for bpm in BPMS for axis in "XY"]
    readings = model.get(addresses)
    return np.array([readings[address] for address in addresses]).reshape(len(BPMS), 2)


def assert_same_optics(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> None:
    assert set(first) == set(second) == set(OPTICS_NAMES)
    for name in OPTICS_NAMES:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


class TestOpticsVariables:
    """Tunes, beta and the true orbit at the BPMs: read-only arrays computed
    from the solved lattice when read, once per solve, and never on a write."""

    def test_optics_are_elementless_read_only_nd_variables(self, unfaulted_model):
        for name, shape in OPTICS_SHAPES.items():
            variable = unfaulted_model.supported_variables[name]
            assert isinstance(variable, PyATReadOnlyNDVariable), name
            assert variable.read_only is True, name
            assert variable.element_name is None, name
            assert variable.shape == shape, name
            assert ":" not in name, name  # never parses as a channel address

    def test_optics_values_have_their_declared_shape_and_are_finite(self, unfaulted_model):
        values = unfaulted_model.get(OPTICS_NAMES)
        for name, shape in OPTICS_SHAPES.items():
            value = values[name]
            assert isinstance(value, np.ndarray), name
            assert value.dtype == np.float64, name
            assert value.shape == shape, name
            assert np.all(np.isfinite(value)), name
        assert np.all((values["tunes"] > 0.0) & (values["tunes"] < 1.0))  # fractional tunes
        assert np.all(values["beta_at_bpms"] > 0.0)

    def test_optics_match_an_independent_optics_pass_at_the_monitors(self, model):
        """pyAT's own linear optics of the live lattice, at every monitor, in
        ring order -- after a write, so a stale boot-time answer cannot pass."""
        nominal_tunes = model.get("tunes")
        model.set({A_QUADRUPOLE: model.get(A_QUADRUPOLE) * 1.001})

        refpts = [i for i, element in enumerate(model.lattice) if isinstance(element, at.Monitor)]
        assert [model.lattice[i].FamName for i in refpts] == BPMS
        _, ringdata, elemdata = at.get_optics(model.lattice, refpts=refpts)

        values = model.get(OPTICS_NAMES)
        np.testing.assert_array_equal(values["tunes"], ringdata.tune)
        np.testing.assert_array_equal(values["beta_at_bpms"], elemdata.beta)
        assert not np.array_equal(values["tunes"], nominal_tunes)

    def test_orbit_at_bpms_is_the_model_bpm_position_in_ring_order(self, model):
        model.set({A_CORRECTOR: 5.0})
        orbit = model.get("orbit_at_bpms")
        np.testing.assert_array_equal(orbit, bpm_positions(model))
        # The kick moved the orbit: not a vacuous zero-against-zero match.
        assert np.any(np.abs(orbit) > 1e-9)

    def test_orbit_at_bpms_is_the_truth_a_bpm_fault_never_reaches(self, model):
        model.set({A_CORRECTOR: 5.0})
        before = model.get("orbit_at_bpms")
        model.set({"BPM01.offset_x": 1e-4, "BPM01.gain_y": 1.5, "BPM02.polarity_x": -1.0})
        np.testing.assert_array_equal(model.get("orbit_at_bpms"), before)

    def test_two_consecutive_optics_reads_compute_once(self, model, optics_calls):
        first = model.get(OPTICS_NAMES)
        second = model.get(OPTICS_NAMES)
        model.get("tunes")
        model.get(["beta_at_bpms", A_BPM_READING])
        assert len(optics_calls) == 1
        assert_same_optics(first, second)

    def test_an_optics_read_mixed_with_scalars_answers_every_name_in_order(self, model):
        names = [A_BPM_READING, "tunes", A_CORRECTOR, "orbit_at_bpms"]
        values = model.get(names)
        assert list(values) == names
        assert values[A_CORRECTOR] == model.get(A_CORRECTOR)
        assert values[A_BPM_READING] == values["orbit_at_bpms"][0, 0]

    def test_a_returned_optics_array_cannot_corrupt_the_next_read(self, model):
        tunes = model.get("tunes")
        expected = tunes.copy()
        tunes[:] = 0.0
        np.testing.assert_array_equal(model.get("tunes"), expected)

    def test_a_setpoint_write_never_runs_the_optics(self, model, optics_calls):
        model.set({A_CORRECTOR: 5.0})
        model.set({A_QUADRUPOLE: model.get(A_QUADRUPOLE) * 1.001, "BPM01.offset_x": 1e-4})
        model.reset()
        assert optics_calls == []

    def test_a_solve_invalidates_the_optics_memo(self, model, optics_calls):
        before = model.get("tunes")
        model.set({A_QUADRUPOLE: model.get(A_QUADRUPOLE) * 1.001})
        after = model.get("tunes")
        assert len(optics_calls) == 2
        assert not np.array_equal(after, before)

    def test_a_rolled_back_write_keeps_the_optics_of_the_restored_solve(self, model, optics_calls):
        """Rollback puts the prior lattice and the prior solve back, so the
        optics already computed for that solve still describe the ring."""
        before = model.get(OPTICS_NAMES)
        fault = {address: model.get(address) * 1.2 for address in quadrupole_setpoints("QF")}
        with pytest.raises(OrbitSolveError):
            model.set(fault)
        assert_same_optics(model.get(OPTICS_NAMES), before)
        assert len(optics_calls) == 1

    @pytest.mark.parametrize("name", OPTICS_NAMES)
    def test_writing_an_optics_variable_is_refused(self, unfaulted_model, name):
        value = np.zeros(OPTICS_SHAPES[name])
        with pytest.raises(ReadOnlyError, match=name):
            unfaulted_model.set({name: value})

    def test_an_optics_variable_is_computed_by_its_model_not_by_itself(self, unfaulted_model):
        """The variable declares the array; computing it once per solve is
        model state, so reading one straight off the simulator is refused."""
        variable = unfaulted_model.supported_variables["tunes"]
        with pytest.raises(NotImplementedError, match="tunes"):
            variable._get(unfaulted_model.simulator)
