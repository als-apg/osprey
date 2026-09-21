"""The model is built from the tree it is served, and from no other.

:class:`~osprey.services.virtual_accelerator.model.pyat.PyATRingModel` is the
facility adapter: handed a served data directory and the channel list its
deployment resolved, it loads that tree's lattice, binds the variables its
``va_bindings.json`` describes, and declares the nominals and bands its
``machine.json`` and ``channel_limits.json`` ship. The tests below pin that
seam rather than any facility's physics -- what reaches the model is what the
tree says, and a tree that contradicts itself refuses to boot.

The ring here is synthetic: a small stable FODO lattice with one cavity, eight
cells, unique names on every bound element. That is the shape a real emitted
tree has in the respects this module cares about -- a ring that solves a 6D
closed orbit, elements a binding can address by ``FamName``, and monitors that
can be told apart -- and nothing here needs a particular accelerator. No
address in this module carries a facility's vocabulary either: the bindings
document is the only thing that pairs an address with an element.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import at
import pytest

from osprey.services.virtual_accelerator.bindings import BindingsError
from osprey.services.virtual_accelerator.lattice.calibration import energy_factor
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_STATIC_NOISY,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.fault_bounds import BPM_ERROR_FIELDS
from osprey.services.virtual_accelerator.model.pyat import (
    OPTICS_NAMES,
    OrbitSolveError,
    PyATRingModel,
    UnknownDeviceError,
)
from osprey.services.virtual_accelerator.model.variables import (
    EnergyVariable,
    KickVariable,
    MonitorVariable,
    RFVariable,
    StrengthVariable,
)

C_LIGHT = 299792458.0

#: The deck this fixture tree was exported at, in GeV.
DECK_ENERGY_GEV = 3.0

#: The focusing strength the ring is built with, and the hardware nominal the
#: calibration below maps onto it -- so the tree boots in its own nominal
#: state, the way an emitted one does.
QUAD_K = 1.1
QUAD_GAIN = 0.01
QUAD_NOMINAL_AMPS = QUAD_K / QUAD_GAIN

#: Radians of kick per amp, and the ring's cavity harmonic number.
KICK_GAIN = 1.0e-6
HARMONIC = 88

# The addresses the fixture document binds. Nothing relates them to the
# element names below; the bindings document does that and only it.
QUAD_SP = "R1:PWR:QUAD_A:07:CUR:SP"
QUAD_RB = "R1:PWR:QUAD_A:07:CUR:RB"
CORR_SP = "R1:PWR:CORR_A:03:CUR:SP"
CAVITY_SP = "R1:RF:CAV_A:01:FREQ:SP"
BPM_X = "R1:DIA:MON_A:12:POS:X"
BEND_SP = "R1:PWR:BEND_A:01:CUR:SP"
STATIC_RB = "R1:VAC:GAU_A:01:PRES:RB"

# The deck's own names for the elements each address drives.
QUAD_ELEMENT = "QF1"
CORR_ELEMENTS = ("HC1", "HC2")
MONITOR_ELEMENT = "BPM1"
CAVITY_ELEMENT = "RFC"


# -- the synthetic emitted tree ----------------------------------------------


def _ring(cells: int = 8, kq: float = QUAD_K) -> at.Lattice:
    """A small stable ring with one cavity and uniquely named magnets.

    Eight FODO cells, each carrying a monitor and a corrector, closed by
    sixteen dipoles. Returned 4D, the way a ring saved out of a facility's
    simulator model arrives.
    """
    angle = 2 * math.pi / (2 * cells)
    elements: list[Any] = []
    for cell in range(1, cells + 1):
        elements += [
            at.Quadrupole(f"QF{cell}", 0.3, kq),
            at.Drift("DR", 1.0),
            at.Dipole(f"BD{cell}A", 1.0, angle),
            at.Monitor(f"BPM{cell}"),
            at.Drift("DR", 1.0),
            at.Quadrupole(f"QD{cell}", 0.3, -kq),
            at.Drift("DR", 1.0),
            at.Corrector(f"HC{cell}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BD{cell}B", 1.0, angle),
            at.Drift("DR", 1.0),
        ]
    ring = at.Lattice(elements, name="fixture", energy=DECK_ENERGY_GEV * 1.0e9, periodicity=1)
    frequency = HARMONIC * C_LIGHT / ring.circumference
    ring.append(at.RFCavity(CAVITY_ELEMENT, 0.0, 1.0e6, frequency, HARMONIC, ring.energy))
    ring.disable_6d()
    return ring


#: The cavity frequency the fixture ring is built at, in MHz -- the hardware
#: unit the rf calibration below states.
CAVITY_NOMINAL_MHZ = HARMONIC * C_LIGHT / _ring().circumference / 1.0e6


def _linear(gain: float, offset: float = 0.0) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _strength(**overrides: Any) -> dict:
    """A quadrupole setpoint, onto one element's ``PolynomB[1]``."""
    body = {
        "kind": "strength",
        "family": "quad_a",
        "setpoint_address": QUAD_SP,
        "readback_address": QUAD_RB,
        "readback": "inverse",
        "element": QUAD_ELEMENT,
        "attribute": "PolynomB",
        "index": 1,
        "slices": [{"element": QUAD_ELEMENT, "weight": 1.0}],
        "owner": "quad_a",
        "calibration": _linear(QUAD_GAIN),
        "monitor_inverse": _linear(1.0 / QUAD_GAIN),
        "nominal": QUAD_NOMINAL_AMPS,
        "energy_scaling": "brho",
        "energy_table": None,
    }
    body.update(overrides)
    return body


def _kick(**overrides: Any) -> dict:
    """A corrector setpoint shared equally over two lattice pieces."""
    body = _strength(
        kind="kick",
        family="corr_a",
        setpoint_address=CORR_SP,
        readback_address=CORR_SP.replace(":SP", ":RB"),
        readback="identity",
        element=CORR_ELEMENTS[0],
        attribute="KickAngle",
        index=0,
        slices=[{"element": name, "weight": 0.5} for name in CORR_ELEMENTS],
        owner="corr_a",
        calibration=_linear(KICK_GAIN),
        monitor_inverse=None,
        nominal=0.0,
    )
    body.update(overrides)
    return body


def _monitor(**overrides: Any) -> dict:
    """One orbit reading: metres on the ring, millimetres on the wire."""
    body = _strength(
        kind="monitor",
        family="mon_a",
        setpoint_address=BPM_X,
        readback_address=None,
        readback="inverse",
        element=MONITOR_ELEMENT,
        attribute="x",
        index=None,
        slices=[{"element": MONITOR_ELEMENT, "weight": 1.0}],
        owner="mon_a",
        calibration=_linear(1.0e-3),
        monitor_inverse=_linear(1.0e3),
        nominal=None,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _rf(**overrides: Any) -> dict:
    """The cavity frequency, written in MHz."""
    body = _strength(
        kind="rf",
        family="cav_a",
        setpoint_address=CAVITY_SP,
        readback_address=None,
        readback="same_as_setpoint",
        element=CAVITY_ELEMENT,
        attribute="Frequency",
        index=None,
        slices=[{"element": CAVITY_ELEMENT, "weight": 1.0}],
        owner="cav_a",
        calibration=_linear(1.0e6),
        monitor_inverse=None,
        nominal=CAVITY_NOMINAL_MHZ,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _energy(**overrides: Any) -> dict:
    """The ring's energy knob: the one binding that drives no element."""
    body = _strength(
        kind="energy",
        family="bend_a",
        setpoint_address=BEND_SP,
        readback_address=BEND_SP.replace(":SP", ":RB"),
        readback="identity",
        element=None,
        attribute=None,
        index=None,
        slices=[],
        owner=None,
        calibration=None,
        monitor_inverse=None,
        nominal=300.0,
        energy_scaling="none",
        energy_table={
            "kind": "table",
            "grid": [270.0, 300.0, 330.0],
            "values": [2.7, DECK_ENERGY_GEV, 3.3],
        },
    )
    body.update(overrides)
    return body


def _channel(address: str, *, partition: str = PARTITION_PYAT_COUPLED, **overrides: Any) -> dict:
    """One manifest channel, in the per-channel schema the manifest carries."""
    ring, system, family, device, field, subfield = address.split(":")
    channel = {
        "address": address,
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": "ai",
        "noise": False,
    }
    channel.update(overrides)
    return channel


def _manifest() -> list[dict]:
    """The channel list a deployment of this tree resolved.

    Order matters in one place: the catalog is built in manifest order, so
    this is the order the energy knob adopts the setpoints it rescales in.
    """
    return [
        _channel(QUAD_SP),
        _channel(QUAD_RB),
        _channel(CORR_SP),
        _channel(CAVITY_SP),
        _channel(BPM_X),
        _channel(BEND_SP),
        _channel(STATIC_RB, partition=PARTITION_STATIC_NOISY),
    ]


def _machine() -> dict:
    """Every nominal and unit the served ``machine.json`` declares."""
    return {
        QUAD_SP: {"value": QUAD_NOMINAL_AMPS, "units": "A"},
        CORR_SP: {"value": 0.0, "units": "A"},
        CAVITY_SP: {"value": CAVITY_NOMINAL_MHZ, "units": "MHz"},
        BPM_X: {"value": 0.0, "units": "mm"},
        BEND_SP: {"value": 300.0, "units": "A"},
    }


def _limits() -> dict:
    """The write bands the served ``channel_limits.json`` ships."""
    return {
        "_version": "1.0",
        "defaults": {"writable": True, "confirm": True},
        QUAD_SP: {"min_value": 0.0, "max_value": 200.0},
        CORR_SP: {"min_value": -10.0, "max_value": 10.0},
        CAVITY_SP: {"min_value": 499.0, "max_value": 500.0},
        BEND_SP: {"min_value": 250.0, "max_value": 350.0},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(
    root: Path,
    *,
    ring: at.Lattice | None = None,
    bindings: list[dict] | None = None,
    machine: dict | None = None,
    limits: dict | None = None,
    digest: str | None = None,
) -> Path:
    """Write a served tree and return the data directory addressing it.

    Args:
        root: the data root to write. ``simulation/`` holds the lattice, the
            bindings and the scenario seed; the write bands sit beside it, the
            way :class:`ManifestPaths` resolves a facility tree.
        ring: the lattice to save; the fixture ring when omitted.
        bindings: the document's bindings; one of each kind when omitted.
        machine: the ``machine.json`` channel block, or ``None`` to write no
            scenario seed at all.
        limits: the ``channel_limits.json`` body, or ``None`` to write none.
        digest: the digest to stamp instead of the saved lattice's own -- how
            a tree whose two files describe different rings is made.

    Returns:
        The data root, ready to hand to :class:`PyATRingModel`.
    """
    paths = ManifestPaths(data_root=root)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(_ring() if ring is None else ring, paths.lattice_json)
    document = {
        "system": "StorageRing",
        "energy_gev": DECK_ENERGY_GEV,
        "lattice_sha256": _sha256(paths.lattice_json) if digest is None else digest,
        "bindings": [_strength(), _kick(), _monitor(), _rf(), _energy()]
        if bindings is None
        else bindings,
    }
    paths.va_bindings.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if machine is not None:
        paths.machine_json.write_text(json.dumps({"name": "fixture", "channels": machine}))
    if limits is not None:
        paths.channel_limits.write_text(json.dumps(limits))
    return root


def _element(model: PyATRingModel, name: str):
    """The ring element a bound name addresses, the way a binding does."""
    return model.lattice[model.element_index(name)]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A served tree whose files all agree -- the boot case."""
    return _tree(
        tmp_path_factory.mktemp("served") / "data",
        machine=_machine(),
        limits=_limits(),
    )


@pytest.fixture(scope="module")
def booted(data_dir: Path) -> PyATRingModel:
    """One boot, shared by every test that only reads the model."""
    return PyATRingModel(data_dir, _manifest())


@pytest.fixture
def model(data_dir: Path) -> PyATRingModel:
    """A fresh model, for the tests that write to it."""
    return PyATRingModel(data_dir, _manifest())


# -- the boot path -----------------------------------------------------------


class TestBootsOnTheServedTree:
    """What the model exposes is what the tree it was handed describes."""

    def test_the_variables_are_the_coupled_channels_of_the_manifest(
        self, booted: PyATRingModel
    ) -> None:
        """Keyed by address, one per coupled channel the model can drive.

        The setpoint echo is the serving layer's mirror of a write, not model
        state, and the channel no lattice backs is not the model's at all.
        Everything else the model declares is named for an element rather than
        an address -- the faults and the optics -- and is checked below.
        """
        addressed = set(booted.supported_variables) - booted.derived_names
        assert addressed == {QUAD_SP, CORR_SP, CAVITY_SP, BPM_X, BEND_SP}

    def test_each_binding_becomes_the_class_that_implements_its_kind(
        self, booted: PyATRingModel
    ) -> None:
        variables = booted.supported_variables
        assert isinstance(variables[QUAD_SP], StrengthVariable)
        assert isinstance(variables[CORR_SP], KickVariable)
        assert isinstance(variables[CAVITY_SP], RFVariable)
        assert isinstance(variables[BPM_X], MonitorVariable)
        assert isinstance(variables[BEND_SP], EnergyVariable)

    def test_the_ring_is_the_lattice_the_tree_ships(
        self, booted: PyATRingModel, data_dir: Path
    ) -> None:
        saved = at.load_lattice(ManifestPaths(data_root=data_dir).lattice_json)
        assert [element.FamName for element in booted.lattice] == [
            element.FamName for element in saved
        ]

    def test_longitudinal_motion_is_on_for_the_cavity(self, booted: PyATRingModel) -> None:
        """``build_ring``'s doing, and the condition the orbit guard assumes."""
        assert booted.lattice.is_6d

    def test_the_declared_nominals_are_the_served_ones(self, booted: PyATRingModel) -> None:
        """Every writable boots holding the tree's own ``machine.json`` value."""
        assert booted.get(QUAD_SP) == pytest.approx(QUAD_NOMINAL_AMPS)
        assert booted.get(BEND_SP) == pytest.approx(300.0)

    def test_a_setpoint_declares_the_served_unit_and_band(self, booted: PyATRingModel) -> None:
        variable = booted.supported_variables[QUAD_SP]
        assert variable.unit == "A"
        assert variable.value_range == (0.0, 200.0)

    def test_each_model_owns_its_own_ring(self, data_dir: Path) -> None:
        """No caching anywhere below: one model's writes never reach another."""
        first = PyATRingModel(data_dir, _manifest())
        first.set({QUAD_SP: 1.05 * QUAD_NOMINAL_AMPS})

        second = PyATRingModel(data_dir, _manifest())

        assert second.get(QUAD_SP) == pytest.approx(QUAD_NOMINAL_AMPS)
        assert _element(second, QUAD_ELEMENT).PolynomB[1] == pytest.approx(QUAD_K)


class TestTheEnergyKnobIsCoupled:
    """``couple_energy_knob`` is not optional, so the boot has to run it."""

    def test_every_rigidity_scaled_setpoint_is_adopted(self, booted: PyATRingModel) -> None:
        """The brho ones in catalog order, and only those: an rf frequency
        does not move with the beam rigidity.

        Without the coupling the knob would write electron-volts and rescale
        nothing -- and the fields a rescale touches would sit outside the
        model's rollback.
        """
        knob = booted.supported_variables[BEND_SP]

        assert [variable.name for variable in knob._scaled] == [QUAD_SP, CORR_SP]

    def test_an_energy_write_rescales_what_it_adopted(self, model: PyATRingModel) -> None:
        """A magnet left at a fixed current follows the energy, as the control
        system's own conversion would put it."""
        before = _element(model, QUAD_ELEMENT).PolynomB[1]

        model.set({BEND_SP: 330.0})

        factor = energy_factor(model.lattice.energy / 1.0e9, DECK_ENERGY_GEV)
        assert _element(model, QUAD_ELEMENT).PolynomB[1] == pytest.approx(before * factor, rel=1e-9)
        assert model.lattice.energy > DECK_ENERGY_GEV * 1.0e9, "the ring moved up in energy"
        assert model.get(QUAD_SP) == pytest.approx(QUAD_NOMINAL_AMPS), "the current has not moved"


class TestTheServedValues:
    """What a read answers, and where the answer comes from."""

    def test_a_writable_is_served_from_its_retained_value(self, model: PyATRingModel) -> None:
        """Get-after-set is the value written, bit for bit -- never a
        re-derivation through the calibration."""
        model.set({QUAD_SP: 123.25})

        assert model.get(QUAD_SP) == 123.25

    def test_a_monitor_is_served_from_the_solve(self, booted: PyATRingModel) -> None:
        """In the facility's own hardware unit: the metres-to-millimetres step
        is the exported inverse curve, not anything this layer multiplies."""
        reading = booted.get(BPM_X)

        assert isinstance(reading, float)
        assert reading == pytest.approx(0.0, abs=1e-6), "an unperturbed ring sits on axis"


class TestWritesReachTheBoundElements:
    """A setpoint is written where its binding says, through its calibration."""

    def test_a_strength_lands_on_the_bound_polynomial_component(self, model: PyATRingModel) -> None:
        model.set({QUAD_SP: 120.0})

        assert _element(model, QUAD_ELEMENT).PolynomB[1] == pytest.approx(120.0 * QUAD_GAIN)

    def test_a_kick_is_shared_equally_between_its_slices(self, model: PyATRingModel) -> None:
        elements = {element.FamName: element for element in model.lattice}

        model.set({CORR_SP: 2.0})

        for name in CORR_ELEMENTS:
            assert elements[name].KickAngle[0] == pytest.approx(0.5 * 2.0 * KICK_GAIN)

    def test_a_write_outside_the_declared_band_is_not_enforced_here(
        self, model: PyATRingModel
    ) -> None:
        """Declared ranges are metadata: enforcement is the record's drive
        limits and the write-safety database, never this model."""
        assert model.supported_variables[CORR_SP].value_range == (-10.0, 10.0)

        model.set({CORR_SP: 20.0})

        assert model.get(CORR_SP) == 20.0


# -- refusals, all read off the served tree ----------------------------------


class TestTheServedTreeIsWhatRefuses:
    """Every boot refusal is a statement about the files this model was given."""

    def test_a_nominal_outside_the_served_band_refuses_the_boot(self, tmp_path: Path) -> None:
        """The facility's own limits exclude the facility's own nominal: the
        model refuses rather than serving a value that cannot be written."""
        machine = _machine() | {QUAD_SP: {"value": 300.0, "units": "A"}}
        data_dir = _tree(tmp_path / "data", machine=machine, limits=_limits())

        with pytest.raises(ValueError, match="out of valid range"):
            PyATRingModel(data_dir, _manifest())

    def test_the_band_is_the_one_this_tree_ships(self, tmp_path: Path) -> None:
        """Same manifest, same nominal, a narrower band: the refusal follows
        the served file rather than any tree the package carries."""
        limits = _limits() | {QUAD_SP: {"min_value": 0.0, "max_value": 20.0}}
        data_dir = _tree(tmp_path / "data", machine=_machine(), limits=limits)

        with pytest.raises(ValueError, match="out of valid range"):
            PyATRingModel(data_dir, _manifest())

    def test_a_tree_with_no_scenario_seed_names_the_file_it_wanted(self, tmp_path: Path) -> None:
        data_dir = _tree(tmp_path / "data", limits=_limits())

        with pytest.raises(FileNotFoundError) as excinfo:
            PyATRingModel(data_dir, _manifest())

        assert str(ManifestPaths(data_root=data_dir).machine_json) in str(excinfo.value)

    def test_a_tree_with_no_write_bands_names_the_file_it_wanted(self, tmp_path: Path) -> None:
        data_dir = _tree(tmp_path / "data", machine=_machine())

        with pytest.raises(FileNotFoundError) as excinfo:
            PyATRingModel(data_dir, _manifest())

        assert str(ManifestPaths(data_root=data_dir).channel_limits) in str(excinfo.value)

    def test_a_lattice_the_bindings_were_not_derived_against_is_refused(
        self, tmp_path: Path
    ) -> None:
        """Every element name, index and nominal in the document was read off
        one particular ring, so the digest is what says this is that ring."""
        data_dir = _tree(tmp_path / "data", machine=_machine(), limits=_limits(), digest="b" * 64)

        with pytest.raises(BindingsError) as excinfo:
            PyATRingModel(data_dir, _manifest())

        assert excinfo.value.key == "lattice_sha256"
        assert str(ManifestPaths(data_root=data_dir).lattice_json) in str(excinfo.value)

    def test_a_bound_element_the_lattice_does_not_carry_is_refused(self, tmp_path: Path) -> None:
        data_dir = _tree(
            tmp_path / "data",
            bindings=[_strength(element="QF99", slices=[{"element": "QF99", "weight": 1.0}])],
            machine=_machine(),
            limits=_limits(),
        )

        with pytest.raises(BindingsError) as excinfo:
            PyATRingModel(data_dir, _manifest())

        assert excinfo.value.key == "bindings[0].slices[0].element"
        assert "QF99" in excinfo.value.message

    def test_a_coupled_address_the_document_binds_to_nothing_refuses(self, tmp_path: Path) -> None:
        """A channel the manifest declares coupled and the document does not
        bind would otherwise serve as a variable that accepts writes, reads
        them back, and moves nothing on the lattice.

        The catalog's own fallback for an unbound address is a plain declared
        variable, and the backend refuses to adopt one: it drives nothing, so
        it is not a variable this model can serve.
        """
        data_dir = _tree(
            tmp_path / "data",
            bindings=[_kick(), _monitor(), _rf(), _energy()],
            machine=_machine(),
            limits=_limits(),
        )

        with pytest.raises(ValueError, match="Action"):
            PyATRingModel(data_dir, _manifest())


class TestSeededMisalignments:
    """A fault seeded at construction, and how a boot failure reads."""

    def test_a_seeded_misalignment_moves_the_orbit(self, data_dir: Path) -> None:
        faulted = PyATRingModel(
            data_dir, _manifest(), element_misalignments={QUAD_ELEMENT: {"dx": 1.0e-4}}
        )

        assert abs(faulted.get(BPM_X)) > 1.0e-4, "the reading is millimetres off axis"

    def test_a_misalignment_naming_an_element_the_ring_has_not_is_refused(
        self, data_dir: Path
    ) -> None:
        with pytest.raises(UnknownDeviceError):
            PyATRingModel(data_dir, _manifest(), element_misalignments={"QF99": {"dx": 1.0e-4}})

    def test_a_misalignment_that_destroys_the_closed_orbit_names_the_fault(
        self, data_dir: Path
    ) -> None:
        """The boot failure has to say what was seeded: an unstable ring at
        construction is otherwise an opaque numerical complaint."""
        misalignments = {f"QF{cell}": {"dx": 0.4, "roll": 0.8} for cell in range(1, 9)}

        with pytest.raises(OrbitSolveError) as excinfo:
            PyATRingModel(data_dir, _manifest(), element_misalignments=misalignments)

        assert "QF1" in str(excinfo.value)
        assert "0.4" in str(excinfo.value)


class TestTheFaultRosterIsTheServedOne:
    """Which devices carry a fault, and what each fault is worth at boot."""

    def test_a_monitor_carries_every_reading_error_field(self, booted: PyATRingModel) -> None:
        declared = set(booted.supported_variables)

        assert {f"{MONITOR_ELEMENT}.{field}" for field in BPM_ERROR_FIELDS} <= declared

    def test_a_magnet_carries_its_calibration(self, booted: PyATRingModel) -> None:
        declared = set(booted.supported_variables)

        assert {f"{QUAD_ELEMENT}.cal_factor", f"{QUAD_ELEMENT}.cal_offset"} <= declared
        assert {f"{CORR_ELEMENTS[0]}.cal_factor", f"{CORR_ELEMENTS[0]}.cal_offset"} <= declared

    def test_a_cavity_carries_no_calibration(self, booted: PyATRingModel) -> None:
        """A frequency is commanded like a magnet current and is not one."""
        faults = {name for name in booted.supported_variables if name.startswith(CAVITY_ELEMENT)}

        assert faults == set()

    def test_a_split_device_carries_one_calibration_at_the_element_it_reads(
        self, booted: PyATRingModel
    ) -> None:
        """The kick is shared over two elements; the fault is the device's."""
        assert f"{CORR_ELEMENTS[1]}.cal_factor" not in booted.supported_variables

    def test_the_whole_declared_roster_is_the_addresses_plus_these(
        self, booted: PyATRingModel
    ) -> None:
        """Nothing else is declared: the model-only half is exactly the faults
        and the optics, which is what the served partition rests on."""
        faults = {f"{MONITOR_ELEMENT}.{field}" for field in BPM_ERROR_FIELDS} | {
            f"{element}.{field}"
            for element in (QUAD_ELEMENT, CORR_ELEMENTS[0])
            for field in ("cal_factor", "cal_offset")
        }
        addresses = {QUAD_SP, CORR_SP, CAVITY_SP, BPM_X, BEND_SP}

        assert set(booted.supported_variables) == addresses | faults | OPTICS_NAMES

    def test_an_unseeded_fault_reads_identity(self, booted: PyATRingModel) -> None:
        assert booted.get(f"{MONITOR_ELEMENT}.gain_x") == 1.0
        assert booted.get(f"{MONITOR_ELEMENT}.offset_x") == 0.0
        assert booted.get(f"{QUAD_ELEMENT}.cal_factor") == 1.0

    def test_a_seed_is_what_the_fault_reads(self, data_dir: Path) -> None:
        seeded = PyATRingModel(
            data_dir, _manifest(), bpm_errors={MONITOR_ELEMENT: {"offset_x": 0.25}}
        )

        assert seeded.get(f"{MONITOR_ELEMENT}.offset_x") == pytest.approx(0.25)

    def test_a_seed_survives_a_reset(self, data_dir: Path) -> None:
        """``reset`` returns a fault to its seed, never to identity."""
        seeded = PyATRingModel(
            data_dir, _manifest(), bpm_errors={MONITOR_ELEMENT: {"offset_x": 0.25}}
        )
        seeded.set({f"{MONITOR_ELEMENT}.offset_x": -0.1})
        seeded.reset()

        assert seeded.get(f"{MONITOR_ELEMENT}.offset_x") == pytest.approx(0.25)

    def test_a_calibration_seed_is_named_in_the_grammar_the_boot_uses(self, data_dir: Path) -> None:
        seeded = PyATRingModel(
            data_dir, _manifest(), corrector_gains={QUAD_ELEMENT: {"factor": -1.0}}
        )

        assert seeded.get(f"{QUAD_ELEMENT}.cal_factor") == -1.0

    def test_a_fault_moves_no_orbit(self, model: PyATRingModel) -> None:
        """The attribute a fault lands on is read by no pass method."""
        before = model.get(BPM_X)
        model.set({f"{MONITOR_ELEMENT}.offset_x": 0.5, f"{QUAD_ELEMENT}.cal_factor": -1.0})

        assert model.get(BPM_X) == pytest.approx(before)

    def test_a_displacement_of_any_size_is_accepted(self, model: PyATRingModel) -> None:
        """A seeded magnitude is what was asked for, in the monitor's unit."""
        model.set({f"{MONITOR_ELEMENT}.offset_x": 5.0e3})

        assert model.get(f"{MONITOR_ELEMENT}.offset_x") == pytest.approx(5.0e3)

    def test_a_gain_outside_its_window_is_refused(self, model: PyATRingModel) -> None:
        with pytest.raises(ValueError, match="gain_x"):
            model.set({f"{MONITOR_ELEMENT}.gain_x": 50.0})

    def test_a_polarity_between_its_two_values_is_refused(self, model: PyATRingModel) -> None:
        with pytest.raises(ValueError, match="polarity_x"):
            model.set({f"{MONITOR_ELEMENT}.polarity_x": 0.5})

    def test_a_negative_noise_amplitude_is_refused(self, model: PyATRingModel) -> None:
        """A noise amplitude is a standard deviation. Below zero it describes
        no distribution, and the model would hold a value every later reading
        raises on -- so the floor is the model's as well as the parser's."""
        with pytest.raises(ValueError, match="noise_x"):
            model.set({f"{MONITOR_ELEMENT}.noise_x": -1.0})

    def test_a_noise_amplitude_of_zero_or_any_size_is_accepted(self, model: PyATRingModel) -> None:
        """Only the floor is a bound: how wide an operator asks the noise to
        be is what was asked for, in the unit the monitor publishes."""
        model.set({f"{MONITOR_ELEMENT}.noise_x": 0.0})
        assert model.get(f"{MONITOR_ELEMENT}.noise_x") == 0.0

        model.set({f"{MONITOR_ELEMENT}.noise_x": 1.0e9})
        assert model.get(f"{MONITOR_ELEMENT}.noise_x") == pytest.approx(1.0e9)

    def test_a_polarity_flip_is_accepted(self, model: PyATRingModel) -> None:
        model.set({f"{MONITOR_ELEMENT}.polarity_x": -1.0})

        assert model.get(f"{MONITOR_ELEMENT}.polarity_x") == -1.0

    def test_a_dimensioned_fault_states_the_unit_of_what_it_perturbs(
        self, booted: PyATRingModel
    ) -> None:
        """Not a unit this package chose: the monitor publishes millimetres
        and the magnet is commanded in amps, so the fault says so too."""
        variables = booted.supported_variables

        assert variables[f"{MONITOR_ELEMENT}.offset_x"].unit == "mm"
        assert variables[f"{MONITOR_ELEMENT}.noise_x"].unit == "mm"
        assert variables[f"{QUAD_ELEMENT}.cal_offset"].unit == "A"
        assert variables[f"{MONITOR_ELEMENT}.gain_x"].unit is None

    def test_a_seed_naming_no_monitor_is_refused_before_the_ring_is_touched(
        self, data_dir: Path
    ) -> None:
        with pytest.raises(UnknownDeviceError, match="QF1"):
            PyATRingModel(data_dir, _manifest(), bpm_errors={QUAD_ELEMENT: {"offset_x": 0.1}})

    def test_a_seed_naming_no_magnet_is_refused(self, data_dir: Path) -> None:
        with pytest.raises(UnknownDeviceError, match=MONITOR_ELEMENT):
            PyATRingModel(data_dir, _manifest(), corrector_gains={MONITOR_ELEMENT: {"factor": 2.0}})

    def test_a_seed_naming_an_unknown_field_is_refused(self, data_dir: Path) -> None:
        with pytest.raises(ValueError, match="offset_z"):
            PyATRingModel(data_dir, _manifest(), bpm_errors={MONITOR_ELEMENT: {"offset_z": 0.1}})

    def test_a_seed_outside_its_bound_is_refused(self, data_dir: Path) -> None:
        with pytest.raises(ValueError, match="gain_x"):
            PyATRingModel(data_dir, _manifest(), bpm_errors={MONITOR_ELEMENT: {"gain_x": 50.0}})

    def test_two_setpoints_on_one_element_disagreeing_on_a_unit_refuse_the_boot(
        self, tmp_path: Path
    ) -> None:
        """One element carries one calibration, and its offset shifts whatever
        arrives on any address bound to it. Two setpoints commanded in
        different units leave that magnitude undefined, so the tree is refused
        rather than served with an offset that means one of the two."""
        second = "R1:PWR:QUAD_B:07:CUR:SP"
        # A second setpoint on the quad's element, writing a different
        # component so the document's own field check admits it.
        bindings = [
            _strength(),
            _kick(),
            _monitor(),
            _rf(),
            _energy(),
            _strength(
                family="quad_b",
                setpoint_address=second,
                readback_address=None,
                readback="same_as_setpoint",
                attribute="PolynomA",
                index=1,
                owner="quad_b",
                monitor_inverse=None,
                nominal=0.0,
            ),
        ]
        machine = {**_machine(), second: {"value": 0.0, "units": "mA"}}
        data_dir = _tree(tmp_path / "data", bindings=bindings, machine=machine, limits=_limits())
        manifest = [*_manifest(), _channel(second)]

        with pytest.raises(ValueError, match="one calibration"):
            PyATRingModel(data_dir, manifest)

    def test_two_setpoints_on_one_element_agreeing_on_a_unit_boot(self, tmp_path: Path) -> None:
        """The case the refusal above has to leave alone: one calibration
        serves both, and its offset is in the unit they share."""
        second = "R1:PWR:QUAD_B:07:CUR:SP"
        bindings = [
            _strength(),
            _kick(),
            _monitor(),
            _rf(),
            _energy(),
            _strength(
                family="quad_b",
                setpoint_address=second,
                readback_address=None,
                readback="same_as_setpoint",
                attribute="PolynomA",
                index=1,
                owner="quad_b",
                monitor_inverse=None,
                nominal=0.0,
            ),
        ]
        machine = {**_machine(), second: {"value": 0.0, "units": "A"}}
        data_dir = _tree(tmp_path / "data", bindings=bindings, machine=machine, limits=_limits())

        booted = PyATRingModel(data_dir, [*_manifest(), _channel(second)])

        assert booted.supported_variables[f"{QUAD_ELEMENT}.cal_offset"].unit == "A"

    def test_a_served_address_spelled_like_a_fault_refuses_the_boot(self, data_dir: Path) -> None:
        """The served / model-only line is drawn by name alone, so a namespace
        that already carries a fault's name would have it served -- and a
        client would write physics state through a channel."""
        collision = f"{MONITOR_ELEMENT}.offset_x"
        manifest = [
            *_manifest(),
            {
                "address": collision,
                "ring": "R1",
                "system": "VAC",
                "family": "GAU_A",
                "device": "02",
                "field": "PRES",
                "subfield": "RB",
                "partition": PARTITION_STATIC_NOISY,
                "record_type": "ai",
                "noise": False,
            },
        ]

        with pytest.raises(ValueError, match=collision):
            PyATRingModel(data_dir, manifest)


class TestTheOpticsArrays:
    """Quantities of the whole solved ring, computed when one is read."""

    def test_the_tunes_are_a_pair(self, booted: PyATRingModel) -> None:
        tunes = booted.get("tunes")

        assert tunes.shape == (2,)

    def test_a_per_monitor_array_has_one_row_per_monitor(self, booted: PyATRingModel) -> None:
        assert booted.get("beta_at_monitors").shape == (1, 2)
        assert booted.get("orbit_at_monitors").shape == (1, 2)

    def test_the_orbit_array_is_the_true_position(self, booted: PyATRingModel) -> None:
        """Metres on the ring; the reading variable publishes millimetres."""
        orbit = booted.get("orbit_at_monitors")

        assert float(orbit[0][0]) == pytest.approx(booted.get(BPM_X) * 1.0e-3, abs=1.0e-12)

    def test_a_returned_array_is_a_copy(self, booted: PyATRingModel) -> None:
        first = booted.get("tunes")
        first[0] = 99.0

        assert booted.get("tunes")[0] != 99.0

    def test_the_arrays_move_with_the_ring(self, model: PyATRingModel) -> None:
        before = model.get("tunes").copy()
        model.set({QUAD_SP: QUAD_NOMINAL_AMPS * 1.05})

        assert model.get("tunes")[0] != pytest.approx(before[0])

    def test_an_optics_array_is_not_read_back_after_a_solve(self, model: PyATRingModel) -> None:
        """The hot path pays nothing for them: a write refreshes the readings
        and leaves the arrays to the next read of one."""
        outputs = model._read_outputs()

        assert OPTICS_NAMES.isdisjoint(outputs)


class TestTheModelReachesNoOtherTree:
    """Ruling: nothing reads the packaged demo tree by default."""

    def test_no_packaged_source_is_named(self) -> None:
        from osprey.services.virtual_accelerator.model import pyat

        source = Path(pyat.__file__).read_text()

        for forbidden in ("PACKAGE_PATHS", "build_manifest", "osprey.templates"):
            assert forbidden not in source

    def test_the_ring_is_never_built_without_a_tree_to_build_it_from(self) -> None:
        """The data directory is positional and required: there is no boot
        that resolves a lattice on its own."""
        with pytest.raises(TypeError):
            PyATRingModel()  # type: ignore[call-arg]
