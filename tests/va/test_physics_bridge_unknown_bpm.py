"""What the fault lookup resolves through the bindings, and what it refuses.

Two things, both about the lookup rather than about any physics:

* the refusal for a seeded monitor or magnet name the served tree has none
  of. A seed naming a device the document binds nothing at would perturb
  nothing at all, and on the live stand-in (whose whole difference from the
  sandbox VA is a shipped monitor offset) that silent drop would make the two
  targets identical while looking configured -- so the model refuses the
  seed, naming every offender, before the ring is touched.
* that neither the model nor the bridge reads a facility fact out of an
  address. Which addresses
  carry a monitor reading, which element each sits at, which transverse axis
  it reads and which element a setpoint drives all come from the model's
  variable catalog -- the served ``va_bindings.json`` resolved by address.

The served tree below is built here and is deliberately hostile to the
grammar the bridge used to assume: its monitor addresses spell their axes
``HORIZ``/``VERT`` under a family ``PSM_A`` while the elements they read are
called ``MON1``..``MON3``, and its corrector setpoint is a ``CUR`` field of a
``PWR`` system. A bridge that rebuilt a device name out of address levels, or
filtered records by an ``X``/``Y`` suffix, would serve nothing here.

The conversion from the solved metres to the unit a monitor publishes is the
*variable's*, applied inside the model (``MonitorVariable``, pinned in
test_pyat_ring_model.py). This module pins the consequence for the readout
model: the bridge perturbs the value the model published, so a seeded offset
is in that published unit.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import at
import pytest

from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_STATIC_NOISY,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel, UnknownDeviceError

C_LIGHT = 299792458.0

#: The deck this fixture tree was exported at, in GeV.
DECK_ENERGY_GEV = 3.0

#: Cells, focusing strength and the harmonic number of the closing cavity.
CELLS = 3
QUAD_K = 1.1
HARMONIC = 32

#: Radians of kick per amp of corrector current.
KICK_GAIN = 1.0e-5

#: Millimetres per metre: the monitor's physics-to-hardware curve, which is
#: where the unit step lives. A reading is published -- and perturbed -- in
#: millimetres because this curve says so, not because anything scales it.
MM_PER_M = 1.0e3

# The addresses the fixture document binds. Nothing relates them to the
# element names below; the bindings document does that and only it.
CORR_SP = "R1:PWR:CORR_A:03:CUR:SP"
CAVITY_SP = "R1:RF:CAV_A:01:FREQ:SP"
STATIC_RB = "R1:VAC:GAU_A:01:PRES:RB"

#: The deck's own name for the element the corrector setpoint drives, and the
#: names of the three monitors -- neither derivable from the addresses above.
CORR_ELEMENT = "HC1"
MONITOR_ELEMENTS = ("MON1", "MON2", "MON3")
CAVITY_ELEMENT = "RFC"


def monitor_address(element: str, axis: str) -> str:
    """The address the fixture document publishes one monitor axis on.

    Spelled the way a facility that never heard of this code base might: the
    device number counts monitors, the family is the facility's own, and the
    axis subfield is a word rather than a letter. The element the reading
    comes from appears nowhere in it.
    """
    device = MONITOR_ELEMENTS.index(element) + 1
    return f"R1:DIA:PSM_A:{device:02d}:POS:{'HORIZ' if axis == 'x' else 'VERT'}"


def _ring() -> at.Lattice:
    """A small stable FODO ring with uniquely named monitors and correctors."""
    angle = 2 * math.pi / (2 * CELLS)
    elements: list[Any] = []
    for cell in range(1, CELLS + 1):
        elements += [
            at.Quadrupole(f"QF{cell}", 0.3, QUAD_K),
            at.Drift("DR", 1.0),
            at.Dipole(f"BD{cell}A", 1.0, angle),
            at.Monitor(MONITOR_ELEMENTS[cell - 1]),
            at.Drift("DR", 1.0),
            at.Quadrupole(f"QD{cell}", 0.3, -QUAD_K),
            at.Drift("DR", 1.0),
            at.Corrector(f"HC{cell}", 0.0, [0.0, 0.0]),
            at.Dipole(f"BD{cell}B", 1.0, angle),
            at.Drift("DR", 1.0),
        ]
    ring = at.Lattice(elements, name="fixture", energy=DECK_ENERGY_GEV * 1.0e9, periodicity=1)
    ring.append(
        at.RFCavity(
            CAVITY_ELEMENT,
            0.0,
            1.0e6,
            HARMONIC * C_LIGHT / ring.circumference,
            HARMONIC,
            ring.energy,
        )
    )
    ring.disable_6d()
    return ring


#: The cavity frequency the fixture ring is built at, in MHz.
CAVITY_NOMINAL_MHZ = HARMONIC * C_LIGHT / _ring().circumference / 1.0e6


def _linear(gain: float, offset: float = 0.0) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _kick() -> dict:
    """The corrector setpoint, onto one element's ``KickAngle[0]``."""
    return {
        "kind": "kick",
        "family": "corr_a",
        "setpoint_address": CORR_SP,
        "readback_address": CORR_SP.replace(":SP", ":RB"),
        "readback": "identity",
        "element": CORR_ELEMENT,
        "attribute": "KickAngle",
        "index": 0,
        "slices": [{"element": CORR_ELEMENT, "weight": 1.0}],
        "owner": "corr_a",
        "calibration": _linear(KICK_GAIN),
        "monitor_inverse": None,
        "nominal": 0.0,
        "energy_scaling": "brho",
        "energy_table": None,
    }


def _monitor(element: str, axis: str) -> dict:
    """One orbit reading: metres on the ring, millimetres on the wire."""
    return {
        "kind": "monitor",
        "family": "psm_a",
        "setpoint_address": monitor_address(element, axis),
        "readback_address": None,
        "readback": "inverse",
        "element": element,
        "attribute": axis,
        "index": None,
        "slices": [{"element": element, "weight": 1.0}],
        "owner": "psm_a",
        "calibration": _linear(1.0 / MM_PER_M),
        "monitor_inverse": _linear(MM_PER_M),
        "nominal": None,
        "energy_scaling": "none",
        "energy_table": None,
    }


def _rf() -> dict:
    """The cavity frequency, written in MHz -- the ring's one other writable."""
    return {
        "kind": "rf",
        "family": "cav_a",
        "setpoint_address": CAVITY_SP,
        "readback_address": None,
        "readback": "same_as_setpoint",
        "element": CAVITY_ELEMENT,
        "attribute": "Frequency",
        "index": None,
        "slices": [{"element": CAVITY_ELEMENT, "weight": 1.0}],
        "owner": "cav_a",
        "calibration": _linear(1.0e6),
        "monitor_inverse": None,
        "nominal": CAVITY_NOMINAL_MHZ,
        "energy_scaling": "none",
        "energy_table": None,
    }


def _bindings(axes: tuple[str, ...] = ("x", "y")) -> list[dict]:
    """The document: one corrector, one cavity, and ``axes`` per monitor."""
    return [
        _kick(),
        _rf(),
        *(_monitor(element, axis) for element in MONITOR_ELEMENTS for axis in axes),
    ]


def _channel(address: str, *, partition: str = PARTITION_PYAT_COUPLED) -> dict:
    """One manifest channel, in the per-channel schema the manifest carries."""
    ring, system, family, device, field, subfield = address.split(":")
    return {
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


def _manifest(axes: tuple[str, ...] = ("x", "y")) -> list[dict]:
    """The channel list a deployment of this tree resolved."""
    return [
        _channel(CORR_SP),
        _channel(CAVITY_SP),
        *(
            _channel(monitor_address(element, axis))
            for element in MONITOR_ELEMENTS
            for axis in axes
        ),
        _channel(STATIC_RB, partition=PARTITION_STATIC_NOISY),
    ]


def _tree(root: Path, *, axes: tuple[str, ...] = ("x", "y")) -> Path:
    """Write a served tree and return the data directory addressing it."""
    paths = ManifestPaths(data_root=root)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(_ring(), paths.lattice_json)
    document = {
        "system": "StorageRing",
        "energy_gev": DECK_ENERGY_GEV,
        "lattice_sha256": hashlib.sha256(paths.lattice_json.read_bytes()).hexdigest(),
        "bindings": _bindings(axes),
    }
    paths.va_bindings.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    channels: dict[str, dict[str, Any]] = {
        CORR_SP: {"value": 0.0, "units": "A"},
        CAVITY_SP: {"value": CAVITY_NOMINAL_MHZ, "units": "MHz"},
    }
    for element in MONITOR_ELEMENTS:
        for axis in axes:
            channels[monitor_address(element, axis)] = {"value": 0.0, "units": "mm"}
    paths.machine_json.write_text(json.dumps({"name": "fixture", "channels": channels}))
    paths.channel_limits.write_text(
        json.dumps(
            {
                "_version": "1.0",
                "defaults": {"writable": True, "confirm": True},
                CORR_SP: {"min_value": -10.0, "max_value": 10.0},
                CAVITY_SP: {
                    "min_value": CAVITY_NOMINAL_MHZ - 1.0,
                    "max_value": CAVITY_NOMINAL_MHZ + 1.0,
                },
            }
        )
    )
    return root


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc In record: just `.set()`."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A served tree whose files all agree -- the boot case."""
    return _tree(tmp_path_factory.mktemp("served") / "data")


@pytest.fixture
def model(data_dir: Path) -> PyATRingModel:
    """The backend the bridges under test serve.

    Function-scoped, so a test that writes a setpoint cannot leave the
    lattice off-nominal for the next one.
    """
    return PyATRingModel(data_dir, _manifest())


class TestASeedNamingNoSuchDeviceIsRefused:
    """FR10: a fault seed the tree has no device for ends the boot.

    Refused where the faults live -- in the model, which declares one per
    served device -- and before the ring is touched, so a machine never comes
    up serving unperturbed readings while looking configured.
    """

    def test_a_monitor_the_tree_has_none_of_names_itself(self, data_dir):
        with pytest.raises(UnknownDeviceError) as excinfo:
            PyATRingModel(data_dir, _manifest(), bpm_errors={"MON9": {"offset_x": 0.1}})

        assert "MON9" in str(excinfo.value)

    def test_a_monitor_the_tree_carries_is_accepted(self, data_dir):
        model = PyATRingModel(data_dir, _manifest(), bpm_errors={"MON1": {"offset_x": 0.1}})

        assert model.get("MON1.offset_x") == pytest.approx(0.1)

    def test_two_unknown_names_are_named_together(self, data_dir):
        """One refusal, both offenders, sorted -- so a run of typos is fixed
        in one pass instead of one boot each."""
        with pytest.raises(UnknownDeviceError) as excinfo:
            PyATRingModel(
                data_dir,
                _manifest(),
                bpm_errors={"MON8": {"gain_x": 1.5}, "MON9": {"offset_x": 0.1}},
            )

        assert "MON8" in str(excinfo.value)
        assert "MON9" in str(excinfo.value)

    def test_a_name_reassembled_from_the_address_is_one_of_the_unknown_ones(self, data_dir):
        """The document's element name is the key, and nothing else is.

        ``PSM_A01`` is what a family-plus-device reconstruction of this
        monitor's address would produce, and the tree binds no monitor by that
        name -- so it is refused, exactly like a typo.
        """
        with pytest.raises(UnknownDeviceError) as excinfo:
            PyATRingModel(data_dir, _manifest(), bpm_errors={"PSM_A01": {"offset_x": 0.1}})

        assert "PSM_A01" in str(excinfo.value)

    def test_a_known_name_beside_an_unknown_one_does_not_rescue_the_boot(self, data_dir):
        """The refusal is not a fallback: a mixed seed is still refused."""
        with pytest.raises(UnknownDeviceError):
            PyATRingModel(
                data_dir,
                _manifest(),
                bpm_errors={"MON1": {"offset_x": 0.05}, "MON9": {"offset_x": 0.1}},
            )


class TestResolvesThroughTheBindings:
    """Every device fact the bridge uses comes from the model's variables."""

    def test_the_served_readings_are_the_documents_monitor_addresses(self, model):
        """No address grammar: these carry neither an ``X``/``Y`` subfield nor
        a family the bridge could have recognised."""
        bridge = PhysicsBridge(model)
        assert set(bridge.bpm_positions()) == {
            monitor_address(element, axis) for element in MONITOR_ELEMENTS for axis in ("x", "y")
        }

    def test_only_the_monitor_records_are_bound(self, model):
        """A coupled record that is not a monitor reading is left alone -- the
        write path drives the setpoints, not this bridge."""
        monitor = FakeRecord()
        setpoint = FakeRecord()
        bridge = PhysicsBridge(model)
        bridge.bind({monitor_address("MON2", "y"): monitor, CORR_SP: setpoint})

        assert monitor.value is not None
        assert setpoint.value is None

    def test_the_bridge_serves_the_value_the_model_published(self, model):
        """It rescales nothing: the unit conversion is the variable's.

        Which is what puts a seeded offset in the monitor's own unit -- the
        readout model perturbs exactly what the model handed over.
        """
        bridge = PhysicsBridge(model)
        bridge.on_setpoint(CORR_SP, 3.0)
        addresses = sorted(bridge.bpm_positions())

        assert bridge.bpm_positions() == pytest.approx(dict(model.get(addresses)))

    def test_a_seeded_gain_multiplies_the_published_reading(self, data_dir):
        model = PyATRingModel(data_dir, _manifest(), bpm_errors={"MON1": {"gain_x": 2.0}})
        bridge = PhysicsBridge(model)
        record = FakeRecord()
        address = monitor_address("MON1", "x")
        bridge.bind({address: record})
        bridge.on_setpoint(CORR_SP, 4.0)

        assert record.value == pytest.approx(2.0 * bridge.bpm_positions()[address], abs=1e-12)

    def test_a_corrector_gain_is_keyed_by_the_bound_element(self, data_dir):
        """``VA_CORR_GAIN`` names ``HC1``, the element, not the address.

        A doubled gain on that element must make a half-sized write land
        exactly where the full one did unseeded -- the seeded fault acts on
        the commanded hardware value, and the binding's own calibration then
        converts it inside the model.
        """
        plain = PhysicsBridge(PyATRingModel(data_dir, _manifest()))
        plain.on_setpoint(CORR_SP, 1.0)

        seeded = PhysicsBridge(
            PyATRingModel(data_dir, _manifest(), corrector_gains={CORR_ELEMENT: {"factor": 2.0}})
        )
        seeded.on_setpoint(CORR_SP, 0.5)

        assert seeded.bpm_positions() == pytest.approx(plain.bpm_positions())

    def test_a_corrector_gain_keyed_by_an_address_derived_name_is_refused(self, data_dir):
        """The reassembled ``CORR_A03`` is not the element, so it names no
        magnet this tree drives and the boot says so."""
        with pytest.raises(UnknownDeviceError) as excinfo:
            PyATRingModel(data_dir, _manifest(), corrector_gains={"CORR_A03": {"factor": 2.0}})

        assert "CORR_A03" in str(excinfo.value)

    def test_a_setpoint_the_model_does_not_drive_is_refused(self, model):
        """The static-noisy channel is served, but no binding drives it."""
        bridge = PhysicsBridge(model)
        with pytest.raises(UnknownDeviceError, match="binds no writable variable"):
            bridge.on_setpoint(STATIC_RB, 1.0)

    def test_a_monitor_address_is_not_a_setpoint(self, model):
        """A reading is read-only however its address is spelled."""
        bridge = PhysicsBridge(model)
        with pytest.raises(UnknownDeviceError, match="binds no writable variable"):
            bridge.on_setpoint(monitor_address("MON1", "x"), 1.0)

    def test_the_rf_setpoint_is_driven_like_any_other_writable(self, model):
        """No field or system check: what the document binds, the bridge drives.

        The old path refused anything that was not a ``CURRENT`` field of a
        ``MAG`` system, which would have refused this cavity outright.
        """
        # A ten-hertz step: enough to see on the element, small enough that
        # the ring still has a stable longitudinal plane to solve for.
        written = CAVITY_NOMINAL_MHZ - 1.0e-5
        bridge = PhysicsBridge(model)
        bridge.on_setpoint(CAVITY_SP, written)

        cavity = model.lattice[model.element_index(CAVITY_ELEMENT)]
        assert cavity.Frequency == pytest.approx(written * 1.0e6, abs=1.0e-3)


class TestAMonitorBoundOnOnePlane:
    """A document binding one axis serves that axis, and only it."""

    @pytest.fixture(scope="class")
    def horizontal_only(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        return _tree(tmp_path_factory.mktemp("horizontal") / "data", axes=("x",))

    def test_the_bound_plane_is_served_and_the_other_is_not(self, horizontal_only):
        model = PyATRingModel(
            horizontal_only, _manifest(axes=("x",)), bpm_errors={"MON1": {"offset_x": 0.05}}
        )
        record = FakeRecord()
        bridge = PhysicsBridge(model)
        bridge.bind({monitor_address("MON1", "x"): record})
        bridge.on_setpoint(CORR_SP, 5.0)

        address = monitor_address("MON1", "x")
        assert set(bridge.bpm_positions()) == {
            monitor_address(element, "x") for element in MONITOR_ELEMENTS
        }
        assert record.value == pytest.approx(bridge.bpm_positions()[address] - 0.05, abs=1e-12)


class _FakeVariable:
    """A read-only variable with whatever binding facts a test gives it."""

    read_only = True

    def __init__(self, **facts: Any) -> None:
        for name, value in facts.items():
            setattr(self, name, value)


class _FakeModel:
    """The narrowest `LUMEModel` surface the bridge touches."""

    def __init__(self, variables: dict[str, Any]) -> None:
        self.supported_variables = variables

    def get(self, names: list[str]) -> dict[str, float]:
        return dict.fromkeys(names, 0.0)


class TestAModelThisBridgeCannotServe:
    """A reading with no monitor behind it is refused, not dropped."""

    def test_a_read_only_variable_with_no_axis_is_refused(self):
        model = _FakeModel({"A:READING": _FakeVariable(element_name="MON1")})
        with pytest.raises(ValueError, match="names the element it sits at"):
            PhysicsBridge(model)

    def test_a_read_only_variable_with_no_element_is_refused(self):
        model = _FakeModel({"A:READING": _FakeVariable(axis="x")})
        with pytest.raises(ValueError, match="names the element it sits at"):
            PhysicsBridge(model)

    def test_two_readings_of_one_axis_at_one_element_are_refused(self):
        model = _FakeModel(
            {
                "A:READING": _FakeVariable(element_name="MON1", axis="x"),
                "B:READING": _FakeVariable(element_name="MON1", axis="x"),
            }
        )
        with pytest.raises(ValueError, match="could only be dropped"):
            PhysicsBridge(model)
