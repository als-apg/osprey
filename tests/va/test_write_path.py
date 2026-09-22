"""What a co-hosted setpoint write means, as the served tree states it.

Two things decide it, and neither is the address text. Which co-hosted
channels reach the lattice is the manifest's ``pyat-coupled`` partition and
its ``subfield`` vocabulary, and what a write *does* -- which element, through
which calibration, and what its readback is then worth -- is the bindings
document. A facility spelling its setpoints ``..._SP``, ``...:CTRL`` or
anything else has the same setpoints as one spelling them ``...:SP``, and a
facility whose readback is a sampled curve rather than an echo has the same
write path as one whose readback is an echo.

The first class below pins the manifest half in pure python. The rest run
against a synthetic emitted tree -- a small stable ring, a bindings document
carrying one binding of every kind, and the ``machine.json`` and
``channel_limits.json`` a served tree ships -- driven through a real
:class:`~osprey.services.virtual_accelerator.model.pyat.PyATRingModel`. The
readback a client reads is the model variable's own conversion, so nothing
short of the real variables would pin it.

No address here carries a facility's vocabulary: the bindings document is the
only thing that pairs an address with an element.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import at
import pytest

from osprey.services.virtual_accelerator.bindings import (
    READBACK_RULES,
    load_bindings,
)
from osprey.services.virtual_accelerator.lattice.calibration import energy_factor
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
)
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
from osprey.services.virtual_accelerator.serving.pvdb import (
    ServingRecords,
    build_serving_pvdb,
)
from osprey.services.virtual_accelerator.serving.write_path import (
    MODE_ECHO,
    MODE_PHYSICS,
    READBACK_IDENTITY,
    READBACK_INVERSE,
    READBACK_SAME_AS_SETPOINT,
    BoundSetpoint,
    CohostWritePath,
    SetpointRoutedModel,
    bound_setpoints,
    physics_setpoint_addresses,
)

# -- the manifest half: pure python, no lattice ------------------------------

#: A facility that separates its address levels with ``_``, not ``:``.
MAGNET_SP = "ZZEXP_MAG_Q1_CURRENT_SP"
MAGNET_RB = "ZZEXP_MAG_Q1_CURRENT_RB"
VALVE_SP = "ZZEXP_VAC_V1_POSITION_SP"
VALVE_RB = "ZZEXP_VAC_V1_POSITION_RB"


def _flat_channel(address: str, *, subfield: str, partition: str, field: str, family: str) -> dict:
    return {
        "address": address,
        "ring": "ZZEXP",
        "system": "MAG",
        "family": family,
        "device": "Q1",
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": "ai",
        "noise": False,
    }


@pytest.fixture
def records():
    """A manifest with no ``:`` in any address and a normal SP/RB vocabulary."""
    return build_serving_pvdb(
        [
            _flat_channel(
                MAGNET_SP,
                subfield="SP",
                partition=PARTITION_PYAT_COUPLED,
                field="CURRENT",
                family="QUAD",
            ),
            _flat_channel(
                MAGNET_RB,
                subfield="RB",
                partition=PARTITION_PYAT_COUPLED,
                field="CURRENT",
                family="QUAD",
            ),
            _flat_channel(
                VALVE_SP,
                subfield="SP",
                partition=PARTITION_SP_ECHO,
                field="POSITION",
                family="VALVE",
            ),
            _flat_channel(
                VALVE_RB,
                subfield="RB",
                partition=PARTITION_SP_ECHO,
                field="POSITION",
                family="VALVE",
            ),
        ],
        async_setpoints=True,
    )


class TestSetpointsComeFromTheManifestNotTheAddress:
    def test_the_served_database_states_its_physics_setpoints(self, records):
        assert records.physics_setpoints == {MAGNET_SP}

    def test_the_write_path_reads_that_set(self, records):
        assert physics_setpoint_addresses(records) == {MAGNET_SP}

    def test_a_setpoint_not_spelled_sp_still_routes_through_physics(self, records):
        path = CohostWritePath(
            records,
            enqueue=lambda values, done=None, reset=False: None,
            physics_setpoints=physics_setpoint_addresses(records),
        )

        assert path.routes[MAGNET_SP].mode == MODE_PHYSICS
        assert path.routes[MAGNET_SP].readback == MAGNET_RB

    def test_an_echo_setpoint_is_still_only_an_echo(self, records):
        path = CohostWritePath(
            records,
            enqueue=lambda values, done=None, reset=False: None,
            physics_setpoints=physics_setpoint_addresses(records),
        )

        assert path.routes[VALVE_SP].mode == MODE_ECHO
        assert path.routes[VALVE_SP].readback == VALVE_RB

    def test_a_pyat_coupled_readback_is_not_a_setpoint(self, records):
        assert MAGNET_RB not in records.physics_setpoints
        assert MAGNET_RB not in records.setpoint_readbacks


# -- the synthetic emitted tree ----------------------------------------------

C_LIGHT = 299792458.0

#: The deck this fixture tree was exported at, in GeV.
DECK_ENERGY_GEV = 3.0

#: The focusing strength the ring is built with, and the hardware nominal the
#: calibration below maps onto it, so the tree boots in its own nominal state.
QUAD_K = 1.1
QUAD_GAIN = 0.01
QUAD_NOMINAL_AMPS = QUAD_K / QUAD_GAIN

#: The quadrupole's way back from physics to hardware. Deliberately *not*
#: ``1 / QUAD_GAIN``: the facility samples the two directions independently,
#: and a fixture whose inverse happens to undo its calibration exactly could
#: not tell an ``inverse`` readback from an ``identity`` one.
QUAD_INVERSE_GAIN = 95.0

#: What one amp on the quadrupole reads back as, through both curves.
QUAD_READBACK_RATIO = QUAD_GAIN * QUAD_INVERSE_GAIN

#: Radians of kick per amp, and the ring's cavity harmonic number.
KICK_GAIN = 1.0e-6
HARMONIC = 88

# The addresses the fixture document binds. Nothing relates them to the
# element names below; the bindings document does that and only it.
QUAD_SP = "R1:PWR:QUAD_A:07:CUR:SP"
QUAD_RB = "R1:PWR:QUAD_A:07:CUR:RB"
CORR_SP = "R1:PWR:CORR_A:03:CUR:SP"
CORR_RB = "R1:PWR:CORR_A:03:CUR:RB"
CAVITY_SP = "R1:RF:CAV_A:01:FREQ:SP"
BPM_X = "R1:DIA:MON_A:12:POS:X"
BEND_SP = "R1:PWR:BEND_A:01:CUR:SP"
BEND_RB = "R1:PWR:BEND_A:01:CUR:RB"
STATIC_RB = "R1:VAC:GAU_A:01:PRES:RB"

# The deck's own names for the elements each address drives.
QUAD_ELEMENT = "QF1"
CORR_ELEMENTS = ("HC1", "HC2")
MONITOR_ELEMENT = "BPM1"
CAVITY_ELEMENT = "RFC"

#: The write bands the served tree ships, as the write path clamps into them.
DRIVE_LIMITS: dict[str, tuple[float, float]] = {
    QUAD_SP: (90.0, 130.0),
    CORR_SP: (-10.0, 10.0),
    CAVITY_SP: (499.0, 500.0),
    BEND_SP: (250.0, 350.0),
}


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
    """A quadrupole setpoint whose readback comes back through the inverse."""
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
        "monitor_inverse": _linear(QUAD_INVERSE_GAIN),
        "nominal": QUAD_NOMINAL_AMPS,
        "energy_scaling": "brho",
        "energy_table": None,
    }
    body.update(overrides)
    return body


def _kick(**overrides: Any) -> dict:
    """A corrector shared over two pieces, reading back what was written."""
    body = _strength(
        kind="kick",
        family="corr_a",
        setpoint_address=CORR_SP,
        readback_address=CORR_RB,
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
    """The cavity frequency: one address carrying setpoint and readback."""
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
        readback_address=BEND_RB,
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


def _manifest(addresses: tuple[str, ...] | None = None) -> list[dict]:
    """The channel list a deployment of this tree resolved.

    Order matters in one place: the catalog is built in manifest order, so
    this is the order the energy knob adopts the setpoints it rescales in.
    """
    served = (
        (QUAD_SP, QUAD_RB, CORR_SP, CORR_RB, CAVITY_SP, BPM_X, BEND_SP, BEND_RB)
        if addresses is None
        else addresses
    )
    channels = [_channel(address) for address in served]
    channels.append(_channel(STATIC_RB, partition=PARTITION_STATIC_NOISY))
    return channels


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
    body: dict[str, Any] = {"_version": "1.0", "defaults": {"writable": True, "confirm": True}}
    for address, (low, high) in DRIVE_LIMITS.items():
        body[address] = {"min_value": low, "max_value": high}
    return body


def _tree(root: Path, *, bindings: list[dict] | None = None) -> Path:
    """Write a served tree and return the data directory addressing it."""
    paths = ManifestPaths(data_root=root)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(_ring(), paths.lattice_json)
    document = {
        "system": "StorageRing",
        "energy_gev": DECK_ENERGY_GEV,
        "lattice_sha256": hashlib.sha256(paths.lattice_json.read_bytes()).hexdigest(),
        "bindings": [_strength(), _kick(), _monitor(), _rf(), _energy()]
        if bindings is None
        else bindings,
    }
    paths.va_bindings.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    paths.machine_json.write_text(json.dumps({"name": "fixture", "channels": _machine()}))
    paths.channel_limits.write_text(json.dumps(_limits()))
    return root


# -- the transport and the loop, reduced to what a write needs ---------------


class FakeDriver:
    """The driver surface, recording instead of serving."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.posted: list[str] = []
        self.completed: list[str] = []
        self.alarms: list[tuple[str, Any, Any]] = []

    def setParam(self, reason: str, value: Any) -> None:  # noqa: N802 - driver contract
        self.values[reason] = value

    def getParam(self, reason: str) -> Any:  # noqa: N802 - driver contract
        return self.values[reason]

    def updatePV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.posted.append(reason)

    def callbackPV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.completed.append(reason)

    def setParamStatus(  # noqa: N802 - driver contract
        self, reason: str, alarm: Any, severity: Any
    ) -> None:
        self.alarms.append((reason, alarm, severity))


class FakeRunLoop:
    """The run loop, for one queued item at a time.

    The runner disables batching, so one item is one cycle is one
    ``model.set`` -- which is also what makes a routed write one lume-pyat
    batch, and therefore one solve that rolls back as a whole. The values of
    every item are kept so a test can prove what the model was offered.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.queue: list[tuple[dict[str, Any], Any]] = []
        self.offered: list[dict[str, Any]] = []

    def enqueue(self, values: dict[str, Any], done: Any = None, _reset: bool = False) -> None:
        self.queue.append((values, done))

    def drain(self) -> None:
        while self.queue:
            values, done = self.queue.pop(0)
            plain = {name: item["value"] for name, item in values.items()}
            self.offered.append(plain)
            error = None
            try:
                self.model.set(plain)
            except Exception as exc:  # noqa: BLE001 - the loop reports, never raises
                error = str(exc)
            if done is not None:
                done(error)


class Served:
    """One booted tree, its serving database and its write path."""

    def __init__(self, data_dir: Path, *, manifest: list[dict], limits: bool = True) -> None:
        self.document = load_bindings(ManifestPaths(data_root=data_dir).va_bindings)
        self.model = PyATRingModel(data_dir, manifest)
        self.records: ServingRecords = build_serving_pvdb(manifest, async_setpoints=True)
        self.bound = bound_setpoints(self.document, self.model.supported_variables)
        self.applied: list[tuple[str, float]] = []
        self.loop = FakeRunLoop(
            SetpointRoutedModel(
                self.model,
                on_setpoint=self._on_setpoint,
                routed=frozenset(self.bound),
            )
        )
        self.path = CohostWritePath(
            self.records,
            enqueue=self.loop.enqueue,
            bound_setpoints=self.bound,
            drive_limits=DRIVE_LIMITS if limits else None,
            refusal_alarm=("WRITE", "MINOR"),
        )
        self.driver = FakeDriver()

    def _on_setpoint(self, address: str, value: float) -> None:
        """The physics hook, as the bridge implements it: apply, then record.

        No calibration of its own -- the commanded hardware value goes to the
        model, and the variable the bindings document built is what converts
        it.
        """
        self.applied.append((address, value))
        self.model.set({address: value})

    def write(self, address: str, value: Any) -> bool:
        accepted = self.path.write(self.driver, address, value)
        self.loop.drain()
        return accepted

    def element(self, name: str) -> Any:
        return self.model.lattice[self.model.element_index(name)]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A served tree whose files all agree -- the boot case."""
    return _tree(tmp_path_factory.mktemp("served") / "data")


@pytest.fixture
def served(data_dir: Path) -> Served:
    """A fresh boot per test: every test here writes to the lattice."""
    return Served(data_dir, manifest=_manifest())


# -- which addresses are setpoints -------------------------------------------


class TestTheDocumentSaysWhichAddressesAreWritten:
    """The writable bindings are the setpoints, and nothing else is."""

    def test_one_entry_per_writable_binding_in_document_order(self, served: Served) -> None:
        assert list(served.bound) == [QUAD_SP, CORR_SP, CAVITY_SP, BEND_SP]

    def test_a_monitor_is_not_a_setpoint(self, served: Served) -> None:
        """A reading is read-only whatever its address looks like: the
        monitor's own address sits in the document's ``setpoint_address``, so
        only ``kind`` can tell it apart -- and it does."""
        assert BPM_X not in served.bound
        assert BPM_X not in served.path.routes

    def test_a_write_to_a_monitor_is_refused(self, served: Served) -> None:
        assert served.path.write(served.driver, BPM_X, 1.0) is False
        assert served.driver.values == {}

    def test_the_two_ends_of_the_tree_name_the_same_setpoints(self, served: Served) -> None:
        """The manifest partition is derived from this document at build
        time, so the served database's declaration and the document's own
        writable set are one set reached from either end."""
        assert physics_setpoint_addresses(served.records) == set(served.bound)

    def test_every_bound_setpoint_routes_through_the_model(self, served: Served) -> None:
        modes = {address: served.path.routes[address].mode for address in served.bound}

        assert modes == dict.fromkeys(served.bound, MODE_PHYSICS)

    def test_an_address_the_deployment_does_not_serve_is_not_routed(self, data_dir: Path) -> None:
        """The document describes the facility; the manifest describes this
        deployment. A binding with no served PV has nothing to write to."""
        served = Served(
            data_dir, manifest=_manifest((QUAD_SP, QUAD_RB, CORR_SP, CORR_RB, BPM_X, BEND_SP))
        )

        assert CAVITY_SP in served.bound
        assert CAVITY_SP not in served.path.routes


# -- what the readback carries -----------------------------------------------


class TestTheReadbackIsTheDocumentsOwn:
    """Each rule of the document, served as the document states it."""

    def test_the_route_records_the_rule_each_readback_follows(self, served: Served) -> None:
        rules = {
            address: served.path.routes[address].route.readback_rule for address in served.bound
        }

        assert rules == {
            QUAD_SP: READBACK_INVERSE,
            CORR_SP: READBACK_IDENTITY,
            CAVITY_SP: READBACK_SAME_AS_SETPOINT,
            BEND_SP: READBACK_IDENTITY,
        }

    def test_the_three_rules_are_the_documents_whole_vocabulary(self) -> None:
        """A fourth rule in the schema must not reach the write path as a
        silent echo of the written value."""
        assert {READBACK_IDENTITY, READBACK_INVERSE, READBACK_SAME_AS_SETPOINT} == set(
            READBACK_RULES
        )

    def test_an_inverse_readback_comes_back_through_monitor_inverse(self, served: Served) -> None:
        """Not the written value, and not an inversion of the calibration:
        the facility's own reverse curve, as the model variable applies it."""
        served.write(QUAD_SP, 120.0)

        assert served.driver.values[QUAD_RB] == pytest.approx(120.0 * QUAD_READBACK_RATIO)
        assert served.driver.values[QUAD_RB] != pytest.approx(120.0)

    def test_the_setpoint_itself_still_carries_the_written_value(self, served: Served) -> None:
        """Whatever the readback is worth, the setpoint reads back what was
        commanded -- a setpoint that did not would make every
        read-modify-write client drift."""
        served.write(QUAD_SP, 120.0)

        assert served.driver.values[QUAD_SP] == 120.0

    def test_an_identity_readback_is_the_written_value(self, served: Served) -> None:
        """The corrector has a calibration, and it is not applied here: an
        identity readback carries no inverse, so there is nothing to map
        through and the value is published unchanged."""
        served.write(CORR_SP, 2.0)

        assert served.driver.values[CORR_SP] == 2.0
        assert served.driver.values[CORR_RB] == 2.0

    def test_same_as_setpoint_owes_no_second_address(self, served: Served) -> None:
        """One address carries both, so an accepted write is the whole of
        what is served on it."""
        assert served.path.routes[CAVITY_SP].readback is None

        served.write(CAVITY_SP, 499.7)

        assert served.driver.values == {CAVITY_SP: 499.7}

    def test_the_energy_knob_reads_back_the_value_it_was_written(self, served: Served) -> None:
        """The energy table is exported in one direction only, so the knob
        has no readback conversion at all -- and its rule says it needs
        none."""
        served.write(BEND_SP, 330.0)

        assert served.driver.values[BEND_SP] == 330.0
        assert served.driver.values[BEND_RB] == 330.0

    def test_the_readback_is_computed_from_the_clamped_value(self, served: Served) -> None:
        """Clamp first, then physics, then readback: the value the model took
        is the value the readback was derived from."""
        served.write(QUAD_SP, 900.0)

        assert served.applied == [(QUAD_SP, 130.0)]
        assert served.driver.values[QUAD_SP] == 130.0
        assert served.driver.values[QUAD_RB] == pytest.approx(130.0 * QUAD_READBACK_RATIO)

    def test_the_readback_does_not_move_when_the_ring_energy_does(self, served: Served) -> None:
        """Both curves scale with the rigidity the same way, so the factors
        cancel: a magnet at a fixed current reads back the same at every
        energy, which is what the control system sees on the machine."""
        served.write(QUAD_SP, 120.0)
        at_deck = served.driver.values[QUAD_RB]

        served.write(BEND_SP, 330.0)
        served.write(QUAD_SP, 120.0)

        assert served.model.lattice.energy == pytest.approx(3.3e9, rel=1e-6)
        assert served.driver.values[QUAD_RB] == pytest.approx(at_deck)

    def test_a_readback_address_the_deployment_does_not_serve_is_not_published(
        self, data_dir: Path
    ) -> None:
        """The document names where the facility serves the readback; this
        deployment serves a namespace of its own. The setpoint is published
        either way, so the write is not lost."""
        served = Served(
            data_dir, manifest=_manifest((QUAD_SP, CORR_SP, CORR_RB, CAVITY_SP, BPM_X, BEND_SP))
        )

        assert served.path.routes[QUAD_SP].readback is None

        served.write(QUAD_SP, 120.0)

        assert served.driver.values == {QUAD_SP: 120.0}

    def test_an_inverse_readback_with_no_model_variable_is_refused(self, served: Served) -> None:
        """Serving the written value instead would report a calibration the
        facility does not have, on the one channel whose point is that it
        does."""
        with pytest.raises(ValueError, match=QUAD_SP):
            bound_setpoints(served.document, {})


# -- what the write does to the lattice --------------------------------------


class TestTheWriteGoesThroughTheModelVariable:
    """The hardware value reaches the model; the variable does the physics."""

    def test_the_model_is_offered_the_commanded_hardware_value(self, served: Served) -> None:
        """Not a calibrated one: the conversion belongs to the variable the
        bindings document built, so nothing on the way converts anything."""
        served.write(QUAD_SP, 120.0)

        assert served.loop.offered == [{QUAD_SP: 120.0}]
        assert served.applied == [(QUAD_SP, 120.0)]

    def test_a_strength_lands_on_the_component_the_binding_names(self, served: Served) -> None:
        served.write(QUAD_SP, 120.0)

        assert served.element(QUAD_ELEMENT).PolynomB[1] == pytest.approx(120.0 * QUAD_GAIN)

    def test_a_kick_is_shared_over_every_slice(self, served: Served) -> None:
        served.write(CORR_SP, 2.0)

        for name in CORR_ELEMENTS:
            assert served.element(name).KickAngle[0] == pytest.approx(0.5 * 2.0 * KICK_GAIN)

    def test_the_rf_write_reaches_the_cavity(self, served: Served) -> None:
        served.write(CAVITY_SP, 499.7)

        assert served.element(CAVITY_ELEMENT).Frequency == pytest.approx(499.7e6)

    def test_the_energy_write_moves_the_ring_and_rescales_with_it(self, served: Served) -> None:
        """One address, one write, and the whole rigidity-scaled machine
        follows it -- the knob's own business, reached because the write path
        routes the energy binding like any other setpoint."""
        before = served.element(QUAD_ELEMENT).PolynomB[1]

        served.write(BEND_SP, 330.0)

        factor = energy_factor(served.model.lattice.energy / 1.0e9, DECK_ENERGY_GEV)
        assert served.element(QUAD_ELEMENT).PolynomB[1] == pytest.approx(before * factor, rel=1e-9)
        assert served.model.get(QUAD_SP) == pytest.approx(QUAD_NOMINAL_AMPS)

    def test_one_write_is_one_batch(self, served: Served) -> None:
        """What makes the rollback semantics the model's own: the run loop
        hands each write over on its own, so one client's write is never
        merged into another's solve."""
        served.write(QUAD_SP, 120.0)
        served.write(CORR_SP, 2.0)

        assert served.loop.offered == [{QUAD_SP: 120.0}, {CORR_SP: 2.0}]


class TestARefusedWriteMovesNothing:
    """A write the model will not take, from the client's side and the ring's."""

    @pytest.fixture
    def unclamped(self, data_dir: Path) -> Served:
        """The same tree with no drive bands, so a refusable value gets
        through to the model at all: the clamp is what normally stops one."""
        return Served(data_dir, manifest=_manifest(), limits=False)

    def test_the_lattice_is_left_where_the_batch_found_it(self, unclamped: Served) -> None:
        """Rollback is lume-pyat's, unchanged: a lost closed orbit undoes
        every field the batch touched."""
        before = unclamped.element(QUAD_ELEMENT).PolynomB[1]

        unclamped.write(QUAD_SP, 1.0e5)

        assert unclamped.element(QUAD_ELEMENT).PolynomB[1] == pytest.approx(before)

    def test_neither_the_setpoint_nor_its_readback_records_the_value(
        self, unclamped: Served
    ) -> None:
        """Withholding the echo is the whole of a refusal: a value recorded
        in the parameter store is what a fresh caget reads, flushed or not."""
        unclamped.write(QUAD_SP, 1.0e5)

        assert QUAD_SP not in unclamped.driver.values
        assert QUAD_RB not in unclamped.driver.values

    def test_the_refusal_leaves_an_alarm_and_completes_the_write(self, unclamped: Served) -> None:
        """Put-completion cannot report failure, so the alarm is the only
        trace -- and completion must fire anyway, or the server library
        postpones every later write to that PV."""
        unclamped.write(QUAD_SP, 1.0e5)

        assert unclamped.driver.alarms == [(QUAD_SP, "WRITE", "MINOR")]
        assert unclamped.driver.completed == [QUAD_SP]


class TestNoReadbackCostsAClientItsCompletion:
    """The readback is the second value of a write, never the first.

    A Channel Access write whose ``callbackPV`` never fires postpones every
    later write to that PV for the life of the process, so a readback that
    cannot be produced must cost one value and not a client's put. Driven
    against the flat-address database: nothing here needs a lattice.
    """

    @staticmethod
    def _path(records: ServingRecords, value: Any) -> tuple[CohostWritePath, FakeDriver]:
        bound = {MAGNET_SP: BoundSetpoint(MAGNET_SP, MAGNET_RB, READBACK_INVERSE, value)}
        return (
            CohostWritePath(
                records,
                enqueue=lambda values, done=None, reset=False: done(None),
                bound_setpoints=bound,
            ),
            FakeDriver(),
        )

    def test_a_conversion_that_raises_leaves_the_setpoint_published(self, records) -> None:
        def explode(_value: Any) -> Any:
            raise ArithmeticError("the curve could not be evaluated")

        path, driver = self._path(records, explode)

        path.write(driver, MAGNET_SP, 5.0)

        assert driver.values == {MAGNET_SP: 5.0}
        assert driver.completed == [MAGNET_SP]

    def test_a_value_no_calibration_describes_is_published_unconverted(self, records) -> None:
        """Text, enum states and flags travel the same path as analog values
        and no conversion means anything for them -- the same reason the
        drive-band clamp lets them through."""
        path, driver = self._path(records, lambda value: value * 2)

        path.write(driver, MAGNET_SP, "OPEN")

        assert driver.values == {MAGNET_SP: "OPEN", MAGNET_RB: "OPEN"}
