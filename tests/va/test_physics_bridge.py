"""What a setpoint write does to a served accelerator, driven by address.

``PhysicsBridge`` is the IOC's whole view of the physics: a write arrives as an
address and a number, and what comes back is every monitor's reading. It holds
one model for the life of the process, and that model holds one lattice, so the
properties that make a served machine behave like a machine are properties of a
*sequence* of writes rather than of any one of them -- writing a device twice
sets it rather than adds to it, two independent devices reach the same state in
either order, and a write the ring cannot take changes nothing at all.

Those are what this module pins, together with the two things layered on top of
the truth: the seeded readout error, which perturbs the value pushed into a
record and never the physics the model published, and ``bind()``, which decides
which records get pushed.

Everything is addressed the way the IOC addresses it. Which addresses are
setpoints, which are readings, which element each drives and which plane each
reads come out of the served ``va_bindings.json``; the actuator the orbit tests
drive is chosen by its binding *kind*, and which transverse plane it moves is
measured rather than assumed. No family name, device count or strength constant
appears below.

The companion module ``test_physics_bridge_unknown_bpm.py`` covers the lookup
itself against a tree built to be hostile to address grammar; this one runs on
the demo facility's own generated tree, where there is a real ring under the
readings.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.ioc.physics_bridge import (
    OrbitSolveError,
    PhysicsBridge,
    UnknownDeviceError,
)
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

#: How far the orbit tests move an actuator from where it sits, in the
#: hardware unit the facility states for it. Large enough to move an orbit far
#: above the solver's own noise, small enough that a ring stays closed.
_STEP = 1.0

#: A seeded readout offset, in the unit the monitor publishes. Nothing derives
#: it from the machine: it is the fault being injected, and what the tests
#: check is that it arrives on the reading and nowhere else.
_OFFSET = 0.25

#: How far apart two readings must be before this suite calls them different.
#: Below any orbit shift a whole amp of kick produces and far above the
#: closed-orbit solver's own repeatability.
_MOVED = 1.0e-12


@pytest.fixture(scope="module")
def document() -> BindingsDocument:
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def channels() -> list[dict]:
    return build_manifest()["channels"]


@pytest.fixture
def model(channels: list[dict]) -> PyATRingModel:
    """A fresh model per test; every bridge below owns its own lattice."""
    return PyATRingModel(PACKAGE_PATHS.data_root, channels)


@pytest.fixture
def bridge(model: PyATRingModel) -> PhysicsBridge:
    return PhysicsBridge(model)


def _actuators(document: BindingsDocument) -> list[Binding]:
    """The bindings an orbit test drives: the ones that kick the beam.

    The document's own word for them. A facility that calls its correctors
    anything at all still states ``kind: kick`` for them, and a tree that binds
    none is a tree with no orbit to steer.
    """
    return [binding for binding in document.bindings if binding.kind == "kick"]


def _readings(document: BindingsDocument) -> list[str]:
    return [binding.setpoint_address for binding in document.bindings if binding.kind == "monitor"]


def _moved(before: dict[str, float], after: dict[str, float]) -> set[str]:
    """Which readings changed between two solves."""
    return {address for address, value in before.items() if abs(after[address] - value) > _MOVED}


class FakeRecord:
    """The one thing this bridge asks of a record: that it can be ``set``."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


class TestTheBridgeServesTheModelsTruth:
    def test_the_boot_readings_are_the_models_own_published_values(
        self, bridge: PhysicsBridge, model: PyATRingModel, document: BindingsDocument
    ) -> None:
        """The bridge converts nothing: a monitor's unit is applied inside the
        model, on the variable the binding built."""
        addresses = _readings(document)
        assert addresses
        assert bridge.bpm_positions() == dict(model.get(addresses))

    def test_a_write_moves_the_readings_the_model_moves(
        self, bridge: PhysicsBridge, model: PyATRingModel, document: BindingsDocument
    ) -> None:
        actuator = _actuators(document)[0]
        before = bridge.bpm_positions()

        bridge.on_setpoint(
            actuator.setpoint_address, float(model.get(actuator.setpoint_address)) + _STEP
        )

        after = bridge.bpm_positions()
        assert _moved(before, after)
        assert after == dict(model.get(_readings(document)))


class TestWritesComposeLikeTheirPhysicalCounterparts:
    def test_writing_one_address_twice_sets_rather_than_adds(
        self, bridge: PhysicsBridge, model: PyATRingModel, document: BindingsDocument
    ) -> None:
        """A setpoint is an absolute value. A bridge that accumulated would
        leave the machine somewhere no client asked for."""
        address = _actuators(document)[0].setpoint_address
        held = float(model.get(address))

        bridge.on_setpoint(address, held + _STEP)
        once = bridge.bpm_positions()
        bridge.on_setpoint(address, held + _STEP)

        assert bridge.bpm_positions() == once

    def test_two_addresses_in_either_order_reach_the_same_state(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        first, second = _actuators(document)[0], _actuators(document)[-1]
        assert first.setpoint_address != second.setpoint_address

        forward = PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels))
        reverse = PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels))
        for bridge, order in ((forward, (first, second)), (reverse, (second, first))):
            for binding in order:
                bridge.on_setpoint(binding.setpoint_address, _STEP)

        assert forward.bpm_positions() == reverse.bpm_positions()

    def test_a_sequence_of_writes_matches_writing_the_final_values_directly(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        """No history is carried: the machine is wherever its setpoints say,
        however it got there."""
        first, second = _actuators(document)[0], _actuators(document)[-1]

        wandered = PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels))
        for address, value in (
            (first.setpoint_address, _STEP),
            (second.setpoint_address, -_STEP),
            (first.setpoint_address, -_STEP),
            (second.setpoint_address, _STEP),
        ):
            wandered.on_setpoint(address, value)

        direct = PhysicsBridge(PyATRingModel(PACKAGE_PATHS.data_root, channels))
        direct.on_setpoint(first.setpoint_address, -_STEP)
        direct.on_setpoint(second.setpoint_address, _STEP)

        assert wandered.bpm_positions() == direct.bpm_positions()


class TestARefusedWriteIsACompleteNoOp:
    def _refuse(self, bridge: PhysicsBridge, binding: Binding) -> dict[str, float]:
        """Escalate one setpoint until the ring loses its closed orbit.

        Searched rather than pinned: which value costs a ring its orbit is the
        ring's own business, and a number chosen against one deck says nothing
        about another.

        Returns:
            The readings the *refused* write found. The writes that succeed on
            the way there are accepted writes and are meant to move the orbit;
            comparing against the readings the search started from would call
            those a rollback failure.
        """
        value = _STEP
        while abs(value) < 1.0e12:
            found = bridge.bpm_positions()
            value *= 10.0
            try:
                bridge.on_setpoint(binding.setpoint_address, value)
            except OrbitSolveError:
                return found
        raise AssertionError(f"{binding.setpoint_address}: no value cost the closed orbit")

    def test_the_readings_are_where_the_refused_write_found_them(
        self, bridge: PhysicsBridge, document: BindingsDocument
    ) -> None:
        actuator = _actuators(document)[0]

        before = self._refuse(bridge, actuator)

        assert bridge.bpm_positions() == before

    def test_the_bridge_still_serves_after_a_refusal(
        self, bridge: PhysicsBridge, document: BindingsDocument
    ) -> None:
        actuator = _actuators(document)[0]
        self._refuse(bridge, actuator)
        before = bridge.bpm_positions()

        bridge.on_setpoint(actuator.setpoint_address, _STEP)

        assert _moved(before, bridge.bpm_positions())


class TestOnlyASetpointThisModelDrivesIsAccepted:
    def test_a_monitor_address_is_not_a_setpoint(
        self, bridge: PhysicsBridge, document: BindingsDocument
    ) -> None:
        """A reading has no write path; accepting one would silently discard
        the value and leave the client believing it landed."""
        with pytest.raises(UnknownDeviceError):
            bridge.on_setpoint(_readings(document)[0], 0.0)

    def test_an_address_the_document_binds_nothing_to_is_refused(
        self, bridge: PhysicsBridge
    ) -> None:
        with pytest.raises(UnknownDeviceError):
            bridge.on_setpoint("no:channel:of:this:tree:exists", 0.0)


class TestTheSeededReadoutErrorIsOnTheReadingNotTheTruth:
    def _seeded(
        self, channels: list[dict], document: BindingsDocument, error: dict[str, float]
    ) -> tuple[PhysicsBridge, dict[str, FakeRecord], str]:
        """A bridge with *error* seeded on the first monitor of the document.

        Returns the bridge, the records it is bound to, and the address whose
        element carries the seeded error.
        """
        monitor = next(binding for binding in document.bindings if binding.kind == "monitor")
        bridge = PhysicsBridge(
            PyATRingModel(PACKAGE_PATHS.data_root, channels, bpm_errors={monitor.element: error})
        )
        records = {address: FakeRecord() for address in _readings(document)}
        bridge.bind(records)
        return bridge, records, monitor.setpoint_address

    def test_an_offset_shifts_the_record_but_not_the_published_truth(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        clean, clean_records, address = self._seeded(channels, document, {})
        seeded, seeded_records, _ = self._seeded(channels, document, {"offset_x": _OFFSET})

        assert seeded.bpm_positions() == clean.bpm_positions()
        assert seeded_records[address].value != pytest.approx(clean_records[address].value)

    def test_the_offset_leaves_the_response_slope_unchanged(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        """An offset is a constant, so it cancels out of a difference. A
        seeded offset that moved a measured response would be a calibration
        error wearing a readout error's name."""
        actuator = _actuators(document)[0]
        slopes = []
        for error in ({}, {"offset_x": _OFFSET}):
            bridge, records, address = self._seeded(channels, document, error)
            low = records[address].value
            bridge.on_setpoint(actuator.setpoint_address, _STEP)
            slopes.append(records[address].value - low)

        assert slopes[0] == pytest.approx(slopes[1])

    def test_a_polarity_flip_anti_correlates_with_the_unflipped_reading(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        actuator = _actuators(document)[0]
        readings = []
        for error in ({}, {"polarity_x": -1.0}):
            bridge, records, address = self._seeded(channels, document, error)
            bridge.on_setpoint(actuator.setpoint_address, _STEP)
            readings.append(records[address].value)

        assert readings[0] != pytest.approx(0.0, abs=_MOVED)
        assert readings[1] == pytest.approx(-readings[0])

    def test_an_error_at_one_monitor_leaves_every_other_alone(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        clean, clean_records, address = self._seeded(channels, document, {})
        seeded, seeded_records, _ = self._seeded(channels, document, {"offset_x": _OFFSET})

        elsewhere = [key for key in clean_records if key != address]
        assert elsewhere
        assert [
            key for key in elsewhere if seeded_records[key].value != clean_records[key].value
        ] == []


class TestSeededNoiseIsReproducible:
    def test_the_same_seed_gives_the_same_readings(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        """A stand-in target that could not be replayed would make every
        comparison against it a coin toss."""
        monitor = next(binding for binding in document.bindings if binding.kind == "monitor")
        runs = []
        for _ in range(2):
            bridge = PhysicsBridge(
                PyATRingModel(
                    PACKAGE_PATHS.data_root,
                    channels,
                    bpm_errors={monitor.element: {"noise_x": _OFFSET}},
                ),
                rng_seed=20260917,
            )
            records = {address: FakeRecord() for address in _readings(document)}
            bridge.bind(records)
            runs.append(records[monitor.setpoint_address].value)

        assert runs[0] == runs[1]
        assert runs[0] != pytest.approx(bridge.bpm_positions()[monitor.setpoint_address])


class TestBindDecidesWhichRecordsArePushed:
    def test_bind_pushes_the_initial_readings_into_the_records(
        self, bridge: PhysicsBridge, document: BindingsDocument
    ) -> None:
        """A client connecting before the first write still gets a reading."""
        records = {address: FakeRecord() for address in _readings(document)}

        bridge.bind(records)

        assert [address for address, record in records.items() if record.value is None] == []

    def test_a_setpoint_record_is_not_pushed(
        self, bridge: PhysicsBridge, document: BindingsDocument
    ) -> None:
        """The write path serves a setpoint's own readback per the binding's
        readback rule; a bridge pushing into it would publish a second,
        unconverted value on the same address."""
        setpoint = _actuators(document)[0].setpoint_address
        records = {setpoint: FakeRecord(), **{a: FakeRecord() for a in _readings(document)}}

        bridge.bind(records)

        assert records[setpoint].value is None

    def test_a_later_write_pushes_updated_readings(
        self, bridge: PhysicsBridge, model: PyATRingModel, document: BindingsDocument
    ) -> None:
        records = {address: FakeRecord() for address in _readings(document)}
        bridge.bind(records)
        before = {address: record.value for address, record in records.items()}
        actuator = _actuators(document)[0]

        bridge.on_setpoint(
            actuator.setpoint_address, float(model.get(actuator.setpoint_address)) + _STEP
        )

        assert _moved(before, {address: record.value for address, record in records.items()})

    def test_a_reading_with_no_record_does_not_prevent_a_write(
        self, bridge: PhysicsBridge, document: BindingsDocument
    ) -> None:
        """An IOC serving a subset of the namespace binds a subset of the
        records, and the rest of the machine still moves."""
        addresses = _readings(document)
        records = {addresses[0]: FakeRecord()}
        bridge.bind(records)
        before = bridge.bpm_positions()

        bridge.on_setpoint(_actuators(document)[0].setpoint_address, _STEP)

        assert _moved(before, bridge.bpm_positions())
        assert records[addresses[0]].value is not None
