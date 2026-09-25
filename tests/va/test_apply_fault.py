"""The stuck-setpoint apply fault, as a client experiences it.

A stuck setpoint is a build-time fixture: the named address still records what
was written to it, and its paired readback never moves again. The point of the
fault is that it is a property of the *substrate* rather than of one client's
view -- a client that reads back its own command is told the truth about the
command, and every client that reads the device sees the same frozen readback.
That distinction is only observable over a real wire, which is what most of
this file is.

The venue and the in-process topology are the ones
``tests/va/test_record_factory.py`` documents: one process, a real ``pcaspy``
server, a real ``pyepics`` client, and ``epics.ca.initialize_libca()`` before
the pcaspy import so the client binds its own libca rather than the second copy
the server extension exports. pcaspy has no loadable macOS arm64 wheel, so the
live classes skip on a developer host and are proven in a linux container and
on CI; the route-table tests need no server and run everywhere.

The address set here is deliberately disjoint from every other module's, so two
servers alive in one pytest session can never answer for each other's names.

The other two apply faults -- BPM reading errors and magnet calibration -- are
model state rather than a route, and need no wire: the model is seeded with
them, holds them as writable variables, and the physics bridge reads them back
on every push and every write. ``TestApplyFaultsLiveInTheModel`` pins that the
bridge serves whatever the model holds now, and keeps no copy of its own.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from queue import Queue
from typing import Any

import pytest


def _free_port() -> str:
    """An unused loopback TCP port, as a string ready for the environment."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


# import-time required because libca latches the EPICS_CA_* environment when
# the C library initialises, which happens on the first `import epics` anywhere
# in the process. Loopback only, on an ephemeral port unless the environment pins one,
# with the server and CAS ports equal -- a search reply carries the server's own
# port, so a server listening anywhere else hands clients a dead address.
os.environ.setdefault("EPICS_CA_ADDR_LIST", "127.0.0.1")
os.environ.setdefault("EPICS_CA_AUTO_ADDR_LIST", "NO")
os.environ.setdefault("EPICS_CA_SERVER_PORT", _free_port())
os.environ.setdefault("EPICS_CAS_SERVER_PORT", os.environ["EPICS_CA_SERVER_PORT"])
os.environ.setdefault("EPICS_CA_REPEATER_PORT", _free_port())

from osprey.services.virtual_accelerator.bindings import (  # noqa: E402
    Binding,
    BindingsDocument,
    load_bindings,
)
from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge  # noqa: E402
from osprey.services.virtual_accelerator.manifest import (  # noqa: E402
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    RECORD_TYPE_ANALOG,
    build_manifest,
)
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS  # noqa: E402
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel  # noqa: E402
from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb  # noqa: E402
from osprey.services.virtual_accelerator.serving.write_path import (  # noqa: E402
    MODE_ECHO,
    MODE_LATCH,
    MODE_PHYSICS,
    CohostWritePath,
    physics_setpoint_addresses,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 22

CA_TIMEOUT_S = 10.0
SETTLE_TIMEOUT_S = 10.0

#: What the model-held fault tests move an actuator by, and the readout offset
#: they seed, in the hardware units the served tree states. Nothing derives
#: either from the machine: they are the command and the fault being injected.
_MODEL_STEP = 1.0
_MODEL_OFFSET = 0.25

RING = "ZZAF"

# A faulted echo pair and an unfaulted one, so the fault can be shown to be per
# channel rather than a partition-wide switch. Distinct devices, because the
# setpoint/readback pairing is keyed on device identity and two pairs sharing
# one key would echo onto each other's readback.
STUCK_SP = f"{RING}:RF:CAV:01:VOLTAGE:SP"
STUCK_RB = f"{RING}:RF:CAV:01:VOLTAGE:RB"
LIVE_SP = f"{RING}:RF:CAV:02:VOLTAGE:SP"
LIVE_RB = f"{RING}:RF:CAV:02:VOLTAGE:RB"

# A faulted magnet: a pyat-coupled setpoint, so the fault can be shown to
# suppress the physics hand-off as well as the echo.
STUCK_MAGNET_SP = f"{RING}:MAG:HCM:01:CURRENT:SP"
STUCK_MAGNET_RB = f"{RING}:MAG:HCM:01:CURRENT:RB"
LIVE_MAGNET_SP = f"{RING}:MAG:HCM:02:CURRENT:SP"
LIVE_MAGNET_RB = f"{RING}:MAG:HCM:02:CURRENT:RB"

# The readbacks are seeded to a nonzero boot value on purpose: a frozen readback
# holding zero is indistinguishable from an unseeded one, so "never moved" would
# be a claim the test could not actually make.
STUCK_BOOT = 2.5
STUCK_SETPOINTS = frozenset({STUCK_SP, STUCK_MAGNET_SP})

BOOT_VALUES = {
    STUCK_SP: STUCK_BOOT,
    STUCK_RB: STUCK_BOOT,
    STUCK_MAGNET_SP: STUCK_BOOT,
    STUCK_MAGNET_RB: STUCK_BOOT,
    LIVE_SP: STUCK_BOOT,
    LIVE_RB: STUCK_BOOT,
    LIVE_MAGNET_SP: STUCK_BOOT,
    LIVE_MAGNET_RB: STUCK_BOOT,
}


def _channel(
    address: str,
    *,
    partition: str,
    subfield: str,
    system: str,
    family: str,
    device: str,
    field: str,
    record_type: str = RECORD_TYPE_ANALOG,
    noise: bool = False,
) -> dict:
    """One synthetic manifest entry, in the shape ``build_serving_pvdb`` reads."""
    return {
        "address": address,
        "ring": RING,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": record_type,
        "noise": noise,
    }


def _pair(system: str, family: str, device: str, field: str, partition: str) -> list[dict]:
    """A setpoint and its paired readback."""
    prefix = f"{RING}:{system}:{family}:{device}:{field}"
    return [
        _channel(
            f"{prefix}:SP",
            partition=partition,
            subfield="SP",
            system=system,
            family=family,
            device=device,
            field=field,
        ),
        _channel(
            f"{prefix}:RB",
            partition=partition,
            subfield="RB",
            system=system,
            family=family,
            device=device,
            field=field,
        ),
    ]


SERVED_CHANNELS = [
    *_pair("RF", "CAV", "01", "VOLTAGE", PARTITION_SP_ECHO),
    *_pair("RF", "CAV", "02", "VOLTAGE", PARTITION_SP_ECHO),
    *_pair("MAG", "HCM", "01", "CURRENT", PARTITION_PYAT_COUPLED),
    *_pair("MAG", "HCM", "02", "CURRENT", PARTITION_PYAT_COUPLED),
]


def _build_records(**kwargs: Any) -> Any:
    return build_serving_pvdb(
        SERVED_CHANNELS, boot_values=BOOT_VALUES, async_setpoints=True, **kwargs
    )


class _RunLoop:
    """The model's thread. See ``test_record_factory.py`` for why it exists."""

    def __init__(self, on_setpoint) -> None:  # test-local callable
        self._on_setpoint = on_setpoint
        self._queue: Queue = Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="va-fault-run-loop")
        self.thread.start()

    def enqueue(self, values: dict, *, done) -> None:  # test-local callable
        self._queue.put((values, done))

    def _run(self) -> None:
        while True:
            values, done = self._queue.get()
            error = None
            try:
                for address, item in values.items():
                    self._on_setpoint(address, item["value"])
            except Exception as exc:  # the loop reports, never raises
                error = str(exc)
            done(error)


class _PhysicsHook:
    """Records every write the lattice would have been given."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def __call__(self, address: str, value: float) -> None:
        self.calls.append((address, value))


class LiveNamespace:
    """A served namespace with one faulted channel in each partition."""

    def __init__(self, records, hook, loop) -> None:
        self.records = records
        self.hook = hook
        self.loop = loop


@pytest.fixture(scope="module")
def live() -> Any:
    """A live Channel Access server serving a faulted namespace."""
    import epics

    epics.ca.initialize_libca()
    pcaspy = pytest.importorskip(
        "pcaspy",
        reason=(
            "the live Channel Access venue needs pcaspy, which has no loadable "
            "macOS arm64 wheel; run this suite in a linux container or on CI"
        ),
    )

    records = _build_records()
    hook = _PhysicsHook()
    loop = _RunLoop(hook)
    path = CohostWritePath(
        records,
        enqueue=loop.enqueue,
        physics_setpoints=physics_setpoint_addresses(records),
        stuck_setpoints=STUCK_SETPOINTS,
        refusal_alarm=(pcaspy.Alarm.WRITE_ALARM, pcaspy.Severity.INVALID_ALARM),
    )

    class LiveDriver(pcaspy.Driver):
        """The production driver's whole body: delegate to the write path."""

        def write(self, reason: str, value: Any) -> bool:  # pcaspy contract
            accepted: bool = path.write(self, reason, value)
            return accepted

    server = pcaspy.SimpleServer()
    server.createPV("", records.pvdb)
    driver = LiveDriver()
    records.attach_driver(driver)

    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            server.process(0.05)

    thread = threading.Thread(target=serve, daemon=True, name="va-fault-cas")
    thread.start()

    if not _wait_until(lambda: _caget(LIVE_RB) is not None):
        stop.set()
        pytest.fail("the Channel Access server never became reachable")

    yield LiveNamespace(records, hook, loop)

    stop.set()
    thread.join(timeout=5)


def _wait_until(predicate, *, timeout: float = SETTLE_TIMEOUT_S) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.05)
    return predicate()


def _caget(address: str) -> Any:
    """Read one value over the wire, never from pyepics' monitor cache.

    ``use_monitor=False`` is load-bearing, not stylistic: pyepics' ``caget``
    otherwise returns whatever its monitor subscription last cached, which
    after a write is whatever arrived BEFORE the write did. This suite asserts
    what a client sees over a real wire, including that a refused write moved
    NOTHING -- and a stale cache never moves, so that assertion would pass for
    free. The sibling live suite was observed reading back a previous test's
    value for exactly this reason.

    Same reasoning, and the same fix, as every read in
    ``scripts/va/build_and_boot_check.sh``.
    """
    import epics

    return epics.caget(
        address, timeout=CA_TIMEOUT_S, connection_timeout=CA_TIMEOUT_S, use_monitor=False
    )


def _caput(address: str, value: float) -> Any:
    import epics

    return epics.caput(address, value, wait=True, timeout=CA_TIMEOUT_S)


def _settle(address: str, expected: float) -> Any:
    last: list[Any] = [None]

    def reached() -> bool:
        last[0] = _caget(address)
        return last[0] is not None and abs(last[0] - expected) < 1e-9

    _wait_until(reached)
    return last[0]


class TestFaultRouting:
    """The route table, before any server exists.

    A faulted setpoint routes to ``MODE_LATCH`` and owes no echo; that is the
    whole mechanism, and it is worth pinning separately from its consequences so
    a regression says which of the two broke.
    """

    def test_a_faulted_setpoint_latches_and_owes_no_echo(self) -> None:
        records = _build_records()
        path = CohostWritePath(
            records,
            physics_setpoints=physics_setpoint_addresses(records),
            stuck_setpoints=STUCK_SETPOINTS,
        )
        route = path.routes[STUCK_SP]

        assert route.mode == MODE_LATCH
        assert route.readback is None

    def test_an_unfaulted_echo_setpoint_still_echoes(self) -> None:
        records = _build_records()
        path = CohostWritePath(records, stuck_setpoints=STUCK_SETPOINTS)

        assert path.routes[LIVE_SP].mode == MODE_ECHO
        assert path.routes[LIVE_SP].readback == LIVE_RB

    def test_an_unfaulted_magnet_still_reaches_the_model(self) -> None:
        records = _build_records()
        loop = _RunLoop(_PhysicsHook())
        path = CohostWritePath(
            records,
            enqueue=loop.enqueue,
            physics_setpoints=physics_setpoint_addresses(records),
            stuck_setpoints=STUCK_SETPOINTS,
        )

        assert path.routes[LIVE_MAGNET_SP].mode == MODE_PHYSICS
        assert path.routes[STUCK_MAGNET_SP].mode == MODE_LATCH

    def test_the_fault_is_off_by_default(self) -> None:
        """No channel is faulted unless it is named: the fixture is opt-in, so
        an unconfigured virtual accelerator serves an unfaulted machine."""
        records = _build_records()
        loop = _RunLoop(_PhysicsHook())
        path = CohostWritePath(
            records,
            enqueue=loop.enqueue,
            physics_setpoints=physics_setpoint_addresses(records),
        )

        # Nothing latches: every setpoint either reaches the model or echoes,
        # and the address this module faults elsewhere is an ordinary echo here.
        assert {route.mode for route in path.routes.values()} == {MODE_ECHO, MODE_PHYSICS}
        assert path.routes[STUCK_SP].mode == MODE_ECHO
        assert path.routes[STUCK_SP].readback == STUCK_RB
        assert path.routes[STUCK_MAGNET_SP].mode == MODE_PHYSICS

    def test_an_address_that_is_not_served_is_inert(self) -> None:
        """``stuck_setpoints`` is a per-channel fixture, not a channel selector:
        naming something that is not served must not invent a route for it, and
        must not disturb the ones that are."""
        records = _build_records()
        path = CohostWritePath(
            records, stuck_setpoints=frozenset({"NOT:A:REAL:CHANNEL:SP", STUCK_SP})
        )

        assert "NOT:A:REAL:CHANNEL:SP" not in path.routes
        assert path.routes[LIVE_SP].mode == MODE_ECHO
        assert path.routes[STUCK_SP].mode == MODE_LATCH

    def test_a_faulted_setpoint_is_still_writable(self) -> None:
        """The fault freezes the device, it does not withdraw the channel: a
        client must still be able to write the setpoint and read its command
        back, which is the difference between a broken magnet and a missing
        one."""
        records = _build_records()
        path = CohostWritePath(records, stuck_setpoints=STUCK_SETPOINTS)

        assert STUCK_SP in path.routes


# The served tree the model-held faults are shown on: the packaged demo tree,
# read the way every other model test reads it. Nothing below is served over a
# wire, so it cannot collide with any live server's names -- and no address is
# written down here at all: the bindings document names the monitor and the
# actuator, and these tests only ask it which.


class _ReadingRecord:
    """A monitor readback record reduced to the one method the bridge calls."""

    def __init__(self) -> None:
        self.value: float | None = None

    def set(self, value: float) -> None:
        self.value = value


class _CommandedRecord(_ReadingRecord):
    """A setpoint record the bridge reads a standing command out of."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def get(self) -> float | None:
        return self.value


def _faulted_model(**seeds: Any) -> PyATRingModel:
    """The packaged tree's model, seeded with ``seeds``."""
    return PyATRingModel(PACKAGE_PATHS.data_root, build_manifest()["channels"], **seeds)


def _document() -> BindingsDocument:
    return load_bindings(PACKAGE_PATHS.va_bindings)


def _a_monitor(document: BindingsDocument) -> Binding:
    """The first monitor reading the document publishes."""
    return next(binding for binding in document.bindings if binding.kind == "monitor")


def _an_actuator(document: BindingsDocument) -> Binding:
    """The first device the document says kicks the beam."""
    return next(binding for binding in document.bindings if binding.kind == "kick")


def _served(model: PyATRingModel, reading: str) -> tuple[PhysicsBridge, _ReadingRecord]:
    """A bridge over ``model`` with one bound monitor reading."""
    bridge = PhysicsBridge(model)
    record = _ReadingRecord()
    bridge.bind({reading: record})
    return bridge, record


class TestApplyFaultsLiveInTheModel:
    """The readout and calibration apply faults, without a server.

    Seeded through ``PyATRingModel``, writable through its public ``set()``
    like any other variable, and read back by the bridge each time it serves
    -- so the value the model holds now is the only one a client ever sees.
    """

    def test_a_seeded_readout_fault_reaches_the_served_reading(self) -> None:
        document = _document()
        monitor, actuator = _a_monitor(document), _an_actuator(document)
        model = _faulted_model(bpm_errors={monitor.element: {"offset_x": _MODEL_OFFSET}})
        bridge, record = _served(model, monitor.setpoint_address)

        bridge.on_setpoint(actuator.setpoint_address, _MODEL_STEP)

        truth = bridge.bpm_positions()[monitor.setpoint_address]
        assert record.value == pytest.approx(truth - _MODEL_OFFSET, abs=1e-12)

    def test_a_readout_fault_written_to_the_model_is_served_on_the_next_push(self) -> None:
        document = _document()
        monitor, actuator = _a_monitor(document), _an_actuator(document)
        model = _faulted_model()
        bridge, record = _served(model, monitor.setpoint_address)

        model.set({f"{monitor.element}.gain_x": 2.0})
        bridge.on_setpoint(actuator.setpoint_address, _MODEL_STEP)

        truth = bridge.bpm_positions()[monitor.setpoint_address]
        assert record.value == pytest.approx(2.0 * truth, abs=1e-12)

    def test_a_calibration_written_to_the_model_scales_the_next_setpoint(self) -> None:
        """A polarity flip is the exact oracle: the ring has to end where the
        opposite command would have left it."""
        document = _document()
        monitor, actuator = _a_monitor(document), _an_actuator(document)
        flipped = _faulted_model()
        bridge, _record = _served(flipped, monitor.setpoint_address)
        flipped.set({f"{actuator.element}.cal_factor": -1.0})
        bridge.on_setpoint(actuator.setpoint_address, _MODEL_STEP)

        plain, _plain_record = _served(_faulted_model(), monitor.setpoint_address)
        plain.on_setpoint(actuator.setpoint_address, -_MODEL_STEP)

        assert bridge.bpm_positions() == pytest.approx(plain.bpm_positions())

    def test_a_model_reset_restores_the_seeded_fault(self) -> None:
        document = _document()
        monitor, actuator = _a_monitor(document), _an_actuator(document)
        model = _faulted_model(bpm_errors={monitor.element: {"offset_x": _MODEL_OFFSET}})
        bridge, record = _served(model, monitor.setpoint_address)

        model.set({f"{monitor.element}.offset_x": -_MODEL_OFFSET})
        model.reset()
        bridge.on_setpoint(actuator.setpoint_address, _MODEL_STEP)

        truth = bridge.bpm_positions()[monitor.setpoint_address]
        assert record.value == pytest.approx(truth - _MODEL_OFFSET, abs=1e-12)

    def test_a_calibration_change_re_commands_what_the_operator_asked_for(self) -> None:
        """What ``refresh`` is for: a model write reaches no setpoint, so the
        magnet would go on delivering the wrong value for its standing command
        until the bridge re-applies it."""
        actuator = _an_actuator(_document())
        address = actuator.setpoint_address
        model = _faulted_model()
        bridge = PhysicsBridge(model)
        command = _CommandedRecord(_MODEL_STEP)
        bridge.bind({address: command}, physics_setpoints=frozenset({address}))
        bridge.on_setpoint(address, _MODEL_STEP)

        model.set({f"{actuator.element}.cal_factor": -1.0})
        bridge.refresh([f"{actuator.element}.cal_factor"])

        plain = PhysicsBridge(_faulted_model())
        plain.on_setpoint(address, -_MODEL_STEP)
        assert bridge.bpm_positions() == pytest.approx(plain.bpm_positions())
        assert command.value == _MODEL_STEP, "the operator's command is read, never rewritten"

    def test_a_calibration_change_leaves_a_stuck_setpoint_where_it_is(self) -> None:
        """A stuck setpoint records what was written to it and hands the model
        nothing, so its record carries a value the ring never took. Re-applying
        that on a calibration change would move the magnet to a value it
        refused -- which is the one thing being stuck means it cannot do."""
        actuator = _an_actuator(_document())
        address = actuator.setpoint_address
        model = _faulted_model()
        bridge = PhysicsBridge(model)
        bridge.follow_stuck_setpoints(lambda: frozenset({address}))
        # What a latched write leaves behind: the record moved, the model did not.
        bridge.bind(
            {address: _CommandedRecord(_MODEL_STEP)}, physics_setpoints=frozenset({address})
        )
        before = bridge.bpm_positions()

        model.set({f"{actuator.element}.cal_factor": -1.0})
        bridge.refresh([f"{actuator.element}.cal_factor"])

        assert model.get(address) == 0.0, "the stuck magnet took the latched command"
        assert bridge.bpm_positions() == pytest.approx(before)

    def test_clearing_the_stuck_fault_lets_the_next_calibration_change_through(self) -> None:
        """The set is read when ``refresh`` runs, not copied when the bridge
        was wired, so clearing the fault at runtime is enough."""
        actuator = _an_actuator(_document())
        address = actuator.setpoint_address
        model = _faulted_model()
        bridge = PhysicsBridge(model)
        stuck = {address}
        bridge.follow_stuck_setpoints(lambda: frozenset(stuck))
        bridge.bind(
            {address: _CommandedRecord(_MODEL_STEP)}, physics_setpoints=frozenset({address})
        )

        stuck.clear()
        model.set({f"{actuator.element}.cal_factor": 1.0})
        bridge.refresh([f"{actuator.element}.cal_factor"])

        assert model.get(address) == pytest.approx(_MODEL_STEP)

    @pytest.mark.parametrize("seed_kwarg", ["bpm_errors", "corrector_gains"])
    def test_the_bridge_takes_no_fault_seed_of_its_own(self, seed_kwarg: str) -> None:
        with pytest.raises(TypeError, match=seed_kwarg):
            PhysicsBridge(model=_faulted_model(), **{seed_kwarg: {}})


class TestLiveStuckEchoPair:
    """A faulted sp-echo channel, over the wire."""

    @pytest.mark.usefixtures("live")
    def test_the_setpoint_latches_the_written_value(self) -> None:
        assert _caput(STUCK_SP, 7.25) == 1

        assert _settle(STUCK_SP, 7.25) == pytest.approx(7.25)

    @pytest.mark.usefixtures("live")
    def test_the_readback_stays_at_its_boot_value(self) -> None:
        assert _caput(STUCK_SP, 4.0) == 1
        _settle(STUCK_SP, 4.0)

        assert _caget(STUCK_RB) == pytest.approx(STUCK_BOOT)

    @pytest.mark.usefixtures("live")
    def test_repeated_writes_never_move_it(self) -> None:
        """Frozen means frozen, not merely lagging by one write."""
        for value in (1.0, -3.0, 11.5):
            assert _caput(STUCK_SP, value) == 1
            _settle(STUCK_SP, value)

        assert _caget(STUCK_RB) == pytest.approx(STUCK_BOOT)

    @pytest.mark.usefixtures("live")
    def test_a_monitoring_client_is_told_nothing(self) -> None:
        """The freeze is in the served value, so there is no monitor event to
        deliver either -- a subscriber sees a device that simply never moves,
        not one that reports a value identical to the last."""
        import epics

        seen: list[float] = []
        readback = epics.PV(STUCK_RB, auto_monitor=True)
        try:
            assert readback.wait_for_connection(timeout=CA_TIMEOUT_S)
            assert readback.get(use_monitor=False) == pytest.approx(STUCK_BOOT)
            readback.add_callback(lambda value=None, **_: seen.append(value))

            assert _caput(STUCK_SP, 6.0) == 1
            _settle(STUCK_SP, 6.0)
            time.sleep(0.5)
        finally:
            # Explicit, in a finally: a PV finalised by the garbage collector
            # tears libca down from the wrong thread.
            readback.disconnect()

        assert [value for value in seen if abs(value - STUCK_BOOT) > 1e-9] == []

    @pytest.mark.usefixtures("live")
    def test_the_write_still_completes(self) -> None:
        """The server library postpones every later write to a PV whose
        asynchronous write never completed, so a fault that skipped completion
        would freeze the setpoint as well as the readback -- and the client
        would hang rather than be told a lie it could detect."""
        assert _caput(STUCK_SP, 1.0) == 1
        assert _caput(STUCK_SP, 2.0) == 1

        assert _settle(STUCK_SP, 2.0) == pytest.approx(2.0)

    @pytest.mark.usefixtures("live")
    def test_the_unfaulted_sibling_still_echoes(self) -> None:
        assert _caput(LIVE_SP, 4.5) == 1

        assert _settle(LIVE_RB, 4.5) == pytest.approx(4.5)

    @pytest.mark.usefixtures("live")
    def test_faulting_one_channel_leaves_its_sibling_alone(self) -> None:
        assert _caput(LIVE_SP, 3.0) == 1
        _settle(LIVE_RB, 3.0)

        assert _caput(STUCK_SP, 8.0) == 1
        _settle(STUCK_SP, 8.0)

        assert _caget(LIVE_RB) == pytest.approx(3.0)


class TestLiveStuckMagnet:
    """A faulted pyat-coupled channel: the machine must not move either.

    Suppressing only the echo would leave a magnet whose readback lies still
    while the lattice underneath it tracks every command -- the orbit would
    move, and no channel would say why.
    """

    @pytest.mark.usefixtures("live")
    def test_the_setpoint_latches_the_written_value(self) -> None:
        assert _caput(STUCK_MAGNET_SP, 5.5) == 1

        assert _settle(STUCK_MAGNET_SP, 5.5) == pytest.approx(5.5)

    @pytest.mark.usefixtures("live")
    def test_the_readback_stays_at_its_boot_value(self) -> None:
        assert _caput(STUCK_MAGNET_SP, -2.0) == 1
        _settle(STUCK_MAGNET_SP, -2.0)

        assert _caget(STUCK_MAGNET_RB) == pytest.approx(STUCK_BOOT)

    def test_the_write_never_reaches_the_model(self, live: Any) -> None:
        live.hook.calls.clear()
        assert _caput(STUCK_MAGNET_SP, 3.5) == 1
        _settle(STUCK_MAGNET_SP, 3.5)

        assert live.hook.calls == []

    def test_an_unfaulted_magnet_still_reaches_the_model_and_echoes(self, live: Any) -> None:
        live.hook.calls.clear()
        assert _caput(LIVE_MAGNET_SP, 1.75) == 1
        _wait_until(lambda: live.hook.calls)

        assert live.hook.calls == [(LIVE_MAGNET_SP, pytest.approx(1.75))]
        assert _settle(LIVE_MAGNET_RB, 1.75) == pytest.approx(1.75)


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_apply_fault.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
