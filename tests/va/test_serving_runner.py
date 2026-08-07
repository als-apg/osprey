"""The co-hosted write path, and the shape of the runner that hosts it.

What a client write to a setpoint does is decided in
:mod:`~osprey.services.virtual_accelerator.serving.write_path`, which holds
no server and no model, so this suite drives the real write path against the
real serving database and a fake Channel Access driver. Nothing here binds a
port, creates a server or imports the CA server extension -- live-CA
behaviour is proven against the deployed container, not here.

The fake driver is faithful on the one point every assertion below rests on:
its parameter store is the served database. A fresh one-shot read (``caget``)
is answered straight out of that store, while a monitoring client is served
only what ``updatePV`` posts. So "moved nothing for any reader" is two
assertions, not one -- the store is unchanged *and* nothing was posted -- and
a write path that recorded a value while withholding the post would fail the
first of them, which is exactly the failure mode the runner's configuration
exists to prevent.

The PVA half of the namespace is stood in for by :class:`FakePvaChannels`,
which is faithful on the mirror-image point: a p4p ``post`` both replaces the
value a one-shot ``get`` is answered with and is the monitor update, so on
that transport "nothing moved" is a single assertion. Its posts are recorded
in the driver's own journal, so the order in which the two views of one
address move -- and the order of both against put-completion -- is one
sequence rather than two that have to be reconciled.

The run loop is stood in for by :class:`FakeRunLoop`, which reproduces the
four steps the real loop takes around each queued item. That is the boundary
of what can be proven in process: the loop itself, the server, and the
subclass that binds them live behind an import of the CA server extension
that this host cannot satisfy. What can still be checked without it -- that
the subclass publishes nothing from a model read, that it attaches the
records only once the driver exists -- is checked structurally, against the
module's own syntax tree, in :class:`TestRunnerShape`.
"""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
)
from osprey.services.virtual_accelerator.serving import write_path as write_path_module
from osprey.services.virtual_accelerator.serving.pvdb import (
    ServingRecords,
    build_serving_pvdb,
)
from osprey.services.virtual_accelerator.serving.write_path import (
    MODE_ECHO,
    MODE_LATCH,
    MODE_PHYSICS,
    NOT_WRITABLE,
    RUNNER_CONFIG_POLICY,
    CohostWritePath,
    SetpointRoutedModel,
    clamp_into,
    physics_setpoint_addresses,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 90

RING = "ZZRS"

# A pyat-coupled magnet: setpoint, its own current readback, and the BPM
# reading the physics hook pushes after a solve.
MAG_SP = f"{RING}:MAG:HCM:01:CURRENT:SP"
MAG_RB = f"{RING}:MAG:HCM:01:CURRENT:RB"
BPM_X = f"{RING}:DIAG:BPM:01:POSITION:X"
# A second magnet, frozen by an apply fault.
STUCK_SP = f"{RING}:MAG:HCM:02:CURRENT:SP"
STUCK_RB = f"{RING}:MAG:HCM:02:CURRENT:RB"
# A plain setpoint/readback echo pair: no physics behind it.
ECHO_SP = f"{RING}:VAC:VALVE:01:POSITION:SP"
ECHO_RB = f"{RING}:VAC:VALVE:01:POSITION:RB"
# Telemetry: driven by the engine, never writable by a client.
TELEM_RB = f"{RING}:VAC:GAUGE:01:PRESSURE:RB"

MAG_BAND = (-10.0, 10.0)
DRIVE_LIMITS = {MAG_SP: MAG_BAND, STUCK_SP: MAG_BAND, ECHO_SP: (0.0, 100.0)}
BOOT_VALUES = {MAG_SP: 1.5, MAG_RB: 1.5, STUCK_SP: 2.5, STUCK_RB: 2.5, ECHO_SP: 4.0, ECHO_RB: 4.0}

# What the fake model reports when read back. No served value may ever carry
# it: every value on the wire comes from a client write or from the physics
# hook's own push, never from reading the model.
POISON = -9999.0

# The reading the physics hook pushes onto the BPM after an accepted write.
# Deliberately unrelated to the written current, so a value that appears on
# the BPM can only have come from the push.
BPM_READING = 0.000123

# The setpoints whose writes go through the physics hook. Derived from the
# built database by `physics_setpoint_addresses`; pinned here as well so a
# test that wires the wrapper directly does not depend on that derivation.
PHYSICS_SETPOINTS = frozenset({MAG_SP, STUCK_SP})

# The addresses served on PVA as well as on Channel Access: exactly the
# model's own variables. Every other co-hosted address has one view only --
# note that a magnet's `:RB` is not among them, because the model describes
# the current that was commanded and the readings that came out, not the
# readback that echoes a command.
PVA_CHANNELS = frozenset({MAG_SP, STUCK_SP, BPM_X})


def _channel(
    address: str,
    *,
    subfield: str,
    partition: str,
    system: str,
    family: str,
    device: str,
    field: str,
) -> dict:
    """One synthetic manifest channel, in the shape ``build_manifest()`` emits."""
    return {
        "address": address,
        "ring": RING,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    }


CHANNELS = [
    _channel(
        MAG_SP,
        subfield="SP",
        partition=PARTITION_PYAT_COUPLED,
        system="MAG",
        family="HCM",
        device="01",
        field="CURRENT",
    ),
    _channel(
        MAG_RB,
        subfield="RB",
        partition=PARTITION_PYAT_COUPLED,
        system="MAG",
        family="HCM",
        device="01",
        field="CURRENT",
    ),
    _channel(
        STUCK_SP,
        subfield="SP",
        partition=PARTITION_PYAT_COUPLED,
        system="MAG",
        family="HCM",
        device="02",
        field="CURRENT",
    ),
    _channel(
        STUCK_RB,
        subfield="RB",
        partition=PARTITION_PYAT_COUPLED,
        system="MAG",
        family="HCM",
        device="02",
        field="CURRENT",
    ),
    _channel(
        BPM_X,
        subfield="X",
        partition=PARTITION_PYAT_COUPLED,
        system="DIAG",
        family="BPM",
        device="01",
        field="POSITION",
    ),
    _channel(
        ECHO_SP,
        subfield="SP",
        partition=PARTITION_SP_ECHO,
        system="VAC",
        family="VALVE",
        device="01",
        field="POSITION",
    ),
    _channel(
        ECHO_RB,
        subfield="RB",
        partition=PARTITION_SP_ECHO,
        system="VAC",
        family="VALVE",
        device="01",
        field="POSITION",
    ),
    _channel(
        TELEM_RB,
        subfield="RB",
        partition=PARTITION_STATIC_NOISY,
        system="VAC",
        family="GAUGE",
        device="01",
        field="PRESSURE",
    ),
]


class FakeDriver:
    """Duck-typed stand-in for the Channel Access driver.

    ``values`` is the served database: a one-shot read is answered from it,
    which is why a rejected write must leave it untouched rather than merely
    skip the monitor post. ``calls`` records every driver operation in order,
    so "committed before signalling completion" is an assertion about a
    sequence rather than about a final state.
    """

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        self.values: dict[str, Any] = dict(values or {})
        self.calls: list[tuple[str, str, Any]] = []

    def setParam(self, reason: str, value: Any) -> None:  # noqa: N802 - driver contract
        self.calls.append(("setParam", reason, value))
        self.values[reason] = value

    def getParam(self, reason: str) -> Any:  # noqa: N802 - driver contract
        return self.values[reason]

    def updatePV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.calls.append(("updatePV", reason, None))

    def callbackPV(self, reason: str) -> None:  # noqa: N802 - driver contract
        self.calls.append(("callbackPV", reason, None))

    def setParamStatus(  # noqa: N802 - driver contract
        self, reason: str, alarm: Any, severity: Any
    ) -> None:
        self.calls.append(("setParamStatus", reason, (alarm, severity)))

    def sequence(self, *reasons: str) -> list[tuple[str, str]]:
        """The operations touching ``reasons``, in order, without values."""
        wanted = set(reasons)
        return [(call, reason) for call, reason, _ in self.calls if reason in wanted]

    def posted(self, reason: str) -> int:
        return sum(1 for call, name, _ in self.calls if call == "updatePV" and name == reason)


class FakePvaChannels:
    """The PVA half of the served namespace: the model's own variables.

    Faithful on the point the PVA assertions rest on. A p4p ``post`` both
    replaces the value a one-shot ``get`` is answered with and delivers the
    monitor update, so unlike Channel Access there is nothing a write can
    record without publishing: "nothing moved on PVA, for either kind of
    reader" is the single assertion that nothing was posted.

    Only the addresses the model describes have a PVA channel at all. Every
    other co-hosted address -- which is most of the namespace, including
    every magnet's paired ``:RB`` -- is silently skipped, exactly as the
    runner's own publisher skips an address it finds no channel for.
    """

    def __init__(
        self,
        driver: FakeDriver,
        served: frozenset[str],
        *,
        failing: frozenset[str] = frozenset(),
    ) -> None:
        self.driver = driver
        self.served = served
        self.failing = failing
        self.values: dict[str, Any] = {}

    def post(self, address: str, value: Any) -> None:
        if address not in self.served:
            return
        if address in self.failing:
            raise RuntimeError(f"PVA transport failure publishing {address}")
        # Into the driver's journal, so that the order the two views of one
        # address move in is one sequence and not two.
        self.driver.calls.append(("post", address, value))
        self.values[address] = value

    def posted(self, address: str) -> int:
        return sum(1 for call, name, _ in self.driver.calls if call == "post" and name == address)


class RecordingCompletion:
    """A PVA put's completion callback, and what it was told.

    Every put owes exactly one of these calls, on every outcome: a put left
    uncompleted blocks the client that issued it until its own timeout.
    """

    def __init__(self) -> None:
        self.errors: list[str | None] = []

    def __call__(self, error: str | None) -> None:
        self.errors.append(error)


class RecordingModel(LUMEModel):
    """A model with the shape the runner drives, and nothing else.

    Reads report :data:`POISON` for every variable rather than the retained
    state: the run loop reads every variable back at the end of each cycle,
    and no served value may derive from that read. A poisoned read makes the
    difference visible instead of coincidental.
    """

    def __init__(self, *, refuse: frozenset[str] = frozenset()) -> None:
        self.refuse = refuse
        self.sets: list[dict[str, Any]] = []
        self.reads: list[list[str]] = []
        self.resets = 0
        self.state: dict[str, float] = {MAG_SP: BOOT_VALUES[MAG_SP], STUCK_SP: 0.0}
        self._vars: dict[str, ScalarVariable] = {
            MAG_SP: ScalarVariable(
                name=MAG_SP,
                default_value=BOOT_VALUES[MAG_SP],
                value_range=MAG_BAND,
                default_validation_config="none",
                read_only=False,
            ),
            STUCK_SP: ScalarVariable(
                name=STUCK_SP,
                default_value=0.0,
                value_range=MAG_BAND,
                default_validation_config="none",
                read_only=False,
            ),
            BPM_X: ScalarVariable(
                name=BPM_X,
                default_value=0.0,
                default_validation_config="none",
                read_only=True,
            ),
        }

    @property
    def supported_variables(self) -> dict[str, ScalarVariable]:
        return self._vars

    def _get(self, names: list[str]) -> dict[str, Any]:
        self.reads.append(list(names))
        return dict.fromkeys(names, POISON)

    def _set(self, values: dict[str, Any]) -> None:
        self.sets.append(dict(values))
        refused = sorted(set(values) & self.refuse)
        if refused:
            # What a lost closed orbit looks like from here: the model raises
            # and has restored itself before it does.
            raise RuntimeError(f"no stable closed orbit after writing {refused}")
        self.state.update(values)

    def reset(self) -> None:
        self.resets += 1


class RecordingBridge:
    """Stands in for ``PhysicsBridge.on_setpoint``.

    Reproduces the three things the real hook does, and records where it was
    called from: apply the calibration to the commanded current, write the
    result to the model, and push the recomputed reading onto the BPM's own
    record. The push goes through the record shim, exactly as the real
    bridge's does, so what lands on the BPM is a value the hook produced and
    not one the run loop read back.
    """

    CALIBRATION = 2.0

    def __init__(self, model: LUMEModel, records: ServingRecords) -> None:
        self.model = model
        self.records = records
        self.calls: list[tuple[str, Any]] = []
        self.threads: list[str] = []

    def on_setpoint(self, address: str, value: Any) -> None:
        self.calls.append((address, value))
        self.threads.append(threading.current_thread().name)
        self.model.set({address: value * self.CALIBRATION})
        self.records.all[BPM_X].set(BPM_READING)


class FakeRunLoop:
    """The serving package's run loop, for one queued item at a time.

    Reproduces the four steps the real loop takes around each item, in order:
    apply the item's values to the model with a single ``set``, read every
    variable back, offer those to the output pass, and call each completion
    callback with the error the cycle failed with -- or ``None``. Batching is
    not reproduced because the runner disables it: with the batching window
    at zero, one item is one cycle is one ``model.set``.
    """

    def __init__(self, model: LUMEModel) -> None:
        self.model = model
        self.queue: list[tuple[dict[str, Any], Any]] = []
        self.outputs: list[dict[str, Any]] = []

    def enqueue(self, values: dict[str, Any], done: Any = None, reset: bool = False) -> None:
        self.queue.append((values, done))

    def drain(self) -> None:
        while self.queue:
            values, done = self.queue.pop(0)
            error = None
            try:
                self.model.set({name: item["value"] for name, item in values.items()})
                # The output pass. The runner publishes nothing from it; the
                # values are kept here so a test can prove they never reached
                # a PV.
                self.outputs.append(self.model.get(list(self.model.supported_variables)))
            except Exception as exc:  # noqa: BLE001 - the loop reports, never raises
                error = str(exc)
            if done is not None:
                done(error)


def _records() -> ServingRecords:
    """The real serving database -- never a mock of it."""
    return build_serving_pvdb(
        CHANNELS,
        drive_limits=DRIVE_LIMITS,
        boot_values=BOOT_VALUES,
        async_setpoints=True,
    )


@pytest.fixture()
def records() -> ServingRecords:
    return _records()


@pytest.fixture()
def driver(records: ServingRecords) -> FakeDriver:
    """A driver seeded from the boot database, then attached, as the runner does."""
    drv = FakeDriver({address: spec["value"] for address, spec in records.pvdb.items()})
    records.attach_driver(drv)
    return drv


@pytest.fixture()
def model() -> RecordingModel:
    return RecordingModel()


@pytest.fixture()
def bridge(model: RecordingModel, records: ServingRecords) -> RecordingBridge:
    return RecordingBridge(model, records)


@pytest.fixture()
def loop(model: RecordingModel, bridge: RecordingBridge) -> FakeRunLoop:
    """The run loop, driving the model through the physics hook.

    Wrapped exactly as the runner wraps it, so the hook runs on whichever
    thread drains the loop.
    """
    return FakeRunLoop(
        SetpointRoutedModel(model, on_setpoint=bridge.on_setpoint, routed=PHYSICS_SETPOINTS)
    )


def _write_path(
    records: ServingRecords,
    loop: FakeRunLoop | None,
    *,
    stuck: frozenset[str] = frozenset(),
    pva: FakePvaChannels | None = None,
) -> CohostWritePath:
    return CohostWritePath(
        records,
        enqueue=loop.enqueue if loop is not None else None,
        physics_setpoints=physics_setpoint_addresses(records),
        stuck_setpoints=stuck,
        drive_limits=DRIVE_LIMITS,
        refusal_alarm=("WRITE_ALARM", "INVALID_ALARM"),
        pva_post=pva.post if pva is not None else None,
    )


@pytest.fixture()
def pva(driver: FakeDriver) -> FakePvaChannels:
    """The PVA channels the runner would serve: the model's variables."""
    return FakePvaChannels(driver, PVA_CHANNELS)


@pytest.fixture()
def path(records: ServingRecords, loop: FakeRunLoop, pva: FakePvaChannels) -> CohostWritePath:
    return _write_path(records, loop, stuck=frozenset({STUCK_SP}), pva=pva)


@dataclass
class Stack:
    """One complete serving arrangement, assembled the way the runner does.

    A test that needs a model which refuses, or a second arrangement to
    compare against, builds one of these rather than reaching around the
    fixtures: the driver, the PVA channels and the records are bound to each
    other, and two arrangements sharing a records object would leave the
    first one's driver detached.
    """

    records: ServingRecords
    model: RecordingModel
    bridge: RecordingBridge
    loop: FakeRunLoop
    driver: FakeDriver
    pva: FakePvaChannels
    path: CohostWritePath

    def journal(self) -> list[tuple[str, str, Any]]:
        """Every operation on either view, in order."""
        return list(self.driver.calls)


def _stack(
    *,
    refuse: frozenset[str] = frozenset(),
    stuck: frozenset[str] = frozenset(),
    served: frozenset[str] = PVA_CHANNELS,
    failing: frozenset[str] = frozenset(),
) -> Stack:
    records = _records()
    model = RecordingModel(refuse=refuse)
    bridge = RecordingBridge(model, records)
    loop = FakeRunLoop(
        SetpointRoutedModel(model, on_setpoint=bridge.on_setpoint, routed=PHYSICS_SETPOINTS)
    )
    driver = FakeDriver({address: spec["value"] for address, spec in records.pvdb.items()})
    records.attach_driver(driver)
    pva = FakePvaChannels(driver, served, failing=failing)
    return Stack(
        records=records,
        model=model,
        bridge=bridge,
        loop=loop,
        driver=driver,
        pva=pva,
        path=_write_path(records, loop, stuck=stuck, pva=pva),
    )


def _write(stack: Stack, address: str, value: Any) -> RecordingCompletion:
    """Write over Channel Access, and drain the loop.

    The recorder returned is always empty, because Channel Access completion
    carries nothing to record. That emptiness is the asymmetry between the
    transports, and it is asserted rather than glossed over.
    """
    stack.path.write(stack.driver, address, value)
    stack.loop.drain()
    return RecordingCompletion()


def _put(stack: Stack, address: str, value: Any) -> RecordingCompletion:
    """Put over PVA, and drain the loop."""
    done = RecordingCompletion()
    stack.path.put(stack.driver, address, value, done=done)
    stack.loop.drain()
    return done


#: The two transports, as ``(name, callable)``. Every property that must hold
#: identically on both is parametrised over this rather than written twice.
TRANSPORTS = [pytest.param(_write, id="ca-write"), pytest.param(_put, id="pva-put")]


class TestRouting:
    """Which behaviour each writable address gets, and which addresses are
    writable at all."""

    def test_physics_setpoints_are_the_writable_half_of_the_coupled_partition(
        self, records: ServingRecords
    ) -> None:
        assert physics_setpoint_addresses(records) == {MAG_SP, STUCK_SP}

    def test_every_setpoint_has_a_route_and_no_readback_has_one(
        self, path: CohostWritePath
    ) -> None:
        assert set(path.routes) == {MAG_SP, STUCK_SP, ECHO_SP}

    def test_modes(self, path: CohostWritePath) -> None:
        routes = path.routes
        assert routes[MAG_SP].mode == MODE_PHYSICS
        assert routes[ECHO_SP].mode == MODE_ECHO
        # Stuck: the fault replaces the behaviour, it does not disable the PV.
        assert routes[STUCK_SP].mode == MODE_LATCH
        assert routes[STUCK_SP].readback is None

    def test_no_hook_means_no_physics_and_no_echo(self, records: ServingRecords) -> None:
        """Parity with a process that has no lattice: a coupled setpoint
        records what was written and propagates nothing."""
        path = _write_path(records, None)
        assert path.routes[MAG_SP].mode == MODE_LATCH
        assert path.routes[MAG_SP].readback is None
        # A setpoint with no physics behind it still echoes if its echo is a
        # plain value copy.
        assert path.routes[ECHO_SP].mode == MODE_ECHO

    def test_physics_setpoint_must_be_asynchronous(self, loop: FakeRunLoop) -> None:
        """A synchronous PV would tell the client the write completed before
        the solve had started."""
        synchronous = build_serving_pvdb(CHANNELS, drive_limits=DRIVE_LIMITS)
        with pytest.raises(ValueError, match="asynchronous"):
            _write_path(synchronous, loop)

    def test_write_path_holds_no_model(self, path: CohostWritePath) -> None:
        """The server thread cannot reach the model even by accident: the
        object it calls has no reference to one."""
        assert not any(isinstance(value, LUMEModel) for value in vars(path).values())


class TestAcceptedWrite:
    """A write the model takes."""

    def test_clamped_value_reaches_the_hook(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, bridge: RecordingBridge
    ) -> None:
        """Clamp first, then physics: the model is never offered a value
        outside the band, so the value it accepts is the value echoed."""
        path.write(driver, MAG_SP, 700.0)
        loop.drain()
        assert bridge.calls == [(MAG_SP, MAG_BAND[1])]

    def test_setpoint_and_echo_both_carry_the_post_clamp_value(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, MAG_SP, 700.0)
        loop.drain()
        assert driver.values[MAG_SP] == MAG_BAND[1]
        assert driver.values[MAG_RB] == MAG_BAND[1]
        # Bit-exact, not merely close: a settle poll compares to 1e-9.
        assert driver.values[MAG_RB] == driver.values[MAG_SP]

    def test_in_band_value_passes_through_untouched(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.values[MAG_SP] == 3.25
        assert driver.values[MAG_RB] == 3.25

    def test_both_addresses_post_a_monitor_event(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.posted(MAG_SP) == 1
        assert driver.posted(MAG_RB) == 1

    def test_values_are_committed_before_completion_is_signalled(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        """A client unblocking on put-completion reads the served database
        immediately; anything committed after the callback would be read as
        the value the write replaced."""
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.sequence(MAG_SP, MAG_RB) == [
            ("setParam", MAG_SP),
            ("updatePV", MAG_SP),
            ("post", MAG_SP),
            ("setParam", MAG_RB),
            ("updatePV", MAG_RB),
            ("callbackPV", MAG_SP),
        ]

    def test_bpm_reading_comes_from_the_hook(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.values[BPM_X] == BPM_READING


class TestRejectedWrite:
    """A write the model refuses. Nothing may move, for any reader."""

    @pytest.fixture()
    def refusing(self) -> Stack:
        return _stack(refuse=frozenset({MAG_SP}))

    def test_one_shot_reader_sees_no_movement(self, refusing: Stack) -> None:
        """The served database is what a fresh read is answered from."""
        _write(refusing, MAG_SP, 3.25)
        assert refusing.driver.values[MAG_SP] == BOOT_VALUES[MAG_SP]
        assert refusing.driver.values[MAG_RB] == BOOT_VALUES[MAG_RB]

    def test_monitoring_reader_sees_no_movement(self, refusing: Stack) -> None:
        """A monitoring client is served what is posted, and nothing is."""
        _write(refusing, MAG_SP, 3.25)
        assert [call for call in refusing.driver.calls if call[0] == "setParam"] == []
        assert refusing.driver.posted(MAG_RB) == 0

    def test_the_other_view_sees_no_movement_either(self, refusing: Stack) -> None:
        """One post is all a PVA reader of either kind would need to see."""
        _write(refusing, MAG_SP, 3.25)
        assert refusing.pva.values == {}

    def test_completion_still_fires(self, refusing: Stack) -> None:
        """Withholding it would postpone every later write to this setpoint,
        for the life of the process."""
        _write(refusing, MAG_SP, 3.25)
        assert ("callbackPV", MAG_SP, None) in refusing.driver.calls

    def test_refusal_raises_an_alarm_after_the_value_is_left_alone(self, refusing: Stack) -> None:
        """Put-completion can only report success, so the alarm is the only
        signal a refusal can leave."""
        _write(refusing, MAG_SP, 3.25)
        assert refusing.driver.sequence(MAG_SP) == [
            ("setParamStatus", MAG_SP),
            ("updatePV", MAG_SP),
            ("callbackPV", MAG_SP),
        ]

    def test_a_later_accepted_write_still_lands(self, refusing: Stack) -> None:
        """The refusal leaves nothing behind that blocks the next write."""
        _write(refusing, MAG_SP, 3.25)
        _write(refusing, ECHO_SP, 7.0)
        assert refusing.driver.values[ECHO_SP] == 7.0
        assert refusing.driver.values[ECHO_RB] == 7.0


class TestModelIsNeverTheSource:
    """No served value derives from reading the model back."""

    def test_the_loop_does_read_every_variable(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, model: RecordingModel
    ) -> None:
        """Without this the next test would pass vacuously: the poisoned
        values have to be available for their absence to mean anything."""
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert loop.outputs and all(value == POISON for value in loop.outputs[0].values())

    def test_no_poisoned_value_reaches_a_pv(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert POISON not in driver.values.values()

    def test_the_bpm_carries_the_pushed_reading_not_the_read_back_one(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.values[BPM_X] == BPM_READING


class TestStuckSetpoint:
    """An apply fault freezes a device's readback -- honestly, for everyone."""

    def test_setpoint_still_records_the_written_value(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, STUCK_SP, 3.25)
        loop.drain()
        assert driver.values[STUCK_SP] == 3.25

    def test_readback_never_moves(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop
    ) -> None:
        path.write(driver, STUCK_SP, 3.25)
        loop.drain()
        assert driver.values[STUCK_RB] == BOOT_VALUES[STUCK_RB]
        assert driver.posted(STUCK_RB) == 0

    def test_the_machine_never_moves_either(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, bridge: RecordingBridge
    ) -> None:
        path.write(driver, STUCK_SP, 3.25)
        loop.drain()
        assert bridge.calls == []

    def test_the_write_still_completes_once_its_value_is_committed(
        self, path: CohostWritePath, driver: FakeDriver
    ) -> None:
        assert path.write(driver, STUCK_SP, 3.25) is True
        assert driver.sequence(STUCK_SP, STUCK_RB) == [
            ("setParam", STUCK_SP),
            ("updatePV", STUCK_SP),
            ("post", STUCK_SP),
            ("callbackPV", STUCK_SP),
        ]


class TestNonPhysicsWrites:
    """Setpoints with no model behind them, and channels with no write at all."""

    def test_echo_pair_follows_immediately_without_the_model(
        self, path: CohostWritePath, driver: FakeDriver, bridge: RecordingBridge, loop: FakeRunLoop
    ) -> None:
        assert path.write(driver, ECHO_SP, 7.0) is True
        assert driver.values[ECHO_SP] == 7.0
        assert driver.values[ECHO_RB] == 7.0
        assert loop.queue == []
        assert bridge.calls == []

    def test_echo_pair_is_committed_before_completion_is_signalled(
        self, path: CohostWritePath, driver: FakeDriver
    ) -> None:
        """Same ordering obligation as a write that waits for the model: the
        client is unblocked by the completion and reads immediately after."""
        path.write(driver, ECHO_SP, 7.0)
        assert driver.sequence(ECHO_SP, ECHO_RB) == [
            ("setParam", ECHO_SP),
            ("updatePV", ECHO_SP),
            ("setParam", ECHO_RB),
            ("updatePV", ECHO_RB),
            ("callbackPV", ECHO_SP),
        ]

    def test_echo_write_is_clamped_too(self, path: CohostWritePath, driver: FakeDriver) -> None:
        path.write(driver, ECHO_SP, 400.0)
        assert driver.values[ECHO_SP] == 100.0
        assert driver.values[ECHO_RB] == 100.0

    def test_telemetry_channel_refuses_the_write_and_does_not_move(
        self, path: CohostWritePath, driver: FakeDriver
    ) -> None:
        boot = driver.values[TELEM_RB]
        assert path.write(driver, TELEM_RB, 42.0) is False
        assert driver.values[TELEM_RB] == boot
        assert driver.calls == []

    def test_readback_of_a_setpoint_pair_is_not_writable(
        self, path: CohostWritePath, driver: FakeDriver
    ) -> None:
        assert path.write(driver, MAG_RB, 42.0) is False
        assert driver.values[MAG_RB] == BOOT_VALUES[MAG_RB]


class TestBothViewsOfOneChannel:
    """An address served twice carries one value, not two that drift."""

    def test_a_channel_access_write_moves_the_pva_view_too(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, pva: FakePvaChannels
    ) -> None:
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert pva.values[MAG_SP] == 3.25

    def test_the_two_views_carry_the_same_value_bit_for_bit(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, pva: FakePvaChannels
    ) -> None:
        """A settle poll compares to 1e-9; a re-derived value would not do."""
        path.write(driver, MAG_SP, 700.0)
        loop.drain()
        assert pva.values[MAG_SP] == driver.values[MAG_SP] == MAG_BAND[1]

    def test_the_pva_view_carries_the_post_clamp_value(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, pva: FakePvaChannels
    ) -> None:
        """The clamp is upstream of the split, so neither view can carry the
        requested value while the other carries the accepted one."""
        path.write(driver, MAG_SP, -700.0)
        loop.drain()
        assert pva.values[MAG_SP] == MAG_BAND[0]

    def test_an_address_with_one_view_is_published_once_and_does_not_raise(
        self, path: CohostWritePath, driver: FakeDriver
    ) -> None:
        """Most of the co-hosted namespace has no PVA channel at all."""
        assert path.write(driver, ECHO_SP, 7.0) is True
        assert driver.values[ECHO_SP] == 7.0
        assert ECHO_SP not in pva_addresses(driver)

    def test_a_readback_with_a_pva_view_is_published_on_both(self) -> None:
        """Nothing in the write path knows which addresses are doubly served;
        the echo goes wherever the setpoint does."""
        stack = _stack(served=frozenset({MAG_SP, MAG_RB}))
        _write(stack, MAG_SP, 3.25)
        assert stack.pva.values == {MAG_SP: 3.25, MAG_RB: 3.25}

    def test_the_stuck_setpoint_latches_on_both_views(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, pva: FakePvaChannels
    ) -> None:
        """The fault freezes the readback, not the record of what was asked
        for -- and it freezes it identically for every reader on either
        transport."""
        path.write(driver, STUCK_SP, 3.25)
        loop.drain()
        assert pva.values[STUCK_SP] == 3.25
        assert driver.values[STUCK_SP] == 3.25
        assert driver.values[STUCK_RB] == BOOT_VALUES[STUCK_RB]

    def test_no_value_read_back_from_the_model_reaches_the_pva_view_either(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, pva: FakePvaChannels
    ) -> None:
        """The output pass offers every variable's read-back value on every
        cycle; the second view is no more allowed to publish one than the
        first is."""
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert loop.outputs and all(value == POISON for value in loop.outputs[0].values())
        assert POISON not in pva.values.values()


def pva_addresses(driver: FakeDriver) -> set[str]:
    """The addresses posted on the PVA view, out of the shared journal."""
    return {name for call, name, _ in driver.calls if call == "post"}


class TestPvaPut:
    """A put takes the write path a Channel Access write takes."""

    def test_the_clamped_value_reaches_the_hook(self) -> None:
        stack = _stack()
        _put(stack, MAG_SP, 700.0)
        assert stack.bridge.calls == [(MAG_SP, MAG_BAND[1])]

    def test_it_moves_the_channel_access_setpoint_and_its_echo(self) -> None:
        stack = _stack()
        _put(stack, MAG_SP, 3.25)
        assert stack.driver.values[MAG_SP] == 3.25
        assert stack.driver.values[MAG_RB] == 3.25

    def test_it_posts_a_monitor_event_on_the_channel_access_view(self) -> None:
        """A CA client monitoring the setpoint sees a put made on the other
        transport, which is the whole point of routing it here."""
        stack = _stack()
        _put(stack, MAG_SP, 3.25)
        assert stack.driver.posted(MAG_SP) == 1
        assert stack.driver.posted(MAG_RB) == 1

    def test_it_echoes_on_its_own_view(self) -> None:
        stack = _stack()
        _put(stack, MAG_SP, 3.25)
        assert stack.pva.values[MAG_SP] == 3.25

    def test_the_bpm_reading_still_comes_from_the_hook(self) -> None:
        stack = _stack()
        _put(stack, MAG_SP, 3.25)
        assert stack.driver.values[BPM_X] == BPM_READING

    def test_an_echo_setpoint_moves_its_pair_without_the_model(self) -> None:
        stack = _stack()
        done = _put(stack, ECHO_SP, 7.0)
        assert stack.driver.values[ECHO_SP] == 7.0
        assert stack.driver.values[ECHO_RB] == 7.0
        assert stack.bridge.calls == []
        assert done.errors == [None]

    def test_a_stuck_setpoint_latches(self) -> None:
        stack = _stack(stuck=frozenset({STUCK_SP}))
        done = _put(stack, STUCK_SP, 3.25)
        assert stack.driver.values[STUCK_SP] == 3.25
        assert stack.driver.values[STUCK_RB] == BOOT_VALUES[STUCK_RB]
        assert stack.bridge.calls == []
        assert done.errors == [None]

    def test_a_put_to_a_channel_that_is_not_a_setpoint_is_refused(self) -> None:
        """Both transports agree on what is writable, not only on values."""
        stack = _stack()
        done = RecordingCompletion()
        assert stack.path.put(stack.driver, TELEM_RB, 42.0, done=done) is False
        assert done.errors == [NOT_WRITABLE]
        assert stack.driver.calls == []
        assert stack.pva.values == {}

    def test_a_put_to_a_readback_is_refused(self) -> None:
        stack = _stack()
        done = RecordingCompletion()
        assert stack.path.put(stack.driver, MAG_RB, 42.0, done=done) is False
        assert done.errors == [NOT_WRITABLE]
        assert stack.driver.values[MAG_RB] == BOOT_VALUES[MAG_RB]

    def test_values_are_committed_before_the_put_is_completed(self) -> None:
        """A PVA client unblocks on completion and reads immediately, just as
        a Channel Access one does."""
        stack = _stack()
        order: list[str] = []
        stack.path.put(
            stack.driver,
            MAG_SP,
            3.25,
            done=lambda error: order.append(f"done:{error}"),
        )
        stack.loop.drain()
        order = [
            *(f"{call}:{reason}" for call, reason, _ in stack.driver.calls if reason == MAG_SP),
            *order,
        ]
        assert order == [
            f"setParam:{MAG_SP}",
            f"updatePV:{MAG_SP}",
            f"post:{MAG_SP}",
            "done:None",
        ]

    def test_the_putting_thread_never_touches_the_model(self) -> None:
        """A p4p worker thread is no more allowed to reach the model than the
        Channel Access server thread is."""
        stack = _stack()
        done = RecordingCompletion()
        worker = threading.Thread(
            target=stack.path.put,
            args=(stack.driver, MAG_SP, 3.25),
            kwargs={"done": done},
            name="pva-worker",
        )
        worker.start()
        worker.join()

        assert stack.bridge.calls == []
        assert stack.driver.values[MAG_SP] == BOOT_VALUES[MAG_SP]
        assert done.errors == []

        stack.loop.drain()
        assert stack.bridge.threads == [threading.current_thread().name]
        assert done.errors == [None]


class TestTransportSymmetry:
    """What a write does may not depend on which transport it arrived on."""

    @pytest.mark.parametrize("send", TRANSPORTS)
    def test_an_accepted_write_moves_both_views(self, send) -> None:  # noqa: ANN001
        stack = _stack()
        send(stack, MAG_SP, 3.25)
        assert stack.driver.values[MAG_SP] == 3.25
        assert stack.driver.values[MAG_RB] == 3.25
        assert stack.pva.values[MAG_SP] == 3.25

    @pytest.mark.parametrize("send", TRANSPORTS)
    def test_a_refused_write_leaves_the_one_shot_readers_where_they_were(
        self,
        send,  # noqa: ANN001
    ) -> None:
        stack = _stack(refuse=frozenset({MAG_SP}))
        send(stack, MAG_SP, 3.25)
        assert stack.driver.values[MAG_SP] == BOOT_VALUES[MAG_SP]
        assert stack.driver.values[MAG_RB] == BOOT_VALUES[MAG_RB]
        assert stack.pva.values == {}

    @pytest.mark.parametrize("send", TRANSPORTS)
    def test_a_refused_write_posts_nothing_to_a_monitoring_reader(
        self,
        send,  # noqa: ANN001
    ) -> None:
        """On Channel Access that is two facts -- nothing recorded, nothing
        posted -- because a value can be recorded without being posted. On
        PVA the post is the record, so it is one."""
        stack = _stack(refuse=frozenset({MAG_SP}))
        send(stack, MAG_SP, 3.25)
        assert [call for call in stack.driver.calls if call[0] == "setParam"] == []
        assert stack.driver.posted(MAG_SP) == 1  # the alarm transition, no value
        assert stack.driver.posted(MAG_RB) == 0
        assert pva_addresses(stack.driver) == set()

    @pytest.mark.parametrize("send", TRANSPORTS)
    def test_a_refused_write_raises_the_alarm_whichever_side_it_came_from(
        self,
        send,  # noqa: ANN001
    ) -> None:
        """The alarm is the channel's condition, not one client's error
        report: a refusal that arrived on PVA is still a refused write to
        that setpoint, and a Channel Access client watching it is owed the
        same signal it would get from a refusal of its own."""
        stack = _stack(refuse=frozenset({MAG_SP}))
        send(stack, MAG_SP, 3.25)
        assert ("setParamStatus", MAG_SP, ("WRITE_ALARM", "INVALID_ALARM")) in stack.driver.calls

    @pytest.mark.parametrize("send", TRANSPORTS)
    def test_the_model_is_offered_the_same_value(self, send) -> None:  # noqa: ANN001
        stack = _stack()
        send(stack, MAG_SP, 700.0)
        assert stack.bridge.calls == [(MAG_SP, MAG_BAND[1])]

    def test_the_two_transports_leave_identical_journals_when_accepted(self) -> None:
        """Everything either transport does to the served channels, in order,
        is the same sequence -- the only difference permitted is which client
        gets told the write finished."""
        assert _journal(_write) == _journal(_put)

    def test_the_two_transports_leave_identical_journals_when_refused(self) -> None:
        assert _journal(_write, refuse=True) == _journal(_put, refuse=True)

    def test_the_journal_comparison_is_not_vacuous(self) -> None:
        """An accepted write and a refused one differ, so the equalities
        above are asserting something."""
        assert _journal(_write) != _journal(_write, refuse=True)
        assert _journal(_write) != []


def _journal(send, *, refuse: bool = False) -> list[tuple[str, str, Any]]:
    """Everything one transport does to the served channels, in order.

    Channel Access completion is dropped, because it is the one operation
    that belongs to a transport rather than to the channel: a PVA put
    completes through its own callback and must not end an asynchronous
    Channel Access write that no client ever started.
    """
    stack = _stack(refuse=frozenset({MAG_SP}) if refuse else frozenset())
    send(stack, MAG_SP, 3.25)
    return [call for call in stack.journal() if call[0] != "callbackPV"]


class TestPutCompletion:
    """How each transport tells a client its write finished.

    The one asymmetry the design keeps, because it is a real difference in
    what the two protocols can express -- and the place where assuming
    symmetry would freeze a setpoint.
    """

    def test_channel_access_completion_carries_no_status(self) -> None:
        """It can only ever report success, which is why a refusal needs the
        alarm to leave any trace at all."""
        stack = _stack(refuse=frozenset({MAG_SP}))
        assert _write(stack, MAG_SP, 3.25).errors == []
        assert ("callbackPV", MAG_SP, None) in stack.driver.calls

    def test_a_pva_put_is_completed_with_the_models_own_reason(self) -> None:
        stack = _stack(refuse=frozenset({MAG_SP}))
        done = _put(stack, MAG_SP, 3.25)
        assert len(done.errors) == 1
        assert "closed orbit" in done.errors[0]

    def test_a_pva_put_that_lands_is_completed_with_no_error(self) -> None:
        stack = _stack()
        assert _put(stack, MAG_SP, 3.25).errors == [None]

    @pytest.mark.parametrize(
        ("address", "stuck"),
        [
            (MAG_SP, frozenset()),
            (ECHO_SP, frozenset()),
            (STUCK_SP, frozenset({STUCK_SP})),
        ],
    )
    def test_every_accepted_put_is_completed_exactly_once(
        self, address: str, stuck: frozenset[str]
    ) -> None:
        """Twice is a protocol error; never blocks the client until its own
        timeout expires."""
        stack = _stack(stuck=stuck)
        assert len(_put(stack, address, 3.25).errors) == 1

    def test_a_refused_put_is_completed_exactly_once(self) -> None:
        stack = _stack(refuse=frozenset({MAG_SP}))
        assert len(_put(stack, MAG_SP, 3.25).errors) == 1

    def test_a_put_never_ends_a_channel_access_asynchronous_write(self) -> None:
        """There is none in flight. Ending one that was never started would
        complete some other client's write."""
        stack = _stack()
        _put(stack, MAG_SP, 3.25)
        assert [call for call in stack.driver.calls if call[0] == "callbackPV"] == []

    def test_a_refused_put_never_ends_one_either(self) -> None:
        stack = _stack(refuse=frozenset({MAG_SP}))
        _put(stack, MAG_SP, 3.25)
        assert [call for call in stack.driver.calls if call[0] == "callbackPV"] == []

    def test_a_channel_access_write_completes_on_channel_access_alone(self) -> None:
        """Symmetrically: a write that arrived on Channel Access owes nothing
        to a PVA operation, because there is none."""
        stack = _stack()
        assert _write(stack, MAG_SP, 3.25).errors == []
        assert ("callbackPV", MAG_SP, None) in stack.driver.calls


class TestPvaPublishFailure:
    """The second view may fail; the first view's client may not pay for it."""

    def test_the_channel_access_write_still_completes(self) -> None:
        """A `callbackPV` that never fires postpones every later write to
        that setpoint for the life of the process, so a failure publishing
        the other view must not be allowed to skip it."""
        stack = _stack(failing=frozenset({MAG_SP}))
        _write(stack, MAG_SP, 3.25)
        assert ("callbackPV", MAG_SP, None) in stack.driver.calls

    def test_the_authoritative_view_still_moves(self) -> None:
        stack = _stack(failing=frozenset({MAG_SP}))
        _write(stack, MAG_SP, 3.25)
        assert stack.driver.values[MAG_SP] == 3.25
        assert stack.driver.values[MAG_RB] == 3.25

    def test_the_put_that_provoked_it_is_still_completed(self) -> None:
        stack = _stack(failing=frozenset({MAG_SP}))
        assert _put(stack, MAG_SP, 3.25).errors == [None]


class TestSingleClamp:
    """One value, one enforcement of the drive band, on either transport."""

    @pytest.fixture()
    def clamps(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, Any]]:
        recorded: list[tuple[Any, Any]] = []
        real = write_path_module.clamp_into

        def spy(value: Any, limits: Any) -> Any:
            recorded.append((value, limits))
            return real(value, limits)

        monkeypatch.setattr(write_path_module, "clamp_into", spy)
        return recorded

    @pytest.mark.parametrize("send", TRANSPORTS)
    def test_a_write_is_clamped_exactly_once(
        self,
        send,  # noqa: ANN001
        clamps: list[tuple[Any, Any]],
    ) -> None:
        """Identical bands make a second clamp invisible in the value, so it
        is counted instead."""
        send(_stack(), MAG_SP, 700.0)
        assert clamps == [(700.0, MAG_BAND)]

    def test_the_band_that_is_enforced_is_the_manifests(
        self, clamps: list[tuple[Any, Any]]
    ) -> None:
        """The drive limits the co-hosted database publishes, which is what
        the write path is given -- not a band read off a model variable."""
        _put(_stack(), ECHO_SP, 400.0)
        assert clamps == [(400.0, DRIVE_LIMITS[ECHO_SP])]


class TestThreadOfExecution:
    """Every model access happens on the thread that drains the loop."""

    def test_the_writing_thread_never_touches_the_model(
        self, path: CohostWritePath, driver: FakeDriver, loop: FakeRunLoop, bridge: RecordingBridge
    ) -> None:
        server_thread = threading.Thread(
            target=path.write, args=(driver, MAG_SP, 3.25), name="ca-server"
        )
        server_thread.start()
        server_thread.join()

        # Returning from the write proves nothing on its own; what proves the
        # separation is that the model has not been touched yet.
        assert bridge.calls == []
        assert driver.values[MAG_SP] == BOOT_VALUES[MAG_SP]

        loop.drain()
        assert bridge.threads == [threading.current_thread().name]


class TestSetpointRoutedModel:
    """The wrapper that puts the physics hook on the run loop's thread."""

    def test_routed_write_goes_to_the_hook_and_not_to_the_model(
        self, model: RecordingModel, bridge: RecordingBridge
    ) -> None:
        routed = SetpointRoutedModel(
            model, on_setpoint=bridge.on_setpoint, routed=frozenset({MAG_SP})
        )
        routed.set({MAG_SP: 1.0})
        assert bridge.calls == [(MAG_SP, 1.0)]
        # One set, and it is the hook's own -- the wrapper does not also
        # write the value, which would apply it twice.
        assert model.sets == [{MAG_SP: 1.0 * RecordingBridge.CALIBRATION}]

    def test_unrouted_write_reaches_the_model_unchanged(
        self, model: RecordingModel, bridge: RecordingBridge
    ) -> None:
        routed = SetpointRoutedModel(
            model, on_setpoint=bridge.on_setpoint, routed=frozenset({MAG_SP})
        )
        routed.set({STUCK_SP: 4.0})
        assert model.sets == [{STUCK_SP: 4.0}]
        assert bridge.calls == []

    def test_empty_batch_still_reaches_the_model(
        self, model: RecordingModel, bridge: RecordingBridge
    ) -> None:
        """The loop's startup cycle carries no values; the model treats that
        as "re-solve and refresh", and dropping it would change boot."""
        routed = SetpointRoutedModel(
            model, on_setpoint=bridge.on_setpoint, routed=frozenset({MAG_SP})
        )
        routed.set({})
        assert model.sets == [{}]

    def test_refusal_propagates(self, records: ServingRecords) -> None:
        model = RecordingModel(refuse=frozenset({MAG_SP}))
        bridge = RecordingBridge(model, records)
        routed = SetpointRoutedModel(
            model, on_setpoint=bridge.on_setpoint, routed=frozenset({MAG_SP})
        )
        with pytest.raises(RuntimeError, match="closed orbit"):
            routed.set({MAG_SP: 1.0})

    def test_reads_and_variables_delegate(
        self, model: RecordingModel, bridge: RecordingBridge
    ) -> None:
        routed = SetpointRoutedModel(
            model, on_setpoint=bridge.on_setpoint, routed=frozenset({MAG_SP})
        )
        assert routed.supported_variables is model.supported_variables
        assert routed.get([BPM_X]) == {BPM_X: POISON}
        routed.reset()
        assert model.resets == 1


class TestClamp:
    """The only enforcement of a drive band that exists anywhere."""

    def test_above_and_below(self) -> None:
        assert clamp_into(700.0, MAG_BAND) == 10.0
        assert clamp_into(-700.0, MAG_BAND) == -10.0

    def test_unbanded_and_non_numeric_pass_through(self) -> None:
        assert clamp_into(700.0, None) == 700.0
        assert clamp_into("open", MAG_BAND) == "open"
        assert clamp_into(True, MAG_BAND) is True


class TestConfigPolicy:
    """The runner configuration the write path depends on."""

    def test_values(self) -> None:
        assert RUNNER_CONFIG_POLICY == {
            "update_rate": 0.0,
            "echo_unconfirmed_writes": False,
            "alarm_on_refused_write": True,
            "clamp_writes": False,
            "control_pvs": False,
        }

    def test_the_runners_own_clamp_is_off(self) -> None:
        """Because this module's is on, for both transports. The runner's
        would enforce the same band on the same PVA puts, and two
        enforcement points on one value is a second thing to keep in step."""
        assert RUNNER_CONFIG_POLICY["clamp_writes"] is False


class TestRealNamespace:
    """The counts and overlaps the co-hosting arrangement rests on.

    Built from the real manifest and the real variable catalog, because the
    decision they justify -- that the model's variables get no Channel Access
    PV of their own -- is only correct if the manifest already describes
    every one of them.
    """

    @pytest.fixture(scope="class")
    def built(self) -> tuple[ServingRecords, dict[str, Any]]:
        from osprey.services.virtual_accelerator.manifest import build_manifest
        from osprey.services.virtual_accelerator.model.catalog import build_variable_catalog

        channels = build_manifest()["channels"]
        return build_serving_pvdb(channels, async_setpoints=True), build_variable_catalog(channels)

    def test_counts(self, built) -> None:  # noqa: ANN001
        records, catalog = built
        assert len(records.pvdb) == 2908
        assert len(records.setpoint_readbacks) == 396
        assert len(physics_setpoint_addresses(records)) == 348
        assert len(catalog) == 492

    def test_every_model_variable_is_already_a_co_hosted_channel(self, built) -> None:  # noqa: ANN001
        """Which is why the base class must not serve them on Channel Access
        as well: the database merge refuses a duplicate name outright."""
        records, catalog = built
        assert set(catalog) <= set(records.pvdb)

    def test_the_physics_setpoints_are_exactly_the_writable_variables(
        self,
        built,  # noqa: ANN001
    ) -> None:
        """Nothing is routed to the model that the model cannot take, and
        nothing writable is left un-routed."""
        records, catalog = built
        writable = {name for name, var in catalog.items() if not var.read_only}
        assert physics_setpoint_addresses(records) == writable


class TestRunnerShape:
    """What the subclass does, asserted against its syntax tree.

    The subclass itself cannot be imported here: it subclasses a class that
    imports the compiled Channel Access server extension, which this host
    has no working build of. Its behaviour is proven against the deployed
    container. What is checked here is the handful of properties whose
    violation would be silent in that container -- publishing a value read
    back from the model, or attaching the records at the wrong moment -- so
    that they fail in a unit run instead.
    """

    @pytest.fixture(scope="class")
    def tree(self) -> ast.Module:
        source = (
            Path(__file__).resolve().parents[2]
            / "src/osprey/services/virtual_accelerator/serving/runner.py"
        )
        return ast.parse(source.read_text())

    def _class(self, tree: ast.Module, name: str) -> ast.ClassDef:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise AssertionError(f"class {name!r} not found")

    def _method(self, tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
        for node in self._class(tree, cls).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"method {cls}.{name} not found")

    def test_output_pass_publishes_nothing(self, tree: ast.Module) -> None:
        """Its whole body is its docstring: there is no path from a model
        read to a served value."""
        body = self._method(tree, "CohostRunner", "_post_outputs").body
        assert len(body) == 1
        assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)

    def test_run_loop_rollback_is_disabled(self, tree: ast.Module) -> None:
        body = self._method(tree, "CohostRunner", "_reset_to_cached_state").body
        assert len(body) == 1
        assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)

    def test_records_are_attached_only_after_the_base_constructor(self, tree: ast.Module) -> None:
        """Before it there is no driver; a record attached earlier writes into
        a spec, and after the server has created the PVs a spec write reaches
        nobody."""
        init = self._method(tree, "CohostRunner", "__init__")
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert "super().__init__" in calls
        assert "records.attach_driver" in calls
        assert calls.index("super().__init__") < calls.index("records.attach_driver")

    def test_the_driver_never_delegates_to_the_stock_write_path(self, tree: ast.Module) -> None:
        """The stock path enqueues a bare model write, bypassing the physics
        hook a setpoint write has to run through."""
        write = self._method(tree, "CohostDriver", "write")
        assert "super" not in ast.unparse(write)

    def test_the_driver_class_is_installed(self, tree: ast.Module) -> None:
        assigns = [
            ast.unparse(node)
            for node in self._class(tree, "CohostRunner").body
            if isinstance(node, ast.Assign)
        ]
        assert "ca_driver_cls = CohostDriver" in assigns

    def test_the_configuration_policy_is_applied(self, tree: ast.Module) -> None:
        init = self._method(tree, "CohostRunner", "__init__")
        assert "config.update(RUNNER_CONFIG_POLICY)" in ast.unparse(init)

    def test_model_variables_get_no_channel_access_pv(self, tree: ast.Module) -> None:
        """They are already in the manifest, which contributes them itself --
        with the record-type mapping and the boot values a spec derived from
        the variable would not carry."""
        add_pv = self._method(tree, "CohostRunner", "_add_pv")
        unparsed = ast.unparse(add_pv)
        assert "super()._add_pv" in unparsed
        # Suppressed by scoping the flag the base implementation reads, and
        # restored in a finally so a raising variable cannot leave the runner
        # serving the rest of the model on Channel Access after all.
        assert "self.supports_ca" in unparsed
        assert "False" in unparsed
        assert any(isinstance(node, ast.Try) for node in add_pv.body)

    def test_the_whole_database_is_contributed(self, tree: ast.Module) -> None:
        extend = ast.unparse(self._method(tree, "CohostRunner", "_extend_pvdb"))
        assert "self._records.pvdb" in extend

    def test_the_pva_publisher_is_wired_into_the_write_path(self, tree: ast.Module) -> None:
        """Without this the write path publishes on Channel Access alone and
        the two views of a setpoint drift apart on every write."""
        init = ast.unparse(self._method(tree, "CohostRunner", "__init__"))
        assert "pva_post=self._post_pva" in init

    def test_the_pva_publisher_is_wired_into_the_record_shims(self, tree: ast.Module) -> None:
        """The sibling of the check above, and it needs to be its own check:
        the write path publishes only what a client writes. Every *reading*
        reaches its PV through the record shim instead, so a shim attached
        without the publisher leaves all 144 BPM readings frozen on PVA while
        setpoints track -- which looks correct, and is the worse failure.

        Asserted against the call node rather than by matching text in the
        constructor's source, so it cannot be satisfied by the argument
        appearing anywhere else in it.
        """
        init = self._method(tree, "CohostRunner", "__init__")
        attaches = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "attach_driver"
        ]

        assert len(attaches) == 1, "the records are attached exactly once"
        published = [
            ast.unparse(keyword.value)
            for keyword in attaches[0].keywords
            if keyword.arg == "pva_post"
        ]
        assert published == ["self._post_pva"]

    def test_pva_puts_are_routed_through_the_write_path(self, tree: ast.Module) -> None:
        """The stock handler would enqueue a bare model write for this one
        variable: no physics hook, no clamp, and no Channel Access view."""
        add_pv = ast.unparse(self._method(tree, "CohostRunner", "_add_pv"))
        assert "channel.put" in add_pv
        assert "self._put" in add_pv
        assert "self.write_path.put" in ast.unparse(self._method(tree, "CohostRunner", "_put"))

    def test_a_read_only_variable_keeps_the_stock_handler(self, tree: ast.Module) -> None:
        """It has no route, and refusing a put to it is what a Channel Access
        write to the same address already gets."""
        add_pv = ast.unparse(self._method(tree, "CohostRunner", "_add_pv"))
        assert "if ro or channel is None" in add_pv

    def test_a_put_arriving_before_the_driver_exists_is_refused(self, tree: ast.Module) -> None:
        """The PVA server is listening from the moment it is created, which
        is before the driver every published value is committed through."""
        put = ast.unparse(self._method(tree, "CohostRunner", "_put"))
        assert "self.ca_driver" in put
        assert "NOT_READY" in put

    def test_the_runner_never_clamps_a_second_time(self, tree: ast.Module) -> None:
        """The base class's clamp is configured off and never called: the
        write path's is the only enforcement of a drive band there is."""
        assert "_clamp_write" not in ast.unparse(tree)

    def test_publishing_on_pva_never_reads_the_model(self, tree: ast.Module) -> None:
        """It publishes the value the client wrote and the model accepted,
        packed into the variable's structure -- never a value read back."""
        post = ast.unparse(self._method(tree, "CohostRunner", "_post_pva"))
        assert "self.model" not in post
        assert ".get(" in post  # the channel lookup, and nothing else


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_serving_runner.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
