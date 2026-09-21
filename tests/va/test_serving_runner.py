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
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from lume.model import LUMEModel
from lume.variables import ScalarVariable, StrVariable

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    PARTITION_STATIC_NOISY,
    RECORD_TYPE_ANALOG,
)
from osprey.services.virtual_accelerator.serving import write_path as write_path_module
from osprey.services.virtual_accelerator.serving.model_stub import NullModel
from osprey.services.virtual_accelerator.serving.model_surface import partition_variables
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
    STUCK_SETPOINTS_VARIABLE,
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
        values have to be available for their absence to mean anything.
        Every variable the wrapped model declares is read back poisoned; the
        wrapper's own stuck set is answered by the wrapper, not the model."""
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert loop.outputs
        assert set(loop.outputs[0]) == {*model.supported_variables, STUCK_SETPOINTS_VARIABLE}
        assert all(loop.outputs[0][name] == POISON for name in model.supported_variables)

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


class TestRouteFollowsTheStuckSet:
    """Which setpoints are stuck is decided per write, from the set in force
    when the write arrives -- so a fault set at runtime takes hold on the next
    write, and clearing it restores what the setpoint did before."""

    def test_a_route_latches_once_stuck_and_echoes_again_once_released(
        self, records: ServingRecords, driver: FakeDriver
    ) -> None:
        """On the no-hook path, where nothing is enqueued and every write is
        published on the thread it arrived on."""
        path = _write_path(records, None)

        path.write(driver, ECHO_SP, 5.0)
        assert driver.values[ECHO_RB] == 5.0

        path.set_stuck_setpoints(frozenset({ECHO_SP}))
        assert path.routes[ECHO_SP].mode == MODE_LATCH
        assert path.routes[ECHO_SP].readback is None
        path.write(driver, ECHO_SP, 6.0)
        # The setpoint still records the command; the readback stays frozen.
        assert driver.values[ECHO_SP] == 6.0
        assert driver.values[ECHO_RB] == 5.0

        path.set_stuck_setpoints(frozenset())
        assert path.routes[ECHO_SP].mode == MODE_ECHO
        assert path.routes[ECHO_SP].readback == ECHO_RB
        path.write(driver, ECHO_SP, 7.0)
        assert driver.values[ECHO_RB] == 7.0

    def test_a_released_physics_route_reaches_the_model_again(
        self,
        path: CohostWritePath,
        driver: FakeDriver,
        loop: FakeRunLoop,
        bridge: RecordingBridge,
    ) -> None:
        path.write(driver, STUCK_SP, 3.25)
        loop.drain()
        assert bridge.calls == []

        path.set_stuck_setpoints(frozenset())
        assert path.routes[STUCK_SP].mode == MODE_PHYSICS
        assert path.routes[STUCK_SP].readback == STUCK_RB
        path.write(driver, STUCK_SP, 3.5)
        loop.drain()
        assert bridge.calls == [(STUCK_SP, 3.5)]
        assert driver.values[STUCK_RB] == 3.5

    def test_a_newly_stuck_physics_route_stops_reaching_the_model(
        self,
        path: CohostWritePath,
        driver: FakeDriver,
        loop: FakeRunLoop,
        bridge: RecordingBridge,
    ) -> None:
        path.set_stuck_setpoints(frozenset({MAG_SP}))
        path.write(driver, MAG_SP, 2.0)
        loop.drain()
        assert bridge.calls == []
        assert driver.values[MAG_SP] == 2.0
        assert driver.values[MAG_RB] == BOOT_VALUES[MAG_RB]
        assert driver.posted(MAG_RB) == 0

    def test_a_write_already_with_the_model_completes_as_routed_on_arrival(
        self,
        path: CohostWritePath,
        driver: FakeDriver,
        loop: FakeRunLoop,
        bridge: RecordingBridge,
    ) -> None:
        """The stuck set changes on the run loop's thread, ordered behind any
        write already queued there: that write predates the fault, so it
        finishes the way it began -- the model takes it and its echo follows."""
        path.write(driver, MAG_SP, 2.0)
        path.set_stuck_setpoints(frozenset({MAG_SP}))
        loop.drain()
        assert bridge.calls == [(MAG_SP, 2.0)]
        assert driver.values[MAG_RB] == 2.0

    def test_a_stuck_physics_route_must_still_be_asynchronous(self, loop: FakeRunLoop) -> None:
        """Stuck at boot is not stuck for good: once the fault is cleared the
        write goes to the model, and a synchronous PV would then tell its
        client the write completed before the solve had started."""
        synchronous = build_serving_pvdb(CHANNELS, drive_limits=DRIVE_LIMITS)
        with pytest.raises(ValueError, match="asynchronous"):
            _write_path(synchronous, loop, stuck=PHYSICS_SETPOINTS)

    def test_physics_routes_with_no_hook_need_not_be_asynchronous(self) -> None:
        """With no model behind the process a coupled setpoint only ever
        latches, so nothing is waiting on a solve."""
        synchronous = build_serving_pvdb(CHANNELS, drive_limits=DRIVE_LIMITS)
        path = _write_path(synchronous, None)
        assert path.routes[MAG_SP].mode == MODE_LATCH


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
        self,
        path: CohostWritePath,
        driver: FakeDriver,
        loop: FakeRunLoop,
        pva: FakePvaChannels,
        model: RecordingModel,
    ) -> None:
        """The output pass offers every variable's read-back value on every
        cycle; the second view is no more allowed to publish one than the
        first is."""
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert loop.outputs
        assert all(loop.outputs[0][name] == POISON for name in model.supported_variables)
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
        """The wrapped model's variables pass through as the same objects;
        the wrapper's own ``stuck_setpoints`` follows them."""
        routed = SetpointRoutedModel(
            model, on_setpoint=bridge.on_setpoint, routed=frozenset({MAG_SP})
        )
        variables = routed.supported_variables
        assert list(variables) == [*model.supported_variables, STUCK_SETPOINTS_VARIABLE]
        assert all(variables[name] is var for name, var in model.supported_variables.items())
        assert routed.get([BPM_X]) == {BPM_X: POISON}
        routed.reset()
        assert model.resets == 1

    def test_no_hook_routes_nothing(self, model: RecordingModel) -> None:
        """With no physics hook, every write -- a routed address included --
        reaches the wrapped model directly."""
        routed = SetpointRoutedModel(model, on_setpoint=None, routed=PHYSICS_SETPOINTS)
        routed.set({MAG_SP: 1.0})
        assert model.sets == [{MAG_SP: 1.0}]

    def test_no_hook_all_routed_batch_is_applied_once(self, model: RecordingModel) -> None:
        routed = SetpointRoutedModel(model, on_setpoint=None, routed=PHYSICS_SETPOINTS)
        routed.set({MAG_SP: 1.0, STUCK_SP: 2.0})
        assert model.sets == [{MAG_SP: 1.0, STUCK_SP: 2.0}]


# A setpoint address no served channel carries.
UNKNOWN_SP = f"{RING}:MAG:HCM:99:CURRENT:SP"
KNOWN_SETPOINTS = frozenset({MAG_SP, STUCK_SP, ECHO_SP})


class StuckSwaps:
    """Stands in for the write path's ``set_stuck_setpoints``.

    Records every set it is handed, and the thread that handed it over.
    """

    def __init__(self, model: RecordingModel | None = None) -> None:
        self.model = model
        self.sets: list[frozenset[str]] = []
        self.threads: list[str] = []
        # The wrapped model's reset count at each call: proves the order of a
        # reset's two steps.
        self.resets_seen: list[int] = []

    def __call__(self, stuck: frozenset[str]) -> None:
        self.sets.append(stuck)
        self.threads.append(threading.current_thread().name)
        if self.model is not None:
            self.resets_seen.append(self.model.resets)


class TestStuckSetpointsVariable:
    """The stuck set, served as one model-only variable of the wrapper.

    The wrapper answers the name itself, so the wrapped model never sees it;
    a write replaces the whole set through ``on_stuck_change`` rather than
    editing one in place, so a reader on another thread sees either the old
    set or the new one and never a set part-way through changing.
    """

    @staticmethod
    def _wrap(
        model: LUMEModel,
        swaps: StuckSwaps,
        *,
        stuck: frozenset[str] = frozenset({STUCK_SP}),
        bridge: RecordingBridge | None = None,
    ) -> SetpointRoutedModel:
        return SetpointRoutedModel(
            model,
            on_setpoint=bridge.on_setpoint if bridge is not None else None,
            routed=frozenset({MAG_SP}),
            stuck_setpoints=stuck,
            known_setpoints=KNOWN_SETPOINTS,
            on_stuck_change=swaps,
        )

    def test_stuck_variable_is_listed_over_null_model(self) -> None:
        """A model with no variables of its own still carries the stuck set,
        which is what lets a lattice-free server take a stuck fault."""
        routed = SetpointRoutedModel(
            NullModel(),
            on_setpoint=None,
            routed=frozenset(),
            stuck_setpoints=frozenset(),
            known_setpoints=KNOWN_SETPOINTS,
            on_stuck_change=StuckSwaps(),
        )
        variables = routed.supported_variables
        assert list(variables) == [STUCK_SETPOINTS_VARIABLE]
        variable = variables[STUCK_SETPOINTS_VARIABLE]
        assert isinstance(variable, StrVariable)
        assert variable.name == STUCK_SETPOINTS_VARIABLE == "stuck_setpoints"
        assert variable.default_value == ""
        assert variable.default_validation_config == "error"
        assert not variable.read_only
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == ""

    def test_stuck_boot_string_is_sorted_and_comma_joined(self, model: RecordingModel) -> None:
        """One canonical spelling per set, so equal sets read back equal."""
        routed = self._wrap(model, StuckSwaps(), stuck=frozenset({STUCK_SP, MAG_SP}))
        boot = f"{MAG_SP},{STUCK_SP}"
        assert routed.supported_variables[STUCK_SETPOINTS_VARIABLE].default_value == boot
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == boot

    def test_stuck_read_is_answered_by_the_wrapper(self, model: RecordingModel) -> None:
        routed = self._wrap(model, StuckSwaps())
        assert routed.get([BPM_X, STUCK_SETPOINTS_VARIABLE]) == {
            BPM_X: POISON,
            STUCK_SETPOINTS_VARIABLE: STUCK_SP,
        }
        routed.get([STUCK_SETPOINTS_VARIABLE])
        # The wrapped model was asked only for its own variable, and not at
        # all for a read of the stuck set alone.
        assert model.reads == [[BPM_X]]

    def test_unknown_stuck_address_is_refused_and_nothing_swapped(
        self, model: RecordingModel
    ) -> None:
        swaps = StuckSwaps()
        routed = self._wrap(model, swaps)
        with pytest.raises(ValueError, match=re.escape(UNKNOWN_SP)):
            routed.set({STUCK_SETPOINTS_VARIABLE: f"{MAG_SP},{UNKNOWN_SP}"})
        assert swaps.sets == []
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == STUCK_SP
        assert model.sets == []

    def test_refused_stuck_write_moves_nothing_else_in_its_batch(
        self, model: RecordingModel, records: ServingRecords
    ) -> None:
        """The stuck set is checked before anything is applied, so a refusal
        cannot leave the rest of the batch half-done."""
        swaps = StuckSwaps()
        bridge = RecordingBridge(model, records)
        routed = self._wrap(model, swaps, bridge=bridge)
        with pytest.raises(ValueError, match=re.escape(UNKNOWN_SP)):
            routed.set({MAG_SP: 1.0, STUCK_SP: 2.0, STUCK_SETPOINTS_VARIABLE: UNKNOWN_SP})
        assert bridge.calls == []
        assert model.sets == []
        assert swaps.sets == []

    def test_valid_stuck_set_swaps(self, model: RecordingModel) -> None:
        swaps = StuckSwaps()
        routed = self._wrap(model, swaps)
        routed.set({STUCK_SETPOINTS_VARIABLE: f" {STUCK_SP} , {MAG_SP},"})
        assert swaps.sets == [frozenset({MAG_SP, STUCK_SP})]
        assert isinstance(swaps.sets[0], frozenset)
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == f"{MAG_SP},{STUCK_SP}"
        # Changing which setpoints are stuck moves no physics, so the wrapped
        # model is not asked for a solve of its own.
        assert model.sets == []

    def test_empty_stuck_string_clears_the_set(self, model: RecordingModel) -> None:
        swaps = StuckSwaps()
        routed = self._wrap(model, swaps)
        routed.set({STUCK_SETPOINTS_VARIABLE: ""})
        assert swaps.sets == [frozenset()]
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == ""

    def test_stuck_write_alongside_a_setpoint_write_applies_both(
        self, model: RecordingModel
    ) -> None:
        swaps = StuckSwaps()
        routed = self._wrap(model, swaps)
        routed.set({STUCK_SETPOINTS_VARIABLE: MAG_SP, STUCK_SP: 4.0})
        assert model.sets == [{STUCK_SP: 4.0}]
        assert swaps.sets == [frozenset({MAG_SP})]

    def test_stuck_write_that_is_not_text_is_refused(self, model: RecordingModel) -> None:
        swaps = StuckSwaps()
        routed = self._wrap(model, swaps)
        with pytest.raises(TypeError):
            routed.set({STUCK_SETPOINTS_VARIABLE: 3.0})
        assert swaps.sets == []

    def test_stuck_change_is_called_on_the_thread_that_sets(self, model: RecordingModel) -> None:
        """Which is the run loop's: the swap needs no lock of its own."""
        swaps = StuckSwaps()
        routed = self._wrap(model, swaps)
        worker = threading.Thread(
            target=routed.set, args=({STUCK_SETPOINTS_VARIABLE: MAG_SP},), name="run-loop"
        )
        worker.start()
        worker.join()
        assert swaps.threads == ["run-loop"]

    def test_reset_restores_the_boot_stuck_set(self, model: RecordingModel) -> None:
        swaps = StuckSwaps(model)
        routed = self._wrap(model, swaps)
        routed.set({STUCK_SETPOINTS_VARIABLE: MAG_SP})
        routed.reset()
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == STUCK_SP
        assert swaps.sets == [frozenset({MAG_SP}), frozenset({STUCK_SP})]
        # The boot set is handed over before the wrapped model resets.
        assert swaps.resets_seen == [0, 0]
        assert model.resets == 1

    def test_stuck_boot_set_may_name_an_unserved_address(self, model: RecordingModel) -> None:
        """A boot fault on an address this server does not carry is inert,
        not a startup failure; only a *write* naming one is refused."""
        routed = self._wrap(model, StuckSwaps(), stuck=frozenset({UNKNOWN_SP}))
        assert routed.get(STUCK_SETPOINTS_VARIABLE) == UNKNOWN_SP

    def test_stuck_name_the_wrapped_model_declares_is_refused(self) -> None:
        """The wrapper answers that name itself, so a wrapped model that
        declared it too would be silently shadowed."""

        class Declaring(NullModel):
            @property
            def supported_variables(self) -> dict[str, Any]:
                return {STUCK_SETPOINTS_VARIABLE: StrVariable(name=STUCK_SETPOINTS_VARIABLE)}

        with pytest.raises(ValueError, match=STUCK_SETPOINTS_VARIABLE):
            SetpointRoutedModel(Declaring(), on_setpoint=None, routed=frozenset())


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
        """The database and the catalog one served tree yields.

        Both are derived from the packaged demo tree and from nothing else,
        the way the entrypoint derives them for a facility's own tree: the
        manifest that tree resolves, the nominals and bands its
        ``machine.json`` and ``channel_limits.json`` carry, and one variable
        factory per binding in its ``va_bindings.json``. Building the catalog
        off a different tree than the database would compare two namespaces
        and prove nothing about either.
        """
        from osprey.services.virtual_accelerator.bindings import load_bindings
        from osprey.services.virtual_accelerator.manifest import build_manifest
        from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
        from osprey.services.virtual_accelerator.model.bindings import build_action_variables
        from osprey.services.virtual_accelerator.model.catalog import build_variable_catalog

        channels = build_manifest(PACKAGE_PATHS)["channels"]
        document = load_bindings(PACKAGE_PATHS.va_bindings)
        catalog = build_variable_catalog(PACKAGE_PATHS, channels, build_action_variables(document))
        return build_serving_pvdb(channels, async_setpoints=True), catalog

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


def _runner_wiring(
    model: LUMEModel,
    records: ServingRecords,
    *,
    on_setpoint: Any = None,
    stuck: frozenset[str] = frozenset(),
) -> tuple[SetpointRoutedModel, CohostWritePath]:
    """The wrapper and the write path, bound to each other as the runner binds them.

    :meth:`TestRunnerShape.test_the_wrapper_is_handed_the_hook_and_the_stuck_set`
    pins the runner's own call to exactly these arguments, so what is proven
    against this pair is proven of the runner's wiring too.
    """
    physics_setpoints = physics_setpoint_addresses(records)
    path = _write_path(records, None, stuck=stuck)
    wrapped = SetpointRoutedModel(
        model,
        on_setpoint=on_setpoint,
        routed=physics_setpoints,
        stuck_setpoints=stuck,
        known_setpoints=frozenset(records.setpoint_readbacks) | physics_setpoints,
        on_stuck_change=path.set_stuck_setpoints,
    )
    return wrapped, path


class TestUnconditionalWrap:
    """The model the runner serves is always the wrapper, a ``NullModel`` included.

    The wrapper is what carries the stuck set, and the stuck set is the one
    fault a server with no lattice behind it can take. So a wrap conditional
    on the physics hook would leave exactly that server with no way to take
    it at runtime.
    """

    def test_the_writable_setpoints_are_exactly_the_routed_ones(
        self, records: ServingRecords
    ) -> None:
        """A stuck-set write may name any address a write can reach, and no
        other: the setpoints paired with a readback and the physics ones."""
        physics_setpoints = physics_setpoint_addresses(records)
        known = frozenset(records.setpoint_readbacks) | physics_setpoints
        assert known == set(_write_path(records, None).routes) == KNOWN_SETPOINTS

    def test_a_wrapped_null_model_serves_nothing(self, records: ServingRecords) -> None:
        """Its only variable is the stuck set, which no manifest lists, so
        the partition leaves it off both transports."""
        wrapped, _ = _runner_wiring(NullModel(), records)
        partition = partition_variables(wrapped, records)
        assert partition.served == {}
        assert list(partition.model_only) == [STUCK_SETPOINTS_VARIABLE]

    def test_a_wrapped_null_model_reads_back_the_boot_stuck_set(
        self, records: ServingRecords
    ) -> None:
        wrapped, _ = _runner_wiring(NullModel(), records, stuck=frozenset({STUCK_SP, ECHO_SP}))
        assert wrapped.get(STUCK_SETPOINTS_VARIABLE) == f"{STUCK_SP},{ECHO_SP}"

    def test_a_lattice_free_server_takes_a_stuck_fault_at_runtime(
        self, records: ServingRecords, driver: FakeDriver
    ) -> None:
        """No hook, no run-loop enqueue: the stuck set still reaches the
        write path, and the next write to the setpoint follows it."""
        wrapped, path = _runner_wiring(NullModel(), records)
        wrapped.set({STUCK_SETPOINTS_VARIABLE: ECHO_SP})
        assert path.routes[ECHO_SP].mode == MODE_LATCH
        path.write(driver, ECHO_SP, 6.0)
        assert driver.values[ECHO_SP] == 6.0
        assert driver.values[ECHO_RB] == BOOT_VALUES[ECHO_RB]

    def test_a_lattice_free_server_releases_a_boot_stuck_fault(
        self, records: ServingRecords, driver: FakeDriver
    ) -> None:
        wrapped, path = _runner_wiring(NullModel(), records, stuck=frozenset({ECHO_SP}))
        assert path.routes[ECHO_SP].mode == MODE_LATCH
        wrapped.set({STUCK_SETPOINTS_VARIABLE: ""})
        assert path.routes[ECHO_SP].mode == MODE_ECHO
        path.write(driver, ECHO_SP, 6.0)
        assert driver.values[ECHO_RB] == 6.0

    def test_a_stuck_write_naming_an_unwritable_address_is_refused(
        self, records: ServingRecords
    ) -> None:
        """A readback is served, but no write reaches it, so it cannot be
        stuck either."""
        wrapped, path = _runner_wiring(NullModel(), records)
        with pytest.raises(ValueError, match=re.escape(ECHO_RB)):
            wrapped.set({STUCK_SETPOINTS_VARIABLE: ECHO_RB})
        assert path.routes[ECHO_SP].mode == MODE_ECHO


#: A reference to the model write token, by either of the names it has in
#: the runner: the constructor argument and the attribute it is kept on.
_TOKEN_NAMES = frozenset({"model_write_token", "_model_write_token"})

#: Calls whose arguments reach a log, a terminal or a warning.
_OUTPUT_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)


def _mentions_token(node: ast.AST) -> bool:
    return any(
        (isinstance(each, ast.Name) and each.id in _TOKEN_NAMES)
        or (isinstance(each, ast.Attribute) and each.attr in _TOKEN_NAMES)
        for each in ast.walk(node)
    )


def _token_leaks(tree: ast.AST) -> list[str]:
    """Every place in ``tree`` that would put the token into text.

    That is a logging, ``print`` or ``warnings.warn`` call, a ``format``
    call, an f-string, a ``%`` interpolation or a raised exception, any of
    which mentions the token. Passing the token on as an argument to another
    call is none of these, which is how the endpoint receives it.
    """
    leaks = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            emits = (isinstance(func, ast.Name) and func.id == "print") or (
                isinstance(func, ast.Attribute) and func.attr in _OUTPUT_METHODS | {"format"}
            )
        else:
            emits = isinstance(node, (ast.JoinedStr, ast.Raise)) or (
                isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
            )
        if emits and _mentions_token(node):
            leaks.append(ast.unparse(node))
    return leaks


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

    def _function(self, tree: ast.Module, name: str) -> ast.FunctionDef:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"function {name!r} not found")

    def _try(self, function: ast.FunctionDef) -> ast.Try:
        tries = [node for node in function.body if isinstance(node, ast.Try)]
        assert len(tries) == 1, f"{function.name} has exactly one try block"
        return tries[0]

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

    def test_the_documents_readback_rules_reach_the_write_path(self, tree: ast.Module) -> None:
        """The runner is the whole distance between the process that reads the
        bindings document and the write path that applies it.

        Dropped here, nothing fails: every coupled readback quietly becomes an
        echo of the value written again, which is exactly what a facility that
        exported a reverse curve does not have.
        """
        init = self._method(tree, "CohostRunner", "__init__")
        built = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "CohostWritePath"
        ]

        assert len(built) == 1, "the write path is built exactly once"
        forwarded = [
            keyword.value for keyword in built[0].keywords if keyword.arg == "bound_setpoints"
        ]
        assert [ast.unparse(value) for value in forwarded] == ["bound"]

    def test_every_setpoint_either_end_knows_about_routes_through_the_model(
        self, tree: ast.Module
    ) -> None:
        """The routed set is the union of the manifest's coupled partition and
        the document's writable bindings -- the same union the write path
        computes for itself.

        Narrowed back to the manifest's half, a setpoint the document binds and
        the manifest does not partition as coupled would latch here while the
        write path routed it: one address, two answers about whether the
        physics serves it.
        """
        init = self._method(tree, "CohostRunner", "__init__")
        wrapped = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SetpointRoutedModel"
        ]

        assert len(wrapped) == 1, "the model is wrapped exactly once"
        routed = [
            ast.unparse(keyword.value) for keyword in wrapped[0].keywords if keyword.arg == "routed"
        ]
        assert routed == ["routed"]
        assigned = [
            ast.unparse(node.value)
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and [ast.unparse(target) for target in node.targets] == ["routed"]
        ]
        assert assigned == ["physics_setpoints | frozenset(bound)"]

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

    def _super_init(self, init: ast.FunctionDef) -> ast.Call:
        calls = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "super().__init__"
        ]
        assert len(calls) == 1, "the base constructor is called exactly once"
        return calls[0]

    def _assignments(self, init: ast.FunctionDef, target: str) -> list[ast.Assign]:
        return [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Assign)
            and [ast.unparse(each) for each in node.targets] == [target]
        ]

    def test_model_only_variables_leave_the_config_before_the_base_constructor(
        self, tree: ast.Module
    ) -> None:
        """The base constructor builds a channel for every configured name
        and rejects only names the model lacks, never names the configuration
        omits. So removal from the configuration is the whole of keeping a
        model-only variable off the wire -- and it has to be done before the
        base constructor reads the configuration, or the channel exists."""
        init = self._method(tree, "CohostRunner", "__init__")
        removals = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.For) and ast.unparse(node.iter) == "self.partition.model_only"
        ]

        assert len(removals) == 1, "the model-only names are removed in one place"
        assert "config['variables']" in ast.unparse(removals[0])
        assert removals[0].end_lineno is not None
        assert removals[0].end_lineno < self._super_init(init).lineno

    def test_the_partition_is_of_the_model_the_base_class_is_handed(self, tree: ast.Module) -> None:
        """Wrapper included. The wrapper declares a variable of its own that
        no manifest lists; partitioning the model it wraps would never see
        that variable, which would then be configured and served."""
        init = self._method(tree, "CohostRunner", "__init__")
        partitions = self._assignments(init, "self.partition")
        super_init = self._super_init(init)

        assert len(partitions) == 1
        assert ast.unparse(partitions[0].value) == "partition_variables(model, records)"
        wraps = self._assignments(init, "model")
        assert wraps, "the model is wrapped before it is partitioned"
        assert all(wrap.lineno < partitions[0].lineno for wrap in wraps)
        assert partitions[0].lineno < super_init.lineno
        handed = [ast.unparse(kw.value) for kw in super_init.keywords if kw.arg == "model"]
        assert handed == ["model"]

    def test_the_run_loop_reads_back_only_served_variables(self, tree: ast.Module) -> None:
        """The base roster is the model's whole namespace. A model-only
        variable read back after every cycle is a model read with no channel
        to go to."""
        roster = ast.unparse(self._method(tree, "CohostRunner", "_cycle_output_names"))

        assert "self.partition.served" in roster
        assert "self.model" not in roster
        assert "model_only" not in roster

    def _wrap(self, init: ast.FunctionDef) -> ast.Assign:
        wraps = self._assignments(init, "model")
        assert len(wraps) == 1, "the model is wrapped exactly once"
        return wraps[0]

    def test_the_model_is_wrapped_unconditionally(self, tree: ast.Module) -> None:
        """A ``NullModel`` included -- see :class:`TestUnconditionalWrap`. The
        wrap is a statement of the constructor's own body, so no branch can
        skip it, and it wraps the model it was given, whatever that is."""
        init = self._method(tree, "CohostRunner", "__init__")
        wrap = self._wrap(init)

        assert wrap in init.body
        assert isinstance(wrap.value, ast.Call)
        assert ast.unparse(wrap.value.func) == "SetpointRoutedModel"
        assert [ast.unparse(arg) for arg in wrap.value.args] == ["model"]
        assert not any(isinstance(node, ast.IfExp) for node in ast.walk(wrap))

    def test_the_wrapper_is_handed_the_hook_and_the_stuck_set(self, tree: ast.Module) -> None:
        """The hook as given, ``None`` included; the boot stuck set; every
        setpoint a write reaches as the addresses a stuck-set write may name;
        and the write path's own swap as what a new set is handed to."""
        wrap = self._wrap(self._method(tree, "CohostRunner", "__init__")).value
        assert isinstance(wrap, ast.Call)
        keywords = {kw.arg: ast.unparse(kw.value) for kw in wrap.keywords}
        assert keywords == {
            "on_setpoint": "on_setpoint",
            "routed": "physics_setpoints",
            "stuck_setpoints": "stuck_setpoints",
            "known_setpoints": "frozenset(records.setpoint_readbacks) | physics_setpoints",
            "on_stuck_change": "self.write_path.set_stuck_setpoints",
        }

    def test_the_write_path_is_built_before_the_wrapper_is_handed_its_swap(
        self, tree: ast.Module
    ) -> None:
        init = self._method(tree, "CohostRunner", "__init__")
        paths = self._assignments(init, "self.write_path")
        assert len(paths) == 1
        assert paths[0].lineno < self._wrap(init).lineno

    def test_the_write_path_starts_from_the_boot_stuck_set(self, tree: ast.Module) -> None:
        """The wrapper and the write path begin from the same set; only a
        write through the wrapper moves the write path's afterwards."""
        init = self._method(tree, "CohostRunner", "__init__")
        path = self._assignments(init, "self.write_path")[0].value
        assert isinstance(path, ast.Call)
        assert ast.unparse(path.func) == "CohostWritePath"
        keywords = {kw.arg: ast.unparse(kw.value) for kw in path.keywords}
        assert keywords["stuck_setpoints"] == "stuck_setpoints"

    def _keyword_only(self, tree: ast.Module) -> dict[str, str]:
        init = self._method(tree, "CohostRunner", "__init__")
        return {
            arg.arg: ast.unparse(arg.annotation) if arg.annotation is not None else ""
            for arg in init.args.kwonlyargs
        }

    @pytest.mark.parametrize(
        ("name", "annotation"),
        [
            ("model_write_token", "str | None"),
            ("backend_name", "str"),
            ("lattice_source", "str"),
        ],
    )
    def test_the_endpoint_arguments_are_keyword_only(
        self, tree: ast.Module, name: str, annotation: str
    ) -> None:
        """``None`` for the token is what disables model writes; it is a
        value the caller passes, not one this constructor assumes."""
        assert self._keyword_only(tree)[name] == annotation

    @pytest.mark.parametrize("name", ["model_write_token", "backend_name", "lattice_source"])
    def test_the_endpoint_arguments_are_kept_as_given(self, tree: ast.Module, name: str) -> None:
        init = self._method(tree, "CohostRunner", "__init__")
        kept = self._assignments(init, f"self._{name}")
        assert [ast.unparse(each.value) for each in kept] == [name]

    def test_the_model_write_token_is_never_logged(self, tree: ast.Module) -> None:
        """Nor printed, warned, formatted into text or raised: the token is
        the one thing that gates a model write, and a log is readable by far
        more people than the write is allowed to."""
        assert _token_leaks(tree) == []

    @pytest.mark.parametrize(
        "leak",
        [
            "LOG.info('armed with %s', self._model_write_token)",
            "print(model_write_token)",
            "warnings.warn(f'token {model_write_token}')",
            "text = 'token %s' % self._model_write_token",
            "raise ValueError(model_write_token)",
            "text = '{}'.format(self._model_write_token)",
        ],
    )
    def test_the_leak_check_catches_a_leak(self, leak: str) -> None:
        """The check above is only as good as its detector: each of these
        would put the token into text, and each is caught."""
        assert _token_leaks(ast.parse(leak))

    def test_the_leak_check_lets_the_token_be_passed_on(self) -> None:
        """Handing the token to the object that checks it is not a leak."""
        passed_on = ast.parse("surface = ModelSurface(token=self._model_write_token)")
        assert _token_leaks(passed_on) == []

    def _surface(self, tree: ast.Module) -> ast.Call:
        hook = self._method(tree, "CohostRunner", "_create_model_info")
        built = [
            node
            for node in ast.walk(hook)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "ModelSurface"
        ]
        assert len(built) == 1, "the model surface is built exactly once"
        return built[0]

    def test_the_model_surface_extends_the_base_class_hook(self, tree: ast.Module) -> None:
        """It extends rather than replaces: the base class's own model_info
        channel is what a client reads the served variable roster from, and
        an override that dropped it would take that channel off the wire
        while every other channel stayed exactly as it was."""
        hook = self._method(tree, "CohostRunner", "_create_model_info")
        statements = [
            node
            for node in hook.body
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
        ]

        assert ast.unparse(statements[0]) == "super()._create_model_info()"

    def test_the_rpc_channel_is_a_provider_of_the_contract_name(self, tree: ast.Module) -> None:
        """The hook is the last moment a provider can still be added: the
        base constructor hands this dictionary to the server straight after
        it, and a provider added later is never served."""
        hook = self._method(tree, "CohostRunner", "_create_model_info")
        registrations = [
            node
            for node in ast.walk(hook)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
        ]

        assert len(registrations) == 1, "one channel is registered"
        target = registrations[0].targets[0]
        assert isinstance(target, ast.Subscript)
        assert ast.unparse(target.value) == "self.providers"
        addressed = ast.unparse(target.slice)
        assert "RPC_PV" in addressed, "the channel is named by the wire contract"
        assert "prefix" in addressed, "under the same prefix every other channel carries"

    def test_the_rpc_channel_is_open_and_answers_calls(self, tree: ast.Module) -> None:
        """A channel built without an initial value stays closed and refuses
        every operation, RPC included -- which reads as a server that is up
        and has no model surface."""
        hook = ast.unparse(self._method(tree, "CohostRunner", "_create_model_info"))

        assert "SharedPV(initial=" in hook
        assert "channel.rpc(self._rpc)" in hook

    def test_the_surface_answers_for_the_model_the_base_class_serves(
        self, tree: ast.Module
    ) -> None:
        """The wrapped model and the partition taken of it: the wrapper's own
        stuck set is a model-only variable like any other, and a surface built
        over the unwrapped model could neither report it nor write it."""
        built = self._surface(tree)

        assert [ast.unparse(arg) for arg in built.args] == [
            "self.model",
            "self.partition",
            "self._records",
        ]

    def test_the_surface_is_handed_the_facts_only_this_boot_knows(self, tree: ast.Module) -> None:
        """Every one of these is a boot fact nothing downstream can recover:
        which backend was built, where its lattice came from, which server
        this is and where it is reached, how the run loop batches, the
        credential a write must present, and what a landed write refreshes."""
        keywords = {kw.arg: ast.unparse(kw.value) for kw in self._surface(tree).keywords}

        assert keywords == {
            "backend_name": "self._backend_name",
            "lattice_source": "self._lattice_source",
            "instance": "_instance_name()",
            "endpoint": "_pva_endpoint()",
            "update_rate": "self.update_rate",
            "model_write_token": "self._model_write_token",
            "refresh": "self._refresh",
        }

    def test_the_reported_endpoint_is_the_port_the_server_binds(self, tree: ast.Module) -> None:
        """Read from the variable the pvAccess server itself binds from, so
        the address a client is told to use cannot drift from the one in
        use."""
        endpoint = ast.unparse(self._function(tree, "_pva_endpoint"))

        assert "EPICS_PVAS_SERVER_PORT" in endpoint
        assert "DEFAULT_PVA_PORT" in endpoint

    def test_the_physics_refresh_is_keyword_only(self, tree: ast.Module) -> None:
        """A model write moves a variable no setpoint is written to, so
        nothing else in this process would notice it: what recomputes the
        readings it feeds is passed in, not assumed."""
        assert self._keyword_only(tree)["refresh"] == "Callable[[list[str]], None]"

    def test_the_physics_refresh_is_kept_as_given(self, tree: ast.Module) -> None:
        init = self._method(tree, "CohostRunner", "__init__")
        kept = self._assignments(init, "self._refresh")

        assert [ast.unparse(each.value) for each in kept] == ["refresh"]

    def test_a_server_with_no_physics_behind_it_refreshes_nothing(self, tree: ast.Module) -> None:
        """The default has to be inert rather than absent: the surface calls
        it after every write that lands."""
        init = self._method(tree, "CohostRunner", "__init__")
        defaults = {
            arg.arg: ast.unparse(default)
            for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)
            if default is not None
        }

        assert defaults["refresh"] == "_refresh_nothing"
        body = self._function(tree, "_refresh_nothing").body
        assert len(body) == 1, "its whole body is its docstring"
        assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)

    def test_the_rpc_handler_never_touches_the_model(self, tree: ast.Module) -> None:
        """It runs on a p4p worker thread, which is no more allowed to touch
        the model than the Channel Access server thread is. The verb is
        dispatched by a job instead, and the job runs on the run loop."""
        rpc = ast.unparse(self._method(tree, "CohostRunner", "_rpc"))

        assert "self.model" not in rpc
        assert "self._surface" not in rpc
        assert "self._dispatch" not in rpc
        # Empty values, so the cycle carrying the job runs no model pass of
        # its own: the job is the whole of what this call costs the loop.
        assert "self._enqueue({}, jobs=[" in rpc

    def test_a_call_arriving_before_the_driver_exists_is_refused(self, tree: ast.Module) -> None:
        """The same window a put is refused in, and for the same reason: the
        PVA server is listening from the moment it is created, which is
        before the driver a served value is committed through exists."""
        rpc = ast.unparse(self._method(tree, "CohostRunner", "_rpc"))

        assert "self.ca_driver" in rpc
        assert "ERR_NOT_READY" in rpc

    def test_a_request_that_is_not_one_is_refused_without_the_run_loop(
        self, tree: ast.Module
    ) -> None:
        """A malformed request never reaches the model, so it never needs the
        loop's thread -- and would otherwise occupy a cycle to be told so."""
        rpc = self._method(tree, "CohostRunner", "_rpc")
        parsing = self._try(rpc)
        enqueues = [
            node
            for node in ast.walk(rpc)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "self._enqueue"
        ]

        assert "parse_request" in ast.unparse(parsing.body)
        refusal = parsing.handlers[0]
        assert "op.done(error=" in ast.unparse(refusal)
        assert len(enqueues) == 1
        assert refusal.end_lineno is not None
        assert refusal.end_lineno < enqueues[0].lineno

    def test_a_call_the_run_loop_never_reaches_is_answered_anyway(self, tree: ast.Module) -> None:
        """A client told nothing waits out its own timeout knowing only that
        the server did not answer; the contract's own message says why."""
        rpc = ast.unparse(self._method(tree, "CohostRunner", "_rpc"))

        assert "threading.Timer(RPC_TIMEOUT_S, call.complete, (error_reply(ERR_TIMEOUT),))" in rpc
        # Daemon, so a call still being waited on never holds up a shutdown.
        assert "timeout.daemon = True" in rpc
        assert "timeout.start()" in rpc

    def test_a_call_is_answered_exactly_once(self, tree: ast.Module) -> None:
        """The job and the timer race to answer, and p4p completes an
        operation once: a second completion is an error the client it was
        meant for never sees."""
        complete = ast.unparse(self._method(tree, "_RpcCall", "complete"))

        assert "with self._lock" in complete
        assert "self._answered" in complete
        assert "return False" in complete

    def test_the_job_answers_its_call_on_every_path(self, tree: ast.Module) -> None:
        """The run loop logs a job that raises and moves on to the next item.
        So a job that returned without answering would cost its client the
        whole timeout, and tell it nothing when the timeout expired."""
        dispatched = self._try(self._method(tree, "CohostRunner", "_answer"))
        closing = ast.unparse(dispatched.finalbody)

        assert "call.complete(reply)" in closing
        assert "ok_reply(" in ast.unparse(dispatched.body)
        assert all("error_reply(" in ast.unparse(handler) for handler in dispatched.handlers)

    def test_the_reply_goes_out_before_the_timer_is_cancelled(self, tree: ast.Module) -> None:
        """Cancelling first would leave a client with nothing at all if the
        reply itself could not be delivered; this way the timer still
        answers, late, with the reason."""
        closing = [
            ast.unparse(statement)
            for statement in self._try(self._method(tree, "CohostRunner", "_answer")).finalbody
        ]

        assert closing.index("call.complete(reply)") < closing.index("timeout.cancel()")

    def test_the_loop_records_what_only_it_can_see(self, tree: ast.Module) -> None:
        """``status`` reports the run loop's own state, which the surface is
        told from the one thread that can read it."""
        answer = ast.unparse(self._method(tree, "CohostRunner", "_answer"))

        assert "surface.record_queue_depth(self.queue.qsize())" in answer
        assert "surface.record_cycle(" in answer

    def test_a_write_the_surface_did_not_refuse_itself_is_still_recorded(
        self, tree: ast.Module
    ) -> None:
        """The surface records the refusals it raises. A write that failed
        some other way is a refused write too, and is the one such failure
        this handler has to record for ``status`` itself -- while a refused
        *read* is recorded by neither, being no write at all."""
        answer = ast.unparse(self._method(tree, "CohostRunner", "_answer"))

        assert "MODEL_WRITE_VERBS" in answer
        assert "surface.record_refusal(" in answer

    def test_the_diff_verb_reads_what_the_control_system_serves(self, tree: ast.Module) -> None:
        """Its whole answer is the served value beside the model's truth, and
        only the driver knows the first of the two."""
        dispatch = ast.unparse(self._method(tree, "CohostRunner", "_dispatch"))

        assert "driver.getParam" in dispatch

    def test_every_verb_the_contract_admits_is_dispatched(self, tree: ast.Module) -> None:
        """A verb the contract admits and this dispatch does not would be
        parsed, enqueued, and answered with whatever the fall-through verb
        happens to be."""
        pytest.importorskip("p4p")
        from osprey.services.virtual_accelerator.serving.model_rpc import VERBS

        dispatch = self._method(tree, "CohostRunner", "_dispatch")
        answered = {
            node.func.attr
            for node in ast.walk(dispatch)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and ast.unparse(node.func.value) == "surface"
        }

        assert answered == set(VERBS)


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_serving_runner.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
