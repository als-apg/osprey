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
from osprey.services.virtual_accelerator.serving.pvdb import (
    ServingRecords,
    build_serving_pvdb,
)
from osprey.services.virtual_accelerator.serving.write_path import (
    MODE_ECHO,
    MODE_LATCH,
    MODE_PHYSICS,
    RUNNER_CONFIG_POLICY,
    CohostWritePath,
    SetpointRoutedModel,
    clamp_into,
    physics_setpoint_addresses,
)

# Floor for this module's own test count -- a guard against a refactor that
# leaves the file importable but empty, which would otherwise pass silently.
MIN_COLLECTED_TESTS = 50

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


@pytest.fixture()
def records() -> ServingRecords:
    """The real serving database -- never a mock of it."""
    return build_serving_pvdb(
        CHANNELS,
        drive_limits=DRIVE_LIMITS,
        boot_values=BOOT_VALUES,
        async_setpoints=True,
    )


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
) -> CohostWritePath:
    return CohostWritePath(
        records,
        enqueue=loop.enqueue if loop is not None else None,
        physics_setpoints=physics_setpoint_addresses(records),
        stuck_setpoints=stuck,
        drive_limits=DRIVE_LIMITS,
        refusal_alarm=("WRITE_ALARM", "INVALID_ALARM"),
    )


@pytest.fixture()
def path(records: ServingRecords, loop: FakeRunLoop) -> CohostWritePath:
    return _write_path(records, loop, stuck=frozenset({STUCK_SP}))


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
    def refusing(self, records: ServingRecords) -> tuple[CohostWritePath, FakeRunLoop, FakeDriver]:
        model = RecordingModel(refuse=frozenset({MAG_SP}))
        bridge = RecordingBridge(model, records)
        loop = FakeRunLoop(
            SetpointRoutedModel(model, on_setpoint=bridge.on_setpoint, routed=PHYSICS_SETPOINTS)
        )
        drv = FakeDriver({address: spec["value"] for address, spec in records.pvdb.items()})
        records.attach_driver(drv)
        return _write_path(records, loop), loop, drv

    def test_one_shot_reader_sees_no_movement(self, refusing) -> None:  # noqa: ANN001
        """The served database is what a fresh read is answered from."""
        path, loop, driver = refusing
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.values[MAG_SP] == BOOT_VALUES[MAG_SP]
        assert driver.values[MAG_RB] == BOOT_VALUES[MAG_RB]

    def test_monitoring_reader_sees_no_movement(self, refusing) -> None:  # noqa: ANN001
        """A monitoring client is served what is posted, and nothing is."""
        path, loop, driver = refusing
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert [call for call in driver.calls if call[0] == "setParam"] == []
        assert driver.posted(MAG_RB) == 0

    def test_completion_still_fires(self, refusing) -> None:  # noqa: ANN001
        """Withholding it would postpone every later write to this setpoint,
        for the life of the process."""
        path, loop, driver = refusing
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert ("callbackPV", MAG_SP, None) in driver.calls

    def test_refusal_raises_an_alarm_after_the_value_is_left_alone(
        self,
        refusing,  # noqa: ANN001
    ) -> None:
        """Put-completion can only report success, so the alarm is the only
        signal a refusal can leave."""
        path, loop, driver = refusing
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        assert driver.sequence(MAG_SP) == [
            ("setParamStatus", MAG_SP),
            ("updatePV", MAG_SP),
            ("callbackPV", MAG_SP),
        ]

    def test_a_later_accepted_write_still_lands(self, refusing) -> None:  # noqa: ANN001
        """The refusal leaves nothing behind that blocks the next write."""
        path, loop, driver = refusing
        path.write(driver, MAG_SP, 3.25)
        loop.drain()
        path.write(driver, ECHO_SP, 7.0)
        loop.drain()
        assert driver.values[ECHO_SP] == 7.0
        assert driver.values[ECHO_RB] == 7.0


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
            "clamp_writes": True,
            "control_pvs": False,
        }


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


def test_this_module_collects_its_whole_suite(request: pytest.FixtureRequest) -> None:
    """Vacuous-green guard: an empty or half-collected module fails here."""
    collected = [
        item
        for item in request.session.items
        if item.nodeid.split("::")[0].endswith("test_serving_runner.py")
    ]

    assert len(collected) >= MIN_COLLECTED_TESTS
