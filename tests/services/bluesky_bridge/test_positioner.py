"""A positioner: a setpoint whose move starts when its start command is written.

Each test drives a real :class:`ConnectorPositioner` over the connector double
of ``_fake_connector``: the double records every read and write, and a write
to the start command delivers a scripted status sequence to the status's
subscribers, so ``osprey_connectors.command.fire`` waits on it exactly as it
waits on a control system. Nothing patches the write path.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from bluesky import Msg, RunEngine
from bluesky import plan_stubs as bps
from bluesky.protocols import Stoppable
from bluesky.utils import RunEngineInterrupted

from osprey.services.bluesky_bridge.devices.connector import (
    PRESS_VALUE,
    ConnectorPositioner,
    ConnectorSettable,
    build_devices,
)
from osprey.services.bluesky_bridge.devices.specs import SettableSpec
from osprey_connectors.command import (
    CommandAct,
    CommandBusyError,
    CommandNoEffectError,
    CommandTimeoutError,
)
from osprey_connectors.errors import ChannelWriteBlockedError
from osprey_connectors.settle import SettleReadError
from tests.services.bluesky_bridge._fake_connector import FakeChannelValue, FakeConnector

SETPOINT, READBACK = "ES:STAGE:POS:SP", "ES:STAGE:POS:RB"
COMMAND, STATUS = "ES:STAGE:MOVE", "ES:STAGE:STATUS"
IDLE, MOVING = 0, 1
#: The act's bound, which is also how long ``fire`` waits for the status to leave idle.
TIMEOUT_S = 0.2
#: Where every test's stage starts, and the target of a move away from it.
START, TARGET = 0.0, 2.0
#: The readback's settle bound around the target.
TOLERANCE = 0.01


class _Stage(FakeConnector):
    """A stage: writing the start command carries the readback to the setpoint.

    With ``moves`` False the readback stays where it is. ``command_written``
    is set once the start command has been written, and ``reads_at_command``
    is how many reads had been made by then. ``arrives_after`` holds the
    readback off the target for that many reads after the start command, as a
    stage still settling does. With ``hold_after_target`` the status's next
    read after the target write waits until it is cancelled.
    """

    def __init__(self, *, separate_readback: bool = True, hold_after_target: bool = False) -> None:
        super().__init__({SETPOINT: START, STATUS: IDLE})
        if separate_readback:
            self.readbacks[READBACK] = START
        self.moves = separate_readback
        self.hold_after_target = hold_after_target
        self.read_held_here = asyncio.Event()
        self.command_written = asyncio.Event()
        self.reads_at_command: int | None = None
        self.arrives_after = 0
        self.responses[COMMAND] = (STATUS, [MOVING, IDLE])

    async def read_channel(self, channel_address: str, timeout: float | None = None):
        if channel_address == READBACK and self.command_written.is_set():
            if self.arrives_after > 0:
                self.arrives_after -= 1
                self.read_calls.append(channel_address)
                return FakeChannelValue(value=START)
        return await super().read_channel(channel_address, timeout)

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        if channel_address == COMMAND and self.moves:
            self.readbacks[READBACK] = self.readbacks[SETPOINT]
        result = await super().write_channel_checked(channel_address, value, **kwargs)
        if channel_address == SETPOINT and self.hold_after_target:
            self.hold_reads.add(STATUS)
        if channel_address == COMMAND:
            self.reads_at_command = len(self.read_calls)
            self.command_written.set()
        return result


def _positioner(connector: FakeConnector, *, separate_readback: bool = True) -> ConnectorPositioner:
    return ConnectorPositioner(
        connector,
        SETPOINT,
        READBACK if separate_readback else None,
        started_by=CommandAct(COMMAND, STATUS, IDLE, TIMEOUT_S),
        name=SETPOINT,
        settle_tolerance=TOLERANCE,
    )


def _notes(error: BaseException) -> str:
    return "\n".join(getattr(error, "__notes__", []))


@pytest.mark.asyncio
async def test_a_move_writes_the_target_then_fires_the_start_command() -> None:
    stage = _Stage()

    await _positioner(stage).set(TARGET)

    assert stage.write_calls == [(SETPOINT, TARGET, {}), (COMMAND, PRESS_VALUE, {"confirm": False})]


@pytest.mark.asyncio
async def test_the_move_ends_when_the_status_leaves_idle_and_returns() -> None:
    stage = _Stage()
    stage.responses[COMMAND] = (STATUS, [MOVING])

    status = _positioner(stage).set(TARGET)
    await stage.command_written.wait()
    await asyncio.sleep(0)
    assert not status.done

    for _address, callback in [s for s in stage.subscribers.values() if s[0] == STATUS]:
        callback(FakeChannelValue(value=IDLE))
    await status

    assert status.success


@pytest.mark.asyncio
async def test_a_busy_status_refuses_the_move_before_anything_is_written() -> None:
    stage = _Stage()
    stage.readbacks[STATUS] = MOVING

    with pytest.raises(CommandBusyError):
        await _positioner(stage).set(TARGET)

    assert stage.write_calls == []


@pytest.mark.asyncio
async def test_a_refused_target_never_fires_the_start_command() -> None:
    stage = _Stage()
    stage.write_side_effect = ChannelWriteBlockedError(SETPOINT, "LIMITS")

    with pytest.raises(ChannelWriteBlockedError):
        await _positioner(stage).set(TARGET)

    assert [address for address, _value, _kwargs in stage.write_calls] == [SETPOINT]


@pytest.mark.asyncio
async def test_a_move_to_where_the_readback_already_sits_passes_without_an_edge() -> None:
    stage = _Stage()
    stage.readbacks[READBACK] = TARGET + TOLERANCE / 2
    stage.readbacks[SETPOINT] = TARGET + TOLERANCE / 2
    del stage.responses[COMMAND]

    status = _positioner(stage).set(TARGET)
    await status

    assert status.success
    assert [address for address, _value, _kwargs in stage.write_calls] == [SETPOINT, COMMAND]
    assert stage.readbacks[STATUS] == IDLE


@pytest.mark.asyncio
async def test_a_start_with_no_effect_fails_when_the_readback_is_elsewhere() -> None:
    stage = _Stage()
    del stage.responses[COMMAND]
    stage.moves = False

    with pytest.raises(CommandNoEffectError) as failed:
        await _positioner(stage).set(TARGET)

    assert f"start had no effect: {READBACK} reads {START}, target {TARGET}" in _notes(failed.value)


@pytest.mark.asyncio
async def test_a_start_with_no_effect_fails_without_a_separate_readback() -> None:
    stage = _Stage(separate_readback=False)
    del stage.responses[COMMAND]

    with pytest.raises(CommandNoEffectError) as failed:
        await _positioner(stage, separate_readback=False).set(TARGET)

    assert "start had no effect" in _notes(failed.value)
    assert stage.readbacks[SETPOINT] == TARGET  # what the setpoint reads proves nothing


@pytest.mark.asyncio
async def test_a_failed_readback_read_after_no_effect_names_the_target_the_setpoint_holds() -> None:
    stage = _Stage()
    del stage.responses[COMMAND]
    stage.moves = False
    stage.readbacks[READBACK] = None

    with pytest.raises(SettleReadError) as failed:
        await _positioner(stage).set(TARGET)

    assert f"{SETPOINT} holds the target {TARGET}" in _notes(failed.value)


@pytest.mark.asyncio
async def test_a_status_that_never_returns_fails_the_move_with_the_bound() -> None:
    stage = _Stage()
    stage.responses[COMMAND] = (STATUS, [MOVING])

    with pytest.raises(CommandTimeoutError) as failed:
        await _positioner(stage).set(TARGET)

    assert failed.value.timeout_s == TIMEOUT_S
    assert f"{SETPOINT} holds the target {TARGET}" in _notes(failed.value)


@pytest.mark.asyncio
async def test_the_readback_settles_after_the_act() -> None:
    stage = _Stage()
    stage.arrives_after = 2

    status = _positioner(stage).set(TARGET)
    await status

    assert status.success
    assert stage.arrives_after == 0
    after_the_act = stage.read_calls[stage.read_calls.index(STATUS, 1) + 1 :]
    assert after_the_act.count(READBACK) == 3


@pytest.mark.asyncio
async def test_a_positioner_without_a_separate_readback_polls_nothing_after_the_act() -> None:
    stage = _Stage(separate_readback=False)

    await _positioner(stage, separate_readback=False).set(TARGET)

    assert stage.reads_at_command is not None
    assert stage.read_calls[stage.reads_at_command :] == []


@pytest.mark.asyncio
async def test_stop_cancels_an_unfinished_move_and_nothing_is_written_after_it() -> None:
    stage = _Stage(hold_after_target=True)
    positioner = _positioner(stage)

    status = positioner.set(TARGET)
    await stage.read_held_here.wait()
    await positioner.stop()

    assert status.done and not status.success
    assert stage.held_read_cancelled.is_set()
    assert stage.write_calls == [(SETPOINT, TARGET, {})]


@pytest.mark.asyncio
async def test_stop_after_a_finished_move_does_nothing() -> None:
    stage = _Stage()
    positioner = _positioner(stage)
    await positioner.set(TARGET)
    written = list(stage.write_calls)

    await positioner.stop()

    assert stage.write_calls == written
    assert positioner.stop_writes() == ()


def test_a_plain_settable_is_not_stoppable() -> None:
    assert not isinstance(ConnectorSettable(FakeConnector(), SETPOINT, name=SETPOINT), Stoppable)
    assert isinstance(_positioner(FakeConnector()), Stoppable)


def test_a_positioner_states_its_target_and_start_writes() -> None:
    positioner = _positioner(FakeConnector())

    assert [(w.address, w.value) for w in positioner.set_writes(TARGET)] == [
        (SETPOINT, TARGET),
        (COMMAND, PRESS_VALUE),
    ]
    assert positioner.channel_addresses() == (SETPOINT, READBACK, COMMAND, STATUS)


def test_a_pause_mid_move_cancels_it_and_an_abort_writes_nothing_after() -> None:
    """OSPREY aborts a run by pausing it, then aborting the paused run.

    The pause lands while the move waits on its status, after the target is
    written and before the start command is: the RunEngine stops the device,
    the move is cancelled, and the abort writes nothing more.
    """
    stage = _Stage(hold_after_target=True)
    devices = asyncio.run(
        build_devices(
            [
                SettableSpec(
                    SETPOINT,
                    SETPOINT,
                    READBACK,
                    settle_tolerance=TOLERANCE,
                    started_by=CommandAct(COMMAND, STATUS, IDLE, TIMEOUT_S),
                )
            ],
            connector=stage,
        )
    )
    positioner = devices[SETPOINT]
    assert isinstance(positioner, ConnectorPositioner)

    def plan():
        yield from bps.abs_set(positioner, TARGET)
        yield Msg("wait_for", None, [lambda: stage.read_held_here.wait()])
        yield Msg("pause")

    run_engine = RunEngine(context_managers=[])
    with pytest.raises(RunEngineInterrupted):
        run_engine(plan())
    assert run_engine.state == "paused"
    assert stage.held_read_cancelled.is_set()

    run_engine.abort()

    assert run_engine.state == "idle"
    assert stage.write_calls == [(SETPOINT, TARGET, {})]
