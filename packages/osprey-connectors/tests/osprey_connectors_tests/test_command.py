"""Firing a command channel: write it once, then wait for its status to leave and return to idle.

The connector here is scripted: it records every call in order, delivers the
status updates a test hands it (on subscribe, during the write, after the
write, one loop turn apart), and answers the one read with the reading a test
gives it. The bounds are real and short; no clock is injected.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from osprey_connectors.command import (
    CommandAct,
    CommandBusyError,
    CommandNoEffectError,
    CommandStatusError,
    CommandTimeoutError,
    fire,
)
from osprey_connectors.control_system import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.errors import (
    ChannelError,
    ChannelReadFailedError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    RefusalReason,
)

COMMAND = "BL:CMD"
STATUS = "BL:STAT"
#: The act's bound: short, so a test that waits it out stays short.
BOUND_S = 0.2


class _Drop:
    """A status connection that drops, in a script of updates."""


DROP = _Drop()


def _value(value: Any, label: str | None = None) -> ChannelValue:
    return ChannelValue(
        value=value, timestamp=datetime.now(UTC), metadata=ChannelMetadata(enum_label=label)
    )


def _act(idle: Any = 0, *, timeout_s: float = BOUND_S) -> CommandAct:
    return CommandAct(command=COMMAND, status=STATUS, idle=idle, timeout_s=timeout_s)


Update = ChannelValue | _Drop


class Scripted:
    """A connector that plays a script of status updates around one command write."""

    def __init__(
        self,
        *,
        reading: ChannelValue | BaseException | None = None,
        on_subscribe: Sequence[Update] = (),
        during_write: Sequence[Update] = (),
        after_write: Sequence[Update] = (),
        write_raises: BaseException | None = None,
        subscribe_raises: BaseException | None = None,
    ) -> None:
        self.reading = _value(0) if reading is None else reading
        self.on_subscribe = list(on_subscribe)
        self.during_write = list(during_write)
        self.after_write = list(after_write)
        self.write_raises = write_raises
        self.subscribe_raises = subscribe_raises
        self.calls: list[str] = []
        self.write_kwargs: dict[str, Any] = {}
        self.written = asyncio.Event()
        self._callback: Callable[[ChannelValue], None] | None = None
        self._on_connection: Callable[[bool], None] | None = None

    def deliver(self, update: Update) -> None:
        if isinstance(update, _Drop):
            assert self._on_connection is not None
            self._on_connection(False)
            return
        assert self._callback is not None
        self._callback(update)

    async def subscribe(
        self,
        channel_address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        self.calls.append(f"subscribe {channel_address}")
        if self.subscribe_raises is not None:
            raise self.subscribe_raises
        self._callback = callback
        self._on_connection = on_connection
        if on_connection is not None:
            on_connection(True)
        for update in self.on_subscribe:
            self.deliver(update)
        return "sub-1"

    async def read_channel(self, channel_address: str) -> ChannelValue:
        self.calls.append(f"read {channel_address}")
        if isinstance(self.reading, BaseException):
            raise self.reading
        return self.reading

    async def write_channel_checked(
        self, channel_address: str, value: Any, **kwargs: Any
    ) -> ChannelWriteResult:
        self.calls.append(f"write {channel_address}={value}")
        self.write_kwargs = kwargs
        self.written.set()
        if self.write_raises is not None:
            raise self.write_raises
        for update in self.during_write:
            self.deliver(update)
        self._play(self.after_write)
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=WriteOutcome.UNREQUESTED,
        )

    def _play(self, updates: list[Update]) -> None:
        """Deliver ``updates`` one loop turn apart, in order."""
        if not updates:
            return
        loop = asyncio.get_running_loop()
        head, *rest = updates

        def step() -> None:
            self.deliver(head)
            self._play(rest)

        loop.call_soon(step)

    async def unsubscribe(self, subscription_id: str) -> None:
        self.calls.append(f"unsubscribe {subscription_id}")

    @property
    def writes(self) -> list[str]:
        return [call for call in self.calls if call.startswith("write ")]

    @property
    def unsubscribes(self) -> list[str]:
        return [call for call in self.calls if call.startswith("unsubscribe ")]


# --- the act ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "timeout_s", [0.0, -1.0, math.inf, math.nan], ids=["zero", "negative", "inf", "nan"]
)
def test_an_act_refuses_a_bound_that_is_not_positive(timeout_s: float) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        _act(timeout_s=timeout_s)


@pytest.mark.parametrize("field", ["command", "status"])
def test_an_act_refuses_an_empty_address(field: str) -> None:
    addresses = {"command": COMMAND, "status": STATUS, field: ""}
    with pytest.raises(ValueError, match=field):
        CommandAct(**addresses, idle=0, timeout_s=BOUND_S)


# --- the straight path ----------------------------------------------------------------


async def test_the_status_is_subscribed_and_read_before_the_command_is_written() -> None:
    connector = Scripted(after_write=[_value(1), _value(0)])
    await fire(connector, _act(), 1)
    assert connector.calls == [
        f"subscribe {STATUS}",
        f"read {STATUS}",
        f"write {COMMAND}=1",
        "unsubscribe sub-1",
    ]


async def test_an_act_ends_on_a_non_idle_update_followed_by_an_idle_one() -> None:
    connector = Scripted(after_write=[_value(1), _value(1), _value(0)])
    loop = asyncio.get_running_loop()
    started = loop.time()
    await fire(connector, _act(), 1)
    # Ended by the idle update, not by the bound running out.
    assert loop.time() - started < BOUND_S


async def test_updates_delivered_during_the_write_count() -> None:
    connector = Scripted(during_write=[_value(1), _value(0)])
    result = await fire(connector, _act(), 1)
    assert result.outcome is WriteOutcome.UNREQUESTED


async def test_an_idle_update_alone_does_not_end_the_act() -> None:
    connector = Scripted(after_write=[_value(0), _value(0)])
    with pytest.raises(CommandNoEffectError):
        await fire(connector, _act(), 1)


async def test_the_command_is_written_with_confirm_false() -> None:
    connector = Scripted(after_write=[_value(1), _value(0)])
    await fire(connector, _act(), 1)
    assert connector.write_kwargs == {"confirm": False}


async def test_the_write_result_is_returned() -> None:
    connector = Scripted(after_write=[_value(1), _value(0)])
    result = await fire(connector, _act(), 1)
    assert result.channel_address == COMMAND
    assert result.value_written == 1
    assert result.outcome is WriteOutcome.UNREQUESTED


# --- fail closed ----------------------------------------------------------------------


async def test_a_status_that_is_not_idle_refuses_the_act_and_writes_nothing() -> None:
    connector = Scripted(reading=_value(1))
    with pytest.raises(CommandBusyError) as caught:
        await fire(connector, _act(), 1)
    assert isinstance(caught.value, ChannelError)
    assert caught.value.reading == 1
    assert connector.writes == []
    assert connector.unsubscribes == ["unsubscribe sub-1"]


async def test_a_non_idle_update_before_the_write_refuses_the_act() -> None:
    connector = Scripted(reading=_value(0), on_subscribe=[_value(1)])
    with pytest.raises(CommandBusyError):
        await fire(connector, _act(), 1)
    assert connector.writes == []


async def test_no_non_idle_update_in_the_start_window_is_no_effect() -> None:
    connector = Scripted()
    with pytest.raises(CommandNoEffectError) as caught:
        await fire(connector, _act(), 1, start_s=BOUND_S / 4)
    assert isinstance(caught.value, TimeoutError)
    assert caught.value.last == 0
    assert caught.value.start_s == BOUND_S / 4
    assert connector.writes == [f"write {COMMAND}=1"]


async def test_idle_not_back_in_the_bound_is_a_timeout() -> None:
    connector = Scripted(after_write=[_value(1), _value(2)])
    with pytest.raises(CommandTimeoutError) as caught:
        await fire(connector, _act(), 1)
    assert isinstance(caught.value, TimeoutError)
    assert caught.value.last == 2
    assert caught.value.timeout_s == BOUND_S
    assert connector.unsubscribes == ["unsubscribe sub-1"]


async def test_each_error_names_command_status_and_bound() -> None:
    act = _act(idle="Done")
    busy = CommandBusyError(act, "Moving")
    assert str(busy) == (
        f"{COMMAND} not written: {STATUS} reads Moving, not Done; the previous act is still running"
    )
    no_effect = CommandNoEffectError(act, "Done", 0.05)
    assert str(no_effect) == f"{COMMAND} was written and {STATUS} did not leave Done within 0.05s"
    timeout = CommandTimeoutError(act, "Moving", BOUND_S)
    assert str(timeout) == (
        f"{COMMAND} was written and {STATUS} did not return to Done within {BOUND_S:g}s "
        "(last: Moving)"
    )
    cause = ChannelUnreachableError(STATUS)
    status = CommandStatusError(act, cause)
    assert str(status) == (
        f"{STATUS} of {COMMAND} cannot be watched: ChannelUnreachableError: {cause}"
    )
    assert isinstance(status, ChannelReadFailedError)
    assert status.addresses == [STATUS]
    assert status.causes == {STATUS: cause}
    reasoned = CommandStatusError(act, reason="the connection dropped")
    assert str(reasoned) == f"{STATUS} of {COMMAND} cannot be watched: the connection dropped"
    assert reasoned.causes == {}
    for error in (busy, no_effect, timeout, status, reasoned):
        assert error.act is act


async def test_a_refused_write_propagates_and_unsubscribes() -> None:
    refusal = ChannelWriteBlockedError(COMMAND, RefusalReason.LIMITS, message="refused")
    connector = Scripted(write_raises=refusal)
    with pytest.raises(ChannelWriteBlockedError) as caught:
        await fire(connector, _act(), 1)
    assert caught.value is refusal
    assert connector.unsubscribes == ["unsubscribe sub-1"]


# --- a status that cannot be watched ---------------------------------------------------


class _NoSubscribe:
    """A connector that has no ``subscribe``, as the connector-host proxy has none."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    async def read_channel(self, _channel_address: str) -> ChannelValue:
        return _value(0)

    async def write_channel_checked(self, channel_address: str, _sent: Any, **_kwargs: Any) -> Any:
        self.writes.append(channel_address)


async def test_a_connector_without_subscribe_is_refused_before_the_write() -> None:
    connector = _NoSubscribe()
    with pytest.raises(CommandStatusError):
        await fire(connector, _act(), 1)
    assert connector.writes == []


@pytest.mark.parametrize("where", ["subscribe", "read", "no value"])
async def test_a_status_read_that_fails_is_a_status_error(where: str) -> None:
    failure = ChannelUnreachableError(STATUS)
    connector = Scripted(
        subscribe_raises=failure if where == "subscribe" else None,
        reading={"read": failure, "no value": _value(None), "subscribe": None}[where],
    )
    with pytest.raises(CommandStatusError) as caught:
        await fire(connector, _act(), 1)
    if where != "no value":
        assert caught.value.causes == {STATUS: failure}
    assert connector.writes == []


@pytest.mark.parametrize("when", ["before the write", "during the write", "after the write"])
async def test_a_dropped_status_fails_the_wait_at_once(when: str) -> None:
    connector = Scripted(
        on_subscribe=[DROP] if when == "before the write" else (),
        during_write=[_value(1), DROP] if when == "during the write" else (),
        after_write=[_value(1), DROP] if when == "after the write" else (),
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(CommandStatusError):
        await fire(connector, _act(), 1)
    # Failed by the drop, not by the bound running out.
    assert loop.time() - started < BOUND_S
    assert len(connector.writes) == (0 if when == "before the write" else 1)
    assert connector.unsubscribes == ["unsubscribe sub-1"]


async def test_an_update_with_no_value_fails_the_wait() -> None:
    connector = Scripted(after_write=[_value(1), _value(None)])
    with pytest.raises(CommandStatusError):
        await fire(connector, _act(), 1)


async def test_a_label_idle_against_an_unlabelled_reading_is_refused_before_the_write() -> None:
    connector = Scripted(reading=_value(1))
    with pytest.raises(CommandStatusError, match="label"):
        await fire(connector, _act(idle="Done"), 1)
    assert connector.writes == []


@pytest.mark.parametrize(
    ("idle", "busy", "done"),
    [
        (False, _value(True), _value(False)),
        ("Done", _value(0, "Moving"), _value(1, "Done")),
        (1, _value(0, "Moving"), _value(1, "Done")),
        (0, _value(3), _value(0)),
        (2.5, _value(3.5), _value(2.5 * (1 + 1e-7))),
        (True, _value(0), _value(1)),
    ],
    ids=["bool", "enum label", "enum index", "int", "float", "true is one"],
)
async def test_idle_compares_by_label_index_and_number(
    idle: Any, busy: ChannelValue, done: ChannelValue
) -> None:
    connector = Scripted(reading=done, after_write=[busy, done])
    result = await fire(connector, _act(idle=idle), 1)
    assert result.outcome is WriteOutcome.UNREQUESTED


# --- cancellation ---------------------------------------------------------------------


class _Blocking(Scripted):
    """A scripted connector whose subscribe or read never answers until cancelled."""

    def __init__(self, blocks_at: str) -> None:
        super().__init__()
        self.blocks_at = blocks_at
        self.reached = asyncio.Event()

    async def subscribe(self, channel_address: str, callback: Any, **kwargs: Any) -> str:
        if self.blocks_at == "subscribe":
            self.calls.append(f"subscribe {channel_address}")
            self.reached.set()
            await asyncio.Event().wait()
        return await super().subscribe(channel_address, callback, **kwargs)

    async def read_channel(self, channel_address: str) -> ChannelValue:
        if self.blocks_at == "read":
            self.calls.append(f"read {channel_address}")
            self.reached.set()
            await asyncio.Event().wait()
        return await super().read_channel(channel_address)


@pytest.mark.parametrize("blocks_at", ["subscribe", "read"])
async def test_cancel_before_the_write_writes_nothing_and_unsubscribes(blocks_at: str) -> None:
    connector = _Blocking(blocks_at)
    task = asyncio.create_task(fire(connector, _act(), 1))
    await connector.reached.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connector.writes == []
    # A subscribe cancelled before it answered left no subscription to cancel.
    expected = [] if blocks_at == "subscribe" else ["unsubscribe sub-1"]
    assert connector.unsubscribes == expected


async def test_cancel_during_the_wait_unsubscribes_and_writes_nothing_more() -> None:
    connector = Scripted(after_write=[_value(1)])
    task = asyncio.create_task(fire(connector, _act(timeout_s=60.0), 1))
    await connector.written.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connector.writes == [f"write {COMMAND}=1"]
    assert connector.unsubscribes == ["unsubscribe sub-1"]


@pytest.mark.parametrize(
    "connector",
    [
        Scripted(after_write=[_value(1), _value(0)]),
        Scripted(reading=_value(1)),
        Scripted(reading=ChannelUnreachableError(STATUS)),
        Scripted(after_write=[_value(1)]),
        Scripted(after_write=[_value(1), DROP]),
        Scripted(write_raises=RuntimeError("lost")),
    ],
    ids=["finished", "busy", "read failed", "timed out", "dropped", "write raised"],
)
async def test_every_exit_unsubscribes_exactly_once(connector: Scripted) -> None:
    with contextlib.suppress(Exception):
        await fire(connector, _act(), 1, start_s=BOUND_S / 4)
    assert connector.unsubscribes == ["unsubscribe sub-1"]
