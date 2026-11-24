"""Fire a command channel and wait for its status channel to say the act is over.

A command is written once; the act it starts is reported by a status channel
that reads a stated ``idle`` value when nothing runs. :func:`fire` subscribes
to the status, refuses when it is not idle, writes the command, and returns
once the status has left idle and come back, within the act's bound. The wait
is the write's confirmation, a stronger one than a re-read, so the one write
:func:`fire` makes passes ``confirm=False``; every other write of the command
is confirmed as its configuration says.

Every failure is closed and typed: :class:`CommandBusyError` (nothing was
written), :class:`CommandNoEffectError` and :class:`CommandTimeoutError` (the
command was written, the status did not move or did not come back), and
:class:`CommandStatusError` (the status cannot be watched). The status is
compared with :func:`~osprey_connectors.control_system.values_match`, the rule
a write's confirmation uses.

This module imports only the std-lib, ``osprey_connectors.errors``, the
package's contract alias and the control-system contract, and takes any
connector with awaitable ``subscribe(address, callback, *, on_connection)``,
``unsubscribe(id)``, ``read_channel(address)`` and
``write_channel_checked(address, value, confirm=...)``.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass

from osprey_connectors._contract_types import ContractAny
from osprey_connectors.control_system import ChannelValue, ChannelWriteResult, values_match
from osprey_connectors.errors import ChannelError, ChannelReadFailedError

__all__ = [
    "CommandAct",
    "CommandBusyError",
    "CommandNoEffectError",
    "CommandStatusError",
    "CommandTimeoutError",
    "fire",
]


@dataclass(frozen=True)
class CommandAct:
    """One command channel, the status channel that reports its act, the idle value and the bound.

    Attributes:
        command: The address the value is written to.
        status: The address of the channel that reports the act.
        idle: What ``status`` reads when nothing runs: a label, a number or a bool.
        timeout_s: The longest one act may take, in seconds.
    """

    command: str
    status: str
    idle: str | int | float | bool
    timeout_s: float

    def __post_init__(self) -> None:
        """Refuse an empty address and a bound that is not a positive finite number.

        Raises:
            ValueError: ``command`` or ``status`` is empty, or ``timeout_s`` is
                not finite and greater than zero.
        """
        for name in ("command", "status"):
            if not getattr(self, name):
                raise ValueError(f"a command act needs a {name} address")
        if not (math.isfinite(self.timeout_s) and self.timeout_s > 0):
            raise ValueError(
                f"timeout_s must be a positive number of seconds, not {self.timeout_s}"
            )


class CommandBusyError(ChannelError):
    """The status was not idle before the write; nothing was written.

    Attributes:
        act: The act that was refused.
        reading: The status value that was not idle.
    """

    def __init__(self, act: CommandAct, reading: object) -> None:
        self.act = act
        self.reading = reading
        super().__init__(
            act.command,
            f"{act.command} not written: {act.status} reads {reading}, not {act.idle}; "
            "the previous act is still running",
        )


class CommandNoEffectError(TimeoutError):
    """The command was written and its status did not leave idle in the start window.

    Attributes:
        act: The act whose status did not move.
        last: The last status value seen.
        start_s: The seconds the status had to leave idle.
    """

    def __init__(self, act: CommandAct, last: object, start_s: float) -> None:
        self.act = act
        self.last = last
        self.start_s = start_s
        super().__init__(
            f"{act.command} was written and {act.status} did not leave {act.idle} "
            f"within {start_s:g}s"
        )


class CommandTimeoutError(TimeoutError):
    """The command was written, the act started, and its status did not return to idle in time.

    Attributes:
        act: The act that did not finish.
        last: The last status value seen.
        timeout_s: The act's bound, in seconds.
    """

    def __init__(self, act: CommandAct, last: object, timeout_s: float) -> None:
        self.act = act
        self.last = last
        self.timeout_s = timeout_s
        super().__init__(
            f"{act.command} was written and {act.status} did not return to {act.idle} "
            f"within {timeout_s:g}s (last: {last})"
        )


class CommandStatusError(ChannelReadFailedError):
    """The status channel cannot be watched or its value cannot be compared with idle.

    Attributes:
        act: The act whose status failed.
        addresses: The status, the one failed channel.
        causes: The status to the exception that failed it; empty when nothing raised.
    """

    def __init__(
        self,
        act: CommandAct,
        cause: BaseException | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        self.act = act
        if cause is not None:
            reason = f"{type(cause).__name__}: {cause}"
        super().__init__(
            [act.status],
            f"{act.status} of {act.command} cannot be watched: {reason}",
            causes={act.status: cause} if cause is not None else None,
        )


class _Dropped:
    """The status connection dropped, as one entry in the update queue."""


_DROPPED = _Dropped()


def _is_idle(act: CommandAct, reading: object) -> bool:
    """Whether a status reading is the act's idle value.

    Raises:
        CommandStatusError: The reading has no value, or ``idle`` is a label
            and the reading carries none to compare it with.
    """
    value: object = getattr(reading, "value", None)
    if value is None:
        raise CommandStatusError(act, reason="no value")
    metadata = getattr(reading, "metadata", None)
    label: str | None = getattr(metadata, "enum_label", None)
    if isinstance(act.idle, str) and label is None:
        raise CommandStatusError(
            act, reason=f"idle is the label {act.idle!r} and the reading {value} carries no label"
        )
    return values_match(act.idle, value, enum_label=label)


async def _read_status(connector: ContractAny, act: CommandAct) -> object:
    """The status's current reading, read once.

    Raises:
        CommandStatusError: The read failed with a
            :class:`~osprey_connectors.errors.ChannelError`.
    """
    try:
        reading: object = await connector.read_channel(act.status)
    except ChannelError as exc:
        raise CommandStatusError(act, exc) from exc
    return reading


async def _next_update(
    updates: asyncio.Queue[ChannelValue | _Dropped], deadline: float
) -> ChannelValue | _Dropped | None:
    """The oldest queued update, waiting until ``deadline`` for one; ``None`` once it passes.

    An update already queued is returned whether or not the deadline has passed.
    """
    if not updates.empty():
        return updates.get_nowait()
    try:
        async with asyncio.timeout_at(deadline):
            return await updates.get()
    except TimeoutError:
        return None


async def fire(
    # Duck-typed by the module's contract: any object with the four awaitable
    # methods the module docstring names, which the package has no public type to name.
    connector: ContractAny,
    act: CommandAct,
    value: object,
    *,
    start_s: float | None = None,
) -> ChannelWriteResult:
    """Write ``value`` to the act's command and wait for its status to leave and return to idle.

    The status is subscribed and read once before anything is written; a status
    that is not idle refuses the act. The command is written once, unconfirmed,
    because the wait confirms it. The act ends on a status update that is not
    idle followed by one that is, each delivered after the write began, updates
    delivered while the write is awaited among them. The time runs on the event
    loop's clock from the write. Every exit after the subscription unsubscribes,
    and nothing is ever written a second time.

    Args:
        connector: The connector the status is watched and the command written through.
        act: The command, its status, its idle value and its bound.
        value: The value written to the command.
        start_s: Seconds the status may take to leave idle; ``None``, or more
            than ``act.timeout_s``, is ``act.timeout_s``.

    Returns:
        The write's result, whose outcome is ``unrequested``.

    Raises:
        CommandBusyError: The status was not idle before the write; nothing
            was written.
        CommandNoEffectError: The command was written and the status did not
            leave idle within ``start_s``.
        CommandTimeoutError: The status left idle and did not return within
            ``act.timeout_s`` of the write.
        CommandStatusError: The connector cannot subscribe, the subscribe or
            the read failed with a
            :class:`~osprey_connectors.errors.ChannelError`, a reading had no
            value, a label ``idle`` met a reading with no label, or the status
            connection dropped. The first three are raised before anything
            is written.
        ValueError: ``start_s`` is not a positive finite number.
        Exception: Whatever ``write_channel_checked`` raises, unchanged.
    """
    if start_s is not None and not (math.isfinite(start_s) and start_s > 0):
        raise ValueError(f"start_s must be a positive number of seconds, not {start_s}")
    start_window = act.timeout_s if start_s is None else min(start_s, act.timeout_s)
    subscribe = getattr(connector, "subscribe", None)
    if subscribe is None:
        raise CommandStatusError(act, reason="the connector cannot subscribe")

    loop = asyncio.get_running_loop()
    updates: asyncio.Queue[ChannelValue | _Dropped] = asyncio.Queue()

    def enqueue(update: ChannelValue | _Dropped) -> None:
        try:
            on_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_loop = False
        if on_loop:
            updates.put_nowait(update)
        else:
            loop.call_soon_threadsafe(updates.put_nowait, update)

    def on_connection(connected: bool) -> None:
        if not connected:
            enqueue(_DROPPED)

    try:
        subscription: str = await subscribe(act.status, enqueue, on_connection=on_connection)
    except ChannelError as exc:
        raise CommandStatusError(act, exc) from exc

    try:
        reading = await _read_status(connector, act)
        newest = reading
        while not updates.empty():
            queued = updates.get_nowait()
            if isinstance(queued, _Dropped):
                raise CommandStatusError(act, reason="the connection dropped")
            newest = queued
        for seen in (reading, newest):
            if not _is_idle(act, seen):
                raise CommandBusyError(act, getattr(seen, "value", None))

        written = loop.time()
        result: ChannelWriteResult = await connector.write_channel_checked(
            act.command, value, confirm=False
        )

        started = False
        last: object = getattr(newest, "value", None)
        while True:
            deadline = written + (act.timeout_s if started else start_window)
            update = await _next_update(updates, deadline)
            if update is None:
                if started:
                    raise CommandTimeoutError(act, last, act.timeout_s)
                raise CommandNoEffectError(act, last, start_window)
            if isinstance(update, _Dropped):
                raise CommandStatusError(act, reason="the connection dropped")
            idle = _is_idle(act, update)
            last = update.value
            if not idle:
                started = True
            elif started:
                return result
    finally:
        await connector.unsubscribe(subscription)
