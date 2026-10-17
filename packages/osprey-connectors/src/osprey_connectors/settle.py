"""Wait for a written setpoint's readback to come within the bound its facility states.

Every caller that waits for a readback after a write uses this one step: the
Bluesky worker for a plan's moves, the sandbox runtime for a script's or a
measurement's steps. The bound has one rule (:func:`settle_bound`): the
setpoint's ``tolerance`` (``absolute``, or ``relative`` times the demand's
magnitude); with none stated, on the simulator only, the readback's motion
band; a caller's floor raises the bound and never lowers it. :func:`settle`
polls each readback through the connector until it is within its bound, and
fails closed with :class:`SettleTimeoutError` when the time runs out. A
readback read that fails, or answers no value, raises :class:`SettleReadError`.

This module imports only the std-lib and ``osprey_connectors.errors``, and
takes any connector with an awaitable ``read_channel(address)`` whose result
carries ``.value``.
"""

from __future__ import annotations

import asyncio
import math
import numbers
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from osprey_connectors.errors import ChannelReadFailedError

__all__ = [
    "POLL_INTERVAL_S",
    "Landing",
    "SettleReadError",
    "SettleTimeoutError",
    "declared_tolerance",
    "settle",
    "settle_bound",
]

#: Seconds between two reads of a readback that is not yet within its bound.
POLL_INTERVAL_S = 0.05


def declared_tolerance(record: Mapping[str, Any] | None, demand: float) -> float | None:
    """The bound a ``tolerance`` record states for a move to ``demand``.

    Args:
        record: The setpoint's ``tolerance`` record, ``{absolute: x}`` in the
            channel's unit or ``{relative: f}``, a fraction of the demand.
        demand: The value the setpoint is driven to.

    Returns:
        ``|absolute|``, or ``|relative| x |demand|``; ``None`` when the record
        states neither.
    """
    if not isinstance(record, Mapping):
        return None
    if "absolute" in record:
        return abs(float(record["absolute"]))
    if "relative" in record:
        return abs(float(record["relative"])) * abs(float(demand))
    return None


def settle_bound(
    tolerance: Mapping[str, Any] | None,
    demand: float,
    *,
    motion: float = 0.0,
    simulated: bool = False,
    floor: float | None = None,
) -> float | None:
    """How close a readback must come to ``demand`` for the move to count as settled.

    Args:
        tolerance: The setpoint's ``tolerance`` record, or ``None``.
        demand: The value the setpoint is driven to.
        motion: The readback's motion band.
        simulated: Whether the machine is the simulator; only there does the
            motion band stand in for a tolerance that is not stated.
        floor: The caller's floor, or ``None``.

    Returns:
        The larger of ``floor`` and the declared bound; ``None`` when nothing
        is declared and the caller gives no floor, so there is nothing to wait for.
    """
    declared = declared_tolerance(tolerance, demand)
    if declared is None and simulated and motion > 0:
        declared = motion
    if declared is None and floor is None:
        return None
    return max(floor or 0.0, declared or 0.0)


@dataclass(frozen=True)
class Landing:
    """One written setpoint, the readback that reports it, its demand and its bound."""

    setpoint: str
    readback: str
    demand: float
    bound: float


class SettleTimeoutError(TimeoutError):
    """A readback did not come within its bound of the demand in time.

    Attributes:
        landing: The first landing that did not settle.
        last: Its readback's last reading.
    """

    def __init__(self, landing: Landing, last: Any, timeout_s: float) -> None:
        self.landing = landing
        self.last = last
        super().__init__(
            f"readback {landing.readback} did not settle to {landing.demand} "
            f"± {landing.bound:g} within {timeout_s:g}s (last read: {last})"
        )


class SettleReadError(ChannelReadFailedError):
    """A readback could not be read while settling, or answered no value.

    Attributes:
        landing: The landing whose readback could not be read.
        addresses: The readback, the one failed channel.
        causes: The readback to the exception its read raised; empty when the
            read answered no value.
    """

    def __init__(self, landing: Landing, cause: BaseException | None) -> None:
        self.landing = landing
        reason = f"{type(cause).__name__}: {cause}" if cause is not None else "no value"
        super().__init__(
            [landing.readback],
            f"readback {landing.readback} of {landing.setpoint} could not be read: {reason}",
            causes={landing.readback: cause} if cause is not None else None,
        )


async def settle(
    connector: Any,
    landings: Sequence[Landing],
    *,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    """Poll each landing's readback until it is within its bound of its demand.

    Every landing is read once per poll until it is inside; a settled landing
    is not read again. A reading with no value is a failed read; any other
    reading that is not a finite number is never inside.

    Args:
        connector: Reads each readback with ``await connector.read_channel(address)``.
        landings: The landings to wait for.
        timeout_s: Seconds from the call after which an unsettled landing fails.
        clock: The clock the deadline is measured on.
        sleep: The pause between two polls.

    Raises:
        SettleTimeoutError: A landing was not inside its bound once ``timeout_s``
            had passed, naming the first such landing in ``landings`` order.
        SettleReadError: A readback read raised or answered no value, naming
            that readback.
    """
    deadline = clock() + timeout_s
    pending = list(landings)
    while pending:
        unsettled: list[tuple[Landing, Any]] = []
        for landing in pending:
            try:
                reading = await connector.read_channel(landing.readback)
            except Exception as exc:
                raise SettleReadError(landing, exc) from exc
            value = getattr(reading, "value", None)
            if value is None:
                raise SettleReadError(landing, None)
            if not _within(value, landing):
                unsettled.append((landing, value))
        if not unsettled:
            return
        if clock() >= deadline:
            landing, last = unsettled[0]
            raise SettleTimeoutError(landing, last, timeout_s)
        pending = [landing for landing, _ in unsettled]
        await sleep(POLL_INTERVAL_S)


def _within(value: Any, landing: Landing) -> bool:
    """Whether ``value`` is a finite number within the landing's bound of its demand."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    reading = float(value)
    return math.isfinite(reading) and abs(reading - landing.demand) <= landing.bound
