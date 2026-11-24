"""A minimal async double for the OSPREY connector the bridge's devices drive.

No Channel Access and no real IOC: ``read_channel`` and
``write_channel_checked`` answer from a mutable readback dict, and the double
logs every call, since the device layer's point is that every read and write
goes through the connector. ``subscribe`` delivers the updates a write's stated
response names, so a command fired through it sees its status move.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeChannelValue:
    """Stand-in for ``osprey_connectors.control_system._base.ChannelValue``."""

    value: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class _FakeWriteResult:
    """Stand-in for a ``ChannelWriteResult`` returned by ``write_channel_checked``.

    Only the fields the device reads: the single owned ``outcome`` word and the
    ``observed_value`` the confirming re-read returned. ``write_channel_checked``
    only ever returns ``confirmed`` or ``unrequested`` — every other outcome
    raises before the device sees a result.
    """

    channel_address: str
    value_written: Any
    outcome: str = "confirmed"
    observed_value: Any = None


class FakeConnector:
    """A minimal async double for ``ControlSystemConnector``.

    ``read_channel`` returns whatever ``readbacks[address]`` currently holds
    (a plain mutable dict, so a test can move the simulated readback after
    the fact). ``write_channel_checked`` either records the call and updates
    the readback (default), or raises/behaves however the test configures it
    via ``write_side_effect``/``echo_readback``.
    """

    def __init__(self, readbacks: dict[str, float] | None = None) -> None:
        self.readbacks: dict[str, float] = dict(readbacks or {})
        self.write_calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.read_calls: list[str] = []
        self.write_side_effect: Exception | None = None
        self.echo_readback = True
        # Maps a written channel address to the (possibly separate) readback
        # address it should echo into. Defaults to echoing into the same
        # address that was written.
        self.echo_target: dict[str, str] = {}
        # The outcome the returned result carries. ``confirmed`` is the default;
        # ``unrequested`` simulates a channel configured ``confirm: false``, the
        # only other outcome ``write_channel_checked`` returns rather than raises.
        self.write_outcome = "confirmed"
        # A write to a key here is answered by the updates its value lists,
        # delivered to the subscribers of the address it names and left as
        # that address's reading: the status a command's act moves.
        self.responses: dict[str, tuple[str, list[Any]]] = {}
        self.subscribers: dict[str, tuple[str, Callable[[Any], None]]] = {}
        # A read of an address here waits until it is cancelled; ``read_held``
        # and ``read_held_here`` are set once such a read is waiting, for a
        # test on another thread and on the read's own loop, and
        # ``held_read_cancelled`` once one has been cancelled.
        self.hold_reads: set[str] = set()
        self.read_held = threading.Event()
        self.read_held_here: asyncio.Event | None = None
        self.held_read_cancelled = threading.Event()

    async def subscribe(
        self,
        channel_address: str,
        callback: Callable[[Any], None],
        *,
        on_connection: Callable[[bool], None] | None = None,  # noqa: ARG002 - the connector subscribe signature
    ) -> str:
        subscription = f"sub-{len(self.subscribers)}"
        self.subscribers[subscription] = (channel_address, callback)
        return subscription

    async def unsubscribe(self, subscription: str) -> None:
        self.subscribers.pop(subscription, None)

    async def read_channel(self, channel_address: str, timeout: float | None = None):  # noqa: ARG002 - the connector read signature
        self.read_calls.append(channel_address)
        if channel_address in self.hold_reads:
            self.read_held.set()
            if self.read_held_here is not None:
                self.read_held_here.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.held_read_cancelled.set()
                raise
        return FakeChannelValue(value=self.readbacks.get(channel_address))

    def _respond(self, channel_address: str) -> None:
        if channel_address not in self.responses:
            return
        status, updates = self.responses[channel_address]
        for value in updates:
            self.readbacks[status] = value
            for address, callback in list(self.subscribers.values()):
                if address == status:
                    callback(FakeChannelValue(value=value))

    async def write_channel_checked(self, channel_address: str, value: Any, **kwargs: Any):
        self.write_calls.append((channel_address, value, kwargs))
        if self.write_side_effect is not None:
            raise self.write_side_effect
        if self.echo_readback:
            target = self.echo_target.get(channel_address, channel_address)
            self.readbacks[target] = value
        self._respond(channel_address)
        # Like a real connector, the value the result observed is of the channel
        # that was WRITTEN — not of any separate readback channel.
        return _FakeWriteResult(
            channel_address,
            value,
            outcome=self.write_outcome,
            observed_value=self.readbacks.get(channel_address),
        )
