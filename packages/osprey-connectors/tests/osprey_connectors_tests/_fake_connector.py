"""A test-only control system: an in-memory store that tells the truth about connection state.

:class:`FakeConnector` honours the :class:`ControlSystemConnector` contract — the
writes gate, limits validation, the six write outcomes, ``subscribe`` with
``on_connection``, ``wait_stale`` — over a plain dictionary. It simulates no
machine: a channel holds exactly what was last put into it (``0.0`` until then),
and every reading is that value, with no noise and no derived channels. A test
that needs a simulated machine serves the simulator instead.

What it adds over a real connector are the hooks a test drives the control
system with:

* :meth:`FakeConnector.set_value` — the control system changing by itself;
* :meth:`FakeConnector.drop` / :meth:`FakeConnector.restore` — a channel going
  away and coming back, reported to its subscribers as ``on_connection(False)``
  and ``on_connection(True)`` followed by the value it then holds;
* a ``never_connects`` list (a ``connect()`` config key) — names that never
  connect until restored: a subscription reports nothing and a read raises
  ``ChannelUnreachableError``, the channel-failure root;
* :meth:`FakeConnector.go_stale` — makes every :meth:`FakeConnector.wait_stale`
  caller raise ``ConnectorUnavailableError``, the lost-connector root, standing
  in for a client transport that has gone stale.

Every delivery happens on the event loop's thread, synchronously inside the call
that caused it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from osprey_connectors.control_system._base import (
    ChannelMetadata,
    ChannelValue,
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
    values_match,
)
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError

__all__ = ["FakeConnector", "FakeSubscription"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FakeSubscription:
    """One subscription: the address, its value callback, its connection callback."""

    address: str
    callback: Callable[[ChannelValue], None]
    on_connection: Callable[[bool], None] | None = None


class FakeConnector(ControlSystemConnector):
    """An in-memory control system with connection-state hooks for tests."""

    def __init__(self) -> None:
        self._connected = False
        self._state: dict[str, Any] = {}
        self._subscriptions: dict[str, FakeSubscription] = {}
        self._subscription_count = 0
        self._never_connects: set[str] = set()
        self._dropped: set[str] = set()
        self._stale = asyncio.Event()
        self._response_delay = 0.0

    # -- lifecycle -----------------------------------------------------------

    async def connect(self, config: dict[str, Any]) -> None:
        """Connect.

        Args:
            config: Optional keys ``never_connects`` (channel names that never
                connect) and ``response_delay_ms`` (a delay before every read
                and write, default 0). Every other key is accepted and ignored,
                so the fake can stand in for any connector type's section.
        """
        self._response_delay = float(config.get("response_delay_ms", 0)) / 1000.0
        self._never_connects = set(config.get("never_connects") or ())

        from osprey_connectors.control_system._limits_validator import LimitsValidator

        self._limits_validator = LimitsValidator.from_config(connector_type=self._connector_type)
        self._connected = True

    async def disconnect(self) -> None:
        self._state.clear()
        self._subscriptions.clear()
        self._dropped.clear()
        self._connected = False

    # -- connection state ----------------------------------------------------

    def _is_up(self, channel_address: str) -> bool:
        """Whether ``channel_address`` is connected (neither never-connecting nor dropped)."""
        return channel_address not in self._never_connects and channel_address not in self._dropped

    def _require_up(self, channel_address: str) -> None:
        if not self._is_up(channel_address):
            raise ChannelUnreachableError(
                channel_address, f"Fake channel '{channel_address}' is not connected"
            )

    def drop(self, channel_address: str) -> None:
        """Take ``channel_address`` down: its subscribers are told ``on_connection(False)``.

        While down the channel delivers no values and a read of it raises
        ``ChannelUnreachableError``; its store still changes. Dropping a channel that is
        already down reports nothing.
        """
        was_up = self._is_up(channel_address)
        self._dropped.add(channel_address)
        if not was_up:
            return
        for sub in list(self._subscriptions.values()):
            if sub.address == channel_address:
                self._report_connection(sub, False)

    def restore(self, channel_address: str) -> None:
        """Bring ``channel_address`` up: ``on_connection(True)``, then the value it holds.

        Undoes :meth:`drop`, and also connects a ``never_connects`` name — a
        channel that comes up late. Restoring a channel that is already up
        reports nothing.
        """
        if self._is_up(channel_address):
            return
        self._dropped.discard(channel_address)
        self._never_connects.discard(channel_address)
        for sub_id, sub in list(self._subscriptions.items()):
            if sub.address == channel_address:
                self._report_connection(sub, True)
                self._deliver(sub_id, sub)

    async def wait_stale(self) -> None:
        """Raise ``ConnectorUnavailableError`` once :meth:`go_stale` has been called; until then, wait."""
        await self._stale.wait()
        raise ConnectorUnavailableError("Fake transport has gone stale")

    def go_stale(self) -> None:
        """Make every :meth:`wait_stale` caller raise, now and from now on (loop thread only)."""
        self._stale.set()

    # -- the store -----------------------------------------------------------

    def _read_value(self, channel_address: str) -> ChannelValue:
        """The reading for ``channel_address``: exactly what the store holds."""
        now = datetime.now(UTC)
        return ChannelValue(
            value=self._state.setdefault(channel_address, 0.0),
            timestamp=now,
            metadata=ChannelMetadata(
                units="", description=f"Fake channel: {channel_address}", timestamp=now
            ),
        )

    def _put(self, channel_address: str, value: Any) -> None:
        self._state[channel_address] = value
        self._notify([channel_address])

    def set_value(self, channel_address: str, value: Any) -> None:
        """Put ``value`` straight into the store and tell the channel's subscribers.

        The control system changing by itself: no writes gate, no limits check,
        no confirmation and no delay.
        """
        self._put(channel_address, value)

    def current_value_reader(self) -> Callable[[str], Any] | None:
        return lambda channel_address: self._read_value(channel_address).value

    async def _confirming_read(self, channel_address: str) -> ChannelValue:
        """Read a channel back to confirm a write (the seam a test patches to fail one)."""
        await asyncio.sleep(self._response_delay)
        self._require_up(channel_address)
        return self._read_value(channel_address)

    # -- reads ---------------------------------------------------------------

    async def read_channel(
        self,
        channel_address: str,
        timeout: float | None = None,  # noqa: ARG002 - the connector read signature
    ) -> ChannelValue:
        await asyncio.sleep(self._response_delay)
        self._require_up(channel_address)
        return self._read_value(channel_address)

    async def read_multiple_channels(
        self, channel_addresses: list[str], timeout: float | None = None
    ) -> dict[str, ChannelValue]:
        return await self._read_concurrently(channel_addresses, timeout)

    async def get_metadata(self, channel_address: str) -> ChannelMetadata:
        self._require_up(channel_address)
        return self._read_value(channel_address).metadata

    async def validate_channel(self, channel_address: str) -> bool:
        return self._is_up(channel_address)

    # -- writes --------------------------------------------------------------

    async def write_channel(
        self,
        channel_address: str,
        value: Any,
        timeout: float | None = None,  # noqa: ARG002 - the connector write signature
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Validate limits, put the value, and confirm it by re-reading unless told not to."""
        if self._limits_validator:
            from osprey_connectors.errors import ChannelLimitsViolationError

            try:
                self._limits_validator.validate(
                    channel_address, value, read_current=self.current_value_reader()
                )
            except ChannelLimitsViolationError:
                raise
            except Exception as e:
                return self._validation_refusal(channel_address, value, e)

        if confirm is None:
            confirm = self._resolve_confirm(channel_address)

        await asyncio.sleep(self._response_delay)
        self._require_up(channel_address)
        try:
            self._put(channel_address, value)
        except Exception as e:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"Fake write failed: {e}",
            )

        if not confirm:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
                notes="Confirmation not requested (fake)",
            )

        try:
            observed = await self._confirming_read(channel_address)
        except Exception as e:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=f"Fake confirming read failed: {e}",
                notes=f"Confirming read raised: {e} (fake)",
            )

        outcome = (
            WriteOutcome.CONFIRMED if values_match(value, observed.value) else WriteOutcome.MISMATCH
        )
        return ChannelWriteResult(
            channel_address=channel_address,
            value_written=value,
            outcome=outcome,
            observed_value=observed.value,
            notes=f"Observed {observed.value}, sent {value} (fake)",
        )

    # -- subscriptions -------------------------------------------------------

    async def subscribe(
        self,
        channel_address: str,
        callback: Callable[[ChannelValue], None],
        *,
        on_connection: Callable[[bool], None] | None = None,
    ) -> str:
        """Subscribe; a connected channel reports ``True`` then its value before this returns.

        Afterwards the callback receives the stored value on every change — a
        write or a :meth:`set_value` — and on :meth:`restore`. A channel that is
        not connected reports nothing until it is restored.
        """
        self._subscription_count += 1
        sub_id = f"fake_{channel_address}_{self._subscription_count}"
        sub = FakeSubscription(channel_address, callback, on_connection)
        self._subscriptions[sub_id] = sub
        if self._is_up(channel_address):
            self._report_connection(sub, True)
            self._deliver(sub_id, sub)
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        self._subscriptions.pop(subscription_id, None)

    def _notify(self, channel_addresses: Iterable[str]) -> None:
        """Deliver what each connected one of ``channel_addresses`` holds to its subscribers.

        A callback that raises is logged and skipped, so one bad subscriber
        cannot fail a write or starve the others.
        """
        addresses = {address for address in channel_addresses if self._is_up(address)}
        for sub_id, sub in list(self._subscriptions.items()):
            if sub.address in addresses:
                self._deliver(sub_id, sub)

    @staticmethod
    def _report_connection(sub: FakeSubscription, up: bool) -> None:
        if sub.on_connection is None:
            return
        try:
            sub.on_connection(up)
        except Exception:
            logger.warning("Fake connection callback failed for %s", sub.address, exc_info=True)

    def _deliver(self, sub_id: str, sub: FakeSubscription) -> None:
        """Deliver the stored value to one subscription still registered under ``sub_id``."""
        if self._subscriptions.get(sub_id) is not sub:
            return
        try:
            sub.callback(self._read_value(sub.address))
        except Exception:
            logger.warning("Fake subscription callback failed for %s", sub.address, exc_info=True)
