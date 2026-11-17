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

    def _is_up(self, address: str) -> bool:
        """Whether ``address`` is connected (neither never-connecting nor dropped)."""
        return address not in self._never_connects and address not in self._dropped

    def _require_up(self, address: str) -> None:
        if not self._is_up(address):
            raise ChannelUnreachableError(address, f"Fake channel '{address}' is not connected")

    def drop(self, address: str) -> None:
        """Take ``address`` down: its subscribers are told ``on_connection(False)``.

        While down the channel delivers no values and a read of it raises
        ``ChannelUnreachableError``; its store still changes. Dropping a channel that is
        already down reports nothing.
        """
        was_up = self._is_up(address)
        self._dropped.add(address)
        if not was_up:
            return
        for sub in list(self._subscriptions.values()):
            if sub.address == address:
                self._report_connection(sub, False)

    def restore(self, address: str) -> None:
        """Bring ``address`` up: ``on_connection(True)``, then the value it holds.

        Undoes :meth:`drop`, and also connects a ``never_connects`` name — a
        channel that comes up late. Restoring a channel that is already up
        reports nothing.
        """
        if self._is_up(address):
            return
        self._dropped.discard(address)
        self._never_connects.discard(address)
        for sub_id, sub in list(self._subscriptions.items()):
            if sub.address == address:
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

    def _read_value(self, address: str) -> ChannelValue:
        """The reading for ``address``: exactly what the store holds."""
        now = datetime.now(UTC)
        return ChannelValue(
            value=self._state.setdefault(address, 0.0),
            sampled_at=now,
            metadata=ChannelMetadata(units="", description=f"Fake channel: {address}"),
        )

    def _put(self, address: str, value: Any) -> None:
        self._state[address] = value
        self._notify([address])

    def set_value(self, address: str, value: Any) -> None:
        """Put ``value`` straight into the store and tell the channel's subscribers.

        The control system changing by itself: no writes gate, no limits check,
        no confirmation and no delay.
        """
        self._put(address, value)

    def current_value_reader(self) -> Callable[[str], Any] | None:
        return lambda address: self._read_value(address).value

    async def _confirming_read(self, address: str) -> ChannelValue:
        """Read a channel back to confirm a write (the seam a test patches to fail one)."""
        await asyncio.sleep(self._response_delay)
        self._require_up(address)
        return self._read_value(address)

    # -- reads ---------------------------------------------------------------

    async def read_channel(
        self,
        address: str,
        *,
        timeout_s: float | None = None,  # noqa: ARG002 - the connector read signature
    ) -> ChannelValue:
        await asyncio.sleep(self._response_delay)
        self._require_up(address)
        return self._read_value(address)

    async def read_multiple_channels(
        self, addresses: list[str], *, timeout_s: float | None = None
    ) -> dict[str, ChannelValue]:
        return await self._read_concurrently(addresses, timeout_s)

    async def get_metadata(self, address: str) -> ChannelMetadata:
        self._require_up(address)
        return self._read_value(address).metadata

    async def validate_channel(self, address: str) -> bool:
        return self._is_up(address)

    # -- writes --------------------------------------------------------------

    async def write_channel(
        self,
        address: str,
        value: Any,
        *,
        timeout_s: float | None = None,  # noqa: ARG002 - the connector write signature
        confirm: bool | None = None,
    ) -> ChannelWriteResult:
        """Validate limits, put the value, and confirm it by re-reading unless told not to."""
        if self._limits_validator:
            from osprey_connectors.errors import ChannelLimitsViolationError

            try:
                self._limits_validator.validate(
                    address, value, read_current=self.current_value_reader()
                )
            except ChannelLimitsViolationError:
                raise
            except Exception as e:
                return self._validation_refusal(address, value, e)

        if confirm is None:
            confirm = self._resolve_confirm(address)

        await asyncio.sleep(self._response_delay)
        self._require_up(address)
        try:
            self._put(address, value)
        except Exception as e:
            return ChannelWriteResult(
                address=address,
                value_written=value,
                outcome=WriteOutcome.FAILED,
                error_message=f"Fake write failed: {e}",
            )

        if not confirm:
            return ChannelWriteResult(
                address=address,
                value_written=value,
                outcome=WriteOutcome.UNREQUESTED,
                notes="Confirmation not requested (fake)",
            )

        try:
            observed = await self._confirming_read(address)
        except Exception as e:
            return ChannelWriteResult(
                address=address,
                value_written=value,
                outcome=WriteOutcome.UNCONFIRMED,
                error_message=f"Fake confirming read failed: {e}",
                notes=f"Confirming read raised: {e} (fake)",
            )

        outcome = (
            WriteOutcome.CONFIRMED if values_match(value, observed.value) else WriteOutcome.MISMATCH
        )
        return ChannelWriteResult(
            address=address,
            value_written=value,
            outcome=outcome,
            observed_value=observed.value,
            notes=f"Observed {observed.value}, sent {value} (fake)",
        )

    # -- subscriptions -------------------------------------------------------

    async def subscribe(
        self,
        address: str,
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
        sub_id = f"fake_{address}_{self._subscription_count}"
        sub = FakeSubscription(address, callback, on_connection)
        self._subscriptions[sub_id] = sub
        if self._is_up(address):
            self._report_connection(sub, True)
            self._deliver(sub_id, sub)
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        self._subscriptions.pop(subscription_id, None)

    def _notify(self, addresses: Iterable[str]) -> None:
        """Deliver what each connected one of ``addresses`` holds to its subscribers.

        A callback that raises is logged and skipped, so one bad subscriber
        cannot fail a write or starve the others.
        """
        up = {address for address in addresses if self._is_up(address)}
        for sub_id, sub in list(self._subscriptions.items()):
            if sub.address in up:
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
