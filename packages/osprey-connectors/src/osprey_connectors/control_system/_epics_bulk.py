"""The Channel Access subscription path: one record per address, on the CA loop.

``EPICSConnector.subscribe`` serves every Channel Access subscriber of one
address from one record, however many subscribers join:

* **One record per normalised address.** ``X`` and ``ca://X`` share one record
  and one pair of aioca monitors. Every piece of record state is read and
  written on the process's CA loop (:mod:`.ca_loop`), where aioca runs every
  monitor callback; subscribers get their values and connection reports posted
  to their own loops.
* **CTRL, then TIME.** The record holds two monitors: CTRL (``DBE_PROPERTY``,
  ``FORMAT_CTRL``, one element -- the record's metadata, not its payload) and
  TIME (``DBE_VALUE | DBE_ALARM``, ``FORMAT_TIME``). libca re-arms both on a
  reconnect, in the order they were opened, and the server posts one initial
  event per subscription, so each connection starts with its CTRL event. A TIME
  event that arrives before its connection's CTRL is held (the latest one) and
  delivered right after it, so every value carries the control data of the
  connection it arrived on. "Up" is reported when that CTRL event arrives,
  ahead of the connection's first value.
* **Every update, once.** Both monitors are opened with ``all_updates=True``.
  aioca's default keeps only the newest pending update, and a drop is queued
  through the loop while values are queued from libca's thread, so a loop that
  lags one reconnect would lose the new connection's CTRL event to the drop
  and never see another. The cost is that a fast monitor queues every event in
  aioca until the CA loop drains it; the handlers only build a value and post
  it, so that backlog is bounded by how far the CA loop lags, not by how fast a
  subscriber is.
* **A drop is believed only when the channel is down.** A drop that reaches the
  handler after the next connection's CTRL event is stale; it is checked
  against libca's own channel state and ignored when the channel is connected.
* **A repoint** (``CaLoop.point_at``) purges every aioca channel without a word
  to the monitors. Each record registers a re-opener: it reports the record
  down and opens both monitors again on the new context.
* **A record left without monitors** -- its re-open failed -- is down, answers
  no metadata waiter, and is opened again by the next subscriber to join it or
  by the next repoint.

aioca is handed in by the connector, so tests can inject a stand-in.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from osprey_connectors.errors import ChannelUnreachableError
from osprey_connectors.logger import get_logger

__all__ = ["CABulk", "normalise_address"]

logger = get_logger("epics_bulk")

#: The scheme prefix a CA address may carry; ``X`` and ``ca://X`` are one channel.
CA_SCHEME = "ca://"


def normalise_address(address: str) -> str:
    """The channel name of a CA address: ``ca://X`` -> ``X``; anything else unchanged."""
    return address.removeprefix(CA_SCHEME)


def _ca_write_access(aioca: Any, name: str) -> bool | None:
    """This client's write access to channel ``name``, or ``None`` if it cannot be told.

    A local libca query on the connected channel, no round trip; callable only
    on the CA loop. A stand-in aioca answers through its ``write_access`` hook.
    """
    hook = getattr(aioca, "write_access", None)
    try:
        if hook is not None:
            return None if (answer := hook(name)) is None else bool(answer)
        catools = aioca._catools
        return bool(catools.cadef.ca_write_access(catools.get_channel(name)))
    except Exception:
        return None


def _ca_channel_connected(aioca: Any, name: str) -> bool | None:
    """Whether libca holds channel ``name`` connected right now; ``None`` if it cannot be told.

    Called from record handlers only: they never run on a purged channel, and
    ``get_channel`` on a purged name would open a new channel rather than answer.
    A stand-in aioca answers through its ``channel_connected`` hook.
    """
    hook = getattr(aioca, "channel_connected", None)
    try:
        if hook is not None:
            return None if (answer := hook(name)) is None else bool(answer)
        catools = aioca._catools
        return bool(catools.cadef.ca_state(catools.get_channel(name)) == catools.cadef.cs_conn)
    except Exception:
        return None


class _Subscriber:
    __slots__ = ("callback", "loop", "on_connection")

    def __init__(
        self,
        callback: Callable[[Any], None],
        loop: asyncio.AbstractEventLoop,
        on_connection: Callable[[bool], None] | None,
    ) -> None:
        self.callback = callback
        self.loop = loop
        self.on_connection = on_connection


class _Record:
    """One address's monitors and connection state. CA loop only."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.subscribers: dict[str, _Subscriber] = {}
        self.ctrl_monitor: Any = None
        self.time_monitor: Any = None
        self.connected = False
        self.ever_connected = False
        # Down from the moment it is opened until its first CTRL event.
        self.down_since: float | None = time.monotonic()
        self.ctrl: Any = None
        self.writable: bool | None = None
        self.held: Any = None
        self.held_timer: asyncio.TimerHandle | None = None
        # The missing-control-data warning, once per connection.
        self.ctrl_missing_warned = False
        self.last_value: Any = None
        self.counts = {"ctrl": 0, "time": 0}
        self.waiters: list[asyncio.Future] = []
        self.reopened_this_connection = False
        # The second-closure warning, once per connection.
        self.second_closure_warned = False
        self.closed = False
        self.reopen: Callable[[], None] | None = None


class CABulk:
    """Every Channel Access subscription of one connector, one record per address."""

    def __init__(
        self,
        aioca: Any,
        ca_loop: Any,
        *,
        value_factory: Callable[[Any, Any, bool | None], Any],
        metadata_factory: Callable[[Any, bool | None], Any],
        held_warning_s: float,
    ) -> None:
        self._aioca = aioca
        self._ca = ca_loop
        self._value_factory = value_factory
        self._metadata_factory = metadata_factory
        self._held_warning_s = held_warning_s
        self._records: dict[str, _Record] = {}
        self._owner: dict[str, str] = {}  # sub_id -> record name
        #: When a record's connection last came up (monotonic); None before any did.
        self.last_connect_at: float | None = None

    # -- subscribe / unsubscribe (CA loop) ---------------------------------

    async def subscribe(
        self,
        address: str,
        sub_id: str,
        callback: Callable[[Any], None],
        loop: asyncio.AbstractEventLoop,
        on_connection: Callable[[bool], None] | None,
        released: Callable[[], bool] | None = None,
    ) -> None:
        """Join ``address``'s record as ``sub_id``, opening it on first need. CA loop.

        ``released`` tells whether the caller already let the subscription go
        (it gave up while this join was queued); such a join opens nothing.
        """
        if released is not None and released():
            return
        name = normalise_address(address)
        record = self._records.get(name)
        if record is None:
            record = _Record(name)
            self._open_or_clear(record)
            reopen = record.reopen = lambda: self._on_repoint(record)
            self._ca.add_reopener(reopen)
            self._records[name] = record
        elif self._without_monitors(record):
            # A failed re-open left it without monitors: open them for this
            # joiner, who learns of a failure the way a first subscriber does.
            self._open_or_clear(record)
            record.reopened_this_connection = False
        elif self._monitor_closed(record):
            self._schedule_reopen(record)
        subscriber = _Subscriber(callback, loop, on_connection)
        record.subscribers[sub_id] = subscriber
        self._owner[sub_id] = name
        if record.connected:
            # A late joiner starts from the record's state, like the first one.
            self._post_connection(record, sub_id, subscriber, True)
            if record.last_value is not None:
                self._post(record, sub_id, subscriber, subscriber.callback, record.last_value)

    def unsubscribe(self, sub_id: str) -> None:
        """Leave the record; the last subscriber closes it. Idempotent. CA loop."""
        name = self._owner.pop(sub_id, None)
        record = self._records.get(name) if name is not None else None
        if record is None:
            return
        record.subscribers.pop(sub_id, None)
        if not record.subscribers:
            self._forget(record)

    def _forget(self, record: _Record) -> None:
        record.closed = True
        self._records.pop(record.name, None)
        if record.reopen is not None:
            self._ca.discard_reopener(record.reopen)
        self._close_monitors(record)
        self._cancel_held_timer(record)
        self._settle_waiters(record, None)

    # -- monitors ----------------------------------------------------------

    def _open_monitors(self, record: _Record) -> None:
        aioca = self._aioca
        record.ctrl_monitor = aioca.camonitor(
            record.name,
            lambda reading: self._guarded(record, self._on_ctrl, reading),
            events=aioca.DBE_PROPERTY,
            format=aioca.FORMAT_CTRL,
            count=1,
            all_updates=True,
            notify_disconnect=True,
        )
        record.time_monitor = aioca.camonitor(
            record.name,
            lambda reading: self._guarded(record, self._on_time, reading),
            events=aioca.DBE_VALUE | aioca.DBE_ALARM,
            format=aioca.FORMAT_TIME,
            all_updates=True,
            notify_disconnect=True,
        )

    def _open_or_clear(self, record: _Record) -> None:
        """Open both monitors; if either fails, close what opened and raise."""
        try:
            self._open_monitors(record)
        except Exception:
            self._close_monitors(record)
            raise

    @staticmethod
    def _without_monitors(record: _Record) -> bool:
        """True for a live record whose monitors a failed re-open left unopened."""
        return not record.closed and record.ctrl_monitor is None and record.time_monitor is None

    def _close_monitors(self, record: _Record) -> None:
        for attr in ("ctrl_monitor", "time_monitor"):
            handle = getattr(record, attr)
            setattr(record, attr, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception as exc:  # teardown is best effort
                    logger.debug("Closing a CA monitor of '%s' failed: %s", record.name, exc)

    def _monitor_closed(self, record: _Record) -> bool:
        """True when aioca closed one of the record's monitors on its own."""
        closed_state = getattr(getattr(self._aioca, "Subscription", None), "CLOSED", 2)
        return any(
            getattr(handle, "state", None) == closed_state
            for handle in (record.ctrl_monitor, record.time_monitor)
        )

    def _guarded(self, record: _Record, handler: Callable[[_Record, Any], None], reading: Any):
        """Run a monitor handler so nothing it raises reaches aioca.

        aioca closes a subscription whose callback raises, and nothing would
        re-open it. A record already forgotten ignores what is still queued.
        """
        if record.closed:
            return
        try:
            handler(record, reading)
        except Exception:
            logger.warning("Dropped a CA monitor event for '%s'", record.name, exc_info=True)

    def _on_ctrl(self, record: _Record, reading: Any) -> None:
        if isinstance(reading, self._aioca.CANothing):
            self._on_drop(record)
            return
        if not record.connected:
            # A new connection: it gets its own one re-open.
            record.reopened_this_connection = False
            record.second_closure_warned = False
        if self._monitor_closed(record) and not self._schedule_reopen(record):
            return
        record.counts["ctrl"] += 1
        record.ctrl = reading
        record.ctrl_missing_warned = False
        record.writable = _ca_write_access(self._aioca, record.name)
        if not record.connected:
            record.connected = True
            record.ever_connected = True
            record.down_since = None
            self.last_connect_at = time.monotonic()
            for sub_id, subscriber in list(record.subscribers.items()):
                self._post_connection(record, sub_id, subscriber, True)
        self._cancel_held_timer(record)
        held, record.held = record.held, None
        if held is not None:
            self._deliver(record, held)
        if record.waiters:
            self._settle_waiters(record, self._current_metadata(record))

    def _on_time(self, record: _Record, reading: Any) -> None:
        if isinstance(reading, self._aioca.CANothing):
            self._on_drop(record)
            return
        if self._monitor_closed(record) and not self._schedule_reopen(record):
            return
        record.counts["time"] += 1
        if record.ctrl is None:
            record.held = reading
            if record.held_timer is None and not record.ctrl_missing_warned:
                record.held_timer = asyncio.get_running_loop().call_later(
                    self._held_warning_s, self._warn_ctrl_missing, record
                )
            return
        self._deliver(record, reading)

    def _warn_ctrl_missing(self, record: _Record) -> None:
        record.held_timer = None
        if not record.closed and record.ctrl is None and record.held is not None:
            record.ctrl_missing_warned = True
            logger.warning(
                "CA channel '%s' sends values but no control data; its values are held",
                record.name,
            )

    def _cancel_held_timer(self, record: _Record) -> None:
        if record.held_timer is not None:
            record.held_timer.cancel()
            record.held_timer = None

    def _on_drop(self, record: _Record) -> None:
        if _ca_channel_connected(self._aioca, record.name):
            # Stale: this drop reached the loop after the next connection began.
            return
        self._mark_down(record)
        if self._monitor_closed(record):
            self._schedule_reopen(record)

    def _schedule_reopen(self, record: _Record) -> bool:
        """Re-open a record whose monitor aioca closed on its own, once per connection.

        Returns False for a second closure in one connection: the record is
        then down, and the event that found it is not delivered.

        Deferred to the next loop iteration, never run inside a monitor
        callback: aioca delivers a drop while walking the channel's
        subscriptions, and closing one of them there changes that set under it.
        """
        if record.reopened_this_connection:
            # A monitor that cannot deliver leaves the record unusable: down.
            self._mark_down(record)
            if not record.second_closure_warned:
                record.second_closure_warned = True
                logger.warning(
                    "A CA monitor of '%s' was closed again in one connection; "
                    "the record stays down until the channel reconnects",
                    record.name,
                )
            return False
        record.reopened_this_connection = True
        asyncio.get_running_loop().call_soon(self._reopen, record)
        return True

    def _reopen(self, record: _Record) -> None:
        if record.closed or not self._monitor_closed(record):
            return
        logger.warning("A CA monitor of '%s' was closed; re-opening it", record.name)
        self._close_monitors(record)
        self._reopen_or_mark_dead(record)

    def _reopen_or_mark_dead(self, record: _Record) -> None:
        """Open the monitors of a record that has none; on failure leave it re-openable.

        Runs as a loop callback or a repoint's re-opener, so nothing above would
        see an exception. A failure leaves the record without monitors: it is
        reported down, its metadata waiters are answered so they read the
        channel instead, and the next subscriber or repoint opens it again.
        """
        try:
            self._open_or_clear(record)
        except Exception:
            self._mark_down(record)
            self._settle_waiters(record, None)
            record.reopened_this_connection = False
            logger.error("Opening the CA monitors of '%s' failed", record.name, exc_info=True)

    def _mark_down(self, record: _Record) -> None:
        record.ctrl_missing_warned = False
        record.ctrl = None
        record.held = None
        record.last_value = None
        self._cancel_held_timer(record)
        if not record.connected:
            return
        record.connected = False
        record.down_since = time.monotonic()
        for sub_id, subscriber in list(record.subscribers.items()):
            self._post_connection(record, sub_id, subscriber, False)

    def _on_repoint(self, record: _Record) -> None:
        """Re-open a record on a fresh context; aioca closed its monitors silently."""
        if record.closed:
            return
        self._mark_down(record)
        self._settle_waiters(record, None)
        record.reopened_this_connection = False
        # The purge already closed both monitors. They are dropped, never
        # closed again: a close flushes CA I/O, and with the old context
        # detached and the new one not yet created, that flush would make
        # libca create a context of its own -- one that delivers nothing.
        record.ctrl_monitor = record.time_monitor = None
        self._reopen_or_mark_dead(record)

    # -- delivery ----------------------------------------------------------

    def _deliver(self, record: _Record, reading: Any) -> None:
        value = self._value_factory(reading, record.ctrl, record.writable)
        record.last_value = value
        for sub_id, subscriber in list(record.subscribers.items()):
            self._post(record, sub_id, subscriber, subscriber.callback, value)

    def _post_connection(
        self, record: _Record, sub_id: str, subscriber: _Subscriber, up: bool
    ) -> None:
        if subscriber.on_connection is not None:
            self._post(record, sub_id, subscriber, _report_connection, subscriber.on_connection, up)

    def _post(
        self, record: _Record, sub_id: str, subscriber: _Subscriber, fn: Callable, *args: Any
    ) -> None:
        try:
            subscriber.loop.call_soon_threadsafe(fn, *args)
        except RuntimeError:
            # The subscriber's loop has closed (osprey.runtime runs one per
            # call). It leaves the record -- after this callback returns, never
            # inside it: aioca may still be walking the channel's subscriptions.
            logger.debug("Dropping a CA subscriber of '%s': its loop has closed", record.name)
            asyncio.get_running_loop().call_soon(self.unsubscribe, sub_id)

    # -- metadata and diagnostics -----------------------------------------

    def _current_metadata(self, record: _Record) -> Any:
        if record.last_value is not None:
            return record.last_value.metadata
        return self._metadata_factory(record.ctrl, record.writable)

    def _settle_waiters(self, record: _Record, result: Any) -> None:
        waiters, record.waiters = record.waiters, []
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(result)

    async def metadata(self, address: str, timeout: float) -> Any:
        """Metadata of a subscribed address, awaiting its CTRL event. CA loop.

        ``None`` when nothing subscribes to ``address``, when the record has no
        monitors, or when it was closed, repointed or left without monitors
        while waiting: the caller then reads the channel.
        Raises :class:`ChannelUnreachableError` when no control data came
        within ``timeout``: the channel did not answer.
        """
        record = self._records.get(normalise_address(address))
        if record is None or self._without_monitors(record):
            return None
        if record.connected and record.ctrl is not None:
            return self._current_metadata(record)
        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        record.waiters.append(waiter)
        try:
            return await asyncio.wait_for(waiter, timeout)
        except TimeoutError:
            raise ChannelUnreachableError(
                record.name,
                f"CA channel '{record.name}' is subscribed but delivered no control data "
                f"within {timeout}s",
            ) from None
        finally:
            if waiter in record.waiters:
                record.waiters.remove(waiter)

    def event_counts(self, address: str) -> dict[str, int]:
        """Cumulative CTRL and TIME events of ``address``'s record. Any thread.

        A diagnostic read without a lock: the counts are plain ints, and a read
        that races an event is off by one at most.
        """
        record = self._records.get(normalise_address(address))
        if record is None:
            return {"ctrl": 0, "time": 0}
        return dict(record.counts)

    def snapshot(self) -> list[tuple[str, bool, float | None, bool]]:
        """``(name, connected, down_since, ever_connected)`` per record. CA loop."""
        return [
            (r.name, r.connected, r.down_since, r.ever_connected) for r in self._records.values()
        ]

    def is_down(self, name: str) -> bool | None:
        """Whether the record for ``name`` is down; ``None`` with no record. CA loop."""
        record = self._records.get(name)
        return None if record is None else not record.connected


def _report_connection(on_connection: Callable[[bool], None], up: bool) -> None:
    """Tell a subscriber its channel's connection state; what it raises stays here."""
    try:
        on_connection(up)
    except Exception:
        logger.warning("A CA connection callback raised", exc_info=True)
