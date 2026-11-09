"""Slack ingestion: Socket Mode envelopes → durable inbox → bounded handler pool.

Two halves, joined only by the :class:`~.inbox.Inbox` on the state volume.

**Ingest** (:class:`Ingest`, its own thread) owns the socket. For every
``events_api`` envelope it :meth:`~.inbox.Inbox.put` s the payload under the
Events API ``event_id`` and acks the envelope **only once the put returned
True** (Decision 3): an ack is Slack's promise not to redeliver, so the event
must be on disk first. A put that cannot be made durable — the inbox is at its
cap, or the disk refused — withholds the ack, so Slack redelivers later, and is
logged (Decision 5). Every other envelope type is acked and ignored. Ingest
never waits on the handler pool, so an ack goes out within Slack's three-second
window however busy the pool is (Decision 7). It is started *before*
:func:`~osprey.bridges.core.runtime.run_forever`, whose startup reconcile can
take longer than Slack's retry window (Decision 4).

**Serve** (:func:`serve`, called by ``run_forever`` once recovery is done) feeds
every pending inbox record — leftovers from a previous process first, since the
inbox is oldest-first, then new arrivals — into a pool of ``pool_size`` threads
running :meth:`~osprey.bridges.core.BridgeRuntime.handle_event`. A record is
removed only when ``handle_event`` **returns**; on a raise it stays and
:meth:`~.inbox.Inbox.note_failure` counts the attempt (the inbox quarantines it
at its ``max_attempts``). Replay is safe because the engine dedups on
``channel:ts``: an already-claimed event comes back ``duplicate``.

Both loops wake at least every :data:`TICK_SEC` (the socket's idle tick on the
ingest side, a bounded wait on the serve side), so ``stop``, the fatal flag and
the ``supervise`` schedule are serviced on a quiet socket. The fatal flag —
set by the web client or the socket on a dead credential (Decision 6) — or the
ingest thread dying makes :func:`serve` raise :class:`~.client.SlackAuthError`;
``main`` maps that to the cannot-start exit code.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from .client import SlackAuthError
from .inbox import Inbox
from .socket import Envelope

logger = logging.getLogger(__name__)

POOL_SIZE = 4
"""Concurrent :meth:`~osprey.bridges.core.BridgeRuntime.handle_event` calls (Decision 5).

Each call blocks for a whole run, so this bounds the runs one bridge drives at
once; records beyond it wait in the inbox, already acked."""

WAKE_INTERVAL = 60.0
"""Seconds between :meth:`~osprey.bridges.core.BridgeRuntime.supervise` calls.

It bounds how long a dead drain thread stays dead, and it is also when a record
whose ``handle_event`` raised is offered again — a failure is retried at the
next wake, not hot-looped through its attempts."""

TICK_SEC = 1.0
"""Longest the serve loop waits for an arrival before re-checking its flags.

Matches the socket's idle tick, so ``stop`` and the fatal flag are noticed
within about a second on either side."""

EVENTS_API = "events_api"
"""The one envelope type that carries a question; every other type is acked and ignored."""


class _Connection(Protocol):
    """What :class:`Ingest` needs from :class:`~.socket.SocketModeConnection`."""

    def __iter__(self) -> Any: ...

    def ack(self, envelope_id: str) -> bool: ...

    def close(self) -> None: ...


class _Runtime(Protocol):
    """What :func:`serve` needs from :class:`~osprey.bridges.core.BridgeRuntime`."""

    def handle_event(self, event: Any) -> str: ...

    def supervise(self) -> Any: ...


class Ingest(threading.Thread):
    """The socket-reading thread: inbox put, then ack, then notify serve.

    Iterates the connection from this one thread (the socket's contract). On
    every item — envelope or idle tick — it checks ``stop`` (return quietly) and
    the fatal flag (raise :class:`~.client.SlackAuthError`). Whatever ends the
    loop other than ``stop`` is kept in :attr:`error` for :func:`serve` to
    surface; the connection is closed on every exit, and :attr:`arrived` is set
    so a waiting serve loop notices at once.

    Attributes:
        arrived: Set after every accepted record (and on exit); :func:`serve`
            waits on it instead of polling the inbox blind.
        error: What ended the loop, or ``None`` (still running, or stopped).
    """

    def __init__(
        self,
        connection: _Connection,
        inbox: Inbox,
        *,
        stop: threading.Event,
        fatal: threading.Event,
    ) -> None:
        """Wire the thread; nothing is read until :meth:`start`.

        Args:
            connection: The Socket Mode link (iterate → ``Envelope | None``; ``ack``; ``close``).
            inbox: The durable spool shared with :func:`serve`.
            stop: Process shutdown signal, the same one handed to ``run_forever``.
            fatal: The shared dead-credential flag (``SlackWebClient.fatal``).
        """
        super().__init__(name="slack-ingest", daemon=True)
        self._connection = connection
        self._inbox = inbox
        self._stop_flag = stop
        self._fatal = fatal
        self.arrived = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            for item in self._connection:
                if self._stop_flag.is_set():
                    return
                if self._fatal.is_set():
                    raise SlackAuthError("slack credential is dead (fatal flag set)")
                if item is not None:
                    self._accept(item)
            if not self._stop_flag.is_set():
                raise RuntimeError("slack socket iteration ended")
        except BaseException as exc:
            self.error = exc
            # Type only: exception text from the transport may carry a URL ticket.
            logger.error("slack ingest stopped (%s)", type(exc).__name__)
        finally:
            self._connection.close()
            self.arrived.set()

    def _accept(self, envelope: Envelope) -> None:
        if envelope.type != EVENTS_API:
            logger.debug("slack ingest: acking an ignored %s envelope", envelope.type)
            self._connection.ack(envelope.envelope_id)
            return
        key = envelope.payload.get("event_id")
        if not isinstance(key, str) or not key:
            # Redelivery would bring back the same unusable payload; ack it, loudly.
            logger.warning("slack ingest: events_api envelope without an event_id; dropped")
            self._connection.ack(envelope.envelope_id)
            return
        if not self._inbox.put(key, envelope.payload):
            logger.warning("slack ingest: %s not stored; ack withheld for redelivery", key)
            return
        self._connection.ack(envelope.envelope_id)
        self.arrived.set()


def serve(
    inbox: Inbox,
    runtime: _Runtime,
    *,
    stop: threading.Event,
    fatal: threading.Event,
    ingest: Ingest | None = None,
    pool_size: int = POOL_SIZE,
    wake_interval: float = WAKE_INTERVAL,
    tick: float = TICK_SEC,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Feed inbox records to ``runtime.handle_event`` until shutdown.

    Each pass: raise if the fatal flag is set or ``ingest`` has died; call
    ``runtime.supervise()`` when a wake is due (which also re-offers records
    whose handler raised); submit pending records not already in flight, oldest
    first, while a pool slot is free; then wait up to ``tick`` for an arrival or
    a freed slot. :meth:`~.inbox.Inbox.pending` lists in-flight records too, so
    an in-flight set keeps a record from being submitted twice.

    On every exit the pool is shut down waiting for in-flight handlers, so each
    of their records is removed or counted before the process goes on.

    Args:
        inbox: The durable spool :class:`Ingest` fills.
        runtime: The started engine.
        stop: Shutdown signal; ends this loop (and ``ingest``'s).
        fatal: The shared dead-credential flag.
        ingest: The running ingest thread; its death is fatal, and its
            :attr:`Ingest.arrived` event is what this loop waits on. ``None``
            serves the inbox alone (records already stored).
        pool_size: Concurrent ``handle_event`` calls.
        wake_interval: Seconds between ``supervise`` calls.
        tick: Longest wait between passes.
        clock: Monotonic clock the wake schedule reads; tests inject one.

    Raises:
        SlackAuthError: The fatal flag is set, or ``ingest`` stopped without
            ``stop`` being set (its own ``SlackAuthError`` is re-raised as is).
    """
    wake = ingest.arrived if ingest is not None else threading.Event()
    lock = threading.Lock()
    in_flight: set[str] = set()
    held_back: set[str] = set()
    pool = ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="slack-handler")

    def handle(key: str, event: Any) -> None:
        failed = False
        try:
            status = runtime.handle_event(event)
        except Exception:
            failed = True
            logger.exception("handle_event failed for %s; record kept", key)
            inbox.note_failure(key)
        else:
            logger.debug("slack event %s: %s", key, status)
            inbox.remove(key)
        finally:
            with lock:
                in_flight.discard(key)
                if failed:
                    held_back.add(key)
            wake.set()

    def feed() -> None:
        with lock:
            free = pool_size - len(in_flight)
            if free <= 0:
                return
            skip = in_flight | held_back
        for key, event in inbox.pending():
            if free <= 0:
                break
            if key in skip:
                continue
            with lock:
                in_flight.add(key)
            pool.submit(handle, key, event)
            free -= 1

    logger.info("slack bridge serving the inbox")
    next_wake = clock() + wake_interval
    try:
        while not stop.is_set():
            _raise_if_dead(fatal, stop, ingest)
            if clock() >= next_wake:
                runtime.supervise()
                with lock:
                    held_back.clear()
                next_wake = clock() + wake_interval
            wake.clear()
            feed()
            wake.wait(tick)
    finally:
        pool.shutdown(wait=True)
    logger.info("slack bridge stopped serving the inbox")


def _raise_if_dead(fatal: threading.Event, stop: threading.Event, ingest: Ingest | None) -> None:
    """Raise :class:`~.client.SlackAuthError` for a set fatal flag or a dead ingest."""
    err = ingest.error if ingest is not None else None
    if fatal.is_set():
        if isinstance(err, SlackAuthError):
            raise err
        raise SlackAuthError("slack credential is dead (fatal flag set)")
    if ingest is not None and not ingest.is_alive() and not stop.is_set():
        if isinstance(err, SlackAuthError):
            raise err
        raise SlackAuthError("slack ingest thread stopped") from err


__all__ = ["EVENTS_API", "POOL_SIZE", "TICK_SEC", "WAKE_INTERVAL", "Ingest", "serve"]
