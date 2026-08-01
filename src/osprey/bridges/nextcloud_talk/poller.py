"""One long-poll thread per Talk room — the adapter's whole ingestion loop.

Core deliberately has no ingestion loop, so this is the component that decides
*when* a message reaches the engine, and every rule in here exists because the
alternative loses messages silently rather than loudly.

**A room thread must never exit.** Nothing in core restarts a dead ingestion
thread: an exception escaping this loop would kill one room permanently while the
process keeps reporting itself healthy and every other room keeps working. So the
entire poll-and-dispatch cycle sits inside a catch-all with the same capped
backoff :mod:`~osprey.bridges.nextcloud_talk.rooms` uses for its type fetch, and
the constants are imported from there rather than restated — one policy, one
definition.

**The room-type hold lives here.** Until
:meth:`RoomDirectory.resolve_once` resolves a room, nothing is fetched from it and
nothing is dispatched. Messages then simply stay unread on the server: the poll
cursor has not moved, so they arrive whole once the type is known. This is why
``parse_event`` must never absorb :class:`RoomTypeUnresolved` — the same
"ignore it" that looks harmless in the parser would consume the message with no
dedup claim and a cursor already past it.

**Two cursors, and the asymmetry between them is the point.** The in-memory
cursor moves as soon as a batch is fetched, so the next long poll asks for the
right window. The *persisted* offset moves only once ``handle_event`` has returned
for every message in that batch. Getting this backwards is the one bug here that
destroys user messages: a persisted offset that runs ahead of what was actually
handled turns any crash into permanent loss, whereas one that lags merely costs a
duplicate delivery, which the engine's dedup claims absorb by design. When a
dispatch does fail, the in-memory cursor is rolled back to the window the failed
poll used — otherwise the persisted offset would be correctly unadvanced while the
*running* process sailed past the message, and it would be re-fetched only if the
bridge happened to restart.

Offset advance is independent of what ``handle_event`` returns. ``ignored``,
``duplicate`` and ``handled`` all mean the engine is done with the message; a
question that ended up parked in the retry queue is the retry queue's business
from then on, and holding the cursor for it would re-deliver it forever.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from osprey.bridges.core import BridgeRuntime

from .client import DEFAULT_POLL_HOLD, TalkClient
from .events import talk_int
from .offsets import OffsetStore
from .rooms import BACKOFF_CAP, BACKOFF_FACTOR, BACKOFF_START, RoomDirectory

logger = logging.getLogger(__name__)

SKIP_HISTORY_ANCHOR = 0
"""Cursor used on a first-ever start when Talk reports no last message.

An empty room has no tail to skip to, and ``0`` reads the room from its
beginning — which for an empty room is the same thing."""


def _message_id(message: Mapping[str, Any]) -> int | None:
    """A message's numeric id, or ``None`` if it has none usable.

    Only used for the cursor fallback, so an unusable id costs nothing: the
    message is still dispatched, it just cannot be the thing the cursor advances
    to. The coercion itself is
    :func:`~osprey.bridges.nextcloud_talk.events.talk_int` — the same rule the
    parser reads an id with, imported rather than restated so the two cannot
    disagree about what Talk sent.
    """
    return talk_int(message.get("id"))


class RoomPoller:
    """The ingestion loop for one Talk room.

    One instance per configured room, each owning a thread. Instances share the
    :class:`TalkClient`, the :class:`RoomDirectory`, the :class:`OffsetStore` and
    the :class:`BridgeRuntime` — all four are thread-safe — and hold only this
    room's cursor privately.

    Typical use from an adapter's ``serve(runtime)``::

        stop = threading.Event()
        pollers = [RoomPoller(room, client=..., rooms=..., offsets=...,
                              runtime=runtime, stop=stop) for room in cfg.rooms]
        for poller in pollers:
            poller.start()
        stop.wait()                     # until SIGTERM
        for poller in pollers:
            poller.join(timeout=...)

    Passing one shared ``stop`` event, as above, is the intended shape: a single
    :meth:`stop` (or a direct ``set()`` from a signal handler) then stops every
    room at once, and each room's backoff wait is interrupted rather than waited
    out.
    """

    def __init__(
        self,
        room: str,
        *,
        client: TalkClient,
        rooms: RoomDirectory,
        offsets: OffsetStore,
        runtime: BridgeRuntime,
        stop: threading.Event | None = None,
        sleep: Callable[[float], object] | None = None,
        poll_hold: float = DEFAULT_POLL_HOLD,
        backoff_start: float = BACKOFF_START,
        backoff_cap: float = BACKOFF_CAP,
    ):
        """Wire a poller for one room.

        Args:
            room: Talk room token to poll.
            client: Shared Talk client.
            rooms: Shared room-type cache. This poller holds on it and never
                dispatches from an unresolved room. Give the *directory* the same
                ``stop.wait`` for its own backoff, so a shutdown during a type
                fetch is prompt too.
            offsets: Shared persisted-offset store.
            runtime: The started engine. Raw Talk messages go to
                :meth:`BridgeRuntime.handle_event`; :meth:`BridgeRuntime.supervise`
                is called on every wake.
            stop: Shutdown signal. Share one event across every room and the
                process's signal handler. A fresh private event is created when
                omitted.
            sleep: Callable that waits the given number of seconds, used for the
                cycle backoff. Defaults to ``stop.wait``, so a shutdown
                interrupts a backoff instead of waiting it out; tests inject a
                recorder to assert the schedule without real delays.
            poll_hold: Seconds Talk is asked to hold an idle long poll open.
            backoff_start: Wait after this room's first failed cycle.
            backoff_cap: Ceiling on that wait.

        Raises:
            ValueError: If ``backoff_start`` is not positive or ``backoff_cap``
                is below it — either turns the guard into a hot retry loop
                against a Nextcloud that is already unwell.
        """
        if backoff_start <= 0:
            raise ValueError(f"backoff_start must be > 0; got {backoff_start}")
        if backoff_cap < backoff_start:
            raise ValueError(
                f"backoff_cap ({backoff_cap}) must be >= backoff_start ({backoff_start})"
            )
        self.room = room
        self.stop_event = stop if stop is not None else threading.Event()
        self._client = client
        self._rooms = rooms
        self._offsets = offsets
        self._runtime = runtime
        self._sleep = sleep if sleep is not None else self.stop_event.wait
        self._poll_hold = poll_hold
        self._backoff_start = backoff_start
        self._backoff_cap = backoff_cap
        self._delay = backoff_start
        # In-memory cursor; None until the first cycle resolves it from the
        # persisted offset or the skip-history anchor.
        self._cursor: int | None = None
        self._thread: threading.Thread | None = None

    # --- thread lifecycle ---------------------------------------------------

    def start(self) -> threading.Thread:
        """Start this room's polling thread and return it.

        Daemon, so it never blocks process exit, and named for logs. Calling this
        again while the thread is alive returns the existing one rather than
        starting a second poller for the same room, which would double-fetch it.
        """
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(target=self.run, name=f"talk-poll-{self.room}", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Signal the loop to finish. Idempotent.

        With a shared stop event this stops every room, which is what a SIGTERM
        handler wants.
        """
        self.stop_event.set()

    def is_alive(self) -> bool:
        """Whether this room's thread is running."""
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the thread to finish. Returns immediately if never started."""
        if self._thread is not None:
            self._thread.join(timeout)

    # --- the loop ------------------------------------------------------------

    def run(self) -> None:
        """Poll this room until :meth:`stop`.

        Does not raise: :meth:`poll_once` absorbs and backs off on everything, so
        the only way out is the stop event.
        """
        logger.info("room %s: polling started", self.room)
        while not self.stop_event.is_set():
            self.poll_once()
        logger.info("room %s: polling stopped", self.room)

    def poll_once(self) -> int:
        """Run one supervise-hold-poll-dispatch cycle.

        The unit the loop repeats, and the unit tests drive directly — calling it
        needs no thread.

        Returns:
            How many messages were handed to the engine: ``0`` for an idle poll,
            for a room still held on its type, and for a failed cycle.

        Raises:
            Nothing. A failure is logged, the room's backoff is waited out, and
            the next call retries — because a raise here would end the room's
            thread and nothing would restart it.
        """
        try:
            handled = self._cycle()
        except Exception as exc:
            # Full traceback on the first failure of a run only. The guard spans
            # the whole cycle, so the stack is the only thing that says which
            # stage broke — but a room that is down for hours would otherwise
            # repeat that traceback at every backoff interval forever.
            first = self._delay == self._backoff_start
            delay = self._penalize()
            logger.warning(
                "room %s: poll cycle failed (%s: %s); retrying in %.0fs",
                self.room,
                type(exc).__name__,
                exc,
                delay,
                exc_info=first,
            )
            self._sleep(delay)
            return 0
        self._delay = self._backoff_start
        return handled

    def _cycle(self) -> int:
        """One cycle, failures and all. See :meth:`poll_once` for the guard."""
        # First, on every wake: the engine's own housekeeping. Deliberately ahead
        # of the hold, so a room whose type has not resolved yet still drives the
        # drain supervision instead of stalling it for as long as Nextcloud is
        # unreachable.
        self._runtime.supervise()

        if self._rooms.resolve_once(self.room) is None:
            # Held. resolve_once has already waited out the directory's own
            # backoff, and the cursor has not moved, so nothing is lost —
            # whatever arrived meanwhile is still on the server.
            return 0

        last_known = self._last_known()
        messages, last_given = self._client.poll_messages(self.room, last_known, self._poll_hold)
        if not messages:
            # Talk's 304: the ordinary idle outcome, not a failure.
            return 0
        return self._dispatch(messages, last_given, last_known)

    def _dispatch(
        self,
        messages: Sequence[Mapping[str, Any]],
        last_given: int | None,
        last_known: int,
    ) -> int:
        """Feed a fetched batch to the engine, then persist the offset.

        Args:
            messages: The batch, in Talk's order.
            last_given: Talk's own next-cursor hint, if it sent a usable one.
            last_known: The cursor this batch was fetched with — what the
                in-memory cursor is rolled back to if a dispatch fails.

        Returns:
            The number of messages handed to :meth:`BridgeRuntime.handle_event`.

        Raises:
            Exception: Whatever ``handle_event`` raised, after rolling the
                in-memory cursor back so the batch is re-fetched.
        """
        self._cursor = self._advanced(messages, last_given, last_known)
        handled = 0
        try:
            for message in messages:
                self._runtime.handle_event(message)
                handled += 1
        except Exception:
            # The persisted offset is left exactly where it was — the whole batch
            # is re-fetched and the engine's dedup claims absorb the messages
            # already handled. Rolling the in-memory cursor back too is what
            # makes the RUNNING process re-fetch; without it only a restart
            # would, and the failed message would wait for one.
            self._cursor = last_known
            raise
        # Every message in the batch came back. Only now is it safe to remember
        # having passed them.
        self._offsets.advance(self.room, self._cursor)
        return handled

    def _advanced(
        self,
        messages: Sequence[Mapping[str, Any]],
        last_given: int | None,
        last_known: int,
    ) -> int:
        """Cursor to use after this batch.

        Talk's ``X-Chat-Last-Given`` is preferred when present — it accounts for
        messages Talk considered but did not return — and the batch's highest id
        is the documented fallback. Never below the cursor the batch was fetched
        with: a server answering with a cursor behind the request would otherwise
        make the same batch arrive forever.
        """
        if last_given is None:
            ids = [mid for mid in (_message_id(message) for message in messages) if mid is not None]
            last_given = max(ids, default=last_known)
        return max(last_given, last_known)

    def _last_known(self) -> int:
        """The cursor for the next poll, resolving it on the first cycle.

        A room with no persisted offset has never been polled, so its backlog is
        skipped: polling starts from the tail Talk reported when the room's type
        resolved, i.e. the room as it stood when this bridge came up. That anchor
        is **persisted immediately**, which is what makes "first-ever start" mean
        once rather than once per restart — otherwise a restart before the room's
        first message would re-skip whatever had arrived in between.
        """
        if self._cursor is not None:
            return self._cursor

        persisted = self._offsets.get(self.room)
        if persisted is not None:
            self._cursor = persisted
            return persisted

        info = self._rooms.info_of(self.room)
        anchor = SKIP_HISTORY_ANCHOR
        if info is not None and info.last_message_id is not None:
            anchor = info.last_message_id
        logger.info(
            "room %s: first-ever start, skipping history from message %d", self.room, anchor
        )
        self._cursor = anchor
        self._offsets.advance(self.room, anchor)
        return anchor

    def _penalize(self) -> float:
        """Consume this room's backoff delay and escalate the next one.

        Reset to ``backoff_start`` after any cycle that did not fail, so a room
        that recovers does not carry a minute-long delay into its next hiccup.
        """
        delay = self._delay
        self._delay = min(delay * BACKOFF_FACTOR, self._backoff_cap)
        return delay
