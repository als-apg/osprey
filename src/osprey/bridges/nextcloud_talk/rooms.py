"""Cached room types, and the hold that keeps a room's messages out of the
pipeline until its type is known.

Talk's mention filter cannot be applied without knowing a room's type: in a
one-to-one conversation every message is addressed to the bot, while in a group
room only an explicit mention is. Guessing wrong is not a cosmetic error — guess
"group" and the bridge ignores every private message, guess "one-to-one" and it
answers every unaddressed line in a busy room. So the type is *fetched*, once per
room, and until it arrives the room is **unresolved** and nothing from it is
dispatched (fail-closed).

Two rules shape this module, and both exist to keep that fail-closed hold from
turning into either a latency tax or silent message loss:

Reads never touch the network.
    :meth:`RoomDirectory.type_of` and friends are pure cache reads.
    ``parse_event`` consults them for *every inbound message*, so a fetch hidden
    behind a read would put an HTTP round trip — and an HTTP timeout — on the
    per-message path.

The hold is poller-side, never parser-side.
    A room's poll thread must not fetch or dispatch until the type resolves; it
    holds by calling :meth:`RoomDirectory.resolve_once` at the top of each cycle
    and looping while it returns ``None``. Messages are then simply *deferred* —
    still unread on the server, cursor unadvanced. The tempting alternative,
    having ``parse_event`` return ``None`` for an unresolved room, would silently
    **discard** those messages: the pipeline treats ``None`` as "not for us", so
    no dedup claim is made and the poll cursor has already moved past them.

The resolved/unresolved distinction is therefore deliberately awkward to ignore.
:meth:`RoomDirectory.type_of` returns ``int | None`` for the poller, whose whole
job is to test for ``None``; but the question ``parse_event`` actually asks is
:meth:`RoomDirectory.is_one_to_one`, which **raises**
:class:`RoomTypeUnresolved` rather than answer for a room it knows nothing about.
That is the failure mode worth engineering against: a bare
``type_of(room) == ROOM_TYPE_ONE_TO_ONE`` quietly reads ``None`` as "group room"
and re-introduces the fail-open guess. Raising instead defers the message —
the poll cycle's catch-all logs it and leaves the offset unadvanced.

The fetch itself retries with a capped exponential backoff and **never raises
out**: a Nextcloud that is down at startup leaves rooms unresolved and their
messages queued on the server, and the process stays up.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from .client import ROOM_TYPE_ONE_TO_ONE, RoomInfo, TalkClient

logger = logging.getLogger(__name__)

# --- backoff policy --------------------------------------------------------

BACKOFF_START = 1.0
"""Seconds to wait after a room's first failed type fetch."""

BACKOFF_CAP = 60.0
"""Ceiling on the wait between type fetches. Reached after six failures
(1, 2, 4, 8, 16, 32, then 60 s forever), so a Nextcloud that is down at startup
is retried roughly once a minute rather than hammered, and a room whose type
never resolves costs one request per minute indefinitely."""

BACKOFF_FACTOR = 2.0
"""Multiplier applied to the wait after each failure, up to :data:`BACKOFF_CAP`."""


class RoomTypeUnresolved(RuntimeError):
    """Raised when a room's type is needed but has not been fetched yet.

    Reaching this means a caller skipped the poller-side hold: nothing should ask
    what kind of room an unresolved room is. It is an exception rather than a
    fallback answer because every fallback is a fail-open guess about the mention
    filter. Raised out of a poll cycle, it defers the message (the cycle's
    catch-all leaves the persisted offset unadvanced) instead of dropping it.
    """


class RoomDirectory:
    """Per-room :class:`~osprey.bridges.nextcloud_talk.client.RoomInfo` cache,
    filled on demand with a capped backoff.

    Thread-safe, and shared by every room thread and the drain thread. The lock
    guards only the dict operations: a fetch's HTTP call and its backoff sleep
    both happen **outside** it, so one room waiting out a 30 s timeout never
    blocks another room's fetch or another thread's per-message
    :meth:`type_of` read.

    Entries are written once and never invalidated. A Talk conversation cannot
    change between one-to-one and group, so there is nothing to expire, and a
    cache that could go empty again would reopen the hold under live traffic.
    """

    def __init__(
        self,
        client: TalkClient,
        *,
        sleep: Callable[[float], object] = time.sleep,
        backoff_start: float = BACKOFF_START,
        backoff_cap: float = BACKOFF_CAP,
    ):
        """Wire the directory to a Talk client.

        Args:
            client: Client used for :meth:`TalkClient.room_info`. Shared with the
                rest of the adapter — this class adds the retry policy that
                client deliberately omits.
            sleep: Callable that waits the given number of seconds. The return
                value is ignored, which is what lets a caller pass
                ``stop.wait`` (a :class:`threading.Event`'s waiter) to make
                backoff waits interruptible at shutdown; tests pass a recorder
                and assert the backoff schedule without real delays.
            backoff_start: Wait after a room's first failed fetch.
            backoff_cap: Ceiling on that wait.

        Raises:
            ValueError: If ``backoff_start`` is not positive or ``backoff_cap``
                is below it — either would turn the backoff into a hot retry loop
                against a Nextcloud that is already struggling.
        """
        if backoff_start <= 0:
            raise ValueError(f"backoff_start must be > 0; got {backoff_start}")
        if backoff_cap < backoff_start:
            raise ValueError(
                f"backoff_cap ({backoff_cap}) must be >= backoff_start ({backoff_start})"
            )
        self._client = client
        self._sleep = sleep
        self._backoff_start = backoff_start
        self._backoff_cap = backoff_cap
        self._lock = threading.Lock()
        self._rooms: dict[str, RoomInfo] = {}
        self._delays: dict[str, float] = {}

    # --- cache reads (no network I/O, ever) ---------------------------------

    def type_of(self, room: str) -> int | None:
        """Talk's conversation type for ``room``, or ``None`` if unresolved.

        A pure cache read: never fetches, never blocks on I/O. ``None`` means
        "not known yet", *not* "group room" — see :meth:`is_one_to_one` for the
        form that cannot be misread that way.

        Args:
            room: Talk room token.

        Returns:
            The cached type, or ``None`` while the room is unresolved.
        """
        with self._lock:
            info = self._rooms.get(room)
        return None if info is None else info.type

    def is_one_to_one(self, room: str) -> bool:
        """Whether ``room`` is a one-to-one conversation (so no mention needed).

        The question ``parse_event`` asks. Total over resolved rooms and loudly
        undefined otherwise, which is the point: there is no return value here
        that could pass an unresolved room off as a group room.

        Args:
            room: Talk room token.

        Returns:
            ``True`` for a one-to-one conversation, ``False`` for a group room.

        Raises:
            RoomTypeUnresolved: If the room's type has not been fetched yet. The
                caller reached past the poller-side hold; the message must be
                deferred, not dropped.
        """
        room_type = self.type_of(room)
        if room_type is None:
            raise RoomTypeUnresolved(
                f"room {room!r} has no cached type yet; nothing may be dispatched from a "
                "room until RoomDirectory.resolve_once() has resolved it"
            )
        return room_type == ROOM_TYPE_ONE_TO_ONE

    def display_name_of(self, room: str) -> str:
        """Cached human-readable name for ``room``, or ``""`` if unresolved.

        For log lines and operator-facing notices only. Returns a string in every
        case — including for an unresolved room — so a log statement never has to
        guard against ``None``.
        """
        with self._lock:
            info = self._rooms.get(room)
        return "" if info is None else info.display_name

    def info_of(self, room: str) -> RoomInfo | None:
        """The whole cached payload for ``room``, or ``None`` if unresolved.

        Note what :attr:`RoomInfo.last_message_id` means here: it is a *snapshot*
        taken when the room resolved, not a live cursor. That makes it the right
        anchor for a first-ever start's skip-history (the room's tail at the
        moment the bridge came up) and the wrong thing to treat as a current
        poll offset.
        """
        with self._lock:
            return self._rooms.get(room)

    # --- population --------------------------------------------------------

    def resolve_once(self, room: str) -> int | None:
        """Return ``room``'s type, making at most one attempt to fetch it.

        This is the poller-side hold. A room thread calls it at the top of every
        cycle and proceeds only on a non-``None`` result:

        * already resolved — returns the cached type immediately, no request and
          no wait, so this is cheap to call on every pass;
        * unresolved — makes exactly **one**
          :meth:`TalkClient.room_info` call and returns the type on success;
        * that call failed — waits out this room's current backoff (via the
          injected ``sleep``) and returns ``None``.

        Never raises: every failure is logged and reported as ``None``. The
        retrying loop is the caller's own ``while not stop.is_set()``, which is
        what keeps the room thread's other per-cycle duties (``supervise()``)
        running while a room is still unresolved, and what guarantees a room
        thread neither exits nor spins.

        Args:
            room: Talk room token.

        Returns:
            The room's type, or ``None`` if it is still unresolved after this
            attempt.
        """
        cached = self.type_of(room)
        if cached is not None:
            return cached

        # Outside the lock: an unreachable Nextcloud makes this block for the
        # client's full timeout, and holding the lock across it would stall every
        # other room's fetch and every per-message type_of read behind it.
        try:
            info = self._client.room_info(room)
        except Exception as exc:
            delay = self._penalize(room)
            logger.warning(
                "room %s: type fetch failed (%s: %s); retrying in %.0fs — no message from "
                "this room is dispatched until its type is known",
                room,
                type(exc).__name__,
                exc,
                delay,
            )
            self._sleep(delay)
            return None

        with self._lock:
            self._rooms[room] = info
            self._delays.pop(room, None)
        logger.info(
            "room %s resolved: type=%d name=%r",
            room,
            info.type,
            info.display_name,
        )
        return info.type

    def _penalize(self, room: str) -> float:
        """Consume ``room``'s current backoff delay and escalate the next one.

        Returns:
            The number of seconds to wait for *this* failure. Subsequent
            failures multiply by :data:`BACKOFF_FACTOR` up to ``backoff_cap``.
            Resolving drops the room's entry — a resolved room is never
            re-fetched, so the delay can never be needed again.
        """
        with self._lock:
            delay = self._delays.get(room, self._backoff_start)
            self._delays[room] = min(delay * BACKOFF_FACTOR, self._backoff_cap)
        return delay
