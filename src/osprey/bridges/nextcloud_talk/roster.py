"""Who is in a Talk room: the participant listing, named by what the room shows.

:class:`ParticipantRoster` lists a room's participants with the bridge account's
existing credential (it is a participant of every room it polls) and names each
signed-in user by one precedence: the ``displayName`` the listing returns, else the
name the bridge has seen that person use or be @mentioned under in that room, else
unknown. A name is never guessed. (Not ``RoomRoster``: that is the core protocol
this adapter implements.)

The constants carry the same values as the other bridges' rosters and are restated
here because one bridge never imports another.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .events import USER_ACTOR_TYPE, _same_account

logger = logging.getLogger(__name__)

ROSTER_TTL_SECONDS = 600.0
"""How long a fetched participant list is reused. It outlasts one run (the worker's
``DISPATCH_TIMEOUT_SEC`` defaults to 300), so the answer's mention rendering
normally reuses the list the dispatch fetched instead of listing twice."""

ROSTER_MAX_MEMBERS = 200
"""The most participants listed per room. Talk answers the whole list unpaged, so
the cap is applied here; it bounds both the payload the agent reads and the memory
one room holds, and a capped list is flagged ``more_not_listed``."""

ROSTER_MAX_ROOMS = 256
"""The most rooms cached at once, oldest evicted first. The roster lives as long as
the process, so without a bound every room ever seen would stay in memory."""

SEEN_MAX_PER_ROOM = 500
"""The most seen names remembered per room, oldest evicted first, for the same
reason."""

EVERYONE_ID = "all"
"""Talk's ``@all`` / ``@"all"`` addresses the whole room. A participant id that
folds to it is never listed, so it can never render."""

_Listed = tuple[float, dict[str, str | None], bool]


class ParticipantRoster:
    """A bounded, thread-safe cache of who is in each Talk room.

    Thread-safe: the room threads record names and the drain thread lists rooms on
    the same instance, so every read and write of the two maps is under one lock
    (the listing call itself runs outside it). Invariants: bounded (see the module
    constants), never raises out of :meth:`members`, never guesses a name (listing
    name, then seen name, then unknown), and lists only signed-in users of that room.
    """

    def __init__(
        self,
        list_participants: Callable[[str], Sequence[Mapping[str, Any]]],
        bot_account: str,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the roster to its listing call.

        Args:
            list_participants: ``room -> attendees``; may raise.
            bot_account: The bridge's own account, never listed.
            now: Clock seam for the TTL.
        """
        self._list_participants = list_participants
        self._bot_account = bot_account
        self._now = now
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, _Listed] = OrderedDict()
        self._seen: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()

    def note_seen(self, room: str, people: Mapping[str, str]) -> None:
        """Record names the room showed. Among seen names the newest wins; a seen name
        never overrides a listing name."""
        if not room or not people:
            return
        with self._lock:
            names = self._seen.pop(room, None) or OrderedDict()
            self._seen[room] = names
            for ident, name in people.items():
                names.pop(ident, None)
                names[ident] = name
            while len(names) > SEEN_MAX_PER_ROOM:
                names.popitem(last=False)
            while len(self._seen) > ROSTER_MAX_ROOMS:
                self._seen.popitem(last=False)

    def members(self, room: str) -> tuple[dict[str, str | None], bool] | None:
        """``({id: name | None}, more_not_listed)`` for ``room``, or ``None``.

        Served from the cache within :data:`ROSTER_TTL_SECONDS`; otherwise listed
        again. Any failure (a lobby ``412`` included) is logged and answered with
        ``None``, and nothing is cached, so the next dispatch retries.
        """
        try:
            if not room:
                return None
            with self._lock:
                cached = self._cache.get(room)
            if cached is None or self._now() - cached[0] >= ROSTER_TTL_SECONDS:
                cached = self._fetch(room)
            _, listed, more = cached
            with self._lock:
                seen = dict(self._seen.get(room) or {})
            return {ident: name or seen.get(ident) for ident, name in listed.items()}, more
        except Exception:
            logger.warning(
                "participants list failed for room %s; the question goes out without the room",
                room,
                exc_info=True,
            )
            return None

    def _fetch(self, room: str) -> _Listed:
        """List ``room`` (outside the lock), keep its signed-in users, cap, and cache."""
        attendees = self._list_participants(room)
        kept: dict[str, str | None] = {}
        for attendee in attendees:
            if not isinstance(attendee, Mapping):
                continue
            if attendee.get("actorType") != USER_ACTOR_TYPE:
                continue
            ident = attendee.get("actorId")
            if not isinstance(ident, str) or not ident:
                continue
            if _same_account(ident, self._bot_account) or ident.casefold() == EVERYONE_ID:
                continue
            if ident in kept:
                continue
            name = attendee.get("displayName")
            kept[ident] = name if isinstance(name, str) and name else None
        more = len(kept) > ROSTER_MAX_MEMBERS
        listed = dict(list(kept.items())[:ROSTER_MAX_MEMBERS])
        entry: _Listed = (self._now(), listed, more)
        with self._lock:
            self._cache.pop(room, None)
            self._cache[room] = entry
            while len(self._cache) > ROSTER_MAX_ROOMS:
                self._cache.popitem(last=False)
        return entry
