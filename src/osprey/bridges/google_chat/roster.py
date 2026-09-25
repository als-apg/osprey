"""Who is in a Google Chat space: the member listing, named by what the space shows.

:class:`SpaceRoster` lists a space's members with the app's own ``chat.bot``
credential (``spaces.members.list``; no new scope, no directory lookup) and names
each member by one precedence: the ``displayName`` the listing returns, else the
name the bridge has seen that person use or be @mentioned under in that space,
else unknown. A name is never guessed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .events import _ids_match

logger = logging.getLogger(__name__)

ROSTER_TTL_SECONDS = 600.0
"""How long a fetched member list is reused. It outlasts one run (the worker's
``DISPATCH_TIMEOUT_SEC`` defaults to 300), so the answer's mention rendering
normally reuses the list the dispatch fetched instead of listing twice."""

ROSTER_MAX_MEMBERS = 200
"""The most members listed per space. Bounds both the payload the agent reads and
the memory one space holds; a capped list is flagged ``more_not_listed``."""

ROSTER_MAX_SPACES = 256
"""The most spaces cached at once, oldest evicted first. The roster lives as long
as the process, so without a bound every space ever seen would stay in memory."""

SEEN_MAX_PER_SPACE = 500
"""The most seen names remembered per space, oldest evicted first, for the same
reason."""

USER_PREFIX = "users/"
"""The resource-name prefix of a Chat person; the roster lists only these."""

_Listed = tuple[float, dict[str, str | None], bool]


class SpaceRoster:
    """A bounded, thread-safe cache of who is in each Chat space.

    Thread-safe: the subscriber thread records names and the drain thread lists
    rooms on the same instance, so every read and write of the two maps is under one
    lock (the listing call itself runs outside it). Three invariants: bounded (see
    the module constants), never raises out of :meth:`members`, and never guesses a
    name (listing name, then seen name, then unknown).
    """

    def __init__(
        self,
        list_members: Callable[[str, int], tuple[Sequence[Mapping[str, Any]], bool]],
        app_id: str,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the roster to its listing call.

        Args:
            list_members: ``(space, limit) -> (memberships, more)``; may raise.
            app_id: The Chat app's own user id, never listed.
            now: Clock seam for the TTL.
        """
        self._list_members = list_members
        self._app_id = app_id
        self._now = now
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, _Listed] = OrderedDict()
        self._seen: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()

    def note_seen(self, space: str, people: Mapping[str, str]) -> None:
        """Record names the space showed. The newest seen name for an id wins among
        seen names; it never overrides a name the listing returns."""
        if not space or not people:
            return
        with self._lock:
            names = self._seen.pop(space, None) or OrderedDict()
            self._seen[space] = names
            for ident, name in people.items():
                names.pop(ident, None)
                names[ident] = name
            while len(names) > SEEN_MAX_PER_SPACE:
                names.popitem(last=False)
            while len(self._seen) > ROSTER_MAX_SPACES:
                self._seen.popitem(last=False)

    def members(self, space: str) -> tuple[dict[str, str | None], bool] | None:
        """``({id: name | None}, more_not_listed)`` for ``space``, or ``None``.

        Served from the cache within :data:`ROSTER_TTL_SECONDS`; otherwise listed
        again. Any failure is logged and answered with ``None``, and nothing is
        cached, so the next dispatch tries again.
        """
        try:
            if not space:
                return None
            with self._lock:
                cached = self._cache.get(space)
            if cached is None or self._now() - cached[0] >= ROSTER_TTL_SECONDS:
                cached = self._fetch(space)
            _, listed, more = cached
            with self._lock:
                seen = dict(self._seen.get(space) or {})
            return {ident: name or seen.get(ident) for ident, name in listed.items()}, more
        except Exception:
            logger.warning(
                "members.list failed for %s; the question goes out without the room",
                space,
                exc_info=True,
            )
            return None

    def _fetch(self, space: str) -> _Listed:
        """List ``space`` (outside the lock), keep its joined humans, and cache them.
        The listing call caps the list at :data:`ROSTER_MAX_MEMBERS`."""
        memberships, more = self._list_members(space, ROSTER_MAX_MEMBERS)
        listed: dict[str, str | None] = {}
        for membership in memberships:
            if not isinstance(membership, Mapping):
                continue
            if membership.get("state") not in (None, "JOINED"):
                continue
            member = membership.get("member")
            if not isinstance(member, Mapping):
                continue
            if member.get("type") not in (None, "HUMAN"):
                continue
            ident = member.get("name")
            if not isinstance(ident, str) or not ident.startswith(USER_PREFIX):
                continue
            if _ids_match(ident, self._app_id) or ident in listed:
                continue
            name = member.get("displayName")
            listed[ident] = name if isinstance(name, str) and name else None
        entry: _Listed = (self._now(), listed, bool(more))
        with self._lock:
            self._cache.pop(space, None)
            self._cache[space] = entry
            while len(self._cache) > ROSTER_MAX_SPACES:
                self._cache.popitem(last=False)
        return entry
