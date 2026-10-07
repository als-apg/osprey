"""Who is in a Teams conversation: the paged member listing, named by what it shows.

:class:`ConversationRoster` lists a conversation's members through the Bot
Connector with the bot's existing bearer (no Graph permission, no consent grant)
and names each member by one precedence: the ``name`` the listing returns, else
the name the bridge has seen that person use or be @mentioned under in that
conversation, else unknown. A name is never guessed.

It mirrors the Google Chat bridge's roster shape for shape; the constants carry
the same values and are restated here because one bridge never imports another.

The roster names people; :func:`share_audience` says who may open a file the bot
shares into a conversation. That audience is read from a fresh, complete listing on
every delivery and never from the roster's cache: a cached list could share with
someone just removed, and a capped one would leave people out.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .events import BOT_ROLE, USER_ID_PREFIX, bot_actor_id

logger = logging.getLogger(__name__)

ROSTER_TTL_SECONDS = 600.0
"""How long a fetched member list is reused. It outlasts one run (the worker's
``DISPATCH_TIMEOUT_SEC`` defaults to 300), so the answer's mention rendering
normally reuses the list the dispatch fetched instead of listing twice."""

ROSTER_MAX_MEMBERS = 200
"""The most members listed per conversation. Bounds both the payload the agent
reads and the memory one conversation holds; a capped list is flagged
``more_not_listed``. Also Teams' documented default page size."""

ROSTER_MAX_CONVERSATIONS = 256
"""The most conversations cached at once, oldest evicted first. The roster lives
as long as the process, so without a bound every conversation ever seen would
stay in memory."""

SEEN_MAX_PER_CONVERSATION = 500
"""The most seen names remembered per conversation, oldest evicted first, for the
same reason."""

_Listed = tuple[float, dict[str, str | None], bool]


class ConversationRoster:
    """A bounded, thread-safe cache of who is in each Teams conversation.

    Thread-safe: the receive threads record names and the drain thread lists
    conversations on the same instance, so every read and write of the two maps is
    under one lock (the listing call itself runs outside it). Three invariants:
    bounded (see the module constants), never raises out of :meth:`members`, and
    never guesses a name (listed name, then seen name, then unknown).
    """

    def __init__(
        self,
        list_members: Callable[[str, str, int], tuple[Sequence[Mapping[str, Any]], bool]],
        app_id: str,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the roster to its listing call.

        Args:
            list_members: ``(service_url, conversation, limit) -> (members, more)``;
                may raise.
            app_id: The bot's app id; the bot is never listed.
            now: Clock seam for the TTL.
        """
        self._list_members = list_members
        self._bot = bot_actor_id(app_id)
        self._now = now
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, _Listed] = OrderedDict()
        self._seen: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()

    def note_seen(self, conversation: str, people: Mapping[str, str]) -> None:
        """Record names the conversation showed. The newest seen name wins among seen
        names; a seen name never overrides a listed name."""
        if not conversation or not people:
            return
        with self._lock:
            names = self._seen.pop(conversation, None) or OrderedDict()
            self._seen[conversation] = names
            for ident, name in people.items():
                names.pop(ident, None)
                names[ident] = name
            while len(names) > SEEN_MAX_PER_CONVERSATION:
                names.popitem(last=False)
            while len(self._seen) > ROSTER_MAX_CONVERSATIONS:
                self._seen.popitem(last=False)

    def members(
        self, service_url: str, conversation: str
    ) -> tuple[dict[str, str | None], bool] | None:
        """``({id: name | None}, more_not_listed)`` for ``conversation``, or ``None``.

        Cached by ``conversation`` only: the service URL addresses the call and is
        not part of who is in the conversation. Served from the cache within
        :data:`ROSTER_TTL_SECONDS`; otherwise listed again. An empty address answers
        ``None`` without a call; any failure is logged, answered with ``None``, and
        nothing is cached, so the next dispatch tries again.
        """
        if not service_url or not conversation:
            return None
        try:
            with self._lock:
                cached = self._cache.get(conversation)
            if cached is None or self._now() - cached[0] >= ROSTER_TTL_SECONDS:
                cached = self._fetch(service_url, conversation)
            _, listed, more = cached
            with self._lock:
                seen = dict(self._seen.get(conversation) or {})
            return {ident: name or seen.get(ident) for ident, name in listed.items()}, more
        except Exception:
            logger.warning(
                "members listing failed for %s; the question goes out without the room",
                conversation,
                exc_info=True,
            )
            return None

    def _fetch(self, service_url: str, conversation: str) -> _Listed:
        """List ``conversation`` (outside the lock), keep its people, and cache them.
        The listing call caps the list at :data:`ROSTER_MAX_MEMBERS`."""
        members, more = self._list_members(service_url, conversation, ROSTER_MAX_MEMBERS)
        listed: dict[str, str | None] = {}
        for member in members:
            if not _is_person(member, self._bot) or member["id"] in listed:
                continue
            ident = member["id"]
            name = member.get("name")
            listed[ident] = name if isinstance(name, str) and name else None
        entry: _Listed = (self._now(), listed, bool(more))
        with self._lock:
            self._cache.pop(conversation, None)
            self._cache[conversation] = entry
            while len(self._cache) > ROSTER_MAX_CONVERSATIONS:
                self._cache.popitem(last=False)
        return entry


def _is_person(member: Any, bot_id: str) -> bool:
    """Whether a listed member is a person: a mapping with a ``29:`` id that is not
    the bot and whose role is not ``bot``."""
    if not isinstance(member, Mapping):
        return False
    ident = member.get("id")
    if not isinstance(ident, str) or not ident.startswith(USER_ID_PREFIX):
        return False
    if ident == bot_id:
        return False
    role = member.get("role")
    return not (isinstance(role, str) and role.lower() == BOT_ROLE)


def share_audience(members: Sequence[Any], app_id: str, tenant_id: str) -> tuple[str, ...]:
    """The directory ids a file shared into a conversation is shared with.

    Every person's ``aadObjectId``, in listing order, without repeats. A member with
    no string ``aadObjectId`` is left out, and so is one whose listing carries a
    ``tenantId`` other than ``tenant_id``: a shared-channel member from another
    directory, whose id names nobody in this one.

    Args:
        members: One complete member listing of the conversation.
        app_id: The bot's app id; the bot is never in the audience.
        tenant_id: The directory the bot and its file library live in.

    Returns:
        The directory ids, possibly empty.
    """
    bot_id = bot_actor_id(app_id)
    audience: list[str] = []
    for member in members:
        if not _is_person(member, bot_id):
            continue
        oid = member.get("aadObjectId")
        if not isinstance(oid, str) or not oid:
            continue
        tenant = member.get("tenantId")
        if tenant is not None and tenant != tenant_id:
            continue
        if oid not in audience:
            audience.append(oid)
    return tuple(audience)
