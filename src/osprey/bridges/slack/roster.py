"""Who is in a Slack channel: the bounded member listing, named by what Slack shows.

:class:`ChannelRoster` lists a channel's members with ``conversations.members``
(bounded to :data:`ROSTER_MAX_MEMBERS`) and looks each one up with ``users.info``,
keeping people only: bots, deactivated accounts, Slackbot and the bridge's own bot
user are left out. Each person is named by one precedence: the profile's display
name, else its real name, else the name the bridge has seen that person use in the
channel, else unknown. A name is never guessed; the login handle is not a name.

It mirrors the Teams bridge's roster shape for shape; the constants carry the same
values and are restated here because one bridge never imports another.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any, Protocol

logger = logging.getLogger(__name__)

__all__ = [
    "ROSTER_MAX_CHANNELS",
    "ROSTER_MAX_MEMBERS",
    "ROSTER_TTL_SECONDS",
    "SEEN_MAX_PER_CHANNEL",
    "ChannelRoster",
]

ROSTER_TTL_SECONDS = 600.0
"""How long a fetched member list is reused. It outlasts one run (the worker's
``DISPATCH_TIMEOUT_SEC`` defaults to 300), so the answer's mention rendering
normally reuses the list the dispatch fetched instead of listing twice."""

ROSTER_MAX_MEMBERS = 200
"""The most members listed per channel. Bounds the payload the agent reads, the
memory one channel holds and the ``users.info`` lookups one listing makes; a capped
list is flagged ``more_not_listed``."""

ROSTER_MAX_CHANNELS = 256
"""The most channels cached at once, oldest evicted first. The roster lives as long
as the process, so without a bound every channel ever seen would stay in memory."""

SEEN_MAX_PER_CHANNEL = 500
"""The most seen names remembered per channel, oldest evicted first, for the same
reason."""

SLACKBOT_USER_ID = "USLACKBOT"
"""Slack's built-in Slackbot user, which ``users.info`` does not mark ``is_bot``."""

_Listed = tuple[float, dict[str, str | None], bool]


class _Client(Protocol):
    """The two Web API calls the roster makes (``SlackWebClient`` provides both)."""

    def conversations_members(
        self, channel: str, *, limit: int | None
    ) -> tuple[list[str], bool]: ...

    def users_info(self, user: str) -> dict[str, Any]: ...


class ChannelRoster:
    """A bounded, thread-safe cache of who is in each Slack channel.

    Thread-safe: the receive thread records names and the drain thread lists
    channels on the same instance, so every read and write of the two maps is under
    one lock (the Web API calls run outside it). Three invariants: bounded (see the
    module constants), never raises out of :meth:`members`, and never guesses a name
    (listed name, then seen name, then unknown).
    """

    def __init__(
        self,
        client: _Client,
        bot_user_id: str,
        *,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """Wire the roster to its Web API client.

        Args:
            client: Provides ``conversations_members`` and ``users_info``; either may
                raise.
            bot_user_id: The bridge's own bot user (``auth.test``'s ``user_id``);
                never listed.
            now: Clock seam for the TTL.
        """
        self._client = client
        self._bot = bot_user_id
        self._now = now
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, _Listed] = OrderedDict()
        self._seen: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()

    def note_seen(self, channel: str, people: Mapping[str, str]) -> None:
        """Record names the channel showed. The newest seen name wins among seen
        names; a seen name never overrides a listed name."""
        if not channel or not people:
            return
        with self._lock:
            names = self._seen.pop(channel, None) or OrderedDict()
            self._seen[channel] = names
            for ident, name in people.items():
                names.pop(ident, None)
                names[ident] = name
            while len(names) > SEEN_MAX_PER_CHANNEL:
                names.popitem(last=False)
            while len(self._seen) > ROSTER_MAX_CHANNELS:
                self._seen.popitem(last=False)

    def members(self, channel: str) -> tuple[dict[str, str | None], bool] | None:
        """``({user_id: name | None}, more_not_listed)`` for ``channel``, or ``None``.

        Served from the cache within :data:`ROSTER_TTL_SECONDS`; otherwise listed
        again. An empty channel answers ``None`` without a call. Any failure — the
        listing or any one member's ``users.info`` — is logged, answered with
        ``None``, and nothing is cached, so the next dispatch tries again: a member
        whose lookup failed cannot be told apart from a bot, so a partial list would
        be a guess.
        """
        if not channel:
            return None
        try:
            with self._lock:
                cached = self._cache.get(channel)
            if cached is None or self._now() - cached[0] >= ROSTER_TTL_SECONDS:
                cached = self._fetch(channel)
            _, listed, more = cached
            with self._lock:
                seen = dict(self._seen.get(channel) or {})
            return {ident: name or seen.get(ident) for ident, name in listed.items()}, more
        except Exception:
            logger.warning(
                "members listing failed for %s; the question goes out without the room",
                channel,
                exc_info=True,
            )
            return None

    def _fetch(self, channel: str) -> _Listed:
        """List ``channel`` and look up each member (outside the lock), keep its
        people, and cache them."""
        ids, more = self._client.conversations_members(channel, limit=ROSTER_MAX_MEMBERS)
        listed: dict[str, str | None] = {}
        looked_up: set[str] = set()
        for ident in ids:
            if not isinstance(ident, str) or not ident or ident in looked_up:
                continue
            looked_up.add(ident)
            if ident in (self._bot, SLACKBOT_USER_ID):
                continue
            user = self._client.users_info(ident)
            if not isinstance(user, Mapping):
                raise TypeError(f"users.info for {ident} answered a {type(user).__name__}")
            if not _is_person(user):
                continue
            listed[ident] = _listed_name(user)
        entry: _Listed = (self._now(), listed, bool(more))
        with self._lock:
            self._cache.pop(channel, None)
            self._cache[channel] = entry
            while len(self._cache) > ROSTER_MAX_CHANNELS:
                self._cache.popitem(last=False)
        return entry


def _is_person(user: Mapping[str, Any]) -> bool:
    """Whether a ``users.info`` user is a person: not a bot or app user, and not
    deactivated."""
    return not (user.get("is_bot") or user.get("is_app_user") or user.get("deleted"))


def _listed_name(user: Mapping[str, Any]) -> str | None:
    """The profile's display name, else its real name, else ``None``. The login
    handle (``name``) is not a name a person goes by, so it is never used."""
    profile = user.get("profile")
    profile = profile if isinstance(profile, Mapping) else {}
    for value in (
        profile.get("display_name"),
        profile.get("real_name"),
        user.get("real_name"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
