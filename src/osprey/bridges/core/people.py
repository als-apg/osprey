"""Who is speaking and who is in the room, as the engine ships them to the agent.

The payload these shapes ride in is documented at
``docs/source/reference/contracts/bridge-dispatch.rst``. This module imports only the
standard library and :mod:`osprey.bridges.core.ports`, which keeps the core's
isolation rule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .ports import RoomPeople

__all__ = [
    "MENTIONS_OFF_NOTE",
    "MENTION_PLACEHOLDER_RE",
    "MENTION_RULE",
    "asker_of",
    "room_payload",
]

MENTION_PLACEHOLDER_RE = re.compile(r"<@([^\s<>]+)>")
"""The one mention form every adapter renders: ``<@ID>``, with the id exactly as the
roster lists it (group 1). An id may contain ``@`` (email-style Nextcloud user ids are
mentionable); it may not contain whitespace, ``<`` or ``>``. Google Chat (``users/…``)
and Teams (``29:…``) ids hold no ``@``, so allowing it does not change what they
match."""

# The two notes below are prose written to be read by the agent, like the pipeline's
# EXPIRED_NOTE: nothing matches on their wording. The payload they ride in is
# documented at docs/source/reference/contracts/bridge-dispatch.rst.
MENTION_RULE = (
    "To @mention a member of this room, write <@ID> with their id exactly as listed "
    "here. Do it only when a person in this room asked you to pass something on or to "
    "notify someone, and mention only the people they meant. Never mention anyone on "
    "your own initiative. An id that is not on this list is posted as plain text. A "
    "member whose name is null is someone the bridge has no name for; do not guess who "
    "it is."
)
MENTIONS_OFF_NOTE = (
    "@mentions are turned off for this deployment; refer to people by name in plain "
    "text. A member whose name is null is someone the bridge has no name for; do not "
    "guess who it is."
)


def asker_of(entry: Mapping[str, Any]) -> dict[str, str | None] | None:
    """Return who asked the entry's question, as ``{"id", "name"}``, or ``None``.

    The one reading of "who asked", so the dispatch payload and the history turn can
    never disagree. Reads the persisted ``sender_id`` and ``sender_display``, keeping
    each only when it is a non-empty ``str``; with neither, the asker is unknown and
    this returns ``None``. Never raises.
    """
    sid = entry.get("sender_id")
    name = entry.get("sender_display")
    sid = sid if isinstance(sid, str) and sid else None
    name = name if isinstance(name, str) and name else None
    if sid is None and name is None:
        return None
    return {"id": sid, "name": name}


def room_payload(people: RoomPeople) -> dict[str, Any] | None:
    """The payload's ``room`` value for what a roster reported, or ``None`` for nobody.

    Lists each member as ``{"id", "name"}`` and carries :data:`MENTION_RULE` when the
    deployment renders mentions, :data:`MENTIONS_OFF_NOTE` otherwise; adds
    ``"more_not_listed": True`` only when the roster capped the list.
    """
    if not people.members:
        return None
    room: dict[str, Any] = {
        "members": [{"id": m.id, "name": m.name} for m in people.members],
        "mentions": MENTION_RULE if people.mentions else MENTIONS_OFF_NOTE,
    }
    if people.more_not_listed:
        room["more_not_listed"] = True
    return room
