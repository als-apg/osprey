"""``resolve_reply_context``: the empty context, pinned for every conversation shape.

Teams delivers no inbound quote to this bot. A bot reads the message a reply points at
only with resource-specific consent granted by a tenant admin, and this deployment asks
for no such permission — so a reply's parent is simply not on the wire. The member exists
anyway, because the seam is one shape for every channel.

That makes this the suite for a member whose whole contract is what it does NOT do, and
each half of it is a silent production failure if it breaks:

* **empty, not ``None``** — the engine treats an empty ``reply_to`` as "no quote" either
  way, but returning the dataclass is what keeps one shape at the seam and gives a future
  quote source somewhere to fill in;
* **empty ``meta``** — the context's meta lands in the same flat persisted entry as the
  engine's bookkeeping, so a member that invented a key here would corrupt an entry rather
  than fail a call;
* **no I/O, no failure path** — the member runs once per claimed question, between the ack
  and the dispatch, and a read that could hang there would delay every answer;
* **independent of the conversation** — a channel thread, a 1:1 and a group chat resolve
  identically, because none of them carries a quote to resolve.

Hermetic: parsing is pure, and the Connector leg is the posting suite's recording stand-in
that nothing here is expected to reach.
"""

from __future__ import annotations

from typing import Any

import pytest

from osprey.bridges.core import RESERVED_ENTRY_KEYS, InboundEvent, ReplyContext
from osprey.bridges.teams.events import resolve_reply_context as resolve_in_events
from tests.bridges.teams.test_posting import (
    QUESTION,
    SENDER_ID,
    TENANT,
    activity,
    chat_activity,
    make_ops,
)

GROUP_CHAT_CONVERSATION_ID = "19:group@thread.v2"
GROUP_CHAT_CONVERSATION = "groupChat"

EMPTY = ReplyContext(reply_to={}, meta={})
"""What every call must return, spelled once."""


def channel_reply() -> dict[str, Any]:
    """A reply inside a channel thread — the conversation id carries its root's id, which
    is the only shape Teams gives that names another message at all."""
    return activity()


def personal_message() -> dict[str, Any]:
    """A 1:1 chat message: a flat transcript with no threads and nobody else to address.

    Teams sends no mention entity at all in a 1:1 chat, and the parse member exempts
    that type from the filter."""
    return chat_activity()


def group_chat_message() -> dict[str, Any]:
    """A group chat: flat like a 1:1, addressed like a channel."""
    return activity(
        conversation={
            "id": GROUP_CHAT_CONVERSATION_ID,
            "conversationType": GROUP_CHAT_CONVERSATION,
            "tenantId": TENANT,
        }
    )


def event_for(wire: dict[str, Any]) -> InboundEvent:
    """The parsed event for one activity, asserted non-``None`` so a builder that stopped
    producing questions fails here instead of silently skipping the assertion."""
    ops, _ = make_ops()
    event = ops.parse_event(wire)
    assert event is not None
    return event


# --- the empty context, per conversation shape -------------------------------


@pytest.mark.parametrize(
    "wire",
    [
        pytest.param(channel_reply(), id="channel-reply"),
        pytest.param(personal_message(), id="one-to-one"),
        pytest.param(group_chat_message(), id="group-chat"),
    ],
)
def test_every_conversation_shape_resolves_the_empty_context(wire: dict[str, Any]) -> None:
    """A thread reply is the one place a Teams quote could plausibly come from, and a 1:1
    is where a user is most likely to expect one; neither carries it to a bot without
    consent this deployment does not ask for, so both resolve to nothing."""
    ops, connector = make_ops()

    context = ops.resolve_reply_context(event_for(wire))

    assert context == EMPTY
    assert context.reply_to == {}
    assert context.meta == {}
    # Resolution is free: no Connector leg, so the member cannot delay the dispatch it
    # sits in front of, and cannot fail it either.
    assert connector.calls == []


def test_the_context_is_returned_rather_than_none() -> None:
    """``None`` would also read as "no quote" today, so the distinction is about the seam:
    one shape for every entry, and a place for a future quote source to fill in."""
    ops, _ = make_ops()

    context = ops.resolve_reply_context(event_for(channel_reply()))

    assert context is not None
    assert isinstance(context, ReplyContext)


def test_the_class_delegates_to_the_events_module() -> None:
    """The engine calls the class and never the module function, so a missing delegation
    would be invisible until an adapter with a real quote source landed."""
    event = event_for(personal_message())
    ops, _ = make_ops()

    assert ops.resolve_reply_context(event) == resolve_in_events(event)


# --- it never raises, on anything --------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(
            InboundEvent(message_id="", text="", sender_id="", sender_display="", history_key=""),
            id="blank-event",
        ),
        pytest.param(
            InboundEvent(
                message_id="19:x:1",
                text=QUESTION,
                sender_id=SENDER_ID,
                sender_display="Ada Lovelace",
                history_key="19:x",
                claim_meta={},
                raw=None,
            ),
            id="no-claim-meta-no-raw",
        ),
    ],
)
def test_it_answers_an_event_it_can_learn_nothing_from(event: InboundEvent) -> None:
    """The member runs after the claim, where a raise is not free: it would cost the
    answer to a question the bridge has already acknowledged. An event carrying no
    Teams-side references at all — an older bridge's entry, a hand-built replay — is
    answered rather than rejected, because the answer never depended on them."""
    ops, _ = make_ops()

    assert ops.resolve_reply_context(event) == EMPTY


# --- what it contributes to the persisted entry ------------------------------


def test_the_meta_collides_with_nothing_because_there_is_none() -> None:
    """``ReplyContext.meta`` merges into the same flat entry as the engine's own
    bookkeeping and the adapter's ``ms_`` claim keys. Empty is disjoint from both — and
    the assertion is written against the real key sets so that a future quote source
    adding meta here has to satisfy it rather than inherit it."""
    ops, _ = make_ops()
    event = event_for(channel_reply())

    context = ops.resolve_reply_context(event)

    assert context is not None
    meta = context.meta
    assert set(meta).isdisjoint(RESERVED_ENTRY_KEYS)
    assert set(meta).isdisjoint(set(event.claim_meta))
