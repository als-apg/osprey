"""Microsoft Teams' wire format in, the engine's :class:`InboundEvent` out.

This is the adapter's inbound translation: which Bot Framework activities are
questions for this bot, what the question text is once the bot's ``@mention`` is
removed, and which Teams-side references have to survive onto the persisted dedup
entry so the rest of the pipeline never needs the wire activity again.

**One wire shape arrives.** The relay enqueues the Bot Framework *activity*
verbatim, so this module parses exactly what the Bot Connector POSTs::

    {"type": "message", "id": "1700000000999", "serviceUrl": "https://smba.../",
     "from": {"id": "29:user", "name": "Alice"},
     "conversation": {"id": "19:room@thread.tacv2;messageid=170...",
                      "conversationType": "channel", "tenantId": "..."},
     "text": "<at>Osprey</at> what is the beam current?",
     "entities": [{"type": "mention", "text": "<at>Osprey</at>",
                   "mentioned": {"id": "28:<app id>", "name": "Osprey"}}]}

**The mention filter is the access control for a channel.** In a channel or a
group chat an activity becomes a question only if its ``mention`` entities name
*this* bot — ``mentioned.id`` is the bot's app id under Teams' ``28:`` actor
prefix. A message addressed to a colleague, or one that merely contains the
literal text ``@Osprey``, is not for the bot and is ignored. A 1:1 chat
(``conversationType == "personal"``) has nobody else to address, so every human
message there is implicitly a question; Teams sends no mention entity at all in
that case. That is the only behavioral difference between the conversation types,
and an activity whose ``conversationType`` is missing is treated as *not*
personal — requiring the mention is the fail-closed direction.

**Text keeps every mention but the bot's.** Teams ships the raw ``<at>…</at>``
markup inline and describes each span in an entity, so the bot's own span is
deleted (it is addressing, not content) while every other mention is replaced by
the person's display name — "ask <at>Alice</at> about the magnets" has to reach
the agent as "ask Alice about the magnets", because who was named is part of the
question. Only the ends are stripped, so pasted code and markdown survive.

**A channel conversation is keyed by its thread root.** Teams gives a reply the
conversation id of the thread it is in (``<channel id>;messageid=<root id>``) but
gives the *root post* the bare channel id, so the two forms have to be normalized
to one or a thread's first turn would never link to its second. The root post's
own activity id IS the root message id, which is what makes the normalization
possible without a read. A chat — personal or group — is one continuous
conversation and keys on the conversation id as sent.

Nothing in the parse path does I/O and nothing there raises: it runs before the
dedup claim, so every Service Bus redelivery pays for it again, and any activity
this module cannot make sense of is answered with ``None`` rather than an
exception that would leave the queue message unsettled and redelivered forever.

**Reply context is empty here, deliberately.** Teams delivers no inbound quote to
a bot without resource-specific-consent permissions the deployment does not ask
for, so there is nothing to resolve and :func:`resolve_reply_context` makes no
call and takes no failure path — see its docstring.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from osprey.bridges.core import InboundEvent, ReplyContext

from .config import TeamsBridgeConfig

logger = logging.getLogger(__name__)

MS_SERVICE_URL = "ms_service_url"
"""Entry key for the Bot Connector base URL this activity arrived from
(``https://smba.trafficmanager.net/emea/``).

Per-tenant and per-region, and NOT derivable from anything else on the entry, so
it is persisted rather than configured: a reply is always posted back to the host
the activity itself named, which is also why the bridge needs no outbound host
table for the Azure cloud it runs in."""

MS_CONVERSATION_ID = "ms_conversation_id"
"""Entry key for the conversation id **exactly as sent** — including the
``;messageid=<root id>`` suffix a channel reply carries.

The as-sent spelling is the one the Bot Connector addresses, so this is the
segment every outbound post puts in its URL. It is deliberately NOT the
normalized :attr:`InboundEvent.history_key`: posting to the normalized form would
address the thread root rather than the conversation the activity came from."""

MS_ACTIVITY_ID = "ms_activity_id"
"""Entry key for the activity id this claim was taken for.

Not recoverable from the rest of the entry: the engine keys the store by
:attr:`InboundEvent.message_id`, which scopes this id by the conversation, and
does not split it back apart. The outbound members need the bare id to reply to
the message rather than to the conversation."""

MS_CONVERSATION_TYPE = "ms_conversation_type"
"""Entry key for Teams' conversation type — ``personal``, ``groupChat`` or
``channel``.

**Written, never read back by the engine.** It does its real work at parse time,
before the entry exists: it exempts a 1:1 chat from the mention filter and picks
which ``history_key`` shape applies. It is kept because it is the one field that
records WHY those two came out as they did — a persisted entry otherwise cannot
answer "was this keyed on the bare conversation id because it was a chat, or
because the channel id already carried its root". Diagnostic only; an entry
written by an older bridge may not carry it at all."""

MS_TENANT_ID = "ms_tenant_id"
"""Entry key for the Microsoft 365 tenant the activity came from.

Persisted for provenance: a single bot registration can be installed in more than
one tenant, and an answer, a give-up note or an audit question about a past run
should be able to say which directory the asker was in without re-reading a wire
activity that is long gone."""

MESSAGE_ACTIVITY_TYPE = "message"
"""The only activity ``type`` that carries a question. ``conversationUpdate``,
``messageReaction``, ``invoke``, ``typing`` and the rest are not questions."""

MENTION_ENTITY_TYPE = "mention"
"""``entities[].type`` of a span that addresses somebody. The only entity type the
filter accepts and the only one whose markup is rewritten."""

BOT_ID_PREFIX = "28:"
"""Teams' actor prefix for a bot. ``mentioned.id`` and a bot-authored activity's
``from.id`` are both the app registration id under this prefix, so it is what
turns the configured ``TEAMS_APP_ID`` into an identity the wire can be matched
against."""

BOT_ROLE = "bot"
"""``from.role`` of a bot-authored activity. Checked alongside the id because the
role is the field Bot Framework documents for the purpose, while the id match is
what still holds when a relay forwards an activity with no role at all."""

PERSONAL_CONVERSATION = "personal"
"""``conversation.conversationType`` of a 1:1 chat with the bot — the only type
exempt from the mention filter."""

CHANNEL_CONVERSATION = "channel"
"""``conversation.conversationType`` of a team channel — the only type whose
conversation id needs normalizing to its thread root."""

ROOT_MESSAGE_MARKER = ";messageid="
"""Separator Teams puts between a channel id and the id of the thread root, in a
reply's conversation id. Its presence is what distinguishes a reply's conversation
id from a root post's bare channel id."""


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` if it is a mapping, else an empty one.

    Every nested read goes through this. Activities are parsed straight off the
    queue, so a field being a string, a list, or absent where a Bot Framework
    object was expected is untrusted-input shape rather than a programming error,
    and must degrade to "not for us" instead of raising.
    """
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    """``value`` if it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def _sequence(value: Any) -> list[Any]:
    """``value`` as a list of entries, normalizing the shapes seen on the wire.

    ``entities`` is a repeated field and arrives as a list, but a relay or a
    hand-built activity has been seen to send a lone object; wrapping it is
    cheaper than dropping a genuine mention. Anything else (a string, a number,
    absent) yields no entries.
    """
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return value
    return []


def bot_actor_id(app_id: str) -> str:
    """The configured app id as the actor id Teams puts on the wire.

    Public because two collaborators must agree on the spelling: the mention
    filter here, and any caller comparing an activity's author against the bot.
    An unset app id yields ``""``, which matches nothing — a bridge with no app id
    configured answers nobody rather than everybody.

    Args:
        app_id: The bot's app-registration (client) id, or ``""``.

    Returns:
        ``"28:<app id>"``, or ``""`` when ``app_id`` is empty.
    """
    return f"{BOT_ID_PREFIX}{app_id}" if app_id else ""


def _mention_entities(activity: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Every ``mention`` entity on the activity, in the order Teams listed them.

    The type is compared case-insensitively: Teams sends ``mention``, but the
    Bot Framework schema is not case-normative and a mention silently classified
    as "some other entity" would drop a genuine question in a channel.
    """
    entities: list[Mapping[str, Any]] = []
    for entry in _sequence(activity.get("entities")):
        entry = _mapping(entry)
        if _text(entry.get("type")).lower() == MENTION_ENTITY_TYPE:
            entities.append(entry)
    return entities


def _mentions_bot(activity: Mapping[str, Any], app_id: str) -> bool:
    """Whether any mention entity names this bot.

    Answers the addressing question only. A mention Teams sent without the
    ``text`` span this module would need to strip still *addresses* the bot, and
    treating it as no mention would drop a real question — so the two concerns
    (does it address us, and which markup can we rewrite) are kept apart.
    """
    actor_id = bot_actor_id(app_id)
    if not actor_id:
        return False
    return any(
        _text(_mapping(entity.get("mentioned")).get("id")) == actor_id
        for entity in _mention_entities(activity)
    )


def _strip_mentions(text: str, activity: Mapping[str, Any], app_id: str) -> str:
    """The question text with the bot's ``<at>`` span gone and the others named.

    Teams leaves the ``<at>Name</at>`` markup inline in ``text`` and describes
    each span in an entity, so the rewrite is driven off the entities rather than
    off a pattern: only a span Teams itself declared is touched, and a literal
    ``<at>`` a user typed is left alone as the content it is.

    The bot's own span is deleted because it is addressing. Every other mention
    becomes the mentioned person's display name, because who was named is part of
    the question — an agent asked to "check with <at>Alice</at>" must be told it
    was Alice, and neither raw markup nor a hole in the sentence says that. A
    mention entity carrying no ``text`` span is left in place: there is nothing to
    match, and guessing at its extent would corrupt the question rather than clean
    it up.

    Args:
        text: The activity's raw ``text``.
        activity: The activity, read for its ``entities``.
        app_id: The configured app id; mentions of anyone else are renamed rather
            than removed.

    Returns:
        The text with only the ends stripped — interior whitespace is left exactly
        as sent, so pasted code and markdown survive.
    """
    actor_id = bot_actor_id(app_id)
    for entity in _mention_entities(activity):
        span = _text(entity.get("text"))
        if not span:
            continue
        mentioned = _mapping(entity.get("mentioned"))
        is_bot = bool(actor_id) and _text(mentioned.get("id")) == actor_id
        text = text.replace(span, "" if is_bot else _text(mentioned.get("name")))
    return text.strip()


def _history_key(conversation_id: str, conversation_type: str, activity_id: str) -> str:
    """The per-conversation transcript key for one activity.

    A channel normalizes to ``<channel id>;messageid=<root id>`` so a thread's
    root post and every reply in it land on one key. Teams gives a reply that
    exact string as its conversation id, while a root post gets the bare channel
    id — and the root post's own activity id *is* the root message id, so the two
    forms reconcile with no read. Anything that is not a channel (a 1:1 chat, a
    group chat) is already one continuous conversation and keys on its
    conversation id as sent.

    Args:
        conversation_id: ``conversation.id`` as sent.
        conversation_type: ``conversation.conversationType`` as sent.
        activity_id: The activity's own id.

    Returns:
        The history key. Empty only if ``conversation_id`` was.
    """
    if conversation_type != CHANNEL_CONVERSATION:
        return conversation_id
    if ROOT_MESSAGE_MARKER in conversation_id:
        return conversation_id
    return f"{conversation_id}{ROOT_MESSAGE_MARKER}{activity_id}"


def _tenant_id(activity: Mapping[str, Any], conversation: Mapping[str, Any]) -> str:
    """The tenant the activity came from, across the two places Teams puts it.

    ``conversation.tenantId`` is the documented home and is preferred;
    ``channelData.tenant.id`` is where it appears on activities from a team
    channel, and has been seen as the only copy. Absent from both is not an error
    — the tenant is provenance, never a decision input — so it degrades to ``""``.
    """
    return _text(conversation.get("tenantId")) or _text(
        _mapping(_mapping(activity.get("channelData")).get("tenant")).get("id")
    )


def parse_event(raw: Any, cfg: TeamsBridgeConfig) -> InboundEvent | None:
    """Parse one raw Teams activity into an :class:`InboundEvent`, or ignore it.

    Free of I/O and never raising, because it runs before the dedup claim: an
    exception here would leave the Service Bus message unsettled and the same
    activity redelivered until its lock expired, forever.

    Ignored (``None``) for: anything that is not an activity mapping; every
    activity whose ``type`` is not ``message``; an activity the bot itself
    authored (the loop guard — without it the bridge would answer its own acks);
    an activity with no conversation id or no activity id to claim it under;
    outside a 1:1 chat, any message that does not @mention this bot; and a message
    whose text is nothing but the mention, which is an address with no question in
    it.

    Args:
        raw: One decoded Bot Framework activity, as the relay enqueued it.
        cfg: Bridge config; only ``app_id`` is read.

    Returns:
        The parsed event, or ``None`` to ignore this one.
    """
    activity = _mapping(raw)
    if _text(activity.get("type")).lower() != MESSAGE_ACTIVITY_TYPE:
        return None

    # The bot must never react to its own activities — its acks and answers
    # included. The Bot Connector should not echo them back, but a loop here
    # would be one that posts to a live channel.
    sender = _mapping(activity.get("from"))
    sender_id = _text(sender.get("id"))
    if _text(sender.get("role")).lower() == BOT_ROLE:
        return None
    actor_id = bot_actor_id(cfg.app_id)
    if actor_id and sender_id == actor_id:
        return None

    conversation = _mapping(activity.get("conversation"))
    conversation_id = _text(conversation.get("id"))
    activity_id = _text(activity.get("id"))
    if not conversation_id or not activity_id:
        # Without both there is no dedup key, so a redelivery could not be
        # recognized and the same question would be answered twice.
        logger.warning("ignoring Teams activity with no conversation or activity id")
        return None

    # The conversation type decides only whether a MISSING mention disqualifies
    # the message; a mention that is present is rewritten either way, since it is
    # addressing rather than part of the question.
    conversation_type = _text(conversation.get("conversationType"))
    if conversation_type != PERSONAL_CONVERSATION and not _mentions_bot(activity, cfg.app_id):
        return None

    text = _strip_mentions(_text(activity.get("text")), activity, cfg.app_id)
    if not text:
        # An address with nothing after it. Dispatching it would spend a run to
        # answer no question, and the person is about to type the real one.
        logger.info("ignoring Teams activity %s: no question left after the mention", activity_id)
        return None

    return InboundEvent(
        # Teams activity ids are unique only within their conversation, so the
        # dedup key is scoped by it — the same discipline a Nextcloud Talk
        # adapter applies to a room-scoped message id.
        message_id=f"{conversation_id}:{activity_id}",
        text=text,
        sender_id=sender_id,
        sender_display=_text(sender.get("name")),
        history_key=_history_key(conversation_id, conversation_type, activity_id),
        claim_meta={
            MS_SERVICE_URL: _text(activity.get("serviceUrl")),
            # As sent, NOT normalized: this is the segment an outbound post
            # addresses, and the normalized key names the thread root instead.
            MS_CONVERSATION_ID: conversation_id,
            MS_ACTIVITY_ID: activity_id,
            MS_CONVERSATION_TYPE: conversation_type,
            MS_TENANT_ID: _tenant_id(activity, conversation),
        },
        # The activity itself, for adapter-internal use only; never persisted.
        raw=activity,
    )


def resolve_reply_context(event: InboundEvent) -> ReplyContext:
    """The empty reply context — Teams delivers no inbound quote to this bot.

    A bot reads a Teams message it was not mentioned in, or the message a reply
    points at, only with resource-specific consent granted by a tenant admin. This
    deployment asks for no such permission, so a reply's parent is simply not on
    the wire: there is nothing to resolve, nothing to read, and no failure path.
    The empty context is returned rather than ``None`` so the seam answers one
    shape for every entry and a future quote source has a place to fill in.

    Args:
        event: The claimed event. Unused — the answer does not depend on it.

    Returns:
        A :class:`ReplyContext` with empty ``reply_to`` and empty ``meta``. The
        engine treats an empty ``reply_to`` as "no quote" and ships none to the
        worker.

    Raises:
        Nothing, on any input.
    """
    return ReplyContext(reply_to={}, meta={})
