"""Talk's wire format in, the engine's :class:`InboundEvent` out.

This is the whole of the adapter's inbound translation: which Talk messages are
questions for the bot, what the question text actually is once Talk's
placeholders are resolved, and which Nextcloud-side references have to survive
onto the persisted dedup entry so the rest of the pipeline never needs the wire
event again.

Three decisions here are load-bearing.

**The mention filter is the access control for a group room.** A group-room
message reaches the agent only if it carries a ``messageParameters`` entry of
type ``user`` whose id is the bot's own account. Matching on the parsed
parameters rather than on the text is what makes it exact: ``@all``, a mention of
some other user, and a message that merely contains the literal string
``@osprey-bot`` are all *not* addressed to the bot and are ignored. In a
one-to-one conversation there is nobody else to address, so no mention is
required — that is the only behavioral difference between the two room types, and
it is why the room type must be known before a message can be judged at all.

**Unresolved room type raises; it does not filter.** ``parse_event`` never
raises on malformed *wire data* — a truncated payload, a missing field, a
``messageParameters`` PHP-empty-array, anything the network hands it is answered
with ``None``. But an unresolved room type is not bad data, it is a violated
precondition: the poller must not fetch or dispatch a room until
:meth:`RoomDirectory.type_of` resolves. So :class:`RoomTypeUnresolved` from
:meth:`RoomDirectory.is_one_to_one` propagates deliberately. Returning ``None``
there would be the one genuinely destructive option available in this module:
``None`` means "not for us", so the engine takes no dedup claim, and the poller's
cursor has already moved past the message — it would be gone for good. Raising
instead leaves the persisted offset unadvanced, so the message is re-fetched once
the type resolves. Deferred, never dropped.

**Text is Talk's own rendering, minus the bot's mention.** Talk sends
``message`` with ``{placeholder}`` tokens and a ``messageParameters`` map giving
each one a display ``name``; what a human reads in the Talk UI is that
substitution. The agent is given the same string, so a question like "what is
wrong in {file0}?" reaches it as "what is wrong in plot.png?" and lines up with
the attachment it receives. The one token removed rather than rendered is the
bot's own mention, which is addressing and not content. Interior whitespace is
left exactly as Talk sent it (pasted code and markdown survive); only the ends
are stripped.

:func:`resolve_reply_context` completes the inbound half. Talk delivers a quoted
message *inline*, as a ``parent`` object on the wire event, so resolving a reply
is pure parsing with no call to make — and the quoted message's own attachment
references are persisted alongside the quote, because after a restart the retry
drain has nothing but the persisted entry and a re-dispatched question would
otherwise silently lose the file its user was pointing at.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from osprey.bridges.core import InboundEvent, ReplyContext

from .config import NextcloudBridgeConfig
from .rooms import RoomDirectory

logger = logging.getLogger(__name__)

MENTION_PARAM_TYPE = "user"
"""``messageParameters`` type of a mention that addresses a specific account.

The only type the filter accepts. Talk uses other types for the mention forms
that are deliberately ignored — ``call`` for ``@all``, plus ``group``,
``circle``, and ``guest`` — so an ``@all`` in a busy room does not summon the
agent for everyone in it."""

FILE_PARAM_TYPE = "file"
"""``messageParameters`` type carrying a shared file's reference."""

COMMENT_MESSAGE_TYPE = "comment"
"""The only ``messageType`` the bridge answers.

Excludes Talk's system events (joins, calls, renames) and its non-text message
kinds (voice messages, deleted comments), none of which carry a question."""

QUOTED_FILES_KEY = "nc_quoted_files"
"""Entry key the **quoted** message's file references ride under.

Deliberately distinct from ``claim_meta``'s ``nc_files``: both land in the same
flat persisted entry, and :meth:`ChannelOps.download_inputs` has to tell an
attachment on *this* message from one on the message it quotes — the two
provenance buckets are reported to the agent differently."""

MAX_QUOTED_TEXT = 2000
"""Character cap on the quoted text shipped as reply context.

The quote is *context*, not the question. Talk allows a 32,000-character message,
and shipping one whole as context would swamp the actual question in the
dispatch payload; 2,000 characters is room for a real quoted question or a pasted
error and little more. The elision marker counts against the cap, so this is a
hard bound on the string."""

ELLIPSIS = "…"
"""Marker appended to a truncated quote, so the agent can see it was cut."""

_PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.-]+)\}")
"""One ``{parameter-key}`` token in Talk's ``message`` field."""


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` if it is a mapping, else an empty one.

    Talk serializes an *absent* ``messageParameters`` as PHP's empty array —
    ``[]``, not ``{}`` — so every parameter map has to be type-checked rather
    than assumed. Same trap as the room payload's ``lastMessage``.
    """
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    """``value`` if it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def talk_int(value: Any) -> int | None:
    """``value`` as an int, or ``None`` if it is not one.

    Talk's numeric fields arrive inconsistently typed — message ids as JSON
    numbers, file ids and sizes as strings (PHP stringifies them) — so both
    spellings are accepted wherever a number is expected.

    Public within the package rather than module-private: the poll loop reads a
    raw message's id with the same rule (see
    :func:`~osprey.bridges.nextcloud_talk.poller._message_id`), and restating it
    there would leave two copies of a wire-format quirk to keep in step.
    ``bool`` is rejected explicitly — it is an ``int`` subclass but never a
    Talk id.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _id_str(value: Any) -> str:
    """A parameter's ``id`` as a string.

    Talk stringifies these (PHP), but an int is tolerated so a payload that
    sends one is not read as an id of ``""``.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return ""


def _same_account(left: str, right: str) -> bool:
    """Whether two Nextcloud account ids name the same account.

    Compared case-insensitively on purpose. Nextcloud logins are
    case-insensitive while its API echoes an account's canonical casing, so an
    operator who sets ``NEXTCLOUD_BOT_ACCOUNT=OSPREY-Bot`` for the uid
    ``osprey-bot`` authenticates fine and then hits two failures at once: the
    loop guard stops recognizing the bot's own messages — the bridge answers its
    own answers, forever, in a live room — and no mention is ever matched, so it
    answers nothing else. Both are silent, and both are avoided by not letting
    casing decide identity.
    """
    return left.casefold() == right.casefold()


def _is_comment(raw: Mapping[str, Any]) -> bool:
    """Whether ``raw`` is a real chat comment rather than a system event.

    Both signals are checked because they answer different questions: a
    non-empty ``systemMessage`` marks the event kind (someone joined, a call
    started), while ``messageType`` distinguishes a comment from Talk's other
    message kinds. An absent ``messageType`` is treated as a comment — the
    ``systemMessage`` check has already excluded the system events, and dropping
    a real question over a missing field is a silent failure.
    """
    if _text(raw.get("systemMessage")):
        return False
    message_type = _text(raw.get("messageType"))
    return message_type in ("", COMMENT_MESSAGE_TYPE)


def _bot_mention_key(params: Mapping[str, Any], bot_account: str) -> str | None:
    """Parameter key of the mention addressing ``bot_account``, if any.

    Args:
        params: The message's ``messageParameters``.
        bot_account: The bot's Nextcloud account id.

    Returns:
        The key whose parameter is a ``user`` mention of the bot (so the caller
        can strip that token from the text), or ``None`` if the message mentions
        the bot nowhere.
    """
    for key, param in params.items():
        param = _mapping(param)
        if param.get("type") != MENTION_PARAM_TYPE:
            continue
        if _same_account(_id_str(param.get("id")), bot_account):
            return key
    return None


def _label(param: Any) -> str:
    """Display label Talk renders a parameter as, or ``""`` if it offers none."""
    return _text(_mapping(param).get("name")).strip()


def _render(body: str, params: Mapping[str, Any], drop: str | None) -> str:
    """Substitute Talk's placeholders in ``body``, removing ``drop`` entirely.

    One pass, so a label that itself contains braces cannot be re-substituted. A
    token with no usable label — and any ``{...}`` the user simply typed — is
    left verbatim rather than deleted: dropping text nobody can account for is
    worse than showing the agent a stray brace.

    Args:
        body: Talk's raw ``message`` field.
        params: The message's ``messageParameters``.
        drop: Parameter key to remove rather than render (the bot's own
            mention), or ``None``.

    Returns:
        The rendered text, stripped at both ends only.
    """

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == drop:
            return ""
        return _label(params.get(key)) or match.group(0)

    return _PLACEHOLDER.sub(substitute, body).strip()


def _file_refs(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Inbound file references from a message's ``messageParameters``.

    Everything :meth:`ChannelOps.download_inputs` needs to re-fetch an
    attachment **from the persisted entry alone**, since by the time the retry
    drain or a post-restart reconcile runs the wire event is long gone. ``path``
    is the one indispensable field — it is what
    :meth:`TalkClient.dav_download` resolves inside the bot's own DAV tree — so
    a reference without one is skipped rather than persisted unusable.

    Args:
        params: The message's ``messageParameters`` (a quoted message's, when
            called for a reply context).

    Returns:
        JSON-plain reference dicts in the order Talk listed them: ``id``,
        ``name``, ``path``, ``size`` (``None`` when Talk sent none), and
        ``mimetype``. Keys are unprefixed — the entry key these ride under
        carries the ``nc_`` namespace, and nesting keeps them from touching the
        flat entry dict at all.
    """
    refs: list[dict[str, Any]] = []
    for key, param in params.items():
        param = _mapping(param)
        if param.get("type") != FILE_PARAM_TYPE:
            continue
        path = _text(param.get("path")).strip()
        if not path:
            logger.warning(
                "ignoring inbound file %r with no DAV path: nothing could fetch it later", key
            )
            continue
        refs.append(
            {
                "id": _id_str(param.get("id")),
                "name": _text(param.get("name")),
                "path": path,
                "size": talk_int(param.get("size")),
                "mimetype": _text(param.get("mimetype")),
            }
        )
    return refs


def parse_event(
    raw: Any,
    room: str,
    cfg: NextcloudBridgeConfig,
    rooms: RoomDirectory,
) -> InboundEvent | None:
    """Parse one raw Talk chat message into an :class:`InboundEvent`.

    Cheap and free of network I/O — it runs before the dedup claim, so every
    redelivery pays for it again, and ``rooms`` is consulted as a pure cache
    read.

    Ignored (``None``) for: anything that is not a Talk message object; system
    events and non-comment message kinds; the bot's own messages (the loop
    guard, without which the bridge would answer itself); a message with no
    usable id; empty text; and, in a **group** room only, any message that does
    not mention the bot account. In a one-to-one conversation no mention is
    needed.

    Args:
        raw: One message object from :meth:`TalkClient.poll_messages`.
        room: Token of the room it was polled from — the authority on where the
            message came from, since it decides both the dedup scope and where
            replies go.
        cfg: Bridge config; only ``bot_account`` is read.
        rooms: The room-type cache. Read-only.

    Returns:
        The parsed event, or ``None`` to ignore the message.

    Raises:
        RoomTypeUnresolved: If ``room``'s type has not resolved yet. This is the
            one exception to the never-raises contract, and it is not about bad
            data: it means a caller bypassed the poller-side hold, which is a
            programming error. It propagates so the poll cycle leaves its
            persisted offset unadvanced and the message is re-fetched later —
            returning ``None`` here would instead discard it silently, with no
            dedup claim and the cursor already past it.
    """
    raw = _mapping(raw)
    if not raw or not _is_comment(raw):
        return None

    actor_id = _text(raw.get("actorId"))
    if _same_account(actor_id, cfg.bot_account):
        return None

    message_id = talk_int(raw.get("id"))
    if message_id is None:
        # Without an id there is no dedup key, and a message that cannot be
        # claimed could be answered again on every redelivery.
        logger.warning("ignoring message in room %s with no usable id: %r", room, raw.get("id"))
        return None

    body = _text(raw.get("message"))
    if not body.strip():
        return None

    # Asked unconditionally, before the mention is even looked for. The rule is
    # that no message is JUDGED until its room's type is known; folding this into
    # the condition below would short-circuit whenever a mention happened to be
    # present, leaving the rule true only by accident.
    one_to_one = rooms.is_one_to_one(room)

    params = _mapping(raw.get("messageParameters"))
    mention_key = _bot_mention_key(params, cfg.bot_account)
    # The room type decides only whether a MISSING mention disqualifies the
    # message; a mention that is present is stripped either way, since it is
    # addressing rather than part of the question.
    if mention_key is None and not one_to_one:
        return None

    text = _render(body, params, mention_key)
    if not text:
        # Nothing but the mention itself — an address with no question.
        return None

    return InboundEvent(
        # Talk message ids are unique only within a room, so the dedup key is
        # room-scoped; two rooms' message 42 must not collide into one claim.
        message_id=f"{room}:{message_id}",
        text=text,
        sender_id=actor_id,
        sender_display=_text(raw.get("actorDisplayName")) or actor_id,
        history_key=f"nextcloud:{room}",
        claim_meta={
            "nc_room": room,
            "nc_message_id": message_id,
            "nc_actor_id": actor_id,
            # Kept because it is not recoverable from sender_id: a guest's or a
            # federated user's actor id is opaque, and only actorType says which
            # kind of account asked.
            "nc_actor_type": _text(raw.get("actorType")),
            "nc_files": _file_refs(params),
        },
        # Talk delivers a quoted message inline, so resolve_reply_context reads
        # this rather than making a call. Adapter-internal; never persisted.
        raw=raw,
    )


# --- reply context ---------------------------------------------------------


def _truncate(text: str, limit: int = MAX_QUOTED_TEXT) -> str:
    """``text`` shortened to at most ``limit`` characters.

    The elision marker is counted against the limit rather than added on top, so
    the return value is a hard bound and a caller sizing a payload around it
    cannot be surprised.
    """
    if len(text) <= limit:
        return text
    return text[: max(limit - len(ELLIPSIS), 0)].rstrip() + ELLIPSIS


def resolve_reply_context(event: InboundEvent) -> ReplyContext | None:
    """Resolve the quoted message a Talk reply points at.

    Free of I/O: Talk ships the quoted message inline as ``parent`` on the wire
    event, so this is pure parsing of :attr:`InboundEvent.raw`.

    The bot's mention is **not** stripped from the quoted text, unlike the
    message's own text. In the quoted message the mention is content — part of
    what was actually said — and rewriting it would misrepresent the quote; it is
    rendered the way Talk displays it, like every other placeholder.

    Args:
        event: The parsed event, whose ``raw`` still holds Talk's payload.

    Returns:
        The reply context, or ``None`` when there is nothing to add: the message
        is not a reply, the parent is a system event or a deleted comment, or the
        parent carries neither text nor a fetchable attachment.

        ``reply_to`` is ``{"sender": ..., "text": ...}`` with the text truncated
        to :data:`MAX_QUOTED_TEXT`. ``meta`` always carries
        :data:`QUOTED_FILES_KEY` when a context is returned — an empty list when
        the quoted message had no attachments — so a persisted entry answers
        "were there quoted files?" the same way whether or not the quote had any.

    Raises:
        Nothing. Reply context is enrichment: the engine treats a failure to
        resolve it as "not a reply" and dispatches the question regardless, so
        every failure here is reported as ``None``.
    """
    try:
        parent = _mapping(_mapping(event.raw).get("parent"))
        if not parent or not _is_comment(parent):
            return None

        params = _mapping(parent.get("messageParameters"))
        # drop=None: nothing is stripped from a quote, not even a mention of the
        # bot — see the note above.
        text = _render(_text(parent.get("message")), params, None)
        refs = _file_refs(params)
        if not text and not refs:
            # A quote that adds neither words nor a file adds nothing; returning
            # a context here would put an empty reply_to in the payload.
            return None

        actor_id = _text(parent.get("actorId"))
        return ReplyContext(
            reply_to={
                "sender": _text(parent.get("actorDisplayName")) or actor_id,
                "text": _truncate(text),
            },
            meta={QUOTED_FILES_KEY: refs},
        )
    except Exception as exc:  # enrichment: never fatal to the dispatch
        logger.warning("could not resolve reply context for %s: %s", event.message_id, exc)
        return None
