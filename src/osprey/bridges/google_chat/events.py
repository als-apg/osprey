"""Google Chat's wire format in, the engine's :class:`InboundEvent` out.

This is the adapter's inbound translation: which Chat events are questions for
this app, what the question text is once the @mention is removed, and which
Google-side references have to survive onto the persisted dedup entry so the
rest of the pipeline never needs the wire event again.

**Two wire shapes arrive and both must parse.** The classic Chat event is what
the HTTP-endpoint delivery mode and every published example show::

    {"type": "MESSAGE", "message": {"name": ..., "sender": ..., "text": ...,
     "thread": {...}, "space": {...}, "annotations": [...]}}

The Workspace **add-on** shape is what Google actually publishes to Pub/Sub for
an app configured through the add-on runtime, and it nests the same Message
resource one level down beside its own space/user context::

    {"commonEventObject": {...},
     "chat": {"user": {...}, "eventTime": ...,
              "messagePayload": {"message": {...}, "space": {...}}}}

:func:`_extract_message` is the switch between them, and it *grafts* the
payload's space and user onto the message so everything downstream sees one
shape. The graft is deliberately conditional — a key already on the message
wins — because the message's own ``space`` is the typed one (it carries
``spaceType``) while the payload's copy has been observed untyped. Grafting
unconditionally would overwrite the typed object with the untyped one and
silently defeat DM detection for every direct message.

**The mention filter is the access control for a room.** Outside a direct
message, a Chat event becomes a question only if its annotations carry a
``USER_MENTION`` naming *this* app: a message mentioning somebody else, or one
that merely contains the literal text ``@Osprey``, is not addressed to the app
and is ignored. In a direct-message space there is nobody else to address, so
every human message is implicitly a question — DM messages usually carry no
mention annotation at all. That is the only behavioral difference between the
two space types.

**Text is Chat's own rendering, minus the app's mention.** Chat ships
``argumentText`` already mention-stripped, so it is preferred outright; the
annotation-offset fallback exists for the events that omit it, and slices
back-to-front so earlier offsets stay valid. Only the ends are stripped —
pasted code and markdown survive intact.

Nothing in the parse path does I/O and nothing there raises: it runs before the
dedup claim, so every Pub/Sub redelivery pays for it again, and any payload this
module cannot make sense of is answered with ``None`` rather than an exception
that would leave the message unacknowledged and redelivered forever.

**Reply context is the one part that makes calls**, and it runs after the claim,
so it is paid for once. Chat's native quote-reply arrives as
``quotedMessageMetadata`` on the Message resource, whose inline snapshot carries
the quoted sender and text — but only the REST view of a message is confirmed to
carry that snapshot, so :func:`resolve_reply_context` reads the event first and
re-reads the current message once through the injected ``get_message`` when the
event came without it. A resolved quote costs one further read, of the *quoted*
message, because its own attachments are visible on no other view and a
quote-reply to an image is precisely the case where the agent needs the pixels
rather than only the knowledge that some message was pointed at. Every one of
those reads is enrichment: a failure resolves to less context, never to a failed
answer.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from osprey.bridges.core import InboundEvent, ReplyContext

from .config import GoogleChatBridgeConfig

logger = logging.getLogger(__name__)

GC_SPACE = "gc_space"
"""Entry key for the Chat space resource name (``spaces/AAAA``) — where a reply
is posted, and the history scope of a direct message."""

GC_THREAD = "gc_thread"
"""Entry key for the thread resource name (``spaces/AAAA/threads/DDDD``), or
``None`` when the message started no thread. Replies are threaded onto it, so a
room conversation stays one visible exchange."""

GC_MESSAGE_NAME = "gc_message_name"
"""Entry key for the Chat message resource name this claim was taken for.

Not recoverable from the rest of the entry: the engine keys the store *by*
:attr:`InboundEvent.message_id` and does not copy it into the entry body, and
the members that have to name the original message to Chat (resolving a quote,
addressing a reply) read only the entry."""

GC_IS_DM = "gc_is_dm"
"""Entry key for whether the space is a direct message.

**Written, never read back.** No delivery path branches on it: an answer is addressed
from :data:`GC_SPACE` and :data:`GC_THREAD`, which already carry wherever the reply has
to go. The DM classification does its real work at parse time, before the entry exists
— it exempts a DM from the mention filter, and it picks ``history_key`` (a DM keys on
the space, a room on its thread) — and both results are baked into the entry by the
time it is persisted.

It is kept because it is the one field that records WHY those two came out as they did.
A persisted entry otherwise cannot answer "was this grouped by space because it was a
DM, or because the message started no thread", which is the first question when a
conversation groups unexpectedly. Diagnostic only: nothing may start branching on it
without persisting it under a contract, since an entry written by an older bridge may
not carry it at all."""

GC_ATTACHMENTS = "gc_attachments"
"""Entry key for the message's own uploaded attachment references — a JSON-plain
list of the dicts :func:`parse_attachments` builds. Read by
:meth:`ChannelOps.download_inputs` off the persisted entry alone."""

GC_DRIVE_ATTACHMENTS = "gc_drive_attachments"
"""Entry key for the display names of Drive attachments, which the app cannot
download with its own credentials. Kept so the answer can tell the user which
file was skipped instead of silently ignoring it."""

GC_QUOTED_ATTACHMENTS = "gc_quoted_attachments"
"""Reply-context meta key for the **quoted** message's fetchable attachment
references, in the same shape :func:`parse_attachments` builds for the message's
own.

Deliberately a different key from :data:`GC_ATTACHMENTS` rather than a merge into
it: both land in one flat persisted entry, and the download member has to tell a
file on *this* message from one on the message it quotes — the engine reports the
two provenances to the agent differently, and folding them together would tell
the agent the user had just attached a file they only pointed at."""

GC_QUOTED_DRIVE_ATTACHMENTS = "gc_quoted_drive_attachments"
"""Reply-context meta key for the display names of Drive attachments on the
quoted message — the same unfetchable-by-design case as
:data:`GC_DRIVE_ATTACHMENTS`, kept so the answer can name what it could not
open."""

REPLY_TO_MAX_CHARS = 2000
"""Character cap on the quoted text shipped as reply context.

The quote identifies *what* is being replied to; it does not need to replay it.
A quoted message can be a full answer chunk (Chat allows 4096 characters), and
shipping one whole would let the context outweigh the question in the dispatch
payload. The elision marker counts against the cap, so this is a hard bound on
the string rather than an approximate one."""

ELLIPSIS = "…"
"""Marker appended to a truncated quote, so the agent can see it was cut."""

MESSAGE_EVENT_TYPE = "MESSAGE"
"""The only classic ``type`` that carries a question. ``ADDED_TO_SPACE``,
``REMOVED_FROM_SPACE`` and the card-interaction types are not questions."""

USER_MENTION = "USER_MENTION"
"""Annotation type of a mention that addresses a specific user. The only type
the filter accepts, and the only type :func:`_strip_mention` slices out."""

BOT_SENDER_TYPE = "BOT"
"""``sender.type`` of an app-authored message. The loop guard: without it the
bridge could answer its own acks and answers forever."""

DM_SPACE_TYPES = frozenset({"DIRECT_MESSAGE", "DM"})
"""The direct-message spellings across the two space schemas — ``spaceType`` on
the current Space resource, ``type`` on its deprecated predecessor. Both are
live on the wire, so both are accepted."""

UPLOADED_SOURCE = "UPLOADED_CONTENT"
"""Attachment source whose ``attachmentDataRef.resourceName`` the app can
download with its own credentials."""

DRIVE_SOURCE = "DRIVE_FILE"
"""Attachment source backed by Drive. It exposes only a ``driveDataRef``, which
the app's own auth cannot read, so these are reported by name rather than
fetched."""


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` if it is a mapping, else an empty one.

    Every nested read goes through this. The events are parsed straight off
    Pub/Sub, so a field being a string, a list, or absent where a Chat resource
    was expected is untrusted-input shape rather than a programming error, and
    must degrade to "not for us" instead of raising.
    """
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    """``value`` if it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def _sequence(value: Any) -> list[Any]:
    """``value`` as a list of entries, normalizing the shapes Chat delivers.

    A repeated field arrives as a list on the REST/classic wire but has been
    observed as a single object in some event payloads, so a lone mapping is
    wrapped rather than skipped. Anything else (a string, a number, absent)
    yields no entries.
    """
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return value
    return []


def _first(mapping: Mapping[str, Any], *keys: str) -> str:
    """The first present, non-empty value across ``keys``, as a string.

    Google delivers attachment and quote fields in camelCase over the classic
    Chat event (``contentName``, ``resourceName``) and in snake_case in some
    add-on forms; accepting both keeps one code path for either wire.
    """
    for key in keys:
        value = mapping.get(key)
        if value:
            return str(value)
    return ""


def _ids_match(mention_user_name: str, app_id: str) -> bool:
    """Whether a mention's ``user.name`` names the configured app.

    Compares last path segments, so a fully-qualified ``users/123`` and a bare
    ``123`` match in either position. The looseness is for the operator's
    benefit — ``GCHAT_APP_ID`` is copied out of the Chat console by hand, and
    the two spellings are equally natural to paste.
    """
    if not mention_user_name or not app_id:
        return False
    return mention_user_name.rsplit("/", 1)[-1] == app_id.rsplit("/", 1)[-1]


def _app_mentions(message: Mapping[str, Any], app_id: str) -> list[Mapping[str, Any]]:
    """The ``USER_MENTION`` annotations that name this app.

    One pass serving two questions that must not be conflated: whether the
    message addresses the app at all (any mention counts), and which character
    ranges to remove from the text (only mentions carrying usable offsets). A
    mention Chat sent without offsets still *addresses* the app — treating it as
    no mention would drop a genuine question in a room.
    """
    mentions: list[Mapping[str, Any]] = []
    for annotation in _sequence(message.get("annotations")):
        annotation = _mapping(annotation)
        if annotation.get("type") != USER_MENTION:
            continue
        user = _mapping(_mapping(annotation.get("userMention")).get("user"))
        if _ids_match(_text(user.get("name")), app_id):
            mentions.append(annotation)
    return mentions


def _mention_spans(message: Mapping[str, Any], app_id: str) -> list[tuple[int, int]]:
    """``(start, end)`` character spans of this app's mentions in ``text``.

    Only mentions carrying integer offsets contribute: an annotation without
    them cannot be sliced out, and guessing at its extent would corrupt the
    question rather than clean it up.
    """
    spans: list[tuple[int, int]] = []
    for annotation in _app_mentions(message, app_id):
        start = annotation.get("startIndex")
        length = annotation.get("length")
        if isinstance(start, int) and isinstance(length, int):
            spans.append((start, start + length))
    return spans


def _strip_mention(message: Mapping[str, Any], app_id: str) -> str:
    """The question text with this app's @mention removed.

    Chat's own ``argumentText`` is already mention-stripped and is preferred
    whenever it is present — including when it is empty, which is Chat saying
    the message was nothing but the address. The fallback slices the mention
    spans out of ``text`` back-to-front so an earlier span's removal cannot
    invalidate a later one's offsets.

    Args:
        message: The Message resource, already normalized by
            :func:`_extract_message`.
        app_id: The configured app id; mentions of anyone else are left in the
            text, since they are content rather than addressing.

    Returns:
        The text with only the ends stripped — interior whitespace is left
        exactly as sent, so pasted code and markdown survive.
    """
    argument_text = message.get("argumentText")
    if isinstance(argument_text, str):
        return argument_text.strip()

    text = _text(message.get("text"))
    for start, end in sorted(_mention_spans(message, app_id), reverse=True):
        text = text[:start] + text[end:]
    return text.strip()


def _extract_message(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The Message resource out of either wire shape, or ``None``.

    The add-on envelope keys its payload by event kind, so any key other than
    ``messagePayload`` (``addedToSpacePayload``, ``removedFromSpacePayload``,
    ...) is a non-message event and yields ``None`` — the same outcome the
    classic shape reaches through its ``type`` field.

    The payload's ``space`` and the envelope's ``user`` are grafted onto the
    returned message **only where the message does not already carry that key**.
    That condition is load-bearing rather than defensive: the message's own
    ``space`` is the typed Space resource and the payload's copy has been seen
    without a ``spaceType``, so an unconditional graft would replace the object
    DM detection reads with one that cannot answer the question.

    Returns:
        A message mapping, or ``None`` if this event carries no message. The
        add-on shape always returns a shallow copy, whether or not either graft
        actually fired; the classic shape has nothing to graft and returns the
        event's own message object. Either way the input is never mutated, which
        is the property callers rely on — not which of the two they got.
    """
    chat = event.get("chat")
    if isinstance(chat, Mapping):
        payload = _mapping(chat.get("messagePayload"))
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return None
        grafted = dict(message)
        if "space" not in grafted and isinstance(payload.get("space"), Mapping):
            grafted["space"] = payload["space"]
        if "sender" not in grafted and isinstance(chat.get("user"), Mapping):
            grafted["sender"] = chat["user"]
        return grafted

    if event.get("type") != MESSAGE_EVENT_TYPE:
        return None
    message = event.get("message")
    return message if isinstance(message, Mapping) else None


def parse_attachments(message: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """A message's attachments, split into fetchable refs and Drive names.

    Public because it runs on two views of a Message resource: the inbound
    event's own message, and the ``spaces.messages.get`` view of a *quoted*
    message when a quote-reply resolves — the quoted message's attachments are
    only visible on that second view.

    Only ``UPLOADED_CONTENT`` attachments become references, and only when they
    carry a resource name: a reference nothing could later fetch is worse than
    no reference, because it looks like an attachment the agent should have
    received. A ``DRIVE_FILE`` attachment is never fetchable with the app's own
    credentials, so its display name is collected separately and the answer can
    say it was skipped.

    Args:
        message: A Message resource (or anything at all — junk yields empties).

    Returns:
        ``(references, drive_names)`` in the order Chat listed them. The
        references are JSON-plain dicts with unprefixed keys — ``content_name``,
        ``content_type``, ``resource_name``, ``source`` — since the entry key
        they ride under carries the ``gc_`` namespace and the nesting keeps them
        out of the flat entry dict entirely.
    """
    references: list[dict[str, Any]] = []
    drive_names: list[str] = []
    for entry in _sequence(_mapping(message).get("attachment")):
        entry = _mapping(entry)
        source = _first(entry, "source")
        content_name = _first(entry, "contentName", "content_name")
        if source == DRIVE_SOURCE:
            drive_names.append(content_name)
            continue
        if source and source != UPLOADED_SOURCE:
            continue
        data_ref = _mapping(entry.get("attachmentDataRef") or entry.get("attachment_data_ref"))
        resource_name = _first(data_ref, "resourceName", "resource_name")
        if not resource_name:
            logger.warning(
                "ignoring inbound attachment %r with no resource name: nothing could fetch it",
                content_name,
            )
            continue
        references.append(
            {
                "content_name": content_name,
                "content_type": _first(entry, "contentType", "content_type"),
                "resource_name": resource_name,
                "source": source or UPLOADED_SOURCE,
            }
        )
    return references, drive_names


def parse_event(raw: Any, cfg: GoogleChatBridgeConfig) -> InboundEvent | None:
    """Parse one raw Chat event into an :class:`InboundEvent`, or ignore it.

    Free of I/O and never raising, because it runs before the dedup claim: an
    exception here would leave the Pub/Sub message unacknowledged and the same
    event redelivered forever, where ``None`` acknowledges it and moves on.

    Ignored (``None``) for: anything that is not a Chat event mapping; every
    non-message event in either wire shape; a message the app itself authored
    (the loop guard); a message with no resource name to claim it under; and,
    outside a direct message, any message that does not @mention this app.

    Args:
        raw: One decoded Chat event, in either wire shape.
        cfg: Bridge config; only ``app_id`` is read.

    Returns:
        The parsed event, or ``None`` to ignore this one.
    """
    event = _mapping(raw)
    message = _extract_message(event) if event else None
    if message is None:
        return None

    # The app must never react to app-authored messages — its own acks and
    # answers included. Google should not dispatch them back to us, but a loop
    # here would be one that posts to a live room.
    sender = _mapping(message.get("sender"))
    if sender.get("type") == BOT_SENDER_TYPE:
        return None

    space = _mapping(message.get("space") or event.get("space"))
    is_dm = space.get("spaceType") in DM_SPACE_TYPES or space.get("type") in DM_SPACE_TYPES

    # The space type decides only whether a MISSING mention disqualifies the
    # message; a mention that is present is stripped either way, since it is
    # addressing rather than part of the question.
    if not is_dm and not _app_mentions(message, cfg.app_id):
        return None

    message_name = _text(message.get("name"))
    if not message_name:
        # Without the resource name there is no dedup key, so a redelivery could
        # not be recognized and the same question would be answered twice.
        logger.warning("ignoring Chat message with no resource name; nothing could claim it")
        return None

    space_name = _text(space.get("name"))
    thread_name = _text(_mapping(message.get("thread")).get("name")) or None
    attachments, drive_attachments = parse_attachments(message)

    return InboundEvent(
        # Chat message names are already globally unique (they embed the space),
        # so unlike a Talk message id this needs no extra scoping.
        message_id=message_name,
        text=_strip_mention(message, cfg.app_id),
        sender_id=_text(sender.get("name")),
        sender_display=_text(sender.get("displayName")),
        # A DM is one continuous conversation: each message there technically
        # starts its own thread, so keying a DM by thread would never link two
        # consecutive turns. A room conversation is scoped to its thread, and
        # falls back to the space only for a message that started none.
        history_key=space_name if is_dm else (thread_name or space_name),
        claim_meta={
            GC_SPACE: space_name,
            GC_THREAD: thread_name,
            GC_MESSAGE_NAME: message_name,
            GC_IS_DM: is_dm,
            GC_ATTACHMENTS: attachments,
            GC_DRIVE_ATTACHMENTS: drive_attachments,
        },
        # The normalized Message resource, not the envelope: a quote-reply
        # arrives inline on it, in the same shape the REST message view returns,
        # so reply resolution reads this rather than re-doing the extraction.
        # Adapter-internal; never persisted.
        raw=message,
    )


# --- reply context ---------------------------------------------------------


@dataclass(frozen=True)
class QuotedMessage:
    """The message a Chat quote-reply points at.

    Built from ``quotedMessageMetadata``: the metadata's ``name`` addresses the
    quoted message, and its inline ``quotedMessageSnapshot`` carries what was
    said. ``sender`` is a display string (``"Alice"``), not a ``users/...``
    resource name — the snapshot has no other spelling of who wrote it, which is
    also all the reply context needs.
    """

    message_name: str
    """Resource name of the quoted message, or ``""`` when the metadata omitted
    it. The handle for the one further read that finds its attachments; without
    it there is nothing to address, and the quote resolves on text alone."""

    sender: str
    """Display name of whoever wrote the quoted message."""

    text: str
    """The quoted message's text, untruncated. Empty for a message that was only
    an image or a card, which is still a quote worth reporting."""


def parse_quoted(message: Any) -> QuotedMessage | None:
    """The quote-reply target on a Message resource, or ``None``.

    Runs on two views of the same resource — the inbound event's message and the
    ``spaces.messages.get`` body — because both carry the metadata in the same
    shape, and which one supplies it is exactly what
    :func:`resolve_reply_context` is deciding.

    Args:
        message: A Message resource (or anything at all — junk yields ``None``).

    Returns:
        The quoted message, or ``None`` when this message is not a quote-reply
        **or** its snapshot carries neither a sender nor text. That second case
        is not the same as "no quote": name-only metadata is unusable as context
        on its own, and answering ``None`` is what lets the caller's re-read fire
        and resolve the actual content instead of shipping an empty quote.
    """
    metadata = _mapping(
        _mapping(message).get("quotedMessageMetadata")
        or _mapping(message).get("quoted_message_metadata")
    )
    snapshot = _mapping(
        metadata.get("quotedMessageSnapshot") or metadata.get("quoted_message_snapshot")
    )
    sender = _first(snapshot, "sender")
    text = _first(snapshot, "text")
    if not sender and not text:
        return None
    return QuotedMessage(message_name=_first(metadata, "name"), sender=sender, text=text)


def _truncate(text: str, limit: int = REPLY_TO_MAX_CHARS) -> str:
    """``text`` shortened to at most ``limit`` characters.

    The elision marker is counted against the limit rather than added on top, so
    the return value is a hard bound and a caller sizing a payload around it
    cannot be surprised.
    """
    if len(text) <= limit:
        return text
    return text[: max(limit - len(ELLIPSIS), 0)].rstrip() + ELLIPSIS


def _quoted_attachments(
    quoted: QuotedMessage,
    get_message: Callable[[str], Mapping[str, Any] | None],
) -> tuple[list[dict[str, Any]], list[str]]:
    """The quoted message's own attachments, via one guarded read.

    The inline snapshot carries sender and text and nothing else, so the quoted
    message's ``attachment`` list is reachable only on its own message view. The
    resource names it yields stay valid for as long as the message exists, so a
    quote-reply to a message posted long before the app was deployed still
    resolves its files.

    Guarded on its own rather than under the caller's guard: the text is the
    reply context, the files are a bonus, and losing the whole quote because a
    second read failed would be a worse answer than losing the files.

    Returns:
        ``(references, drive_names)`` as :func:`parse_attachments` builds them,
        or two empty lists when the metadata named no message, the read failed,
        or the body was not a message.
    """
    if not quoted.message_name:
        return [], []
    try:
        return parse_attachments(get_message(quoted.message_name))
    except Exception as exc:
        logger.warning(
            "could not read attachments of quoted message %s: %s", quoted.message_name, exc
        )
        return [], []


def resolve_reply_context(
    event: InboundEvent,
    get_message: Callable[[str], Mapping[str, Any] | None],
) -> ReplyContext | None:
    """Resolve the message a Chat quote-reply points at.

    Event first: when the wire event's own ``quotedMessageMetadata`` carries a
    usable snapshot, nothing is read. Otherwise the current message is re-read
    **once** and the same parser runs on that view, because only the REST view is
    confirmed to carry the snapshot — one cheap guarded read per non-reply
    message, and the INFO log names which path answered so production logs can
    settle the wire-shape question and retire the fallback if the event always
    carries it.

    Args:
        event: The claimed event; its ``raw`` still holds the normalized Message
            resource :func:`parse_event` parsed.
        get_message: Reads one Message resource by name, answering ``None`` on
            failure. Injected rather than imported so this module stays free of
            the Chat transport.

    Returns:
        The reply context, or ``None`` when the message is not a quote-reply or
        resolution failed. ``reply_to`` is ``{"sender": ..., "text": ...}`` with
        the text truncated to :data:`REPLY_TO_MAX_CHARS`; ``meta`` always carries
        both quoted-attachment keys — empty lists when the quoted message had no
        files — so a persisted entry answers "were there quoted files?" the same
        way whether or not there were any.

    Raises:
        Nothing. Reply context is enrichment, so every failure is reported as
        ``None`` and the question is answered without the quote.
    """
    try:
        quoted = parse_quoted(event.raw)
        source = "the event"
        if quoted is None:
            # The one-shot fallback: the CURRENT message, re-read for the
            # snapshot the event may not have carried.
            quoted = parse_quoted(get_message(event.message_id))
            source = "messages.get"
        if quoted is None:
            logger.info("no reply context for %s: neither view carried a quote", event.message_id)
            return None

        logger.info("reply context for %s resolved from %s", event.message_id, source)
        references, drive_names = _quoted_attachments(quoted, get_message)
        return ReplyContext(
            reply_to={"sender": quoted.sender, "text": _truncate(quoted.text)},
            meta={
                GC_QUOTED_ATTACHMENTS: references,
                GC_QUOTED_DRIVE_ATTACHMENTS: drive_names,
            },
        )
    except Exception as exc:  # enrichment: never fatal to the dispatch
        logger.warning("could not resolve reply context for %s: %s", event.message_id, exc)
        return None
