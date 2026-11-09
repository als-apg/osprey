"""Slack's wire format in, the engine's :class:`InboundEvent` out.

This is the adapter's inbound translation. It owns the ``sl_*`` entry keys that
carry Slack-side references onto the persisted dedup entry, so the rest of the
pipeline never needs the wire event again; it owns :func:`normalize_text`,
which turns Slack's ``mrkdwn`` markup into the plain question the agent reads; and
it owns :func:`parse_event`, which decides whether an Events API payload is a
question for this bot at all and builds the :class:`InboundEvent` when it is.

**What counts as a question.** Only ``message`` and ``app_mention`` events, from a
person in the installing workspace, that are not edits, joins or other subtypes
(``file_share`` excepted), not hidden, not from any bot (this one included), and —
outside a 1:1 DM — that address the bot with its ``<@BOT>`` token (``app_mention``
is addressing by definition). A channel question arrives only as ``app_mention``,
which carries no ``channel_type``, so the conversation kind is derived from the
channel id when Slack leaves it out.

**Text keeps every mention but the bot's.** Slack ships mentions inline as
``<@U…>`` tokens. The bot's own token is deleted (it is addressing, not content);
every other user is replaced by their display name when the event itself shows it
(the ``user_profile`` data already on the event — never a lookup), else by
``@U…``, because who was named is part of the question. Links, channel references
and broadcasts become their readable forms, and Slack's three HTML escapes are
undone. Only the ends are stripped, so pasted code and markdown survive.

Nothing here does I/O and nothing here raises: it runs before the dedup claim, so
every redelivery pays for it again, and an input this module cannot make sense of
degrades to an empty string rather than an exception.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from osprey.bridges.core import InboundEvent

__all__ = [
    "SL_CHANNEL",
    "SL_CHANNEL_TYPE",
    "SL_FILES",
    "SL_PARENT_FILES",
    "SL_TEAM",
    "SL_TEXT_HEAD",
    "SL_THREAD_TS",
    "SL_TS",
    "BotIdentity",
    "normalize_text",
    "parse_event",
    "people_seen",
]

SL_CHANNEL = "sl_channel"
"""Entry key for the Slack conversation id the question arrived in (``C…``,
``G…`` or ``D…``) — where the answer is posted."""

SL_TS = "sl_ts"
"""Entry key for the question message's own ``ts`` — the message the bridge
reacts to and, for a channel question, the thread root when none was given."""

SL_THREAD_TS = "sl_thread_ts"
"""Entry key for the ``thread_ts`` the answer goes into; empty for a top-level
DM answer, which is posted into the conversation rather than a thread."""

SL_CHANNEL_TYPE = "sl_channel_type"
"""Entry key for the conversation kind: ``im`` | ``mpim`` | ``channel`` |
``group``."""

SL_TEAM = "sl_team"
"""Entry key for the workspace (team) id the event belongs to."""

SL_FILES = "sl_files"
"""Entry key for the files attached to the question itself: a list of
``{id, name, mimetype, size, url}`` refs, downloaded later by the ops layer."""

SL_TEXT_HEAD = "sl_text_head"
"""Entry key for the first line of the question text, used as the quote prefix
of a DM answer."""

SL_PARENT_FILES = "sl_parent_files"
"""Reply-context meta key for the files attached to the thread parent a reply
quotes — the same ``{id, name, mimetype, size, url}`` ref shape as
:data:`SL_FILES`, downloaded into the quoted bucket."""

_TOKEN = re.compile(r"<([^<>]*)>")
_USER = re.compile(r"@([UW][A-Z0-9]+)(?:\|.*)?", re.DOTALL)


def _unescape(text: str) -> str:
    """Undo Slack's escapes; ``&amp;`` last, so ``&amp;lt;`` becomes ``&lt;``."""
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


def _split_label(body: str) -> tuple[str, str]:
    target, _, label = body.partition("|")
    return target, label


def _render_token(body: str, bot_user_id: str, names: Mapping[str, str]) -> str:
    """Render the inside of one ``<…>`` token as readable text."""
    if body.startswith("@"):
        user = _USER.fullmatch(body)
        if user is None:
            return _unescape(body)
        user_id = user.group(1)
        if bot_user_id and user_id == bot_user_id:
            return ""
        name = names.get(user_id)
        if isinstance(name, str) and name.strip():
            return name.strip()
        return f"@{user_id}"
    if body.startswith("#"):
        target, label = _split_label(body[1:])
        return f"#{_unescape(label or target)}"
    if body.startswith("!"):
        target, label = _split_label(body[1:])
        if label:
            label = _unescape(label)
            return label if label.startswith("@") else f"@{label}"
        kind, _, ident = target.partition("^")
        return f"@{_unescape(ident or kind)}"
    target, label = _split_label(body)
    url = _unescape(target)
    if label:
        return f"{_unescape(label)} ({url})"
    return url


def normalize_text(text: str, bot_user_id: str, names: Mapping[str, str]) -> str:
    """Return Slack message *text* as the plain question the agent reads.

    Removes only the bot's own ``<@BOT>``/``<@BOT|x>`` token; other users'
    ``<@U…>`` become their display name from *names* (user id → name, taken
    from data already on the event) when known, else ``@U…``.
    ``<url|text>`` becomes ``text (url)``, a bare ``<url>`` becomes ``url``,
    ``<#C…|name>`` becomes ``#name``, and ``<!here>``-style broadcasts become
    ``@here``. ``&amp; &lt; &gt;`` are unescaped and only the ends are stripped.
    Pure and never raises: a non-string *text* yields ``""``.
    """
    if not isinstance(text, str):
        return ""
    bot = bot_user_id if isinstance(bot_user_id, str) else ""
    known: Mapping[str, str] = names if isinstance(names, Mapping) else {}
    try:
        parts: list[str] = []
        pos = 0
        for match in _TOKEN.finditer(text):
            parts.append(_unescape(text[pos : match.start()]))
            parts.append(_render_token(match.group(1), bot, known))
            pos = match.end()
        parts.append(_unescape(text[pos:]))
        return "".join(parts).strip()
    except Exception:  # defensive: never raise on untrusted input
        return _unescape(text).strip()


ACCEPTED_EVENT_TYPES = frozenset({"message", "app_mention"})
"""The only inner ``event.type`` values that can carry a question."""

APP_MENTION = "app_mention"
"""The event a channel question arrives as; exempt from the mention-token check."""

FILE_SHARE_SUBTYPE = "file_share"
"""The one ``message`` subtype that is still a person's question (with files)."""

DM_CHANNEL_TYPE = "im"
"""``channel_type`` of a 1:1 DM — the only kind answered without a mention."""

DM_CHANNEL_PREFIX = "D"
"""Prefix of a 1:1 DM channel id, used when the event omits ``channel_type``."""

PRIVATE_CHANNEL_PREFIX = "G"
"""Prefix of a private-channel id, derived as ``group`` when ``channel_type`` is absent."""

_USER_ID = re.compile(r"[UW][A-Z0-9]+")


@dataclass(frozen=True)
class BotIdentity:
    """Who the bot is, from ``auth.test`` at wiring time — never fetched in parse.

    ``bot_user_id`` (``U…``) is the id the bot posts as and is mentioned by;
    ``bot_id`` (``B…``) is its bot-profile id; ``team_id`` is the installing
    workspace, against which shared-channel traffic is rejected.
    """

    bot_user_id: str
    bot_id: str
    team_id: str


def _mapping(value: Any) -> Mapping[str, Any]:
    """``value`` if it is a mapping, else an empty one (untrusted wire shape)."""
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    """``value`` if it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def _ctx_text(ctx: Any, name: str) -> str:
    """One string attribute of the bot context, ``""`` when absent or mistyped."""
    try:
        return _text(getattr(ctx, name, ""))
    except Exception:
        return ""


def _display_name(event: Mapping[str, Any]) -> str:
    """The sender's name from ``user_profile``: display name, else real name, else ``""``."""
    profile = _mapping(event.get("user_profile"))
    for key in ("display_name", "real_name"):
        name = _text(profile.get(key)).strip()
        if name:
            return name
    return ""


def _channel_type(event: Mapping[str, Any], channel: str) -> str:
    """``channel_type`` when Slack sent it, else derived from the channel id."""
    sent = _text(event.get("channel_type"))
    if sent:
        return sent
    if channel.startswith(DM_CHANNEL_PREFIX):
        return DM_CHANNEL_TYPE
    if channel.startswith(PRIVATE_CHANNEL_PREFIX):
        return "group"
    return "channel"


def _mentions_bot(text: str, bot_user_id: str) -> bool:
    """Whether *text* carries the bot's own ``<@BOT>`` or ``<@BOT|label>`` token."""
    if not bot_user_id:
        return False
    return f"<@{bot_user_id}>" in text or f"<@{bot_user_id}|" in text


def _is_foreign(payload: Mapping[str, Any], event: Mapping[str, Any], team_id: str) -> bool:
    """Shared-channel traffic: an ext-shared channel, or a sender from another team."""
    if payload.get("is_ext_shared_channel") is True:
        return True
    if not team_id:
        return False
    return any((team := _text(event.get(key))) and team != team_id for key in ("user_team", "team"))


def _file_refs(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``{id, name, mimetype, size, url}`` refs of the files on the event."""
    files = event.get("files")
    if not isinstance(files, list):
        return []
    refs: list[dict[str, Any]] = []
    for entry in files:
        entry = _mapping(entry)
        file_id = _text(entry.get("id"))
        url = _text(entry.get("url_private_download")) or _text(entry.get("url_private"))
        if not file_id or not url:
            continue
        size = entry.get("size")
        refs.append(
            {
                "id": file_id,
                "name": _text(entry.get("name")) or _text(entry.get("title")) or file_id,
                "mimetype": _text(entry.get("mimetype")),
                "size": size if isinstance(size, int) and not isinstance(size, bool) else 0,
                "url": url,
            }
        )
    return refs


def _parse(raw: Any, ctx: Any) -> InboundEvent | None:
    payload = _mapping(raw)
    event = _mapping(payload.get("event"))
    event_type = _text(event.get("type"))
    if event_type not in ACCEPTED_EVENT_TYPES:
        return None
    if event_type == APP_MENTION and "edited" in event:
        return None
    subtype = event.get("subtype")
    if subtype is not None and subtype != FILE_SHARE_SUBTYPE:
        return None

    bot_user_id = _ctx_text(ctx, "bot_user_id")
    user = _text(event.get("user"))
    # Loop guard: no bot's post is a question, this bot's least of all.
    if event.get("bot_id") or event.get("app_id"):
        return None
    if not user or (bot_user_id and user == bot_user_id):
        return None
    if event.get("hidden"):
        return None
    if _is_foreign(payload, event, _ctx_text(ctx, "team_id")):
        return None

    channel = _text(event.get("channel"))
    ts = _text(event.get("ts"))
    if not channel or not ts:
        # No dedup key: a redelivery could not be recognized.
        return None

    channel_type = _channel_type(event, channel)
    is_dm = channel_type == DM_CHANNEL_TYPE
    raw_text = _text(event.get("text"))
    if not is_dm and event_type != APP_MENTION and not _mentions_bot(raw_text, bot_user_id):
        return None

    display = _display_name(event)
    text = normalize_text(raw_text, bot_user_id, {user: display} if display else {})
    files = _file_refs(event)
    if not text and not files:
        return None

    thread_ts = _text(event.get("thread_ts"))
    team = _text(payload.get("team_id")) or _ctx_text(ctx, "team_id")
    return InboundEvent(
        message_id=f"{channel}:{ts}",
        text=text,
        sender_id=user,
        sender_display=display or user,
        history_key=channel if is_dm else f"{channel}:{thread_ts or ts}",
        claim_meta={
            SL_CHANNEL: channel,
            SL_TS: ts,
            SL_THREAD_TS: thread_ts if is_dm else (thread_ts or ts),
            SL_CHANNEL_TYPE: channel_type,
            SL_TEAM: team,
            SL_FILES: files,
            SL_TEXT_HEAD: text.split("\n", 1)[0].strip(),
        },
        # The payload itself, for adapter-internal use only; never persisted.
        raw=raw,
    )


def parse_event(raw: Any, ctx: Any) -> InboundEvent | None:
    """Parse one Events API ``event_callback`` payload, or ignore it.

    Free of I/O and never raising: it runs before the dedup claim, on every
    redelivery, over untrusted input.

    Ignored (``None``) for: an inner event type other than ``message`` /
    ``app_mention``; an ``app_mention`` carrying ``edited``; any ``subtype``
    except ``file_share``; a post with ``bot_id``/``app_id`` or by the bot's own
    user; a ``hidden`` event; an ext-shared channel or a ``user_team``/``team``
    other than the installing workspace; outside a 1:1 DM, a ``message`` without
    the bot's mention token; no channel or ``ts``; and nothing left — no text after
    normalization and no files.

    Args:
        raw: The ``event_callback`` payload (with ``event``, ``team_id``,
            ``is_ext_shared_channel``), exactly as the inbox stored it.
        ctx: The bot identity — any object with ``bot_user_id``, ``bot_id`` and
            ``team_id`` string attributes, normally a :class:`BotIdentity`.

    Returns:
        The parsed event, or ``None`` to ignore this one.
    """
    try:
        return _parse(raw, ctx)
    except Exception:  # defensive: never raise on untrusted input
        return None


def people_seen(raw: Any) -> tuple[str, dict[str, str]]:
    """The channel and ``{user id: name}`` for the person an event shows.

    Only the sender, named from its ``user_profile`` (display name, else real
    name), is recorded — and only when it is a person (``U…``/``W…``) and the
    post is not a bot's. Free of I/O and never raising: a payload this cannot
    read gives ``("", {})``.
    """
    try:
        event = _mapping(_mapping(raw).get("event"))
        channel = _text(event.get("channel"))
        if not channel:
            return "", {}
        user = _text(event.get("user"))
        name = _display_name(event)
        if event.get("bot_id") or not name or not _USER_ID.fullmatch(user):
            return channel, {}
        return channel, {user: name}
    except Exception:
        return "", {}
