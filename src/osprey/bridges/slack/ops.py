"""Slack's :class:`~osprey.bridges.core.ChannelOps` implementation.

:class:`SlackOps` is I/O only. Every ordering, dedup, retry and crash-recovery
decision stays in :mod:`osprey.bridges.core`; what lives here is the Slack-shaped
call sequence and the wording a conversation actually sees. The posting members
compose one collaborator — :class:`~osprey.bridges.slack.client.SlackWebClient`
for the Web API leg — and :meth:`~SlackOps.parse_event` holds no wire knowledge
of its own: it delegates to :mod:`osprey.bridges.slack.events`, which owns the
Events API shape.

**Every post-claim member reads the persisted ``entry`` and nothing else.** None
of them touches the live event, which is what makes one implementation serve the
live path, the retry drain and startup reconcile identically — on the latter two
the event is long gone. The keys read here are the ``sl_`` prefixed ones the
claim stamped (:data:`~osprey.bridges.slack.events.SL_CHANNEL` and its siblings),
imported rather than re-spelled so the reader and the writer of a key cannot
drift apart.

The failure contracts are **asymmetric and load-bearing**;
:mod:`osprey.bridges.core.ports` holds the full table and each member restates its
own. ``post_ack``, ``post_resumed`` and ``post_superseded`` swallow, because a lost
courtesy message must never cost a user their answer; ``post_answer``,
``post_queued`` and ``post_giveup`` raise, because the engine's "leave it queued
and try again" is what stands between a transport hiccup and a question never
answered. A refusal the Web API will never reverse (an archived channel, a bot
removed from it) arrives as :class:`~osprey.bridges.core.errors.UndeliverableError`
and propagates untouched, so the engine settles the entry instead of retrying.

Where a reply goes
------------------
A channel, private channel or group DM question is answered in its thread
(``thread_ts`` or, for a top-level question, its own ``ts`` — the claim already
resolved which into :data:`~osprey.bridges.slack.events.SL_THREAD_TS`). A 1:1 DM
is answered at the top level of the conversation, unless the question was itself
asked in a thread. A top-level DM has no thread to say which message is being
answered, so its ack, answer and notices open with :func:`quote_prefix`.

Who is in the room
------------------
:meth:`~SlackOps.room_people` (the optional
:class:`~osprey.bridges.core.ports.RoomRoster` port) lists a channel, private
channel or group DM through one :class:`~osprey.bridges.slack.roster.ChannelRoster`
the instance owns. A 1:1 DM answers ``None``: its only other member is the bot, so
the question goes out with its asker alone. ``post_answer`` consults the same
roster only when the answer holds a ``<@ID>`` placeholder.

Wording mirrors the Teams bridge and never leaks machine detail: no exception
text, no run id, no ``give_up_reason``, and in particular no ``result["error"]``.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from osprey.bridges.core import (
    MAX_DELIVERED_DOC_BYTES,
    MENTION_PLACEHOLDER_RE,
    ChannelOps,
    InboundEvent,
    InputDownload,
    ReplyContext,
    RoomMember,
    RoomPeople,
    RoomRoster,
    artifact_descriptors,
    ext_for_mime,
    fetch_artifact,
    queued_since,
    safe_label,
    unique_stems,
    upload_name,
)
from osprey.bridges.core.text import chunk_text

from .client import FileTooLargeError, ForeignHostError, NotAFileError, SlackWebClient
from .config import SlackBridgeConfig
from .events import (
    DM_CHANNEL_TYPE,
    SL_CHANNEL,
    SL_CHANNEL_TYPE,
    SL_FILES,
    SL_PARENT_FILES,
    SL_TEXT_HEAD,
    SL_THREAD_TS,
    SL_TS,
    BotIdentity,
    _display_name,
    _file_refs,
    _mapping,
    _text,
    normalize_text,
)
from .events import parse_event as parse_slack_event
from .formatting import markdown_to_mrkdwn, render_mentions
from .roster import ChannelRoster

logger = logging.getLogger(__name__)

# --- conversation-facing wording --------------------------------------------

ACK_TEXT = "On it — working on this now."
"""Immediate "your question was heard" signal, posted before the long dispatch."""

VERSION_TAG_SUFFIX = "  (OSPREY {tag})"
"""Appended to :data:`ACK_TEXT` when a release tag resolved, so every
conversation shows which release answered. An untagged deployment renders the
ack without it rather than with an empty parenthetical."""

EMPTY_ANSWER_TEXT = "The run finished without producing any text."
"""Posted when a completed run carries no answer text. Something must always land,
or a successful-but-silent run is indistinguishable from a broken bridge."""

ERROR_TEXT = (
    "I couldn't complete that request. Please try again, or ask an operator to check the "
    "bridge log."
)
"""Posted for any terminal status that is not ``completed``. The result's own
``error`` string is internal — a stack frame, a provider message, a URL with a
token in it — and never reaches the conversation."""

QUEUED_TEXT = (
    "Queued — the service I need is unavailable right now. I'll answer here once it "
    "runs, so there's no need to re-send this."
)
"""First-park notice. A re-park is silent (the engine posts this only once)."""

RESUMED_TEXT = "Resuming your request queued {since} — service is restored, answer follows."
"""Posted right before the delayed answer of a request the user was told was queued.
``{since}`` is ``at <stamp> (<elapsed> ago)`` from :func:`~osprey.bridges.core.queued_since`,
or ``earlier`` when the entry carries no usable time."""

GIVEUP_TEXT = (
    "I couldn't run this automatically and I've stopped retrying, so this question will "
    "not be answered. Please re-send it later, or ask an operator to look into it."
)
"""Abandonment notice. Honest about the outcome and silent about the reason — the
machine-side ``give_up_reason`` stays in store meta."""

SUPERSEDED_TEXT = "Superseded by a newer question in this conversation — I won't answer this one."
"""Posted on the OLDER queued question, so the note lands on the message it is
about rather than reading as a comment on the newest one."""

SKIPPED_FILES_NOTE = "I couldn't attach: {names}."
"""One-line note naming the files the answer promised and the delivery dropped.

Posted as its own message because the answer has already landed by the time the
files are fetched. It says nothing about *why*; the diagnosis is in the bridge
log."""

NAME_SEPARATOR = ", "
"""How the note joins several names, so it stays one line."""

ANSWER_CHUNK_CHARS = 4000
"""Characters per answer message — Slack's recommended ceiling for one message's
``text``. The hard limit is far higher, but Slack truncates and degrades very long
messages in some clients; several readable messages beat one clipped one."""

QUOTE_LINE_CHARS = 120
"""How much of the question's first line a top-level DM reply quotes back: a
pointer at what is being answered, not a second copy of it."""

QUOTE_ELLIPSIS = "…"
"""Marks a quote that was cut, so a truncated question never reads as a short one."""

MAX_QUOTED_TEXT = 2000
"""Hard bound on the thread parent's text a reply carries in ``reply_to``. The
:data:`QUOTE_ELLIPSIS` of a cut parent is counted inside it, so a caller sizing a
payload around the bound is never surprised."""

# --- inbound files ------------------------------------------------------------
#
# These never reach the conversation: they ride the dispatch payload so the AGENT can
# tell the user which attachment it did not get and why. No URL, token or exception
# text — the diagnosis goes to the log.

MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
"""Per-file download cap. Equal to the engine's fresh-input total budget: a single
file over it could never ship, so it is not fetched. A ref whose declared ``size``
is already over it is skipped before any request."""

DEFAULT_INPUT_FILENAME = "attachment"
"""Stand-in name for an inbound file whose ref carries neither name nor id."""

FOREIGN_HOST_REASON = "couldn't be fetched (not hosted by Slack)"
"""Skip reason for a ref whose URL is not ``https://files.slack.com`` — refused
before any request, so the bot token never leaves for another host."""

NOT_A_FILE_REASON = "couldn't be fetched (Slack answered with a page, not the file)"
"""Skip reason for a download that answered something other than the file's bytes
(Slack's login page, what a missing ``files:read`` scope gets)."""

TOO_LARGE_REASON = f"too large to fetch (over {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB)"
"""Skip reason for a file over :data:`MAX_DOWNLOAD_BYTES`, quoting the same constant
the cap comes from so the two cannot drift."""

DOWNLOAD_FAILED_REASON = "couldn't be fetched from Slack"
"""Skip reason for every other failure: a deleted file, a transport error, a ref
without a URL."""

UPLOAD_FALLBACK_MIME = "application/octet-stream"
"""The upload Content-Type of an artifact the worker served without one."""


def _escape(text: str) -> str:
    """``text`` with mrkdwn's three control characters escaped.

    Applied to text that is not the agent's answer (the quoted question, file
    names): it is shown as typed, so a ``<!here>`` or ``<@U…>`` in it pings nobody.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ack_text(version_tag: str) -> str:
    """The ack body for a deployment tagged ``version_tag``.

    Args:
        version_tag: The resolved release tag from
            :meth:`~osprey.bridges.slack.config.SlackBridgeConfig.from_env`.
            Empty means untagged.

    Returns:
        :data:`ACK_TEXT`, with :data:`VERSION_TAG_SUFFIX` appended only when the
        tag is non-empty.
    """
    if not version_tag:
        return ACK_TEXT
    return f"{ACK_TEXT}{VERSION_TAG_SUFFIX.format(tag=version_tag)}"


def skipped_files_note(names: Sequence[str]) -> str:
    """The note for files that were not delivered.

    Public because the e2e lane asserts the posted text by equality against it.
    The names are escaped, since a file name is not mrkdwn.

    Args:
        names: The dropped files' names, in delivery order.

    Returns:
        :data:`SKIPPED_FILES_NOTE` filled with the names, on one line.
    """
    return SKIPPED_FILES_NOTE.format(names=NAME_SEPARATOR.join(_escape(n) for n in names))


def _is_top_level_dm(entry: Mapping[str, Any]) -> bool:
    """Whether the entry's reply is posted at the top level of a 1:1 DM."""
    thread_ts = entry.get(SL_THREAD_TS)
    return entry.get(SL_CHANNEL_TYPE) == DM_CHANNEL_TYPE and not (
        isinstance(thread_ts, str) and thread_ts
    )


def quote_prefix(entry: Mapping[str, Any]) -> str:
    """The blockquote a top-level DM reply opens with, or ``""``.

    A threaded reply sits under the question itself, so it quotes nothing. A
    top-level DM reply has no thread, so it names what it answers. The line comes
    from :data:`~osprey.bridges.slack.events.SL_TEXT_HEAD`, else the first line of
    the engine's ``text``; it is escaped, so the quote is shown as typed and never
    becomes a mention or a broadcast.

    Pure and total: every degenerate shape (a threaded or non-DM entry, a missing
    or blank question) resolves to ``""``.

    Args:
        entry: The persisted entry.

    Returns:
        ``"> <first line>\\n\\n"``, the line capped at :data:`QUOTE_LINE_CHARS`
        characters and marked with :data:`QUOTE_ELLIPSIS` when cut, or ``""``.
    """
    if not _is_top_level_dm(entry):
        return ""
    head = entry.get(SL_TEXT_HEAD)
    if not isinstance(head, str) or not head.strip():
        head = entry.get("text")
    if not isinstance(head, str):
        return ""
    line = head.strip().split("\n", 1)[0].strip()
    if not line:
        return ""
    if len(line) > QUOTE_LINE_CHARS:
        line = f"{line[:QUOTE_LINE_CHARS]}{QUOTE_ELLIPSIS}"
    return f"> {_escape(line)}\n\n"


def _address(entry: Mapping[str, Any]) -> tuple[str, str | None]:
    """The ``(channel, thread_ts)`` a reply is posted to; ``thread_ts`` ``None``
    means the top level of a 1:1 DM.

    Raising here is what lets each posting member apply its OWN contract to a
    malformed entry — ``post_ack`` swallows it, the members that must raise let it
    through. A non-DM entry with no thread at all is malformed too: posting it at
    the channel's top level would answer in front of everyone, out of context.

    Raises:
        ValueError: If the entry names no channel, or a non-DM entry no thread.
    """
    channel = entry.get(SL_CHANNEL)
    if not isinstance(channel, str) or not channel:
        raise ValueError(f"entry names no {SL_CHANNEL}")
    thread_ts = entry.get(SL_THREAD_TS)
    if isinstance(thread_ts, str) and thread_ts:
        return channel, thread_ts
    if entry.get(SL_CHANNEL_TYPE) == DM_CHANNEL_TYPE:
        return channel, None
    ts = entry.get(SL_TS)
    if isinstance(ts, str) and ts:
        return channel, ts
    raise ValueError(f"non-DM entry names no {SL_THREAD_TS}")


def _truncate(text: str, limit: int = MAX_QUOTED_TEXT) -> str:
    """``text`` cut to at most ``limit`` characters, the ellipsis counted inside."""
    if len(text) <= limit:
        return text
    return text[: max(limit - len(QUOTE_ELLIPSIS), 0)].rstrip() + QUOTE_ELLIPSIS


def _sender_name(message: Mapping[str, Any]) -> str:
    """Who posted a thread parent: profile display name, else real name, else the
    posting bot's name, else the legacy ``username``, else the user id."""
    return (
        _display_name(message)
        or _text(_mapping(message.get("bot_profile")).get("name")).strip()
        or _text(message.get("username")).strip()
        or _text(message.get("user"))
    )


def _refs(entry: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """The file refs persisted under ``key``, anything of the wrong shape dropped:
    this runs on a JSON round-trip of whatever was stored, and a malformed store
    must cost attachments, never the dispatch."""
    refs = entry.get(key)
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, Mapping)]


def _input_filename(ref: Mapping[str, Any]) -> str:
    """The filename one inbound file ships under: its name, else its file id, else
    :data:`DEFAULT_INPUT_FILENAME`. A payload label, never a storage path."""
    fallback = safe_label(_text(ref.get("id")) or None, DEFAULT_INPUT_FILENAME)
    return safe_label(_text(ref.get("name")) or None, fallback)


def _worker_http(cfg: SlackBridgeConfig) -> httpx.Client:
    """A client for the worker's artifact byte route — a different host, auth and
    timeout budget from the Web API's — with proxy handling from config rather than
    the ambient environment. Built per delivery, so the adapter holds no state."""
    return httpx.Client(trust_env=cfg.core.trust_env)


def _answer_chunks(result: Mapping[str, Any]) -> list[str]:
    """The messages :meth:`SlackOps.post_answer` should post, in order.

    A non-completed status becomes :data:`ERROR_TEXT`; ``result["error"]`` is never
    posted. A completed run's text is converted to mrkdwn **on a local copy** — the
    stored ``text_output`` is replayed as history and must stay Markdown — exactly
    once and BEFORE the split, then cut at :data:`ANSWER_CHUNK_CHARS` without ever
    splitting a ``<@ID>`` placeholder, since each chunk's mentions are rendered on
    their own. Never empty.
    """
    if result.get("status") != "completed":
        return [ERROR_TEXT]
    text = result.get("text_output")
    if not isinstance(text, str) or not text.strip():
        return [EMPTY_ANSWER_TEXT]
    return chunk_text(
        markdown_to_mrkdwn(text), ANSWER_CHUNK_CHARS, keep_whole=MENTION_PLACEHOLDER_RE
    ) or [EMPTY_ANSWER_TEXT]


class SlackOps:
    """Slack's platform I/O behind the ``ChannelOps`` seam.

    Thread-safe and free of per-dispatch state: the engine shares ONE instance
    across the handler pool and the retry-drain thread, and each member derives
    everything it needs from the ``entry`` it is handed. The instance holds only
    its injected collaborators and the thread-safe channel roster cache.
    """

    def __init__(
        self,
        cfg: SlackBridgeConfig,
        client: SlackWebClient,
        ctx: BotIdentity,
    ) -> None:
        """Wire the adapter to its config, the Web API client and the bot identity.

        Args:
            cfg: The bridge config; read for the ack's version tag and the
                mentions switch.
            client: The Web API client every post goes through. Tests inject a
                recording stand-in with the same ``post_message`` signature.
            ctx: Who the bot is, from ``auth.test`` at wiring time; handed to
                :func:`~osprey.bridges.slack.events.parse_event`, never fetched
                there.
        """
        self._cfg = cfg
        self._client = client
        self._ctx = ctx
        self._roster = ChannelRoster(client, ctx.bot_user_id)

    # --- inbound members ---------------------------------------------------

    def parse_event(self, event: Any) -> InboundEvent | None:
        """Parse one Events API ``event_callback`` payload, or ignore it.

        Free of I/O and never raises — it runs before the dedup claim, on every
        redelivery. See :func:`~osprey.bridges.slack.events.parse_event`; this
        member contributes only the bot identity.
        """
        return parse_slack_event(event, self._ctx)

    def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
        """The thread parent a reply quotes, or ``None``. Never raises.

        A question is a reply when its wire event carries a ``thread_ts`` other than
        its own ``ts``; a top-level question (or a thread root) makes no call. The
        parent comes from ONE ``conversations.replies`` lookup (``limit=1``; Slack
        lists the parent first) and is the message whose ``ts`` is that
        ``thread_ts``.

        Args:
            event: The parsed event; :attr:`~osprey.bridges.core.InboundEvent.raw`
                still holds the Events API payload.

        Returns:
            ``ReplyContext(reply_to={"sender", "text"}, meta={SL_PARENT_FILES: refs})``
            — the text normalized like a question's and cut to
            :data:`MAX_QUOTED_TEXT`, the refs in the ``sl_files`` shape (an empty
            list when the parent had none). ``None`` when the question is not a
            reply, the lookup failed, the parent is missing, or the parent carries
            neither text nor files.
        """
        try:
            raw = event.raw
            wire = raw.get("event") if isinstance(raw, Mapping) else None
            if not isinstance(wire, Mapping):
                return None
            ts = _text(wire.get("ts"))
            thread_ts = _text(wire.get("thread_ts"))
            if not ts or not thread_ts or thread_ts == ts:
                return None
            channel = _text(event.claim_meta.get(SL_CHANNEL)) or _text(wire.get("channel"))
            if not channel:
                return None
            messages = self._client.conversations_replies(channel, thread_ts, limit=1)
            parent = next(
                (m for m in messages if isinstance(m, Mapping) and m.get("ts") == thread_ts),
                None,
            )
            if parent is None:
                logger.info(
                    "thread parent %s not found in %s; no reply context", thread_ts, channel
                )
                return None
            sender = _sender_name(parent)
            user = _text(parent.get("user"))
            names = {user: sender} if user and sender and sender != user else {}
            text = normalize_text(_text(parent.get("text")), self._ctx.bot_user_id, names)
            refs = _file_refs(parent)
            if not text and not refs:
                return None
            return ReplyContext(
                reply_to={"sender": sender, "text": _truncate(text)},
                meta={SL_PARENT_FILES: refs},
            )
        except Exception:
            logger.warning(
                "reply context lookup failed for %s; dispatching without it",
                event.message_id,
                exc_info=True,
            )
            return None

    def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
        """Fetch the files this exchange referenced. Never raises.

        Reads refs **from the persisted entry only** — the question's own from
        :data:`~osprey.bridges.slack.events.SL_FILES`, the thread parent's from
        :data:`~osprey.bridges.slack.events.SL_PARENT_FILES` — so the retry drain
        and startup reconcile fetch exactly what the live path would. The two
        provenance buckets stay separate: the engine reports them to the agent
        differently. Budgeting across files is the engine's, not done here.

        Args:
            entry: The persisted entry.

        Returns:
            The two buckets of files and skip notes; exactly ``InputDownload()``
            when the entry references no files, which keeps the dispatch payload
            byte-identical to the text-only path.
        """
        own_refs = _refs(entry, SL_FILES)
        quoted_refs = _refs(entry, SL_PARENT_FILES)
        if not own_refs and not quoted_refs:
            return InputDownload()
        files, skipped = self._download_refs(own_refs)
        quoted_files, quoted_skipped = self._download_refs(quoted_refs)
        return InputDownload(
            files=files, skipped=skipped, quoted_files=quoted_files, quoted_skipped=quoted_skipped
        )

    def _download_refs(
        self, refs: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Download one bucket's refs. Never raises.

        Returns:
            ``(files, skipped)``: worker-payload dicts for what was fetched and
            ``{"filename", "reason"}`` notes for what was not. Every ref lands in
            exactly one of the two, in Slack's order.
        """
        files: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for ref in refs:
            filename = _input_filename(ref)
            reason = self._download_one(ref)
            if isinstance(reason, str):
                skipped.append({"filename": filename, "reason": reason})
                continue
            files.append(
                {
                    "filename": filename,
                    "mime": _text(ref.get("mimetype")),
                    "content_b64": base64.b64encode(reason).decode("ascii"),
                    # Fresh input, budgeted against the worker's fresh-file caps.
                    "ingest": True,
                }
            )
        return files, skipped

    def _download_one(self, ref: Mapping[str, Any]) -> bytes | str:
        """One ref's bytes, or the skip reason it is reported under. Never raises."""
        url = _text(ref.get("url"))
        file_id = _text(ref.get("id"))
        if not url:
            logger.warning("inbound file %r has no download url; skipping", file_id)
            return DOWNLOAD_FAILED_REASON
        size = ref.get("size")
        if isinstance(size, int) and not isinstance(size, bool) and size > MAX_DOWNLOAD_BYTES:
            logger.warning("inbound file %r declares %d bytes, over the cap", file_id, size)
            return TOO_LARGE_REASON
        try:
            return self._client.download(url, MAX_DOWNLOAD_BYTES)
        except ForeignHostError:
            logger.warning("inbound file %r is not hosted by Slack; not fetched", file_id)
            return FOREIGN_HOST_REASON
        except FileTooLargeError:  # before NotAFileError: it is a subclass
            logger.warning("inbound file %r is over the download cap", file_id)
            return TOO_LARGE_REASON
        except NotAFileError:
            logger.warning("inbound file %r answered a page, not a file", file_id, exc_info=True)
            return NOT_A_FILE_REASON
        except Exception:
            logger.warning("could not download inbound file %r", file_id, exc_info=True)
            return DOWNLOAD_FAILED_REASON

    # --- posting members ---------------------------------------------------

    def _post_text(self, entry: Mapping[str, Any], text: str) -> None:
        """Post ``text`` where the entry's replies go. Raises whatever the client
        raised; every wrapping member owns its own failure contract."""
        channel, thread_ts = _address(entry)
        self._client.post_message(channel, text, thread_ts=thread_ts)

    def post_ack(self, entry: Mapping[str, Any]) -> None:
        """Post the immediate "on it" reply. Never raises.

        Best-effort by contract: a Web API hiccup, a refusal or a malformed entry
        here must cost the user a courtesy message and nothing more.
        """
        try:
            self._post_text(entry, quote_prefix(entry) + ack_text(self._cfg.version_tag))
        except Exception:
            logger.warning(
                "ack post failed for %s; dispatch continues", entry.get(SL_TS), exc_info=True
            )

    def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the terminal answer into the conversation. **Raises** on failure.

        Converted to mrkdwn and split once (:func:`_answer_chunks`), each chunk's
        mention placeholders rendered, then posted in order, stopping at the first
        failure so the raise reaches the engine — which re-delivers from the FIRST
        chunk (at-least-once, deliberately: a silently dropped tail would read as a
        complete answer). Only the first message carries :func:`quote_prefix`, and
        the quote is never mention-rendered.

        Args:
            entry: The persisted entry; supplies the address and the quote.
            result: The terminal result. Only ``status`` and ``text_output`` are read.

        Raises:
            ValueError: If the entry names no destination.
            UndeliverableError: If Slack refused the post for good.
            SlackApiError: Any other failed post.
        """
        prefix = quote_prefix(entry)
        chunks = _answer_chunks(result)
        roster, enabled = self._mention_roster(entry, chunks)
        for index, chunk in enumerate(chunks):
            lead = prefix if index == 0 else ""
            self._post_text(entry, lead + self._render(chunk, roster, enabled))

    def room_people(self, entry: Mapping[str, Any]) -> RoomPeople | None:
        """Who is in the entry's conversation, or ``None`` when that is unknown.

        The :class:`~osprey.bridges.core.ports.RoomRoster` member. Never raises, and
        lists only this conversation's people. A 1:1 DM, an entry naming no
        channel, and a failed listing all answer ``None`` — the question then goes
        out with its asker only. The asker is named from the entry when they are
        listed without a name, so a re-dispatch after a restart still names who
        asked; a ``sender_display`` that is only the asker's id is not a name. The
        asker is never added when the listing does not contain them.

        Args:
            entry: The persisted entry; supplies the channel, its kind and the asker.

        Returns:
            The listed members (bot excluded), the mentions switch and whether the
            listing was cut short; or ``None``.
        """
        try:
            channel = entry.get(SL_CHANNEL)
            if not isinstance(channel, str) or not channel:
                return None
            if entry.get(SL_CHANNEL_TYPE) == DM_CHANNEL_TYPE:
                return None
            listed = self._roster.members(channel)
            if listed is None:
                return None
            names, more = dict(listed[0]), listed[1]
            sender_id = entry.get("sender_id")
            sender_display = entry.get("sender_display")
            if (
                isinstance(sender_id, str)
                and sender_id in names
                and names[sender_id] is None
                and isinstance(sender_display, str)
                and sender_display.strip()
                and sender_display != sender_id
            ):
                names[sender_id] = sender_display.strip()
            return RoomPeople(
                members=tuple(RoomMember(i, n) for i, n in names.items()),
                mentions=self._cfg.mentions,
                more_not_listed=more,
            )
        except Exception:
            logger.warning(
                "room roster failed; the question goes out without the room", exc_info=True
            )
            return None

    def _mention_roster(
        self, entry: Mapping[str, Any], chunks: Sequence[str]
    ) -> tuple[Mapping[str, str | None], bool]:
        """The roster this answer's mentions render against, and whether to render
        live.

        ``({}, False)`` when no chunk holds a placeholder, so an answer with no
        mention costs no members call; a roster failure also answers ``({}, False)``
        and every ``<@ID>`` posts as plain ``@ID``. Any conversation kind is looked
        up, a 1:1 DM included, since its listing names the asker.
        """
        try:
            if not any(MENTION_PLACEHOLDER_RE.search(chunk) for chunk in chunks):
                return {}, False
            channel = entry.get(SL_CHANNEL)
            if not isinstance(channel, str) or not channel:
                return {}, False
            listed = self._roster.members(channel)
            return (listed[0] if listed else {}), self._cfg.mentions and listed is not None
        except Exception:
            logger.warning("mention roster failed; posting plain text", exc_info=True)
            return {}, False

    @staticmethod
    def _render(chunk: str, roster: Mapping[str, str | None], enabled: bool) -> str:
        """``chunk`` with its placeholders rendered; a render failure posts each as
        plain ``@ID`` so it never costs the answer."""
        try:
            return render_mentions(chunk, roster, enabled=enabled)
        except Exception:
            logger.warning("mention rendering failed; posting plain text", exc_info=True)
            return MENTION_PLACEHOLDER_RE.sub(lambda m: "@" + m.group(1), chunk)

    def post_queued(
        self,
        entry: Mapping[str, Any],
        result: Mapping[str, Any],  # noqa: ARG002 - channel-ops seam signature; the result carries machine failure detail
    ) -> None:
        """Post the first-park "your question is waiting" notice. **Raises** on
        failure; the engine parks the entry first and logs the raise."""
        self._post_text(entry, quote_prefix(entry) + QUEUED_TEXT)

    def post_resumed(
        self,
        entry: Mapping[str, Any],
        result: Mapping[str, Any],  # noqa: ARG002 - channel-ops seam signature; channels that word the line by outcome read the result
    ) -> None:
        """Post the "resuming your queued request" line. Never raises: the answer
        follows whether or not this line did."""
        since = queued_since(entry)
        text = RESUMED_TEXT.format(since=f"at {since}" if since else "earlier")
        try:
            self._post_text(entry, quote_prefix(entry) + text)
        except Exception:
            logger.warning(
                "resume notice failed for %s; the answer follows regardless",
                entry.get(SL_TS),
                exc_info=True,
            )

    def post_giveup(self, entry: Mapping[str, Any]) -> None:
        """Post the honest abandonment notice. **Raises** on failure, so the engine
        leaves the entry queued for another attempt rather than abandoning it
        silently."""
        self._post_text(entry, quote_prefix(entry) + GIVEUP_TEXT)

    def post_superseded(self, entry: Mapping[str, Any]) -> None:
        """Post the "a newer question replaced this" note. Never raises: the
        coalesce has already committed and the note is a courtesy."""
        try:
            self._post_text(entry, quote_prefix(entry) + SUPERSEDED_TEXT)
        except Exception:
            logger.warning(
                "superseded note failed for %s; the entry is superseded regardless",
                entry.get(SL_TS),
                exc_info=True,
            )

    def deliver_files(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, str]:
        """Upload the run's artifacts where the answer went. Never raises.

        Each artifact is fetched from the worker's byte route
        (:func:`~osprey.bridges.core.fetch_artifact`) and uploaded into the
        answer's placement — its thread, or the top level of a 1:1 DM. Names are
        settled once over every descriptor (:func:`~osprey.bridges.core.unique_stems`)
        before any fetch; the extension comes from the served bytes, since the
        descriptor's ``delivered_mime`` only predicted them. Slack renders images
        natively, so nothing is resized. One failed fetch or upload costs only its
        own file; every file that did not land is named in ONE
        :func:`skipped_files_note`, posted after the loop.

        **Returns ``{}`` unconditionally — including on complete success.** The
        engine treats a returned URL as fetchable by an *unauthenticated* GET (that
        is how it re-injects a prior artifact); every Slack file URL needs the bot
        token, so returning one would break re-injection far from its cause.
        Re-injection falls back to the worker byte route, which always works. Do not
        "fix" this to return the file permalink.

        Args:
            entry: The persisted entry; supplies the placement.
            result: The terminal result; supplies ``run_id`` and ``artifacts``.

        Returns:
            An empty mapping, always.
        """
        try:
            run_id = result.get("run_id") or entry.get("run_id")
            artifacts = result.get("artifacts")
            descriptors = artifact_descriptors(artifacts if isinstance(artifacts, list) else None)
            if not isinstance(run_id, str) or not run_id or not descriptors:
                return {}
            channel, thread_ts = _address(entry)
            stems = unique_stems(descriptors)
            undelivered: list[str] = []
            with _worker_http(self._cfg) as http:
                for descriptor in descriptors:
                    artifact_id = descriptor.get("artifact_id")
                    if not isinstance(artifact_id, str) or artifact_id not in stems:
                        continue  # artifact_descriptors already drops these
                    name = self._deliver_one(
                        http, channel, thread_ts, run_id, descriptor, stems[artifact_id]
                    )
                    if name is not None:
                        undelivered.append(name)
        except Exception:
            logger.warning(
                "file delivery failed for %s; the answer stands", entry.get(SL_TS), exc_info=True
            )
            return {}
        if undelivered:
            try:
                self._post_text(entry, skipped_files_note(undelivered))
            except Exception:
                logger.warning(
                    "skipped-file note failed for %s; the files are gone either way",
                    entry.get(SL_TS),
                    exc_info=True,
                )
        return {}

    def _deliver_one(
        self,
        http: httpx.Client,
        channel: str,
        thread_ts: str | None,
        run_id: str,
        descriptor: Mapping[str, Any],
        stem: str,
    ) -> str | None:
        """Fetch and upload one artifact. Never raises.

        Returns:
            ``None`` when it landed; otherwise the name the skipped-files note
            calls it by.
        """
        artifact_id = descriptor["artifact_id"]
        predicted = descriptor.get("delivered_mime")
        predicted = predicted if isinstance(predicted, str) else None
        try:
            fetched = fetch_artifact(
                http, self._cfg.core, run_id, artifact_id, max_bytes=MAX_DELIVERED_DOC_BYTES
            )
        except Exception:
            # fetch_artifact promises not to raise; belt for a future change.
            logger.warning("artifact fetch failed for %s/%s", run_id, artifact_id, exc_info=True)
            fetched = None
        if fetched is None:
            # The bytes were never seen: name it by what the descriptor predicted.
            return upload_name(
                stem, ".png" if predicted == "image/png" else ext_for_mime(predicted)
            )
        if fetched.is_png:
            name, mime = upload_name(stem, ".png"), "image/png"
        else:
            name = upload_name(stem, ext_for_mime(fetched.content_type or predicted))
            mime = fetched.content_type or UPLOAD_FALLBACK_MIME
        try:
            self._client.upload(channel, thread_ts, name, fetched.data, mime)
        except Exception:
            logger.warning(
                "upload of artifact %s/%s into %s failed",
                run_id,
                artifact_id,
                channel,
                exc_info=True,
            )
            return name
        return None

    # --- pure members ------------------------------------------------------

    def coalesce_key(self, entry: Mapping[str, Any]) -> Sequence[str]:
        """Group queued entries by conversation and sender.

        Returns:
            ``[history_key, sender_id]``, or ``()`` — "stands alone, never
            coalesce" — when either is missing. Standing alone is the fail-closed
            direction: a wrong group silently discards someone's question, a missed
            group only costs a duplicate run.
        """
        history_key = entry.get("history_key")
        sender_id = entry.get("sender_id")
        if not isinstance(history_key, str) or not history_key:
            return ()
        if not isinstance(sender_id, str) or not sender_id:
            return ()
        return [history_key, sender_id]


# NOT dead code, and not to be "cleaned up": this is the whole static conformance
# check for the ChannelOps seam. Assigning a SlackOps to it makes mypy verify every
# member's FULL signature — which a runtime ``isinstance`` protocol check cannot do
# (it only looks for the names). Nothing runs it; the value is at type-check time.
def _static_conformance(ops: SlackOps) -> ChannelOps:
    return ops


# Same, for the optional RoomRoster port.
def _static_room_roster(ops: SlackOps) -> RoomRoster:
    return ops
