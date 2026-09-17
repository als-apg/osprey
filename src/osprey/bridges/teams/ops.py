"""Microsoft Teams' :class:`~osprey.bridges.core.ChannelOps` implementation.

:class:`TeamsOps` is I/O only. Every ordering, dedup, retry and crash-recovery
decision stays in :mod:`osprey.bridges.core`; what lives here is the Teams-shaped
call sequence and the wording a conversation actually sees. The posting members
compose one collaborator — :class:`~osprey.bridges.teams.client.ConnectorClient`
for the Bot Connector leg — and the inbound members hold no wire knowledge of
their own: :meth:`~TeamsOps.parse_event` delegates to
:mod:`osprey.bridges.teams.events`, which owns the activity shape.

**Every post-claim member reads the persisted ``entry`` and nothing else.** None
of them touches a live wire activity, which is what makes one implementation
serve the live path, the retry drain and startup reconcile identically — on the
latter two the activity is long gone, and a member that reached for it would
work in testing and fail after a restart. The keys read here are the ``ms_``
prefixed ones the claim stamped (:data:`~osprey.bridges.teams.events.MS_SERVICE_URL`
and its siblings), imported rather than re-spelled so the reader and the writer
of a key cannot drift apart.

The failure contracts are **asymmetric and load-bearing** — the engine's
crash-safety ordering is built on them, so inverting one here silently breaks
delivery rather than failing a test. :mod:`osprey.bridges.core.ports` holds the
full table, and each member here restates its own. The asymmetry in one line:
``post_ack`` and ``post_superseded`` swallow, because a lost courtesy message
must never cost a user their answer; ``post_answer`` and ``post_giveup`` raise,
because the engine's "leave it queued and try again" is the only thing standing
between a transport hiccup and a question that is never answered at all.

Wording is deliberately dull and never leaks machine detail: no exception text,
no run id, no ``give_up_reason``, and in particular no ``result["error"]``. A
user in a control room needs to know whether their question is answered,
waiting, or abandoned — the diagnosis belongs in the bridge log, which is where
it stays.

Threading and the quote
-----------------------
Teams gives the bridge one addressing mechanism and two meanings for it. Every
post is a reply to the activity the claim was taken for, sent to
``{ms_service_url}/v3/conversations/{ms_conversation_id}/activities/{ms_activity_id}``.
In a **channel** that lands the message inside the question's own thread, so the
reply is visibly attached to what it answers and needs to say nothing about it.
In a **1:1 or group chat** there are no threads at all: the reply lands at the
bottom of a flat transcript, where a bare "On it" is ambiguous the moment two
people are talking. So chat replies open with a markdown blockquote of the
question's first line — enough to identify it, capped and elided so a pasted
wall of text cannot become a second wall of text quoted back.

:func:`ack_text` and :func:`quote_prefix` are public for the same reason: the
e2e lane asserts the posted text by *equality* against them. A test that
re-spelled the ack would prove only that someone typed it twice.
"""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from osprey.bridges.core import (
    ChannelOps,
    CoreConfig,
    FetchedArtifact,
    InboundEvent,
    InputDownload,
    ReplyContext,
    artifact_descriptors,
    fetch_artifact,
    safe_label,
)
from osprey.bridges.core.text import chunk_text

from .client import ConnectorClient, MessageSizeTooBig, TokenSource
from .config import TeamsBridgeConfig
from .events import (
    CHANNEL_CONVERSATION,
    MESSAGE_ACTIVITY_TYPE,
    MS_ACTIVITY_ID,
    MS_CONVERSATION_ID,
    MS_CONVERSATION_TYPE,
    MS_SERVICE_URL,
)
from .events import parse_event as parse_teams_event
from .events import resolve_reply_context as resolve_teams_reply_context
from .formatting import markdown_to_teams

logger = logging.getLogger(__name__)

# --- conversation-facing wording --------------------------------------------

ACK_TEXT = "On it — working on this now."
"""Immediate "your question was heard" signal, posted before the long dispatch."""

VERSION_TAG_SUFFIX = "  (OSPREY {tag})"
"""Appended to :data:`ACK_TEXT` when a release tag resolved, so every
conversation shows which release answered. An untagged deployment — a source
checkout with no installed distribution — renders the ack without it rather than
with an empty parenthetical."""

EMPTY_ANSWER_TEXT = "The run finished without producing any text."
"""Posted when a completed run carries no answer text. Something must always land,
or a successful-but-silent run is indistinguishable from a broken bridge."""

ERROR_TEXT = (
    "I couldn't complete that request. Please try again, or ask an operator to check the "
    "bridge log."
)
"""Posted for any terminal status that is not ``completed``. The result's own
``error`` string is internal — a stack frame, a provider message, a URL with a
token in it — and never reaches the conversation.

It says nothing about what the run did before it stopped: ``error``, ``timeout``
and ``cancelled`` all land here, because what the user can do about any of them
is the same."""

QUEUED_TEXT = (
    "Queued — the service I need is unavailable right now. I'll answer here once it "
    "runs, so there's no need to re-send this."
)
"""First-park notice. A re-park is silent (the engine posts this only once)."""

GIVEUP_TEXT = (
    "I couldn't run this automatically and I've stopped retrying, so this question will "
    "not be answered. Please re-send it later, or ask an operator to look into it."
)
"""Abandonment notice. Honest about the outcome and silent about the reason — the
machine-side ``give_up_reason`` stays in store meta."""

SUPERSEDED_TEXT = "Superseded by a newer question in this conversation — I won't answer this one."
"""Posted on the OLDER queued question, so the note lands on the message it is
about rather than reading as a comment on the newest one."""

ANSWER_CHUNK_CHARS = 20_000
"""Characters per answer message, well under the Connector's activity size cap.

Teams refuses an activity larger than 100 KB with a 413; at four bytes per
character a chunk this size is at most 80 KB, which is the ceiling Microsoft
recommends staying beneath. The margin is not slack to be reclaimed — a single
answer can be mostly multi-byte characters, and the cap is measured on the
serialized body, not on the character count. A 413 is still handled rather than
assumed away (:meth:`TeamsOps.post_answer` re-splits at
``ANSWER_CHUNK_CHARS // 2``), because the encoding of any one answer is not
knowable here."""

QUOTE_LINE_CHARS = 120
"""How much of the question's first line a flat-chat reply quotes back.

Long enough that the quote identifies a real question and short enough that it
cannot itself become the message: what is wanted is a pointer at what is being
answered, not a second copy of it pasted above every reply."""

QUOTE_ELLIPSIS = "…"
"""Marks a quote that was cut, so a truncated question never reads as a short
one the bridge misunderstood."""

TEXT_FORMAT_MARKDOWN = "markdown"
"""``textFormat`` every outbound activity declares. Teams renders an activity
without one as plain text in some clients, which would show the quote's ``>``
and the answer's formatting as literal characters."""

# --- artifact fetch seam ----------------------------------------------------

ArtifactFetcher = Callable[[httpx.Client, CoreConfig, str, str], FetchedArtifact | None]
"""How one of a run's output artifacts becomes bytes, called
``(http, core_cfg, run_id, artifact_id)``. Defaults to
:func:`~osprey.bridges.core.fetch_artifact`; tests inject a fake, which is what
keeps the worker out of this module's test path entirely."""


def ack_text(version_tag: str) -> str:
    """The ack body for a deployment tagged ``version_tag``.

    Args:
        version_tag: The resolved release tag, already decided once in
            :meth:`~osprey.bridges.teams.config.TeamsBridgeConfig.from_env`.
            Empty means untagged.

    Returns:
        :data:`ACK_TEXT`, with :data:`VERSION_TAG_SUFFIX` appended only when the
        tag is non-empty.
    """
    if not version_tag:
        return ACK_TEXT
    return f"{ACK_TEXT}{VERSION_TAG_SUFFIX.format(tag=version_tag)}"


def quote_prefix(entry: Mapping[str, Any]) -> str:
    """The blockquote a flat-chat reply opens with, or ``""``.

    A channel reply is threaded under the question itself, so it quotes nothing:
    the quote would be noise directly below the message it repeats. A 1:1 or
    group chat has no threads, so the reply names what it answers.

    Pure and total — it reads only the engine's own ``text`` claim field and the
    adapter's conversation type, and every degenerate shape (a missing type, a
    missing or blank question, an entry written by an older bridge) resolves to
    ``""`` rather than to a blockquote of nothing.

    Args:
        entry: The persisted entry.

    Returns:
        ``"> <first line>\\n\\n"``, the line capped at :data:`QUOTE_LINE_CHARS`
        characters and marked with :data:`QUOTE_ELLIPSIS` when it was cut, or
        ``""`` for a channel and for an entry that carries no question.
    """
    if entry.get(MS_CONVERSATION_TYPE) == CHANNEL_CONVERSATION:
        return ""
    text = entry.get("text")
    if not isinstance(text, str):
        return ""
    line = text.strip().split("\n", 1)[0].strip()
    if not line:
        return ""
    if len(line) > QUOTE_LINE_CHARS:
        line = f"{line[:QUOTE_LINE_CHARS]}{QUOTE_ELLIPSIS}"
    return f"> {line}\n\n"


def _address(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    """The ``(service_url, conversation_id, activity_id)`` a reply is posted to.

    All three are stamped by the claim and none is derivable from the others, so
    a missing one is a malformed entry rather than a degraded one: there is no
    weaker place to post. Raising here is what lets each posting member apply its
    OWN contract to the failure — ``post_ack`` swallows it, the members that must
    raise let it through — instead of this helper deciding for all of them.

    Args:
        entry: The persisted entry.

    Returns:
        The three address segments, exactly as the inbound activity sent them.

    Raises:
        ValueError: If any segment is absent or not a non-empty string.
    """
    segments = []
    for key in (MS_SERVICE_URL, MS_CONVERSATION_ID, MS_ACTIVITY_ID):
        value = entry.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"entry names no {key}")
        segments.append(value)
    return segments[0], segments[1], segments[2]


def _message_activity(text: str) -> dict[str, Any]:
    """One outbound message activity carrying ``text``.

    The Connector is handed this verbatim as the request body, so everything an
    activity needs to be rendered the way the bridge intends is spelled here
    rather than left to a client default.
    """
    return {
        "type": MESSAGE_ACTIVITY_TYPE,
        "textFormat": TEXT_FORMAT_MARKDOWN,
        "text": text,
    }


def _answer_chunks(result: Mapping[str, Any]) -> list[str]:
    """The messages :meth:`TeamsOps.post_answer` should post, in order.

    A non-completed status becomes :data:`ERROR_TEXT`; the result's own ``error``
    string is internal and never posted. A completed run's text is rewritten into
    the subset Teams renders **on a local copy** and split at
    :data:`ANSWER_CHUNK_CHARS` — the caller's ``result["text_output"]`` is never
    touched, because the same string is replayed to the agent as conversation
    history and the rewrite is a presentation concern that must not follow it
    there.

    The rewrite runs exactly once and BEFORE the split, which is the only order
    that is safe: rewriting each chunk instead would let a table or a fenced
    block that the splitter kept whole be converted with half its context
    missing, and the transform can change the text's length, so splitting first
    would not even guarantee the chunks still fit.

    Args:
        result: The terminal result. Only ``status`` and ``text_output`` are read.

    Returns:
        One or more messages, each within :data:`ANSWER_CHUNK_CHARS`. Never
        empty: a completed run with no usable text still gets
        :data:`EMPTY_ANSWER_TEXT`, so "it worked but said nothing" never looks
        like a dead bridge. The guard is re-applied to the transform's output
        because a body that was nothing but markup can come back empty.
    """
    if result.get("status") != "completed":
        return [ERROR_TEXT]
    text = result.get("text_output")
    if not isinstance(text, str) or not text.strip():
        return [EMPTY_ANSWER_TEXT]
    return chunk_text(markdown_to_teams(text), ANSWER_CHUNK_CHARS) or [EMPTY_ANSWER_TEXT]


# --- outbound image delivery ------------------------------------------------
#
# Teams has no file-upload leg in this bridge: an image is delivered by POSTing
# another message activity that carries the bytes inline as a ``data:`` URL. That
# is the whole reason the two bounds below exist and why they are the adapter's
# own rather than the shared ones in :mod:`osprey.bridges.core.artifacts` — those
# bound what a bridge that UPLOADS may send, and an inline attachment is charged
# against the Connector's request body instead.

IMAGE_BOX_PX = 1024
"""Side of the square box every delivered PNG is fitted into.

A plot rendered for a paper arrives several thousand pixels wide; Teams shows it
at a few hundred either way. Fitting it here is what turns a multi-megabyte
artifact into an activity the Connector accepts, and the box is square so the
same number bounds both dimensions whatever the aspect."""

MAX_ATTACHMENT_BYTES = 1024 * 1024
"""Ceiling on one delivered image, measured on the RE-ENCODED bytes.

A base64 ``data:`` URL costs a third more than the bytes it carries, and the
whole activity — text, attachments and all — is one Connector request body. An
image still over this after the fit is dropped rather than posted: an oversize
inline attachment does not render smaller, it renders broken or is refused
outright, and either is worse for the user than being told the image was left
out."""

ATTACHMENT_CONTENT_TYPE = "image/png"
"""``contentType`` of every delivered attachment. PNG is the only type this
bridge delivers, and the re-encode is what makes the claim true regardless of
what the worker served."""

DATA_URL_PREFIX = f"data:{ATTACHMENT_CONTENT_TYPE};base64,"
"""What an inline ``contentUrl`` starts with, composed from the content type so
the two cannot disagree."""

SKIPPED_IMAGES_NOTE = "I couldn't attach: {names}."
"""One-line note naming the images the answer promised and the delivery dropped.

Posted as its own message because by the time the images are fetched the answer
has already landed — :meth:`~TeamsOps.deliver_files` runs after
:meth:`~TeamsOps.post_answer`, and a message that has been posted cannot be
appended to. It says nothing about *why*: oversize and "this deployment has no
Pillow" read identically to a user, who can act on neither, and the diagnosis is
in the bridge log. Saying nothing at all is the option this rules out — an
answer that discusses a plot nobody can see is indistinguishable from a broken
bridge."""

NAME_SEPARATOR = ", "
"""How the note joins several names, so it stays one line."""


def skipped_images_note(names: Sequence[str]) -> str:
    """The note for images that were not delivered.

    Public for the same reason :func:`ack_text` is: the e2e lane asserts the
    posted text by equality against it, and a test that re-spelled the wording
    would prove only that someone typed it twice.

    Args:
        names: The dropped images' names, in delivery order.

    Returns:
        :data:`SKIPPED_IMAGES_NOTE` filled with the names, on one line.
    """
    return SKIPPED_IMAGES_NOTE.format(names=NAME_SEPARATOR.join(names))


def _image_name(descriptor: Mapping[str, Any], artifact_id: str) -> str:
    """The filename one image is attached and, if dropped, named under.

    The worker's ``filename`` hint is user-visible text that lands in a
    serialized activity body, so it goes through
    :func:`~osprey.bridges.core.safe_label` — which strips CR/LF and bounds the
    length — before it is used. An artifact carrying no hint falls back to its
    id, which every descriptor has.
    """
    hint = descriptor.get("filename")
    return safe_label(hint if isinstance(hint, str) else None, f"{artifact_id}.png")


def _fit_png(image_module: Any, data: bytes) -> bytes | None:
    """Re-encode ``data`` as a PNG fitted into :data:`IMAGE_BOX_PX`, or ``None``.

    ``thumbnail`` only ever shrinks, which is the wanted behaviour: an image
    already inside the box is re-encoded unchanged rather than blown up to fill
    it. LANCZOS is the resampling filter because these are plots — a cheaper
    filter aliases thin traces and gridlines into a mess at the moment the
    image is most reduced.

    Args:
        image_module: ``PIL.Image``, passed in because the import lives inside
            :meth:`~TeamsOps.deliver_files` — this module must import with no
            Pillow installed.
        data: The fetched PNG bytes.

    Returns:
        The re-encoded bytes, or ``None`` if the payload could not be decoded or
        re-encoded — a truncated or malformed PNG costs its own delivery and
        nothing else.
    """
    try:
        with image_module.open(io.BytesIO(data)) as image:
            image.load()
            fitted = image.copy()
        fitted.thumbnail((IMAGE_BOX_PX, IMAGE_BOX_PX), image_module.LANCZOS)
        buffer = io.BytesIO()
        fitted.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        logger.warning("image artifact could not be re-encoded; skipping", exc_info=True)
        return None


def _attachment_activity(name: str, data: bytes) -> dict[str, Any]:
    """One message activity carrying ``data`` as an inline PNG attachment.

    Deliberately one attachment per activity: clients disagree about how a
    multi-attachment message renders (a carousel in some, a single image in
    others), and one image per message is the only shape that looks the same
    everywhere. The text is empty because the answer it belongs to has already
    been posted — a caption here would repeat it.
    """
    return {
        **_message_activity(""),
        "attachments": [
            {
                "contentType": ATTACHMENT_CONTENT_TYPE,
                "contentUrl": DATA_URL_PREFIX + base64.b64encode(data).decode("ascii"),
                "name": name,
            }
        ],
    }


class TeamsOps:
    """Teams' platform I/O behind the ``ChannelOps`` seam.

    Thread-safe and free of per-dispatch state: the engine shares ONE instance
    across the Service Bus receive threads and the retry-drain thread, and each
    member derives everything it needs from the ``entry`` it is handed. The
    collaborators it holds are themselves thread-safe — the connector client
    serializes its HTTP leg behind its own lock, and ``httpx.Client`` is safe to
    share.
    """

    def __init__(
        self,
        cfg: TeamsBridgeConfig,
        client: ConnectorClient | None = None,
        worker_http: httpx.Client | None = None,
        artifact_fetcher: ArtifactFetcher = fetch_artifact,
    ) -> None:
        """Wire the adapter to its config, the Connector client and the worker.

        Every collaborator is explicit and injectable, and each has a default
        derived from ``cfg``, so a test that exercises one member need not name
        the others.

        Args:
            cfg: The bridge config. The posting members address conversations
                from the entry rather than from config — the Connector host is
                whatever the inbound activity named — so ``cfg`` is read for the
                bot identity behind the mention filter, the ack's version tag,
                and the worker route the artifact fetch takes.
            client: Connector client to use instead of one built from ``cfg``.
                Tests inject a recording stand-in this way, which is what keeps
                AAD and the Bot Connector out of this module's test path.
            worker_http: Client for the WORKER's artifact byte route — a
                different host, auth scheme and timeout budget from the
                Connector's, so deliberately not the Connector client's.
                ``trust_env`` comes from config for the same reason it does
                there: a proxy inherited from a dev shell must not mount itself
                in front of the worker.
            artifact_fetcher: How an artifact's bytes are fetched. See
                :data:`ArtifactFetcher`.
        """
        self._cfg = cfg
        self._client = client if client is not None else ConnectorClient(cfg, TokenSource(cfg))
        self._http = (
            worker_http if worker_http is not None else httpx.Client(trust_env=cfg.core.trust_env)
        )
        self._fetch_artifact = artifact_fetcher

    # --- inbound members ---------------------------------------------------

    def parse_event(self, event: Any) -> InboundEvent | None:
        """Parse one raw Teams activity into an :class:`InboundEvent`, or ignore it.

        Free of I/O and never raises, because it runs BEFORE the dedup claim:
        every Service Bus redelivery pays for it again, and an exception would
        leave the message unsettled and redelivered forever. See
        :func:`~osprey.bridges.teams.events.parse_event` for which activities
        count as questions for this bot — the mention filter outside a 1:1 chat
        is the access control for a channel.

        Args:
            event: One decoded activity.

        Returns:
            The parsed event, or ``None`` to ignore this one.
        """
        return parse_teams_event(event, self._cfg)

    # --- posting members ---------------------------------------------------

    def _reply(self, entry: Mapping[str, Any], activity: dict[str, Any]) -> None:
        """Post one activity as a reply to the entry's activity.

        The single place an entry becomes a Connector address: every outbound
        post — text or attachment — goes through here, so the address contract
        (:func:`_address`) is applied once and identically. Raises whatever the
        Connector raised — every wrapping member owns its own failure contract,
        and a helper that swallowed here would quietly impose ``post_ack``'s on
        all of them.
        """
        service_url, conversation_id, activity_id = _address(entry)
        self._client.reply(service_url, conversation_id, activity_id, activity)

    def _post_text(self, entry: Mapping[str, Any], text: str) -> None:
        """Post one message activity carrying ``text``; see :meth:`_reply`."""
        self._reply(entry, _message_activity(text))

    def post_ack(self, entry: Mapping[str, Any]) -> None:
        """Post the immediate "on it" reply. Never raises.

        Best-effort by contract: the engine calls this as the first thing after
        a won claim, before the long dispatch, so a Connector hiccup, a refused
        bearer or a malformed entry here must cost the user a courtesy message
        and nothing more.

        In a channel the reply is threaded under the question and is the ack
        wording alone; in a flat chat it opens with :func:`quote_prefix` so the
        user can tell which of their messages it answers.
        """
        try:
            self._post_text(entry, quote_prefix(entry) + ack_text(self._cfg.version_tag))
        except Exception:
            logger.warning(
                "ack post failed for %s; dispatch continues",
                entry.get(MS_ACTIVITY_ID),
                exc_info=True,
            )

    def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the terminal answer into the conversation. **Raises** on failure.

        The answer is rewritten for Teams and split once (:func:`_answer_chunks`),
        then posted in order, stopping at the first failure so the raise reaches
        the engine. The engine re-queues the entry and the drain re-delivers from
        the FIRST chunk, which may repost chunks that already landed: that is
        at-least-once delivery, accepted deliberately, because the alternative —
        silently dropping the tail — makes a truncated answer read as a complete
        one.

        Only the first message carries :func:`quote_prefix`, and only in a flat
        chat. Repeating the quote on every chunk of a long answer would bury the
        answer in its own question; a channel answer is threaded under the
        question already and quotes nothing.

        **The 413 re-split.** The chunk size is a character count and the
        Connector's cap is a byte count, so an answer of mostly multi-byte
        characters can still be refused. A refused chunk is re-split alone at
        half the limit and its halves are posted, then the loop continues with
        the NEXT chunk — the chunks already posted are not reposted, and the ones
        after it are not re-sized on the strength of one over-large neighbour. A
        413 on a half is not caught: two halvings mean the cap is not what this
        path assumes, and the engine's retry is a better answer than an unbounded
        split. The re-split runs on the text that was actually attempted, prefix
        included, so a quote is not lost by being made smaller.

        Args:
            entry: The persisted entry; supplies the address and the quote.
            result: The terminal result. Only ``status`` and ``text_output`` are
                read — ``error``, the run id and the rest stay out of the
                conversation.

        Raises:
            ValueError: If the entry names no destination.
            ConnectorError: Whatever the Connector raised on a failed post,
                including a :class:`~osprey.bridges.teams.client.MessageSizeTooBig`
                on an already-halved chunk.
            TokenError: If no bearer could be obtained for the post.
        """
        prefix = quote_prefix(entry)
        for index, chunk in enumerate(_answer_chunks(result)):
            text = f"{prefix}{chunk}" if index == 0 else chunk
            try:
                self._post_text(entry, text)
            except MessageSizeTooBig:
                logger.warning(
                    "connector refused answer chunk %d for %s as too large; re-splitting it",
                    index,
                    entry.get(MS_ACTIVITY_ID),
                    exc_info=True,
                )
                for half in chunk_text(text, ANSWER_CHUNK_CHARS // 2):
                    self._post_text(entry, half)

    def post_queued(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the first-park "your question is waiting" notice. May raise.

        The engine parks the entry BEFORE calling this and logs the raise, so a
        lost notice costs a notification and never the request.

        Args:
            entry: The persisted entry.
            result: The retryable failure that triggered the park. Deliberately
                unread — it carries machine failure detail, and the user's answer
                to "what now?" is the same whatever the cause.
        """
        self._post_text(entry, quote_prefix(entry) + QUEUED_TEXT)

    def post_giveup(self, entry: Mapping[str, Any]) -> None:
        """Post the honest abandonment notice. **Raises** on failure.

        The engine marks the entry terminal only after this returns, so a raise
        leaves it queued for another attempt next cycle. A user who is never told
        their question was abandoned waits forever; a duplicate notice is a much
        smaller harm.

        The machine-side ``give_up_reason`` stays in store meta and never reaches
        the conversation.
        """
        self._post_text(entry, quote_prefix(entry) + GIVEUP_TEXT)

    def post_superseded(self, entry: Mapping[str, Any]) -> None:
        """Post the "a newer question replaced this" note. Never raises.

        The coalesce CAS has already committed by the time the engine calls this,
        so a failure here must not disturb the pass: the entry is superseded
        either way, and the note is a courtesy.

        In a flat chat the note opens with the superseded question's own quote,
        which is the only thing that says WHICH question was dropped — in a
        channel the reply is threaded under it and needs no quote.
        """
        try:
            self._post_text(entry, quote_prefix(entry) + SUPERSEDED_TEXT)
        except Exception:
            logger.warning(
                "superseded note failed for %s; the entry is superseded regardless",
                entry.get(MS_ACTIVITY_ID),
                exc_info=True,
            )

    # --- inbound file downloads --------------------------------------------

    def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
        """No inbound files: the empty download, always. Never raises.

        A bot reads a Teams message's attachments only with resource-specific
        consent granted by a tenant admin, and this deployment asks for no such
        permission — so the claim stamped no file references and there is nothing
        to fetch. The empty value is returned rather than a partial one so the
        dispatch payload stays byte-identical to the text-only path, and the
        member exists rather than being omitted because the seam is one shape for
        every channel.

        Args:
            entry: The persisted entry. Unused — there are no references in it.

        Returns:
            Exactly ``InputDownload()``: both provenance buckets empty, no skip
            notes. A skip note would be wrong here, because nothing was declined;
            nothing was offered.
        """
        return InputDownload()

    # --- inbound reply context ---------------------------------------------

    def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
        """Resolve the quoted message for ``event``. Never raises.

        Delegates to :func:`~osprey.bridges.teams.events.resolve_reply_context`,
        which owns the answer — Teams delivers no inbound quote to a bot without
        resource-specific consent, so the context is empty for every entry and
        needs no I/O at all. Keeping it there rather than inlining it here is
        what lets that module's tests pin the shape without constructing an
        adapter.
        """
        return resolve_teams_reply_context(event)

    # --- pure members ------------------------------------------------------

    def coalesce_key(self, entry: Mapping[str, Any]) -> Sequence[str]:
        """Group queued entries by conversation and sender.

        When dispatch is down and several questions pile up, the drain keeps only
        the newest entry per key and supersedes the rest. Keying on
        ``[history_key, sender_id]`` collapses one person's repeated asking in one
        conversation — a chat's whole transcript, a channel's thread, since that
        is what ``history_key`` already encodes — while never collapsing two
        different people's questions into one.

        Args:
            entry: The persisted entry. Only the engine's own claim fields are
                read, so this is pure and cannot raise.

        Returns:
            ``[history_key, sender_id]``, or ``()`` — the engine's "stands alone,
            never coalesce" sentinel — when either is missing. Standing alone is
            the fail-closed direction: a wrong group silently discards someone's
            question, while a missed group only costs a duplicate run.
        """
        history_key = entry.get("history_key")
        sender_id = entry.get("sender_id")
        if not isinstance(history_key, str) or not history_key:
            return ()
        if not isinstance(sender_id, str) or not sender_id:
            return ()
        return [history_key, sender_id]

    # --- outbound artifact delivery ----------------------------------------

    def deliver_files(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, str]:
        """Post the run's PNG artifacts into the conversation. Never raises.

        Each artifact is fetched from the worker's byte route, fitted into
        :data:`IMAGE_BOX_PX` and re-encoded, then posted as its own activity
        carrying the bytes inline as a ``data:`` URL — Teams' Bot Connector has
        no upload leg, so an inline attachment is the delivery. Artifacts are
        independent: one that cannot be fetched, decoded or posted costs only
        itself. Images that were dropped are named to the user in one
        :func:`skipped_images_note` message, posted last.

        **Only PNGs are delivered.** Documents are out of scope for v1 — they
        would need a Graph permission and a SharePoint upload — so a non-PNG
        artifact is ignored silently rather than named in the note: the note
        promises images, and an unfetchable artifact whose bytes were never seen
        cannot be claimed to have been one.

        ``Pillow`` is imported HERE rather than at module scope, which is what
        lets the whole adapter import on a deployment that installed the bridge
        without its optional extra. With it missing every image is skipped with
        the same note, because a user can act on neither cause and the
        distinction belongs in the log.

        **Returns ``{}`` unconditionally — including on complete success. This
        is not an oversight.** The engine treats a returned URL as re-fetchable
        by an *unauthenticated* GET, which is how it re-injects a prior artifact
        into a later run. An inline ``data:`` URL is not a URL anything can
        fetch, and the Connector mints no public link, so returning nothing is
        what makes re-injection fall back to the worker byte route — which
        always works. Do not "fix" this to return the data URL.

        Best-effort by contract: the answer text has already landed via
        :meth:`post_answer`, and no delivery failure may un-deliver it.

        Args:
            entry: The persisted entry; supplies the destination address.
            result: The terminal result; supplies ``run_id`` and ``artifacts``.

        Returns:
            An empty mapping, always.
        """
        artifacts = result.get("artifacts")
        descriptors = artifact_descriptors(artifacts if isinstance(artifacts, list) else None)
        if not descriptors:
            return {}
        # The live path takes the run id from the result; the drain's re-attached
        # delivery has it persisted on the entry as well.
        run_id = result.get("run_id") or entry.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            logger.debug("no run id to deliver artifacts for; text only")
            return {}
        try:
            from PIL import Image
        except Exception:
            # Not an error: Pillow is optional, and a bridge without it is a
            # supported deployment that answers in text and says what it dropped.
            logger.info("Pillow is not installed; images will not be delivered")
            image_module = None
        else:
            image_module = Image

        skipped: list[str] = []
        for descriptor in descriptors:
            name = self._deliver_one(entry, run_id, descriptor, image_module)
            if name is not None:
                skipped.append(name)
        if skipped:
            try:
                self._post_text(entry, skipped_images_note(skipped))
            except Exception:
                logger.warning(
                    "skipped-image note failed for %s; the images are gone either way",
                    entry.get(MS_ACTIVITY_ID),
                    exc_info=True,
                )
        return {}

    def _deliver_one(
        self,
        entry: Mapping[str, Any],
        run_id: str,
        descriptor: Mapping[str, Any],
        image_module: Any,
    ) -> str | None:
        """Fetch, fit and post one artifact. Never raises.

        The descriptor only *names* the artifact: its ``delivered_mime`` is a
        prediction made before anything was rendered, so an artifact whose
        conversion failed arrives as its original bytes under a different type.
        Routing on the prediction would drop exactly those artifacts, and
        silently — so the fetched bytes decide the path and the descriptor
        supplies only the name.

        Args:
            entry: The persisted entry; supplies the destination address.
            run_id: Run the artifact belongs to.
            descriptor: Normalized artifact descriptor.
            image_module: ``PIL.Image``, or ``None`` on a deployment without
                Pillow — in which case every image this sees is skipped.

        Returns:
            The name to put in the skip note, or ``None`` when there is nothing
            to tell the user: the image was posted, or the bytes were never
            known to be an image at all.
        """
        artifact_id = descriptor["artifact_id"]
        try:
            fetched = self._fetch_artifact(self._http, self._cfg.core, run_id, artifact_id)
        except Exception:
            # fetch_artifact promises not to raise; the seam takes any callable.
            logger.warning("artifact fetch failed for %s/%s", run_id, artifact_id, exc_info=True)
            return None
        if fetched is None:
            # The fetcher logged why. Nothing is known about what the bytes were,
            # so this is not reported as a dropped image.
            return None
        if not fetched.is_png:
            logger.debug("artifact %s is not a PNG; not delivered to Teams", artifact_id)
            return None
        name = _image_name(descriptor, artifact_id)
        if image_module is None:
            return name
        data = _fit_png(image_module, fetched.data)
        if data is None:
            return name
        if len(data) > MAX_ATTACHMENT_BYTES:
            logger.warning(
                "image artifact %s is %d bytes after the fit, over the inline budget",
                artifact_id,
                len(data),
            )
            return name
        try:
            self._reply(entry, _attachment_activity(name, data))
        except Exception:
            # The answer already landed; an image that could not be posted must
            # not cost it. The note is not attempted either — it would travel the
            # same leg that just failed.
            logger.warning("image artifact %s not delivered; text only", artifact_id, exc_info=True)
        return None


# NOT dead code, and not to be "cleaned up": this is the whole static conformance check
# for the seam. Assigning a TeamsOps to a ChannelOps makes mypy verify every member's
# FULL signature — parameter names, types, arity, return type — which a runtime
# ``isinstance`` protocol check cannot do (it only looks for the names). Nothing runs it
# and nothing constructs at import; the value is entirely in type-check time.
def _static_conformance(ops: TeamsOps) -> ChannelOps:
    return ops
