"""Google Chat's :class:`~osprey.bridges.core.ChannelOps` implementation.

:class:`GoogleChatOps` is I/O only. Every ordering, dedup, retry, and crash-recovery
decision stays in :mod:`osprey.bridges.core`; what lives here is the Chat-shaped call
sequence and the wording a space actually sees. The **posting** members —
:meth:`~GoogleChatOps.post_ack`, :meth:`~GoogleChatOps.post_answer`,
:meth:`~GoogleChatOps.post_queued`, :meth:`~GoogleChatOps.post_giveup`,
:meth:`~GoogleChatOps.post_superseded` — compose two collaborators:
:class:`~osprey.bridges.google_chat.client.ChatClient` for the REST leg and
:mod:`osprey.bridges.google_chat.gcs` for republishing a run's artifacts as public
objects the ``cardsV2`` widgets can point at.

The **inbound** members hold no wire knowledge of their own.
:meth:`~GoogleChatOps.parse_event` and :meth:`~GoogleChatOps.resolve_reply_context`
delegate to :mod:`osprey.bridges.google_chat.events`, which owns Chat's two wire shapes,
and :meth:`~GoogleChatOps.download_inputs` delegates the media GET to
:mod:`osprey.bridges.google_chat.inbound`. What the members add is the seam's arity —
those module functions take the config and the message reader explicitly, and the engine
supplies neither — and, for the download, the pairing of each bucket's references with
the Drive names that bucket could not fetch.

**Every post-claim member reads the persisted ``entry`` and nothing else.** None of them
touches a live wire event, which is what makes one implementation serve the live path,
the retry drain, and startup reconcile identically — on the latter two the wire event is
long gone, and a member that reached for it would work in testing and fail after a
restart. The keys read here are the ``gc_``-prefixed ones the claim stamped
(:data:`~osprey.bridges.google_chat.events.GC_SPACE`,
:data:`~osprey.bridges.google_chat.events.GC_THREAD`), imported rather than re-spelled so
the reader and the writer of a key cannot drift apart.

The failure contracts are **asymmetric and load-bearing** — the engine's crash-safety
ordering is built on them, so inverting one here silently breaks delivery rather than
failing a test:

===================  ==========================================================
member               on failure
===================  ==========================================================
``post_ack``         swallow — a missing "on it" must never cost the user an
                     answer
``post_answer``      **raise** — the raise IS the redelivery signal; swallowing
                     silently drops the answer the run already paid for
``post_queued``      may raise — the engine parks the entry before calling, so
                     a lost notice never loses the request, and the pipeline
                     logs the raise rather than acting on it
``post_giveup``      **raise** — the engine marks the entry terminal only after
                     the notice lands; lost silence is worse than a duplicate
``post_superseded``  swallow — the coalesce CAS already committed
``deliver_files``    cannot fail — it posts nothing, and only reports where
                     ``post_answer`` already published
===================  ==========================================================

Wording is deliberately dull and never leaks machine detail: no exception text, no run
id, no ``give_up_reason``, and in particular no ``result["error"]``. A user in a control
room needs to know whether their question is answered, waiting, or abandoned — the
diagnosis belongs in the bridge log, which is where it stays.

Artifacts
---------
Chat delivers a run's output artifacts as ``cardsV2`` riding the answer itself, so the
publish happens inside :meth:`~GoogleChatOps.post_answer` rather than in a separate
delivery member: a card is part of a message body, and there is no way to attach one to a
message that already posted. The publish is nonetheless best-effort throughout — no
failure anywhere in it may cost the user the text answer, which is the property the whole
member is arranged around.

Each artifact is published on its own publisher call rather than a bucket at a time. Both
publishers preserve input order but silently drop what they could not publish, so a short
result is only attributable when exactly one artifact was offered — and attribution is
what two things need: the ``{artifact_id: public URL}`` map the history layer stamps onto
this turn's descriptors, and the re-offer below. The cost is one storage client per
artifact, which is noise beside the fetch and upload each one already does.

An image the PNG publisher **declines** — the fallback-conversion case, where the
descriptor claims PNG and the worker serves something else — is re-offered to
:func:`~osprey.bridges.google_chat.gcs.publish_documents`, which reads no mime off the
descriptor and so publishes it under the type it really has. It arrives as a document
button instead of a broken image widget, and is never dropped.

Because the publish already happened, :meth:`~GoogleChatOps.deliver_files` has no I/O left
to do — but the engine still asks it where each artifact ended up, to stamp ``public_url``
onto this turn's history descriptors. The map therefore has to survive from the post to
that call, which is what the run-keyed stash is for: written at the end of the publish,
popped by ``deliver_files``, and bounded so a run whose ``deliver_files`` never arrives
cannot accumulate.

Prior artifacts
---------------
Those stamped URLs are what :func:`build_prior_artifact_fetcher` exists to use. A follow-up
question re-injects the bytes of an earlier turn's images, and the engine's own fetcher
knows only the worker byte route — which is the *ephemeral* copy: the worker sweeps a
finished run's files, and a restart clears them, while the GCS object the space already
links to is permanent. The injected fetcher tries the worker first and falls back to the
descriptor's ``public_url``, so a conversation can still look back at a plot the worker no
longer has.
"""

from __future__ import annotations

import logging
import threading
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
    bare_mime,
    fetch_artifact,
    safe_label,
)

from .client import ChatClient, chunk_text
from .config import GoogleChatBridgeConfig
from .events import (
    GC_ATTACHMENTS,
    GC_DRIVE_ATTACHMENTS,
    GC_MESSAGE_NAME,
    GC_QUOTED_ATTACHMENTS,
    GC_QUOTED_DRIVE_ATTACHMENTS,
    GC_SPACE,
    GC_THREAD,
)
from .events import parse_event as parse_chat_event
from .events import resolve_reply_context as resolve_chat_reply_context
from .formatting import markdown_to_chat
from .gcs import IMAGE_CONTENT_TYPE, publish_artifacts, publish_documents
from .inbound import (
    DEFAULT_FILENAME,
    DOWNLOAD_FAILED_REASON,
    DRIVE_SKIP_REASON,
    MediaSession,
    build_media_session,
    download_attachments,
)

logger = logging.getLogger(__name__)

# --- space-facing wording ---------------------------------------------------

ACK_TEXT = "On it 🦅 — running your request, one moment…"
"""Immediate "your question was heard" signal, posted before the long dispatch."""

VERSION_TAG_SUFFIX = "  [{tag}]"
"""Appended to :data:`ACK_TEXT` when a release tag is configured, so every conversation
shows which release answered. An unset tag renders the ack without it."""

EMPTY_ANSWER_TEXT = "(the run produced no text output)"
"""Posted when a completed run carries no answer text. Something must always land, or a
successful-but-silent run is indistinguishable from a broken bridge."""

ERROR_TEXT = "⚠️ The run failed or timed out."
"""Posted for any terminal status that is not ``completed``. The result's own ``error``
string is internal — a stack frame, a provider message, a URL with a token in it — and
never reaches the space."""

QUEUED_TEXT = (
    "⏳ Osprey hit a system problem — your request is queued and will run "
    "automatically once service is restored."
)
"""First-park notice. A re-park is silent (the engine posts this only once). It claims
only what the code guarantees: queued, and retried automatically. No notification is
promised, because nothing is filed to send one."""

GIVEUP_TEXT = "⚠️ We couldn't run this automatically — please re-send your question."
"""Abandonment notice. Honest about the outcome and silent about the reason."""

SUPERSEDED_TEXT = "⏭️ Superseded by a newer question in this conversation — I won't answer it."
"""Posted into the OLD message's own thread, so the note lands on the question it is
about rather than looking like a comment on the newest one."""

DOCUMENT_BUTTON_TEXT = "📄 Open {filename}"
"""Label of one document button. The filename is the publisher's display label, already
sanitized — it names the file, it does not address it."""

DEFAULT_DOCUMENT_LABEL = "file"
"""Button label for a published document that carries no filename at all."""

# --- publisher seams --------------------------------------------------------

ImagePublisher = Callable[[GoogleChatBridgeConfig, str, Sequence[str]], list[str]]
"""How image artifacts become public URLs. Defaults to
:func:`~osprey.bridges.google_chat.gcs.publish_artifacts`; tests inject a fake, which is
what keeps GCS and the worker out of this module's test path entirely."""

DocumentPublisher = Callable[
    [GoogleChatBridgeConfig, str, Sequence[Mapping[str, Any]]], list[dict[str, str]]
]
"""How non-image artifacts become public URLs, as
:func:`~osprey.bridges.google_chat.gcs.publish_documents`'s
``{url, filename, mime}`` results."""

STASH_LIMIT = 32
"""How many runs' published-URL maps :meth:`GoogleChatOps._stash_urls` keeps.

The stash exists to carry one map from :meth:`~GoogleChatOps.post_answer` to the
``deliver_files`` that follows it microseconds later on the same thread, so the live
occupancy is the number of answers being posted at once — two threads' worth. Entries
linger only for a run whose ``deliver_files`` never arrives at all: ``post_answer`` raised
after the artifacts were already published, or the process was interrupted between the
two. The bound turns that from an unbounded leak into a few kilobytes, and 32 leaves so
much headroom over the live occupancy that eviction cannot reach a map that is still on
its way to being read."""

# --- prior-artifact re-fetch ------------------------------------------------

PRIOR_FETCH_TIMEOUT = 30.0
"""Budget for one leg of a prior-artifact re-fetch. Applied per route, so a stalled worker
cannot eat the fallback's budget as well."""

PriorArtifactFetcher = Callable[
    [CoreConfig, str | None, Mapping[str, Any], int], FetchedArtifact | None
]
"""The engine's prior-artifact fetch seam, called
``(cfg, run_id, descriptor, max_bytes)``. See
:attr:`~osprey.bridges.core.pipeline.PipelineDeps.fetch_prior_artifact`."""


def build_prior_artifact_fetcher(
    *,
    worker_http: httpx.Client | None = None,
    public_http: httpx.Client | None = None,
) -> PriorArtifactFetcher:
    """Build the fetcher that recovers one prior artifact's bytes. Never raises.

    The engine's own default reads the worker byte route and stops there, which for this
    channel throws away the better copy: the worker's file is *ephemeral* — swept once the
    run is old, gone after a restart — while the GCS object the space is already linking to
    is permanent. So this one tries both, in the order that keeps the fallback rare:

    1.  the worker, addressed from the descriptor's ``entry_id``, over a client with
        ``trust_env=False``. The worker is an internal service reached directly, and a
        proxy inherited from the deployment's environment must never mount itself in front
        of it — the same reasoning that makes ``False`` the default of
        :attr:`~osprey.bridges.core.CoreConfig.trust_env`, applied here unconditionally
        because this leg's destination is internal by construction;
    2.  the descriptor's ``public_url`` — the object this bridge published in
        :meth:`GoogleChatOps.post_answer` and the engine stamped into history — over a
        **proxy-aware** client, ``trust_env=cfg.trust_env``. This leg leaves the
        deployment's network for ``storage.googleapis.com``, so a site that sits behind an
        egress proxy can reach it only through that proxy, which is exactly the case
        ``trust_env`` is set true for.

    The public leg carries **no ``Authorization`` header**. The worker token authenticates
    the internal route and has no business travelling to a third-party host; the object is
    anonymously readable, which is the property the whole delivery path is built on.

    Args:
        worker_http: Client for the worker leg. ``None`` builds one per call and closes it;
            an injected one is left open for its owner. Injected by the entrypoint's
            wiring, and by tests over an :class:`httpx.MockTransport`.
        public_http: Client for the ``public_url`` leg, same ownership rule.

    Returns:
        A fetcher matching :data:`PriorArtifactFetcher`. An exhausted budget short-circuits
        to ``(None, None)`` before either route is tried. A missing run id or an unusable
        ``entry_id`` skips only the WORKER leg — the ``public_url`` leg still runs and can
        still succeed, which is the point of having two routes. ``(None, None)`` is what
        comes back when every route available to that descriptor has failed: a 404 or
        transport error on both, or a body over budget. An unknown served type is reported
        as ``None`` rather than guessed, which makes the engine skip that artifact and flag
        its descriptor.
    """

    def fetch(
        cfg: CoreConfig, run_id: str | None, descriptor: Mapping[str, Any], max_bytes: int
    ) -> FetchedArtifact | None:
        if max_bytes <= 0:
            return None
        entry_id = descriptor.get("entry_id")
        if run_id and isinstance(entry_id, str) and entry_id:
            fetched = _fetch_from_worker(cfg, run_id, entry_id, max_bytes, worker_http)
            if fetched is not None:
                return fetched
        # Every worker-side failure falls through, not just the 404 this exists for:
        # ``fetch_artifact`` reports a swept artifact, a transport error and an oversize
        # body identically, and the fallback is harmless in each case — it is one GET
        # against a URL this bridge published itself.
        return _fetch_public(cfg, descriptor.get("public_url"), max_bytes, public_http)

    return fetch


def _fetch_from_worker(
    cfg: CoreConfig,
    run_id: str,
    artifact_id: str,
    max_bytes: int,
    http: httpx.Client | None,
) -> FetchedArtifact | None:
    """The worker leg of :func:`build_prior_artifact_fetcher`. Never raises.

    :func:`~osprey.bridges.core.fetch_artifact` already swallows everything it can hit; the
    guard here covers building the client itself, which is a live failure mode of its own.
    """
    owns = http is None
    client: httpx.Client | None = None
    try:
        client = http or httpx.Client(timeout=PRIOR_FETCH_TIMEOUT, trust_env=False)
        return fetch_artifact(
            client, cfg, run_id=run_id, artifact_id=artifact_id, max_bytes=max_bytes
        )
    except Exception:
        logger.warning(
            "worker route unusable for prior artifact %s/%s; trying the published URL",
            run_id,
            artifact_id,
            exc_info=True,
        )
        return None
    finally:
        if owns and client is not None:
            client.close()


def _fetch_public(
    cfg: CoreConfig, url: object, max_bytes: int, http: httpx.Client | None
) -> FetchedArtifact | None:
    """The ``public_url`` leg of :func:`build_prior_artifact_fetcher`. Never raises.

    Args:
        cfg: Supplies ``trust_env`` for the client this may build.
        url: The descriptor's ``public_url``, straight off a JSON round-trip — a descriptor
            that carries none, or carries something that is not a string, simply has no
            fallback.
        max_bytes: What is left of the engine's re-injection budget. A body over it is
            dropped whole rather than truncated, exactly as the worker route drops one.
        http: Client to use, or ``None`` to build and close one here.

    Returns:
        The object as served, or ``None``. Built here rather than by
        :func:`~osprey.bridges.core.fetch_artifact` because this leg reads a public
        object store instead of the worker byte route — the same shape from a
        different source, so the seam sees one type either way. The content type is
        normalized by :func:`~osprey.bridges.core.bare_mime`; an object served with no
        header at all reports ``None``, which the engine treats as "not provably an
        image" and skips.
    """
    if not isinstance(url, str) or not url:
        return None
    owns = http is None
    client: httpx.Client | None = None
    try:
        client = http or httpx.Client(timeout=PRIOR_FETCH_TIMEOUT, trust_env=cfg.trust_env)
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.content
        mime = bare_mime(resp.headers.get("content-type"))
    except Exception as exc:
        # Broad on purpose, for the reason ``fetch_artifact`` documents: the URL comes off
        # a stored descriptor, and a malformed one raises at construction time, before any
        # transport error is possible. A prior image is an enrichment — nothing here may
        # cost the dispatch.
        logger.warning("prior artifact fetch failed for %s: %s", url, exc)
        return None
    finally:
        if owns and client is not None:
            client.close()
    if len(data) > max_bytes:
        logger.warning(
            "prior artifact %s oversize (%d bytes > %d), skipping", url, len(data), max_bytes
        )
        return None
    return FetchedArtifact(data=data, content_type=mime)


def _attempt_order(
    descriptors: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split descriptors into ``(try_image_first, documents_only)``.

    This reads ``delivered_mime`` — the prediction that :func:`fetch_artifact`
    refuses to route on — and that is safe HERE and nowhere else, because of what
    the two lists are used for. Getting this wrong costs an artifact its image
    CARD, never its delivery: an artifact whose image attempt declines is
    re-offered to the document path, which makes no claim about the payload and
    publishes it under whatever it turned out to be. Nothing is dropped on the
    strength of the prediction, which is the whole of osprey #503.

    What it buys is one fetch instead of two. The two paths fetch under different
    budgets (image bytes vs document bytes), so attempting the image path for
    every artifact would fetch each document twice — once to be declined, once to
    be published. The prediction is a good enough guess to avoid that, and is
    allowed to be wrong.

    Args:
        descriptors: Normalized descriptors, in worker order.

    Returns:
        ``(images, documents)``. An entry predicting PNG, and one predicting
        nothing at all (a bare id from an older worker), try the image path
        first; everything else goes straight to documents.
    """
    images: list[Mapping[str, Any]] = []
    documents: list[Mapping[str, Any]] = []
    for descriptor in descriptors:
        predicted = descriptor.get("delivered_mime")
        if predicted is None or predicted == IMAGE_CONTENT_TYPE:
            images.append(descriptor)
        else:
            documents.append(descriptor)
    return images, documents


def _space(entry: Mapping[str, Any]) -> str:
    """The Chat space to post into, or ``""`` when the entry names none."""
    space = entry.get(GC_SPACE)
    return space if isinstance(space, str) else ""


def _thread(entry: Mapping[str, Any]) -> str | None:
    """The thread to reply under, or ``None`` to start a new one.

    A malformed or absent thread degrades to an un-threaded message rather than failing
    the post: an answer in a new thread still reaches the user, while a raise here would
    cost them the answer entirely.
    """
    thread = entry.get(GC_THREAD)
    return thread if isinstance(thread, str) and thread else None


def _refs(entry: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    """The inbound attachment references stored under ``key``.

    Tolerant by design: this runs on a JSON round-trip of whatever was persisted, and a
    malformed store must cost attachments, never the dispatch.

    Args:
        entry: The persisted entry.
        key: :data:`~osprey.bridges.google_chat.events.GC_ATTACHMENTS` or
            :data:`~osprey.bridges.google_chat.events.GC_QUOTED_ATTACHMENTS`.

    Returns:
        The reference mappings, in the order Chat listed them, with anything of the
        wrong shape dropped.
    """
    refs = entry.get(key)
    if not isinstance(refs, list):
        return []
    return [ref for ref in refs if isinstance(ref, Mapping)]


def _drive_names(entry: Mapping[str, Any], key: str) -> list[str]:
    """The Drive-attachment display names stored under ``key``.

    Args:
        entry: The persisted entry.
        key: :data:`~osprey.bridges.google_chat.events.GC_DRIVE_ATTACHMENTS` or
            :data:`~osprey.bridges.google_chat.events.GC_QUOTED_DRIVE_ATTACHMENTS`.

    Returns:
        The names, in the order Chat listed them. A non-string entry is dropped rather
        than reported: a skip note exists to tell the user *which* file was lost, and
        one that cannot name the file says nothing.
    """
    names = entry.get(key)
    if not isinstance(names, list):
        return []
    return [name for name in names if isinstance(name, str)]


def _skip(name: object, reason: str) -> dict[str, str]:
    """One ``{"filename", "reason"}`` note for a file that was not fetched.

    The name is sanitized exactly as
    :func:`~osprey.bridges.google_chat.inbound.download_attachments` sanitizes the ones
    it reports, so the notes this module adds and the notes it collects are
    indistinguishable to the agent that relays them.
    """
    return {
        "filename": safe_label(name if isinstance(name, str) else None, DEFAULT_FILENAME),
        "reason": reason,
    }


def _ack_text(version_tag: str) -> str:
    """The ack body for a deployment tagged ``version_tag`` (empty for untagged)."""
    if not version_tag:
        return ACK_TEXT
    return f"{ACK_TEXT}{VERSION_TAG_SUFFIX.format(tag=version_tag)}"


def _answer_chunks(result: Mapping[str, Any]) -> list[str]:
    """The messages :meth:`GoogleChatOps.post_answer` should post, in order.

    A non-completed status becomes :data:`ERROR_TEXT`. A completed run's text is
    rewritten into Chat's formatting subset **on a local copy** and split at Chat's
    per-message ceiling (:func:`~osprey.bridges.google_chat.client.chunk_text`); the
    caller's ``result["text_output"]`` is never touched, because the same string is
    replayed to the agent as conversation history and the transform is a presentation
    concern that must not follow it there.

    Args:
        result: The terminal result. Only ``status`` and ``text_output`` are read.

    Returns:
        One or more messages, each within Chat's limit. Never empty: a completed run with
        no usable text still gets :data:`EMPTY_ANSWER_TEXT`, so "it worked but said
        nothing" never looks like a dead bridge.
    """
    if result.get("status") != "completed":
        return [ERROR_TEXT]
    text = result.get("text_output")
    if not isinstance(text, str) or not text.strip():
        return [EMPTY_ANSWER_TEXT]
    # The transform can empty a body that was only markup to begin with, so the guard is
    # re-applied to its output rather than only to the input.
    return chunk_text(markdown_to_chat(text)) or [EMPTY_ANSWER_TEXT]


def _image_card(run_id: str, urls: Sequence[str]) -> dict[str, Any]:
    """One ``cardsV2`` card with an image widget per published URL, in order.

    Each widget carries an ``onClick.openLink`` to its own URL: Chat downscales the inline
    preview, so tapping the image opens the full-resolution object instead.
    """
    widgets = [{"image": {"imageUrl": url, "onClick": {"openLink": {"url": url}}}} for url in urls]
    return {"cardId": f"artifacts-{run_id}", "card": {"sections": [{"widgets": widgets}]}}


def _document_card(run_id: str, documents: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """One ``cardsV2`` card with a button per published document, in order."""
    buttons = [
        {
            "text": DOCUMENT_BUTTON_TEXT.format(
                filename=document.get("filename") or DEFAULT_DOCUMENT_LABEL
            ),
            "onClick": {"openLink": {"url": document["url"]}},
        }
        for document in documents
    ]
    return {
        "cardId": f"documents-{run_id}",
        "card": {"sections": [{"widgets": [{"buttonList": {"buttons": buttons}}]}]},
    }


class GoogleChatOps:
    """Google Chat's platform I/O behind the ``ChannelOps`` seam.

    Thread-safe and free of per-dispatch state: the engine shares ONE instance across the
    subscriber thread and the drain thread, and each member derives everything it needs
    from the ``entry`` it is handed. The one piece of state that does span calls — the
    published-URL stash — is guarded by its own lock, popped on read, and bounded at
    :data:`STASH_LIMIT` runs, so it can neither be read twice nor grow without limit.
    """

    def __init__(
        self,
        cfg: GoogleChatBridgeConfig,
        client: ChatClient | None = None,
        publisher: ImagePublisher = publish_artifacts,
        doc_publisher: DocumentPublisher = publish_documents,
        media_session: MediaSession | None = None,
    ) -> None:
        """Wire the adapter to its config, the Chat REST client, and the two publishers.

        Every collaborator is explicit and injectable, and each has a default derived from
        ``cfg``, so a test that exercises one member need not name the others.

        Args:
            cfg: The bridge config. The posting members address spaces from the entry
                rather than from config; ``cfg`` is read for the ack's version tag and
                handed to the publishers, which take the bucket and the worker route off
                it.
            client: Chat client to use instead of one built from ``cfg``. Tests inject a
                client over a mock discovery service, which is what lets this module's
                whole suite run with no Google library installed.
            publisher: How image artifacts are published. See :data:`ImagePublisher`.
            doc_publisher: How documents are published. See :data:`DocumentPublisher`.
            media_session: Session for the inbound attachment downloads. Tests inject an
                ``httpx.Client`` over a ``MockTransport``; production leaves it ``None``
                and one is built from ``cfg.sa_key`` on the first download rather than
                here — see :meth:`_session`.
        """
        self._cfg = cfg
        self._client = client if client is not None else ChatClient(cfg)
        self._publisher = publisher
        self._doc_publisher = doc_publisher
        self._media = media_session
        self._media_lock = threading.Lock()
        self._stash_lock = threading.Lock()
        self._stashed_urls: dict[str, dict[str, str]] = {}

    # --- inbound members ---------------------------------------------------

    def parse_event(self, event: Any) -> InboundEvent | None:
        """Parse one raw Chat event into an :class:`InboundEvent`, or ignore it.

        Free of I/O and never raises, because it runs BEFORE the dedup claim: every
        Pub/Sub redelivery pays for it again, and an exception would leave the message
        unacknowledged and redelivered forever. See
        :func:`~osprey.bridges.google_chat.events.parse_event` for which events count as
        questions for this app — the mention filter outside a direct message is the
        access control for a room.

        Args:
            event: One decoded Chat event, in either wire shape.

        Returns:
            The parsed event, or ``None`` to ignore this one.
        """
        return parse_chat_event(event, self._cfg)

    def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
        """Resolve the message a Chat quote-reply points at, or ``None``. Never fatal.

        Chat's quote arrives inline on the event when it arrives at all, so this is
        mostly parsing — but the two views that may be missing (the current message's
        snapshot, the quoted message's attachments) are reads, and they are made through
        the same client the posts use. It is injected rather than imported by
        :mod:`~osprey.bridges.google_chat.events` so that module stays free of the Chat
        transport, which is what lets the whole reply-context suite run against a plain
        callable.

        Every one of those reads is enrichment: a failure resolves to less context, never
        to a failed answer.
        """
        return resolve_chat_reply_context(event, self._client.get_message)

    def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
        """Fetch the inbound files this exchange referenced. Never raises.

        Reads the references **from the persisted entry and nothing else** — the
        message's own from :data:`~osprey.bridges.google_chat.events.GC_ATTACHMENTS`, the
        quoted message's from
        :data:`~osprey.bridges.google_chat.events.GC_QUOTED_ATTACHMENTS`, which the engine
        persisted flat into the same entry off the resolved :class:`ReplyContext`. That is
        the whole point of the member: by the time the retry drain or a post-restart
        reconcile calls it, the wire event is long gone, so an implementation that reached
        for one would work in testing and silently lose every attachment after a restart.

        The two provenance buckets stay separate end to end. The engine reports own
        attachments at the payload's top level and quoted ones inside ``reply_to``, so a
        user can tell whether the file *they* attached or the one they *quoted* was
        dropped — mixing them would misattribute the loss, and would tell the agent a user
        had just sent a file they only pointed at.

        A Drive-backed attachment is not fetchable with the app's own credential, so each
        name the parse step split off becomes a
        :data:`~osprey.bridges.google_chat.inbound.DRIVE_SKIP_REASON` note in its own
        bucket instead of being dropped: "I could not open that one" is an answer, silence
        is not. Those notes follow that bucket's download skips, because splitting the two
        kinds apart discarded Chat's original interleaving and it cannot be recovered
        here.

        No capability or budget negotiation happens here: every referenced file is
        downloaded, and the engine's single memoized probe decides whether the bytes
        actually ship. Duplicating that decision here would probe the dispatch pair from
        the adapter, which it must never do.

        Args:
            entry: The persisted entry.

        Returns:
            The two buckets of files and skip notes. Exactly ``InputDownload()`` when the
            entry references no file of either kind, which keeps the dispatch payload
            byte-identical to the text-only path.
        """
        own_refs = _refs(entry, GC_ATTACHMENTS)
        quoted_refs = _refs(entry, GC_QUOTED_ATTACHMENTS)
        own_drive = _drive_names(entry, GC_DRIVE_ATTACHMENTS)
        quoted_drive = _drive_names(entry, GC_QUOTED_DRIVE_ATTACHMENTS)
        if not (own_refs or quoted_refs or own_drive or quoted_drive):
            return InputDownload()

        files, skipped = self._download(own_refs)
        quoted_files, quoted_skipped = self._download(quoted_refs)
        return InputDownload(
            files=files,
            skipped=[*skipped, *(_skip(name, DRIVE_SKIP_REASON) for name in own_drive)],
            quoted_files=quoted_files,
            quoted_skipped=[
                *quoted_skipped,
                *(_skip(name, DRIVE_SKIP_REASON) for name in quoted_drive),
            ],
        )

    def _download(
        self, refs: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Download one bucket's fetchable references. Never raises.

        Args:
            refs: That bucket's reference mappings, straight off the entry.

        Returns:
            ``(files, skipped)`` as
            :func:`~osprey.bridges.google_chat.inbound.download_attachments` builds them —
            every reference lands in exactly one of the two.

        A session that cannot be built at all (an unreadable key file, ``google-auth`` not
        installed) turns the whole bucket into "download failed" notes rather than an
        empty result: a misconfigured deployment must tell the user their files were lost,
        not answer as though they had sent none.
        """
        if not refs:
            return [], []
        try:
            session = self._session()
        except Exception:
            logger.warning(
                "no media session; reporting %d inbound attachment(s) as failed",
                len(refs),
                exc_info=True,
            )
            return [], [_skip(ref.get("content_name"), DOWNLOAD_FAILED_REASON) for ref in refs]
        return download_attachments(session, refs)

    def _session(self) -> MediaSession:
        """The media-download session, built from ``cfg.sa_key`` on first use.

        Built here rather than in the constructor because building it reads a key file
        and imports ``google-auth``, and this class is constructed on paths that download
        nothing at all — a text-only conversation, the drain posting a queued answer —
        where that cost and that dependency would be paid for no reason. It also keeps
        the ops layer constructible on a machine with no Google libraries installed,
        which is what the whole test suite relies on.

        Lock-guarded because the engine shares ONE instance across the subscriber thread
        and the drain thread: unguarded, two first downloads could race and build two
        credentialed sessions, one of which is then silently discarded.
        """
        with self._media_lock:
            if self._media is None:
                self._media = build_media_session(self._cfg.sa_key)
            return self._media

    # --- posting members ---------------------------------------------------

    def _require_space(self, entry: Mapping[str, Any]) -> str:
        """The space to post into.

        Raises:
            ValueError: If the entry names no space. Callers that must not raise catch
                this along with everything else; the members that must raise let it
                through, which is correct — an entry with no destination cannot be
                delivered by any later attempt either, and the retry queue's give-up
                ceiling is what eventually stops it.
        """
        space = _space(entry)
        if not space:
            raise ValueError(f"entry carries no {GC_SPACE}: nothing can be posted")
        return space

    def post_ack(self, entry: Mapping[str, Any]) -> None:
        """Post the immediate "on it" reply, threaded. Never raises.

        Best-effort by contract: the engine calls this as the first thing after a won
        claim, before the long dispatch, so a Chat hiccup here must cost the user a
        courtesy message and nothing more.
        """
        try:
            self._client.create_message(
                self._require_space(entry), _thread(entry), _ack_text(self._cfg.version_tag)
            )
        except Exception:
            logger.warning(
                "ack post failed for %s; dispatch continues",
                entry.get(GC_MESSAGE_NAME),
                exc_info=True,
            )

    def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the terminal answer into the space. **Raises** on failure.

        The sequence, and why it is this one:

        1. the answer text is transformed and split into Chat-sized chunks;
        2. the run's artifacts are published **before any chunk is posted**, because the
           cards they produce are part of the final chunk's body and cannot be attached
           to a message that already landed — and doing it up front keeps a slow publish
           out of the middle of a half-posted answer;
        3. the chunks post in order, with the cards riding the LAST one only. A card on a
           standalone message could land even if a text chunk failed, and a redelivery
           would then duplicate just the card; riding the final chunk means it only ever
           posts alongside the text it illustrates.

        A ``cardsV2`` create that fails on the final chunk is retried text-only. We cannot
        tell from here whether the create landed server-side before raising (a timeout
        after the write committed), so that retry risks a rare double-post of the final
        chunk — accepted deliberately, because silently dropping the user's answer is
        worse, and the multi-chunk path is already not crash-atomic.

        Every artifact failure — no bucket, no run id, a publisher returning nothing or
        raising — degrades to a text-only answer. The text always goes out.

        Args:
            entry: The persisted entry; supplies the space and the thread.
            result: The terminal result. ``status``, ``text_output``, ``run_id`` and
                ``artifacts`` are read; ``error`` stays out of the space.

        Raises:
            ValueError: If the entry names no space.
            Exception: Whatever the Chat client raises on a failed post. The engine turns
                the raise into "stay queued and re-deliver next cycle", which may repost
                chunks that already landed — at-least-once delivery, accepted because a
                truncated answer reads as a complete one.
        """
        space = self._require_space(entry)
        thread = _thread(entry)
        chunks = _answer_chunks(result)
        cards = self._publish_cards(entry, result)
        last = len(chunks) - 1
        for index, chunk in enumerate(chunks):
            if index == last and cards:
                try:
                    self._client.create_message(space, thread, chunk, cards=cards)
                except Exception:
                    logger.warning(
                        "cardsV2 create failed for the final chunk of %s; retrying text-only",
                        result.get("run_id"),
                        exc_info=True,
                    )
                    self._client.create_message(space, thread, chunk)
            else:
                self._client.create_message(space, thread, chunk)

    def post_queued(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Post the first-park "your request is queued" notice, threaded.

        May raise: the engine parks the entry BEFORE calling this and logs the raise, so a
        lost notice costs a notification and never the request.

        Args:
            entry: The persisted entry.
            result: The retryable failure that triggered the park. Deliberately unread —
                it carries machine failure detail, and the user's answer to "what now?" is
                the same whatever the cause.
        """
        self._client.create_message(self._require_space(entry), _thread(entry), QUEUED_TEXT)

    def post_giveup(self, entry: Mapping[str, Any]) -> None:
        """Post the honest abandonment notice, threaded. **Raises** on failure.

        The engine marks the entry terminal only after this returns, so a raise leaves it
        queued for another attempt next cycle. A user who is never told their question was
        abandoned waits forever; a duplicate notice is a much smaller harm.

        The machine-side ``give_up_reason`` stays in store meta and never reaches the
        space.
        """
        self._client.create_message(self._require_space(entry), _thread(entry), GIVEUP_TEXT)

    def post_superseded(self, entry: Mapping[str, Any]) -> None:
        """Post the "a newer question replaced this" note, threaded. Never raises.

        The coalesce CAS has already committed by the time the engine calls this, so a
        failure here must not disturb the pass.
        """
        try:
            self._client.create_message(self._require_space(entry), _thread(entry), SUPERSEDED_TEXT)
        except Exception:
            logger.warning(
                "superseded note failed for %s; the entry is superseded regardless",
                entry.get(GC_MESSAGE_NAME),
                exc_info=True,
            )

    # --- artifact publication ----------------------------------------------

    def deliver_files(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, str]:
        """Report where this run's artifacts were republished. Never raises.

        **Posts nothing.** Chat carries a run's artifacts as ``cardsV2`` riding the answer
        itself, so by the time the engine calls this the artifacts have already been
        published and delivered by :meth:`post_answer` — a card cannot be attached to a
        message that has already landed, which is why the work happens there and not here.
        What is left is the bookkeeping half of the contract: the engine stamps the
        returned URLs onto this turn's history descriptors as ``public_url``, and those
        stamps are what let a later question re-fetch the artifact through
        :func:`build_prior_artifact_fetcher` once the worker has swept it.

        The URLs qualify under the contract's unauthenticated-GET rule: they address
        anonymously readable GCS objects, which they must be anyway — Chat itself fetches
        them server-side to render the cards.

        The map is popped, not read: nothing else consumes it, and leaving it behind would
        keep every answered run's URLs alive for the life of the process.

        Args:
            entry: The persisted entry; carries the run id on the drain's re-attached
                delivery, where the result may not.
            result: The terminal result; supplies ``run_id``.

        Returns:
            ``{artifact_id: public URL}`` for what this run published, or ``{}`` — for a
            text-only answer, a run whose publish failed entirely, an unknown run id, or a
            second call for the same run. Every path out of here is a dict lookup, so
            "never raises" is a property of the code rather than of a caught exception.
        """
        # The live path takes the run id from the result; the drain's re-attached
        # delivery has it persisted on the entry as well.
        run_id = result.get("run_id") or entry.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return {}
        return self._pop_urls(run_id)

    def _publish_cards(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Publish the run's artifacts and build the cards for them. Never raises.

        Also stashes the ``{artifact_id: public URL}`` map for the run, which is how the
        engine later learns where each artifact was republished.

        Ordering is fixed: the image card first, the document card second, so an
        image-only answer carries exactly the body it did before documents were
        deliverable at all. Within the document card, artifacts that were documents to
        begin with come first and re-offered images follow — their position among the
        buttons carries no meaning, and appending keeps the descriptor order intact for
        the ones whose bucket was right.

        Args:
            entry: The persisted entry; carries the run id on the drain's re-attached
                delivery, where the result may not.
            result: The terminal result; supplies ``run_id`` and ``artifacts``.

        Returns:
            The cards to attach to the final chunk — empty for a non-completed run, an
            unknown run id, no artifacts, or nothing successfully published.
        """
        if result.get("status") != "completed":
            return []
        # The live path takes the run id from the result; the drain's re-attached
        # delivery has it persisted on the entry as well.
        run_id = result.get("run_id") or entry.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return []
        artifacts = result.get("artifacts")
        descriptors = artifact_descriptors(artifacts if isinstance(artifacts, list) else None)
        if not descriptors:
            return []
        images, documents = _attempt_order(descriptors)

        urls: list[str] = []
        mapping: dict[str, str] = {}
        declined: list[Mapping[str, Any]] = []
        for descriptor in images:
            artifact_id = descriptor.get("artifact_id")
            if not isinstance(artifact_id, str) or not artifact_id:
                continue  # artifact_descriptors already drops these; belt for a future shape
            url = self._publish_image(run_id, artifact_id)
            if url is None:
                # Declined (or failed): re-offered below rather than dropped. See the
                # module docstring — the document path makes no claim about the payload,
                # so it publishes the artifact under whatever it turned out to be.
                declined.append(descriptor)
                continue
            urls.append(url)
            mapping[artifact_id] = url

        published: list[dict[str, str]] = []
        for candidate in [*documents, *declined]:
            document = self._publish_document(run_id, candidate)
            if document is None:
                continue
            published.append(document)
            artifact_id = candidate.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                mapping[artifact_id] = document["url"]

        self._stash_urls(run_id, mapping)
        cards: list[dict[str, Any]] = []
        if urls:
            cards.append(_image_card(run_id, urls))
        if published:
            cards.append(_document_card(run_id, published))
        return cards

    def _publish_image(self, run_id: str, artifact_id: str) -> str | None:
        """Publish one artifact as a public PNG. Never raises.

        Offered alone so the answer is attributable: the publisher returns URLs in input
        order and drops what it could not publish, so exactly one input is the only shape
        in which a short result identifies *which* artifact it was about. Anything but a
        single usable URL is treated as "not deliverable as an image" — which is also why
        a raise is not distinguished from a decline here. The publisher is documented not
        to raise, but the property that the user gets an answer at all is not something to
        rest on another module's promise.

        Args:
            run_id: Run the artifact belongs to.
            artifact_id: Worker artifact id to publish.

        Returns:
            The public URL, or ``None`` — the caller re-offers those on the document path.
        """
        try:
            urls = self._publisher(self._cfg, run_id, [artifact_id])
        except Exception:
            logger.warning(
                "image publisher raised for %s/%s; offering it as a document",
                run_id,
                artifact_id,
                exc_info=True,
            )
            return None
        if len(urls) != 1 or not urls[0]:
            return None
        return urls[0]

    def _publish_document(
        self, run_id: str, descriptor: Mapping[str, Any]
    ) -> dict[str, str] | None:
        """Publish one artifact as a public document. Never raises.

        Offered alone for the same reason as :meth:`_publish_image`: the published result
        carries no artifact id of its own, so the id it belongs to is recoverable only
        from a one-in, one-out call. A count that does not match is not guessed at — an
        artifact carrying no URL is strictly better than one carrying somebody else's.

        Args:
            run_id: Run the artifact belongs to.
            descriptor: The artifact's run-status descriptor, passed through unchanged.
                It may come from either bucket: the publisher reads no mime off it, which
                is what makes a declined image safe to re-offer as-is.

        Returns:
            The publisher's ``{url, filename, mime}`` result, or ``None``.
        """
        try:
            published = self._doc_publisher(self._cfg, run_id, [descriptor])
        except Exception:
            logger.warning(
                "document publisher raised for %s/%s; no document card",
                run_id,
                descriptor.get("artifact_id"),
                exc_info=True,
            )
            return None
        if len(published) != 1:
            return None
        document = published[0]
        url = document.get("url")
        if not isinstance(url, str) or not url:
            return None
        return document

    def _stash_urls(self, run_id: str, urls: Mapping[str, str]) -> None:
        """Record where this run's artifacts were republished, evicting the oldest map
        once :data:`STASH_LIMIT` runs are held.

        The engine asks for the map through :meth:`deliver_files` — a member Chat has
        nothing to do in, since the artifacts already rode the answer's cards — so the map
        has to survive from the post to that call. Nothing is stashed for a run that
        published nothing, which is the common case.

        The bound is what makes the stash safe to hold on an instance the whole process
        shares. A map is normally popped moments after it is written, but the runs that
        never get there — an answer that raised after publishing, an interrupted process —
        would otherwise pile up for the life of the bridge. Evicting the OLDEST is the
        direction that matches how the stash is used: the entry most likely to still be
        wanted is the one just written, and the one least likely is the one that has sat
        unread the longest.

        Re-stashing a run that is already held refreshes its position rather than leaving
        it where it was, so a redelivery cannot leave a run's own map near the eviction
        edge while newer runs push past it.

        Lock-guarded because the subscriber thread and the drain thread post answers
        concurrently through this one instance, and eviction reads and writes the same
        dict.

        Args:
            run_id: The run the answer was posted for.
            urls: ``{artifact_id: public URL}``; an empty map is not stored.
        """
        if not urls:
            return
        with self._stash_lock:
            self._stashed_urls.pop(run_id, None)
            self._stashed_urls[run_id] = dict(urls)
            while len(self._stashed_urls) > STASH_LIMIT:
                # Insertion-ordered: the first key is the oldest write still held.
                oldest = next(iter(self._stashed_urls))
                dropped = self._stashed_urls.pop(oldest)
                logger.warning(
                    "dropping the stashed URLs of %s (%d artifact(s)); its delivery never "
                    "arrived and %d runs are now held",
                    oldest,
                    len(dropped),
                    STASH_LIMIT,
                )

    def _pop_urls(self, run_id: str) -> dict[str, str]:
        """Take this run's published-URL map out of the stash, or ``{}`` if it has none."""
        with self._stash_lock:
            return self._stashed_urls.pop(run_id, {})

    # --- pure members ------------------------------------------------------

    def coalesce_key(self, entry: Mapping[str, Any]) -> Sequence[str]:
        """Group queued entries by conversation and sender.

        When dispatch is down and several questions pile up, the drain keeps only the
        newest entry per key and supersedes the rest. Keying on
        ``[history_key, sender_id]`` collapses one person's repeated asking in one
        conversation — a DM's whole space, a room's thread, since that is what
        ``history_key`` already encodes — while never collapsing two different people's
        questions into one.

        Args:
            entry: The persisted entry. Only the engine's own claim fields are read, so
                this is pure and cannot raise.

        Returns:
            ``[history_key, sender_id]``, or ``()`` — the engine's "stands alone, never
            coalesce" sentinel — when either is missing. Standing alone is the
            fail-closed direction: a wrong group silently discards someone's question,
            while a missed group only costs a duplicate run.
        """
        history_key = entry.get("history_key")
        sender_id = entry.get("sender_id")
        if not isinstance(history_key, str) or not history_key:
            return ()
        if not isinstance(sender_id, str) or not sender_id:
            return ()
        return [history_key, sender_id]


# NOT dead code, and not to be "cleaned up": this is the whole static conformance check
# for the seam. Assigning a GoogleChatOps to a ChannelOps makes mypy verify every member's
# FULL signature — parameter names, types, arity, return type — which a runtime
# ``isinstance`` protocol check cannot do (it only looks for the names). Nothing runs it
# and nothing constructs at import; the value is entirely in type-check time.
def _static_conformance(ops: GoogleChatOps) -> ChannelOps:
    return ops
