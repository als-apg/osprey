"""The ``ChannelOps`` seam: every platform side effect the bridge engine invokes.

The engine (pipeline / retry_queue / reconcile / drain wiring) owns the safety-critical
*ordering* — **claim -> ack -> reply context -> dispatch** (the ack is the FIRST thing after
a won claim, so no channel I/O ever delays the user's "on it" signal), send before terminal
mark, history only after a successful post — while every platform I/O behind those steps is
injected through :class:`ChannelOps`. The design generalizes the two proven channels (Google Chat and email):
everything the two share verbatim lives in the engine; everywhere they diverge is a protocol
member.

Destination addressing is deliberately NOT modeled here. Chat posts into ``(space, thread)``,
email replies to ``(sender, in_reply_to, references)``, Nextcloud Talk posts into a room
token — so the engine never sees an address. The adapter stamps whatever it needs into
:attr:`InboundEvent.claim_meta`, the engine persists that meta verbatim on the dedup claim,
and every outbound member receives the persisted ``entry`` back and reads its own fields out
of it. That round-trip is also what makes the retry drain and startup reconcile work: both
operate *only* on persisted entries, long after the wire event is gone.

Failure contracts are part of the seam (the engine's crash-safety ordering depends on them):

============================  =========================================================
member                        contract on failure
============================  =========================================================
``parse_event``               return ``None`` (drop/ignore); never raise on bad input
``resolve_reply_context``     return ``None``; reply context is enrichment, never fatal
``post_ack``                  best-effort: swallow internally, never raise
``download_inputs``           per-file failures become ``skipped`` notes; never raise
``post_answer``               MUST raise — the engine turns the raise into "stay queued
                              and re-deliver next cycle" (drain) or ``post_failed``
                              (reconcile); a swallowed failure would silently drop the
                              answer
``deliver_files``             best-effort: degrade to text-only, never raise — the
                              answer text has already landed via ``post_answer`` and
                              must not be un-delivered by a failed file upload
``post_queued``               may raise — the engine parks the entry BEFORE posting the
                              notice, so a lost notice never loses the request
``post_giveup``               MUST raise — the engine marks the entry terminal only
                              after the notice lands; a lost give-up notice is silence,
                              worse than the rare duplicate a retry could produce
``post_superseded``           best-effort: the coalesce CAS is already committed, so a
                              failed note must never disturb the pass
``coalesce_key``              pure function of the entry; never raise
============================  =========================================================

This module is import-isolated on purpose: no channel imports, no osprey dispatch
internals, no agent SDK, not even sibling ``osprey.bridges.core`` modules — an adapter can
type-check against the seam without dragging in the engine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "RESERVED_ENTRY_KEYS",
    "ChannelOps",
    "InboundEvent",
    "InputDownload",
    "ReplyContext",
]

# Entry keys the ENGINE writes on the persisted dedup entry. The adapter-owned
# namespaces — :attr:`InboundEvent.claim_meta` and :attr:`ReplyContext.meta` — land in
# the SAME flat entry dict, so an adapter key colliding with any of these is a silent
# overwrite in one direction or the other (e.g. a stamped ``reply_to`` clobbered by the
# engine's persisted reply context; a stamped ``attempts`` corrupting the drain's
# give-up ceilings). The pipeline rejects ``claim_meta`` collisions at claim time and
# strips ``ReplyContext.meta`` collisions (loudly) before persisting.
#
# Writers, for the record: the pipeline claims ``text`` / ``sender_id`` /
# ``sender_display`` / ``history_key`` and persists ``reply_to``; the dedup store
# stamps ``run_id`` / ``status`` / ``claimed_at`` / ``settled_at``; the retry queue's
# park stamps ``failure_class`` / ``num_tool_calls`` / ``queued_at`` /
# ``first_queued_at`` / ``coalesce_key`` / ``attempts``; the pipeline's re-queue path
# stamps ``queued_at`` / ``first_queued_at``; the drain stamps ``retried`` /
# ``attempts`` / ``notfound_count`` / ``gate_seen_open`` / ``give_up_reason``.
RESERVED_ENTRY_KEYS: frozenset[str] = frozenset(
    {
        "text",
        "sender_id",
        "sender_display",
        "history_key",
        "run_id",
        "status",
        "claimed_at",
        "settled_at",
        "reply_to",
        "queued_at",
        "first_queued_at",
        "attempts",
        "retried",
        "failure_class",
        "num_tool_calls",
        "coalesce_key",
        "notfound_count",
        "gate_seen_open",
        "give_up_reason",
    }
)


@dataclass(frozen=True)
class InboundEvent:
    """A parsed, channel-neutral inbound message — what :meth:`ChannelOps.parse_event`
    distills out of a raw wire event for the engine to claim and dispatch.

    Only the fields the engine itself consumes are first-class; everything
    channel-specific rides in :attr:`claim_meta`. Adapters may subclass (as a dataclass)
    to carry extra parse state privately, or stash the wire event in :attr:`raw`.
    """

    message_id: str
    """Channel-supplied opaque dedup key, unique per deliverable message. The engine
    treats it as a black box (claim key, log token) — it is the ADAPTER's job to scope
    it when the platform does not guarantee globally unique message ids (e.g. a
    Nextcloud Talk adapter keys ``f"{room_token}:{message_id}"``; email synthesizes a
    hash when ``Message-ID`` is absent)."""

    text: str
    """The question: mention-stripped, quote-stripped message text. The engine claims
    it, ships it to the dispatcher, and re-dispatches from the persisted copy after a
    crash — so it must be complete on its own, without the wire event."""

    sender_id: str
    """Stable machine identity of the sender (``users/123``, an email address, a Talk
    actor id). Persisted for provenance and used in channel coalesce keys."""

    sender_display: str
    """Human display name of the sender, for logs and reply context."""

    history_key: str
    """The adapter-computed per-conversation key the transcript store is scoped by.
    First-class because it is NOT reconstructible from the other fields: it encodes
    channel policy (a Chat DM keys on space while a room keys on thread; email keys on
    sender; Talk keys on room token), and the retry drain needs it to append history
    when a queued run finally completes — long after ``is_dm``-style wire context is
    gone. Empty string means "no conversation history for this message"."""

    claim_meta: Mapping[str, Any] = field(default_factory=dict)
    """Channel-specific fields the engine persists VERBATIM on the dedup claim and
    hands back (inside ``entry``) to every outbound member: destination/threading
    (``space``/``thread``, ``in_reply_to``/``references``, a room token) and inbound
    attachment refs for :meth:`ChannelOps.download_inputs` to re-fetch on a
    re-dispatch. Must be JSON-plain (it round-trips through the on-disk store).

    This meta and :attr:`ReplyContext.meta` land in the SAME flat entry dict as the
    engine's own bookkeeping: keys MUST NOT collide with
    :data:`RESERVED_ENTRY_KEYS` (the pipeline raises at claim time on a
    collision), and the two adapter namespaces must not collide with each other."""

    raw: Any = None
    """The original wire event, for adapter-internal use only (e.g.
    :meth:`ChannelOps.resolve_reply_context` re-reading an inline quote snapshot).
    NEVER persisted or otherwise touched by the engine."""


@dataclass(frozen=True)
class ReplyContext:
    """Resolved reply-to context for a message that quotes/replies to another.

    Returned by :meth:`ChannelOps.resolve_reply_context`; the engine ships
    :attr:`reply_to` to the worker and persists both fields so every re-dispatch path
    replays the same context without re-resolving.
    """

    reply_to: Mapping[str, str]
    """``{"sender": <display name>, "text": <quoted text>}`` — the payload contract the
    worker already consumes. The adapter truncates ``text`` to its own limit."""

    meta: Mapping[str, Any] = field(default_factory=dict)
    """Extra JSON-plain fields persisted into the entry beside ``reply_to`` — e.g. the
    quoted message's attachment refs, so :meth:`ChannelOps.download_inputs` can fetch
    the quoted files on both the live and the re-dispatch path.

    Lands in the SAME flat entry dict as :attr:`InboundEvent.claim_meta` and the
    engine's own bookkeeping: keys MUST NOT collide with
    :data:`RESERVED_ENTRY_KEYS` (the pipeline strips such keys, loudly, rather than
    let them corrupt the entry), and must not collide with the adapter's own
    ``claim_meta`` keys."""


@dataclass(frozen=True)
class InputDownload:
    """Inbound files fetched (or declined) for one dispatch, in two provenance buckets.

    File entries are worker-payload-shaped dicts (``filename`` / ``content_b64`` /
    ``mime`` ...); skip entries are ``{"filename": ..., "reason": ...}`` notes for
    anything the adapter could not or would not fetch, shipped to the agent so it can
    relay why. The OWN bucket (``files`` / ``skipped``) holds the message's own
    attachments; the QUOTED bucket (``quoted_files`` / ``quoted_skipped``) holds
    attachments of the quoted/replied-to message a resolved
    :class:`ReplyContext` pointed at. A channel with no quote mechanic (email, Talk
    without ``parent``) simply leaves the quoted bucket empty.

    The ENGINE owns the shared fresh-file budget and applies it across ``files`` then
    ``quoted_files``, in that order (own attachments win a tight budget), routing each
    capability/trim skip into the MATCHING bucket. Own-message provenance is folded to
    top-level payload keys (``input_files`` / ``skipped_attachments``); quoted
    provenance is folded INSIDE ``reply_to`` (``reply_to["attachments"]`` = delivered
    quoted filenames, ``reply_to["skipped_attachments"]`` = quoted skips), so the agent
    knows those files came from the quoted message, not the current one. Quoted files
    are meaningful only when a reply context resolved (there is a ``reply_to`` to fold
    into).

    ``InputDownload()`` (all four empty) is the no-op: a text-only message must leave
    the dispatch payload byte-identical to the no-attachment path.
    """

    files: Sequence[Mapping[str, Any]] = ()
    skipped: Sequence[Mapping[str, Any]] = ()
    quoted_files: Sequence[Mapping[str, Any]] = ()
    quoted_skipped: Sequence[Mapping[str, Any]] = ()


@runtime_checkable
class ChannelOps(Protocol):
    """Platform I/O a channel adapter implements; the engine owns all ordering and
    store state around these calls (see the module docstring for the per-member
    failure contracts — they are load-bearing).

    Every post-claim member takes the persisted ``entry`` dict (the claim's
    ``claim_meta`` plus the engine's own bookkeeping) rather than the wire event, so
    the same implementation serves the live path, the retry drain, and startup
    reconcile identically.

    **Concurrency.** The engine may invoke any member concurrently: the retry drain
    runs as its own daemon thread alongside live ingestion, and a channel may ingest
    from more than one path (e.g. one thread per room). Implementations must be
    thread-safe and must not assume per-dispatch instance affinity — the same
    ``ChannelOps`` object serves every thread. The engine's own per-dispatch state
    (the memoized ``input_files`` capability probe) is scoped to a single dispatch
    and never shared across threads. Members may block (``download_inputs`` and the
    posts do real I/O); the engine never requires them to return promptly.
    """

    def parse_event(self, event: Any) -> InboundEvent | None:
        """Parse one raw wire event into an :class:`InboundEvent`, or ``None`` to
        ignore it (not addressed to the bot, untrusted sender, empty question,
        non-message event). Must be cheap and I/O-free: it runs BEFORE the dedup
        claim, so redeliveries pay this cost repeatedly."""
        ...

    def resolve_reply_context(self, event: InboundEvent) -> ReplyContext | None:
        """Resolve the platform's native reply/quote mechanic for ``event``, or
        ``None`` when it is not a reply (or resolution failed — never fatal). Runs
        once, AFTER the claim, so duplicates never pay for it; adapters whose wire
        events carry the reply inline (Talk's ``parent``) just read ``event.raw``,
        while adapters needing a lookup (Chat's ``messages.get`` fallback) do their
        I/O here."""
        ...

    def post_ack(self, entry: Mapping[str, Any]) -> None:
        """Give the user an immediate "on it" signal in the conversation (an emoji
        reaction, a short message). Best-effort; a channel with no ack idiom (email)
        implements this as a no-op."""
        ...

    def download_inputs(self, entry: Mapping[str, Any]) -> InputDownload:
        """Fetch the inbound files referenced by the persisted ``entry``: its own
        attachment refs into the OWN bucket (``files`` / ``skipped``), any
        quoted-message refs from ``ReplyContext.meta`` into the QUOTED bucket
        (``quoted_files`` / ``quoted_skipped``) — see :class:`InputDownload` for how
        the engine budgets and folds the two. Called at most once per dispatch;
        whether the fetched bytes are actually shipped is decided by the ENGINE's
        single memoized ``input_files`` capability probe — the adapter never probes
        worker capabilities itself. Also the re-dispatch path: refs come from the
        entry, never from a live wire event."""
        ...

    def post_answer(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Deliver the terminal answer TEXT into the conversation: the completed
        answer, or the channel's error wording for a non-completed status (raw
        internal error strings must never leak). Raise on delivery failure — the
        engine relies on the raise for its at-least-once redelivery."""
        ...

    def deliver_files(
        self, entry: Mapping[str, Any], result: Mapping[str, Any]
    ) -> Mapping[str, str]:
        """Deliver the run's OUTPUT artifacts into the conversation (fetch from the
        worker, upload/republish platform-side). Called after :meth:`post_answer`
        succeeded; degrades to text-only on failure, never raises. Returns
        ``{artifact_id: public_url}`` for artifacts republished at a stable URL (may
        be empty) — the engine stamps these onto the history turn's descriptors.

        A returned URL MUST be retrievable by an unauthenticated GET from the
        engine: it is re-fetched later for prior-artifact re-injection, and the
        engine holds no channel credentials. A channel that cannot publish such a
        URL (e.g. a WebDAV store that would 401, short of minting world-readable
        share links) returns ``{}`` for that artifact — re-injection then degrades
        to the worker byte route only.

        A channel whose answer and artifacts ride ONE payload (email: a single MIME
        message carries the text and the attachments) folds the artifacts into
        :meth:`post_answer` and implements this member as a no-op returning ``{}``.
        Such a channel MUST keep the artifact fetch/attach best-effort INSIDE
        ``post_answer``, so ``post_answer`` still raises only on transport failure —
        a lost artifact must never un-deliver the answer (see the failure table)."""
        ...

    def post_queued(self, entry: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        """Tell the user their request is parked and will run once service is
        restored. Posted on the FIRST park only (a re-park is silent); ``result`` is
        the retryable-failure result that triggered the park, for channels that
        surface any of it."""
        ...

    def post_giveup(self, entry: Mapping[str, Any]) -> None:
        """Post the honest "couldn't run this automatically" notice for an abandoned
        entry. Raise on failure — the engine leaves the entry queued and retries the
        notice next cycle. The machine-side reason stays in store meta
        (``give_up_reason``), never in the user-facing wording."""
        ...

    def post_superseded(self, entry: Mapping[str, Any]) -> None:
        """Post the one-line note that a newer question in the same conversation
        superseded this queued one (into the OLD message's own thread). Best-effort;
        never raises."""
        ...

    def coalesce_key(self, entry: Mapping[str, Any]) -> str | Sequence[str]:
        """The channel's grouping key for collapsing redundant queued entries: entries
        sharing a key keep only the newest (Chat groups on
        ``[history_key, sender_id]``; email on the References-thread root). Return
        ``""`` (or ``()``) for "stands alone — never coalesce". Pure function of the
        persisted entry; the engine normalizes and persists it via
        :func:`osprey.bridges.core.retry.coalesce_key` at park time."""
        ...
