"""The per-message pipeline: ``handle_event``, the ordering specification itself.

Every dispatch bridge runs the same safety-critical sequence for one inbound message;
this module owns that sequence once, parameterized by the
:class:`~osprey.bridges.core.ports.ChannelOps` seam so no adapter ever re-authors it:

1.  ``ops.parse_event`` — ``None`` means the event is not for us: return ``"ignored"``.
2.  ``dedup.claim`` — atomically claim the message BEFORE anything fires. The claim
    persists the question ``text``, sender provenance, ``history_key`` and the
    adapter's ``claim_meta`` (destination/threading/attachment refs), so a crash or a
    redelivery can never re-dispatch, and every later step — including the retry drain
    and startup reconcile, long after the wire event is gone — works off the persisted
    entry alone. A lost claim returns ``"duplicate"`` without firing anything.
3.  ``ops.post_ack`` — the user's "on it" signal, BEFORE the long dispatch.
4.  ``ops.resolve_reply_context`` — after the claim, so duplicates never pay for it;
    the resolved ``reply_to`` (plus its meta, e.g. quoted attachment refs) is persisted
    so every re-dispatch path replays the same context without re-resolving.
5.  ``history.recent`` — the conversation-so-far rides the payload so follow-ups
    resolve their referents.
6.  ONE memoized ``input_files`` capability probe for the whole dispatch, shared by
    the attachment fold (the own and quoted buckets of ``ops.download_inputs``,
    budgeted own-first, with own provenance top-level and quoted provenance folded
    into ``reply_to``) and the prior-image re-injection. The probe itself
    (:func:`~osprey.bridges.core.capabilities.pair_supports`) is deliberately
    uncached, so the memoization here is strictly per-dispatch: a mid-session
    downgrade is observed on the next dispatch.
7.  ``dispatcher.run`` — with an ``on_run_id`` callback that persists the run id
    (non-terminal ``in_flight``) BEFORE the long worker poll, so reconcile can
    re-attach after a crash mid-poll.
8.  Settle the terminal result off the persisted entry: a retryable failure is parked
    in the retry queue (:func:`osprey.bridges.core.retry_queue.park` — the drain
    replays it later; no history is written for an unfinished exchange); anything else
    is final — ``ops.post_answer`` (which raises on a failed send: the entry is then
    re-queued, run id intact, for the drain to re-attach and re-deliver — never marked
    terminal, never appended to history), then the terminal status mark, then
    best-effort ``ops.deliver_files``, then the history append.

Returns ``"ignored"`` | ``"duplicate"`` | ``"handled"``.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from . import retry, retry_queue
from .artifacts import FetchedArtifact, fetch_artifact
from .capabilities import pair_supports
from .config import CoreConfig
from .dedup import DedupStore
from .dispatch_client import DispatchClient
from .history import HistoryStore
from .ports import RESERVED_ENTRY_KEYS, ChannelOps, InputDownload

logger = logging.getLogger(__name__)

__all__ = [
    "EXPIRED_NOTE",
    "IMAGE_MIMES",
    "MAX_FILES",
    "MAX_TOTAL_BYTES",
    "OVER_BUDGET_REASON",
    "REINJECT_MAX_IMAGES",
    "REINJECT_MAX_TOTAL_BYTES",
    "SERVER_TOO_OLD_REASON",
    "TOO_MANY_FILES_REASON",
    "PipelineDeps",
    "build_extra",
    "handle_event",
]

# --- Fresh-file budget --------------------------------------------------------------
# The worker validates all ``ingest=True`` files of one dispatch against a single count
# cap and the fresh half of its total decoded ceiling; a violation is a fatal,
# non-retryable 400. ``ops.download_inputs`` returns two provenance buckets (own /
# quoted), and the ENGINE enforces the shared budget across ``files`` then
# ``quoted_files``, in that order, so the two sources can never sum past the caps —
# routing each trim's skip note into the matching bucket.
MAX_FILES = 5
MAX_TOTAL_BYTES = 10 * 1024 * 1024

# --- Prior-image re-injection budget ------------------------------------------------
# Shares the worker's total input ceiling with the fresh budget above: 10MiB fresh +
# 8MiB replayed priors. Only images are re-injected.
IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})
REINJECT_MAX_IMAGES = 3
REINJECT_MAX_TOTAL_BYTES = 8 * 1024 * 1024

# Skip reason for a downloaded file the pair cannot ingest (capability probe said no).
# The bytes are never shipped to a worker that would drop them; the reason rides the
# payload so the agent can relay why.
SERVER_TOO_OLD_REASON = "couldn't ingest (server too old)"

# Skip reasons for files the engine's shared fresh budget declined.
TOO_MANY_FILES_REASON = f"too many attachments (limit {MAX_FILES}); this file was not processed"
OVER_BUDGET_REASON = "attachments exceed the total size budget; this file was not processed"

# The note flagged onto a prior-artifact descriptor whose bytes could not be brought
# back (404 / swept / over budget / network). It rides in the outgoing
# ``conversation_so_far``; trigger prompts key on this EXACT phrase to tell the user
# the file is no longer available rather than silently retrying. Never fatal — a
# flagged descriptor is skipped, not dropped.
EXPIRED_NOTE = "may have expired"

# Belt-and-suspenders mirror of ``HistoryStore.MAX_ARTIFACTS_PER_TURN``: the store
# already caps (keeping the tail), but we assemble OLDEST-first so its tail-keep
# retains the NEWEST, and we never hand it an unbounded list.
MAX_TURN_DESCRIPTORS = 10

# Timeout for the default prior-artifact byte fetch against the worker.
_PRIOR_FETCH_TIMEOUT = 30.0


def _default_fetch_prior_artifact(
    cfg: CoreConfig, run_id: str | None, descriptor: Mapping[str, Any], max_bytes: int
) -> FetchedArtifact | None:
    """Fetch one prior artifact via the worker byte route; ``None`` on any failure.

    Returns the served Content-Type alongside the bytes, because the descriptor's
    ``delivered_mime`` is only a prediction of what a fetch would produce. An
    adapter whose channel republishes artifacts at a stable URL (and stamps
    ``public_url`` on the descriptor via ``deliver_files``) injects its own
    fetcher with a fallback for artifacts the worker has since swept."""
    entry_id = descriptor.get("entry_id")
    if not run_id or not isinstance(entry_id, str) or not entry_id or max_bytes <= 0:
        return None
    with httpx.Client(timeout=_PRIOR_FETCH_TIMEOUT, trust_env=cfg.trust_env) as http:
        return fetch_artifact(http, cfg, run_id=run_id, artifact_id=entry_id, max_bytes=max_bytes)


@dataclass
class PipelineDeps:
    """Injected collaborators — makes the per-message handler unit-testable.

    ``park`` is the retry-queue seam, defaulting to the real
    :func:`osprey.bridges.core.retry_queue.park`; it is called
    ``(ops, dedup, message_id, entry, result)`` on a retryable terminal failure.
    ``probe_capability`` is the per-dispatch ``input_files`` probe, called
    ``(cfg, "input_files")``; ``fetch_prior_artifact`` recovers one prior IMAGE
    artifact, called ``(cfg, run_id, descriptor, max_bytes)`` and returning a
    :class:`~osprey.bridges.core.artifacts.FetchedArtifact` (bytes plus the type they
    were served under) or ``None``. All three are injectable so tests run no network
    I/O.
    """

    cfg: CoreConfig
    ops: ChannelOps
    dedup: DedupStore
    dispatcher: DispatchClient
    history: HistoryStore | None = None
    park: Callable[..., Any] = retry_queue.park
    probe_capability: Callable[..., bool] = pair_supports
    fetch_prior_artifact: Callable[..., FetchedArtifact | None] = _default_fetch_prior_artifact


def _run_id_persister(dedup: DedupStore, message_id: str) -> Callable[[str], None]:
    """Build a ``DispatchClient.run(on_run_id=...)`` callback.

    Records the run_id with a NON-terminal status BEFORE the long worker poll, so
    startup reconcile can re-attach if we crash mid-poll. Only ``run_id``/``status``
    are written — the claim persisted the message meta, and
    :meth:`~osprey.bridges.core.dedup.DedupStore.record` preserves it.
    """

    def persist(run_id: str) -> None:
        dedup.record(message_id, run_id=run_id, status="in_flight")

    return persist


def _dispatch(
    dispatcher: DispatchClient,
    dedup: DedupStore,
    message_id: str,
    text: str,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fire one dispatch for ``message_id``, persisting its run id mid-handshake.

    The single dispatch call for every path that fires a question — the live pipeline,
    the drain's retry, and startup reconcile's re-dispatch — so all three get the
    crash-safety persister (:func:`_run_id_persister`) and none can drift. An empty (or
    absent) ``extra`` is sent as no ``extra`` at all, keeping a text-only dispatch's
    payload identical whether or not the engine looked for context to attach.
    """
    persist = _run_id_persister(dedup, message_id)
    if extra:
        return dispatcher.run(text, extra=dict(extra), on_run_id=persist)
    return dispatcher.run(text, on_run_id=persist)


def _capability_probe(deps: PipelineDeps) -> Callable[[], bool]:
    """A memoized ``input_files`` capability probe for ONE dispatch.

    Returns a zero-arg callable that runs :attr:`PipelineDeps.probe_capability` at
    most once and caches the verdict, so the attachment fold and the prior-image
    re-injection — every feature that gates on the pair advertising ``input_files`` —
    share a SINGLE probe per dispatch. The probe fires lazily on first call, so a
    dispatch that ships no bytes never probes at all; and the memo lives only for
    this dispatch, so a mid-session downgrade is observed on the next one.
    """
    verdict: dict[str, bool] = {}

    def probe() -> bool:
        if "v" not in verdict:
            verdict["v"] = bool(deps.probe_capability(deps.cfg, "input_files"))
        return verdict["v"]

    return probe


def build_extra(
    deps: PipelineDeps,
    entry: Mapping[str, Any],
    *,
    history_key: str,
    reply_to: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the dispatch ``extra`` for one message off its persisted ``entry``.

    The single payload assembler: :func:`handle_event` calls it on the live path (with
    the just-resolved ``reply_to``) and :func:`osprey.bridges.core.runtime.rebuild_extra`
    calls it for every RE-dispatch (with the persisted one), so a replayed question can
    never quietly ship less than the live one did. In order: the conversation-so-far, so
    follow-ups ("now plot it over 24h") resolve their referents; the reply context, which
    the quoted-attachment fold then folds provenance into; this message's attachments;
    and the newest prior image artifacts — the last two off ONE memoized capability
    probe, so a dispatch shipping no bytes never probes at all.

    ``history_key`` is passed rather than read off ``entry`` because the live path knows
    it from the parsed event; empty means "no conversation history for this message".
    Returns ``{}`` for a text-only entry with no history, leaving the payload
    byte-identical to the no-attachment path.
    """
    extra: dict[str, Any] = {}

    turns: list[dict[str, Any]] = []
    if deps.history is not None and history_key:
        turns = deps.history.recent(history_key)
        if turns:
            extra["conversation_so_far"] = turns

    # A COPY: the quoted-attachment fold mutates ``reply_to``, and handing out either
    # the store's own nested dict or the adapter's mapping would let per-dispatch keys
    # leak. Set BEFORE the fold, which folds quoted provenance inside it.
    if reply_to:
        extra["reply_to"] = dict(reply_to)

    probe = _capability_probe(deps)
    _fold_inputs(deps.ops.download_inputs(entry), extra, probe)
    _reinject_prior_images(deps, extra, turns, probe)
    return extra


def handle_event(event: Any, deps: PipelineDeps) -> str:
    """Process one raw wire event. Returns a short status string for logging.

    Return values: ``"ignored"`` (not addressed to us), ``"duplicate"``
    (redelivery), or ``"handled"``. See the module docstring for the ordering
    specification this function IS.
    """
    parsed = deps.ops.parse_event(event)
    if parsed is None:
        return "ignored"

    # An adapter claim_meta key colliding with an engine-owned entry field would be a
    # silent overwrite in one direction or the other (a stamped ``reply_to`` clobbered
    # by the persisted reply context; a stamped ``attempts`` corrupting the drain's
    # give-up ceilings). Reject it HERE, before anything is claimed or fired, so the
    # bug fails in the adapter's first test instead of corrupting entries in
    # production.
    collisions = RESERVED_ENTRY_KEYS.intersection(parsed.claim_meta)
    if collisions:
        raise ValueError(
            f"claim_meta for {parsed.message_id} collides with engine-owned entry "
            f"fields {sorted(collisions)}; see osprey.bridges.core.ports.RESERVED_ENTRY_KEYS"
        )

    # Atomically claim the message BEFORE dispatch: a crash/redelivery can't re-fire
    # it, and two concurrent redeliveries can't both pass. The claim persists the
    # question text (so a restart that crashed before a run_id existed can
    # re-dispatch), sender provenance (the one field no other store ever sees),
    # history_key (adapter channel policy, not reconstructible from the meta), and
    # the adapter's claim_meta verbatim (destination/threading/attachment refs).
    if not deps.dedup.claim(
        parsed.message_id,
        text=parsed.text,
        sender_id=parsed.sender_id,
        sender_display=parsed.sender_display,
        history_key=parsed.history_key,
        **dict(parsed.claim_meta),
    ):
        logger.info("duplicate delivery for %s — skipping", parsed.message_id)
        return "duplicate"

    entry = deps.dedup.get(parsed.message_id) or {}
    deps.ops.post_ack(entry)

    reply_to: dict[str, str] | None = None

    # Resolve the channel's native reply/quote mechanic AFTER the claim (duplicates
    # never pay for it). Persist reply_to + its meta (e.g. quoted attachment refs)
    # before dispatch so every re-dispatch path replays the same context without
    # re-resolving; ``extra`` gets its own copy to mutate.
    reply = deps.ops.resolve_reply_context(parsed)
    if reply is not None:
        reply_to = dict(reply.reply_to)
        # ReplyContext.meta lands in the same flat entry namespace as the engine's
        # bookkeeping. The claim already committed, so a collision here cannot fail
        # the message — strip the offending keys LOUDLY instead of letting them
        # corrupt the entry (or clobber the reply_to persisted right below).
        meta = dict(reply.meta)
        meta_collisions = RESERVED_ENTRY_KEYS.intersection(meta)
        if meta_collisions:
            logger.error(
                "ReplyContext.meta for %s collides with engine-owned entry fields %s; "
                "dropping those keys (see osprey.bridges.core.ports.RESERVED_ENTRY_KEYS)",
                parsed.message_id,
                sorted(meta_collisions),
            )
            for key in meta_collisions:
                del meta[key]
        deps.dedup.record(
            parsed.message_id, run_id=None, status="pending", reply_to=reply_to, **meta
        )

    # Assemble the dispatch payload off the persisted entry (refreshed: it now
    # carries the reply meta too). The freshly resolved reply context is passed in
    # rather than re-read, so the live path never depends on the store's round-trip.
    entry = deps.dedup.get(parsed.message_id) or entry
    extra = build_extra(deps, entry, history_key=parsed.history_key, reply_to=reply_to)

    result = _dispatch(deps.dispatcher, deps.dedup, parsed.message_id, parsed.text, extra)

    # Settle the terminal result off the persisted entry — independent of the
    # inbound event, exactly like the drain and reconcile paths.
    entry = deps.dedup.get(parsed.message_id) or {}
    _settle_terminal(deps, parsed.message_id, entry, result)
    return "handled"


# --- input fold ---------------------------------------------------------------------


def _decoded_size(content_b64: Any) -> int:
    """Decoded byte size of one file's ``content_b64``; undecodable counts as zero
    (defensive; never raises)."""
    if not isinstance(content_b64, str) or not content_b64:
        return 0
    try:
        return len(base64.b64decode(content_b64))
    except Exception:
        return 0


def _apply_fresh_budget(
    buckets: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> list[list[dict[str, Any]]]:
    """Enforce the SINGLE shared fresh-file budget across ``buckets`` in order.

    Each bucket is ``(files, skipped)``; the count and byte caps are shared across
    all buckets (own-message files first, quoted after — so own attachments win a
    tight budget), and each over-budget file degrades to a skip note in ITS OWN
    bucket's ``skipped`` list, never a sibling's. Returns the kept files per bucket.
    """
    kept_per_bucket: list[list[dict[str, Any]]] = []
    count = 0
    total = 0
    for files, skipped in buckets:
        kept: list[dict[str, Any]] = []
        for item in files:
            if count >= MAX_FILES:
                skipped.append({"filename": item.get("filename"), "reason": TOO_MANY_FILES_REASON})
                continue
            size = _decoded_size(item.get("content_b64"))
            if total + size > MAX_TOTAL_BYTES:
                skipped.append({"filename": item.get("filename"), "reason": OVER_BUDGET_REASON})
                continue
            kept.append(item)
            count += 1
            total += size
        kept_per_bucket.append(kept)
    return kept_per_bucket


def _fold_inputs(inputs: InputDownload, extra: dict[str, Any], probe: Callable[[], bool]) -> None:
    """Budget the downloaded files and capability-gate their delivery into ``extra``,
    keeping the two provenance buckets of :class:`~osprey.bridges.core.ports.InputDownload`
    separate end to end.

    Three outcomes, applied per bucket off ONE shared budget and ONE shared probe:

      * no deliverable file  -> ship only the skip notes (if any); the capability
        probe never runs (nothing to send).
      * capable pair         -> ``input_files`` (adapter-shaped, ``ingest=True``,
        own files before quoted) plus the skip notes.
      * non-capable pair     -> NO ``input_files`` (bytes an old worker would drop
        are never shipped); each undeliverable file gains a
        :data:`SERVER_TOO_OLD_REASON` skip so the agent can relay why.

    Own-message provenance rides top-level: ``extra["input_files"]`` /
    ``extra["skipped_attachments"]``. Quoted provenance rides INSIDE
    ``extra["reply_to"]``: delivered filenames as ``reply_to["attachments"]``, skips
    as ``reply_to["skipped_attachments"]`` — so the agent knows those files came from
    the quoted message, not the current one. Quoted files without a resolved
    ``reply_to`` are an adapter contract violation and are dropped with a warning
    (there is no quoted bucket to report them in).

    An empty :class:`~osprey.bridges.core.ports.InputDownload` is a strict no-op:
    ``extra`` stays byte-identical to the text-only path.
    """
    skipped = [dict(s) for s in inputs.skipped]
    quoted_skipped = [dict(s) for s in inputs.quoted_skipped]
    quoted_files = [dict(f) for f in inputs.quoted_files]

    reply_to = extra.get("reply_to")
    if (quoted_files or quoted_skipped) and not isinstance(reply_to, dict):
        logger.warning(
            "download_inputs returned quoted files/skips but no reply context resolved; "
            "dropping the quoted bucket"
        )
        quoted_files, quoted_skipped = [], []

    files, quoted = _apply_fresh_budget(
        [([dict(f) for f in inputs.files], skipped), (quoted_files, quoted_skipped)]
    )
    # ONE probe for both buckets (memoized by the caller); no deliverable file
    # anywhere means the probe never runs — there is nothing to send.
    capable = probe() if (files or quoted) else False

    # Own bucket -> top-level provenance.
    if files:
        if capable:
            extra["input_files"] = files
            extra["skipped_attachments"] = skipped
        else:
            extra["skipped_attachments"] = skipped + [
                {"filename": f.get("filename"), "reason": SERVER_TOO_OLD_REASON} for f in files
            ]
    elif skipped:
        # Every own file was declined: the skip reasons still ride the payload so
        # the agent can relay them.
        extra["skipped_attachments"] = skipped

    # Quoted bucket -> provenance INSIDE reply_to. Delivered bytes still ride
    # input_files (after the own files), but the filenames/skips land on reply_to
    # so the agent knows they came from the quoted message.
    if quoted and isinstance(reply_to, dict):
        if capable:
            extra.setdefault("input_files", []).extend(quoted)
            reply_to["attachments"] = [f.get("filename") for f in quoted]
        else:
            quoted_skipped = quoted_skipped + [
                {"filename": f.get("filename"), "reason": SERVER_TOO_OLD_REASON} for f in quoted
            ]
    if quoted_skipped and isinstance(reply_to, dict):
        reply_to["skipped_attachments"] = quoted_skipped


# --- prior-image re-injection -------------------------------------------------------


def _content_hashes(input_files: Any) -> set[bytes]:
    """SHA-256 digests of the bytes already in ``input_files`` (this message's own
    and quoted attachments), for dedup against re-injected priors. Tolerates a
    missing key or an item without decodable bytes."""
    hashes: set[bytes] = set()
    for item in input_files or []:
        b64 = item.get("content_b64") if isinstance(item, dict) else None
        if not isinstance(b64, str) or not b64:
            continue
        try:
            hashes.add(hashlib.sha256(base64.b64decode(b64)).digest())
        except Exception:
            continue
    return hashes


def _flag_expired(turns: list[dict[str, Any]], ti: int, ai: int) -> None:
    """Flag one prior-artifact descriptor with :data:`EXPIRED_NOTE` WITHOUT mutating
    the :class:`~osprey.bridges.core.history.HistoryStore`'s copy.

    ``HistoryStore.recent`` shallow-copies each turn but SHARES the nested artifact
    list and descriptor dicts with the persisted store, so we replace the turn's
    artifact list (and the one descriptor) with fresh objects the store never sees.
    Idempotent and safe to call repeatedly for the same turn."""
    turn = turns[ti]
    arts = list(turn.get("artifacts") or [])
    if not (0 <= ai < len(arts)) or not isinstance(arts[ai], dict):
        return
    arts[ai] = {**arts[ai], "note": EXPIRED_NOTE}
    turn["artifacts"] = arts


def _reinject_prior_images(
    deps: PipelineDeps,
    extra: dict[str, Any],
    turns: list[dict[str, Any]],
    probe: Callable[[], bool],
) -> None:
    """Re-inject the newest prior IMAGE artifacts (bytes) as ``ingest=False`` files.

    A follow-up dispatch already carries the prior turns in ``conversation_so_far``
    (descriptors and all); this additionally ships the BYTES of the newest
    :data:`REINJECT_MAX_IMAGES` prior images (across all turns, within
    :data:`REINJECT_MAX_TOTAL_BYTES`) so the agent can look at them again, not just
    read a descriptor. Runs AFTER the current message's own files are folded.

    Gated on the SAME ``input_files`` capability probe as the current-message fold:
    a non-capable pair gets no bytes anywhere (the descriptors still ride
    ``conversation_so_far`` untouched). Each selected image is pulled via
    :attr:`PipelineDeps.fetch_prior_artifact` against the shared budget; a fetch
    that fails or would bust the budget flags its descriptor with
    :data:`EXPIRED_NOTE` and is skipped — never fatal, never blocks dispatch. Bytes
    are deduped by SHA-256 against the files the current message already attached
    (and against each other), so a user who re-attaches the very plot they are
    asking about does not ship it twice.
    """
    if not turns:
        return

    # Collect image descriptors chronologically (turns oldest-first; within a turn,
    # descriptors are already oldest-first), remembering each one's turn+index so a
    # failed fetch can flag exactly that descriptor.
    candidates: list[tuple[int, int, str | None, dict[str, Any]]] = []
    for ti, turn in enumerate(turns):
        run_id = turn.get("run_id")
        for ai, desc in enumerate(turn.get("artifacts") or []):
            if isinstance(desc, dict) and desc.get("delivered_mime") in IMAGE_MIMES:
                candidates.append((ti, ai, run_id, desc))
    if not candidates:
        return

    # Probe only now — never for a dispatch with no prior images to re-inject.
    if not probe():
        return

    # Newest wins the budget: keep the newest N, process them newest-first so a
    # tight budget starves the OLDER images, then re-inject in chronological order.
    selected = candidates[-REINJECT_MAX_IMAGES:]
    seen = _content_hashes(extra.get("input_files"))
    injected: list[dict[str, Any]] = []
    remaining = REINJECT_MAX_TOTAL_BYTES
    for ti, ai, run_id, desc in reversed(selected):
        fetched = deps.fetch_prior_artifact(deps.cfg, run_id, desc, remaining)
        if fetched is None:
            _flag_expired(turns, ti, ai)
            continue
        # ``delivered_mime`` only PREDICTED an image (it is stamped at run
        # completion, before any conversion runs); the byte route's Content-Type
        # is what was actually served. An artifact whose conversion failed comes
        # back as its original, non-image bytes — re-injecting those as the image
        # the agent is told to look at is worse than not re-injecting them, and
        # the descriptor rides ``conversation_so_far`` either way.
        mime = fetched.content_type or desc.get("delivered_mime")
        if mime not in IMAGE_MIMES:
            logger.warning(
                "prior artifact served as %s, not the predicted %s; not re-injected",
                mime,
                desc.get("delivered_mime"),
            )
            continue
        data = fetched.data
        digest = hashlib.sha256(data).digest()
        if digest in seen:
            continue  # already attached to this message (or an earlier prior)
        seen.add(digest)
        injected.append(
            {
                "filename": desc.get("filename"),
                "mime": mime,
                "content_b64": base64.b64encode(data).decode("ascii"),
                "ingest": False,
            }
        )
        remaining -= len(data)

    if injected:
        injected.reverse()  # restore chronological order among the re-injected
        extra.setdefault("input_files", []).extend(injected)


# --- terminal settle ----------------------------------------------------------------


def _settle_terminal(
    deps: PipelineDeps, message_id: str, entry: dict[str, Any], result: dict[str, Any]
) -> None:
    """Settle a terminal dispatch result off the persisted ``entry``.

    A retryable failure (provider/infrastructure) is parked in the retry queue via
    :attr:`PipelineDeps.park` — the user gets the queued notice from ``park`` itself,
    and the history append is SKIPPED (the exchange is unfinished; the drain records
    it when the queued run completes). ``post_queued`` raising AFTER the park CAS
    committed is swallowed here: a lost notice never loses the request.

    Every other result is final, in crash-safe order: send the answer FIRST
    (``post_answer`` raises on a failed delivery — the entry is then re-queued with
    its run id intact so the drain re-attaches and re-delivers next cycle, and no
    terminal mark or history append happens), THEN record the terminal status, then
    best-effort ``deliver_files`` (never raises), then the history append with the
    returned ``{artifact_id: public_url}`` stamped onto this turn's descriptors.
    """
    if retry.is_retryable(result):
        try:
            deps.park(deps.ops, deps.dedup, message_id, entry, result)
        except Exception:
            logger.exception("queued notice failed for %s; entry stays parked", message_id)
        return

    try:
        deps.ops.post_answer(entry, result)
    except Exception:
        logger.exception(
            "answer delivery failed for %s; re-queueing for the drain to re-deliver", message_id
        )
        _requeue_for_redelivery(deps, message_id, entry, result)
        return

    # Answer landed: only run_id/status change here; the claim persisted the rest.
    deps.dedup.record(message_id, run_id=result.get("run_id"), status=result.get("status", "error"))
    urls = _deliver_files(deps, message_id, entry, result)
    _append_history(deps, entry, result, urls)


def _deliver_files(
    deps: PipelineDeps, message_id: str, entry: Mapping[str, Any], result: Mapping[str, Any]
) -> dict[str, str]:
    """Deliver the run's OUTPUT artifacts, returning the ``{artifact_id: public_url}``
    map of whatever the channel republished (empty when it republished nothing).

    The one file-delivery step for every path that has already posted an answer — the
    live settle and the drain's re-attached delivery. ``deliver_files`` degrades to
    text-only rather than raising, by contract; a misbehaving adapter that raises anyway
    must not un-deliver the text that already landed, so the raise is logged and treated
    as "nothing republished" — the terminal mark and the history append still happen.
    """
    try:
        return dict(deps.ops.deliver_files(entry, result) or {})
    except Exception:
        logger.exception("file delivery raised for %s; text already sent", message_id)
        return {}


def _requeue_for_redelivery(
    deps: PipelineDeps, message_id: str, entry: dict[str, Any], result: dict[str, Any]
) -> None:
    """Keep an entry whose answer could not be delivered QUEUED (run id intact).

    The drain's re-attach path then probes the persisted run and re-delivers next
    cycle (at-least-once, bounded by the give-up ceilings) — a transient post failure
    must never terminal-mark an undelivered answer or silently drop it. Stamps the
    queue bookkeeping the drain's pacing/ceiling policy reads; ``attempts`` is left
    to the drain (this was a delivery failure, not a dispatch attempt)."""
    now = time.time()
    deps.dedup.transition(
        message_id,
        str(entry.get("status") or "pending"),
        "queued",
        run_id=result.get("run_id") or entry.get("run_id"),
        queued_at=now,
        first_queued_at=entry.get("first_queued_at") or entry.get("queued_at") or now,
    )


# --- history ------------------------------------------------------------------------


def _append_history(
    deps: PipelineDeps,
    entry: dict[str, Any],
    result: dict[str, Any],
    artifact_urls: dict[str, str] | None = None,
) -> None:
    """Record this exchange under the entry's persisted ``history_key`` so the NEXT
    question in the conversation carries it. A completed run with text keeps the
    real answer, the producing ``run_id``, and this turn's artifact descriptors (so
    a follow-up can refer back to a plot the agent already made); anything else
    (timeout, error, or a completed run with no text) keeps the QUESTION with a
    marker so a follow-up still resolves its referent — the raw error text is never
    replayed, and a failed turn carries no descriptors. Best-effort: a history write
    must never fail a delivery that already landed.
    """
    if deps.history is None:
        return
    key = entry.get("history_key")
    if not key:
        return
    question = entry.get("text", "")
    answer = result.get("text_output") or ""
    try:
        if result.get("status") == "completed" and answer:
            deps.history.append(
                key,
                question,
                answer,
                run_id=result.get("run_id"),
                artifacts=_turn_descriptors(result, artifact_urls),
            )
        else:
            deps.history.append_failed(key, question)
    except Exception:
        logger.exception("history append failed for %s; continuing", key)


def _turn_descriptors(
    result: dict[str, Any], artifact_urls: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Assemble this turn's artifact descriptors, OLDEST-first.

    Order is inputs (what the user sent, older) then outputs (what the run produced,
    newer), each in delivery order, so the history store's tail-keeping cap retains
    the NEWEST when it trims. Each descriptor is
    ``{entry_id, filename, delivered_mime, public_url?, origin}`` with ``origin`` in
    ``{"input", "output"}``. Only OUTPUT descriptors ever carry ``public_url``
    (stamped from the ``deliver_files`` return); INPUT files are never republished,
    so never get one — a hard invariant."""
    url_map = artifact_urls if isinstance(artifact_urls, dict) else {}
    descriptors = _input_descriptors(result) + _output_descriptors(result, url_map)
    return descriptors[-MAX_TURN_DESCRIPTORS:]


def _output_descriptors(
    result: dict[str, Any], artifact_urls: dict[str, str]
) -> list[dict[str, Any]]:
    """OUTPUT descriptors from the run-status ``artifacts`` field (descriptor dicts,
    or bare id strings from an older worker treated as PNG images). A ``public_url``
    is stamped only when the artifact was republished by ``deliver_files`` (its id is
    in ``artifact_urls``)."""
    descriptors: list[dict[str, Any]] = []
    for a in result.get("artifacts") or []:
        entry_id: str
        filename: Any = None
        mime: Any = "image/png"
        if isinstance(a, str):
            if not a:
                continue
            entry_id = a
        elif isinstance(a, dict):
            raw_id = a.get("artifact_id")
            if not (isinstance(raw_id, str) and raw_id):
                continue
            entry_id = raw_id
            filename = a.get("filename")
            mime = a.get("delivered_mime")
        else:
            continue
        descriptor: dict[str, Any] = {
            "entry_id": entry_id,
            "filename": filename,
            "delivered_mime": mime,
            "origin": "output",
        }
        url = artifact_urls.get(entry_id)
        if url:
            descriptor["public_url"] = url
        descriptors.append(descriptor)
    return descriptors


def _input_descriptors(result: dict[str, Any]) -> list[dict[str, Any]]:
    """INPUT descriptors from the run-status ``input_artifacts`` field
    (``[{entry_id, filename, mime}]`` from the worker). Inputs are never
    republished, so a descriptor NEVER carries a ``public_url``."""
    descriptors: list[dict[str, Any]] = []
    for a in result.get("input_artifacts") or []:
        if not isinstance(a, dict):
            continue
        entry_id = a.get("entry_id")
        if not (isinstance(entry_id, str) and entry_id):
            continue
        descriptors.append(
            {
                "entry_id": entry_id,
                "filename": a.get("filename"),
                "delivered_mime": a.get("mime"),
                "origin": "input",
            }
        )
    return descriptors
