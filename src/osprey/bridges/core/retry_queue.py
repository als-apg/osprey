"""The retry queue's write side: park, supersede-at-enqueue, give-up notice.

The drain loop (:mod:`osprey.bridges.core.drain`) owns *reading* the queue — which
parked entry is eligible, which has aged out, which gets redispatched. This module owns
the three places something is *put into* or *taken out of* the queue by a decision the
drain does not make:

  * :func:`park` — a retryable failure becomes a ``queued`` entry plus an honest notice.
  * :func:`supersede_at_enqueue` — parking a newer message drops the older queued
    siblings it makes redundant (called by :func:`park`; exposed for tests/diagnostics).
  * :func:`give_up` / :func:`supersede_note` — the two channel-side notices the drain
    invokes through its callbacks, bound to :class:`~osprey.bridges.core.ports.ChannelOps`.

Every store mutation goes through :meth:`~osprey.bridges.core.dedup.DedupStore.transition`,
the compare-and-set primitive: it is what keeps this module from racing the drain thread.
A CAS that loses (the entry already moved out of the status we expected) is never an
error here — it means another actor legitimately owns that entry now, and we do nothing.

Ordering is the safety property, and it differs per notice because the failure contracts
in :mod:`osprey.bridges.core.ports` differ:

  * ``post_queued`` **may raise**: the entry is parked and its siblings superseded
    *before* the notice is posted, so a lost notice costs the user a message, never
    their request.
  * ``post_giveup`` **must raise**: :func:`give_up` posts and does NOT mark the entry
    terminal — the drain does that only after this returns, so a failed notice leaves
    the entry queued for the next cycle rather than silently dropped.
  * ``post_superseded`` **never raises**: the coalesce CAS is already committed, so a
    failed note is swallowed and must not disturb the rest of the pass.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import retry
from .dedup import DedupStore
from .history import HistoryStore
from .ports import ChannelOps

logger = logging.getLogger(__name__)

# Status a queued entry is flipped to when a newer question in the same conversation
# coalesces over it. This is a TERMINAL drop — the earlier request is intentionally
# abandoned in favor of the newer one and must never be re-dispatched. For
# restart-safety the string MUST be a member of
# :data:`~osprey.bridges.core.config.TERMINAL_STATUSES`; otherwise
# :meth:`~osprey.bridges.core.dedup.DedupStore.reconcile` re-surfaces a superseded
# entry on the next start and re-runs the dropped question.
SUPERSEDED_STATUS = "superseded"

__all__ = [
    "SUPERSEDED_STATUS",
    "give_up",
    "park",
    "supersede_at_enqueue",
    "supersede_note",
]


def park(
    ops: ChannelOps,
    dedup: DedupStore,
    message_id: str,
    entry: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    now: Callable[[], float] = time.time,
) -> bool:
    """Park a retryable-failure request in the retry queue for the drain loop.

    Returns ``True`` iff this call actually parked the entry. Steps, in order:

      1. Flip the entry to ``queued`` — a CAS off its current (non-terminal) status,
         stamping the retry bookkeeping the drain reads. ``failure_class`` and
         ``num_tool_calls`` are copied verbatim from ``result``: ``None`` (unknown) is
         preserved as ``None``, never coerced to ``0``, because
         :func:`osprey.bridges.core.retry.may_redispatch` is fail-closed on unknown.
         ``queued_at`` is the per-attempt pacing clock and refreshes on every park;
         ``first_queued_at`` is the give-up-ceiling anchor and is stamped only once, so
         a cycling entry cannot re-age forever. ``attempts`` starts at ``0`` on the
         first park and is otherwise left to the drain, which increments it per
         redispatch — a re-park never resets it.
      2. Only if that CAS won, supersede the older queued siblings sharing this entry's
         coalesce key (see :func:`supersede_at_enqueue`).
      3. On the FIRST park only, post the "your request is queued" notice. A re-park is
         silent: the user was already told once, and the next message in that
         conversation is either the answer or the give-up notice.

    The CAS losing (``False``) means someone else already moved this entry — a racing
    drain, a late terminal callback — so nothing is parked, nothing is superseded and
    no notice is posted.

    **The park CAS runs before supersession on purpose.** Supersession is only ever
    justified by the newer message actually taking the older one's place: cancelling an
    older queued question on behalf of a newer one that did not itself park would leave
    the user with no answer to either and no explanation for the first. Callers must
    honor the same precondition — :func:`park` is called on a retryable failure, never
    on a dispatch that is still in flight or that succeeded.

    History is deliberately NOT appended: the exchange is unfinished. The drain appends
    under the persisted ``history_key`` once the queued run finally completes.

    Reads destination/threading/sender only from the persisted ``entry``, so it serves
    the live pipeline and the drain's re-park identically. ``post_queued`` may raise
    (see :mod:`osprey.bridges.core.ports`); the raise propagates to the caller *after*
    the queue state is committed, so a lost notice never loses the request.
    """
    key = retry.coalesce_key({"coalesce_key": ops.coalesce_key(entry)})

    parked_at = now()
    first_park = not entry.get("first_queued_at") and not entry.get("queued_at")
    meta: dict[str, Any] = {
        "failure_class": result.get("failure_class"),
        "num_tool_calls": result.get("num_tool_calls"),
        "queued_at": parked_at,
        "first_queued_at": entry.get("first_queued_at") or entry.get("queued_at") or parked_at,
        # Persisted JSON-plain (a list) and normalized back through
        # retry.coalesce_key on every read, so the on-disk shape round-trips.
        "coalesce_key": list(key),
    }
    if first_park:
        meta["attempts"] = 0

    from_status = str(entry.get("status") or "pending")
    if not dedup.transition(message_id, from_status, "queued", **meta):
        logger.info(
            "park of %s lost the CAS off %r; leaving it to whoever owns it now",
            message_id,
            from_status,
        )
        return False

    supersede_at_enqueue(ops, dedup, message_id, key)

    if first_park:
        ops.post_queued(dedup.get(message_id) or {**dict(entry), **meta}, result)
    return True


def supersede_at_enqueue(
    ops: ChannelOps,
    dedup: DedupStore,
    message_id: str,
    key: tuple,
) -> list[str]:
    """Drop every OLDER queued entry sharing ``key``, returning the ids dropped.

    Called by :func:`park` right after ``message_id`` itself became ``queued``, so every
    other entry in the queue predates it: a newer question in the same conversation
    makes the older queued one redundant. Each match is CAS-flipped ``queued ->
    superseded`` (a terminal drop) and gets its own one-line note in its OWN thread —
    the user is never left wondering why an earlier question went unanswered.

    ``key`` is the normalized tuple from :func:`osprey.bridges.core.retry.coalesce_key`.
    The empty tuple is the *no-group* sentinel: a keyless entry stands alone and is
    never coalesced (keyless entries must not collapse into each other).

    The CAS is the race guard: a drain that already claimed an entry (``queued ->
    in_flight``) wins, and we then neither drop nor note it. Notes go through
    :func:`supersede_note`, so they are best-effort here exactly as they are on the
    drain side — the flip is already committed, and a failing ``post_superseded`` is
    logged while the pass continues with the remaining siblings.
    """
    if not key:
        return []
    dropped: list[str] = []
    for mid, sibling in dedup.queued().items():
        if mid == message_id or retry.coalesce_key(sibling) != key:
            continue
        if not dedup.transition(mid, "queued", SUPERSEDED_STATUS):
            continue
        dropped.append(mid)
        logger.info("coalesced %s into %s", mid, message_id)
        supersede_note(ops, dedup, mid, sibling)
    return dropped


def give_up(
    ops: ChannelOps,
    dedup: DedupStore,
    message_id: str,
    entry: Mapping[str, Any],
    *,
    history: HistoryStore | None = None,
) -> None:
    """Post the honest "couldn't run this automatically" notice for an abandoned entry.

    Binds :meth:`~osprey.bridges.core.ports.ChannelOps.post_giveup` for the drain's
    ``give_up`` callback. It deliberately does NOT mark the entry terminal: the drain
    does that only after this returns, so a notice that fails to send leaves the entry
    queued for the next cycle. A lost give-up notice is silence, which is worse than
    the rare duplicate a retry could produce — hence the raise propagates.

    The failed turn is recorded in conversation history (so the transcript shows the
    question was asked and not answered) only AFTER the notice landed, and best-effort:
    a history write must never turn a delivered notice into a retried one.
    """
    current = dict(dedup.get(message_id) or entry)
    ops.post_giveup(current)

    key = current.get("history_key")
    if history is not None and key:
        try:
            history.append_failed(key, current.get("text", ""))
        except Exception:
            logger.exception("give-up history append failed for %s; continuing", message_id)


def supersede_note(
    ops: ChannelOps,
    dedup: DedupStore,
    message_id: str,
    entry: Mapping[str, Any],
) -> None:
    """Note that a queued entry was coalesced away by a newer sibling.

    Binds :meth:`~osprey.bridges.core.ports.ChannelOps.post_superseded` for the drain's
    optional ``supersede`` callback, covering DRAIN-side collapses (:func:`park`'s
    enqueue path already notes its own drops). No history is written for a dropped
    turn. Best-effort: the coalesce CAS is already committed, so a failed note must
    never disturb the drain pass.
    """
    try:
        ops.post_superseded(dict(dedup.get(message_id) or entry))
    except Exception:
        logger.exception("supersede note post failed for %s; continuing", message_id)
