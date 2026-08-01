"""Startup crash recovery: drive to terminal whatever was in flight when we stopped.

A bridge claims a message BEFORE it dispatches, and marks the entry terminal only
AFTER the answer has been posted. That ordering is what makes delivery at-least-once
— but it also means a crash anywhere in between leaves a non-terminal entry that
nothing else will ever pick up: the channel's own redelivery is defeated by the dedup
claim, and the retry drain deliberately owns only ``queued`` entries. This module is
the other half of that contract. It runs once at startup, over
:meth:`~osprey.bridges.core.dedup.DedupStore.reconcile` (non-terminal and not
``queued``), and finishes each entry.

Three cases, distinguished by what the entry already carries:

``retried`` and no ``run_id``
    A drain redispatch that crashed after CAS-flipping ``queued -> in_flight`` but
    before a run existed. The drain owns retried entries, so hand it straight back to
    the queue — startup must never dispatch one, so the health gate and the retry
    min-age always mediate the next attempt.

a ``run_id``
    Crashed mid-poll: a run WAS started and may already have reached hardware.
    **Re-attach** — poll the worker to terminal and deliver. Never re-dispatch: a
    second run of the same question is a second set of side effects.

no ``run_id`` but a persisted ``text``
    Crashed after the claim, before the dispatcher handshake yielded a run id. No run
    was ever started, so a **re-dispatch** is safe — which is precisely why the claim
    persists ``text`` before anything is fired.

Anything else (no run id, no text) is left alone rather than guessed at.

Delivery goes through :class:`~osprey.bridges.core.ports.ChannelOps`, whose failure
contracts carry the crash-safety here: ``post_answer`` MUST raise, and a raise during
reconcile marks the entry ``post_failed``. That is deliberately harsher than the drain's
"stay queued and re-deliver next cycle": startup runs once per process, so an entry
whose destination is permanently gone (a deleted thread, a bounced address) would
otherwise be retried on every restart forever, and a raise out of here would crashloop
the bridge before it ever subscribed.

Two seams are injectable so the module stays independent of the rest of the engine:
``build_extra`` (re-assemble a re-dispatch's context — conversation replay, reply
context, re-downloaded attachments) and ``settle`` (what to do with a terminal result).
Both default to a self-contained implementation; the runtime overrides them with the
pipeline's versions so a reconciled result settles exactly like a live one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .config import TERMINAL_STATUSES
from .dedup import DedupStore
from .dispatch_client import DispatchClient
from .pipeline import _dispatch
from .ports import ChannelOps

__all__ = [
    "ReconcileDeps",
    "ReconcileReport",
    "deliver_terminal",
    "reconcile_inflight",
]

logger = logging.getLogger(__name__)


class ReconcileDeps(Protocol):
    """The collaborators reconcile needs, as a structural type.

    A Protocol rather than a concrete dataclass so the runtime's own (larger)
    dependency bundle can be passed straight in — reconcile reads exactly these three
    attributes and nothing else.
    """

    dedup: DedupStore
    dispatcher: DispatchClient
    ops: ChannelOps


@dataclass
class ReconcileReport:
    """What one startup pass did, for the runtime's startup log and for tests.

    ``failed`` counts entries whose dispatch/poll raised and were left for the next
    startup; ``post_failed`` counts entries abandoned because the answer could not be
    delivered. Both are non-fatal — a reconcile pass never raises.
    """

    reattached: int = 0
    redispatched: int = 0
    requeued: int = 0
    skipped: int = 0
    failed: int = 0
    post_failed: int = 0

    @property
    def total(self) -> int:
        """Every non-terminal entry this pass looked at."""
        return (
            self.reattached
            + self.redispatched
            + self.requeued
            + self.skipped
            + self.failed
            + self.post_failed
        )


def deliver_terminal(
    deps: ReconcileDeps,
    message_id: str,
    entry: Mapping[str, Any],
    result: Mapping[str, Any],
) -> bool:
    """Deliver a reconciled terminal result and settle the entry. Returns whether the
    answer landed.

    Send BEFORE the terminal mark, always: the entry stays reconcilable until the user
    actually has the answer. A ``post_answer`` raise (its contract — see
    :mod:`~osprey.bridges.core.ports`) means the answer did NOT land, and the entry is
    marked ``post_failed`` rather than retried on every future restart. File delivery
    is best-effort by contract and never un-delivers the text that already landed.

    This is the DEFAULT settlement, deliberately the conservative common denominator of
    the two proven channels: no retry-queue park and no history append. The runtime
    injects the pipeline's settlement instead, so a retryable failure discovered at
    startup is parked for the drain and a delivered answer is recorded in history.
    """
    try:
        deps.ops.post_answer(entry, result)
    except Exception:
        logger.exception("reconcile post failed for %s; marking post_failed", message_id)
        deps.dedup.update_status(message_id, "post_failed")
        return False
    try:
        deps.ops.deliver_files(entry, result)
    except Exception:
        # Contractually unreachable (deliver_files degrades to text-only rather than
        # raising), but the answer text has already landed — a misbehaving adapter must
        # not cost us the terminal mark and leave the entry to be re-delivered forever.
        logger.exception("reconcile file delivery raised for %s; text already sent", message_id)
    status = result.get("status", "error")
    deps.dedup.update_status(message_id, status if status in TERMINAL_STATUSES else "error")
    return True


def reconcile_inflight(
    deps: ReconcileDeps,
    *,
    build_extra: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    settle: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Any] | None = None,
) -> ReconcileReport:
    """Finish every in-flight message left over from the previous process.

    Call once at startup, before the inbound loop starts: the entries handled here are
    invisible to every other mechanism, and a message the user is still waiting on
    should be answered before new ones compete for the worker.

    ``build_extra`` re-assembles the dispatch context for a re-dispatch off the
    persisted entry (fresh conversation history, the persisted reply context,
    attachments re-downloaded from the persisted refs). Default ``None`` means "replay
    the question alone" — correct but thinner than a live dispatch, so the runtime
    passes the pipeline's builder. ``settle`` handles a terminal result and defaults to
    :func:`deliver_terminal`.

    Never raises: a per-entry dispatch failure leaves that entry for the next startup
    and the pass continues. Returns a :class:`ReconcileReport` of what it did.
    """

    def default_settle(mid: str, current: Mapping[str, Any], body: Mapping[str, Any]) -> Any:
        return deliver_terminal(deps, mid, current, body)

    settle_result = settle if settle is not None else default_settle
    report = ReconcileReport()

    for message_id, entry in deps.dedup.reconcile().items():
        run_id = entry.get("run_id")

        # A drain redispatch that crashed before a run_id existed. The drain owns it;
        # dispatching from startup would bypass the health gate and the retry min-age
        # that exist to keep a failing service from being hammered on every restart.
        if entry.get("retried") and not run_id:
            logger.info("returning retried in-flight %s to the queue", message_id)
            deps.dedup.transition(message_id, entry.get("status", "in_flight"), "queued")
            report.requeued += 1
            continue

        text = entry.get("text")
        reattached = bool(run_id)
        try:
            if run_id:
                # RE-ATTACH. The run exists and may already have touched hardware, so
                # this path must never fire a second dispatch. A fresh poll budget:
                # the run has been going since before the crash, and the worker's own
                # cap — not our previous, now-lost deadline — bounds it.
                logger.info("reconciling in-flight run %s (%s)", run_id, message_id)
                result = deps.dispatcher.poll_worker(run_id)
            elif text:
                # RE-DISPATCH. No run was ever started (that is what "no run_id" means
                # for a non-retried entry), so replaying the persisted question is the
                # only way this message is ever answered.
                logger.info("re-dispatching %s (crashed before a run_id existed)", message_id)
                extra = build_extra(entry) if build_extra is not None else None
                result = _dispatch(deps.dispatcher, deps.dedup, message_id, text, extra)
            else:
                # Neither a run to re-attach nor a question to replay: there is nothing
                # correct to do, and guessing would either drop the entry's dedup claim
                # or invent a dispatch. Leave it for a human.
                logger.info("in-flight %s has no run_id and no text; leaving", message_id)
                report.skipped += 1
                continue
        except Exception:
            logger.exception("reconcile failed for %s; leaving for next startup", message_id)
            report.failed += 1
            continue

        # Re-read: the re-dispatch persister has just written the new run_id, and the
        # settlement (and the channel post it drives) must see the entry as stored.
        current = deps.dedup.get(message_id) or dict(entry)
        if settle_result(message_id, current, result) is False:
            # Only the default settlement reports delivery failure; an injected one
            # returning None counts as delivered (it owns its own outcome handling).
            report.post_failed += 1
        elif reattached:
            report.reattached += 1
        else:
            report.redispatched += 1

    return report
