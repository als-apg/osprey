"""Retry-queue drain loop: channel-neutral scheduling over parked entries.

When a bridge cannot dispatch an inbound message (the pipeline is down, the gate
is closed), it parks the claim as ``status == "queued"`` instead of dropping it.
This module owns replaying those parked entries once the pipeline recovers.

:func:`drain_once` is one pass:

  1. Consult the health gate. If it is closed, do nothing — replaying into a
     still-broken pipeline would just re-fail (and possibly re-file alerts).
  2. Coalesce redundant fresh entries: among run_id-less entries sharing a
     channel-supplied :func:`osprey.bridges.core.retry.coalesce_key`, keep only
     the newest and supersede the rest (``queued -> superseded`` via the store's
     CAS :meth:`~osprey.bridges.core.dedup.DedupStore.transition`), announcing
     each drop via the optional ``supersede`` callback. This runs on the FULL
     queue snapshot, BEFORE the age filter, so a newest-but-too-young message
     still supersedes an older eligible sibling (otherwise both would eventually
     be answered).
  3. Keep the survivors aged past ``retry_min_age``
     (:func:`osprey.bridges.core.retry.is_eligible`).
  4. For each surviving entry:
       * give-up check (:func:`osprey.bridges.core.retry.give_up_due`): abandon
         via the give-up callback, then mark the entry terminal (a failed notice
         leaves it queued to retry; the reason lands in ``give_up_reason`` meta);
       * **re-attach** when it already has a ``run_id`` — probe the worker once
         (:meth:`~osprey.bridges.core.dispatch_client.DispatchClient.status`): a
         ``completed`` run is delivered (callback) and marked terminal (a
         failed delivery leaves it queued so the next cycle re-delivers); a
         non-terminal run is left for the next cycle; a terminal FAILURE body is
         the run's definitive outcome, so the retry decision is applied — one
         redispatch iff :func:`~osprey.bridges.core.retry.is_retryable` AND
         :func:`~osprey.bridges.core.retry.may_redispatch` both pass, else give
         up; two *consecutive* 404s across cycles (tracked in entry meta) mean
         the run is gone with its outcome UNKNOWN, so give up — an unknown
         outcome is never re-dispatched;
       * otherwise **redispatch** — but only when a KNOWN ``num_tool_calls == 0``
         in the entry meta proves the failed attempt took no side-effecting
         action (:func:`~osprey.bridges.core.retry.may_redispatch`, fail-closed):
         unknown or nonzero means give up (the honest "couldn't safely re-run"
         path), NEVER re-execute. When provably safe: flip ``queued ->
         in_flight`` (CAS, stamping ``retried``) then hand off to the channel's
         dispatch callback, which re-fires and delivers (the existing
         ``_run_id_persister`` pattern).

Every channel-specific side effect (posting a reply, delivering a re-attached
result, filing a give-up notice) is injected through :class:`DrainCallbacks`, so
``osprey.bridges.core`` carries ZERO channel imports and ZERO osprey dispatch
imports. The store lifecycle (CAS transitions, terminal marking, 404
bookkeeping) stays here.

:func:`run_drain_thread` and :func:`ensure_alive` are the supervision helpers a
channel entrypoint uses to run the loop as a daemon thread and restart it if it
ever dies. One drain is shared by the whole bridge process — never one per
conversation — because the queue it walks is the process-wide dedup store.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import TERMINAL_STATUSES
from .dedup import DedupStore
from .dispatch_client import DispatchClient
from .health_gate import gate_open
from .retry import (
    ceiling_age,
    coalesce_key,
    give_up_due,
    is_eligible,
    is_retryable,
    may_redispatch,
)

logger = logging.getLogger(__name__)

# Two consecutive 404s from the worker (across drain cycles) mean the run record
# is gone for good — the worker never registered it, or it aged out — so the
# entry can never be re-attached and is abandoned. One 404 alone can be a brief
# registration race, so a single miss only defers to the next cycle.
NOTFOUND_GIVE_UP = 2

# Terminal status stamped on an abandoned entry: it is done, must leave both the
# queued() set and the reconcile() set so it is never reprocessed. "error" is an
# existing terminal status (see config.TERMINAL_STATUSES) — no new status needed.
_GIVE_UP_STATUS = "error"


@dataclass
class DrainCallbacks:
    """Channel-specific I/O the drain invokes. Everything here is pure channel
    behavior — the drain owns all dedup-store state, so a callback never needs to
    write status/run_id (except the ``_run_id_persister`` the channel wires into
    its own dispatch, for crash-safety).

    All callbacks run on the drain thread and may block (the dispatch callback
    runs a full worker poll). A raising callback is logged and never kills the
    loop; per-callback retry semantics are documented on each field.

    This is the drain's slice of the :class:`~osprey.bridges.core.ports.ChannelOps`
    seam, expressed as plain callables so the loop stays testable with three
    lambdas. The failure contracts below are the same ones ``ports.py``
    documents for ``post_answer`` / ``post_giveup`` / ``post_superseded``; a
    ``ChannelOps`` implementation satisfies them through the thin binding the
    runtime wiring builds (the callbacks also own store-independent channel
    steps — history append, file delivery — that no single protocol member
    covers).
    """

    dispatch: Callable[[str, dict[str, Any]], Any]
    """Redispatch a run-id-less queued entry from its persisted question.
    ``(message_id, entry)``. The channel re-fires the dispatch (with its
    ``_run_id_persister`` for crash-safety), delivers the reply, and records the
    terminal status itself — the drain has already flipped the entry to
    ``in_flight`` and does not touch it afterward."""

    deliver: Callable[[str, dict[str, Any], dict[str, Any]], Any]
    """Deliver a re-attached run's COMPLETED result. ``(message_id, entry,
    result_body)`` where ``result_body`` is the raw worker status body — only
    ``status == "completed"`` bodies reach this callback (terminal failures are
    routed through the retry decision instead). Raise on a failed send: the
    entry then STAYS QUEUED and the next cycle re-attaches and re-delivers
    (at-least-once, bounded by the give-up ceilings). Channels should mirror
    the same crash-safety ordering internally: send the reply first, then
    settle the store status."""

    give_up: Callable[[str, dict[str, Any]], Any]
    """Abandon an entry: notify the user / file a give-up issue. ``(message_id,
    entry)``. The drain marks the entry terminal only AFTER this returns — a
    notice that fails to send leaves the entry queued for the next cycle (a
    lost give-up notice is silence, worse than the rare duplicate a race could
    produce). The drain stamps its machine-side reason into the terminal
    transition meta (``give_up_reason``) for the audit trail."""

    supersede: Callable[[str, dict[str, Any]], Any] | None = None
    """Optional: note that an entry was coalesced away in favor of a newer
    sibling. ``(message_id, entry)``, called only after the drain WON the
    ``queued -> superseded`` CAS (at most once per entry). A channel that
    already posts its own supersede note at enqueue time (so the queue rarely
    holds duplicates) may leave this ``None``; the drain then coalesces
    silently."""


@dataclass
class DrainDeps:
    """Collaborators for :func:`drain_once`, injected for testability.

    ``cfg`` supplies the retry tunables (``retry_min_age`` / ``retry_give_up`` /
    ``retry_lifetime_cap`` / ``drain_interval``) and, for the default gate, the
    dispatcher/worker coordinates and ``trust_env`` that
    :func:`osprey.bridges.core.health_gate.gate_open` reads. ``gate`` / ``now`` /
    ``sleep`` are overridable so tests drive the clock and health signal
    directly.
    """

    cfg: Any
    dedup: DedupStore
    dispatcher: DispatchClient
    callbacks: DrainCallbacks
    gate: Callable[[], bool] | None = None
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep

    def gate_is_open(self) -> bool:
        """Health gate for this pass — the injected ``gate`` or, by default,
        :func:`osprey.bridges.core.health_gate.gate_open` against ``cfg`` (which
        builds its client honoring ``cfg.trust_env``)."""
        if self.gate is not None:
            return self.gate()
        return gate_open(self.cfg)


def _newest_first(entries: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Sort ``(message_id, entry)`` pairs newest-request first.

    "Newest" follows the request's FIRST park time (``first_queued_at``,
    falling back to ``queued_at``) — a re-park refreshes ``queued_at``, and
    sorting on that would let an old question in a retry loop out-rank a
    genuinely newer one during coalescing."""
    return sorted(
        entries,
        key=lambda item: item[1].get("first_queued_at") or item[1].get("queued_at", 0),
        reverse=True,
    )


def _ceiling_reason(entry: dict[str, Any], now: float, cfg: Any) -> str:
    """Human-readable reason for an age-ceiling give-up (same
    :func:`~osprey.bridges.core.retry.ceiling_age` anchor the ceilings
    measure)."""
    age = ceiling_age(entry, now)
    if age >= cfg.retry_lifetime_cap:
        return f"queued for {age:.0f}s, past the retry lifetime cap ({cfg.retry_lifetime_cap:.0f}s)"
    return (
        f"queued for {age:.0f}s without a successful retry, past the give-up age "
        f"({cfg.retry_give_up:.0f}s)"
    )


def _select(deps: DrainDeps, now: float) -> list[tuple[str, dict[str, Any]]]:
    """Queued entries to process this pass: coalesced first, THEN age-filtered.

    Coalescing runs on the FULL queue snapshot before the ``retry_min_age``
    filter so a newest-but-too-young message still supersedes an older eligible
    sibling — filtering first would answer the old one this pass and the new
    one a few passes later (both). Run-id-less entries sharing a coalesce key
    collapse to the newest — the older duplicates are superseded in the store —
    and the survivors are then age-filtered (re-attach entries, which have a
    run_id, are never coalesced and always pass through individually).
    """
    survivors: list[tuple[str, dict[str, Any]]] = []
    groups: dict[tuple, list[tuple[str, dict[str, Any]]]] = {}
    for mid, entry in deps.dedup.queued().items():
        # A dispatched entry (has a run_id) is re-attached on its own; only fresh
        # entries coalesce. An empty coalesce key is the no-group sentinel.
        key = coalesce_key(entry)
        if entry.get("run_id") or key == ():
            survivors.append((mid, entry))
        else:
            groups.setdefault(key, []).append((mid, entry))

    for members in groups.values():
        newest, *older = _newest_first(members)
        survivors.append(newest)
        for mid, entry in older:
            # CAS: only supersede if still queued (a racing pass may have taken it).
            if deps.dedup.transition(mid, "queued", "superseded"):
                logger.info("coalesced %s into %s", mid, newest[0])
                if deps.callbacks.supersede is not None:
                    try:
                        deps.callbacks.supersede(mid, entry)
                    except Exception:
                        logger.exception("supersede callback failed for %s", mid)
    return [(mid, entry) for mid, entry in survivors if is_eligible(entry, now, deps.cfg)]


def _reattach(deps: DrainDeps, message_id: str, entry: dict[str, Any], run_id: str) -> None:
    """Probe a persisted run once and act on its state (see module docstring)."""
    body = deps.dispatcher.status(run_id)

    if body is None:  # 404 — run not (yet) known to the worker
        misses = entry.get("notfound_count", 0)
        misses = (misses if isinstance(misses, int) and not isinstance(misses, bool) else 0) + 1
        if misses >= NOTFOUND_GIVE_UP:
            logger.info(
                "run %s gone (%d consecutive 404s); giving up %s", run_id, misses, message_id
            )
            _abandon(
                deps,
                message_id,
                entry,
                f"run {run_id} is unknown to the worker ({misses} consecutive 404s); "
                "its outcome is unknown, so it is never re-dispatched",
            )
        else:
            # Persist the miss and wait a cycle. CAS keeps it write-safe: the
            # meta lands only while the entry is still queued (a plain record()
            # could stomp a concurrent queued -> in_flight flip back to queued).
            deps.dedup.transition(message_id, "queued", "queued", notfound_count=misses)
        return

    status = body.get("status")
    if status not in TERMINAL_STATUSES:
        # Still running — clear any stale miss streak and try again next cycle.
        if entry.get("notfound_count"):
            deps.dedup.transition(message_id, "queued", "queued", notfound_count=0)
        return

    if status == "completed":
        # Deliver FIRST, then mark. A raising callback leaves the entry queued
        # (run_id intact) so the NEXT cycle re-attaches and re-delivers: a
        # transient post failure retries (at-least-once), bounded by the
        # give_up_due ceilings — never a first-failure terminal post_failed.
        # Channels mirror this crash-safety ordering internally (email: send
        # the reply, then settle the status) so a crash between the two
        # re-delivers rather than drops.
        try:
            deps.callbacks.deliver(message_id, entry, body)
        except Exception:
            logger.exception("deliver failed for %s; left queued to retry next cycle", message_id)
            return
        deps.dedup.transition(message_id, "queued", status)
        return

    # A terminal FAILURE body — the persisted run's definitive outcome. Not an
    # answer to deliver: apply the retry decision instead. One redispatch iff the
    # failure class is retryable AND a known-zero tool-call count (entry meta or
    # this body) proves the run took no action; anything else is abandoned.
    if is_retryable(body) and may_redispatch(entry, body):
        logger.info(
            "run %s terminal %r is safely retryable; redispatching %s", run_id, status, message_id
        )
        _redispatch(deps, message_id, entry, result=body)
    else:
        logger.info(
            "run %s terminal %r not safely retryable; giving up %s", run_id, status, message_id
        )
        _abandon(
            deps,
            message_id,
            entry,
            f"run {run_id} failed terminally "
            f"(failure_class={body.get('failure_class')!r}) and cannot be safely re-run",
        )


def _redispatch(
    deps: DrainDeps,
    message_id: str,
    entry: dict[str, Any],
    result: dict[str, Any] | None = None,
) -> None:
    """Safety-gate a queued entry, claim it (CAS), and hand it to dispatch.

    Re-dispatching re-executes the agent from scratch, so it must be PROVEN
    side-effect free: a KNOWN tool-call count of exactly 0 across the entry's
    persisted meta and (on the re-attach path) the observed terminal ``result``.
    Unknown or nonzero fails closed — the entry takes the give-up path (the
    channel's honest "couldn't safely re-run" notice), NEVER a re-execution.
    """
    if not may_redispatch(entry, result or {}):
        logger.info(
            "cannot prove %s side-effect free (num_tool_calls unknown/nonzero); giving up",
            message_id,
        )
        _abandon(
            deps,
            message_id,
            entry,
            "the failed attempt cannot be proven side-effect free "
            "(num_tool_calls unknown or nonzero), so re-running it is not safe",
        )
        return
    # Flip queued -> in_flight atomically so a concurrent pass can't double-fire;
    # only the winner runs the callback. `retried` and the incremented `attempts`
    # both feed give_up_due later (it honors either signal).
    attempts = entry.get("attempts")
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 0:
        attempts = 0
    won = deps.dedup.transition(
        message_id, "queued", "in_flight", retried=True, run_id=None, attempts=attempts + 1
    )
    if not won:
        return
    deps.callbacks.dispatch(message_id, entry)


def _abandon(deps: DrainDeps, message_id: str, entry: dict[str, Any], reason: str) -> None:
    """Fire the give-up callback, then mark the entry terminal (CAS-guarded).

    Callback FIRST: a give-up notice that fails to send leaves the entry queued
    for the next cycle rather than silently terminal — a lost notice is
    silence, worse than the rare duplicate a race could produce. ``reason`` is
    the drain's machine-side explanation; it never reaches the callback (the
    surface stays ``(message_id, entry)``) but is stamped into the terminal
    transition meta as ``give_up_reason`` for the audit trail.
    """
    try:
        deps.callbacks.give_up(message_id, entry)
    except Exception:
        logger.exception("give_up callback failed for %s; leaving queued", message_id)
        return
    deps.dedup.transition(message_id, "queued", _GIVE_UP_STATUS, give_up_reason=reason)


def drain_once(deps: DrainDeps) -> None:
    """Run one drain pass. No-op (returns early) when the health gate is closed.

    Each entry is processed inside its own try/except so a single failure — a
    worker error, a callback raising — is logged and skipped, never aborting the
    rest of the pass.
    """
    if not deps.gate_is_open():
        logger.debug("drain skipped: health gate closed")
        return

    now = deps.now()
    for message_id, entry in _select(deps, now):
        try:
            # Give-up takes precedence: an entry past its ceilings is abandoned
            # before we spend a redispatch/probe on it. The gate is open now, so
            # this outage-clearing counts as "gate observed open since enqueue".
            gate_seen = bool(entry.get("gate_seen_open"))
            if give_up_due(entry, now, gate_seen, deps.cfg):
                _abandon(deps, message_id, entry, _ceiling_reason(entry, now, deps.cfg))
                continue

            run_id = entry.get("run_id")
            if run_id:
                _reattach(deps, message_id, entry, run_id)
            else:
                _redispatch(deps, message_id, entry)

            # Record that the gate was open while this entry was still queued, so
            # a FUTURE pass can give up at the give-up age even with no retry.
            # Stamped after give_up_due so an entry gets one fair pass first. The
            # CAS applies it only if the entry is STILL queued (redispatch/deliver
            # may have moved it on) — atomically, unlike a get-then-record.
            if not gate_seen:
                deps.dedup.transition(message_id, "queued", "queued", gate_seen_open=True)
        except Exception as exc:  # isolate per-entry: one failure never aborts the pass
            logger.warning("drain: entry %s failed this pass: %s", message_id, exc)


# --- thread supervision ----------------------------------------------------


def run_drain_thread(deps: DrainDeps, stop: threading.Event | None = None) -> threading.Thread:
    """Start and return a daemon thread that calls :func:`drain_once` every
    ``cfg.drain_interval`` seconds until ``stop`` is set.

    ONE thread per bridge process, shared across every conversation: the pass it
    runs walks the whole dedup queue, so a per-room thread would multiply the
    same work and race on the same CAS transitions.

    The loop body is fully guarded: any exception escaping ``drain_once`` is
    logged and the loop continues, so the drain thread does not die on a
    transient fault. Daemon so it never blocks process exit; named for logs.
    """
    stop = stop or threading.Event()

    def _loop() -> None:
        logger.info("drain loop started (interval=%.0fs)", deps.cfg.drain_interval)
        while not stop.is_set():
            try:
                drain_once(deps)
            except Exception:
                logger.exception("drain_once raised; continuing")
            # Interruptible sleep: stop.set() wakes it immediately for shutdown.
            stop.wait(deps.cfg.drain_interval)

    thread = threading.Thread(target=_loop, name="bridge-drain", daemon=True)
    thread.start()
    return thread


def ensure_alive(
    thread: threading.Thread | None,
    factory: Callable[[], threading.Thread],
) -> threading.Thread:
    """Return a live drain thread: ``thread`` if it is still alive, else a fresh
    one from ``factory``.

    A channel's supervision path calls this periodically so a drain thread that
    somehow died (an unforeseen crash in the loop harness) is transparently
    replaced without restarting the whole bridge.
    """
    if thread is not None and thread.is_alive():
        return thread
    logger.warning("drain thread not alive; restarting")
    return factory()
