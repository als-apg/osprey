"""The process shell: build the engine's collaborators, recover, and supervise the drain.

This is what an adapter's ``__main__`` calls. It owns everything about running a bridge
EXCEPT ingestion:

* :func:`build_deps` — construct the collaborator set (stores, dispatch client, pipeline
  bundle) from a :class:`~osprey.bridges.core.config.CoreConfig` plus the adapter's
  :class:`~osprey.bridges.core.ports.ChannelOps` implementation.
* :func:`start` — run :func:`~osprey.bridges.core.reconcile.reconcile_inflight` FIRST
  (crash recovery completes before a single new message is accepted), then start the one
  supervised drain thread.
* :class:`BridgeRuntime` — the handle the adapter drives: ``handle_event`` per inbound
  message, ``supervise`` on its periodic wake, ``shutdown`` on exit.
* :func:`run_forever` — the whole of the above around the adapter's own blocking loop.

**The adapter owns ingestion.** How messages arrive is the one thing that does not
generalize (a Pub/Sub streaming pull, an IMAP poll, a thread per Talk room), so the
adapter runs its own inbound loop and calls :meth:`BridgeRuntime.handle_event` per
message. Core deliberately has no ingestion loop: putting one here would force every
channel through one arrival model, and the ordering guarantees that actually matter all
live downstream of ``handle_event``.

**One drain thread per process**, shared across every conversation: a drain pass walks
the whole dedup queue, so a per-room thread would multiply the same work and race on the
same CAS transitions. It is a daemon (never blocks process exit) and supervised — a
thread that somehow died is replaced without restarting the bridge.

The re-dispatch paths (the drain's ``dispatch`` callback and reconcile's re-dispatch of
an entry that crashed before a run id existed) must send the SAME payload the live path
would and settle its result the SAME way, or a replayed question silently loses its
conversation history, its reply context, and its retry-parking. So this module binds the
pipeline's own helpers for both rather than re-deriving them; they are module-private to
say "not adapter API", not "not engine API", and the one-implementation rule is the
whole point of the wiring living here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from . import retry_queue
from .config import CoreConfig
from .dedup import DedupStore
from .dispatch_client import DispatchClient
from .drain import DrainCallbacks, DrainDeps, ensure_alive, run_drain_thread
from .health_gate import gate_open
from .history import HistoryStore
from .pipeline import (
    PipelineDeps,
    _append_history,
    _deliver_files,
    _dispatch,
    _settle_terminal,
    build_extra,
    handle_event,
)
from .ports import ChannelOps
from .reconcile import ReconcileReport, reconcile_inflight

logger = logging.getLogger(__name__)

__all__ = [
    "BridgeRuntime",
    "build_deps",
    "build_drain_callbacks",
    "build_drain_deps",
    "rebuild_extra",
    "run_forever",
    "start",
]


# --- collaborator construction ------------------------------------------------------


def build_deps(
    cfg: CoreConfig,
    ops: ChannelOps,
    *,
    dedup: DedupStore | None = None,
    history: HistoryStore | None = None,
    dispatcher: DispatchClient | None = None,
) -> PipelineDeps:
    """Build the engine's collaborator set from ``cfg`` and the adapter's ``ops``.

    Returns the :class:`~osprey.bridges.core.pipeline.PipelineDeps` bundle every engine
    entry point takes — it also satisfies
    :class:`~osprey.bridges.core.reconcile.ReconcileDeps` structurally, so one bundle
    serves the live pipeline, startup reconcile and the drain wiring, and there is no
    second place a collaborator could be constructed differently.

    The stores are file-backed at the configured paths and the dispatch client honors
    ``cfg.trust_env``; each is injectable for tests and for an adapter that needs a
    pre-configured instance. The remaining :class:`PipelineDeps` seams (``park``,
    ``probe_capability``, ``fetch_prior_artifact``) keep their real defaults — override
    them with :func:`dataclasses.replace` on the returned bundle, which is also how an
    adapter opts out of conversation history entirely (``history=None``).
    """
    return PipelineDeps(
        cfg=cfg,
        ops=ops,
        dedup=dedup if dedup is not None else DedupStore(cfg.dedup_path),
        dispatcher=dispatcher if dispatcher is not None else DispatchClient(cfg),
        history=history if history is not None else HistoryStore(cfg.history_path),
    )


# --- re-dispatch context ------------------------------------------------------------


def rebuild_extra(deps: PipelineDeps, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the dispatch ``extra`` for a RE-DISPATCH off a persisted entry.

    Every re-dispatch call site (the drain's retry, startup reconcile of an entry that
    crashed before a run id existed) goes through here, so a replayed question carries
    what the live path assembled: a FRESH ``conversation_so_far`` (its follow-up referent
    may have landed since the first attempt), the reply context persisted at claim time
    (replayed, never re-resolved — the quoted message may be gone), this message's
    attachments re-downloaded from the persisted refs, and the newest prior image
    artifacts re-injected — all off the SAME single memoized capability probe the live
    path uses. No bytes are ever persisted, so re-downloading is the only way to recover
    them.

    Returns ``{}`` for a plain text-only entry with no history, leaving the re-dispatch
    payload byte-identical to a live text-only dispatch.
    """
    return build_extra(
        deps,
        entry,
        history_key=str(entry.get("history_key") or ""),
        reply_to=entry.get("reply_to"),
    )


# --- drain wiring -------------------------------------------------------------------


def _drain_dispatch(deps: PipelineDeps, message_id: str, entry: dict[str, Any]) -> None:
    """Drain callback: redispatch a run-id-less queued entry from its persisted question.

    The drain has already CAS-flipped the entry ``queued -> in_flight`` (stamping
    ``retried`` and incrementing ``attempts``) and does not touch it afterward, so this
    binding owns the outcome: rebuild the live payload, re-fire with the crash-safety
    persister, and settle exactly as the live path does — which re-parks a fresh
    retryable failure for the next pass, unchanged.
    """
    text = entry.get("text")
    if not text:
        # Nothing to replay and nothing to re-attach: leaving it queued would retry
        # forever. Terminal, so it leaves both the queued and the reconcile sets.
        logger.error("queued %s has no text to redispatch; abandoning", message_id)
        deps.dedup.update_status(message_id, "error")
        return

    result = _dispatch(deps.dispatcher, deps.dedup, message_id, text, rebuild_extra(deps, entry))

    # Re-read: the persister has just written the new run id, and the settlement (and
    # the channel post it drives) must see the entry as stored.
    current = deps.dedup.get(message_id) or dict(entry)
    _settle_terminal(deps, message_id, current, result)


def _drain_deliver(
    deps: PipelineDeps, message_id: str, entry: dict[str, Any], result_body: dict[str, Any]
) -> None:
    """Drain callback: deliver a re-attached run's COMPLETED result.

    Only ``completed`` bodies reach here (terminal failures are policy-routed inside the
    drain), so this is always the final answer. ``post_answer`` raising is NOT swallowed:
    the raise is what leaves the entry queued for the next cycle to re-attach and
    re-deliver, instead of dropping the answer. The drain marks the entry terminal only
    after this returns, so nothing here writes status.
    """
    current = deps.dedup.get(message_id) or dict(entry)
    # The raw worker status body omits the run id; graft the persisted one on so
    # artifact delivery and the history turn can address the run.
    result = {**result_body, "run_id": current.get("run_id")}

    deps.ops.post_answer(current, result)
    urls = _deliver_files(deps, message_id, current, result)
    _append_history(deps, current, result, urls)


def build_drain_callbacks(deps: PipelineDeps) -> DrainCallbacks:
    """Bind the drain's four callbacks over the ``ChannelOps`` implementation in ``deps``.

    :class:`~osprey.bridges.core.drain.DrainCallbacks` is deliberately a NARROWER surface
    than :class:`~osprey.bridges.core.ports.ChannelOps` — it is the drain's slice of the
    seam plus the store-independent channel steps (history append, file delivery) no
    single protocol member covers. Adapters implement ``ChannelOps`` only; this binding
    is the one place the two surfaces meet, so the failure contracts (``post_answer`` and
    ``post_giveup`` propagate; ``post_superseded`` is best-effort) hold identically for
    every channel.
    """
    return DrainCallbacks(
        dispatch=lambda mid, entry: _drain_dispatch(deps, mid, entry),
        deliver=lambda mid, entry, body: _drain_deliver(deps, mid, entry, body),
        give_up=lambda mid, entry: retry_queue.give_up(
            deps.ops, deps.dedup, mid, entry, history=deps.history
        ),
        supersede=lambda mid, entry: retry_queue.supersede_note(deps.ops, deps.dedup, mid, entry),
    )


def build_drain_deps(
    deps: PipelineDeps,
    *,
    gate: Callable[[], bool] | None = None,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> DrainDeps:
    """Assemble the :class:`~osprey.bridges.core.drain.DrainDeps` the drain loop runs on.

    The health gate defaults to :func:`~osprey.bridges.core.health_gate.gate_open` over
    the same ``cfg`` — dispatcher ``/health`` AND worker ``/health``, plus the alert-issue
    check only when it is configured. A facility with its own readiness signal injects
    ``gate`` instead; tests inject it (with ``now``/``sleep``) to drive the loop without
    network or real time.
    """
    return DrainDeps(
        cfg=deps.cfg,
        dedup=deps.dedup,
        dispatcher=deps.dispatcher,
        callbacks=build_drain_callbacks(deps),
        gate=gate if gate is not None else (lambda: gate_open(deps.cfg)),
        now=now,
        sleep=sleep,
    )


# --- the running bridge -------------------------------------------------------------


@dataclass
class BridgeRuntime:
    """A started engine: the collaborator bundle plus the live drain thread.

    Handed to the adapter's inbound loop, which calls :meth:`handle_event` per message,
    :meth:`supervise` whenever it wakes, and :meth:`shutdown` on exit.
    """

    deps: PipelineDeps
    drain: DrainDeps
    stop: threading.Event
    thread: threading.Thread

    def handle_event(self, event: Any) -> str:
        """Run one raw wire event through the pipeline. Returns ``"ignored"``,
        ``"duplicate"`` or ``"handled"`` for the adapter's logs."""
        return handle_event(event, self.deps)

    def supervise(self) -> threading.Thread:
        """Replace the drain thread if it died; otherwise a no-op returning the live one.

        Call it from the adapter's periodic wake — queued work must never stall silently
        because an unforeseen fault killed the loop harness. The replacement shares this
        runtime's stop event, so :meth:`shutdown` still stops it.
        """
        self.thread = ensure_alive(self.thread, lambda: run_drain_thread(self.drain, self.stop))
        return self.thread

    def shutdown(self) -> None:
        """Signal the drain thread to stop. It is a daemon, so this is about stopping
        work promptly, not about being allowed to exit; an in-progress pass finishes."""
        self.stop.set()


def start(
    cfg: CoreConfig,
    ops: ChannelOps,
    *,
    deps: PipelineDeps | None = None,
    gate: Callable[[], bool] | None = None,
    stop: threading.Event | None = None,
) -> BridgeRuntime:
    """Recover, start the drain, and return the runtime handle — in that order.

    Startup crash recovery runs FIRST and to completion: the entries it finishes are
    invisible to every other mechanism (the channel's own redelivery is defeated by the
    dedup claim, and the drain owns only ``queued`` entries), and a user still waiting on
    an answer should get it before new messages compete for the worker. Reconcile is
    given the pipeline's own context builder and settlement, so a result recovered at
    startup ships the same payload and settles the same way a live one does — a retryable
    failure discovered here is parked for the drain rather than reported as an error, and
    a delivered answer is recorded in conversation history.

    Then the single drain daemon thread starts. Only after both does the caller get a
    handle it could feed messages through.
    """
    bundle = deps if deps is not None else build_deps(cfg, ops)
    drain_deps = build_drain_deps(bundle, gate=gate)

    report = _recover(bundle)
    if report.total:
        logger.info(
            "startup reconcile: %d in-flight (reattached=%d redispatched=%d requeued=%d "
            "skipped=%d failed=%d post_failed=%d)",
            report.total,
            report.reattached,
            report.redispatched,
            report.requeued,
            report.skipped,
            report.failed,
            report.post_failed,
        )

    stop_event = stop if stop is not None else threading.Event()
    return BridgeRuntime(
        deps=bundle,
        drain=drain_deps,
        stop=stop_event,
        thread=run_drain_thread(drain_deps, stop_event),
    )


def _recover(deps: PipelineDeps) -> ReconcileReport:
    """Startup crash recovery with the pipeline's context builder and settlement."""
    return reconcile_inflight(
        deps,
        build_extra=lambda entry: rebuild_extra(deps, entry),
        settle=lambda mid, entry, result: _settle_terminal(deps, mid, dict(entry), dict(result)),
    )


def run_forever(
    cfg: CoreConfig,
    ops: ChannelOps,
    serve: Callable[[BridgeRuntime], None],
    *,
    deps: PipelineDeps | None = None,
    gate: Callable[[], bool] | None = None,
    stop: threading.Event | None = None,
) -> None:
    """The adapter entry point: start the engine, run the adapter's inbound loop, stop.

    ``serve`` is the adapter's own blocking ingestion loop. It receives the
    :class:`BridgeRuntime` and is expected to call :meth:`~BridgeRuntime.handle_event`
    per inbound message and :meth:`~BridgeRuntime.supervise` whenever it wakes. When it
    returns (or raises), the drain is signalled to stop.
    """
    runtime = start(cfg, ops, deps=deps, gate=gate, stop=stop)
    try:
        serve(runtime)
    finally:
        runtime.shutdown()
