"""Channel-agnostic core shared by OSPREY dispatch bridges.

A thin HTTP client of the OSPREY dispatcher/worker pair, plus the crash-safe local
stores and the ordering-critical message pipeline every channel needs. A channel
adapter composes this package with a :class:`ChannelOps` implementation and adds
only its own wire format on top.

The engine talks to the pair over HTTP only — ``POST {dispatcher}/webhook/{trigger}``,
``GET {dispatcher}/dispatch/{dispatch_id}``, ``GET {worker}/dispatch/{run_id}``, the
worker artifact byte route, and ``GET /health`` on both. It has **no channel imports**
(no chat/email specifics), **no imports of osprey dispatch internals**, and **no agent
SDK imports**. Keep it that way: the isolation is what lets a bridge run as its own
process against any deployment of the pair.

Layers:
    Promoted primitives
        store — lock-guarded JSON file store (persistence substrate)
        config — ``CoreConfig`` and the terminal-status set
        dedup — at-least-once claim/CAS-transition store
        history — per-conversation transcript replayed on each dispatch
        retry — pure, fail-closed retry policy
        capabilities — fail-closed peer capability probe
        health_gate — drain gate over dispatcher/worker health
        artifacts — worker artifact fetch and normalization
        dispatch_client — three-step dispatch handshake
        drain — supervised retry-queue drain thread

    Hoisted pipeline
        ports — the ``ChannelOps`` seam adapters implement
        pipeline — ``handle_event``, the per-message ordering specification
        retry_queue — park / give-up / supersede-at-enqueue
        reconcile — startup crash recovery for in-flight messages
        runtime — collaborator wiring and drain supervision
"""

from .artifacts import (
    KNOWN_EXTENSIONS,
    MAX_DELIVERED_DOC_BYTES,
    MAX_DELIVERED_IMAGE_BYTES,
    PNG_MAGIC,
    FetchedArtifact,
    artifact_descriptors,
    artifact_ids,
    bare_mime,
    ext_for_mime,
    fetch_artifact,
    safe_label,
)
from .capabilities import pair_supports
from .config import TERMINAL_STATUSES, CoreConfig
from .dedup import DedupStore
from .dispatch_client import DispatchClient, DispatchPipelineError
from .drain import DrainCallbacks, DrainDeps, drain_once, ensure_alive, run_drain_thread
from .health_gate import gate_open
from .history import HistoryStore
from .pipeline import PipelineDeps, handle_event
from .ports import (
    RESERVED_ENTRY_KEYS,
    ChannelOps,
    InboundEvent,
    InputDownload,
    ReplyContext,
)
from .reconcile import ReconcileDeps, ReconcileReport, deliver_terminal, reconcile_inflight
from .retry import coalesce_key, give_up_due, is_eligible, is_retryable, may_redispatch
from .retry_queue import SUPERSEDED_STATUS, give_up, park, supersede_at_enqueue, supersede_note
from .runtime import BridgeRuntime, build_deps, build_drain_deps, run_forever, start
from .store import JsonFileStore

__all__ = [
    # config + persistence substrate
    "TERMINAL_STATUSES",
    "CoreConfig",
    "DedupStore",
    "HistoryStore",
    "JsonFileStore",
    # retry policy
    "coalesce_key",
    "give_up_due",
    "is_eligible",
    "is_retryable",
    "may_redispatch",
    # the pair over HTTP
    "KNOWN_EXTENSIONS",
    "MAX_DELIVERED_DOC_BYTES",
    "MAX_DELIVERED_IMAGE_BYTES",
    "PNG_MAGIC",
    "DispatchClient",
    "DispatchPipelineError",
    "FetchedArtifact",
    "artifact_descriptors",
    "bare_mime",
    "artifact_ids",
    "ext_for_mime",
    "fetch_artifact",
    "gate_open",
    "pair_supports",
    "safe_label",
    # the seam adapters implement
    "RESERVED_ENTRY_KEYS",
    "ChannelOps",
    "InboundEvent",
    "InputDownload",
    "ReplyContext",
    # engine entry points
    "SUPERSEDED_STATUS",
    "BridgeRuntime",
    "DrainCallbacks",
    "DrainDeps",
    "PipelineDeps",
    "ReconcileDeps",
    "ReconcileReport",
    "build_deps",
    "build_drain_deps",
    "deliver_terminal",
    "drain_once",
    "ensure_alive",
    "give_up",
    "handle_event",
    "park",
    "reconcile_inflight",
    "run_drain_thread",
    "run_forever",
    "start",
    "supersede_at_enqueue",
    "supersede_note",
]
