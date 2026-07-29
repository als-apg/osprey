"""Channel-agnostic core shared by OSPREY dispatch bridges.

A thin HTTP client of the OSPREY dispatcher/worker pair, plus the crash-safe local
stores every channel needs. A channel adapter composes this package and adds only its
own wire format on top.

The engine talks to the pair over HTTP only — ``POST {dispatcher}/webhook/{trigger}``,
``GET {dispatcher}/dispatch/{dispatch_id}``, ``GET {worker}/dispatch/{run_id}``, the
worker artifact byte route, and ``GET /health`` on both. It has **no channel imports**
(no chat/email specifics), **no imports of osprey dispatch internals**, and **no agent
SDK imports**. Keep it that way: the isolation is what lets a bridge run as its own
process against any deployment of the pair.

Modules:
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
"""

from .artifacts import (
    MAX_ARTIFACT_BYTES,
    MAX_DOC_BYTES,
    PNG_MAGIC,
    artifact_ids,
    ext_for_mime,
    fetch_artifact,
    safe_label,
    split_artifacts,
)
from .capabilities import pair_supports
from .config import TERMINAL_STATUSES, CoreConfig
from .dedup import DedupStore
from .dispatch_client import DispatchClient, DispatchError
from .drain import DrainCallbacks, DrainDeps, drain_once, ensure_alive, run_drain_thread
from .health_gate import gate_open
from .history import HistoryStore
from .retry import coalesce_key, give_up_due, is_eligible, is_retryable, may_redispatch
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
    "MAX_ARTIFACT_BYTES",
    "MAX_DOC_BYTES",
    "PNG_MAGIC",
    "DispatchClient",
    "DispatchError",
    "artifact_ids",
    "ext_for_mime",
    "fetch_artifact",
    "gate_open",
    "pair_supports",
    "safe_label",
    "split_artifacts",
    # retry-queue drain
    "DrainCallbacks",
    "DrainDeps",
    "drain_once",
    "ensure_alive",
    "run_drain_thread",
]
