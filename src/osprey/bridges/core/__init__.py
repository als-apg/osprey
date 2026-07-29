"""Channel-agnostic core shared by OSPREY dispatch bridges.

The crash-safe local stores and shared configuration every channel needs. A channel
adapter composes this package and adds only its own wire format on top.

The engine has **no channel imports** (no chat/email specifics), **no imports of
osprey dispatch internals**, and **no agent SDK imports**. Keep it that way: the
isolation is what lets a bridge run as its own process against any deployment of the
dispatcher/worker pair.

Modules:
    store — lock-guarded JSON file store (persistence substrate)
    config — ``CoreConfig`` and the terminal-status set
    dedup — at-least-once claim/CAS-transition store
    history — per-conversation transcript replayed on each dispatch
"""

from .config import TERMINAL_STATUSES, CoreConfig
from .dedup import DedupStore
from .history import HistoryStore
from .store import JsonFileStore

__all__ = [
    "TERMINAL_STATUSES",
    "CoreConfig",
    "DedupStore",
    "HistoryStore",
    "JsonFileStore",
]
