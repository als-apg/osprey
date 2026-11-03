"""The agent-record archive: an append-only copy of what a deployment recorded about its agent.

``osprey archive`` copies artifact stores and plan-queue history into a
day-bucketed tree with a sha256 manifest, beside the agent record's own daily
export. It never deletes or rewrites a copy.
"""

from osprey.services.archive.manifest import ArchiveState, load_state
from osprey.services.archive.run import (
    SOURCE_TABLE,
    ArchiveDestinationError,
    PassResult,
    run_pass,
)

__all__ = [
    "SOURCE_TABLE",
    "ArchiveDestinationError",
    "ArchiveState",
    "PassResult",
    "load_state",
    "run_pass",
]
