"""The agent-record archive: an append-only copy of what a deployment recorded about its agent.

``osprey archive`` copies transcripts, dispatch run records, artifact stores,
plan-queue history, the audit ledger and completed days of telemetry into a
day-bucketed tree with a sha256 manifest. It never deletes or rewrites a copy.
"""

from osprey.services.archive.manifest import ArchiveState, load_state
from osprey.services.archive.run import (
    SOURCE_TABLE,
    ArchiveDestinationError,
    PassResult,
    run_pass,
)
from osprey.services.archive.telemetry_export import TelemetryExporter

__all__ = [
    "SOURCE_TABLE",
    "ArchiveDestinationError",
    "ArchiveState",
    "PassResult",
    "TelemetryExporter",
    "load_state",
    "run_pass",
]
