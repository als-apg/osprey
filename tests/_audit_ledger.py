"""Read back what the audit writer filed, for suites that assert on the ledger.

The writer keeps one JSONL file per surface under a directory per identity, so
a suite reads ``<zone>/<acting identity>/<surface>.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

from osprey_connectors.identity import acting_identity


def ledger_records(zone: Path, surface: str) -> list[dict]:
    """Every record the acting identity filed on *surface* under *zone*, oldest first.

    ``[]`` when nothing was filed (the writer creates the file on first record).
    """
    path = zone / acting_identity() / f"{surface}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
