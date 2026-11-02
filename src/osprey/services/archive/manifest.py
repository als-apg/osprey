"""The append-only ``MANIFEST.jsonl`` of each archive day, and the state folded from them.

Every day directory of the archive holds one ``MANIFEST.jsonl``, appended to and
never rewritten. Four line kinds:

- ``file`` — one copied or exported file: ``source``, ``path`` (relative to the
  archive root), ``size``, ``sha256``, ``source_mtime_ns``, ``archived_at``; a
  file exported from the agent record also carries its ``day`` and ``rows``;
- ``agent_record_day`` — one exported arrival day of the agent record: ``day``
  and ``rows`` (``0`` for a day with no entries, which has no file). It is the
  last line of that day's export, in that arrival day's own manifest; a day
  without it is not exported;
- ``agent_record_purged`` — the record's archive of one arrival day was purged:
  ``day``. It clears that day's earlier ``agent_record_day`` lines;
- ``pass`` — the last line of every pass: ``started_at``, ``completed_at``,
  ``files``, ``bytes``, ``errors``, ``skipped``.

The archive keeps no other state. What a writer needs to know — the last copy
of each source, which agent-record days are already exported — is folded from
the manifests in day order, line by line, so the latest line about a day wins
and a re-run is correct because nothing is ever deleted or rewritten. A line of
any other kind is ignored.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_NAME = "MANIFEST.jsonl"

#: The line kind that marks one arrival day of the agent record exported.
AGENT_RECORD_DAY = "agent_record_day"

#: The line kind that clears a day's :data:`AGENT_RECORD_DAY` mark.
AGENT_RECORD_PURGED = "agent_record_purged"

#: A day directory's name: the UTC date its pass started.
DAY_DIR_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class SourceRecord:
    """The last archived copy of one source file."""

    size: int
    source_mtime_ns: int | None
    sha256: str


@dataclass
class ArchiveState:
    """What the manifests of an archive say, folded in day order."""

    sources: dict[str, SourceRecord] = field(default_factory=dict)
    agent_record_days: set[str] = field(default_factory=set)
    last_pass: dict[str, Any] | None = None


def day_dirs(dest: Path) -> list[Path]:
    """The day directories under *dest*, oldest first."""
    try:
        entries = list(dest.iterdir())
    except OSError:
        return []
    return sorted(p for p in entries if DAY_DIR_PATTERN.match(p.name) and p.is_dir())


def _read_lines(manifest: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping an unreadable line %d of %s", lineno, manifest)
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def load_state(dest: Path) -> ArchiveState:
    """Fold every ``<dest>/<day>/MANIFEST.jsonl`` in day order.

    A line that does not parse (a crash mid-append leaves a torn last line) is
    skipped with a warning rather than failing the fold.

    Raises:
        OSError: If a manifest exists but cannot be read.
    """
    state = ArchiveState()
    for day in day_dirs(dest):
        manifest = day / MANIFEST_NAME
        if not manifest.is_file():
            continue
        for record in _read_lines(manifest):
            kind = record.get("kind")
            if kind == "file" and isinstance(record.get("source"), str):
                state.sources[record["source"]] = SourceRecord(
                    size=int(record.get("size", -1)),
                    source_mtime_ns=record.get("source_mtime_ns"),
                    sha256=str(record.get("sha256", "")),
                )
            elif kind == AGENT_RECORD_DAY and isinstance(record.get("day"), str):
                state.agent_record_days.add(record["day"])
            elif kind == AGENT_RECORD_PURGED and isinstance(record.get("day"), str):
                state.agent_record_days.discard(record["day"])
            elif kind == "pass":
                state.last_pass = record
    return state


def append_record(manifest: Path, record: dict[str, Any], *, file_mode: int) -> bool:
    """Append one line to *manifest*, creating it at *file_mode* when absent.

    Returns whether the file was created, so the caller can hand it to the
    archive root's owner.
    """
    created = not manifest.exists()
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(manifest, os.O_WRONLY | os.O_APPEND | os.O_CREAT, file_mode)
    try:
        if created:
            os.fchmod(fd, file_mode)
        data = line.encode("utf-8")
        while data:
            data = data[os.write(fd, data) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    return created
