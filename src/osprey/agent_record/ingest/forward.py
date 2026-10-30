"""The transcript forwarder: a host session's transcripts into the stack's record, by the spool.

Claude Code writes a host session's transcripts readable by the account that
runs it alone, so the stack's agent-record, which runs as the image's user,
never reads them. While the stack owns the record, the host session forwards
them instead: the same :class:`~osprey.agent_record.ingest.tailer.TranscriptTailer`
the host-local record runs, over the same root and relpath prefix, appends into
a :class:`SpoolForward` rather than into a record.

Invariants:

- Every entry keeps the transcript key the tailer gives it
  (:func:`~osprey.agent_record.contract.transcript_key`), and goes to the host
  identity's own spool directory as a forwarded transcript line
  (:func:`~osprey.agent_record.contract.forwarded_transcript`). The record
  ingests it under the transcript route; an entry the host-local record already
  filed under that key is a duplicate there, so a takeover files nothing twice.
- One append is one sealed, fsynced segment
  (:func:`~osprey.agent_record.writer.write_batch`): a batch is on disk, whole,
  before the tailer commits the position past it, and a failed write commits
  nothing.
- The committed positions are the forwarder's own, kept in one JSON file
  (:class:`ForwardState`) and never in the record's state namespace. A file
  read again from an older position only resends entries the record drops as
  duplicates.
- The forwarder knows no arrival day, so the retention floor is the
  configured window before now.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from osprey.agent_record import writer
from osprey.agent_record.contract import (
    STATE_TIME_FIELD,
    WRITER_ERRORS_FILE,
    AppendResult,
    BackendHealth,
    Entry,
    RecordBackendError,
    decode_state,
    encode_state,
    spool_line_of,
    state_time,
)
from osprey_connectors.safety_file import atomic_replace

__all__ = ["FORWARD_BACKEND", "ForwardState", "SpoolForward"]

#: The name a forwarder's health gives its backend.
FORWARD_BACKEND = "spool-forward"

#: Mode of the state file: the host identity's group shares the state directory.
_STATE_FILE_MODE = 0o664


class ForwardState:
    """The forwarder's committed positions: one JSON object of state name to value.

    Read once when built; every :meth:`put_state` writes the whole file again,
    atomically, so a reader sees the old file or the new one. One forwarder at
    a time writes it: the caller holds the forward lock.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._states: dict[str, Mapping[str, Any]] = {}
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RecordBackendError(f"the forward state {self.path} is unreadable: {exc}") from exc
        try:
            stored = json.loads(text)
        except ValueError as exc:
            raise RecordBackendError(f"the forward state {self.path} is unreadable: {exc}") from exc
        if not isinstance(stored, dict):
            raise RecordBackendError(f"the forward state {self.path} is not a JSON object")
        for name, value in stored.items():
            self._states[name] = decode_state(name, json.dumps(value), self.path)

    def get_state(self, name: str) -> Mapping[str, Any] | None:
        return self._states.get(name)

    def count_state(self, prefix: str, since: datetime | None = None) -> int:
        """How many positions are named ``<prefix>…``, stamped at or after *since* if given."""
        stamp = state_time(since) if since is not None else None
        return sum(
            1
            for name, value in self._states.items()
            if name.startswith(prefix)
            and (stamp is None or str(value.get(STATE_TIME_FIELD) or "") >= stamp)
        )

    def put_state(self, name: str, value: Mapping[str, Any]) -> None:
        encode_state(name, value)
        states = {**self._states, name: dict(value)}
        data = json.dumps(states, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            atomic_replace(self.path, data, mode=_STATE_FILE_MODE)
        except OSError as exc:
            raise RecordBackendError(
                f"the forward state {self.path} was not written: {exc}"
            ) from exc
        self._states = states


class SpoolForward:
    """A transcript sink that forwards each append into one spool directory.

    Args:
        spool_dir: The host identity's own spool directory.
        state: Where the forwarded positions are kept.
    """

    def __init__(self, spool_dir: Path | str, state: ForwardState) -> None:
        self.spool_dir = Path(spool_dir)
        self.state = state

    def append(self, entries: Sequence[Entry]) -> AppendResult:
        """Write *entries* as one sealed segment; every one counts as appended.

        Raises:
            RecordBackendError: If the segment was not written.
        """
        lines = [spool_line_of(entry, uuid.uuid4().hex) for entry in entries]
        if writer.write_batch(self.spool_dir, lines) is None:
            raise RecordBackendError(
                f"the forwarded transcript entries were not written to {self.spool_dir}; "
                f"see its {WRITER_ERRORS_FILE}"
            )
        return AppendResult(appended=len(lines))

    def get_state(self, name: str) -> Mapping[str, Any] | None:
        return self.state.get_state(name)

    def put_state(self, name: str, value: Mapping[str, Any]) -> None:
        self.state.put_state(name, value)

    def count_state(self, prefix: str, since: datetime | None = None) -> int:
        return self.state.count_state(prefix, since)

    def health(self) -> BackendHealth:
        """Not ok: a forwarder holds no record, so it knows no arrival day."""
        return BackendHealth(
            ok=False, backend=FORWARD_BACKEND, detail="a forwarder holds no record"
        )
