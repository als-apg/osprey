"""The agent record as a dispatch worker's login reads it, served in process.

:class:`WorkerRecord` stands in for the worker's
:class:`~osprey.agent_record.client.RecordClient`. Every read first ingests
what the test's spool zone holds (the worker's own ``run`` lines), then asks a
real :class:`~osprey.agent_record.reader.RecordReader` under the worker's
``Source`` scope, and decodes the answer exactly as the client decodes the
service's — so the worker's routes see what they would see from a deployed
record, with no service and no network. A run's answer is a transcript fact,
which no spool carries: :meth:`WorkerRecord.answer` puts one in the record for
a session, as the tailer would.
"""

from __future__ import annotations

import threading
from collections.abc import Collection
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from osprey.agent_record import contract as c
from osprey.agent_record.client import ClientAnswer, _decode
from osprey.agent_record.reader import RecordReader

__all__ = ["WorkerRecord", "worker_record"]


class WorkerRecord:
    """The record one dispatch worker reads, fed by a :class:`tests.conftest.SpoolZone`.

    Args:
        zone: The test's spool zone; the worker writes into its dispatch
            identity's directory.
        unavailable: When set, every read answers unavailable with this reason.
    """

    def __init__(self, zone: Any, *, unavailable: str | None = None) -> None:
        self.zone = zone
        self.identity = zone.dispatch
        self.unavailable = unavailable
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._turn = 0

    # The worker resolves its client per call; this double is its own client.
    def client(self, **_: Any) -> WorkerRecord:
        return self

    def calls_to(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def answer(self, session_id: str, text: str) -> None:
        """Record *text* as the final answer of *session_id*'s transcript."""
        with self._lock:
            self._turn += 1
            self.zone.recorded_entries()
            self.zone._backend.append(
                [
                    c.Entry(
                        key=f"cc:{self.identity}:{session_id}:u{self._turn}:0",
                        kind=c.KIND_ANSWER,
                        ts=datetime.now(UTC),
                        identity=self.identity,
                        body={"text": text},
                        session_id=session_id,
                    )
                ]
            )

    def _read(self, name: str, narrow_runs: Collection[str], **kwargs: Any) -> ClientAnswer:
        self.calls.append((name, {**kwargs, "narrow_runs": frozenset(narrow_runs)}))
        if self.unavailable is not None:
            return ClientAnswer.failed(self.unavailable)
        narrow = c.Narrow(run_ids=frozenset(narrow_runs)) if narrow_runs else None
        scope = c.Source(self.identity, narrow)
        with self._lock:
            self.zone.recorded_entries()
            reader = RecordReader(self.zone._backend, self.zone.config())
            args = {k: v for k, v in kwargs.items() if k != "wait"}
            answer = getattr(reader, name)(scope, **args)
        return _decode(httpx.Response(200, json=answer.to_json()))

    def run(
        self,
        run_id: str,
        *,
        wait: Any = None,
        narrow_session: str | None = None,  # noqa: ARG002 - the client's signature
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        return self._read("run", narrow_runs, run_id=run_id, wait=wait)

    def entries(
        self,
        kind: str,
        since: datetime | None = None,
        limit: int | None = None,
        *,
        source: str | None = None,
        wait: Any = None,
        narrow_session: str | None = None,  # noqa: ARG002 - the client's signature
        narrow_runs: Collection[str] = (),
    ) -> ClientAnswer:
        return self._read(
            "entries", narrow_runs, kind=kind, since=since, limit=limit, source=source, wait=wait
        )


@pytest.fixture
def worker_record(monkeypatch, spool_zone, spool_for) -> WorkerRecord:
    """The dispatch worker writes as the zone's dispatch identity and reads this record."""
    from osprey.mcp_server.dispatch_worker import dispatch_api

    spool_for(spool_zone.dispatch)
    record = WorkerRecord(spool_zone)
    monkeypatch.setattr(dispatch_api, "record_client", record.client)
    monkeypatch.setattr(dispatch_api, "_settled", dispatch_api.OrderedDict())
    return record
