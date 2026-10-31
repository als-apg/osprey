"""Builders shared by the contract cases: entries, a reader, and the two ingest paths.

Cases that pin what ingest stores (identity, ``on_behalf_of``, run refusals,
secrets, transcript marks) feed the backend through the real
:class:`~osprey.agent_record.ingest.spool.SpoolConsumer` and
:class:`~osprey.agent_record.ingest.tailer.TranscriptTailer`; every other case
appends entries directly.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from itertools import count
from pathlib import Path
from typing import Any

import duckdb

from osprey.agent_record import contract as c
from osprey.agent_record import reader as rd
from osprey.agent_record.ingest.spool import SpoolConsumer
from osprey.agent_record.ingest.tailer import TranscriptTailer
from osprey.agent_record.service import IngestDrainer
from tests.agent_record.test_identities import _config

T0 = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
NOW = T0 + timedelta(hours=1)

ALICE = "alice"
BOB = "bob"
W1 = "dispatch-worker-1"
W2 = "dispatch-worker-2"
BRIDGE = "teams-bridge"
QS = "queueserver"
SIDECAR = "sidecar"

#: Terminals alice and bob, two isolated dispatch workers, one bridge, one
#: queueserver lane and the host identity ``builder``.
CONFIG = _config(workers=2, bridges=("teams_bridge",), lanes=True)

#: The transcript root name of each source, as agent-record mounts them.
ROOTS = {
    ALICE: f"{ALICE}-claude-projects",
    BOB: f"{BOB}-claude-projects",
    W1: "dispatch_claude_projects_1",
    W2: "dispatch_claude_projects_2",
}

PROJECT = "-data-workspace"

_ids = count(1)


def at(seconds: float, base: datetime = T0) -> datetime:
    return base + timedelta(seconds=seconds)


# --- entries appended directly ---------------------------------------------------------


def entry(
    identity: str,
    key: str,
    kind: str,
    t: float,
    body: dict | None = None,
    *,
    arrived: float | None = None,
    **fields: Any,
) -> c.Entry:
    return c.Entry(
        key=key,
        kind=kind,
        ts=at(t),
        identity=identity,
        body=body if body is not None else {},
        inserted_at=at(t if arrived is None else arrived),
        **fields,
    )


def cc(identity: str, session: str, n: int, kind: str, t: float, body=None, **fields):
    """A transcript entry of *session*, keyed as the parser keys line ``u<n>`` block 0."""
    return entry(
        identity, f"cc:{identity}:{session}:u{n}:0", kind, t, body, session_id=session, **fields
    )


def accept(
    identity: str,
    run_id: str,
    session: str,
    requester: str | None,
    t: float,
    *,
    arrived: float | None = None,
) -> c.Entry:
    body: dict = {"phase": c.RUN_ACCEPT, "run_id": run_id, "session_id": session}
    if requester:
        body["requester"] = requester
    return entry(
        identity,
        c.run_key(run_id, c.RUN_ACCEPT),
        c.KIND_RUN,
        t,
        body,
        arrived=arrived,
        session_id=session,
        run_id=run_id,
    )


def terminal(identity: str, run_id: str, t: float) -> c.Entry:
    body: dict = {"phase": c.RUN_TERMINAL, "status": "done"}
    return entry(identity, c.run_key(run_id, c.RUN_TERMINAL), c.KIND_RUN, t, body, run_id=run_id)


def turn(
    n: int, run_id: str, t: float, *, conversation: str = "c1", asked_by: str = "", question=""
) -> c.Entry:
    body = {"conversation": conversation, "run_id": run_id, "question": question or f"q{n}"}
    if asked_by:
        body["asked_by"] = asked_by
    return entry(BRIDGE, f"sp:{BRIDGE}:t{n}", c.KIND_BRIDGE_TURN, t, body, run_id=run_id)


# --- reading ------------------------------------------------------------------------------


class Clock:
    """A monotonic clock that moves only when the reader sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def reader(
    backend,
    *,
    now: datetime | None = None,
    clock: Clock | None = None,
    drain: rd.Drainer | None = None,
) -> rd.RecordReader:
    clock = clock or Clock()
    return rd.RecordReader(
        backend,
        CONFIG,
        clock=clock,
        sleep=clock.sleep,
        now=(lambda: now) if now is not None else None,
        drain=drain,
    )


def drainer(backend, spool_root: Path, mounts: Path) -> IngestDrainer:
    """The service's drainer over *spool_root* and every transcript root under *mounts*;
    it wakes nothing (a case runs ingest itself)."""
    consumer = SpoolConsumer(backend, spool_root, config=CONFIG)
    tailer = TranscriptTailer(backend, {name: mounts / name for name in ROOTS.values()}, CONFIG)
    return IngestDrainer(CONFIG, consumer, tailer, lambda: None)


def rows(backend) -> list[c.Entry]:
    return backend.select(c.ScopeFilter(), c.Query())


def flat(nodes) -> list[str]:
    out: list[str] = []
    for node in nodes:
        out.extend(i.key for i in node.walk())
    return out


def session_ids(answer: rd.ReadAnswer) -> set[str]:
    return {r["session_id"] for r in answer.data}


def export_rows(backend, day: date, dest: Path) -> list[dict[str, Any]]:
    """The rows *backend* exports for arrival day *day*, read back from the Parquet file."""
    written = backend.export_parquet(day, dest)
    con = duckdb.connect()
    try:
        result = con.execute("SELECT * FROM read_parquet(?)", [str(dest)])
        names = [d[0] for d in result.description]
        out = [dict(zip(names, row, strict=True)) for row in result.fetchall()]
    finally:
        con.close()
    assert written == len(out)
    return out


# --- the spool path -------------------------------------------------------------------------


def spool_line(kind: str, body: dict | None = None, *, when: datetime, **ids: Any) -> c.SpoolLine:
    ids.setdefault("entry_id", f"e{next(_ids)}")
    return c.SpoolLine(kind=kind, ts=when.isoformat(), body=body or {"what": "x"}, **ids)


def spool(root: Path, identity: str, *lines: c.SpoolLine) -> Path:
    """One sealed segment of *lines* in *identity*'s spool directory."""
    directory = root / identity
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / c.SegmentName("h", 7, next(_ids), sealed=True).name
    path.write_bytes(b"".join((line.to_json() + "\n").encode() for line in lines))
    return path


def consume(backend, root: Path, *, secrets=()) -> Any:
    return SpoolConsumer(backend, root, config=CONFIG, secrets=secrets).consume()


# --- the transcript path ----------------------------------------------------------------------


def stamp(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def user_line(session: str, text: str, *, uuid: str, when: datetime) -> bytes:
    return _line(
        {
            "parentUuid": None,
            "isSidechain": False,
            "sessionId": session,
            "type": "user",
            "message": {"role": "user", "content": text},
            "uuid": uuid,
            "timestamp": stamp(when),
        }
    )


def assistant_line(session: str, blocks: list[dict], *, uuid: str, when: datetime) -> bytes:
    return _line(
        {
            "parentUuid": None,
            "isSidechain": False,
            "sessionId": session,
            "type": "assistant",
            "message": {"role": "assistant", "model": "claude-opus-5-5", "content": blocks},
            "uuid": uuid,
            "timestamp": stamp(when),
        }
    )


def text_line(session: str, text: str, *, uuid: str, when: datetime) -> bytes:
    return assistant_line(session, [{"type": "text", "text": text}], uuid=uuid, when=when)


def relpath(session: str) -> str:
    """A session's main transcript, relative to ``projects/``."""
    return f"{PROJECT}/{session}.jsonl"


def write_transcript(mounts: Path, identity: str, session: str, data: bytes) -> Path:
    """Append *data* to *identity*'s transcript of *session* under *mounts*."""
    path = mounts / ROOTS[identity] / relpath(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as handle:
        handle.write(data)
    return path


def tail(backend, mounts: Path, *, secret_set=(), now: datetime | None = None) -> Any:
    """One tailer poll over every transcript root that exists under *mounts*."""
    roots = {name: mounts / name for name in ROOTS.values() if (mounts / name).is_dir()}
    report = TranscriptTailer(backend, roots, CONFIG, secret_set=secret_set).poll_once(now)
    assert not report.errors, report.errors
    return report
