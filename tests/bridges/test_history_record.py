"""Bridge history through the real record: spool writer, agent-record service, drain.

The bridge engine runs its real :func:`~osprey.bridges.core.pipeline.handle_event`
with a :class:`~osprey.bridges.core.history.ConversationHistory` that writes through
the spool writer and reads through a :class:`~osprey.agent_record.client.RecordClient`
logged in as the bridge, against an agent-record service running over an embedded
backend. A stand-in dispatch worker records each run the way the real one does: its
``run`` accept and terminal lines in its own spool and its session transcript under
its transcript root.

Pins that the second question sees the first answer, also when the run entries or
the transcript land only after the bridge wrote its turn (the read's drain commits
them first), and that a bridge sees only its own turns.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record import credentials as creds
from osprey.agent_record import identities as ids
from osprey.agent_record import service
from osprey.agent_record.backends.embedded import EmbeddedBackend
from osprey.agent_record.client import RecordClient
from osprey.bridges.core.config import CoreConfig
from osprey.bridges.core.dedup import DedupStore
from osprey.bridges.core.history import ConversationHistory
from osprey.bridges.core.pipeline import HISTORY_RECORD_KEY, PipelineDeps, handle_event
from osprey.bridges.core.ports import InboundEvent
from osprey.interfaces._serving import run_app_server
from tests.agent_record.test_identities import _config
from tests.bridges.conftest import RecordingChannelOps

TEAMS = "teams-bridge"
GCHAT = "gchat-bridge"
WORKER = "dispatch-worker-1"
PROJECT = "-data-workspace"
CONVERSATION = "room/AAA"
FIRST = "what is the orbit doing?"
ANSWER = "The orbit is stable."


def _token(login: str) -> str:
    return f"token-of-{login}-0123456789"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


@pytest.fixture
def config(spool_zone) -> dict:
    cfg = _config(workers=1, bridges=("teams_bridge", "gchat_bridge"))
    cfg["project_root"] = str(spool_zone.root.parent / "project")
    return cfg


@pytest.fixture
def mounts(tmp_path) -> Path:
    return tmp_path / "transcripts"


@pytest.fixture
def url(config, spool_zone, mounts, tmp_path) -> Iterator[str]:
    spool_zone.root.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        creds.verifier_line(login, _token(login)) + "\n" for login in ids.credential_table(config)
    )
    env = {
        creds.VERIFIER_ENV: _b64(lines),
        creds.SECRETS_ENV: _b64(json.dumps([])),
        service.SPOOL_ROOT_ENV: str(spool_zone.root),
        service.RECORD_DIR_ENV: str(tmp_path / "record"),
        service.TRANSCRIPTS_DIR_ENV: str(mounts),
    }
    with EmbeddedBackend(tmp_path / "record" / "record.duckdb") as backend:
        app = service.create_app(config=config, backend=backend, env=env)
        with run_app_server(app) as base:
            yield base


def _history(url: str, identity: str) -> ConversationHistory:
    login = ids.login_for(identity)
    return ConversationHistory(RecordClient(url, login, _token(login)))


class Worker:
    """The dispatch worker's side of a run: its ``run`` lines and its transcript.

    ``late`` names what lands only when :meth:`land` is called, after the bridge has
    written its turn: ``"run"`` (the accept and terminal lines) or ``"transcript"``.
    """

    def __init__(self, spool_root: Path, mounts: Path, config: dict, late: str | None):
        self.spool_root = spool_root
        self.root = mounts / next(n for n, i in ids.transcript_roots(config).items() if i == WORKER)
        self.late = late
        self.pending: list[Callable[[], None]] = []
        self.n = 0

    def run(self, question: str, text: str) -> str:
        self.n += 1
        run_id, session = f"run-{self.n}", str(uuid.uuid4())
        now = datetime.now(UTC)
        lines = [
            c.SpoolLine(
                entry_id=f"{run_id}-accept",
                kind=c.KIND_RUN,
                ts=now.isoformat(),
                body={"phase": c.RUN_ACCEPT, "run_id": run_id, "session_id": session},
                run_id=run_id,
                session_id=session,
            ),
            c.SpoolLine(
                entry_id=f"{run_id}-terminal",
                kind=c.KIND_RUN,
                ts=(now + timedelta(seconds=3)).isoformat(),
                body={"phase": c.RUN_TERMINAL, "status": "completed"},
                run_id=run_id,
                session_id=session,
            ),
        ]
        transcript = _event(session, "user", question, now + timedelta(seconds=1))
        transcript += _event(session, "assistant", text, now + timedelta(seconds=2))

        def write_run() -> None:
            directory = self.spool_root / WORKER
            directory.mkdir(parents=True, exist_ok=True)
            segment = c.SegmentName("worker", 7, self.n, sealed=True).name
            payload = "".join(line.to_json() + "\n" for line in lines)
            (directory / segment).write_text(payload, encoding="utf-8")

        def write_transcript() -> None:
            path = self.root / PROJECT / f"{session}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as handle:
                handle.write(transcript)

        for name, step in (("run", write_run), ("transcript", write_transcript)):
            if name == self.late:
                self.pending.append(step)
            else:
                step()
        return run_id

    def land(self) -> None:
        for step in self.pending:
            step()
        self.pending.clear()


def _event(session: str, role: str, text: str, when: datetime) -> bytes:
    content: Any = text if role == "user" else [{"type": "text", "text": text}]
    message: dict[str, Any] = {"role": role, "content": content}
    if role == "assistant":
        message["model"] = "claude-opus-5-5"
    line = {
        "parentUuid": None,
        "isSidechain": False,
        "sessionId": session,
        "type": role,
        "message": message,
        "uuid": str(uuid.uuid4()),
        "timestamp": when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    }
    return (json.dumps(line) + "\n").encode("utf-8")


class WorkerDispatcher:
    """A dispatch client whose runs the :class:`Worker` records; each answers *text*."""

    def __init__(self, worker: Worker) -> None:
        self.worker = worker
        self.calls: list[dict[str, Any]] = []

    def run(self, question: str, extra: dict | None = None, *, on_run_id=None) -> dict[str, Any]:
        self.calls.append({"question": question, "extra": extra})
        run_id = self.worker.run(question, ANSWER)
        if on_run_id is not None:
            on_run_id(run_id)
        return {"status": "completed", "text_output": ANSWER, "run_id": run_id}


def _event_for(n: int, text: str) -> InboundEvent:
    return InboundEvent(
        message_id=f"m{n}",
        text=text,
        sender_id="users/7",
        sender_display="Pat",
        history_key=CONVERSATION,
    )


@pytest.mark.parametrize("late", [None, "run", "transcript"])
def test_the_second_question_sees_the_first_answer(
    url, config, spool_zone, spool_for, mounts, tmp_path, late
):
    spool_for(TEAMS)
    worker = Worker(spool_zone.root, mounts, config, late)
    ops = RecordingChannelOps(parse_result=_event_for(1, FIRST))
    deps = PipelineDeps(
        cfg=CoreConfig(),
        ops=ops,
        dedup=DedupStore(str(tmp_path / "dedup.json")),
        dispatcher=WorkerDispatcher(worker),
        history=_history(url, TEAMS),
        probe_capability=lambda cfg, capability: True,
    )

    assert handle_event({}, deps) == "handled"
    # What the worker had not recorded yet lands only now, after the bridge's turn.
    worker.land()
    ops.parse_result = _event_for(2, "and yesterday?")
    assert handle_event({}, deps) == "handled"

    first, second = (call["extra"] for call in deps.dispatcher.calls)
    assert "conversation_so_far" not in first and HISTORY_RECORD_KEY not in first
    assert HISTORY_RECORD_KEY not in second, second.get(HISTORY_RECORD_KEY)
    (turn,) = second["conversation_so_far"]
    assert (turn["question"], turn["answer"], turn["run_id"]) == (FIRST, ANSWER, "run-1")
    assert turn["asked_by"] == {"id": "users/7", "name": "Pat"}
    turns = spool_zone.recorded_entries(c.KIND_BRIDGE_TURN)
    assert sorted(e.body["question"] for e in turns) == ["and yesterday?", FIRST]
    assert {e.identity for e in turns} == {TEAMS}


def test_a_bridge_sees_only_its_own_turns(url, config, spool_zone, spool_for, mounts):
    worker = Worker(spool_zone.root, mounts, config, None)
    run_id = worker.run(FIRST, ANSWER)
    teams, gchat = _history(url, TEAMS), _history(url, GCHAT)
    spool_for(TEAMS)
    teams.append(CONVERSATION, FIRST, run_id=run_id)
    spool_for(GCHAT)
    gchat.append(CONVERSATION, "a gchat question", run_id=run_id)
    gchat.append_failed("gchat-only", "another")

    seen = teams.read(CONVERSATION)
    assert seen.record is None, seen.reason
    assert [t["question"] for t in seen.turns] == [FIRST]
    assert teams.read("gchat-only").turns == []
    assert [t["question"] for t in gchat.read(CONVERSATION).turns] == ["a gchat question"]
