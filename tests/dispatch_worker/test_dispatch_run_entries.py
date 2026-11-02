"""The dispatch API records each run as two ``run`` entries in its own spool.

``run:<run_id>:accept`` is written when the run is accepted, before the agent
starts, and names the session minted for the run and who asked for it.
``run:<run_id>:terminal`` is written exactly once, by whichever path ends the
run, and holds the run's outcome fields and none of its text. These tests drive
the dispatch API with its agent call stubbed, write as a dispatch worker into
the test's own spool root, and read the entries back through the spool
consumer.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest

from osprey.agent_record import contract as c
from osprey.agent_runner import ResultEvent, TextEvent, ThinkingEvent
from osprey.bridges.core.config import CoreConfig
from osprey.bridges.core.dispatch_client import DispatchClient
from osprey.mcp_server.dispatch_worker import dispatch_api, sdk_runner

#: The terminal entry's body keys: its phase, its run, and the outcome fields.
TERMINAL_KEYS = frozenset({c.RUN_PHASE_FIELD, "run_id", *dispatch_api.RUN_OUTCOME_FIELDS})

#: The status-body fields a run's terminal entry does not hold, and why: the
#: answer is the transcript's text, and the tool-call count is derived from the
#: run's timeline at read time.
NOT_IN_THE_TERMINAL_ENTRY = frozenset({"text_output", "num_tool_calls"})

_CANNED: dict[str, Any] = {
    "status": "completed",
    "text_output": "the answer",
    "tool_calls": [],
    "error": None,
    "duration_sec": 0.25,
    "cost_usd": 0.03,
    "num_turns": 2,
}


#: Every ``(kind, body)`` the dispatch API handed the writer in this test. The
#: record keeps the first line of a run key, so a second terminal write would
#: not show in the read-back; the writer calls are where it would.
_WRITES: list[tuple[str, dict[str, Any]]] = []


@pytest.fixture(autouse=True)
def _worker(monkeypatch, tmp_path, spool_for, spool_zone, worker_record):  # noqa: ARG001 - fixtures
    """A dispatch worker with empty run stores, writing into the test's spool and reading it."""
    spool_for(spool_zone.dispatch)
    real_write = dispatch_api.record_writer.write
    _WRITES.clear()

    def _counted(kind, body, **ids):
        _WRITES.append((kind, dict(body)))
        return real_write(kind, body, **ids)

    monkeypatch.setattr(dispatch_api.record_writer, "write", _counted)
    monkeypatch.setattr(dispatch_api, "_runs", {})
    monkeypatch.setattr(dispatch_api, "_queues", {})
    monkeypatch.setattr(dispatch_api, "_tasks", {})
    monkeypatch.setattr(dispatch_api, "_run_input_seam", {})
    monkeypatch.setenv("OSPREY_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.setattr(dispatch_api, "describe_run_artifacts", lambda run_id: [])
    monkeypatch.setattr(dispatch_api, "describe_run_input_artifacts", lambda run_id: [])


def _request(**fields: Any) -> dispatch_api.DispatchRequest:
    return dispatch_api.DispatchRequest(prompt="why did BPM 3 drift?", allowed_tools=[], **fields)


def _run_entries(recorded_entries, run_id: str, phase: str) -> list[c.Entry]:
    return [e for e in recorded_entries(c.KIND_RUN) if e.key == c.run_key(run_id, phase)]


async def _dispatch(request: dispatch_api.DispatchRequest) -> str:
    """Accept *request* the way the route does and return its run id."""
    response = await dispatch_api.dispatch(request)
    return response.run_id


async def _finish(run_id: str) -> None:
    """Wait for a run's task that ends on its own."""
    task = dispatch_api._tasks.get(run_id)
    if task is not None:
        await task


# ---------------------------------------------------------------------------
# The accept entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_accept_is_recorded_before_the_agent_starts(monkeypatch, recorded_entries):
    """The accept is in the spool when the agent is called, and names the
    session handed to the agent, the requester and the accept time."""
    seen: dict[str, Any] = {}

    async def _agent(**kw):
        seen["session_id"] = kw["session_id"]
        seen["accepts"] = _run_entries(recorded_entries, kw["run_id"], c.RUN_ACCEPT)
        return dict(_CANNED)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _agent)
    run_id = await _dispatch(_request(owner="alice"))
    await _finish(run_id)

    (accept,) = seen["accepts"]
    record = dispatch_api._runs[run_id]
    assert accept.kind == c.KIND_RUN
    assert accept.identity == "dispatch-worker-1"
    assert accept.run_id == run_id
    assert accept.session_id == seen["session_id"] == record["session_id"]
    assert dict(accept.body) == {
        c.RUN_PHASE_FIELD: c.RUN_ACCEPT,
        "run_id": run_id,
        c.RUN_SESSION_FIELD: seen["session_id"],
        "requester": "alice",
        "created_at": record["created_at"],
    }
    assert c.accept_session(accept) == seen["session_id"]


@pytest.mark.asyncio
async def test_a_fire_nobody_asked_for_has_no_requester(monkeypatch, recorded_entries):
    monkeypatch.setattr(sdk_runner, "run_dispatch", lambda **kw: _canned())
    run_id = await _dispatch(_request())
    await _finish(run_id)

    (accept,) = _run_entries(recorded_entries, run_id, c.RUN_ACCEPT)
    assert accept.body["requester"] is None


@pytest.mark.asyncio
async def test_a_refused_owner_is_no_requester(monkeypatch, recorded_entries):
    """The requester is the owner after ``owner_from_header``'s guard."""
    monkeypatch.setattr(sdk_runner, "run_dispatch", lambda **kw: _canned())
    run_id = await _dispatch(_request(owner="<script>"))
    await _finish(run_id)

    (accept,) = _run_entries(recorded_entries, run_id, c.RUN_ACCEPT)
    assert accept.body["requester"] is None


async def _canned() -> dict[str, Any]:
    return dict(_CANNED)


# ---------------------------------------------------------------------------
# The session the accept names is the transcript's
# ---------------------------------------------------------------------------


@pytest.fixture
def real_runner(monkeypatch, tmp_path):
    """The real ``run_dispatch`` over a stand-in agent CLI.

    The stand-in streams the given events and files a transcript the way the
    CLI does: one JSONL file named for the session the SDK option forced, whose
    lines carry that ``sessionId``. Returns ``(set_events, transcripts_dir)``.
    """
    monkeypatch.setattr(
        "osprey.agent_runner.artifact_resolve.deployed_agent_data_root",
        lambda: tmp_path / "agent-data",
    )
    monkeypatch.setattr("osprey.agent_runner.clean_env.build_clean_env", lambda **kw: {})
    monkeypatch.setattr(
        "osprey.agent_runner.sdk_context.build_system_prompt", lambda *a, **k: "system"
    )
    monkeypatch.setattr("osprey_connectors.config.get_facility_timezone", lambda *a, **k: "UTC")
    transcripts = tmp_path / "projects" / "render"
    events: list[Any] = []

    async def fake_stream(project_dir, prompt, **kw):  # noqa: ARG001 - matches the stream_query signature
        transcripts.mkdir(parents=True, exist_ok=True)
        session = kw["session_id"]
        line = {"type": "user", "sessionId": session, "message": {"content": prompt}}
        (transcripts / f"{session}.jsonl").write_text(json.dumps(line) + "\n")
        for event in events:
            yield event

    monkeypatch.setattr(sdk_runner, "stream_query", fake_stream)
    return events.extend, transcripts


def _result_event(**overrides: Any) -> ResultEvent:
    fields: dict[str, Any] = {
        "subtype": "success",
        "is_error": False,
        "num_turns": 3,
        "duration_ms": 0,
        "session_id": "s",
        "total_cost_usd": 0.07,
        "usage": None,
        "result": None,
        "api_error_status": None,
    }
    fields.update(overrides)
    return ResultEvent(**fields)


@pytest.mark.asyncio
async def test_the_accepts_session_is_the_transcripts_session_id(real_runner, recorded_entries):
    set_events, transcripts = real_runner
    set_events([TextEvent(text="fine", parent_tool_use_id=None), _result_event()])

    run_id = await _dispatch(_request(owner="alice"))
    await _finish(run_id)

    (transcript,) = transcripts.glob("*.jsonl")
    transcript_session = json.loads(transcript.read_text().splitlines()[0])["sessionId"]
    (accept,) = _run_entries(recorded_entries, run_id, c.RUN_ACCEPT)
    assert c.accept_session(accept) == transcript_session
    assert accept.session_id == transcript_session
    assert dispatch_api._runs[run_id]["session_id"] == transcript_session


# ---------------------------------------------------------------------------
# The terminal entry: one per run, on every path
# ---------------------------------------------------------------------------


async def _completed(monkeypatch) -> str:
    monkeypatch.setattr(sdk_runner, "run_dispatch", lambda **kw: _canned())
    run_id = await _dispatch(_request(owner="alice"))
    await _finish(run_id)
    return run_id


async def _agent_error(monkeypatch) -> str:
    async def _agent(**kw):
        return {
            **_CANNED,
            "status": "error",
            "text_output": "",
            "error": "max turns",
            "failure_class": "run",
            "num_tool_calls": 0,
            "session_id": kw["session_id"],
        }

    monkeypatch.setattr(sdk_runner, "run_dispatch", _agent)
    run_id = await _dispatch(_request())
    await _finish(run_id)
    return run_id


async def _timeout(monkeypatch) -> str:
    async def _slow(**kw):
        await asyncio.sleep(5)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _slow)
    monkeypatch.setattr(dispatch_api, "DISPATCH_TIMEOUT_SEC", 0)
    run_id = await _dispatch(_request())
    await _finish(run_id)
    return run_id


async def _worker_fault(monkeypatch) -> str:
    async def _boom(**kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(sdk_runner, "run_dispatch", _boom)
    run_id = await _dispatch(_request())
    await _finish(run_id)
    return run_id


async def _user_cancel(monkeypatch) -> str:
    started = asyncio.Event()

    async def _hang(**kw):
        started.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _hang)
    run_id = await _dispatch(_request())
    await started.wait()
    task = dispatch_api._tasks[run_id]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return run_id


async def _stale_sweep(monkeypatch) -> str:
    """The sweep ends the run, then the cancelled task unwinds and writes nothing."""
    started = asyncio.Event()

    async def _hang(**kw):
        started.set()
        await asyncio.sleep(30)

    monkeypatch.setattr(sdk_runner, "run_dispatch", _hang)
    run_id = await _dispatch(_request(owner="alice"))
    await started.wait()
    task = dispatch_api._tasks[run_id]
    dispatch_api._runs[run_id]["created_at"] = time.time() - (
        dispatch_api.DISPATCH_TIMEOUT_SEC + 90
    )
    dispatch_api._sweep_stale_runs()
    with pytest.raises(asyncio.CancelledError):
        await task
    return run_id


_PATHS = {
    "completed": (_completed, "completed", False),
    "agent-error": (_agent_error, "error", False),
    "timeout": (_timeout, "error", False),
    "worker-fault": (_worker_fault, "error", False),
    "user-cancel": (_user_cancel, "error", True),
    "stale-sweep": (_stale_sweep, "error", False),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(_PATHS))
async def test_every_terminal_path_records_exactly_one_terminal_entry(
    path, monkeypatch, recorded_entries
):
    drive, status, cancelled = _PATHS[path]
    run_id = await drive(monkeypatch)

    (terminal,) = _run_entries(recorded_entries, run_id, c.RUN_TERMINAL)
    (accept,) = _run_entries(recorded_entries, run_id, c.RUN_ACCEPT)
    record = dispatch_api._runs[run_id]
    phases = [
        body[c.RUN_PHASE_FIELD]
        for kind, body in _WRITES
        if kind == c.KIND_RUN and body.get("run_id") == run_id
    ]
    assert phases == [c.RUN_ACCEPT, c.RUN_TERMINAL]

    assert terminal.run_id == run_id
    assert terminal.identity == "dispatch-worker-1"
    assert set(terminal.body) == TERMINAL_KEYS
    assert terminal.body[c.RUN_PHASE_FIELD] == c.RUN_TERMINAL
    assert terminal.body["status"] == status == record["status"]
    assert terminal.body["cancelled"] is cancelled
    for field in dispatch_api.RUN_OUTCOME_FIELDS:
        expected = record.get(field)
        if field in ("artifacts", "input_artifacts"):
            expected = expected or []
        elif field == "cancelled":
            expected = bool(expected)
        assert terminal.body[field] == expected, field
    if status == "error":
        assert terminal.body["failure_class"]
        assert terminal.body["error"]
    # The terminal entry names no session: its run's accept does, and every
    # outcome (the error outcomes included) carries that same session.
    assert not terminal.session_id
    assert record["session_id"] == c.accept_session(accept)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(_PATHS))
async def test_the_terminal_entry_holds_no_text(path, monkeypatch, recorded_entries):
    """No prompt, answer, tool payload or thinking: those are the transcript's."""
    drive, _, _ = _PATHS[path]
    run_id = await drive(monkeypatch)

    (terminal,) = _run_entries(recorded_entries, run_id, c.RUN_TERMINAL)
    for key in ("prompt", "text_output", "tool_calls", "thinking", "stderr", "session_id"):
        assert key not in terminal.body
    assert "why did BPM 3 drift?" not in json.dumps(dict(terminal.body))
    assert "the answer" not in json.dumps(dict(terminal.body))


@pytest.mark.asyncio
async def test_no_run_record_file_is_written(monkeypatch, tmp_path):
    """The record holds the run; the worker keeps no file of its own about it."""
    await _completed(monkeypatch)

    assert list((tmp_path / "project").rglob("*.json")) == []


# ---------------------------------------------------------------------------
# The field set, pinned against the bridge's dispatch client
# ---------------------------------------------------------------------------


class _ReadKeys(dict):
    """A status body that notes every key its reader asks for."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read: set[str] = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)


def _status_body_keys_the_client_reads(body: dict[str, Any]) -> set[str]:
    """The keys ``DispatchClient.poll_worker`` reads from a terminal status body."""
    recording = _ReadKeys(body)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    with httpx.Client(transport=transport) as http:
        client = DispatchClient(CoreConfig(worker_url="http://worker"), http, sleep=lambda s: None)
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(httpx.Response, "json", lambda self, **kw: recording)
            client.poll_worker("r1")
    return recording.read


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["completed", "agent-error", "user-cancel"])
async def test_the_terminal_entry_holds_every_status_field_the_client_reads(
    path, monkeypatch, recorded_entries
):
    """Every field the dispatch client takes from the status body is in the
    terminal entry with the status body's value, except the answer text and
    the derived tool-call count."""
    drive, _, _ = _PATHS[path]
    run_id = await drive(monkeypatch)
    body = await dispatch_api.get_dispatch_result(run_id)
    (terminal,) = _run_entries(recorded_entries, run_id, c.RUN_TERMINAL)

    read = _status_body_keys_the_client_reads(dict(body))
    assert "status" in read and "failure_class" in read, read
    assert read - NOT_IN_THE_TERMINAL_ENTRY <= set(terminal.body), (
        read - NOT_IN_THE_TERMINAL_ENTRY - set(terminal.body)
    )
    for key in read - NOT_IN_THE_TERMINAL_ENTRY:
        expected = body.get(key)
        if key in ("artifacts", "input_artifacts"):
            expected = expected or []
        assert terminal.body[key] == expected, key


# ---------------------------------------------------------------------------
# Thinking reaches neither the status body nor the run entries
# ---------------------------------------------------------------------------


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(k) for k in value} | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


@pytest.mark.asyncio
async def test_no_thinking_reaches_the_status_body_or_the_run_entries(
    real_runner, recorded_entries, worker_record
):
    set_events, _ = real_runner
    set_events(
        [
            ThinkingEvent(text="secret chain of thought"),
            TextEvent(text="the answer", parent_tool_use_id=None),
            _result_event(),
        ]
    )

    run_id = await _dispatch(_request(owner="alice"))
    await _finish(run_id)
    # The answer is the transcript's fact; the record holds it as the tailer would.
    worker_record.answer(dispatch_api._runs[run_id]["session_id"], "the answer")
    body = await dispatch_api.get_dispatch_result(run_id)
    dashboard = await dispatch_api.dashboard_runs()

    assert body["status"] == "completed"
    assert body["text_output"] == "the answer"
    for shown in (body, dashboard, *(dict(e.body) for e in recorded_entries(c.KIND_RUN))):
        assert not any("thinking" in k.lower() for k in _keys(shown))
        assert "secret chain of thought" not in json.dumps(shown, default=str)
