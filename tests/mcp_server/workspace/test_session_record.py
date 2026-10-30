"""The session readers read the agent record: drained, paged, scoped by the login.

Pins how :mod:`osprey.mcp_server.session` reads a session — a drain naming the
session on the first page, the cursor after it, ``narrow_session`` only inside a
dispatch worker, an unavailable or partial answer never shown as a complete
empty one — and how timeline nodes become the session log's events, the
conversation's turns and a subagent's steps. Through a running agent-record
service: a step the session just took is in the next ``session_log`` and
handoff read, and another person's session reads empty.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from osprey.agent_record.client import ClientAnswer, RecordClient
from osprey.agent_record.contract import Drain
from osprey.mcp_server import session as s
from tests.agent_record.fakes import FakeRecordClient
from tests.mcp_server.conftest import get_tool_fn

SID = "sess-record-1"


def _page(nodes, *, nxt=None, complete=True, reason=""):
    return ClientAnswer(data=nodes, complete=complete, next=nxt, reason=reason)


def _node(kind, key, ts, body, **extra):
    node = {"kind": kind, "key": key, "ts": ts, "body": body, "agent_id": "", "tool_use_id": ""}
    node.update(extra)
    return node


# --- the read ----------------------------------------------------------------------


class TestReadTimeline:
    def test_first_page_drains_the_session_and_later_pages_follow_the_cursor(self, monkeypatch):
        monkeypatch.delenv(s.DISPATCH_RUN_ENV, raising=False)
        fake = FakeRecordClient(
            {"timeline": [_page([{"kind": "prompt"}], nxt="c1"), _page([{"kind": "answer"}])]}
        )
        read = s.read_timeline(SID, client=fake)
        calls = fake.calls_to("timeline")
        assert calls[0]["wait"] == Drain(session_id=SID)
        assert calls[0]["after"] is None
        assert calls[1]["after"] == "c1" and calls[1]["wait"] is None
        assert all(c["narrow_session"] is None for c in calls)
        assert read.record == s.RECORD_COMPLETE
        assert [n["kind"] for n in read.nodes] == ["prompt", "answer"]

    def test_inside_a_dispatch_worker_every_page_is_narrowed_to_the_session(self, monkeypatch):
        monkeypatch.setenv(s.DISPATCH_RUN_ENV, "1")
        fake = FakeRecordClient({"timeline": [_page([], nxt="c1"), _page([])]})
        s.read_timeline(SID, client=fake)
        assert [c["narrow_session"] for c in fake.calls_to("timeline")] == [SID, SID]

    def test_an_agent_read_uses_the_agent_timeline(self, monkeypatch):
        monkeypatch.delenv(s.DISPATCH_RUN_ENV, raising=False)
        fake = FakeRecordClient({"agent_timeline": _page([{"kind": "prompt"}])})
        read = s.read_timeline(SID, agent_id="abc", client=fake)
        (call,) = fake.calls_to("agent_timeline")
        assert (call["session_id"], call["agent_id"]) == (SID, "abc")
        assert call["wait"] == Drain(session_id=SID)
        assert read.nodes == [{"kind": "prompt"}]

    def test_an_unavailable_record_reads_unavailable(self):
        read = s.read_timeline(SID, client=FakeRecordClient(unavailable="connection refused"))
        assert read.record == s.RECORD_UNAVAILABLE
        assert read.reason == "connection refused"
        assert not read.available

    def test_a_partial_answer_reads_incomplete(self):
        fake = FakeRecordClient(
            {"timeline": _page([{"kind": "prompt"}], complete=False, reason="wait mark not passed")}
        )
        read = s.read_timeline(SID, client=fake)
        assert read.record == s.RECORD_INCOMPLETE
        assert "wait mark" in read.reason
        assert read.nodes == [{"kind": "prompt"}]

    def test_a_later_page_that_fails_keeps_what_was_read_as_incomplete(self):
        fake = FakeRecordClient(
            {"timeline": [_page([{"kind": "prompt"}], nxt="c1"), ClientAnswer.failed("gone")]}
        )
        read = s.read_timeline(SID, client=fake)
        assert read.record == s.RECORD_INCOMPLETE
        assert read.nodes == [{"kind": "prompt"}]

    def test_more_pages_than_the_bound_read_incomplete(self, monkeypatch):
        monkeypatch.setattr(s, "MAX_TIMELINE_PAGES", 2)
        fake = FakeRecordClient({"timeline": _page([], nxt="again")})
        assert s.read_timeline(SID, client=fake).record == s.RECORD_INCOMPLETE


class TestOwnSessions:
    def test_rows_are_kept_for_the_identity_only(self):
        rows = [
            {"session_id": "a", "identities": ["alice"], "persons": ["alice"]},
            {"session_id": "b", "identities": ["bob"], "persons": ["bob"]},
            {"session_id": "d", "identities": ["dispatch-worker-1"], "persons": ["alice"]},
        ]
        fake = FakeRecordClient({"list_sessions": ClientAnswer(data=rows, complete=True)})
        kept, answer = s.own_sessions("alice", client=fake)
        assert [r["session_id"] for r in kept] == ["a", "d"]
        assert answer.complete
        assert fake.calls_to("list_sessions")[0]["wait"] == Drain()

    def test_unavailable_lists_nothing_and_says_so(self):
        kept, answer = s.own_sessions("alice", client=FakeRecordClient(unavailable="down"))
        assert kept == [] and not answer.available


class TestCurrentSessionId:
    def test_the_harness_id_comes_first(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", "forced")
        assert s.current_session_id(tmp_path) == "forced"

    def test_else_the_newest_local_transcript(self, monkeypatch, tmp_path):
        monkeypatch.delenv("OSPREY_TELEMETRY_SESSION_ID", raising=False)
        monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
        with patch(
            "osprey.mcp_server.workspace.transcript_reader.TranscriptReader.find_current_transcript",
            return_value=tmp_path / "abc-123.jsonl",
        ):
            assert s.current_session_id(tmp_path) == "abc-123"
        assert s.current_session_id(None) is None


# --- the conversions ---------------------------------------------------------------


def _subagent_session() -> list[dict]:
    call = _node(
        "tool_call",
        "cc:alice:S:a1:0",
        "2026-02-19T12:00:01+00:00",
        {"id": "t-mcp", "name": "mcp__osprey_workspace__archiver_read", "input": {"pv": "X"}},
        result={
            "kind": "tool_result",
            "ts": "2026-02-19T12:00:02+00:00",
            "body": {"content": json.dumps({"error": True, "message": "nope"})},
        },
    )
    inner = _node(
        "tool_call",
        "cc:alice:S:b1:0",
        "2026-02-19T12:00:04+00:00",
        {"id": "t-in", "name": "mcp__controls__channel_read", "input": {}},
        agent_id="abc",
        result={
            "kind": "tool_result",
            "ts": "2026-02-19T12:00:05+00:00",
            "body": {"content": "1.0"},
        },
    )
    agent = _node(
        "tool_call",
        "cc:alice:S:a2:0",
        "2026-02-19T12:00:03+00:00",
        {"id": "t-agent", "name": "Agent", "input": {"subagent_type": "data-visualizer"}},
        result={"kind": "subagent", "ts": "2026-02-19T12:00:06+00:00", "body": {"agent_id": "abc"}},
        subagent={"agent_id": "abc", "timeline": [inner]},
    )
    builtin = _node("tool_call", "cc:alice:S:a3:0", "2026-02-19T12:00:07+00:00", {"name": "Bash"})
    return [call, agent, builtin]


class TestSessionEvents:
    def test_mcp_calls_and_subagents_become_the_logs_events(self):
        events = s.session_events(_subagent_session(), "S")
        assert [(e["type"], e.get("tool") or e.get("agent_type")) for e in events] == [
            ("tool_call", "archiver_read"),
            ("agent_start", "data-visualizer"),
            ("tool_call", "channel_read"),
            ("agent_stop", "data-visualizer"),
        ]
        first, start, inner, stop = events
        assert first["is_error"] is True, "an OSPREY error envelope is an error"
        assert first["server"] == "osprey_workspace" and first["session_id"] == "S"
        assert first["agent_id"] is None
        assert inner["agent_id"] == start["agent_id"] == stop["agent_id"] == "abc"
        assert stop["timestamp"] == "2026-02-19T12:00:06+00:00"

    def test_a_subagent_whose_call_is_missing_is_still_bracketed(self):
        inner = _subagent_session()[1]["subagent"]["timeline"]
        orphan = {
            "kind": "subagent",
            "orphan": True,
            "subagent": {"agent_id": "abc", "timeline": inner},
        }
        events = s.session_events([orphan], "S")
        assert [e["type"] for e in events] == ["agent_start", "tool_call", "agent_stop"]

    def test_agent_steps_follow_the_subagents_stream(self):
        prompt = _node("prompt", "cc:a:S:p:0", "t0", {"text": "Plot X"})
        think = _node("thinking", "cc:a:S:q:0", "t1", {"text": "hmm"})
        words = _node("answer", "cc:a:S:q:1", "t1", {"text": "Reading X"})
        call = _node(
            "tool_call",
            "cc:a:S:q:2",
            "t1",
            {"id": "t9", "name": "mcp__c__read", "input": {"pv": "X"}},
            result={"ts": "t2", "body": {"content": "y" * 3000, "is_error": True}},
        )
        later = _node("prompt", "cc:a:S:r:0", "t3", {"text": "not the task prompt"})
        steps = s.agent_steps([prompt, think, words, call, later])
        assert [st["kind"] for st in steps] == [
            "prompt",
            "thinking",
            "reasoning",
            "tool_call",
            "tool_result",
        ]
        result = steps[-1]
        assert result["tool_use_id"] == "t9" and result["is_error"] is True
        assert len(result["text"]) == s.MAX_STEP_RESULT_LENGTH

    def test_chat_history_is_one_turn_per_line_with_words(self):
        nodes = [
            _node("prompt", "cc:a:S:u1:0", "t0", {"text": "Hello"}),
            _node("prompt", "cc:a:S:m1:0", "t1", {"text": "<command-name>/model"}),
            _node("prompt", "cc:a:S:x1:0", "t1", {"text": "meta", "is_meta": True}),
            _node("thinking", "cc:a:S:a1:0", "t2", {"text": "private"}),
            _node("answer", "cc:a:S:a1:1", "t2", {"text": "Hi"}),
            _node("answer", "cc:a:S:a1:2", "t2", {"text": "there"}),
            _node("answer", "cc:a:S:s1:0", "t3", {"text": "sub"}, agent_id="abc"),
        ]
        assert s.chat_history(nodes) == [
            {"role": "user", "content": "Hello", "timestamp": "t0"},
            {"role": "assistant", "content": "Hi\nthere", "timestamp": "t2"},
        ]


# --- through the service -------------------------------------------------------------

from tests.agent_record.test_drain import (  # noqa: E402,F401
    PROJECT,
    _cfg,
    _env,
    _transcript,
    slow_ingest,
)
from tests.agent_record.test_service import _fresh_writer, _token  # noqa: E402,F401

NOW = datetime.now(UTC).replace(microsecond=0)


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _user(session: str, uuid: str, content, when: datetime) -> bytes:
    stamp = when.isoformat().replace("+00:00", "Z")
    return _line(
        {
            "type": "user",
            "sessionId": session,
            "uuid": uuid,
            "timestamp": stamp,
            "message": {"role": "user", "content": content},
        }
    )


def _assistant(session: str, uuid: str, blocks: list, when: datetime) -> bytes:
    stamp = when.isoformat().replace("+00:00", "Z")
    return _line(
        {
            "type": "assistant",
            "sessionId": session,
            "uuid": uuid,
            "timestamp": stamp,
            "message": {"role": "assistant", "content": blocks},
        }
    )


def _step(session: str, n: int, when: datetime) -> bytes:
    """One MCP tool call and its result, numbered *n*."""
    call = {
        "type": "tool_use",
        "id": f"toolu_{n}",
        "name": f"mcp__osprey_workspace__step_{n}",
        "input": {},
    }
    result = {"type": "tool_result", "tool_use_id": f"toolu_{n}", "content": f"done {n}"}
    return _assistant(session, f"a-{n}", [call], when) + _user(
        session, f"r-{n}", [result], when + timedelta(milliseconds=500)
    )


@pytest.fixture
def record(tmp_path):
    from osprey.agent_record import service
    from osprey.agent_record.backends.embedded import EmbeddedBackend
    from osprey.interfaces._serving import run_app_server

    config = _cfg(tmp_path)
    with EmbeddedBackend(tmp_path / "record.duckdb") as backend:
        app = service.create_app(config=config, backend=backend, env=_env(tmp_path, config))
        with run_app_server(app) as url:
            yield url


@pytest.mark.usefixtures("slow_ingest")
async def test_a_step_is_in_the_next_session_log_and_handoff_read(tmp_path, record, monkeypatch):
    from osprey.mcp_server.workspace.tools.session_log import session_log

    alice = RecordClient(record, "alice", _token("alice"))
    monkeypatch.setattr(s, "record_client", lambda: alice)
    monkeypatch.setenv("OSPREY_TELEMETRY_SESSION_ID", SID)
    monkeypatch.delenv(s.DISPATCH_RUN_ENV, raising=False)
    root = "alice-claude-projects"
    _transcript(tmp_path, root, SID, _user(SID, "u-0", "Read X", NOW - timedelta(seconds=9)))
    _transcript(tmp_path, root, SID, _step(SID, 1, NOW - timedelta(seconds=8)))
    fn = get_tool_fn(session_log)

    with patch(
        "osprey.mcp_server.workspace.tools.session_log.deployed_render_dir",
        return_value=tmp_path,
    ):
        first = json.loads(await fn())
        assert first["record"] == "complete", first
        assert [e["tool"] for e in first["events"]] == ["step_1"]

        _transcript(tmp_path, root, SID, _step(SID, 2, NOW - timedelta(seconds=4)))
        _transcript(
            tmp_path,
            root,
            SID,
            _assistant(SID, "a-done", [{"type": "text", "text": "X is 1.0"}], NOW),
        )
        second = json.loads(await fn())

    assert second["record"] == "complete", second
    assert [e["tool"] for e in second["events"]] == ["step_1", "step_2"]
    handoff = s.read_timeline(SID, client=alice)
    assert handoff.record == s.RECORD_COMPLETE
    assert s.chat_history(handoff.nodes, max_chars=None)[-1]["content"] == "X is 1.0"


@pytest.mark.usefixtures("slow_ingest")
def test_another_persons_session_reads_empty(tmp_path, record, monkeypatch):
    monkeypatch.delenv(s.DISPATCH_RUN_ENV, raising=False)
    bobs = "sess-bob-1"
    _transcript(tmp_path, "bob-claude-projects", bobs, _user(bobs, "u-b", "bob's secret", NOW))
    _transcript(tmp_path, "alice-claude-projects", SID, _user(SID, "u-a", "mine", NOW))
    bob = RecordClient(record, "bob", _token("bob"))
    alice = RecordClient(record, "alice", _token("alice"))

    assert "bob's secret" in json.dumps(s.read_timeline(bobs, client=bob).nodes)
    read = s.read_timeline(bobs, client=alice)
    assert read.record == s.RECORD_COMPLETE
    assert read.nodes == []
    rows, _ = s.own_sessions("alice", client=alice)
    assert [r["session_id"] for r in rows] == [SID]
