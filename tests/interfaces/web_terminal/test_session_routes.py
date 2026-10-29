"""Tests for session-related routes.

The session routes read the agent record through ``routes.session.record_client``,
which these tests replace with a fake: what each route asks the record for
(the session, a drain naming it, the subagent), and how it renders a complete,
an incomplete and an unavailable answer.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.agent_record.client import ClientAnswer
from osprey.agent_record.contract import Drain
from osprey.interfaces.web_terminal.routes import session as session_routes
from tests.agent_record.fakes import FakeRecordClient
from tests.interfaces.web_terminal._fakes import FakePtySession, PoolChatSession
from tests.interfaces.web_terminal._started_app import terminal_client

SESSION_KEY = "11111111-2222-3333-4444-555555555555"
CLEARED_TRANSCRIPT = "99999999-8888-7777-6666-555555555555"


@pytest.fixture
def client(workspace_dir):
    with terminal_client(workspace_dir) as c:
        yield c


@pytest.fixture
def record(monkeypatch):
    """A fake record the routes read; tests set ``record.answers``."""
    fake = FakeRecordClient()
    monkeypatch.setattr(session_routes, "record_client", lambda: fake)
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    return fake


def _row(session_id, identity, prompt, last_ts, entries):
    return {
        "session_id": session_id,
        "persons": [identity],
        "identities": [identity],
        "first_ts": last_ts,
        "last_ts": last_ts,
        "entries": entries,
        "prompt": prompt,
        "run_ids": [],
        "terminal_session": "",
    }


class TestListSessionsEndpoint:
    def test_returns_this_terminals_sessions(self, client, record):
        """GET /api/sessions lists the terminal identity's sessions, newest first."""
        client.app.state.terminal_user = "alice"
        record.answers["list_sessions"] = ClientAnswer(
            data=[
                _row(
                    "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                    "alice",
                    "Help me tune the beam",
                    "2026-02-17T10:00:00+00:00",
                    42,
                ),
                _row(
                    "bbbbbbbb-0000-0000-0000-000000000000",
                    "bob",
                    "Not mine",
                    "2026-02-17T09:00:00+00:00",
                    3,
                ),
                _row(SESSION_KEY, "alice", "Read BPM values", "2026-02-16T08:00:00+00:00", 10),
            ],
            complete=True,
        )
        resp = client.get("/api/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert [s["session_id"] for s in data["sessions"]] == [
            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            SESSION_KEY,
        ]
        assert data["sessions"][0] == {
            "session_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "first_message": "Help me tune the beam",
            "last_modified": "2026-02-17T10:00:00+00:00",
            "message_count": 42,
        }
        assert data["record"] == "complete"
        assert record.calls_to("list_sessions")[0]["wait"] == Drain()

    def test_an_unavailable_record_is_not_an_empty_list(self, client, monkeypatch):
        monkeypatch.setattr(
            session_routes, "record_client", lambda: FakeRecordClient(unavailable="down")
        )
        data = client.get("/api/sessions").json()
        assert data["sessions"] == []
        assert data["record"] == "unavailable"
        assert data["record_reason"] == "down"


class TestRestartEndpoint:
    def test_restart_empties_both_registries(self, client):
        """POST /api/terminal/restart kills the pooled PTY and the chat child alike."""
        app = client.app
        registry = app.state.pty_registry
        pty = FakePtySession()
        registry._sessions["k"] = pty

        operator_registry = app.state.operator_registry
        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession", PoolChatSession
        ):
            chat, _ = client.portal.call(operator_registry.get_or_create_chat_session, "c", "/tmp")

        resp = client.post("/api/terminal/restart")

        assert resp.status_code == 200
        assert registry.get_session("k") is None
        assert not pty.is_alive
        assert operator_registry.get_chat_session("c") is None
        assert chat.stop_calls == 1


class TestSessionScopedDiagnostics:
    """The diagnostics routes read the session (or the key's transcript) from the record."""

    _CURRENT = (
        "osprey.mcp_server.workspace.transcript_reader.TranscriptReader.find_current_transcript"
    )

    @staticmethod
    def _call_nodes():
        return [
            {
                "kind": "tool_call",
                "key": "cc:alice:abc-123:a1:0",
                "ts": "2026-02-19T12:00:00+00:00",
                "body": {"id": "t1", "name": "mcp__controls__channel_read", "input": {}},
                "result": {"ts": "2026-02-19T12:00:01+00:00", "body": {"content": "1.0"}},
            }
        ]

    def test_session_agents_with_session_id(self, client, record):
        """GET /api/session-agents?session_id=<id> drains and reads that session."""
        record.answers["timeline"] = ClientAnswer(data=self._call_nodes(), complete=True)
        resp = client.get("/api/session-agents?session_id=abc-123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tool_calls_by_agent"]["main"][0]["tool"] == "channel_read"
        assert data["record"] == "complete"
        (call,) = record.calls_to("timeline")
        assert call["session_id"] == "abc-123"
        assert call["wait"] == Drain(session_id="abc-123")
        assert call["narrow_session"] is None

    def test_without_session_id_the_current_transcript_names_the_session(
        self, client, record, tmp_path
    ):
        """With no session_id the routes read the session of the current transcript."""
        with patch(self._CURRENT, return_value=tmp_path / "cur-1.jsonl"):
            assert client.get("/api/session-agents").status_code == 200
            assert client.get("/api/session-chat").status_code == 200
        assert [c["session_id"] for c in record.calls_to("timeline")] == ["cur-1", "cur-1"]

    def test_without_any_session_nothing_is_read(self, client, record):
        with patch(self._CURRENT, return_value=None):
            data = client.get("/api/session-log").json()
        assert data["events"] == [] and data["record"] == "complete"
        assert record.calls_to("timeline") == []

    @pytest.mark.parametrize(
        "path",
        [
            "/api/session-agents",
            "/api/session-log",
            "/api/session-chat",
            "/api/session-agent-timeline?agent_id=x&",
        ],
    )
    def test_an_unavailable_record_says_so(self, client, monkeypatch, path):
        monkeypatch.setattr(
            session_routes, "record_client", lambda: FakeRecordClient(unavailable="down")
        )
        sep = "" if path.endswith("&") else "?"
        data = client.get(f"{path}{sep}session_id=abc-123").json()
        assert data["record"] == "unavailable"
        assert data["record_reason"] == "down"

    def test_an_incomplete_read_is_marked_incomplete(self, client, record):
        record.answers["timeline"] = ClientAnswer(
            data=self._call_nodes(), complete=False, reason="wait mark not passed"
        )
        data = client.get("/api/session-log?session_id=abc-123").json()
        assert data["total_events"] == 1
        assert data["record"] == "incomplete"
        assert data["record_reason"] == "wait mark not passed"

    def test_session_log_since_compares_instants(self, client, record):
        record.answers["timeline"] = ClientAnswer(data=self._call_nodes(), complete=True)
        data = client.get("/api/session-log?session_id=abc-123&since=2026-02-19T12:00:00Z").json()
        assert data["total_events"] == 1

    def test_session_agent_timeline_reads_the_subagent(self, client, record):
        record.answers["agent_timeline"] = ClientAnswer(
            data=[
                {"kind": "prompt", "key": "cc:a:s:p:0", "ts": "t0", "body": {"text": "Plot"}},
                {"kind": "answer", "key": "cc:a:s:q:0", "ts": "t1", "body": {"text": "Done"}},
            ],
            complete=True,
        )
        data = client.get(
            "/api/session-agent-timeline?agent_id=agent-xyz&session_id=abc-123"
        ).json()
        assert [st["kind"] for st in data["timeline"]] == ["prompt", "reasoning"]
        assert data["count"] == 2
        (call,) = record.calls_to("agent_timeline")
        assert (call["session_id"], call["agent_id"]) == ("abc-123", "agent-xyz")
        assert call["wait"] == Drain(session_id="abc-123")

    @staticmethod
    def _point_key_at_cleared_transcript(client):
        """Map SESSION_KEY to the transcript a ``/clear`` moved it to."""
        client.app.state.transcript_map = {SESSION_KEY: CLEARED_TRANSCRIPT}
        client.app.state.transcript_map_provisional = False

    @pytest.mark.parametrize(
        ("path", "read"),
        [
            ("/api/session-agents?", "timeline"),
            ("/api/session-log?", "timeline"),
            ("/api/session-chat?", "timeline"),
            ("/api/session-agent-timeline?agent_id=agent-xyz&", "agent_timeline"),
        ],
    )
    def test_a_route_reads_the_transcript_the_key_points_at(self, client, record, path, read):
        """A session key is mapped to its current transcript before the record is read."""
        self._point_key_at_cleared_transcript(client)
        resp = client.get(f"{path}session_id={SESSION_KEY}")

        assert resp.status_code == 200
        (call,) = record.calls_to(read)
        assert call["session_id"] == CLEARED_TRANSCRIPT
