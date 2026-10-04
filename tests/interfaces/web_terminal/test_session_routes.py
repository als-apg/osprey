"""Tests for session-related routes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.session_discovery import SessionInfo
from tests.interfaces.web_terminal._fakes import FakePtySession, PoolChatSession

SESSION_KEY = "11111111-2222-3333-4444-555555555555"
CLEARED_TRANSCRIPT = "99999999-8888-7777-6666-555555555555"


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def client(workspace_dir):
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


class TestListSessionsEndpoint:
    def test_returns_session_list(self, client):
        """GET /api/sessions returns session metadata."""
        mock_sessions = [
            SessionInfo(
                session_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                first_message="Help me tune the beam",
                last_modified=datetime(2026, 2, 17, 10, 0, 0, tzinfo=UTC),
                message_count=42,
            ),
            SessionInfo(
                session_id="11111111-2222-3333-4444-555555555555",
                first_message="Read BPM values",
                last_modified=datetime(2026, 2, 16, 8, 0, 0, tzinfo=UTC),
                message_count=10,
            ),
        ]

        with patch(
            "osprey.interfaces.web_terminal.routes.session.SessionDiscovery.list_sessions",
            return_value=mock_sessions,
        ):
            resp = client.get("/api/sessions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 2
        assert data["sessions"][0]["session_id"] == "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        assert data["sessions"][0]["first_message"] == "Help me tune the beam"
        assert data["sessions"][0]["message_count"] == 42
        assert data["sessions"][1]["session_id"] == "11111111-2222-3333-4444-555555555555"


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
    """Verify session diagnostics endpoints accept ?session_id= param."""

    # TranscriptReader is imported inside the endpoint functions (lazy import),
    # so we patch at its canonical module location.
    _TR = "osprey.mcp_server.workspace.transcript_reader.TranscriptReader"

    def test_session_agents_with_session_id(self, client):
        """GET /api/session-agents?session_id=<id> uses read_session_by_id."""
        mock_events = [
            {
                "type": "tool_call",
                "tool": "channel_read",
                "agent_id": None,
                "timestamp": "2026-02-19T12:00:00Z",
            },
        ]
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_session_by_id.return_value = mock_events
            resp = client.get("/api/session-agents?session_id=abc-123")

        assert resp.status_code == 200
        instance.read_session_by_id.assert_called_once_with("abc-123")
        instance.read_current_session.assert_not_called()

    def test_session_agents_without_session_id(self, client):
        """GET /api/session-agents falls back to read_current_session."""
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_current_session.return_value = []
            resp = client.get("/api/session-agents")

        assert resp.status_code == 200
        instance.read_current_session.assert_called_once()

    def test_session_chat_without_session_id(self, client):
        """GET /api/session-chat falls back to read_current_chat_history."""
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_current_chat_history.return_value = []
            resp = client.get("/api/session-chat")

        assert resp.status_code == 200
        instance.read_current_chat_history.assert_called_once()

    @pytest.mark.parametrize(
        ("query", "expected"),
        [("&session_id=abc-123", "abc-123"), ("", None)],
        ids=["with-session-id", "without"],
    )
    def test_session_agent_timeline_passes_the_session_id(self, client, query, expected):
        """GET /api/session-agent-timeline forwards the session id, or None."""
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_agent_timeline.return_value = []
            resp = client.get(f"/api/session-agent-timeline?agent_id=agent-xyz{query}")

        assert resp.status_code == 200
        instance.read_agent_timeline.assert_called_once_with("agent-xyz", session_id=expected)

    @staticmethod
    def _point_key_at_cleared_transcript(client):
        """Map SESSION_KEY to the transcript a ``/clear`` moved it to."""
        client.app.state.transcript_map = {SESSION_KEY: CLEARED_TRANSCRIPT}
        client.app.state.transcript_map_provisional = False

    def test_session_agents_reads_the_transcript_the_key_points_at(self, client):
        """GET /api/session-agents maps a session key to its current transcript."""
        self._point_key_at_cleared_transcript(client)
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_session_by_id.return_value = []
            resp = client.get(f"/api/session-agents?session_id={SESSION_KEY}")

        assert resp.status_code == 200
        instance.read_session_by_id.assert_called_once_with(CLEARED_TRANSCRIPT)

    def test_session_log_reads_the_transcript_the_key_points_at(self, client):
        """GET /api/session-log maps a session key to its current transcript."""
        self._point_key_at_cleared_transcript(client)
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_session_by_id.return_value = []
            resp = client.get(f"/api/session-log?session_id={SESSION_KEY}")

        assert resp.status_code == 200
        instance.read_session_by_id.assert_called_once_with(CLEARED_TRANSCRIPT)

    def test_session_agent_timeline_reads_the_transcript_the_key_points_at(self, client):
        """GET /api/session-agent-timeline maps a session key to its current transcript."""
        self._point_key_at_cleared_transcript(client)
        with patch(self._TR) as MockReader:
            instance = MockReader.return_value
            instance.read_agent_timeline.return_value = []
            resp = client.get(
                f"/api/session-agent-timeline?agent_id=agent-xyz&session_id={SESSION_KEY}"
            )

        assert resp.status_code == 200
        instance.read_agent_timeline.assert_called_once_with(
            "agent-xyz", session_id=CLEARED_TRANSCRIPT
        )
