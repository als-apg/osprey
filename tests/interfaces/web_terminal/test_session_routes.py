"""Tests for session-related routes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app
from osprey.interfaces.web_terminal.session_discovery import SessionInfo


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "osprey-workspace"
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
    def test_returns_empty_list(self, client):
        """GET /api/sessions returns empty list when registry is empty."""
        with (
            patch(
                "osprey.interfaces.web_terminal.routes.SessionRegistry.known_ids",
                return_value=set(),
            ),
            patch(
                "osprey.interfaces.web_terminal.routes.SessionDiscovery.list_sessions",
                return_value=[],
            ),
        ):
            resp = client.get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []

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

        known = {"aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "11111111-2222-3333-4444-555555555555"}
        with (
            patch(
                "osprey.interfaces.web_terminal.routes.SessionRegistry.known_ids",
                return_value=known,
            ),
            patch(
                "osprey.interfaces.web_terminal.routes.SessionDiscovery.list_sessions",
                return_value=mock_sessions,
            ),
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
    def test_restart_cleans_all(self, client):
        """POST /api/terminal/restart calls cleanup_all()."""
        app = client.app
        registry = app.state.pty_registry

        with patch.object(registry, "cleanup_all") as mock_cleanup:
            resp = client.post("/api/terminal/restart")

        assert resp.status_code == 200
        mock_cleanup.assert_called_once()


class TestResolveWorkspace:
    def test_file_tree_with_session_id(self, workspace_dir, client):
        """File tree with session_id scopes to sessions subdir."""
        session_dir = workspace_dir / "sessions" / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        session_dir.mkdir(parents=True)
        (session_dir / "test.txt").write_text("scoped content")

        resp = client.get(
            "/api/files/tree?session_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        )
        assert resp.status_code == 200
        data = resp.json()
        names = [c["name"] for c in data.get("children", [])]
        assert "test.txt" in names

    def test_file_tree_without_session_id(self, workspace_dir, client):
        """File tree without session_id uses base workspace."""
        (workspace_dir / "base_file.txt").write_text("base content")

        resp = client.get("/api/files/tree")
        assert resp.status_code == 200
        data = resp.json()
        names = [c["name"] for c in data.get("children", [])]
        assert "base_file.txt" in names

    def test_invalid_session_id_ignored(self, workspace_dir, client):
        """Invalid session_id (path traversal attempt) falls back to base."""
        (workspace_dir / "base_file.txt").write_text("safe")

        resp = client.get("/api/files/tree?session_id=../../../etc")
        assert resp.status_code == 200
        data = resp.json()
        # Should not have traversed — uses base workspace
        names = [c["name"] for c in data.get("children", [])]
        assert "base_file.txt" in names
