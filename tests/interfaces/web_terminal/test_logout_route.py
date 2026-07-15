"""Tests for the server-side logout route (task 4.1, closing M2).

``/api/terminal/restart`` reconnects (the client immediately respawns a
fresh PTY under the same flow); ``/api/terminal/logout`` must not leave
anything resumable behind — PTY *or* operator-mode (Agent SDK) session —
so the next visitor at a shared browser cannot inherit the prior user's
warm session of either kind. This mirrors the ``TestRestartEndpoint``
harness in ``test_session_routes.py`` and the dual-registry cleanup in
``restart_terminal`` (routes/panels.py).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app


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


class TestLogoutEndpoint:
    def test_logout_terminates_registry_sessions(self, client):
        """POST /api/terminal/logout empties the PTY registry pool."""
        app = client.app
        registry = app.state.pty_registry

        # Seed a warm session directly into the pool (as if a prior PTY had
        # been detached-but-kept-alive by a normal disconnect).
        session, _ = registry.get_or_create_session("some-claude-session-id", "echo")
        assert registry.get_session("some-claude-session-id") is session

        resp = client.post("/api/terminal/logout")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

        # The prior session must be gone — not merely detached/still-resumable.
        assert registry.get_session("some-claude-session-id") is None

    def test_logout_calls_cleanup_all(self, client):
        """Logout reuses the existing PtyRegistry.cleanup_all primitive."""
        app = client.app
        registry = app.state.pty_registry

        with patch.object(registry, "cleanup_all") as mock_cleanup:
            resp = client.post("/api/terminal/logout")

        assert resp.status_code == 200
        mock_cleanup.assert_called_once()

    def test_logout_also_cleans_operator_registry(self, client):
        """Logout must not leave a warm operator-mode (Agent SDK) session behind.

        A live agent with tool access is more sensitive than a bare shell
        PTY, so the M2 hazard is worse if this is skipped. Mirrors
        ``restart_terminal``'s dual-registry cleanup (routes/panels.py).
        """
        app = client.app
        operator_registry = app.state.operator_registry

        with patch.object(operator_registry, "cleanup_all", new_callable=AsyncMock) as mock_cleanup:
            resp = client.post("/api/terminal/logout")

        assert resp.status_code == 200
        mock_cleanup.assert_awaited_once()
