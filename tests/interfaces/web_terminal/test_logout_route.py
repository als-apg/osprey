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


class TestLogoutRevokesBrowserSessions:
    """Logout has to invalidate the *credential*, not just the terminal.

    Emptying the PTY and operator pools closes the warm-session hazard;
    it does nothing about the cookie the browser is still holding, which
    the gate would go on accepting. These cover the other half: the
    server-side revocation that makes the logout real, and the delete
    cookie that is the browser-side courtesy layered over it.

    The repo's ``TestClient`` seam injects the operator secret on every
    request, so these requests are admitted regardless of what cookies
    they carry. That is deliberate here: the assertions are about what the
    credentials holder and the on-disk store say afterwards, not about
    whether a later request is refused.
    """

    def test_every_candidate_cookie_is_revoked(self, client):
        """Two cookies of the same name mean two live sessions to drop.

        A page on a sibling host can make the browser send a second cookie
        under the app's own cookie name. The gate accepts either one, so a
        logout that revoked only the first would leave a live session
        behind — see ``read_cookies`` in ``common_middleware``.
        """
        from osprey.interfaces.common_middleware import session_cookie_name
        from osprey.interfaces.web_auth import get_web_credentials

        credentials = get_web_credentials(client.app)
        first = credentials.create_session()
        second = credentials.create_session()
        assert credentials.verify_session(first)
        assert credentials.verify_session(second)

        name = session_cookie_name()
        resp = client.post(
            "/api/terminal/logout",
            headers={"cookie": f"{name}={first}; {name}={second}"},
        )

        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 2
        assert credentials.verify_session(first) is False
        assert credentials.verify_session(second) is False

    def test_candidates_split_across_two_cookie_headers_are_revoked(self, client):
        """Repeated ``Cookie`` headers must be read the way the gate reads them.

        HTTP/2 permits a client to split its cookie header, and the gate
        joins the halves back together (``_scope_headers`` in
        ``common_middleware``) so a session offered only in the second one
        is still honoured. Logout has to see the same thing: a credential
        the gate admits but logout cannot see is a session that survives
        its own logout. Not reachable through the shipped topology today —
        uvicorn speaks HTTP/1.1 and nginx concatenates the field lines
        before proxying — so this pins the agreement rather than a live
        hole, and keeps it pinned if anything ever serves HTTP/2 directly.
        """
        from osprey.interfaces.common_middleware import session_cookie_name
        from osprey.interfaces.web_auth import get_web_credentials

        credentials = get_web_credentials(client.app)
        first = credentials.create_session()
        second = credentials.create_session()

        name = session_cookie_name()
        resp = client.post(
            "/api/terminal/logout",
            headers=[("cookie", f"{name}={first}"), ("cookie", f"{name}={second}")],
        )

        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 2
        assert credentials.verify_session(first) is False
        assert credentials.verify_session(second) is False

    def test_delete_cookie_is_not_marked_secure(self, client):
        """The expiry must land on the plain-http shape as well as on https.

        A browser matches a cookie for deletion by name/domain/path and
        ignores the rest of the attributes, so omitting ``Secure`` still
        clears a cookie that was set with it — while including it would
        have the browser discard the delete outright over plain ``http``,
        which is the single-user loopback shape. This is the one place the
        exchange's derived ``Secure``
        (``WebAuthMiddleware._cookie_is_secure``) is deliberately not
        mirrored.
        """
        from osprey.interfaces.common_middleware import session_cookie_name

        resp = client.post("/api/terminal/logout")
        name = session_cookie_name()

        deletes = [
            value
            for key, value in resp.headers.multi_items()
            if key.lower() == "set-cookie" and value.startswith(f"{name}=;")
        ]
        assert len(deletes) == 1
        header = deletes[0]
        assert "Max-Age=0" in header
        assert "Path=/" in header
        assert "HttpOnly" in header
        assert "SameSite=Lax" in header
        assert "Secure" not in header

    def test_revocation_reaches_the_on_disk_store(self, tmp_path, monkeypatch, workspace_dir):
        """A revocation that lived only in memory would be undone by a restart.

        The store is the half that survives the process, and a restart is
        precisely when nobody is watching, so the digests have to be gone
        from the file — not merely from the map. The store directory and
        the port are published *before* the holder is built (and the
        process holder reset), because both are read once at population
        time; this test therefore builds its own app rather than reusing
        the module ``client`` fixture, whose app is constructed first.
        """
        import json

        from osprey.interfaces.common_middleware import session_cookie_name
        from osprey.interfaces.web_auth import (
            SESSION_STORE_DIR_ENV,
            _digest,
            get_web_credentials,
            reset_web_credentials,
        )

        monkeypatch.setenv(SESSION_STORE_DIR_ENV, str(tmp_path))
        monkeypatch.setenv("OSPREY_WEB_PORT", "8123")
        reset_web_credentials()

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as store_client:
                credentials = get_web_credentials(store_client.app)
                session = credentials.create_session()

                store_path = tmp_path / "sessions-8123.json"
                assert _digest(session) in json.loads(store_path.read_text())["sessions"]

                name = session_cookie_name()
                resp = store_client.post(
                    "/api/terminal/logout",
                    headers={"cookie": f"{name}={session}"},
                )

        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 1
        assert _digest(session) not in json.loads(store_path.read_text())["sessions"]

    def test_logout_without_a_cookie_still_cleans_up(self, client):
        """No credential offered is not an error — the pools still go.

        A browser whose cookie has already expired, or a client that never
        had one, must still be able to end its terminal session; and it
        still gets the delete header, because a cookie this handler cannot
        see may yet be scoped in a way that makes the browser clear it.
        """
        from osprey.interfaces.common_middleware import session_cookie_name

        app = client.app
        registry = app.state.pty_registry
        registry.get_or_create_session("some-claude-session-id", "echo")

        resp = client.post("/api/terminal/logout")

        assert resp.status_code == 200
        assert resp.json()["sessions_revoked"] == 0
        assert any(
            value.startswith(f"{session_cookie_name()}=;")
            for key, value in resp.headers.multi_items()
            if key.lower() == "set-cookie"
        )
        assert registry.get_session("some-claude-session-id") is None
