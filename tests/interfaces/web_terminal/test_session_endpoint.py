"""Contract tests for ``GET /api/session`` — the authenticated liveness probe.

Task 2.3 (session-endpoint-and-probe): single-user must not get stuck in a
permanent reconnect loop after a restart mints a fresh secret. ``/api/session``
is a tiny credentialed endpoint that answers 200 to a live operator credential
and 401 to none — the app-level 401 the frontend probe (``probeSessionExpired``
in ``api.js``) reads to tell an expired session apart from a flaky network.

The endpoint's value is entirely in the auth gate that fronts it, so these tests
set auth up **explicitly** — a known operator secret on
``app.state.web_credentials``, credentials constructed by hand — and opt the
whole module out of the shared TestClient auth seam (``pytestmark`` below). The
seam force-sets the operator header on every request, which would silently admit
the very unauthenticated request the 401 case asserts is refused.

The apps are driven without the ``TestClient`` lifespan (no ``with`` block): the
gate reads only ``app.state.web_credentials``, seeded directly, and skipping
startup keeps the tests off the PTY/file-watcher machinery the probe never
touches.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from osprey.interfaces.common_middleware import (
    EXEMPT_PATHS,
    OPERATOR_SECRET_HEADER,
    session_cookie_name,
)
from osprey.interfaces.web_auth import WebCredentials
from osprey.interfaces.web_terminal.app import create_app

pytestmark = pytest.mark.no_auth_seam

OPERATOR_SECRET = "operator-secret-value"
PANEL_TOKEN = "panel-token-value"
WEB_PORT = "8899"


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    """A real web-terminal app with a known operator secret on its gate.

    The port is pinned through the environment so the gate and the cookie name
    agree, and the credential holder is set directly on ``app.state`` — the
    object the gate reads.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", WEB_PORT)
    application = create_app(shell_command="echo")
    application.state.web_credentials = WebCredentials(
        operator_secret=OPERATOR_SECRET, panel_token=PANEL_TOKEN
    )
    return application


def _client(app) -> TestClient:
    """A fresh client with an empty cookie jar."""
    client = TestClient(app)
    client.cookies.clear()
    return client


def test_session_is_not_exempt_from_the_auth_gate():
    """The probe must sit behind auth — never in the middleware exempt set.

    Guards the contract directly: if ``/api/session`` were ever added to
    ``EXEMPT_PATHS`` the gate would stop 401'ing an expired credential and the
    single-user reconnect loop would silently return.
    """
    assert "/api/session" not in EXEMPT_PATHS


def test_valid_operator_secret_header_gets_200(app):
    """A live operator-secret header reaches the handler and gets a 200."""
    response = _client(app).get("/api/session", headers={OPERATOR_SECRET_HEADER: OPERATOR_SECRET})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_session_cookie_gets_200(app):
    """The browser path: a live session cookie authenticates the probe."""
    session_id = app.state.web_credentials.create_session()
    client = _client(app)
    client.cookies.set(session_cookie_name(WEB_PORT), session_id)

    response = client.get("/api/session")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_no_credential_gets_401(app):
    """No secret and no cookie means a 401 — the signal a stale session needs.

    This is the whole reason the endpoint exists: the app itself (not a
    perimeter) refuses an unauthenticated request, so ``probeSessionExpired``
    sees a definite 401 and breaks the reconnect loop.
    """
    response = _client(app).get("/api/session")
    assert response.status_code == 401


def test_wrong_operator_secret_gets_401(app):
    """A stale/wrong operator secret is refused, not admitted."""
    response = _client(app).get("/api/session", headers={OPERATOR_SECRET_HEADER: "not-the-secret"})
    assert response.status_code == 401


def test_expired_session_cookie_gets_401(app):
    """A cookie whose session the gate no longer knows yields a 401.

    Models the restart case: a browser holds a cookie for a session the fresh
    process never minted, so the gate rejects it and the frontend learns the
    session expired instead of reconnecting forever.
    """
    client = _client(app)
    client.cookies.set(session_cookie_name(WEB_PORT), "stale-unknown-session-id")

    response = client.get("/api/session")
    assert response.status_code == 401
