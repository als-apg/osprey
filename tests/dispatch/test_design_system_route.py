"""Tests for the dispatch server's ``/design-system/{path}`` asset route (PLAN Task 4.1).

FastMCP has no static-file mount (unlike the FastAPI interfaces, which mount
``StaticFiles`` at ``/design-system``), so the dispatch dashboard's
tokens.css/base.css/theme-boot.js/theme-manager.js are served through a
hand-rolled ``@mcp.custom_route`` in ``osprey.dispatch.server`` instead. This
route is ungated (like the ``/dashboard`` shell), so its path-traversal guard
is the only thing standing between an anonymous request and the rest of the
filesystem — see ``server.py``'s ``design_system_asset`` for the guard itself.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from osprey.dispatch import server


@pytest.fixture(autouse=True)
def _reset_mcp_routes():
    """Reset the shared FastMCP singleton's routes around each test.

    Mirrors ``tests/unit/dispatch/test_server_routes.py``: ``create_server()``
    mutates a module-level FastMCP singleton and appends dashboard routes to
    it on every call, so each test starts from a clean slate.
    """
    baseline = list(server.mcp._additional_http_routes)
    yield
    server.mcp._additional_http_routes = baseline


@pytest.fixture
def app(tmp_path, monkeypatch):
    """Build the dispatcher ASGI app with no triggers configured.

    The design-system route needs none of the trigger/webhook machinery, so
    point ``TRIGGERS_YML`` at a file that doesn't exist — ``create_server()``
    handles that by starting with an empty trigger list.
    """
    monkeypatch.setenv("TRIGGERS_YML", str(tmp_path / "does-not-exist.yml"))
    return server.create_server().http_app()


def test_serves_known_asset_with_200_and_css_content_type(app):
    with TestClient(app) as client:
        resp = client.get("/design-system/css/tokens.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]
    assert "--bg-primary" in resp.text


def test_serves_known_js_asset(app):
    with TestClient(app) as client:
        resp = client.get("/design-system/js/theme-manager.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]


def test_missing_file_returns_404(app):
    with TestClient(app) as client:
        resp = client.get("/design-system/css/does-not-exist.css")
    assert resp.status_code == 404


def test_traversal_attempt_returns_404_not_403(app):
    """A path-traversal probe must 404, never 403 (don't leak existence).

    Percent-encodes the dot segments so httpx's client-side URL normalization
    (which collapses literal ".." before the request is ever sent, landing on
    an unrelated matched/unmatched route rather than this one) doesn't defeat
    the traversal attempt before it reaches the server. Asserting the exact
    body also confirms this 404 is the guard's own response and not a generic
    route-miss from the request landing somewhere else entirely.
    """
    with TestClient(app) as client:
        resp = client.get("/design-system/%2e%2e/%2e%2e/server.py")
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Not found"}


def test_deeper_traversal_escape_returns_404(app):
    """A path segment that resolves outside the static root via a deeper escape."""
    with TestClient(app) as client:
        resp = client.get("/design-system/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
    assert resp.status_code == 404
