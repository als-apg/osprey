"""Tests for the panel reverse-proxy X-Forwarded-Prefix header."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from osprey.interfaces.web_terminal.routes.proxy import _rewrite_content


def _make_client(workspace_dir, custom_panels, base_path=""):
    """Create a TestClient with custom panels configured."""
    enabled = set(UNIVERSAL_PANELS)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom_panels, None),
        ),
    ):
        app = create_app(shell_command="echo", base_path=base_path)
        with TestClient(app) as c:
            yield app, c


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def app_and_client(workspace_dir):
    """App + client with a custom panel (my-dash → http://localhost:9000)."""
    custom = [
        {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"},
    ]
    yield from _make_client(workspace_dir, custom)


class TestProxyForwardedPrefix:
    def test_x_forwarded_prefix_set(self, app_and_client):
        """Proxy sets X-Forwarded-Prefix header when forwarding to a panel."""
        app, client = app_and_client

        captured_headers = {}

        # Mock the proxy_client's .request() method (used for non-SSE requests).
        async def fake_request(*, method, url, headers, content):
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                json={"ok": True},
                headers={"content-type": "application/json"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/api/status")
        assert resp.status_code == 200
        assert captured_headers.get("x-forwarded-prefix") == "/panel/my-dash"

    def test_nonexistent_panel_returns_404(self, app_and_client):
        """Request to an unknown panel ID returns 404."""
        _app, client = app_and_client
        resp = client.get("/panel/nonexistent/anything")
        assert resp.status_code == 404

    def test_vendor_js_skips_rewriting(self, app_and_client):
        """Vendor JS files are passed through without path rewriting."""
        app, client = app_and_client

        js_body = 'var x = "/static/js/foo.js";'

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/vendor/plotly-3.3.1.min.js")
        assert resp.status_code == 200
        # Vendor path — body must NOT be rewritten
        assert resp.text == js_body

    def test_non_vendor_js_is_rewritten(self, app_and_client):
        """Non-vendor JS files still get path rewriting."""
        app, client = app_and_client

        js_body = 'var x = "/static/js/foo.js";'

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/js/gallery.js")
        assert resp.status_code == 200
        # Non-vendor path — body MUST be rewritten
        assert "/panel/my-dash/static/js/foo.js" in resp.text

    def test_dashboard_prefix_is_rewritten(self, app_and_client):
        """Root-absolute /dashboard/* paths get prefixed so iframe-embedded
        dashboards (e.g. the event dispatcher) reach their own origin through
        the panel proxy rather than escaping to the web-terminal root."""
        app, client = app_and_client

        html_body = (
            "<script>"
            'fetch("/dashboard/triggers");'
            "fetch('/dashboard/runs');"
            'new EventSource("/dashboard/stream/abc");'
            "</script>"
        )

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=html_body,
                headers={"content-type": "text/html"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/dashboard")
        assert resp.status_code == 200
        assert '"/panel/my-dash/dashboard/triggers"' in resp.text
        assert "'/panel/my-dash/dashboard/runs'" in resp.text
        assert '"/panel/my-dash/dashboard/stream/abc"' in resp.text


class TestRewriteContentBasePath:
    """`_rewrite_content` prefixes with the reverse-proxy base path."""

    def test_no_base_path_backcompat(self):
        body = 'a="/static/x";b="/api";href="/"'
        out = _rewrite_content(body, "ariel")
        assert '"/panel/ariel/static/x"' in out
        assert '"/panel/ariel/api"' in out
        assert 'href="/panel/ariel/"' in out

    def test_with_base_path(self):
        body = 'a="/static/x";b="/api";href="/"'
        out = _rewrite_content(body, "ariel", "/user/a")
        assert '"/user/a/panel/ariel/static/x"' in out
        assert '"/user/a/panel/ariel/api"' in out
        assert 'href="/user/a/panel/ariel/"' in out


class TestProxyUnderBasePath:
    """End-to-end: the panel proxy honors the app's base path."""

    @pytest.fixture
    def app_and_client(self, workspace_dir):
        custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"}]
        yield from _make_client(workspace_dir, custom, base_path="/user/a")

    def test_forwarded_prefix_and_rewrite_include_base_path(self, app_and_client):
        app, client = app_and_client

        captured_headers = {}

        async def fake_request(*, method, url, headers, content):
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                text='var x = "/static/js/foo.js";',
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/user/a/panel/my-dash/static/js/gallery.js")
        assert resp.status_code == 200
        assert captured_headers.get("x-forwarded-prefix") == "/user/a/panel/my-dash"
        assert "/user/a/panel/my-dash/static/js/foo.js" in resp.text


class TestEventsPanelTokenInjection:
    """The EVENTS panel proxy injects the dispatcher bearer token server-side."""

    @pytest.fixture
    def app_and_client_events(self, workspace_dir):
        custom = [
            {"id": "events", "label": "EVENTS", "url": "http://localhost:8020"},
            {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"},
        ]
        yield from _make_client(workspace_dir, custom)

    def test_events_panel_injects_bearer(self, app_and_client_events, monkeypatch):
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "sekret")

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/events/dashboard/state")
        assert resp.status_code == 200
        assert captured.get("authorization") == "Bearer sekret"

    def test_non_events_panel_not_injected(self, app_and_client_events, monkeypatch):
        app, client = app_and_client_events
        monkeypatch.setenv("EVENT_DISPATCHER_TOKEN", "sekret")

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/api/status")
        assert resp.status_code == 200
        assert "authorization" not in {k.lower() for k in captured}

    def test_events_panel_no_token_no_header(self, app_and_client_events, monkeypatch):
        app, client = app_and_client_events
        monkeypatch.delenv("EVENT_DISPATCHER_TOKEN", raising=False)

        captured = {}

        async def fake_request(*, method, url, headers, content):
            captured.update(headers)
            return httpx.Response(
                200, json={"ok": True}, headers={"content-type": "application/json"}
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/events/dashboard/state")
        assert resp.status_code == 200
        assert "authorization" not in {k.lower() for k in captured}
