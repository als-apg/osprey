"""Tests for the panel reverse-proxy X-Forwarded-Prefix header."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app


def _make_client(workspace_dir, custom_panels):
    """Create a TestClient with custom panels configured."""
    enabled = set(UNIVERSAL_PANELS)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom_panels),
        ),
    ):
        app = create_app(shell_command="echo")
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
            '<script>'
            'fetch("/dashboard/triggers");'
            "fetch('/dashboard/runs');"
            'new EventSource("/dashboard/stream/abc");'
            '</script>'
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
