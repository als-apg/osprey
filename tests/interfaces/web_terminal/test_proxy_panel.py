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
