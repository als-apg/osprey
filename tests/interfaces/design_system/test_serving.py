"""Tests that all six interface apps serve the shared design system.

Each app mounts ``src/osprey/interfaces/design_system/static`` at
``/design-system`` (mirroring the existing ``SHARED_FONTS_DIR`` mount idiom).
The web-terminal panel reverse-proxy also rewrites ``/design-system/...``
references so proxied panels resolve the shared assets through their own
``/panel/{id}/`` prefix.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient


class TestDesignSystemMountedInEveryApp:
    """GET /design-system/css/tokens.css must succeed in all six apps.

    None of these apps need their lifespan to run to serve a static mount
    (the mount is registered synchronously inside ``create_app``), so a
    plain ``TestClient`` — without entering it as a context manager — is
    enough and avoids spinning up PTYs, watchers, or backend clients.
    """

    def test_web_terminal(self):
        from osprey.interfaces.web_terminal.app import create_app

        app = create_app(shell_command="echo")
        resp = TestClient(app).get("/design-system/css/tokens.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert resp.text.strip()

    def test_artifacts(self, tmp_path):
        from osprey.interfaces.artifacts.app import create_app

        app = create_app(workspace_root=tmp_path)
        resp = TestClient(app).get("/design-system/css/tokens.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert resp.text.strip()

    def test_ariel(self):
        from osprey.interfaces.ariel.app import create_app

        app = create_app()
        resp = TestClient(app).get("/design-system/css/tokens.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert resp.text.strip()

    def test_channel_finder(self):
        from osprey.interfaces.channel_finder.app import create_app

        app = create_app(project_cwd="/tmp/test-project")
        resp = TestClient(app).get("/design-system/css/tokens.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert resp.text.strip()

    def test_tuning(self):
        from osprey.interfaces.tuning.app import create_app

        app = create_app()
        resp = TestClient(app).get("/design-system/css/tokens.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert resp.text.strip()

    def test_lattice_dashboard(self, tmp_path):
        from osprey.interfaces.lattice_dashboard.app import create_app

        app = create_app(workspace_root=tmp_path)
        resp = TestClient(app).get("/design-system/css/tokens.css")
        assert resp.status_code == 200
        assert "text/css" in resp.headers["content-type"]
        assert resp.text.strip()


class TestDesignSystemProxyRewrite:
    """/panel/{id}/... proxy rewrites /design-system/... references."""

    @pytest.fixture
    def workspace_dir(self, tmp_path):
        ws = tmp_path / "_agent_data"
        ws.mkdir()
        return ws

    @pytest.fixture
    def app_and_client(self, workspace_dir):
        from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app

        custom = [
            {"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"},
        ]
        enabled = set(UNIVERSAL_PANELS)
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value={"watch_dir": str(workspace_dir)},
            ),
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=(enabled, custom, None),
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                yield app, c

    def test_design_system_prefix_is_rewritten(self, app_and_client):
        """HTML served through /panel/{id}/ has /design-system/... rewritten
        to /panel/{id}/design-system/... so followers resolve the shared
        tokens/CSS/JS through their own panel origin."""
        app, client = app_and_client

        html_body = (
            '<link rel="stylesheet" href="/design-system/css/tokens.css">'
            '<script src="/design-system/js/theme-boot.js"></script>'
        )

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=html_body,
                headers={"content-type": "text/html"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/")
        assert resp.status_code == 200
        assert '"/panel/my-dash/design-system/css/tokens.css"' in resp.text
        assert '"/panel/my-dash/design-system/js/theme-boot.js"' in resp.text

    def test_design_system_prefix_rewritten_in_css(self, app_and_client):
        """CSS responses (e.g. url() references) also get the prefix rewrite."""
        app, client = app_and_client

        css_body = "body { background: url('/design-system/img/noise.png'); }"

        async def fake_request(*, method, url, headers, content):
            return httpx.Response(
                status_code=200,
                text=css_body,
                headers={"content-type": "text/css"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=fake_request)

        resp = client.get("/panel/my-dash/static/css/theme.css")
        assert resp.status_code == 200
        assert "'/panel/my-dash/design-system/img/noise.png'" in resp.text
