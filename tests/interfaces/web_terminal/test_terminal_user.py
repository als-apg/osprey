"""Tests for per-user identity surfacing in the web terminal.

``OSPREY_TERMINAL_USER`` and ``OSPREY_TERMINAL_LANDING_URL`` are read
env-over-config (mirroring ``app.state.app_name``) and passed into the
``index.html`` template context by the ``root()`` route.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal import app as app_module
from osprey.interfaces.web_terminal.app import create_app


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory for the app to watch."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    (ws / "README.md").write_text("# Test workspace\n")
    return ws


@pytest.fixture
def client(workspace_dir):
    """Create a test client with mocked config active through lifespan."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


class TestTerminalUser:
    """``OSPREY_TERMINAL_USER`` -> ``app.state.terminal_user`` (env-only, no config key)."""

    def test_set_from_env(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict("os.environ", {"OSPREY_TERMINAL_USER": "alice"}),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                assert app.state.terminal_user == "alice"
                assert c.get("/").status_code == 200

    def test_empty_when_unset(self, client):
        # The shared `client` fixture supplies no OSPREY_TERMINAL_USER.
        assert client.app.state.terminal_user == ""


class TestLandingURL:
    """``OSPREY_TERMINAL_LANDING_URL`` -> ``app.state.landing_url`` (env-only, no config key)."""

    def test_set_from_env(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict(
                "os.environ",
                {"OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal"},
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                assert app.state.landing_url == "https://facility.example/portal"
                assert c.get("/").status_code == 200

    def test_empty_when_unset(self, client):
        # The shared `client` fixture supplies no OSPREY_TERMINAL_LANDING_URL.
        assert client.app.state.landing_url == ""


class TestRootContext:
    """root() must forward terminal_user/landing_url into the index.html context."""

    def test_context_includes_terminal_user_and_landing_url(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch.dict(
                "os.environ",
                {
                    "OSPREY_TERMINAL_USER": "bob",
                    "OSPREY_TERMINAL_LANDING_URL": "https://facility.example/portal",
                },
            ),
        ):
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                captured = {}
                original = app_module.templates.TemplateResponse

                def _capture(request, name, context=None, *args, **kwargs):
                    captured.update(context or {})
                    return original(request, name, context, *args, **kwargs)

                with patch.object(app_module.templates, "TemplateResponse", side_effect=_capture):
                    resp = c.get("/")

                assert resp.status_code == 200
                assert captured["terminal_user"] == "bob"
                assert captured["landing_url"] == "https://facility.example/portal"

    def test_context_empty_when_unset(self, client):
        captured = {}
        original = app_module.templates.TemplateResponse

        def _capture(request, name, context=None, *args, **kwargs):
            captured.update(context or {})
            return original(request, name, context, *args, **kwargs)

        with patch.object(app_module.templates, "TemplateResponse", side_effect=_capture):
            resp = client.get("/")

        assert resp.status_code == 200
        assert captured["terminal_user"] == ""
        assert captured["landing_url"] == ""
