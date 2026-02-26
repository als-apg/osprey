"""Tests for dynamic web panel registration (Phase 3)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app


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


@pytest.fixture
def client_with_custom_panels(workspace_dir):
    """Client where app.state.custom_panels is pre-populated."""
    panels = [
        {"id": "my-dashboard", "label": "DASHBOARD", "url": "http://localhost:9000"},
        {"id": "grafana", "label": "GRAFANA", "url": "http://localhost:3000"},
    ]
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_custom_panels",
            return_value=panels,
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


class TestCustomPanelsAPI:
    def test_custom_panels_api(self, client_with_custom_panels):
        """GET /api/custom-panels returns configured panels."""
        resp = client_with_custom_panels.get("/api/custom-panels")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["id"] == "my-dashboard"
        assert data[0]["label"] == "DASHBOARD"
        assert data[1]["id"] == "grafana"

    def test_empty_custom_panels(self, client):
        """No web.panels in config returns empty list."""
        resp = client.get("/api/custom-panels")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_panel_focus_custom_id(self, client_with_custom_panels):
        """Custom panel ID is accepted by POST /api/panel-focus."""
        resp = client_with_custom_panels.post(
            "/api/panel-focus",
            json={"panel": "my-dashboard"},
        )
        assert resp.status_code == 200
        assert resp.json()["active_panel"] == "my-dashboard"

    def test_panel_focus_unknown_id(self, client_with_custom_panels):
        """Unknown ID not in known or custom panels returns 422."""
        resp = client_with_custom_panels.post(
            "/api/panel-focus",
            json={"panel": "nonexistent-panel"},
        )
        assert resp.status_code == 422
