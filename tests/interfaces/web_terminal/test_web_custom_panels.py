"""Tests for web panel configuration (template-driven panel filtering)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    BUILTIN_PANELS,
    UNIVERSAL_PANELS,
    _load_panel_config,
    create_app,
)


@pytest.fixture
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _make_client(workspace_dir, enabled_panels=None, custom_panels=None):
    """Create a TestClient with the given panel config."""
    if enabled_panels is None:
        enabled_panels = set(UNIVERSAL_PANELS)
    if custom_panels is None:
        custom_panels = []
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled_panels, custom_panels),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client(workspace_dir):
    """Client with only universal panels (no domain panels)."""
    yield from _make_client(workspace_dir)


@pytest.fixture
def client_all_panels(workspace_dir):
    """Client with all built-in panels enabled."""
    yield from _make_client(workspace_dir, enabled_panels=set(BUILTIN_PANELS))


@pytest.fixture
def client_with_custom_panels(workspace_dir):
    """Client with custom panels added."""
    custom = [
        {"id": "my-dashboard", "label": "DASHBOARD", "url": "http://localhost:9000"},
        {"id": "grafana", "label": "GRAFANA", "url": "http://localhost:3000"},
    ]
    enabled = set(UNIVERSAL_PANELS)
    yield from _make_client(workspace_dir, enabled_panels=enabled, custom_panels=custom)


# ---- Unit tests for _load_panel_config ----


class TestLoadPanelConfig:
    def test_no_web_section(self):
        """Missing web section returns only universal panels."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={},
        ):
            enabled, custom = _load_panel_config()
        assert enabled == UNIVERSAL_PANELS
        assert custom == []

    def test_empty_panels(self):
        """Empty web.panels returns only universal panels."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {}}},
        ):
            enabled, custom = _load_panel_config()
        assert enabled == UNIVERSAL_PANELS
        assert custom == []

    def test_domain_panels_enabled(self):
        """Domain panels with enabled: true are added."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "ariel": {"enabled": True},
                        "tuning": {"enabled": True},
                    }
                }
            },
        ):
            enabled, custom = _load_panel_config()
        assert "ariel" in enabled
        assert "tuning" in enabled
        assert "channel-finder" not in enabled
        # Universal panels are always present
        assert UNIVERSAL_PANELS <= enabled

    def test_domain_panel_disabled(self):
        """Domain panels with enabled: false are excluded."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "ariel": {"enabled": False},
                        "tuning": {"enabled": True},
                    }
                }
            },
        ):
            enabled, custom = _load_panel_config()
        assert "ariel" not in enabled
        assert "tuning" in enabled

    def test_domain_panel_bare_true(self):
        """A bare `true` value (not dict) enables the panel."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {"ariel": True}}},
        ):
            enabled, custom = _load_panel_config()
        assert "ariel" in enabled

    def test_domain_panel_dict_defaults_enabled(self):
        """A dict without explicit enabled key defaults to enabled."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {"ariel": {}}}},
        ):
            enabled, custom = _load_panel_config()
        assert "ariel" in enabled

    def test_custom_panel_extracted(self):
        """Non-builtin panel IDs are returned as custom panels."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "my-grafana": {
                            "label": "GRAFANA",
                            "url": "http://grafana.local:3000",
                            "health_endpoint": "/api/health",
                        }
                    }
                }
            },
        ):
            enabled, custom = _load_panel_config()
        assert len(custom) == 1
        assert custom[0]["id"] == "my-grafana"
        assert custom[0]["label"] == "GRAFANA"
        assert custom[0]["url"] == "http://grafana.local:3000"
        assert custom[0]["healthEndpoint"] == "/api/health"

    def test_mixed_builtin_and_custom(self):
        """Both builtin and custom panels are handled correctly."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "ariel": {"enabled": True},
                        "channel-finder": {"enabled": False},
                        "my-dash": {"label": "DASH", "url": "http://localhost:9000"},
                    }
                }
            },
        ):
            enabled, custom = _load_panel_config()
        assert "ariel" in enabled
        assert "channel-finder" not in enabled
        assert len(custom) == 1
        assert custom[0]["id"] == "my-dash"

    def test_config_load_failure(self):
        """Config load failure returns universal panels only."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            side_effect=RuntimeError("no config"),
        ):
            enabled, custom = _load_panel_config()
        assert enabled == UNIVERSAL_PANELS
        assert custom == []


# ---- API endpoint tests ----


class TestPanelsAPI:
    def test_panels_api_universal_only(self, client):
        """GET /api/panels returns only universal panels when no domain panels enabled."""
        resp = client.get("/api/panels")
        assert resp.status_code == 200
        data = resp.json()
        enabled_set = set(data["enabled"])
        assert enabled_set == UNIVERSAL_PANELS
        assert data["custom"] == []

    def test_panels_api_all_panels(self, client_all_panels):
        """GET /api/panels returns all panels when all enabled."""
        resp = client_all_panels.get("/api/panels")
        assert resp.status_code == 200
        data = resp.json()
        enabled_set = set(data["enabled"])
        assert enabled_set == BUILTIN_PANELS

    def test_panels_api_with_custom(self, client_with_custom_panels):
        """GET /api/panels returns custom panels."""
        resp = client_with_custom_panels.get("/api/panels")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["custom"]) == 2
        assert data["custom"][0]["id"] == "my-dashboard"

    def test_panel_focus_enabled_panel(self, client_all_panels):
        """POST /api/panel-focus accepts enabled panel IDs."""
        resp = client_all_panels.post(
            "/api/panel-focus",
            json={"panel": "ariel"},
        )
        assert resp.status_code == 200
        assert resp.json()["active_panel"] == "ariel"

    def test_panel_focus_custom_id(self, client_with_custom_panels):
        """Custom panel ID is accepted by POST /api/panel-focus."""
        resp = client_with_custom_panels.post(
            "/api/panel-focus",
            json={"panel": "my-dashboard"},
        )
        assert resp.status_code == 200
        assert resp.json()["active_panel"] == "my-dashboard"

    def test_panel_focus_disabled_panel(self, client):
        """Disabled panel ID returns 422."""
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "ariel"},
        )
        assert resp.status_code == 422

    def test_panel_focus_unknown_id(self, client):
        """Completely unknown ID returns 422."""
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "nonexistent-panel"},
        )
        assert resp.status_code == 422
