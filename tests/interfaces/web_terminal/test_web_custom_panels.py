"""Tests for web panel configuration (template-driven panel filtering)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import (
    BUILTIN_PANELS,
    UNIVERSAL_PANELS,
    _load_panel_config,
    create_app,
)
from osprey.profiles.web_panels import BUILTIN_PANEL_LABELS


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
            return_value=(enabled_panels, custom_panels, None),
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
            enabled, custom, _default = _load_panel_config()
        assert enabled == UNIVERSAL_PANELS
        assert custom == []

    def test_empty_panels(self):
        """Empty web.panels returns only universal panels."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {}}},
        ):
            enabled, custom, _default = _load_panel_config()
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
                        "lattice": {"enabled": True},
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" in enabled
        assert "lattice" in enabled
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
                        "lattice": {"enabled": True},
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" not in enabled
        assert "lattice" in enabled

    def test_domain_panel_bare_true(self):
        """A bare `true` value (not dict) enables the panel."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {"ariel": True}}},
        ):
            enabled, custom, _default = _load_panel_config()
        assert "ariel" in enabled

    def test_domain_panel_dict_defaults_enabled(self):
        """A dict without explicit enabled key defaults to enabled."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {"ariel": {}}}},
        ):
            enabled, custom, _default = _load_panel_config()
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
            enabled, custom, _default = _load_panel_config()
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
            enabled, custom, _default = _load_panel_config()
        assert "ariel" in enabled
        assert "channel-finder" not in enabled
        assert len(custom) == 1
        assert custom[0]["id"] == "my-dash"

    def test_events_panel_is_url_backed_custom(self):
        """The control-assistant EVENTS panel is URL-backed.

        It must be emitted as a custom panel (carrying url/path/health) so the
        web terminal renders the iframe tab, not silently collapsed into the
        builtin `enabled` set where its url is discarded and no frontend tab exists.
        """
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={
                "web": {
                    "panels": {
                        "events": {
                            "label": "EVENTS",
                            "url": "http://localhost:8020",
                            "path": "/dashboard",
                            "health_endpoint": "/health",
                        }
                    }
                }
            },
        ):
            enabled, custom, _default = _load_panel_config()
        assert "events" not in enabled  # not silently dropped as a builtin
        events = [p for p in custom if p["id"] == "events"]
        assert len(events) == 1
        assert events[0]["url"] == "http://localhost:8020"
        assert events[0]["path"] == "/dashboard"
        assert events[0]["healthEndpoint"] == "/health"
        assert events[0]["label"] == "EVENTS"

    def test_config_load_failure(self):
        """Config load failure returns universal panels only."""
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            side_effect=RuntimeError("no config"),
        ):
            enabled, custom, _default = _load_panel_config()
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


# ---- Helpers for new feature tests ----

_LAN_ADDR = [(2, 1, 6, "", ("10.0.0.5", 0))]
_LOOPBACK_ADDR = [(2, 1, 6, "", ("127.0.0.1", 0))]
_GETADDRINFO_TARGET = "osprey.interfaces.web_terminal.routes.panels.socket.getaddrinfo"


def _make_client_with_runtime_panels(workspace_dir, allowlist=None):
    """Yield a TestClient whose lifespan has allow_runtime_panels enabled.

    Patches both the panel-config loader and the raw osprey config so the
    lifespan sets ``app.state.allow_runtime_panels = True``.
    """
    web_cfg: dict = {"allow_runtime_panels": True}
    if allowlist is not None:
        web_cfg["runtime_panel_allowlist"] = allowlist
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(set(UNIVERSAL_PANELS), [], None),
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": web_cfg},
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


def _make_client_with_hidden_panel(workspace_dir, hidden_panel_id, enabled_panels):
    """Yield a TestClient where *hidden_panel_id* is enabled but hidden:true."""
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(set(enabled_panels), [], None),
        ),
        patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={"web": {"panels": {hidden_panel_id: {"enabled": True, "hidden": True}}}},
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


@pytest.fixture
def client_runtime_panels(workspace_dir):
    """Client with allow_runtime_panels: True and no allowlist."""
    yield from _make_client_with_runtime_panels(workspace_dir)


@pytest.fixture
def client_runtime_panels_allowlist(workspace_dir):
    """Client with allow_runtime_panels: True and allowlist=['grafana.lan']."""
    yield from _make_client_with_runtime_panels(workspace_dir, allowlist=["grafana.lan"])


# ---- Guard: _load_panel_config 3-tuple contract ----


class TestLoadPanelConfigContract:
    def test_returns_three_tuple(self):
        """_load_panel_config always returns a 3-tuple (enabled, custom, default)."""
        # Arrange
        with patch(
            "osprey.utils.workspace.load_osprey_config",
            return_value={},
        ):
            # Act
            result = _load_panel_config()

        # Assert
        assert isinstance(result, tuple), "result must be a tuple"
        assert len(result) == 3, "result must have exactly 3 elements"
        enabled, custom, default = result
        assert isinstance(enabled, (set, frozenset))
        assert isinstance(custom, list)


# ---- Six-key /api/panels response shape ----


class TestPanelsAPIShape:
    def test_response_has_six_keys(self, client):
        """GET /api/panels includes all six keys: enabled, custom, default, visible, active, labels."""
        # Arrange — client has only universal panels

        # Act
        resp = client.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "custom" in data
        assert "default" in data
        assert "visible" in data
        assert "active" in data
        assert "labels" in data

    def test_labels_map_enabled_builtin_ids_to_display_names(self, client_all_panels):
        """labels contains BUILTIN_PANEL_LABELS entries for each enabled built-in panel."""
        # Arrange — client_all_panels enables the full BUILTIN_PANELS set

        # Act
        resp = client_all_panels.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        labels = resp.json()["labels"]
        for panel_id in BUILTIN_PANELS:
            assert panel_id in labels, f"missing label for {panel_id!r}"
            assert labels[panel_id] == BUILTIN_PANEL_LABELS[panel_id]

    def test_visible_defaults_to_all_enabled_when_no_hidden_flags(self, client_all_panels):
        """visible equals the full enabled set when no hidden:true flags are configured."""
        # Arrange — no hidden flags in config (load_osprey_config returns {})

        # Act
        resp = client_all_panels.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        data = resp.json()
        assert set(data["visible"]) == set(data["enabled"])

    def test_active_is_null_on_fresh_start(self, client):
        """active is null immediately after server start (no panel has been focused yet)."""
        # Arrange — fresh TestClient, no panel-focus POST issued

        # Act
        resp = client.get("/api/panels")

        # Assert
        assert resp.status_code == 200
        assert resp.json()["active"] is None


# ---- Hidden panel visibility ----


class TestHiddenPanels:
    def test_hidden_builtin_absent_from_visible_but_present_in_enabled(self, workspace_dir):
        """A panel with hidden:true is omitted from visible but kept in enabled."""
        # Arrange — ariel enabled in _load_panel_config but hidden:true in raw config
        enabled = {"artifacts", "ariel"}
        gen = _make_client_with_hidden_panel(workspace_dir, "ariel", enabled)
        client = next(gen)

        try:
            # Act
            resp = client.get("/api/panels")

            # Assert
            assert resp.status_code == 200
            data = resp.json()
            assert "ariel" in data["enabled"], "ariel should be in enabled"
            assert "ariel" not in data["visible"], "ariel should not be in visible when hidden:true"
        finally:
            try:
                next(gen)
            except StopIteration:
                pass


# ---- POST /api/panel-visibility ----


class TestPanelVisibilityAPI:
    def test_valid_panel_mutates_visible_panels_state(self, client_all_panels):
        """POST /api/panel-visibility with a known id updates app.state.visible_panels."""
        # Arrange — all panels enabled; hide ariel to get a deterministic starting state
        client_all_panels.post("/api/panel-visibility", json={"panel": "ariel", "visible": False})

        # Act — show ariel again
        resp = client_all_panels.post(
            "/api/panel-visibility", json={"panel": "ariel", "visible": True}
        )

        # Assert
        assert resp.status_code == 200
        assert resp.json()["panel"] == "ariel"
        assert resp.json()["visible"] is True
        assert "ariel" in client_all_panels.app.state.visible_panels

    def test_valid_panel_broadcasts_panel_visibility_event(self, client_all_panels):
        """POST /api/panel-visibility broadcasts the exact event dict via the broadcaster."""
        # Arrange — replace broadcaster.broadcast with a mock to capture calls
        mock_broadcast = MagicMock()
        client_all_panels.app.state.broadcaster.broadcast = mock_broadcast

        # Act
        client_all_panels.post("/api/panel-visibility", json={"panel": "ariel", "visible": False})

        # Assert
        mock_broadcast.assert_called_once_with(
            {"type": "panel_visibility", "panel": "ariel", "visible": False}
        )

    def test_unknown_panel_returns_422(self, client):
        """POST /api/panel-visibility returns 422 for a panel id that is not enabled."""
        # Arrange — client has only universal panels; "ariel" is disabled

        # Act
        resp = client.post("/api/panel-visibility", json={"panel": "ariel", "visible": True})

        # Assert
        assert resp.status_code == 422

    def test_unknown_panel_does_not_broadcast(self, client):
        """No broadcast is emitted when the panel id is unknown (422 path)."""
        # Arrange
        mock_broadcast = MagicMock()
        client.app.state.broadcaster.broadcast = mock_broadcast

        # Act
        client.post("/api/panel-visibility", json={"panel": "nonexistent", "visible": True})

        # Assert
        mock_broadcast.assert_not_called()


# ---- POST /api/panels/register ----


class TestPanelRegisterAPI:
    def test_register_forbidden_when_runtime_panels_disabled(self, client):
        """POST /api/panels/register returns 403 when allow_runtime_panels is False."""
        # Arrange — default client has allow_runtime_panels=False

        # Act
        resp = client.post(
            "/api/panels/register",
            json={"id": "my-panel", "label": "MY", "url": "http://grafana.lan:3000"},
        )

        # Assert
        assert resp.status_code == 403

    def test_register_success_stores_raw_url_in_custom_panels(self, client_runtime_panels):
        """Successful registration stores the original (raw) URL in app.state.custom_panels."""
        # Arrange
        raw_url = "http://grafana.lan:3000"

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": raw_url},
            )

        # Assert
        assert resp.status_code == 200
        stored = client_runtime_panels.app.state.custom_panels
        assert len(stored) == 1
        assert stored[0]["url"] == raw_url, "raw URL must be preserved in state for proxy"

    def test_register_success_broadcasts_with_proxy_url_path_and_health(
        self, client_runtime_panels
    ):
        """Successful registration broadcasts /panel/{id} URL plus path and healthEndpoint."""
        # Arrange
        mock_broadcast = MagicMock()
        client_runtime_panels.app.state.broadcaster.broadcast = mock_broadcast

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            client_runtime_panels.post(
                "/api/panels/register",
                json={
                    "id": "grafana",
                    "label": "GRAFANA",
                    "url": "http://grafana.lan:3000",
                    "path": "/d/abc",
                    "health_endpoint": "/api/health",
                },
            )

        # Assert
        mock_broadcast.assert_called_once()
        event = mock_broadcast.call_args[0][0]
        assert event["type"] == "panel_register"
        assert event["url"] == "/panel/grafana"
        assert event["path"] == "/d/abc"
        assert event["healthEndpoint"] == "/api/health"

    def test_register_duplicate_id_replaces_not_duplicates(self, client_runtime_panels):
        """Re-registering an existing id replaces the entry (no duplicates, len unchanged)."""
        # Arrange — register "grafana" once
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA", "url": "http://grafana.lan:3000"},
            )

        # Act — re-register "grafana" with a different URL
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "grafana", "label": "GRAFANA 2", "url": "http://grafana.lan:4000"},
            )

        # Assert
        assert resp.status_code == 200
        stored = client_runtime_panels.app.state.custom_panels
        assert len(stored) == 1, "duplicate id must not create a second entry"
        assert stored[0]["label"] == "GRAFANA 2"

    def test_register_loopback_address_returns_422(self, client_runtime_panels):
        """A URL whose host resolves to a loopback address is rejected with 422."""
        # Arrange — getaddrinfo returns 127.0.0.1

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LOOPBACK_ADDR):
            resp = client_runtime_panels.post(
                "/api/panels/register",
                json={"id": "internal", "label": "INT", "url": "http://grafana.lan:3000"},
            )

        # Assert
        assert resp.status_code == 422

    def test_register_non_http_scheme_returns_422(self, client_runtime_panels):
        """A URL with a non-http/https scheme is rejected with 422 without a DNS lookup."""
        # Arrange — ftp:// is not a valid panel URL scheme

        # Act
        resp = client_runtime_panels.post(
            "/api/panels/register",
            json={"id": "ftp-panel", "label": "FTP", "url": "ftp://files.example.com/"},
        )

        # Assert
        assert resp.status_code == 422

    def test_register_builtin_id_returns_422(self, client_runtime_panels):
        """Using a built-in panel id (e.g. 'ariel') for registration is rejected with 422."""
        # Arrange — "ariel" is in BUILTIN_PANELS

        # Act
        resp = client_runtime_panels.post(
            "/api/panels/register",
            json={"id": "ariel", "label": "ARIEL", "url": "http://grafana.lan:3000"},
        )

        # Assert
        assert resp.status_code == 422

    def test_register_host_not_in_allowlist_returns_422(self, client_runtime_panels_allowlist):
        """A host not in runtime_panel_allowlist is rejected with 422 even with a LAN address."""
        # Arrange — allowlist contains only "grafana.lan"; "other-host.lan" is not listed

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels_allowlist.post(
                "/api/panels/register",
                json={
                    "id": "other",
                    "label": "OTHER",
                    "url": "http://other-host.lan:3000",
                },
            )

        # Assert
        assert resp.status_code == 422

    def test_register_host_in_allowlist_succeeds(self, client_runtime_panels_allowlist):
        """A host present in runtime_panel_allowlist is accepted when the address is not blocked."""
        # Arrange — allowlist contains "grafana.lan"; URL host matches

        # Act
        with patch(_GETADDRINFO_TARGET, return_value=_LAN_ADDR):
            resp = client_runtime_panels_allowlist.post(
                "/api/panels/register",
                json={
                    "id": "grafana",
                    "label": "GRAFANA",
                    "url": "http://grafana.lan:3000",
                },
            )

        # Assert
        assert resp.status_code == 200
