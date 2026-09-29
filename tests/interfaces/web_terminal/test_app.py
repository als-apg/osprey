"""Tests for OSPREY Web Terminal app factory and routes."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import yaml
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app


@pytest.fixture
def workspace_dir(tmp_path):
    """An empty watched directory."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def client(workspace_dir):
    """Create a test client with mocked config active through lifespan."""
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command=["echo"])
        with TestClient(app) as c:
            yield c


class TestProjectDir:
    def test_project_dir_sets_project_cwd(self, tmp_path, workspace_dir):
        """Verify create_app(project_dir=...) sets app.state.project_cwd."""
        project = tmp_path / "my-project"
        project.mkdir()

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command=["echo"], project_dir=str(project))
            with TestClient(app):
                assert app.state.project_cwd == str(project.resolve())

    def test_default_project_cwd_is_cwd(self, workspace_dir):
        """Without project_dir, project_cwd defaults to os.getcwd()."""
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command=["echo"])
            with TestClient(app):
                from pathlib import Path

                assert app.state.project_cwd == str(Path.cwd().resolve())


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "web_terminal"

    def test_health_reports_ca_user_in_process(self, client, monkeypatch):
        """``ca_user`` is the name this process's uid resolves to, read live."""
        import os
        import pwd

        monkeypatch.delenv("OSPREY_CONTROL_IDENTITY_SKIPPED", raising=False)
        data = client.get("/health").json()
        assert data["ca_user"] == pwd.getpwuid(os.getuid()).pw_name
        assert data["control_identity_skipped"] is None

    def test_health_reports_skipped_identity(self, client, monkeypatch):
        """The entrypoint's skip reason reaches ``osprey health`` unchanged."""
        monkeypatch.setenv("OSPREY_CONTROL_IDENTITY_SKIPPED", "non-root-start")
        data = client.get("/health").json()
        assert data["control_identity_skipped"] == "non-root-start"


class TestPanelFocus:
    def test_get_panel_focus_default_none(self, client):
        resp = client.get("/api/panel-focus")
        assert resp.status_code == 200
        assert resp.json()["active_panel"] is None


class TestPanelsOpenTiles:
    """``GET /api/panels`` reports tile occupancy and how stale it is.

    ``visible`` is launcher-rail membership; ``open_tiles`` is what a browser
    last reported as actually on screen. The two freshness companions exist so
    a consumer can tell "no client has ever reported" from "reported N seconds
    ago" instead of trusting a possibly-abandoned list.

    The payload carries three distinct states that must never collapse into
    each other: never reported (all null), unknown occupancy (null list, real
    age, dock false), and known occupancy (a list, possibly empty).
    """

    def test_panels_open_tiles_all_null_before_any_report(self, client):
        """Never reported is all-null — emphatically not a known-empty screen."""
        body = client.get("/api/panels").json()
        assert body["open_tiles"] is None
        assert body["open_tiles_age_s"] is None
        assert body["open_tiles_dock"] is None

    def test_panels_dock_less_report_is_unknown_occupancy(self, client):
        """A watching-but-blind client: null tiles, real age, dock false."""
        client.post("/api/panel-layout", json={"tiles": [], "dock": False})
        body = client.get("/api/panels").json()
        assert body["open_tiles"] is None
        assert body["open_tiles_age_s"] is not None
        assert body["open_tiles_dock"] is False

    def test_panels_known_empty_is_distinct_from_unknown(self, client):
        """A dock client reporting [] means the operator closed everything."""
        client.post("/api/panel-layout", json={"tiles": [], "dock": True})
        body = client.get("/api/panels").json()
        assert body["open_tiles"] == []
        assert body["open_tiles_dock"] is True

    def test_panels_open_tiles_reflect_the_last_report(self, client):
        client.post("/api/panel-layout", json={"tiles": ["artifacts"], "dock": True})
        body = client.get("/api/panels").json()
        assert body["open_tiles"] == ["artifacts"]
        assert body["open_tiles_dock"] is True

    def test_panels_open_tiles_age_is_seconds_since_the_report(self, client):
        import time

        client.post("/api/panel-layout", json={"tiles": ["artifacts"], "dock": True})
        fresh = client.get("/api/panels").json()["open_tiles_age_s"]
        assert 0 <= fresh < 60

        # Age the stored report; the payload must report it as stale, not fresh.
        client.app.state.open_tiles_ts = time.time() - 3600
        aged = client.get("/api/panels").json()["open_tiles_age_s"]
        assert aged >= 3600

    def test_panels_open_tiles_are_independent_of_rail_membership(self, client):
        """Closing every tile leaves the rail alone — occupancy is not membership."""
        before = client.get("/api/panels").json()["visible"]
        client.post("/api/panel-layout", json={"tiles": [], "dock": True})
        after = client.get("/api/panels").json()
        assert after["open_tiles"] == []
        assert after["visible"] == before


class TestHeaderAppName:
    """web.app_name surfaces as an optional header badge for deployment ID."""

    def test_app_name_renders_when_set(self, workspace_dir):
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch(
                "osprey.interfaces.web_terminal.app._load_web_ui_config",
                return_value={"app_name": "Control Room A"},
            ),
        ):
            app = create_app(shell_command=["echo"])
            with TestClient(app) as c:
                assert app.state.app_name == "Control Room A"
                body = c.get("/").text
                assert "header-deployment" in body
                assert "Control Room A" in body

    def test_app_name_absent_when_unset(self, client):
        # The shared `client` fixture supplies no `web` section.
        body = client.get("/").text
        assert "header-deployment" not in body

    def test_env_var_overrides_config(self, workspace_dir):
        # OSPREY_WEB_APP_NAME wins over web.app_name so containers sharing one
        # baked config image can still be named individually.
        cfg = {"watch_dir": str(workspace_dir)}
        with (
            patch(
                "osprey.interfaces.web_terminal.app._load_web_config",
                return_value=cfg,
            ),
            patch(
                "osprey.interfaces.web_terminal.app._load_web_ui_config",
                return_value={"app_name": "From Config"},
            ),
            patch.dict("os.environ", {"OSPREY_WEB_APP_NAME": "From Env"}),
        ):
            app = create_app(shell_command=["echo"])
            with TestClient(app) as c:
                assert app.state.app_name == "From Env"
                assert "From Env" in c.get("/").text


class TestHookDebugRoutes:
    """``/api/hooks/debug-status`` and ``/api/hooks/debug-log`` read the project live."""

    @staticmethod
    def _client(tmp_path, debug: bool):
        config_file = tmp_path / "config.yml"
        config_file.write_text(yaml.dump({"hooks": {"debug": debug}}))
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(tmp_path / "ws")},
        ):
            app = create_app(
                config_path=str(config_file),
                shell_command=["echo"],
                project_dir=str(tmp_path),
            )
        return TestClient(app)

    @pytest.mark.parametrize("debug", [True, False])
    def test_debug_status_reads_the_config(self, tmp_path, monkeypatch, debug):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        with self._client(tmp_path, debug) as client:
            resp = client.get("/api/hooks/debug-status")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is debug

    def test_debug_log_returns_entries_newest_first(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        log_dir = tmp_path / ".claude" / "hooks"
        log_dir.mkdir(parents=True)
        entries = [
            {
                "ts": "2026-03-02T10:00:00Z",
                "hook": "PreToolUse",
                "tool": "Bash",
                "status": "allowed",
            },
            {
                "ts": "2026-03-02T10:00:01Z",
                "hook": "PreToolUse",
                "tool": "Write",
                "status": "blocked",
                "detail": "safety check",
            },
        ]
        (log_dir / "hook_debug.jsonl").write_text("\n".join(json.dumps(e) for e in entries))

        with self._client(tmp_path, True) as client:
            resp = client.get("/api/hooks/debug-log?limit=50")

        assert resp.status_code == 200
        assert [e["tool"] for e in resp.json()["entries"]] == ["Write", "Bash"]

    def test_debug_log_is_empty_without_a_log_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        with self._client(tmp_path, True) as client:
            resp = client.get("/api/hooks/debug-log")
        assert resp.status_code == 200
        assert resp.json()["entries"] == []
