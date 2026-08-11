"""Tests for OSPREY Web Terminal app factory and routes."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory with sample files."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()

    # Create sample files
    scripts = ws / "scripts"
    scripts.mkdir()
    (scripts / "analysis.py").write_text("import numpy as np\nprint('hello')\n")

    data = ws / "data"
    data.mkdir()
    (data / "results.json").write_text('{"key": "value"}\n')

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


class TestAppCreation:
    def test_create_app_returns_fastapi(self):
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={},
        ):
            app = create_app()
        assert isinstance(app, FastAPI)
        assert app.title == "OSPREY Web Terminal"

    def test_create_app_with_custom_shell(self):
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={},
        ):
            app = create_app(shell_command="zsh")
        assert isinstance(app, FastAPI)


class TestProjectDir:
    def test_project_dir_sets_project_cwd(self, tmp_path, workspace_dir):
        """Verify create_app(project_dir=...) sets app.state.project_cwd."""
        project = tmp_path / "my-project"
        project.mkdir()

        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo", project_dir=str(project))
            with TestClient(app):
                assert app.state.project_cwd == str(project.resolve())

    def test_default_project_cwd_is_cwd(self, workspace_dir):
        """Without project_dir, project_cwd defaults to os.getcwd()."""
        with patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ):
            app = create_app(shell_command="echo")
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


class TestFileTreeEndpoint:
    def test_file_tree_returns_structure(self, client):
        resp = client.get("/api/files/tree")
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "directory"
        assert "children" in data

    def test_file_tree_contains_files(self, client):
        resp = client.get("/api/files/tree")
        data = resp.json()
        names = [c["name"] for c in data["children"]]
        assert "scripts" in names
        assert "data" in names
        assert "README.md" in names

    def test_directories_sorted_before_files(self, client):
        resp = client.get("/api/files/tree")
        data = resp.json()
        types = [c["type"] for c in data["children"]]
        dir_indices = [i for i, t in enumerate(types) if t == "directory"]
        file_indices = [i for i, t in enumerate(types) if t == "file"]
        if dir_indices and file_indices:
            assert max(dir_indices) < min(file_indices)


class TestFileContentEndpoint:
    def test_read_file_content(self, client):
        resp = client.get("/api/files/content/README.md")
        assert resp.status_code == 200
        data = resp.json()
        assert data["path"] == "README.md"
        assert "# Test workspace" in data["content"]
        assert data["extension"] == ".md"

    def test_read_nested_file(self, client):
        resp = client.get("/api/files/content/scripts/analysis.py")
        assert resp.status_code == 200
        data = resp.json()
        assert "import numpy" in data["content"]

    def test_path_traversal_blocked(self, client):
        # URL-level .. is normalized by the HTTP framework, so we test
        # with a path that bypasses URL normalization
        resp = client.get("/api/files/content/../../../etc/passwd")
        # Framework normalizes the path, so we get 403 or 404
        assert resp.status_code in (403, 404)

    def test_path_traversal_encoded(self, client):
        resp = client.get("/api/files/content/..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (403, 404)

    def test_nonexistent_file_404(self, client):
        resp = client.get("/api/files/content/does_not_exist.txt")
        assert resp.status_code == 404

    def test_directory_not_a_file(self, client):
        resp = client.get("/api/files/content/scripts")
        assert resp.status_code == 400


class TestPanelFocus:
    def test_get_panel_focus_default_none(self, client):
        resp = client.get("/api/panel-focus")
        assert resp.status_code == 200
        assert resp.json()["active_panel"] is None

    def test_set_panel_focus_artifacts(self, client):
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "artifacts"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["active_panel"] == "artifacts"

    def test_get_reflects_set(self, client):
        client.post("/api/panel-focus", json={"panel": "artifacts"})
        resp = client.get("/api/panel-focus")
        assert resp.json()["active_panel"] == "artifacts"

    def test_set_unknown_panel_422(self, client):
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "unknown"},
        )
        assert resp.status_code == 422

    def test_set_panel_focus_with_url(self, client):
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "artifacts", "url": "http://localhost:8086/gallery"},
        )
        assert resp.status_code == 200
        assert resp.json()["active_panel"] == "artifacts"

    def test_set_panel_focus_broadcasts_event(self, client):
        """SSE broadcast should be called with a panel_focus event."""
        app = client.app
        broadcaster = app.state.broadcaster

        # Subscribe before sending
        q = broadcaster.subscribe()

        client.post("/api/panel-focus", json={"panel": "artifacts"})

        # The event should be in the queue
        assert not q.empty()
        event = q.get_nowait()
        assert event["type"] == "panel_focus"
        assert event["panel"] == "artifacts"

        broadcaster.unsubscribe(q)


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


class TestStaticServing:
    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]


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
            app = create_app(shell_command="echo")
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
            app = create_app(shell_command="echo")
            with TestClient(app) as c:
                assert app.state.app_name == "From Env"
                assert "From Env" in c.get("/").text
