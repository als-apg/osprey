"""Tests for OSPREY Web Terminal app factory and routes."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal.dock_layout import layout_keys
from tests.interfaces.web_terminal._started_app import terminal_client


@pytest.fixture
def client(workspace_dir):
    """Create a test client with mocked config active through lifespan."""
    with terminal_client(workspace_dir) as c:
        yield c


class TestProjectDir:
    def test_project_dir_sets_project_cwd(self, tmp_path, workspace_dir):
        """Verify create_app(project_dir=...) sets app.state.project_cwd."""
        project = tmp_path / "my-project"
        project.mkdir()

        with terminal_client(workspace_dir, project_dir=str(project)) as c:
            assert c.app.state.project_cwd == str(project.resolve())

    def test_default_project_cwd_is_cwd(self, workspace_dir):
        """Without project_dir, project_cwd defaults to os.getcwd()."""
        with terminal_client(workspace_dir) as c:
            from pathlib import Path

            assert c.app.state.project_cwd == str(Path.cwd().resolve())


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


#: A dock holding only the terminal: every service tile closed.
_TERMINAL_ONLY = {"tile": "t1", "tabs": ["terminal"], "showing": "terminal"}
#: A dock with ARTIFACTS beside the terminal.
_WITH_ARTIFACTS = {
    "split": "row",
    "items": [
        {"tile": "t1", "tabs": ["artifacts"], "showing": "artifacts", "size": 0.5},
        {"tile": "t2", "tabs": ["terminal"], "showing": "terminal", "size": 0.5},
    ],
}


def _service_keys(layout) -> list[str]:
    """The service keys of a reported dock ``layout`` in reading order, terminal excluded."""
    return [key for key in layout_keys(layout) if key != "terminal"]


def _report(client, layout, *, dock: bool = True) -> None:
    """POST a client's dock-layout report, asserting it was recorded."""
    resp = client.post(
        "/api/panel-layout", json={"layout": layout, "dock": dock, "client_id": "page-1"}
    )
    assert resp.status_code == 200, resp.text


class TestPanelsLayoutOccupancy:
    """``GET /api/panels`` reports tile occupancy and how stale it is.

    ``visible`` is launcher-rail membership; ``layout`` is the dock layout a
    browser last reported (``POST /api/panel-layout``). The two freshness
    companions exist so a consumer can tell "no client has ever reported" from
    "reported N seconds ago" instead of trusting a possibly-abandoned layout.

    The payload carries three distinct states that must never collapse into
    each other: never reported (all null), unknown occupancy (null layout, real
    age, dock false), and known occupancy (a tree, possibly terminal-only).
    """

    def test_panels_layout_all_null_before_any_report(self, client):
        """Never reported is all-null — emphatically not a known-empty screen."""
        body = client.get("/api/panels").json()
        assert body["layout"] is None
        assert body["layout_age_s"] is None
        assert body["layout_dock"] is None

    def test_panels_dock_less_report_is_unknown_occupancy(self, client):
        """A watching-but-blind client: null layout, real age, dock false."""
        _report(client, None, dock=False)
        body = client.get("/api/panels").json()
        assert body["layout"] is None
        assert body["layout_age_s"] is not None
        assert body["layout_dock"] is False

    def test_panels_known_empty_is_distinct_from_unknown(self, client):
        """A terminal-only dock means the operator closed every service tile."""
        _report(client, _TERMINAL_ONLY)
        body = client.get("/api/panels").json()
        assert _service_keys(body["layout"]) == []
        assert body["layout_dock"] is True

    def test_panels_layout_reflects_the_last_report(self, client):
        _report(client, _WITH_ARTIFACTS)
        body = client.get("/api/panels").json()
        assert _service_keys(body["layout"]) == ["artifacts"]
        assert body["layout_dock"] is True

    def test_panels_layout_age_is_seconds_since_the_report(self, client):
        import time

        _report(client, _WITH_ARTIFACTS)
        fresh = client.get("/api/panels").json()["layout_age_s"]
        assert 0 <= fresh < 60

        # Age the stored report; the payload must report it as stale, not fresh.
        client.app.state.layout_ts = time.time() - 3600
        aged = client.get("/api/panels").json()["layout_age_s"]
        assert aged >= 3600

    def test_panels_layout_is_independent_of_rail_membership(self, client):
        """Closing every tile leaves the rail alone — occupancy is not membership."""
        before = client.get("/api/panels").json()["visible"]
        client.post("/api/panel-layout", json={"tiles": [], "dock": True})
        after = client.get("/api/panels").json()
        assert _service_keys(after["layout"]) == []
        assert after["visible"] == before


class TestHeaderAppName:
    """web.app_name surfaces as the deployment line of the ☰ More menu."""

    def test_app_name_renders_when_set(self, workspace_dir):
        with terminal_client(
            workspace_dir,
            patch(
                "osprey.interfaces.web_terminal.app._load_web_ui_config",
                return_value={"app_name": "Control Room A"},
            ),
        ) as c:
            assert c.app.state.app_name == "Control Room A"
            body = c.get("/").text
            assert "more-menu-sub" in body
            assert "Control Room A" in body

    def test_app_name_absent_when_unset(self, client):
        # The shared `client` fixture supplies no `web` section.
        body = client.get("/").text
        assert "more-menu-sub" not in body

    def test_env_var_overrides_config(self, workspace_dir):
        # OSPREY_WEB_APP_NAME wins over web.app_name so containers sharing one
        # baked config image can still be named individually.
        with terminal_client(
            workspace_dir,
            patch(
                "osprey.interfaces.web_terminal.app._load_web_ui_config",
                return_value={"app_name": "From Config"},
            ),
            patch.dict("os.environ", {"OSPREY_WEB_APP_NAME": "From Env"}),
        ) as c:
            assert c.app.state.app_name == "From Env"
            assert "From Env" in c.get("/").text
