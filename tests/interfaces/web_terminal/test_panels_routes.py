"""Tests for the panels payload route (`routes/panels.py`).

Task 3.3 (project-key-endpoint): ``GET /api/panels`` carries a stable, opaque
``project_key`` — a truncated sha256 of the resolved project directory — that
the client uses as the ``osprey-dock-layout-<project_key>`` localStorage suffix
for per-project dock-layout persistence.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes.panels import router

_HEX16 = re.compile(r"\A[0-9a-f]{16}\Z")


def _make_app(project_cwd) -> FastAPI:
    """A minimal app exposing the panels router with a set ``project_cwd``.

    Every other field read by ``get_panels`` is accessed via ``getattr`` with a
    default, so a bare state carrying only ``project_cwd`` exercises the route.
    """
    app = FastAPI()
    app.include_router(router)
    app.state.project_cwd = str(project_cwd)
    return app


def _panels(app: FastAPI) -> dict:
    with TestClient(app) as client:
        resp = client.get("/api/panels")
    assert resp.status_code == 200
    return resp.json()


def test_project_key_present_and_shape(tmp_path):
    """The payload carries a 16-char lowercase-hex ``project_key``."""
    body = _panels(_make_app(tmp_path))
    assert "project_key" in body
    assert isinstance(body["project_key"], str)
    assert _HEX16.match(body["project_key"])


def test_project_key_matches_resolved_sha256(tmp_path):
    """The key is the first 16 hex chars of sha256(resolved project dir)."""
    expected = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()[:16]
    body = _panels(_make_app(tmp_path))
    assert body["project_key"] == expected


def test_project_key_stable_across_requests(tmp_path):
    """Two requests to one app return the same key (restart-stable)."""
    app = _make_app(tmp_path)
    assert _panels(app)["project_key"] == _panels(app)["project_key"]


def test_project_key_stable_across_equivalent_paths(tmp_path):
    """Equivalent paths (trailing slash / ``.`` segment) resolve to one key."""
    plain = _panels(_make_app(tmp_path))["project_key"]
    trailing = _panels(_make_app(str(tmp_path) + "/"))["project_key"]
    dotted = _panels(_make_app(tmp_path / "."))["project_key"]
    assert plain == trailing == dotted


def test_project_key_distinct_across_directories(tmp_path):
    """Two apps with different project dirs return different keys."""
    dir_a = tmp_path / "project-a"
    dir_b = tmp_path / "project-b"
    dir_a.mkdir()
    dir_b.mkdir()
    key_a = _panels(_make_app(dir_a))["project_key"]
    key_b = _panels(_make_app(dir_b))["project_key"]
    assert key_a != key_b


def test_project_key_does_not_disturb_existing_fields(tmp_path):
    """Existing payload fields are unchanged; project_key is purely additive."""
    app = _make_app(tmp_path)
    app.state.enabled_panels = {"ariel", "channels"}
    app.state.custom_panels = [{"id": "grafana", "label": "GRAFANA", "url": "http://x:3000"}]
    app.state.default_panel = "ariel"
    app.state.visible_panels = ["ariel"]
    app.state.active_panel = "ariel"
    app.state.allow_runtime_panels = True
    app.state.panel_presets = [{"name": "L1", "panels": ["ariel"]}]
    app.state.web_ui_mode = "simple"

    body = _panels(app)

    assert set(body["enabled"]) == {"ariel", "channels"}
    assert body["custom"] == [{"id": "grafana", "label": "GRAFANA", "url": "/panel/grafana"}]
    assert body["default"] == "ariel"
    assert body["visible"] == ["ariel"]
    assert body["active"] == "ariel"
    assert body["allow_runtime_panels"] is True
    assert body["presets"] == [{"name": "L1", "panels": ["ariel"]}]
    assert body["ui_mode"] == "simple"
    # The additive key sits alongside the established shape.
    assert set(body) == {
        "enabled",
        "custom",
        "default",
        "visible",
        "active",
        "labels",
        "allow_runtime_panels",
        "presets",
        "ui_mode",
        "project_key",
    }


def test_project_key_matches_documented_construction(tmp_path):
    """Key equals sha256(resolved path)[:16] — guards against digest drift."""
    body = _panels(_make_app(tmp_path))
    full = hashlib.sha256(str(Path(tmp_path).resolve()).encode("utf-8")).hexdigest()
    assert body["project_key"] == full[:16]
    assert len(body["project_key"]) == 16
