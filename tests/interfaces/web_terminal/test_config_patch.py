"""Tests for PATCH /api/config endpoint (comment-preserving config updates).

Uses a minimal FastAPI app with just the config routes to avoid lifespan
complexity (PTY, file watchers, etc.) that can crash in test environments.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.routes import router

SAMPLE_CONFIG = """\
# ============================================================
# Test Config
# ============================================================
# Comments must survive PATCH operations.

project_name: "test-project"

control_system:
  type: "mock"  # Options: mock | epics
  writes_enabled: false  # Master safety switch
  limits_checking:
    enabled: false
    on_violation: "skip"

approval:
  global_mode: "selective"

artifact_server:
  host: "127.0.0.1"
  port: 8086
  auto_launch: true
"""


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project with config.yml."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(project_dir):
    """Minimal FastAPI test client with just the routes router and config state."""
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = project_dir / "config.yml"
    app.state.project_cwd = str(project_dir)
    with TestClient(app) as c:
        yield c


class TestPatchEndpoint:
    """Test PATCH /api/config for structured field updates."""

    def test_patch_boolean_field(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": True}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["fields_updated"] == 1

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["control_system"]["writes_enabled"] is True

    def test_patch_string_field(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.type": "epics"}},
        )
        assert resp.status_code == 200

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["control_system"]["type"] == "epics"

    def test_patch_numeric_field(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={"updates": {"artifact_server.port": 9999}},
        )
        assert resp.status_code == 200

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["artifact_server"]["port"] == 9999

    def test_patch_multiple_fields(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={
                "updates": {
                    "control_system.writes_enabled": True,
                    "control_system.type": "epics",
                    "artifact_server.port": 7777,
                    "approval.global_mode": "disabled",
                }
            },
        )
        assert resp.status_code == 200
        assert resp.json()["fields_updated"] == 4

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["control_system"]["writes_enabled"] is True
        assert data["control_system"]["type"] == "epics"
        assert data["artifact_server"]["port"] == 7777
        assert data["approval"]["global_mode"] == "disabled"

    def test_patch_preserves_comments(self, client, project_dir):
        client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": True}},
        )
        text = (project_dir / "config.yml").read_text()
        assert "# ============================================================" in text
        assert "# Test Config" in text
        assert "# Comments must survive PATCH operations." in text
        assert "# Options: mock | epics" in text
        assert "# Master safety switch" in text

    def test_patch_creates_backup(self, client, project_dir):
        client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": True}},
        )
        backup = project_dir / "config.yml.bak"
        assert backup.exists()
        backup_text = backup.read_text()
        assert "writes_enabled: false" in backup_text

    def test_patch_empty_updates_rejected(self, client):
        resp = client.patch("/api/config", json={"updates": {}})
        assert resp.status_code == 422

    def test_patch_no_config_file(self, client):
        client.app.state.config_path = Path("/nonexistent/config.yml")
        resp = client.patch(
            "/api/config",
            json={"updates": {"key": "value"}},
        )
        assert resp.status_code == 404

    def test_patch_preserves_key_order(self, client, project_dir):
        original = yaml.safe_load((project_dir / "config.yml").read_text())
        original_keys = list(original.keys())

        client.patch(
            "/api/config",
            json={"updates": {"artifact_server.port": 1234}},
        )

        updated = yaml.safe_load((project_dir / "config.yml").read_text())
        updated_keys = list(updated.keys())
        assert original_keys == updated_keys


class TestGetEndpoint:
    """Verify GET /api/config still works."""

    def test_get_returns_sections_and_raw(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200
        body = resp.json()
        assert "sections" in body
        assert "raw" in body
        assert "path" in body
        assert "# Test Config" in body["raw"]


class TestPutEndpointStillWorks:
    """Ensure the existing PUT /api/config still works for raw YAML saves."""

    def test_put_raw_yaml(self, client, project_dir):
        new_yaml = "project_name: updated\nkey: value\n"
        resp = client.put(
            "/api/config",
            json={"raw": new_yaml},
        )
        assert resp.status_code == 200
        assert resp.json()["requires_restart"] is True

        text = (project_dir / "config.yml").read_text()
        assert "project_name: updated" in text

    def test_put_invalid_yaml_rejected(self, client):
        resp = client.put(
            "/api/config",
            json={"raw": "invalid: yaml: [unterminated"},
        )
        assert resp.status_code == 422
