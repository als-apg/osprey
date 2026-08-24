"""Tests for PATCH /api/config endpoint (comment-preserving config updates).

Uses a minimal FastAPI app with just the config routes to avoid lifespan
complexity (PTY, file watchers, etc.) that can crash in test environments.

These cover the *mechanics* of the patch — type handling, comment preservation,
the backup, the error shapes. They therefore drive the endpoint with keys it may
actually write: ``control_system.*`` and ``approval.*`` are in the protected set
and now come back 403, which is exercised in test_config_routes.py. Using one
here would turn a mechanics test into a refusal test without saying so, and
several of these assert on the file rather than the status, so they would pass
vacuously against a file nothing had touched.
"""

from __future__ import annotations

from pathlib import Path

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
  enabled: true

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
            json={"updates": {"artifact_server.auto_launch": False}},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["fields_updated"] == 1

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["artifact_server"]["auto_launch"] is False

    def test_patch_string_field(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={"updates": {"artifact_server.host": "0.0.0.0"}},
        )
        assert resp.status_code == 200

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["artifact_server"]["host"] == "0.0.0.0"

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
                    "artifact_server.auto_launch": False,
                    "artifact_server.host": "0.0.0.0",
                    "artifact_server.port": 7777,
                    "project_name": "renamed-project",
                }
            },
        )
        assert resp.status_code == 200
        assert resp.json()["fields_updated"] == 4

        data = yaml.safe_load((project_dir / "config.yml").read_text())
        assert data["artifact_server"]["auto_launch"] is False
        assert data["artifact_server"]["host"] == "0.0.0.0"
        assert data["artifact_server"]["port"] == 7777
        assert data["project_name"] == "renamed-project"

    def test_patch_preserves_comments(self, client, project_dir):
        resp = client.patch(
            "/api/config",
            json={"updates": {"artifact_server.auto_launch": False}},
        )
        # Asserted explicitly: every other assertion here is about the file, so a
        # refused patch would satisfy them all against untouched bytes.
        assert resp.status_code == 200
        text = (project_dir / "config.yml").read_text()
        assert "# ============================================================" in text
        assert "# Test Config" in text
        assert "# Comments must survive PATCH operations." in text
        assert "# Options: mock | epics" in text
        assert "# Master safety switch" in text

    def test_patch_creates_backup(self, client, project_dir):
        # Relocated, as the note here anticipated: the backup is a pre-write copy
        # of the old file exactly as before, but it lands in the agent-data state
        # zone instead of beside config.yml. SAMPLE_CONFIG names no
        # `agent_data.base_dir`, so the zone is the framework default anchored on
        # the project -- resolved through the same helpers the route uses rather
        # than spelled out, since the location following the *config* is the
        # property that matters (see test_config_routes.py for the relocation
        # case). Beside-the-config is asserted absent: the render is root-owned
        # after the container split, so a new file there is a 500, not a backup.
        from osprey.utils.workspace import agent_data_base_dir, anchored_path

        resp = client.patch(
            "/api/config",
            json={"updates": {"artifact_server.auto_launch": False}},
        )
        assert resp.status_code == 200

        zone = anchored_path(agent_data_base_dir(yaml.safe_load(SAMPLE_CONFIG)), project_dir)
        backup = zone / "config-backups" / "config.yml.bak"
        assert backup.exists()
        assert not (project_dir / "config.yml.bak").exists()
        backup_text = backup.read_text()
        assert "auto_launch: true" in backup_text

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
        # The replacement keeps SAMPLE_CONFIG's protected blocks (control_system,
        # approval) exactly as they are and moves only unprotected keys. PUT is
        # gated on a protected-set *document diff* now, so a body that dropped
        # them -- as this one used to -- is a refusal, and this test would be
        # asserting the gate rather than the raw-save mechanics it is about.
        new_yaml = SAMPLE_CONFIG.replace('project_name: "test-project"', "project_name: updated")
        new_yaml += "key: value\n"
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
