"""Tests for config-route auto-regeneration of Claude Code artifacts.

Complements test_config_patch.py (which tests the YAML mutation mechanics) by
verifying the GitHub #244 fix: a config write via PATCH/PUT re-renders the
``.claude/`` artifacts so safety-critical fields (the writes_enabled kill-switch)
actually take effect on the next terminal restart.

Uses a minimal FastAPI app over the routes router (no lifespan) pointed at a real
built project so ``regenerate_claude_code`` has artifacts to update.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal.routes import router
from osprey.utils.config_writer import config_update_fields


@pytest.fixture
def built_project(tmp_path):
    """A real OSPREY project with writes disabled (channel_write baked into deny)."""
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="route-regen",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )
    config_update_fields(project_dir / "config.yml", {"control_system.writes_enabled": False})
    manager.regenerate_claude_code(project_dir)
    return project_dir


@pytest.fixture
def client(built_project):
    app = FastAPI()
    app.include_router(router)
    app.state.config_path = built_project / "config.yml"
    app.state.project_cwd = str(built_project)
    with TestClient(app) as c:
        yield c


def _deny(project_dir):
    settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
    return settings["permissions"]["deny"]


class TestConfigRouteRegen:
    def test_patch_regenerates_artifacts(self, client, built_project):
        assert "mcp__controls__channel_write" in _deny(built_project)

        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": True}},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "regenerated" in body
        assert any("settings.json" in f for f in body["regenerated"])
        # The on-disk artifact the respawned agent reads is now in sync.
        assert "mcp__controls__channel_write" not in _deny(built_project)

    def test_patch_noop_when_already_in_sync(self, client, built_project):
        # Patching a non-safety field that is already at its value → nothing to regen.
        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": False}},
        )
        assert resp.status_code == 200
        assert resp.json()["regenerated"] == []
        assert "mcp__controls__channel_write" in _deny(built_project)

    def test_patch_fails_open_when_regen_raises(self, client, built_project, monkeypatch):
        """A regen error must never undo a config write that already succeeded."""

        def boom(self, project_dir):
            raise RuntimeError("regen exploded")

        monkeypatch.setattr(TemplateManager, "regen_if_drift", boom)
        resp = client.patch(
            "/api/config",
            json={"updates": {"control_system.writes_enabled": True}},
        )
        assert resp.status_code == 200
        assert resp.json()["regenerated"] == []
        # The config write persisted despite the regen failure.
        import yaml

        cfg = yaml.safe_load((built_project / "config.yml").read_text())
        assert cfg["control_system"]["writes_enabled"] is True

    def test_put_regenerates_artifacts(self, client, built_project):
        raw = (
            (built_project / "config.yml")
            .read_text()
            .replace("writes_enabled: false", "writes_enabled: true")
        )
        resp = client.put("/api/config", json={"raw": raw})
        assert resp.status_code == 200
        body = resp.json()
        assert "regenerated" in body
        assert any("settings.json" in f for f in body["regenerated"])
        assert "mcp__controls__channel_write" not in _deny(built_project)
