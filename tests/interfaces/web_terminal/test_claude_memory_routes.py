"""Tests for /api/claude-memory routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace_dir(tmp_path):
    ws = tmp_path / "osprey-workspace"
    ws.mkdir()
    return ws


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() to a temp directory."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


@pytest.fixture()
def client(workspace_dir, fake_home):
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


@pytest.fixture()
def memory_dir(client, fake_home):
    """Create the memory directory that the service will resolve to."""
    # The app's project_cwd is set by create_app; we need to figure out
    # the encoded path. Get it via the service.
    from osprey.interfaces.web_terminal.claude_memory_service import ClaudeMemoryService

    # Access project_cwd from the app state
    app = client.app
    project_cwd = app.state.project_cwd
    svc = ClaudeMemoryService(project_cwd)
    d = svc._resolve_memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestListMemoryFiles:
    def test_empty(self, client, memory_dir):
        resp = client.get("/api/claude-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["files"] == []
        assert data["count"] == 0

    def test_with_files(self, client, memory_dir):
        (memory_dir / "MEMORY.md").write_text("# Main\n", encoding="utf-8")
        (memory_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")

        resp = client.get("/api/claude-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        names = {f["filename"] for f in data["files"]}
        assert names == {"MEMORY.md", "notes.md"}


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class TestGetMemoryFile:
    def test_read_existing(self, client, memory_dir):
        (memory_dir / "test.md").write_text("hello\n", encoding="utf-8")
        resp = client.get("/api/claude-memory/test.md")
        assert resp.status_code == 200
        assert resp.json()["content"] == "hello\n"

    def test_read_nonexistent(self, client, memory_dir):
        resp = client.get("/api/claude-memory/missing.md")
        assert resp.status_code == 404

    def test_read_invalid_filename(self, client, memory_dir):
        resp = client.get("/api/claude-memory/.hidden.md")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreateMemoryFile:
    def test_create_new(self, client, memory_dir):
        resp = client.post(
            "/api/claude-memory",
            json={"filename": "new.md", "content": "# New\n"},
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "new.md"
        assert (memory_dir / "new.md").exists()

    def test_create_existing(self, client, memory_dir):
        (memory_dir / "exist.md").write_text("x\n", encoding="utf-8")
        resp = client.post(
            "/api/claude-memory",
            json={"filename": "exist.md", "content": "y\n"},
        )
        assert resp.status_code == 409

    def test_create_missing_filename(self, client, memory_dir):
        resp = client.post(
            "/api/claude-memory",
            json={"content": "x"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdateMemoryFile:
    def test_update_existing(self, client, memory_dir):
        (memory_dir / "test.md").write_text("old\n", encoding="utf-8")
        resp = client.put(
            "/api/claude-memory/test.md",
            json={"content": "new\n"},
        )
        assert resp.status_code == 200
        assert (memory_dir / "test.md").read_text(encoding="utf-8") == "new\n"

    def test_update_nonexistent(self, client, memory_dir):
        resp = client.put(
            "/api/claude-memory/missing.md",
            json={"content": "x"},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDeleteMemoryFile:
    def test_delete_existing(self, client, memory_dir):
        (memory_dir / "doomed.md").write_text("x\n", encoding="utf-8")
        resp = client.delete("/api/claude-memory/doomed.md")
        assert resp.status_code == 200
        assert not (memory_dir / "doomed.md").exists()

    def test_delete_nonexistent(self, client, memory_dir):
        resp = client.delete("/api/claude-memory/missing.md")
        assert resp.status_code == 404
