"""Tests for /api/claude-memory routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from osprey.agent_runner.project_paths import claude_project_dir
from osprey.interfaces.web_terminal.app import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace_dir(tmp_path):
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Redirect ``Path.home()`` to a temp directory and unset ``CLAUDE_CONFIG_DIR``.

    The memory directory lives under ``$CLAUDE_CONFIG_DIR`` when it is set, so a
    developer or container shell exporting it would otherwise aim these tests at
    a real Claude Code state root.
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return tmp_path


# ``fake_home`` redirects ``Path.home()``, which is where the memory routes resolve their
# directory.
@pytest.fixture()
def client(workspace_dir, fake_home):  # noqa: ARG001
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(workspace_dir)},
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as c:
            yield c


def _memory_dir_of(client) -> Path:
    """Where Claude Code keeps this app's project memory."""
    return claude_project_dir(client.app.state.project_cwd) / "memory"


@pytest.fixture()
def memory_dir(client):
    """The memory directory the routes resolve to, created."""
    d = _memory_dir_of(client)
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


class TestListMemoryFiles:
    @pytest.mark.usefixtures("memory_dir")
    def test_empty(self, client):
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

    def test_empty_when_the_memory_dir_is_missing(self, client):
        assert not _memory_dir_of(client).exists()

        resp = client.get("/api/claude-memory")

        assert resp.status_code == 200
        assert resp.json() == {"files": [], "count": 0}

    def test_lists_only_what_the_gallery_can_open(self, client, memory_dir):
        """The listing and the per-file routes share one filename grammar.

        A dot-prefixed ``.md`` matches a ``*.md`` glob but is refused by every
        per-file route, so listing it would show an entry the operator can
        neither open nor delete.
        """
        (memory_dir / "MEMORY.md").write_text("# Main\n", encoding="utf-8")
        (memory_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
        (memory_dir / "data.json").write_text("{}", encoding="utf-8")
        (memory_dir / ".hidden.md").write_text("hidden\n", encoding="utf-8")

        names = {f["filename"] for f in client.get("/api/claude-memory").json()["files"]}

        assert names == {"MEMORY.md", "notes.md"}
        assert client.get("/api/claude-memory/.hidden.md").status_code == 422

    def test_primary_flag_and_line_count(self, client, memory_dir):
        (memory_dir / "MEMORY.md").write_text("x\n", encoding="utf-8")
        (memory_dir / "other.md").write_text("line1\nline2\nline3", encoding="utf-8")

        files = {f["filename"]: f for f in client.get("/api/claude-memory").json()["files"]}

        assert files["MEMORY.md"]["is_primary"] is True
        assert files["other.md"]["is_primary"] is False
        # A final line without a newline still counts.
        assert files["other.md"]["line_count"] == 3

    def test_honours_claude_config_dir(self, client, tmp_path, monkeypatch):
        """Memory lives under ``$CLAUDE_CONFIG_DIR/projects``, not ``~/.claude``.

        The per-user web-terminal container sets ``CLAUDE_CONFIG_DIR`` and
        ``HOME`` to one mounted volume; the ``~/.claude`` spelling then names a
        directory Claude Code never writes, and the gallery reads empty.
        """
        config_dir = tmp_path / "data" / "claude-config"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
        memory = _memory_dir_of(client)
        assert memory.parent.parent == config_dir / "projects"
        memory.mkdir(parents=True)
        (memory / "x.md").write_text("x\n", encoding="utf-8")

        names = [f["filename"] for f in client.get("/api/claude-memory").json()["files"]]

        assert names == ["x.md"]


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


class TestGetMemoryFile:
    def test_read_existing(self, client, memory_dir):
        (memory_dir / "test.md").write_text("# Test\nHello\n", encoding="utf-8")
        resp = client.get("/api/claude-memory/test.md")
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test.md"
        assert body["content"] == "# Test\nHello\n"
        assert body["line_count"] == 2
        assert body["is_primary"] is False

    @pytest.mark.usefixtures("memory_dir")
    def test_read_nonexistent(self, client):
        resp = client.get("/api/claude-memory/missing.md")
        assert resp.status_code == 404

    @pytest.mark.usefixtures("memory_dir")
    def test_read_invalid_filename(self, client):
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
        assert (memory_dir / "new.md").read_text(encoding="utf-8") == "# New\n"

    def test_create_makes_the_memory_dir(self, client):
        resp = client.post("/api/claude-memory", json={"filename": "first.md", "content": "# 1\n"})

        assert resp.status_code == 200
        assert (_memory_dir_of(client) / "first.md").is_file()

    def test_create_existing(self, client, memory_dir):
        (memory_dir / "exist.md").write_text("x\n", encoding="utf-8")
        resp = client.post(
            "/api/claude-memory",
            json={"filename": "exist.md", "content": "y\n"},
        )
        assert resp.status_code == 409

    @pytest.mark.usefixtures("memory_dir")
    def test_create_missing_filename(self, client):
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

    @pytest.mark.usefixtures("memory_dir")
    def test_update_nonexistent(self, client):
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
        assert resp.json()["deleted"] is True
        assert not (memory_dir / "doomed.md").exists()

    @pytest.mark.usefixtures("memory_dir")
    def test_delete_nonexistent(self, client):
        resp = client.delete("/api/claude-memory/missing.md")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Filename grammar
# ---------------------------------------------------------------------------

#: Names every operation refuses. None carries a ``/``, so each reaches the
#: service through the ``{filename}`` path segment as well as a POST body.
INVALID_NAMES = ["bad\\.md", "no-extension", ".hidden.md", "has spaces.md"]

#: Names that carry a separator: only a POST body can deliver them.
SEPARATOR_NAMES = ["../escape.md", "../../etc/passwd", "path/traversal.md"]

VALID_NAMES = ["MEMORY.md", "debugging.md", "my-notes.md", "topic_1.md", "A.md", "notes..v2.md"]


class TestFilenameValidation:
    @pytest.mark.usefixtures("memory_dir")
    @pytest.mark.parametrize("name", INVALID_NAMES)
    @pytest.mark.parametrize("verb", ["get", "put", "delete"])
    def test_an_invalid_name_is_refused_on_every_path_operation(self, client, verb, name):
        url = f"/api/claude-memory/{quote(name, safe='')}"
        if verb == "put":
            resp = client.put(url, json={"content": "x"})
        else:
            resp = getattr(client, verb)(url)

        assert resp.status_code == 422

    @pytest.mark.parametrize("name", INVALID_NAMES + SEPARATOR_NAMES)
    def test_an_invalid_name_is_refused_on_create(self, client, name):
        resp = client.post("/api/claude-memory", json={"filename": name, "content": "x"})

        assert resp.status_code == 422
        assert not _memory_dir_of(client).exists()

    @pytest.mark.parametrize("name", VALID_NAMES)
    def test_valid_names_are_served(self, client, memory_dir, name):
        (memory_dir / name).write_text("x\n", encoding="utf-8")

        resp = client.get(f"/api/claude-memory/{name}")

        assert resp.status_code == 200
        assert resp.json()["filename"] == name
