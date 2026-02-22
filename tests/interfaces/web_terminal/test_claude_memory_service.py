"""Tests for ClaudeMemoryService — reads/writes Claude Code native memory files."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.interfaces.web_terminal.claude_memory_service import (
    ClaudeMemoryService,
    MemoryFileExistsError,
    MemoryFileNotFoundError,
    MemoryValidationError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Redirect Path.home() to a temp directory."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    return tmp_path


@pytest.fixture()
def project_dir(tmp_path):
    """Return a fake project directory."""
    p = tmp_path / "projects" / "test-project"
    p.mkdir(parents=True)
    return p


@pytest.fixture()
def service(project_dir, fake_home):
    """Create a ClaudeMemoryService for the fake project."""
    return ClaudeMemoryService(project_dir)


@pytest.fixture()
def memory_dir(service):
    """Return the resolved memory directory, pre-created."""
    d = service._resolve_memory_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Path Resolution
# ---------------------------------------------------------------------------


class TestResolveMemoryDir:
    def test_encodes_path(self, service, fake_home):
        """Memory dir uses the encoded project path."""
        result = service._resolve_memory_dir()
        # Must be under ~/.claude/projects/<encoded>/memory/
        assert str(result).startswith(str(fake_home))
        assert result.name == "memory"
        assert ".claude/projects/" in str(result)

    def test_uses_resolved_path(self, tmp_path, fake_home):
        """Service resolves relative paths before encoding."""
        s = ClaudeMemoryService(tmp_path / "a" / ".." / "b")
        d = s._resolve_memory_dir()
        # The .. should be resolved away
        assert ".." not in str(d)


# ---------------------------------------------------------------------------
# List Files
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_empty_when_dir_missing(self, service):
        """Returns empty list when memory directory doesn't exist."""
        assert service.list_files() == []

    def test_empty_when_dir_empty(self, service, memory_dir):
        """Returns empty list when memory directory has no .md files."""
        assert service.list_files() == []

    def test_lists_md_files_only(self, service, memory_dir):
        """Only .md files are returned; other files are ignored."""
        (memory_dir / "MEMORY.md").write_text("# Main\n", encoding="utf-8")
        (memory_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
        (memory_dir / "data.json").write_text("{}", encoding="utf-8")
        (memory_dir / ".hidden.md").write_text("hidden\n", encoding="utf-8")

        files = service.list_files()
        names = {f["filename"] for f in files}
        # data.json excluded (not .md); .hidden.md excluded by glob pattern
        assert "MEMORY.md" in names
        assert "notes.md" in names
        assert "data.json" not in names

    def test_primary_flag(self, service, memory_dir):
        """MEMORY.md is flagged as primary."""
        (memory_dir / "MEMORY.md").write_text("x\n", encoding="utf-8")
        (memory_dir / "other.md").write_text("y\n", encoding="utf-8")

        files = service.list_files()
        primary = [f for f in files if f["is_primary"]]
        assert len(primary) == 1
        assert primary[0]["filename"] == "MEMORY.md"

    def test_line_count(self, service, memory_dir):
        """Line count is accurate."""
        (memory_dir / "test.md").write_text("line1\nline2\nline3\n", encoding="utf-8")
        files = service.list_files()
        assert files[0]["line_count"] == 3


# ---------------------------------------------------------------------------
# Read File
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_read_existing(self, service, memory_dir):
        """Read returns content and metadata."""
        (memory_dir / "test.md").write_text("# Test\nHello\n", encoding="utf-8")
        result = service.read_file("test.md")
        assert result["filename"] == "test.md"
        assert result["content"] == "# Test\nHello\n"
        assert result["line_count"] == 2
        assert result["is_primary"] is False

    def test_read_nonexistent_raises(self, service, memory_dir):
        with pytest.raises(MemoryFileNotFoundError):
            service.read_file("missing.md")

    def test_read_invalid_filename(self, service, memory_dir):
        with pytest.raises(MemoryValidationError):
            service.read_file("../escape.md")


# ---------------------------------------------------------------------------
# Create File
# ---------------------------------------------------------------------------


class TestCreateFile:
    def test_create_new(self, service, memory_dir):
        """Create writes file and returns metadata."""
        result = service.create_file("new-topic.md", "# New\n")
        assert result["filename"] == "new-topic.md"
        assert (memory_dir / "new-topic.md").read_text(encoding="utf-8") == "# New\n"

    def test_create_existing_raises(self, service, memory_dir):
        (memory_dir / "existing.md").write_text("x\n", encoding="utf-8")
        with pytest.raises(MemoryFileExistsError):
            service.create_file("existing.md", "y\n")

    def test_create_invalid_filename(self, service, memory_dir):
        with pytest.raises(MemoryValidationError):
            service.create_file("no-extension", "x")

    def test_create_creates_dir(self, service):
        """Memory directory is auto-created if it doesn't exist."""
        result = service.create_file("first.md", "# First\n")
        assert result["filename"] == "first.md"
        assert service._resolve_memory_dir().is_dir()


# ---------------------------------------------------------------------------
# Update File
# ---------------------------------------------------------------------------


class TestUpdateFile:
    def test_update_existing(self, service, memory_dir):
        (memory_dir / "test.md").write_text("old\n", encoding="utf-8")
        result = service.update_file("test.md", "new\n")
        assert result["filename"] == "test.md"
        assert (memory_dir / "test.md").read_text(encoding="utf-8") == "new\n"

    def test_update_nonexistent_raises(self, service, memory_dir):
        with pytest.raises(MemoryFileNotFoundError):
            service.update_file("missing.md", "x")


# ---------------------------------------------------------------------------
# Delete File
# ---------------------------------------------------------------------------


class TestDeleteFile:
    def test_delete_existing(self, service, memory_dir):
        (memory_dir / "doomed.md").write_text("x\n", encoding="utf-8")
        result = service.delete_file("doomed.md")
        assert result["deleted"] is True
        assert not (memory_dir / "doomed.md").exists()

    def test_delete_nonexistent_raises(self, service, memory_dir):
        with pytest.raises(MemoryFileNotFoundError):
            service.delete_file("missing.md")


# ---------------------------------------------------------------------------
# Filename Validation
# ---------------------------------------------------------------------------


class TestFilenameValidation:
    @pytest.mark.parametrize("filename", [
        "../escape.md",
        "../../etc/passwd",
        "path/traversal.md",
        "bad\\.md",
        "",
        "no-extension",
        ".hidden.md",
        "has spaces.md",
    ])
    def test_invalid_filenames(self, service, memory_dir, filename):
        with pytest.raises(MemoryValidationError):
            service.read_file(filename)

    @pytest.mark.parametrize("filename", [
        "MEMORY.md",
        "debugging.md",
        "my-notes.md",
        "topic_1.md",
        "A.md",
    ])
    def test_valid_filenames(self, service, memory_dir, filename):
        (memory_dir / filename).write_text("x\n", encoding="utf-8")
        result = service.read_file(filename)
        assert result["filename"] == filename
