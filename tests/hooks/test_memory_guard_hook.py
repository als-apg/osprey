"""Tests for the osprey_memory_guard hook.

This hook restricts the Write tool to Claude Code native memory files only.
Writes targeting ``~/.claude/projects/<encoded>/memory/*.md`` are allowed;
everything else is denied. Non-Write tools pass through without opinion.
"""

from pathlib import Path

import pytest


def _memory_dir_for(project_dir: str) -> Path:
    """Mirror the hook's resolve_memory_dir logic for test assertions."""
    abs_project = str(Path(project_dir).resolve())
    encoded = abs_project.replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded / "memory"


# -- Allow cases --


@pytest.mark.unit
def test_allows_write_to_memory_md(tmp_path, hook_runner):
    """Write to a .md file inside the Claude memory directory is allowed."""
    memory_dir = _memory_dir_for(str(tmp_path))
    target = str(memory_dir / "channels.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# BPM Channels\n"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


@pytest.mark.unit
def test_allows_primary_memory_file(tmp_path, hook_runner):
    """MEMORY.md (the primary memory file) is allowed."""
    memory_dir = _memory_dir_for(str(tmp_path))
    target = str(memory_dir / "MEMORY.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# Memories\n"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


# -- Deny cases --


@pytest.mark.unit
def test_denies_write_to_arbitrary_path(tmp_path, hook_runner):
    """Write to an arbitrary file outside memory directory is denied."""
    target = str(tmp_path / "evil.py")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "import os; os.system('rm -rf /')"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_write_to_project_claude_md(tmp_path, hook_runner):
    """Write to the project-level CLAUDE.md (system prompt) is denied."""
    target = str(tmp_path / "CLAUDE.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "overwrite system prompt"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_non_md_in_memory_dir(tmp_path, hook_runner):
    """Write to the memory directory but with a non-.md extension is denied."""
    memory_dir = _memory_dir_for(str(tmp_path))
    target = str(memory_dir / "script.py")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "import evil"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_subdirectory_in_memory_dir(tmp_path, hook_runner):
    """Write to a subdirectory within the memory dir is denied (direct children only)."""
    memory_dir = _memory_dir_for(str(tmp_path))
    target = str(memory_dir / "subdir" / "notes.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "sneaky"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_path_traversal(tmp_path, hook_runner):
    """Write with path traversal components is denied."""
    memory_dir = _memory_dir_for(str(tmp_path))
    target = str(memory_dir / ".." / ".." / ".." / ".bashrc")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "alias rm='rm -rf /'"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_empty_file_path(tmp_path, hook_runner):
    """Write with no file_path is denied."""
    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": "", "content": "something"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_missing_file_path(tmp_path, hook_runner):
    """Write with no file_path key at all is denied."""
    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"content": "something"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


# -- Pass-through cases --


@pytest.mark.unit
def test_non_write_tool_passes_through(tmp_path, hook_runner):
    """Non-Write tools are not affected by this hook."""
    result = hook_runner(
        "osprey_memory_guard.py",
        "Read",
        {"file_path": "/etc/passwd"},
        cwd=tmp_path,
    )

    assert result is None  # No opinion


@pytest.mark.unit
def test_mcp_tool_passes_through(tmp_path, hook_runner):
    """MCP tools are not affected by this hook."""
    result = hook_runner(
        "osprey_memory_guard.py",
        "mcp__workspace__some_tool",
        {"content": "some memory"},
        cwd=tmp_path,
    )

    assert result is None  # No opinion


# -- Deny message quality --


@pytest.mark.unit
def test_deny_message_includes_allowed_directory(tmp_path, hook_runner):
    """Deny message tells the agent where it CAN write."""
    target = str(tmp_path / "nope.txt")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "test"},
        cwd=tmp_path,
    )

    assert result is not None
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert "WRITE DENIED" in reason
    assert "memory" in reason.lower()


# -- CLAUDE_PROJECT_DIR env var --


@pytest.mark.unit
def test_uses_claude_project_dir_env(tmp_path, hook_runner, monkeypatch):
    """Hook respects CLAUDE_PROJECT_DIR env var over CWD."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    memory_dir = _memory_dir_for(str(project_dir))
    target = str(memory_dir / "notes.md")

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# Notes"},
        cwd=tmp_path,  # CWD is different from project dir
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
