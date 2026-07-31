"""Tests for the osprey_memory_guard hook.

This hook restricts the Write tool to Claude Code native memory files only.
Writes targeting ``~/.claude/projects/<encoded>/memory/*.md`` are allowed;
everything else is denied. Non-Write tools pass through without opinion.
"""

import pytest

from osprey.agent_runner.project_paths import encode_claude_project_path


@pytest.fixture
def memory_guard(hook_module):
    """The hook module, for tests that call its helpers directly."""
    return hook_module("osprey_memory_guard")


@pytest.fixture
def memory_dir_for(memory_guard, hook_home):
    """Resolve a project's memory directory the way the hook does.

    Delegates to the hook's own ``resolve_memory_dir`` so the expected paths
    cannot drift from the implementation they are meant to describe. The result
    is pinned under ``hook_home`` — the curated ``$HOME`` the hook subprocess
    runs under — so a break in the ``$HOME`` isolation surfaces here instead of
    as a write into the developer's real home directory.
    """

    def _resolve(project_dir):
        memory_dir = memory_guard.resolve_memory_dir(str(project_dir))
        assert hook_home in memory_dir.parents
        return memory_dir

    return _resolve


# -- Path encoding --


@pytest.mark.unit
def test_memory_dir_is_home_relative(tmp_path, memory_guard, hook_home):
    """The memory dir is ``$HOME/.claude/projects/<encoded>/memory``."""
    encoded = encode_claude_project_path(tmp_path)

    assert memory_guard.resolve_memory_dir(str(tmp_path)) == (
        hook_home / ".claude" / "projects" / encoded / "memory"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "project_name",
    ["plain", "with spaces", "dotted.name.v2", "under_score+plus@sign"],
    ids=["plain", "spaces", "dots", "non-alnum"],
)
def test_encoding_matches_packaged_helper(tmp_path, memory_guard, project_name):
    """The hook's inlined encoding agrees with ``encode_claude_project_path``.

    The hook duplicates the regex on purpose: it runs inside user projects where
    the ``osprey`` package may not be importable at hook-execution time. Nothing
    but this test keeps the copies together, and drift is silent — Claude Code
    would store memories under one directory name while the guard allows writes
    to another.
    """
    project_dir = tmp_path / project_name
    project_dir.mkdir()

    memory_dir = memory_guard.resolve_memory_dir(str(project_dir))

    assert memory_dir.parent.name == encode_claude_project_path(project_dir)


# -- Write predicate --


@pytest.mark.unit
def test_predicate_allows_md_direct_child(tmp_path, memory_guard, memory_dir_for):
    """A ``.md`` file directly inside the memory dir is allowed."""
    memory_dir = memory_dir_for(tmp_path)

    assert memory_guard.is_allowed_memory_write(str(memory_dir / "channels.md"), memory_dir)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("build_path", "rejected_for"),
    [
        (lambda d: d / "script.py", "not a .md file"),
        (lambda d: d.parent / "notes.md", "outside the memory directory"),
        (lambda d: d / "subdir" / "notes.md", "nested rather than a direct child"),
        (lambda d: d / "subdir" / ".." / "notes.md", "traversal in the raw input"),
    ],
    ids=["wrong-suffix", "outside-dir", "nested", "traversal"],
)
def test_predicate_rejects(tmp_path, memory_guard, memory_dir_for, build_path, rejected_for):
    """Each rejection branch is reachable on its own.

    The ``traversal`` case resolves back to a legitimate memory file and is
    turned down only by the literal ``..`` check on the caller's string.
    """
    memory_dir = memory_dir_for(tmp_path)

    assert not memory_guard.is_allowed_memory_write(str(build_path(memory_dir)), memory_dir), (
        f"should be rejected: {rejected_for}"
    )


# -- Allow cases --


@pytest.mark.unit
def test_allows_write_to_memory_md(tmp_path, hook_runner, memory_dir_for):
    """Write to a .md file inside the Claude memory directory is allowed."""
    target = str(memory_dir_for(tmp_path) / "channels.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# BPM Channels\n"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


@pytest.mark.unit
def test_allows_primary_memory_file(tmp_path, hook_runner, memory_dir_for):
    """MEMORY.md (the primary memory file) is allowed."""
    target = str(memory_dir_for(tmp_path) / "MEMORY.md")

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
def test_denies_non_md_in_memory_dir(tmp_path, hook_runner, memory_dir_for):
    """Write to the memory directory but with a non-.md extension is denied."""
    target = str(memory_dir_for(tmp_path) / "script.py")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "import evil"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_subdirectory_in_memory_dir(tmp_path, hook_runner, memory_dir_for):
    """Write to a subdirectory within the memory dir is denied (direct children only)."""
    target = str(memory_dir_for(tmp_path) / "subdir" / "notes.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "sneaky"},
        cwd=tmp_path,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.unit
def test_denies_path_traversal(tmp_path, hook_runner, memory_dir_for):
    """Write with path traversal components is denied."""
    target = str(memory_dir_for(tmp_path) / ".." / ".." / ".." / ".bashrc")

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
        "mcp__osprey_workspace__some_tool",
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
def test_uses_claude_project_dir_env(tmp_path, hook_runner, memory_dir_for, monkeypatch):
    """Hook respects CLAUDE_PROJECT_DIR env var over CWD."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    target = str(memory_dir_for(project_dir) / "notes.md")

    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project_dir))

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# Notes"},
        cwd=tmp_path,  # CWD is different from project dir
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


@pytest.mark.unit
@pytest.mark.parametrize(
    "stdin",
    ["", "{nope", "[]", "[1,2,3]"],
    ids=["empty", "invalid-json", "wrong-shape", "wrong-shape-truthy"],
)
def test_malformed_stdin_fails_open(tmp_path, hook_runner_raw, stdin):
    """Unusable stdin yields no opinion — not a deny.

    The guard denies writes it can see and does not recognise, but a closed
    pipe, a truncated write, or a non-object payload — falsy (``[]``) or
    truthy (``[1,2,3]``) — names no tool at all. With no ``tool_name`` to
    match, the hook exits 0 silently and Write is left to the rest of the
    permission chain. The truthy payload is the one that slips past an
    emptiness check, so it has to be rejected on shape.
    """
    returncode, stdout, stderr = hook_runner_raw(
        "osprey_memory_guard.py",
        tool_name=None,
        tool_input=None,
        cwd=tmp_path,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr
