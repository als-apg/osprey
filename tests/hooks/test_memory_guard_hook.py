"""Tests for the osprey_memory_guard hook.

This hook restricts the Write tool to Claude Code native memory files only.
Writes targeting ``~/.claude/projects/<encoded>/memory/*.md`` are allowed;
everything else is denied. Non-Write tools pass through without opinion.
"""

from pathlib import Path

import pytest

from osprey.agent_runner.project_paths import encode_claude_project_path
from osprey.cli.templates.artifact_library import parse_hook_frontmatter


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


def test_memory_dir_is_home_relative(tmp_path, memory_guard, hook_home):
    """The memory dir is ``$HOME/.claude/projects/<encoded>/memory``."""
    encoded = encode_claude_project_path(tmp_path)

    assert memory_guard.resolve_memory_dir(str(tmp_path)) == (
        hook_home / ".claude" / "projects" / encoded / "memory"
    )


def test_memory_dir_honours_claude_config_dir(tmp_path, memory_guard, monkeypatch):
    """``CLAUDE_CONFIG_DIR`` is the state root; ``$HOME/.claude`` is only the fallback.

    The per-user web-terminal container sets ``CLAUDE_CONFIG_DIR`` and ``HOME``
    to one mounted volume. Claude Code then keeps memory under
    ``$CLAUDE_CONFIG_DIR/projects/<encoded>/memory`` — a hook that still spells
    ``$HOME/.claude`` names a directory nothing writes to, and denies every
    memory write in the container.
    """
    config_dir = tmp_path / "data" / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    encoded = encode_claude_project_path(tmp_path)

    assert memory_guard.resolve_memory_dir(str(tmp_path)) == (
        config_dir / "projects" / encoded / "memory"
    )


def test_allows_memory_write_under_claude_config_dir(tmp_path, hook_runner, monkeypatch):
    """End to end: a write into the ``$CLAUDE_CONFIG_DIR`` memory dir is allowed."""
    config_dir = tmp_path / "data" / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    project_dir = tmp_path / "build"
    project_dir.mkdir()
    memory_dir = config_dir / "projects" / encode_claude_project_path(project_dir) / "memory"
    memory_dir.mkdir(parents=True)

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": str(memory_dir / "notes.md"), "content": "# Notes"},
        cwd=project_dir,
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


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


def test_predicate_allows_md_direct_child(tmp_path, memory_guard, memory_dir_for):
    """A ``.md`` file directly inside the memory dir is allowed."""
    memory_dir = memory_dir_for(tmp_path)

    assert memory_guard.is_allowed_memory_write(str(memory_dir / "channels.md"), memory_dir)


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


def test_non_write_tool_passes_through(tmp_path, hook_runner):
    """Non-Write tools are not affected by this hook."""
    result = hook_runner(
        "osprey_memory_guard.py",
        "Read",
        {"file_path": "/etc/passwd"},
        cwd=tmp_path,
    )

    assert result is None  # No opinion


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


def test_guarded_tools_are_exactly_the_frontmatter_matcher(memory_guard):
    """The tools the hook judges are the tools its ``tools:`` matcher routes to it.

    The build copies the frontmatter matcher verbatim into ``PreToolUse``, so a
    tool listed there and missing from ``_GUARDED_TOOLS`` would reach the hook and
    be waved through, and a tool guarded here but absent there never arrives.
    """
    meta = parse_hook_frontmatter(Path(memory_guard.__file__))

    assert memory_guard._GUARDED_TOOLS == frozenset(meta["tools"].split("|"))


# -- Plan file --


@pytest.fixture
def plans_root(tmp_path, monkeypatch):
    """A throwaway config dir, pinned as ``$CLAUDE_CONFIG_DIR``: the plans directory's parent."""
    config_dir = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    return config_dir


def test_plans_dir_honours_claude_config_dir(plans_root, memory_guard):
    """The CLI keeps plans under ``$CLAUDE_CONFIG_DIR``, and the hook names the same place."""
    assert memory_guard.resolve_plans_dir() == plans_root / "plans"


def test_plans_dir_defaults_under_home(memory_guard, hook_home, monkeypatch):
    """With the variable unset the plans directory is ``~/.claude/plans``."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    assert memory_guard.resolve_plans_dir() == hook_home / ".claude" / "plans"


@pytest.mark.usefixtures("plans_root")
def test_plan_predicate_allows_md_direct_child(memory_guard):
    """A ``.md`` file directly inside the plans directory is a plan file."""
    plans_dir = memory_guard.resolve_plans_dir()

    assert memory_guard.is_allowed_plan_write(str(plans_dir / "fuzzy-moseying-quail.md"), plans_dir)


@pytest.mark.parametrize(
    ("build_path", "rejected_for"),
    [
        (lambda root: root / "plans" / "notes.txt", "not a .md file"),
        (lambda root: root / "plans" / "sub" / "notes.md", "nested rather than a direct child"),
        (lambda root: root / "plans-evil" / "notes.md", "a sibling sharing the plans prefix"),
        (lambda root: root / "notes.md", "the config dir above the plans directory"),
        (lambda root: root / "plans", "the plans directory itself"),
        (lambda root: root / "plans" / ".." / "notes.md", "traversal in the raw input"),
        (
            lambda root: root / "plans" / ".." / "plans" / "notes.md",
            "traversal that resolves back inside the plans directory",
        ),
        (
            lambda root: root / "projects" / "enc" / "memory" / "notes.md",
            "a memory file, which the plan allowance never reaches",
        ),
    ],
    ids=[
        "wrong-suffix",
        "nested",
        "sibling-prefix",
        "config-dir",
        "plans-dir-itself",
        "traversal",
        "traversal-back-in",
        "memory-dir",
    ],
)
def test_plan_predicate_rejects(plans_root, memory_guard, build_path, rejected_for):
    """Each refused neighbour is refused on its own, so none rests on another's check."""
    plans_dir = memory_guard.resolve_plans_dir()

    assert not memory_guard.is_allowed_plan_write(str(build_path(plans_root)), plans_dir), (
        f"should be rejected: {rejected_for}"
    )


def test_allows_plan_file_while_planning(plans_root, hook_runner, tmp_path):
    """In plan mode the plan file is writable: that is how a plan reaches ExitPlanMode."""
    target = str(plans_root / "plans" / "fuzzy-moseying-quail.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# Plan\n"},
        cwd=tmp_path,
        hook_input_extra={"permission_mode": "plan"},
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_plan_file_is_refused_outside_planning(plans_root, hook_runner, tmp_path):
    """The plan file is writable only while the session plans; any other mode refuses it.

    The hook allows it only when ``permission_mode`` is ``plan``, so every other mode,
    or none reported, takes one branch. ``acceptEdits`` stands for them: there the CLI
    itself would let the write through, so the hook's refusal is the only gate.
    """
    target = str(plans_root / "plans" / "fuzzy-moseying-quail.md")

    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": target, "content": "# Plan\n"},
        cwd=tmp_path,
        hook_input_extra={"permission_mode": "acceptEdits"},
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_planning_does_not_widen_the_write_allowance(plans_root, hook_runner, tmp_path):
    """In plan mode a neighbour of the plans directory stays refused, and the refusal
    tells the agent where a plan file goes.

    Each neighbour shape is refused by the predicate in ``test_plan_predicate_rejects``;
    this run shows the hook applies it, with a sibling that shares the plans prefix.
    """
    result = hook_runner(
        "osprey_memory_guard.py",
        "Write",
        {"file_path": str(plans_root / "plans-evil" / "notes.md"), "content": "x"},
        cwd=tmp_path,
        hook_input_extra={"permission_mode": "plan"},
    )

    assert result is not None
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    reason = result["hookSpecificOutput"]["permissionDecisionReason"]
    assert f"{plans_root / 'plans'}/<name>.md" in reason
