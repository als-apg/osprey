"""The two safety boundaries of a deployment root, pinned as literal outcomes.

What executed python code may not write, and which root entries never enter a
container image, are both computed from the deployment-root layout. These tests
state each boundary's result on one example deployment as a literal, so a
change to either boundary, whatever table it comes from, is a visible edit
here rather than a silent consequence of a layout change elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_cmd import _stage_source_zone
from osprey.mcp_server.python_executor.executor import resolve_protected_roots

#: Every entry the example deployment holds at its root: the whole known
#: layout, the retired layer directory, the secrets in several spellings,
#: git's directory, the merged compose document a deploy writes, and two
#: entries the layout does not name.
#: A name ending in ``/`` is a directory.
EXAMPLE_ROOT_ENTRIES: tuple[str, ...] = (
    "profile.yml",
    "providers.yml",
    "triggers.yml",
    "ci-extra.yml",
    "osprey.service",
    ".dockerignore",
    "data/",
    "tiers/",
    "personas/",  # retired-spelling
    "apps/",
    "profiles/",
    "scripts/",
    "watchdogs/",
    ".osprey/",
    "rules/",
    "skills/",
    "agents/",
    "commands/",
    "output-styles/",
    "hooks/",
    "web-terminal-context/",
    "mcp_servers/",
    "services/",
    "project/",
    "build/",
    "var/",
    ".env",
    ".env.example",
    ".env.users",
    ".env.local",
    ".env.shared",
    ".gitignore",
    ".gitlab-ci.yml",
    "README.md",
    "CLAUDE.md",
    "CLAUDE.local.md",
    ".claude/",
    ".mcp.json",
    ".git/",
    ".osprey-compose.yml",
    "docs/",
    "notes.txt",
)


@pytest.fixture
def deployment(tmp_path: Path) -> Path:
    root = (tmp_path / "deployment").resolve()
    root.mkdir()
    for entry in EXAMPLE_ROOT_ENTRIES:
        if entry.endswith("/"):
            directory = root / entry.rstrip("/")
            directory.mkdir()
            (directory / "keep.txt").write_text("x\n", encoding="utf-8")
        else:
            (root / entry).write_text("# example\n", encoding="utf-8")
    return root


def test_executed_code_may_not_write_exactly_these_paths(deployment: Path) -> None:
    protected = resolve_protected_roots(deployment)
    relative = [path.relative_to(deployment).as_posix() for path in protected]

    assert len(relative) == len(set(relative)), f"duplicate protected roots: {relative}"
    assert set(relative) == {
        # The render zone.
        "build",
        # The pre-upgrade audit tree and the agent-record roots.
        "var/audit",
        "var/spool",
        "var/agent_record",
        "var/state",
        "var/archive",
        # Source files and directories.
        "profile.yml",
        "providers.yml",
        "triggers.yml",
        "ci-extra.yml",
        "osprey.service",
        ".dockerignore",
        "data",
        "tiers",
        "personas",  # retired-spelling
        "apps",
        "profiles",
        "scripts",
        "watchdogs",
        ".osprey",
        # Convention sources.
        "rules",
        "skills",
        "agents",
        "commands",
        "output-styles",
        "hooks",
        "web-terminal-context",
        "mcp_servers",
        "services",
        "project",
    }
    # Every example entry no protected path covers stays writable.
    covered = {path for path in relative if "/" not in path}
    assert {entry.rstrip("/") for entry in EXAMPLE_ROOT_ENTRIES} - covered == {
        # The agent-record tree, protected only below its root.
        "var",
        # Every `.env` spelling.
        ".env",
        ".env.example",
        ".env.users",
        ".env.local",
        ".env.shared",
        # The repo's own Claude Code files.
        "CLAUDE.md",
        "CLAUDE.local.md",
        ".claude",
        ".mcp.json",
        # Repository files the layout names.
        ".gitignore",
        ".gitlab-ci.yml",
        "README.md",
        # Git's directory and the merged compose document.
        ".git",
        ".osprey-compose.yml",
        # Entries the layout does not name.
        "docs",
        "notes.txt",
    }


def test_executed_code_write_boundary_includes_entries_not_yet_on_disk(tmp_path: Path) -> None:
    empty = (tmp_path / "empty").resolve()
    empty.mkdir()

    relative = {path.relative_to(empty).as_posix() for path in resolve_protected_roots(empty)}

    assert {"tiers", "personas", "var/audit", ".osprey"} <= relative  # retired-spelling


def test_image_staging_leaves_out_exactly_these_root_entries(
    deployment: Path, tmp_path: Path
) -> None:
    image_root = tmp_path / "image"
    image_root.mkdir()

    _stage_source_zone(deployment, image_root, build_inputs=())

    staged = {entry.name for entry in image_root.iterdir()}
    left_out = {entry.rstrip("/") for entry in EXAMPLE_ROOT_ENTRIES} - staged
    assert staged <= {entry.rstrip("/") for entry in EXAMPLE_ROOT_ENTRIES}
    assert left_out == {
        # Every `.env` spelling.
        ".env",
        ".env.example",
        ".env.users",
        ".env.local",
        ".env.shared",
        # The repo's own Claude Code files.
        "CLAUDE.md",
        "CLAUDE.local.md",
        ".claude",
        ".mcp.json",
        # The repo's own image-exclusion list, consumed rather than copied.
        ".dockerignore",
        # The fixed generated names and git's directory.
        "build",
        "var",
        ".git",
        ".osprey-compose.yml",
    }
    assert staged == {
        # Source files and directories.
        "profile.yml",
        "providers.yml",
        "triggers.yml",
        "ci-extra.yml",
        "osprey.service",
        "data",
        "tiers",
        "personas",  # retired-spelling
        "apps",
        "profiles",
        "scripts",
        "watchdogs",
        ".osprey",
        # Convention sources.
        "rules",
        "skills",
        "agents",
        "commands",
        "output-styles",
        "hooks",
        "web-terminal-context",
        "mcp_servers",
        "services",
        "project",
        # Repository files the layout names.
        ".gitignore",
        ".gitlab-ci.yml",
        "README.md",
        # Entries the layout does not name.
        "docs",
        "notes.txt",
    }
