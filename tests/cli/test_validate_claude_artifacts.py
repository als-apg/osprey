"""Unit tests for ``validate_agent_tools_against_permissions``.

These exercise the validator in isolation against hand-built ``.claude/``
trees, independent of the full template-render path. They lock down the
"backed" rule: an agent's ``mcp__`` tool must appear in ``permissions.allow``
*or* ``permissions.ask`` (approval-gated tools are backed, just prompted),
while a tool in neither list is real drift and must be reported.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from osprey.cli.validate_claude_artifacts import (
    validate_agent_tools_against_permissions,
)


def _write_project(
    tmp_path: Path,
    *,
    allow: list[str],
    ask: list[str],
    agent_tools: str,
    deny: list[str] | None = None,
    agent_name: str = "specialist",
) -> Path:
    """Build a minimal rendered project: one settings.json + one agent."""
    claude = tmp_path / ".claude"
    (claude / "agents").mkdir(parents=True)
    permissions: dict[str, list[str]] = {"allow": allow, "ask": ask}
    if deny is not None:
        permissions["deny"] = deny
    (claude / "settings.json").write_text(
        json.dumps({"permissions": permissions}),
        encoding="utf-8",
    )
    (claude / "agents" / f"{agent_name}.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {agent_name}
            description: Test agent.
            tools: {agent_tools}
            ---

            # {agent_name}
            """
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_tool_in_allow_passes(tmp_path):
    """A tool present in permissions.allow is backed."""
    project = _write_project(
        tmp_path,
        allow=["mcp__osprey_workspace__data_read"],
        ask=[],
        agent_tools="mcp__osprey_workspace__data_read, Read",
    )
    assert validate_agent_tools_against_permissions(project) == []


def test_ask_gated_tool_passes(tmp_path):
    """A tool present only in permissions.ask is backed (approval-gated).

    Mirrors pyat-specialist declaring ``mcp__python__execute``, which the
    python server renders into ``permissions.ask`` (permissions_allow=[],
    permissions_ask=["execute"]) — available to the agent, just prompted.
    """
    project = _write_project(
        tmp_path,
        allow=[],
        ask=["mcp__python__execute"],
        agent_tools="mcp__python__execute, Read",
    )
    assert validate_agent_tools_against_permissions(project) == []


def test_tool_in_neither_list_fails(tmp_path):
    """A tool in neither allow nor ask is unbacked and must be reported."""
    project = _write_project(
        tmp_path,
        allow=["mcp__osprey_workspace__data_read"],
        ask=["mcp__python__execute"],
        agent_tools="mcp__nonexistent__phantom, Read",
    )
    errors = validate_agent_tools_against_permissions(project)
    assert any("specialist" in e and "mcp__nonexistent__phantom" in e for e in errors), (
        f"expected error naming the unbacked tool; got: {errors}"
    )


def test_ask_gated_and_allow_mix_passes(tmp_path):
    """An agent may draw from both lists at once."""
    project = _write_project(
        tmp_path,
        allow=["mcp__osprey_workspace__data_read"],
        ask=["mcp__python__execute"],
        agent_tools="mcp__python__execute, mcp__osprey_workspace__data_read, Read",
    )
    assert validate_agent_tools_against_permissions(project) == []


def test_tool_in_ask_and_deny_fails(tmp_path):
    """deny wins at runtime — a tool in ask AND deny is not actually backed."""
    project = _write_project(
        tmp_path,
        allow=[],
        ask=["mcp__python__execute"],
        deny=["mcp__python__execute"],
        agent_tools="mcp__python__execute, Read",
    )
    errors = validate_agent_tools_against_permissions(project)
    assert any("specialist" in e and "mcp__python__execute" in e for e in errors), (
        f"expected denied tool to fail validation; got: {errors}"
    )


def test_wildcard_still_rejected(tmp_path):
    """Wildcards are rejected regardless of the ask/allow membership rule."""
    project = _write_project(
        tmp_path,
        allow=["mcp__osprey_workspace__data_read"],
        ask=["mcp__python__execute"],
        agent_tools="mcp__osprey_workspace__*, Read",
    )
    errors = validate_agent_tools_against_permissions(project)
    assert any("wildcard" in e.lower() for e in errors), (
        f"expected wildcard-rejection error; got: {errors}"
    )
