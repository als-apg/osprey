"""Unit tests for ``validate_agent_tools_against_permissions`` and the agent model pin check.

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
    agent_file_models,
    agent_model_pin_errors,
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
        allow=["mcp__osprey_workspace__artifact_read"],
        ask=[],
        agent_tools="mcp__osprey_workspace__artifact_read, Read",
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
        allow=["mcp__osprey_workspace__artifact_read"],
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
        allow=["mcp__osprey_workspace__artifact_read"],
        ask=["mcp__python__execute"],
        agent_tools="mcp__python__execute, mcp__osprey_workspace__artifact_read, Read",
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
        allow=["mcp__osprey_workspace__artifact_read"],
        ask=["mcp__python__execute"],
        agent_tools="mcp__osprey_workspace__*, Read",
    )
    errors = validate_agent_tools_against_permissions(project)
    assert any("wildcard" in e.lower() for e in errors), (
        f"expected wildcard-rejection error; got: {errors}"
    )


# --- agent model pins ---


def _write_agent(project: Path, name: str, model: str | None) -> Path:
    """Write ``.claude/agents/<name>.md``, with a ``model:`` line when *model* is given."""
    agents = project / ".claude" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    model_line = f"model: {model}\n" if model is not None else ""
    (agents / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Test agent.\n{model_line}---\n\n# {name}\n",
        encoding="utf-8",
    )
    return agents


def test_agent_file_models_reads_each_model_line(tmp_path):
    _write_agent(tmp_path, "logbook-search", "claude-sonnet-5")
    agents = _write_agent(tmp_path, "site-helper", None)
    (agents / "notes.md").write_text("no frontmatter here\n", encoding="utf-8")

    assert agent_file_models(agents) == {
        "logbook-search": "claude-sonnet-5",
        "notes": None,
        "site-helper": None,
    }
    assert agent_file_models(tmp_path / "missing") == {}


def test_a_pin_the_render_applies_passes(tmp_path):
    agents = _write_agent(tmp_path, "logbook-search", "gpt-6-luna")

    assert (
        agent_model_pin_errors(agents, {"logbook-search": "gpt-6-luna"}, ["logbook-search"]) == []
    )


def test_a_pin_naming_no_agent_is_refused_with_the_agent_names(tmp_path):
    agents = _write_agent(tmp_path, "logbook-search", "claude-sonnet-5")

    errors = agent_model_pin_errors(
        agents, {"logbok-search": "claude-sonnet-5"}, ["logbook-deep-research"]
    )

    assert len(errors) == 1
    assert (
        "claude_code.agent_models.logbok-search: no agent is called 'logbok-search'" in (errors[0])
    )
    assert "Agents: logbook-deep-research, logbook-search." in errors[0]


def test_a_pin_on_an_agent_this_render_does_not_ship_is_not_an_error(tmp_path):
    agents = _write_agent(tmp_path, "logbook-search", "claude-sonnet-5")

    assert (
        agent_model_pin_errors(
            agents, {"channel-finder": "claude-haiku-4-5"}, ["channel-finder", "logbook-search"]
        )
        == []
    )


def test_a_pin_the_agent_file_overrides_is_refused(tmp_path):
    agents = _write_agent(tmp_path, "logbook-search", "claude-sonnet-5")

    errors = agent_model_pin_errors(agents, {"logbook-search": "gpt-6-luna"}, [])

    assert len(errors) == 1
    assert "gpt-6-luna is not what agents/logbook-search.md runs (claude-sonnet-5)" in errors[0]


def test_a_pin_on_an_agent_file_with_no_model_line_is_refused(tmp_path):
    agents = _write_agent(tmp_path, "site-helper", None)

    errors = agent_model_pin_errors(agents, {"site-helper": "claude-haiku-4-5"}, [])

    assert len(errors) == 1
    assert "(it has no model: line)" in errors[0]
