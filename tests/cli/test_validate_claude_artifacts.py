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

import pytest

from osprey.cli.validate_claude_artifacts import (
    _normalised_text,
    _parse_frontmatter,
    _unreadable_frontmatter,
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


def test_tool_under_a_denied_namespace_fails(tmp_path):
    """deny wins at runtime for a namespace glob as for a literal entry."""
    project = _write_project(
        tmp_path,
        allow=["mcp__plugin_x__y"],
        ask=[],
        deny=["mcp__plugin_*"],
        agent_tools="mcp__plugin_x__y, Read",
    )
    errors = validate_agent_tools_against_permissions(project)
    assert len(errors) == 1, errors
    assert "mcp__plugin_x__y" in errors[0]


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


# --- the frontmatter block: it closes on a line that is exactly --- ---------------


@pytest.mark.parametrize(
    "text",
    [
        "---\nname: a\n---\nbody\n",
        "---\nname: a\n---",
        "---\nname: a\n---\nbody\n\n---\n\nmore\n",
    ],
    ids=["plain", "close-at-end-of-file", "horizontal-rule-in-body"],
)
def test_the_block_closes_on_an_exact_line(text):
    assert _parse_frontmatter(text) == {"name": "a"}
    assert not _unreadable_frontmatter(text)


@pytest.mark.parametrize(
    "line", ["---x: 1", "--- x: 1", "----"], ids=["dashes-then-text", "dashes-space-text", "four"]
)
def test_a_line_starting_with_dashes_does_not_close_the_block(line):
    text = f"---\nname: a\n{line}\nhooks: 1\n---\nbody\n"
    assert _parse_frontmatter(text) != {"name": "a"}
    assert _unreadable_frontmatter(text)


def test_a_block_no_exact_line_closes_is_unreadable():
    text = "---\nname: a\nhooks: 1\n"
    assert _parse_frontmatter(text) is None
    assert _unreadable_frontmatter(text)


@pytest.mark.parametrize("text", ["# title\n\nbody\n", "body\n---\nname: a\n---\n", ""])
def test_text_that_opens_no_block_has_none(text):
    assert _parse_frontmatter(text) is None
    assert not _unreadable_frontmatter(text)


@pytest.mark.parametrize(
    "first",
    ["----", "---x: 1", "\ufeff---", "---\u00a0", "---\f", "---\v", "---\r", " ---"],
    ids=[
        "four",
        "dashes-then-text",
        "byte-order-mark",
        "no-break-space",
        "form-feed",
        "vertical-tab",
        "carriage-return",
        "leading-space",
    ],
)
def test_a_first_line_a_looser_reader_opens_on_is_unreadable(first):
    text = f"{first}\nname: a\nhooks: 1\n---\nbody\n"
    assert _parse_frontmatter(text) is None
    assert _unreadable_frontmatter(text)


@pytest.mark.parametrize(
    "line", ["---\u00a0", " ---", "---\f", "\u00a0---"], ids=["nbsp", "space", "ff", "nbsp-first"]
)
def test_a_closing_candidate_inside_the_block_is_unreadable(line):
    text = f"---\nname: a\n{line}\nhooks: 1\n---\nbody\n"
    assert _unreadable_frontmatter(text)


@pytest.mark.parametrize(
    "text",
    ["--- \nname: a\n---\nbody\n", "---\nname: a\n---\t\nbody\n---\nmore\n"],
    ids=["opening-trailing-space", "closing-trailing-tab"],
)
def test_a_delimiter_with_trailing_blanks_is_unreadable(text):
    assert _unreadable_frontmatter(text)


def _read_back(tmp_path: Path, raw: bytes) -> str:
    path = tmp_path / "SKILL.md"
    path.write_bytes(raw)
    return _normalised_text(path)


@pytest.mark.parametrize(
    "raw",
    [
        b"---\rhooks: {}\n---\n",
        b"---\nname: x\rhooks: {}\n---\nbody\n",
        b"---\nname: x\n---\rhooks: {}\n---\nbody\n",
    ],
    ids=["opening-line", "inside-the-block", "closing-line"],
)
def test_a_lone_carriage_return_in_the_frontmatter_is_unreadable(tmp_path, raw):
    """The check sees the characters the copy carries: a lone CR is not a line end to it."""
    assert _unreadable_frontmatter(_read_back(tmp_path, raw))


def test_crlf_line_endings_with_exact_delimiters_read(tmp_path):
    text = _read_back(tmp_path, b"---\r\nname: a\r\nmodel: m\r\n---\r\nbody\r\n")
    assert not _unreadable_frontmatter(text)
    assert _parse_frontmatter(text) == {"name": "a", "model": "m"}


@pytest.mark.parametrize(
    "body", [b"one\rtwo\n", b"--- \nmore\n"], ids=["lone-carriage-return", "trailing-blanks"]
)
def test_the_body_after_a_closed_block_stays_readable(tmp_path, body):
    text = _read_back(tmp_path, b"---\nname: a\n---\n" + body)
    assert not _unreadable_frontmatter(text)
    assert _parse_frontmatter(text) == {"name": "a"}


def test_an_agent_file_with_a_byte_order_mark_is_audited(tmp_path):
    """The audit reads the frontmatter the loader reads, byte-order mark or not."""
    project = _write_project(
        tmp_path, allow=[], ask=[], agent_tools="mcp__nonexistent__phantom, Read"
    )
    agent = project / ".claude" / "agents" / "specialist.md"
    agent.write_bytes(b"\xef\xbb\xbf" + agent.read_bytes().replace(b"\n", b"\r\n"))

    errors = validate_agent_tools_against_permissions(project)

    assert any("mcp__nonexistent__phantom" in e for e in errors), errors
    assert agent_file_models(agent.parent) == {"specialist": None}


def test_an_agent_file_with_unreadable_frontmatter_is_reported(tmp_path):
    project = _write_project(tmp_path, allow=[], ask=[], agent_tools="Read")
    agent = project / ".claude" / "agents" / "specialist.md"
    agent.write_text(
        "---\u00a0\nname: specialist\ntools: mcp__nonexistent__phantom\n---\n", encoding="utf-8"
    )

    errors = validate_agent_tools_against_permissions(project)

    assert len(errors) == 1, errors
    assert "specialist.md" in errors[0] and "does not parse as a YAML mapping" in errors[0]


@pytest.mark.parametrize(
    "lead",
    ["\n", "\n\n", " \n", " \n", "﻿\n", "\t\n  "],
    ids=["blank-line", "two-blank-lines", "spaces", "no-break-space", "byte-order-mark", "mixed"],
)
def test_a_block_a_stripping_reader_opens_is_unreadable(lead):
    """A reader that strips the file first opens a block after leading blank lines."""
    text = f"{lead}---\nname: a\nhooks: 1\n---\nbody\n"
    assert _parse_frontmatter(text) is None
    assert _unreadable_frontmatter(text)


@pytest.mark.parametrize(
    "char",
    ["\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", " ", " "],
    ids=["vt", "ff", "fs", "gs", "rs", "nel", "line-separator", "paragraph-separator"],
)
def test_a_line_break_other_than_newline_in_the_block_is_unreadable(char):
    """A reader that splits on every line break sees other lines than this one does."""
    text = f"---\nname: a{char}hooks: 1\n---\nbody\n"
    assert _unreadable_frontmatter(text)


def test_a_line_break_other_than_newline_in_the_body_is_readable():
    text = "---\nname: a\n---\nbody more\fend\n"
    assert _parse_frontmatter(text) == {"name": "a"}
    assert not _unreadable_frontmatter(text)


def test_a_markdown_rule_after_text_stays_readable():
    text = "# title\n\nbody\n\n---\n\nmore\n"
    assert not _unreadable_frontmatter(text)
