"""The SQL tool in the guidance every ARIEL agent surface renders.

``ariel_sql_tool`` is false where the build profile authors
``ariel.database.uri`` without ``ariel.database.readonly_uri``: the ARIEL
server then does not serve ``sql_query``, so no rendered body names it or
tells the agent to reach for SQL, and the subagents' tool lines and the
main agent's do-not-call list leave it out.
"""

from __future__ import annotations

import re

import pytest

from osprey.cli.templates.claude_code import (
    _ariel_read_tools,
    _ariel_sql_tool,
    config_derived_context,
)
from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import resolve_agents, resolve_servers
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR

DEEP_RESEARCH = "claude_code/claude/agents/logbook-deep-research.md.j2"
SKILL = "claude_code/claude/skills/logbook-deep-research/SKILL.md.j2"
CLAUDE_ARIEL = "claude_code/CLAUDE.logbook.md.j2"
CLAUDE_MD = "claude_code/CLAUDE.md.j2"
BODIES = [DEEP_RESEARCH, SKILL, CLAUDE_ARIEL]

AUTHORED = "postgresql://ariel@logbook-db.example.org:5432/ariel"
AUTHORED_RO = "postgresql://ariel_ro@logbook-db.example.org:5432/ariel"


def _ctx(sql_tool: bool) -> dict:
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
        "phoebus_agent_access": "read",
        "facility_name": "Test Facility",
        "facility_permissions": {},
        "ariel_attachment_view": True,
        "ariel_sql_tool": sql_tool,
        "ariel_read_tools": _ariel_read_tools({"ariel_sql_tool": sql_tool}),
    }
    ctx["servers"] = resolve_servers({}, ctx)
    ctx["agents"] = resolve_agents({}, ctx, resolved_servers=ctx["servers"])
    ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
    ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
    return ctx


def _render(path: str, sql_tool: bool) -> str:
    return TemplateManager().jinja_env.get_template(path).render(**_ctx(sql_tool))


@pytest.mark.parametrize("path", BODIES)
def test_withheld_renders_name_no_sql_tool(path):
    text = _render(path, False)
    assert "sql_query" not in text
    assert "SQL" not in text


@pytest.mark.parametrize("path", BODIES)
def test_offered_renders_name_the_sql_tool(path):
    assert "sql_query" in _render(path, True)


@pytest.mark.parametrize("sql_tool", [True, False])
def test_the_deep_research_tools_line_follows_the_flag(sql_tool):
    match = re.search(r"^tools: (.*)$", _render(DEEP_RESEARCH, sql_tool), re.MULTILINE)
    assert match
    tools = [t.strip() for t in match.group(1).split(",")]
    assert ("mcp__ariel__sql_query" in tools) is sql_tool
    assert "mcp__ariel__entries_by_ids" in tools


@pytest.mark.parametrize("sql_tool", [True, False])
def test_the_main_agent_do_not_call_list_follows_the_flag(sql_tool):
    assert ("sql_query" in _ariel_read_tools({"ariel_sql_tool": sql_tool})) is sql_tool
    assert ("sql_query" in _render(CLAUDE_MD, sql_tool)) is sql_tool


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, True),
        ({"ariel": {}}, True),
        ({"ariel": {"database": {"uri": AUTHORED}}}, False),
        ({"ariel": {"database": {"uri": AUTHORED, "readonly_uri": AUTHORED_RO}}}, True),
    ],
    ids=["no-ariel", "derived", "authored-uri", "authored-readonly"],
)
def test_ariel_sql_tool_reads_the_profile(config, expected):
    assert _ariel_sql_tool(config) is expected


def test_config_derived_context_withholds_the_tool_for_an_authored_store(tmp_path):
    ctx = config_derived_context({"ariel": {"database": {"uri": AUTHORED}}}, tmp_path)
    assert ctx["ariel_sql_tool"] is False
    assert "sql_query" not in ctx["ariel_read_tools"]
    assert config_derived_context({}, tmp_path)["ariel_sql_tool"] is True
