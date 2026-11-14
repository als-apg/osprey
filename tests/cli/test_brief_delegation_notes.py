"""An agent's delegation note renders under its entry in the operator brief.

The generic agent loop of the brief renders an agent's ``delegation`` note in
place of its routing ``description`` when the agent carries one. A note moved
out of a hand-written block therefore renders byte for byte as the block did.
"""

from __future__ import annotations

import pytest

from osprey.cli.templates.manager import TemplateManager
from osprey.registry.mcp import resolve_agents, resolve_servers
from osprey_connectors.workspace import DEFAULT_AGENT_DATA_BASE_DIR

CLAUDE_MD = "claude_code/CLAUDE.md.j2"
CLAUDE_CF = "claude_code/CLAUDE.channel-finder.md.j2"
BRIEFS = [CLAUDE_MD, CLAUDE_CF]

# The facility-knowledge paragraph of CLAUDE.md.j2, as an author writes it in
# an agent file: first line the summary, each further line a sentence.
KB_NOTE = (
    "Looks up facility-specific documented knowledge: subsystem descriptions, device "
    "specifications, operational procedures, safety rules, physics references, and "
    "terminology.\n"
    "For ANY question about documented facility content — procedures, subsystem "
    "descriptions, device specs, safety rules, or facility-specific terminology — "
    "delegate to **kb-fixture**. Do NOT call `list_concepts`, `read_concept`, or "
    "`search` yourself; use **kb-fixture** to retrieve and synthesize bundle content."
)


def _ctx(enabled: dict[str, bool], extra: list[dict] | None = None) -> dict:
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
        "phoebus_agent_access": "read",
        "facility_name": "Test Facility",
        "facility_permissions": {},
    }
    ctx["servers"] = resolve_servers({}, ctx)
    agents = resolve_agents({}, ctx, resolved_servers=ctx["servers"])
    for a in agents:
        if a["name"] in enabled:
            a["enabled"] = enabled[a["name"]]
    agents.extend(extra or [])
    ctx["agents"] = agents
    ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
    ctx["enabled_agents"] = {a["name"] for a in agents if a["enabled"]}
    return ctx


def _agent(name: str, *, enabled: bool = True, delegation: str | None = None) -> dict:
    return {
        "name": name,
        "enabled": enabled,
        "description": f"Routing text for {name}.",
        "is_custom": True,
        "requires_ask_tools": [],
        "delegation": delegation,
    }


def _render(path: str, ctx: dict) -> str:
    return TemplateManager().jinja_env.get_template(path).render(**ctx)


def _bullet_block(text: str, name: str) -> str:
    """The bullet for ``name``, up to the next bullet or blank line."""
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith(f"- **{name}**"))
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("- ") or not line.strip():
            break
        block.append(line)
    return "".join(block)


def test_note_renders_byte_identical_to_the_hand_written_block():
    a = _render(CLAUDE_MD, _ctx({"facility-knowledge": True, "pyat-specialist": False}))
    b = _render(
        CLAUDE_MD,
        _ctx(
            {"facility-knowledge": False, "pyat-specialist": False},
            [_agent("kb-fixture", delegation=KB_NOTE)],
        ),
    )
    expected = _bullet_block(a, "facility-knowledge").replace("facility-knowledge", "kb-fixture")
    assert _bullet_block(b, "kb-fixture") == expected


@pytest.mark.parametrize("path", BRIEFS)
def test_note_renders_under_the_bullet_with_continuation_indent(path):
    text = _render(path, _ctx({}, [_agent("kb-fixture", delegation=KB_NOTE)]))
    first, second = KB_NOTE.split("\n")
    assert _bullet_block(text, "kb-fixture") == f"- **kb-fixture** — {first}\n  {second}\n"


@pytest.mark.parametrize("path", BRIEFS)
def test_agent_without_note_renders_its_description(path):
    text = _render(path, _ctx({}, [_agent("plain-fixture")]))
    assert _bullet_block(text, "plain-fixture") == (
        "- **plain-fixture** — Routing text for plain-fixture.\n"
    )


@pytest.mark.parametrize("path", BRIEFS)
def test_disabled_agent_with_note_renders_nothing(path):
    text = _render(path, _ctx({}, [_agent("kb-fixture", enabled=False, delegation=KB_NOTE)]))
    assert "kb-fixture" not in text
