"""Every artifact-filing subagent hands back a pointer, not a copy.

A subagent that calls ``submit_response`` has two outputs: the artifact, which
is durable and is what the gallery and downstream tools read, and the text it
returns to the parent session, which is transient context. When the return
text repeats the artifact body, the parent sees no reason to focus the
artifact and re-narrates the tables into chat instead — the user gets the
answer twice in the transcript and never in the gallery. The channel finder
always carried an explicit closing contract; the other agents ended on
``**Results** (artifact_id: 5): ...``, and a small model fills that ellipsis
with the whole artifact.

Two claims, rendered on ``control_assistant`` in the graph paradigm so every
agent template is exercised:

1. **Every agent whose frontmatter carries ``submit_response`` ships the shared
   hand-back section**, and none ships the open-ended ellipsis.
2. **The orchestrator's artifact rule tells it what to do with a hand-back**:
   focus the artifact and relay the headline, read it only when the next step
   needs the detail.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.cli.templates.manager import TemplateManager
from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents

pytestmark = pytest.mark.unit

_SUBMIT_RESPONSE = "mcp__osprey_workspace__submit_response"
_OPEN_ENDED_ELLIPSIS = re.compile(r"\(artifact_id: \d+\): \.\.\.")
#: The id-line label each agent hands back; the channel finder keeps the
#: ``**Channels found**`` list its report-format tests already pin.
_LABELS = {"channel-finder": "Channels found"}


@pytest.fixture(scope="module")
def rendered_project(tmp_path_factory) -> Path:
    return TemplateManager().create_project(
        project_name="handback-contract",
        output_dir=tmp_path_factory.mktemp("render"),
        data_bundle="control_assistant",
        context={"channel_finder_mode": "graph"},
    )


def _submitting_agents(project_dir: Path) -> dict[str, str]:
    """Rendered prompt text of every agent whose tool surface carries ``submit_response``.

    Agent names come from the frontmatter the dispatch worker parses — the same
    rule that decides what is delegable — and equal the file stems on this render.
    """
    return {
        name: (project_dir / ".claude" / "agents" / f"{name}.md").read_text(encoding="utf-8")
        for name, tools in parse_project_agents(project_dir).items()
        if tools and _SUBMIT_RESPONSE in tools
    }


def test_fixture_renders_the_whole_roster(rendered_project):
    """Precondition: the agents under test are actually on this render."""
    names = set(_submitting_agents(rendered_project))
    assert {
        "channel-finder",
        "facility-knowledge-graph",
        "facility-knowledge",
        "logbook-search",
        "logbook-deep-research",
        "pyat-specialist",
    } <= names, names


def test_every_submitting_agent_ships_the_handback_section(rendered_project):
    for name, text in _submitting_agents(rendered_project).items():
        assert "## Handing Back" in text, f"{name}: no hand-back section"
        assert f"**{_LABELS.get(name, 'Results')}** (artifact_id: <id>)" in text, (
            f"{name}: the id line is not spelled out with its label"
        )
        assert "artifact_read" in text, f"{name}: must say the parent reads the artifact itself"
        assert not _OPEN_ENDED_ELLIPSIS.search(text), (
            f"{name}: the open-ended '(artifact_id: N): ...' invites echoing the artifact body"
        )


def test_orchestrator_focuses_a_handed_back_artifact(rendered_project):
    rule = (rendered_project / ".claude" / "rules" / "artifacts.md").read_text(encoding="utf-8")
    assert "## Subagent Hand-Backs" in rule
    tail = rule.split("## Subagent Hand-Backs", 1)[1]
    assert "artifact_focus" in tail
    assert "artifact_read" in tail
