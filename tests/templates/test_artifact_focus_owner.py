"""Focusing an artifact belongs to the session that answers the user.

``artifact_focus`` selects an artifact in the gallery. A subagent files its
answer and hands back the id; the session that answers the user focuses it.
Two claims, on the control-assistant render:

1. No agent file names ``artifact_focus`` unless its ``tools:`` list holds it.
2. The artifacts rule, which every session reads, addresses the call to the
   main session, never to whoever created the artifact.
"""

from __future__ import annotations

from pathlib import Path

from tests._builds import BuiltProject

from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents

FOCUS_TOOL = "artifact_focus"
AGENTS = Path(".claude") / "agents"
RULE = Path(".claude") / "rules" / "artifacts.md"


def test_no_agent_names_artifact_focus_without_holding_it(
    built_control_assistant: BuiltProject,
) -> None:
    render = built_control_assistant.build_dir
    surfaces = parse_project_agents(render)
    assert surfaces, f"the render holds no agent under {AGENTS}"
    offenders = []
    for path in sorted((render / AGENTS).glob("*.md")):
        if FOCUS_TOOL not in path.read_text(encoding="utf-8"):
            continue
        tools = surfaces.get(path.stem) or frozenset()
        if not any(tool.endswith(f"__{FOCUS_TOOL}") for tool in tools):
            offenders.append(path.name)
    assert offenders == [], f"{offenders} name {FOCUS_TOOL} but do not hold the tool"


def test_the_artifacts_rule_leaves_focus_to_the_main_session(
    built_control_assistant: BuiltProject,
) -> None:
    rule = (built_control_assistant.build_dir / RULE).read_text(encoding="utf-8")
    assert f"After creating an artifact, call `{FOCUS_TOOL}" not in rule
    section = rule.split("## Directing User Attention", 1)[1].split("\n## ", 1)[0]
    assert "main session" in section, section
    assert "subagent" in section.lower(), section
