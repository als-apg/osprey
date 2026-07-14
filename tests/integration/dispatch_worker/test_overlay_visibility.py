"""Fast-provisioning visibility test for a fully-baked project image.

Guards the fix for a real provisioning gap: the dispatch worker used to run a
lean image and regenerate ``.claude`` from ``config.yml`` at startup, which
restored built-in artifacts only. Facility *overlay* skills/agents/rules are
recorded in ``config.yml`` by name, not by content, so they were silently
dropped on regeneration — and ``data/`` had no regen path at all. The fix
makes the worker run the full project image, which bakes overlays and
``data/`` in via ``COPY .`` at build time.

This is the fast, pure-filesystem complement to the real-container e2e: given
a project directory that emulates a fully-baked image (overlay agent/skill/
rule files and a ``data/`` file physically present on disk, as ``COPY .``
would leave them), it asserts the worker's own visibility and tool-policy
machinery treats the overlay artifacts as first-class:

* ``parse_project_agents`` (agent_surfaces.py) sees the overlay agent and its
  declared tool surface — the parser reads whatever ``.claude/agents/*.md``
  files physically exist, so a baked-in overlay agent is indistinguishable
  from a built-in one.
* ``make_pretooluse_hook`` (tool_policy.py) does NOT deny a subagent-dispatch
  call targeting the overlay agent on the "not declared" ground — the same
  ground it uses (and this test separately proves it uses) to reject a
  genuinely undeclared agent name.
* The overlay skill/rule files and ``data/channel_limits.json`` are present
  and readable, standing in for the ``data/`` regen gap the fix closed.

No container, network, or Claude Agent SDK involvement — pure filesystem
fixture plus in-process calls into the two dispatch_worker modules.
"""

import json
from pathlib import Path

from osprey.mcp_server.dispatch_worker.agent_surfaces import parse_project_agents
from osprey.mcp_server.dispatch_worker.tool_policy import make_pretooluse_hook

OVERLAY_AGENT_NAME = "facility-overlay"
OVERLAY_SKILL_NAME = "facility-overlay-skill"
OVERLAY_RULE_NAME = "facility-overlay-rule"

OVERLAY_AGENT_TOOLS = frozenset(
    {"mcp__osprey_workspace__facility_description", "mcp__controls__channel_read", "Read"}
)

# Mirrors the frontmatter shape the templates render (see e.g.
# claude/agents/data-visualizer.md.j2): name/description/summary/tools with
# tools: as the template's comma-separated scalar form.
OVERLAY_AGENT_MD = f"""---
name: {OVERLAY_AGENT_NAME}
description: "Facility-specific overlay agent shipped by the facility profile, not by OSPREY."
summary: Facility overlay agent
tools: {", ".join(sorted(OVERLAY_AGENT_TOOLS))}
disallowedTools: Bash, Write, Edit
---

# Facility Overlay Agent

Facility-specific delegate baked into the project image via `COPY .`.
"""

# Mirrors the SKILL.md frontmatter shape (see e.g. skills/demo-gallery/SKILL.md).
OVERLAY_SKILL_MD = f"""---
name: {OVERLAY_SKILL_NAME}
description: >
  Facility-specific overlay skill shipped by the facility profile, not by OSPREY.
summary: Facility overlay skill
---

# Facility Overlay Skill

Facility-specific procedure baked into the project image via `COPY .`.
"""

OVERLAY_RULE_MD = """# Facility Overlay Rule

Facility-specific operating rule baked into the project image via `COPY .`.
"""

CHANNEL_LIMITS = {"RING:BR:HCOR:01:CM:SP": {"low": -5.0, "high": 5.0}}


def _write(path: Path, content: str) -> Path:
    """Write ``content`` to ``path``, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _baked_project(tmp_path: Path) -> Path:
    """Build a project dir emulating a fully-baked project image.

    Lays down an overlay agent, an overlay skill, an overlay rule, and a
    ``data/`` file the same way ``COPY .`` would leave them at image-build
    time — no distinction between "built-in" and "overlay" artifacts on
    disk, because the fix is precisely that both are baked together.
    """
    project_dir = tmp_path / "project"

    _write(project_dir / ".claude" / "agents" / f"{OVERLAY_AGENT_NAME}.md", OVERLAY_AGENT_MD)
    _write(
        project_dir / ".claude" / "skills" / OVERLAY_SKILL_NAME / "SKILL.md",
        OVERLAY_SKILL_MD,
    )
    _write(project_dir / ".claude" / "rules" / f"{OVERLAY_RULE_NAME}.md", OVERLAY_RULE_MD)
    _write(
        project_dir / "data" / "channel_limits.json",
        json.dumps(CHANNEL_LIMITS, indent=2),
    )

    return project_dir


def _main_thread_delegation(tool_name: str, subagent_type: str) -> dict:
    """Build a main-thread Task/Agent dispatch call input in the hook's wire shape."""
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"subagent_type": subagent_type},
    }


def _decision(result: dict) -> str | None:
    return (result or {}).get("hookSpecificOutput", {}).get("permissionDecision")


def _reason(result: dict) -> str:
    return (result or {}).get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


class TestOverlayAgentVisibility:
    """The overlay agent is parsed exactly like a built-in agent would be."""

    def test_parse_project_agents_sees_the_overlay_agent(self, tmp_path):
        # Arrange
        project_dir = _baked_project(tmp_path)

        # Act
        surfaces = parse_project_agents(project_dir)

        # Assert — overlay agent present with its declared tool surface, not
        # dropped as an unrecognized/non-built-in artifact
        assert OVERLAY_AGENT_NAME in surfaces
        assert surfaces[OVERLAY_AGENT_NAME] == OVERLAY_AGENT_TOOLS


class TestOverlayAgentAcceptedForDispatch:
    """A dispatch call targeting the overlay agent is not denied as undeclared."""

    async def test_delegation_to_overlay_agent_is_not_denied_as_undeclared(self, tmp_path):
        # Arrange — the hook is built from the SAME parse the worker would run
        # against the baked project dir
        project_dir = _baked_project(tmp_path)
        surfaces = parse_project_agents(project_dir)
        hook = make_pretooluse_hook(trigger_tools=[], agent_surfaces=surfaces, denied_tools=[])
        input_data = _main_thread_delegation("Task", OVERLAY_AGENT_NAME)

        # Act
        result = await hook(input_data, "t1", None)

        # Assert — {} means "no decision": the overlay agent is recognized,
        # so delegation is granted by declaration (mirrors a built-in agent)
        assert result == {}

    async def test_delegation_to_a_genuinely_undeclared_agent_is_still_denied(self, tmp_path):
        # Arrange — control case proving the hook's "not declared" ground is
        # live and would have caught the overlay agent had it been dropped
        # (the pre-fix behavior this test guards against)
        project_dir = _baked_project(tmp_path)
        surfaces = parse_project_agents(project_dir)
        hook = make_pretooluse_hook(trigger_tools=[], agent_surfaces=surfaces, denied_tools=[])
        input_data = _main_thread_delegation("Task", "phantom-agent")

        # Act
        result = await hook(input_data, "t1", None)

        # Assert
        assert _decision(result) == "deny"
        assert "not declared" in _reason(result)
        assert "phantom-agent" in _reason(result)


class TestOverlayArtifactsPresentOnDisk:
    """The overlay skill/rule and the data/ file exist as COPY . would leave them."""

    def test_overlay_skill_file_present(self, tmp_path):
        # Arrange
        project_dir = _baked_project(tmp_path)

        # Act
        skill_path = project_dir / ".claude" / "skills" / OVERLAY_SKILL_NAME / "SKILL.md"

        # Assert
        assert skill_path.is_file()
        assert f"name: {OVERLAY_SKILL_NAME}" in skill_path.read_text(encoding="utf-8")

    def test_overlay_rule_file_present(self, tmp_path):
        # Arrange
        project_dir = _baked_project(tmp_path)

        # Act
        rule_path = project_dir / ".claude" / "rules" / f"{OVERLAY_RULE_NAME}.md"

        # Assert
        assert rule_path.is_file()
        assert "Facility Overlay Rule" in rule_path.read_text(encoding="utf-8")

    def test_data_directory_file_present_and_parseable(self, tmp_path):
        # Arrange — data/ has no regen path pre-fix; this stands in for that gap
        project_dir = _baked_project(tmp_path)

        # Act
        data_path = project_dir / "data" / "channel_limits.json"
        parsed = json.loads(data_path.read_text(encoding="utf-8"))

        # Assert
        assert data_path.is_file()
        assert parsed == CHANNEL_LIMITS
