"""Prompt Registry — declarative catalog of all Claude Code prompt artifacts.

Each prompt artifact produced by ``osprey init`` / ``osprey claude regen``
is cataloged here with a canonical name, template path, output path, and
metadata.  The registry enables:

* ``osprey prompts list`` — show all artifacts and their override status
* ``osprey prompts scaffold <name>`` — create an editable override copy
* Override resolution during ``_create_claude_code_integration()``
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptArtifact:
    """A single prompt artifact managed by OSPREY.

    Attributes:
        canonical_name: Stable identifier, e.g. ``"agents/channel-finder"``.
        template_path: Relative to ``templates/claude_code/``, e.g.
            ``"claude/agents/channel-finder.md.j2"``.
        output_path: Relative to project root, e.g.
            ``".claude/agents/channel-finder.md"``.
        description: Human-readable, shown in ``osprey prompts list``.
    """

    canonical_name: str
    template_path: str
    output_path: str
    description: str


def _get_default_artifacts() -> list[PromptArtifact]:
    """Return the declarative list of all known prompt artifacts."""
    return [
        # ── Top-level files ──────────────────────────────────────────
        PromptArtifact(
            canonical_name="claude-md",
            template_path="CLAUDE.md.j2",
            output_path="CLAUDE.md",
            description="CLAUDE.md system prompt",
        ),
        PromptArtifact(
            canonical_name="mcp-json",
            template_path="mcp.json.j2",
            output_path=".mcp.json",
            description="MCP server configuration",
        ),
        PromptArtifact(
            canonical_name="settings-json",
            template_path="claude/settings.json.j2",
            output_path=".claude/settings.json",
            description="Claude Code permissions & hooks",
        ),
        # ── Agents ───────────────────────────────────────────────────
        PromptArtifact(
            canonical_name="agents/channel-finder",
            template_path="claude/agents/channel-finder.md.j2",
            output_path=".claude/agents/channel-finder.md",
            description="Channel-finder sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/data-visualizer",
            template_path="claude/agents/data-visualizer.md.j2",
            output_path=".claude/agents/data-visualizer.md",
            description="Data visualization sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/graph-analyst",
            template_path="claude/agents/graph-analyst.md.j2",
            output_path=".claude/agents/graph-analyst.md",
            description="Graph analysis sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/literature-search",
            template_path="claude/agents/literature-search.md.j2",
            output_path=".claude/agents/literature-search.md",
            description="Literature search sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/logbook-search",
            template_path="claude/agents/logbook-search.md.j2",
            output_path=".claude/agents/logbook-search.md",
            description="Logbook search sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/logbook-deep-research",
            template_path="claude/agents/logbook-deep-research.md.j2",
            output_path=".claude/agents/logbook-deep-research.md",
            description="Logbook deep-research sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/matlab-search",
            template_path="claude/agents/matlab-search.md.j2",
            output_path=".claude/agents/matlab-search.md",
            description="MATLAB MML search sub-agent",
        ),
        PromptArtifact(
            canonical_name="agents/wiki-search",
            template_path="claude/agents/wiki-search.md.j2",
            output_path=".claude/agents/wiki-search.md",
            description="Wiki search sub-agent",
        ),
        # ── Rules ────────────────────────────────────────────────────
        PromptArtifact(
            canonical_name="rules/safety",
            template_path="claude/rules/safety.md",
            output_path=".claude/rules/safety.md",
            description="Safety & tool confinement rules",
        ),
        PromptArtifact(
            canonical_name="rules/error-handling",
            template_path="claude/rules/error-handling.md",
            output_path=".claude/rules/error-handling.md",
            description="Error taxonomy & response protocols",
        ),
        PromptArtifact(
            canonical_name="rules/artifacts",
            template_path="claude/rules/artifacts.md",
            output_path=".claude/rules/artifacts.md",
            description="Artifact-first reuse rule",
        ),
        PromptArtifact(
            canonical_name="rules/facility",
            template_path="claude/rules/facility.md.j2",
            output_path=".claude/rules/facility.md",
            description="Facility identity & context",
        ),
        PromptArtifact(
            canonical_name="rules/workflows",
            template_path="claude/rules/workflows.md",
            output_path=".claude/rules/workflows.md",
            description="Task planning, agent delegation, and parallel execution",
        ),
        PromptArtifact(
            canonical_name="rules/code-generation",
            template_path="claude/rules/code-generation.md.j2",
            output_path=".claude/rules/code-generation.md",
            description="Code generation safety rules (control_assistant only)",
        ),
        PromptArtifact(
            canonical_name="rules/timezone",
            template_path="claude/rules/timezone.md.j2",
            output_path=".claude/rules/timezone.md",
            description="Facility timezone context for timestamp interpretation",
        ),
        # ── Hooks ────────────────────────────────────────────────────
        PromptArtifact(
            canonical_name="hooks/approval",
            template_path="claude/hooks/osprey_approval.py",
            output_path=".claude/hooks/osprey_approval.py",
            description="Human-approval gate hook",
        ),
        PromptArtifact(
            canonical_name="hooks/writes-check",
            template_path="claude/hooks/osprey_writes_check.py",
            output_path=".claude/hooks/osprey_writes_check.py",
            description="Control-system write detection hook",
        ),
        PromptArtifact(
            canonical_name="hooks/error-guidance",
            template_path="claude/hooks/osprey_error_guidance.py",
            output_path=".claude/hooks/osprey_error_guidance.py",
            description="Error-guidance injection hook",
        ),
        PromptArtifact(
            canonical_name="hooks/limits",
            template_path="claude/hooks/osprey_limits.py",
            output_path=".claude/hooks/osprey_limits.py",
            description="Channel limits validation hook",
        ),
        PromptArtifact(
            canonical_name="hooks/notebook-update",
            template_path="claude/hooks/osprey_notebook_update.py",
            output_path=".claude/hooks/osprey_notebook_update.py",
            description="Notebook artifact update hook",
        ),
        PromptArtifact(
            canonical_name="hooks/memory-guard",
            template_path="claude/hooks/osprey_memory_guard.py",
            output_path=".claude/hooks/osprey_memory_guard.py",
            description="Memory guard hook",
        ),
        PromptArtifact(
            canonical_name="hooks/cf-feedback-capture",
            template_path="claude/hooks/osprey_cf_feedback_capture.py",
            output_path=".claude/hooks/osprey_cf_feedback_capture.py",
            description="Channel finder feedback capture hook",
        ),
        PromptArtifact(
            canonical_name="hooks/hook-log",
            template_path="claude/hooks/osprey_hook_log.py",
            output_path=".claude/hooks/osprey_hook_log.py",
            description="Shared hook logging utility",
        ),
        # ── Skills ──────────────────────────────────────────────────
        PromptArtifact(
            canonical_name="skills/session-report",
            template_path="claude/skills/session-report/SKILL.md",
            output_path=".claude/skills/session-report/SKILL.md",
            description="Session report generation skill",
        ),
        PromptArtifact(
            canonical_name="skills/session-report/reference",
            template_path="claude/skills/session-report/reference.md",
            output_path=".claude/skills/session-report/reference.md",
            description="CSS/JS reference patterns for session reports",
        ),
        PromptArtifact(
            canonical_name="skills/setup-mode",
            template_path="claude/skills/setup-mode/SKILL.md.j2",
            output_path=".claude/skills/setup-mode/SKILL.md",
            description="Configuration diagnostic and troubleshooting skill",
        ),
        # ── Commands ─────────────────────────────────────────────────
        PromptArtifact(
            canonical_name="commands/diagnose",
            template_path="claude/commands/diagnose.md",
            output_path=".claude/commands/diagnose.md",
            description="Operational failure diagnosis slash command",
        ),
    ]


class PromptRegistry:
    """Catalog of all known Claude Code prompt artifacts.

    Usage::

        registry = PromptRegistry.default()
        artifact = registry.get("agents/channel-finder")
        for name in registry.all_names():
            print(name)
    """

    def __init__(self, artifacts: list[PromptArtifact]) -> None:
        self._by_name: dict[str, PromptArtifact] = {a.canonical_name: a for a in artifacts}
        self._by_output: dict[str, PromptArtifact] = {a.output_path: a for a in artifacts}

    @classmethod
    def default(cls) -> PromptRegistry:
        """Create a registry populated with the default artifact list."""
        return cls(_get_default_artifacts())

    def get(self, name: str) -> PromptArtifact | None:
        """Look up an artifact by canonical name."""
        return self._by_name.get(name)

    def get_by_output(self, output_path: str) -> PromptArtifact | None:
        """Look up an artifact by its output path (relative to project root)."""
        return self._by_output.get(output_path)

    def all_artifacts(self) -> list[PromptArtifact]:
        """Return all artifacts in registration order."""
        return list(self._by_name.values())

    def all_names(self) -> list[str]:
        """Return all canonical names, sorted."""
        return sorted(self._by_name.keys())
