"""Tests for the pyat-specialist AgentDefinition registration and wiring.

Verifies that:
- The agent appears in FRAMEWORK_AGENTS with the correct metadata and a
  ``python`` server dependency.
- AGENT_DEFAULT_TIERS pins pyat-specialist to the ``sonnet`` tier.
- resolve_agents() enables pyat-specialist by default when the python server is
  resolved, disables it when the python server is disabled, and honors a config
  override.
- The agent template renders the exact tools/disallowedTools/maxTurns frontmatter
  and the distinctive body behaviors, and produces no output when disabled.
- The control-assistant CLAUDE.md renders exactly one pyat-specialist roster
  bullet, the delegation prohibition, and the documented-vs-computed boundary in
  both the pyat-specialist and facility-knowledge blocks.

All template assertions pin what the artifacts actually contain (truthful pins).
"""

from __future__ import annotations

import re

import pytest

from osprey.build.claude_code_resolver import AGENT_DEFAULT_TIERS
from osprey.registry.mcp import FRAMEWORK_AGENTS, resolve_agents, resolve_servers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_ctx(**overrides):
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
    }
    ctx.update(overrides)
    return ctx


def _get_agent(agents: list[dict]) -> dict:
    matches = [a for a in agents if a["name"] == "pyat-specialist"]
    assert len(matches) == 1, "Expected exactly one pyat-specialist agent"
    return matches[0]


# ---------------------------------------------------------------------------
# FRAMEWORK_AGENTS catalog + tier
# ---------------------------------------------------------------------------


class TestPyatSpecialistAgentCatalog:
    """The AgentDefinition is correctly wired in FRAMEWORK_AGENTS."""

    def test_present_in_framework_agents(self):
        assert "pyat-specialist" in FRAMEWORK_AGENTS

    def test_name_field(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.name == "pyat-specialist"

    def test_description_is_non_empty(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.description

    def test_server_dependency_is_python(self):
        """pyat-specialist computes via mcp__python__execute — it needs the python server."""
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.server_dependency == "python"

    def test_enabled_by_default(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.default_enabled is True

    def test_is_not_custom(self):
        adef = FRAMEWORK_AGENTS["pyat-specialist"]
        assert adef.is_custom is False

    def test_default_tier_is_sonnet(self):
        assert AGENT_DEFAULT_TIERS["pyat-specialist"] == "sonnet"


# ---------------------------------------------------------------------------
# resolve_agents() output
# ---------------------------------------------------------------------------


class TestPyatSpecialistAgentResolved:
    """resolve_agents() gates pyat-specialist on the python server dependency."""

    def test_appears_in_resolved_agents(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        names = [a["name"] for a in agents]
        assert "pyat-specialist" in names

    def test_enabled_by_default_when_python_resolved(self):
        """Default build: the python server is enabled, so pyat-specialist is enabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        assert "python" in {s["name"] for s in servers if s["enabled"]}
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["enabled"] is True

    def test_disabled_when_python_server_disabled(self):
        """Disabling the python server must disable the pyat-specialist agent."""
        ctx = _base_ctx()
        cfg = {"servers": {"python": {"enabled": False}}}
        servers = resolve_servers(cfg, ctx)
        assert "python" not in {s["name"] for s in servers if s["enabled"]}
        agents = resolve_agents(cfg, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["enabled"] is False

    def test_can_be_disabled_via_config(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"pyat-specialist": {"enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        agent = _get_agent(agents)
        assert agent["enabled"] is False

    def test_is_not_custom_in_resolved(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["is_custom"] is False

    def test_description_propagates(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["description"]


# ---------------------------------------------------------------------------
# Agent template rendering
# ---------------------------------------------------------------------------


class TestPyatSpecialistAgentTemplate:
    """The agent template renders the exact frontmatter and body behaviors."""

    @pytest.fixture()
    def template_manager(self):
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager()

    def _full_ctx(self, enabled: bool = True, **overrides):
        ctx = _base_ctx(**overrides)
        claude_code_cfg = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_name", "Test Facility")
        ctx["servers"] = resolve_servers(claude_code_cfg, ctx)
        ctx["agents"] = []
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {"pyat-specialist"} if enabled else set()
        return ctx

    def _render(self, tm, ctx: dict) -> str:
        return tm.jinja_env.get_template("claude_code/claude/agents/pyat-specialist.md.j2").render(
            **ctx
        )

    def test_renders_when_enabled(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert rendered.strip()

    def test_renders_empty_when_disabled(self, template_manager):
        ctx = self._full_ctx(enabled=False)
        rendered = self._render(template_manager, ctx)
        assert not rendered.strip(), "Template must produce no output when agent is disabled"

    def test_frontmatter_name_field(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "name: pyat-specialist" in rendered

    def test_frontmatter_tools_exact(self, template_manager):
        """The tools: line pins exactly the six allowed tools, in order."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        expected = (
            "tools: mcp__python__execute, mcp__osprey_workspace__submit_response, "
            "mcp__osprey_workspace__artifact_save, mcp__osprey_workspace__data_list, "
            "mcp__osprey_workspace__data_read, Read"
        )
        assert expected in rendered

    def test_frontmatter_disallowed_tools_exact(self, template_manager):
        """The disallowedTools: line pins exactly the eleven blocked tools, in order."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        expected = (
            "disallowedTools: Bash, Write, Edit, Glob, Grep, WebFetch, WebSearch, "
            "NotebookEdit, Task, Skill, Agent"
        )
        assert expected in rendered

    def test_frontmatter_max_turns(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "maxTurns: 30" in rendered

    def test_no_readwrite_write_edit_in_tools(self, template_manager):
        """Compute-only agent: no Write/Edit/Bash in the tools: line."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        tools_line = next((ln for ln in rendered.splitlines() if ln.startswith("tools:")), "")
        for banned in ("Bash", "Write", "Edit"):
            assert banned not in tools_line

    def test_body_execution_mode_readonly_pin(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert 'execution_mode="readonly"' in rendered
        assert 'Never use `execution_mode="readwrite"`.' in rendered

    def test_body_disable_6d(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "disable_6d" in rendered
        assert "ring4d.disable_6d()" in rendered

    def test_body_edge_case_silent_nan(self, template_manager):
        """Edge case: pyAT returns NaN without raising — must guard results."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "NaN without raising" in rendered

    def test_body_edge_case_trace_stability_guard(self, template_manager):
        """Edge case: one-turn map |trace| < 2 stability test."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "|trace| must be < 2 per plane" in rendered

    def test_body_edge_case_module_not_found(self, template_manager):
        """Edge case: ModuleNotFoundError → report the environment limitation and stop."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        normalized = " ".join(rendered.split())
        assert "ModuleNotFoundError" in rendered
        assert "Report that environment limitation plainly and stop" in normalized

    def test_body_edge_case_type_enumeration_over_name_pattern(self, template_manager):
        """Edge case: enumerate elements by pyAT type, never by FamName pattern."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "get_uint32_index" in rendered
        assert "Do **not** scan `FamName`" in rendered

    def test_body_edge_case_flag_heavy_runs(self, template_manager):
        """Edge case: flag heavy DA/LMA/tracking runs before launching them."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "Flag heavy runs before launching them" in rendered

    def test_body_native_type_conversion_instruction(self, template_manager):
        """Native-type coercion before save (json.dumps default=str lossily stringifies)."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        normalized = " ".join(rendered.split())
        assert "convert every computed quantity to a NATIVE Python type" in normalized
        assert "lossily stringifies" in rendered


# ---------------------------------------------------------------------------
# Control-assistant CLAUDE.md rendering
# ---------------------------------------------------------------------------


class TestPyatSpecialistClaudeMd:
    """CLAUDE.md renders the roster bullet, delegation prohibition, and boundary."""

    @pytest.fixture()
    def template_manager(self):
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager()

    def _full_ctx(self, **overrides):
        ctx = _base_ctx(**overrides)
        claude_code_cfg = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_name", "Test Facility")
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers(claude_code_cfg, ctx)
        ctx["agents"] = resolve_agents(claude_code_cfg, ctx, resolved_servers=ctx["servers"])
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
        return ctx

    def _render(self, tm, ctx: dict) -> str:
        return tm.jinja_env.get_template("claude_code/CLAUDE.md.j2").render(**ctx)

    def test_pyat_specialist_enabled_in_default_build(self, template_manager):
        """Sanity: the default build enables pyat-specialist (python server on)."""
        ctx = self._full_ctx()
        assert "pyat-specialist" in ctx["enabled_agents"]

    def test_exactly_one_roster_bullet(self, template_manager):
        """The roster lists pyat-specialist exactly once — no duplicate from the fallthrough loop."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        matches = re.findall(r"^- \*\*pyat-specialist\*\*", rendered, re.MULTILINE)
        assert len(matches) == 1

    def test_delegation_prohibition_text(self, template_manager):
        """The roster tells the parent NOT to compute lattice quantities itself."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        assert (
            "do NOT compute lattice quantities (orbit, tunes, optics functions, "
            "response matrices) via `mcp__python__execute` yourself" in rendered
        )

    def test_documented_vs_computed_in_pyat_block(self, template_manager):
        """The pyat-specialist block carries the documented-vs-computed boundary."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        assert (
            "Anything that must be *computed* on the lattice model goes to **pyat-specialist**."
            in rendered
        )

    def test_documented_vs_computed_in_facility_knowledge_block(self, template_manager):
        """The facility-knowledge block carries the documented-vs-computed boundary too."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        assert (
            "Anything that requires *computing* on the lattice model — orbit, tunes, "
            "optics functions, response matrices — goes to **pyat-specialist**, not here."
            in rendered
        )

    def test_documented_vs_computed_phrase_in_both_blocks(self, template_manager):
        """The 'Documented vs. computed:' lead-in appears in both agent blocks."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, ctx)
        assert rendered.count("Documented vs. computed:") == 2
