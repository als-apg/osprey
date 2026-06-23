"""Tests for the facility-knowledge AgentDefinition registration.

Verifies that:
- The agent appears in FRAMEWORK_AGENTS with the correct metadata.
- server_dependency is NOT set (facility-knowledge has no server dep).
- resolve_agents() returns facility-knowledge enabled in a default build.
- The agent template renders correctly with the four allowed tools.
- The gate keyword in enabled_agents guards the template output.
"""

from __future__ import annotations

import pytest

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
    matches = [a for a in agents if a["name"] == "facility-knowledge"]
    assert len(matches) == 1, "Expected exactly one facility-knowledge agent"
    return matches[0]


# ---------------------------------------------------------------------------
# FRAMEWORK_AGENTS catalog
# ---------------------------------------------------------------------------


class TestFacilityKnowledgeAgentCatalog:
    """The AgentDefinition is correctly wired in FRAMEWORK_AGENTS."""

    def test_present_in_framework_agents(self):
        assert "facility-knowledge" in FRAMEWORK_AGENTS

    def test_name_field(self):
        adef = FRAMEWORK_AGENTS["facility-knowledge"]
        assert adef.name == "facility-knowledge"

    def test_description_is_non_empty(self):
        adef = FRAMEWORK_AGENTS["facility-knowledge"]
        assert adef.description

    def test_no_condition(self):
        """facility-knowledge must be unconditional — no gating on a config key."""
        adef = FRAMEWORK_AGENTS["facility-knowledge"]
        assert adef.condition is None

    def test_no_server_dependency(self):
        """facility-knowledge does not declare a server_dependency."""
        adef = FRAMEWORK_AGENTS["facility-knowledge"]
        assert adef.server_dependency is None

    def test_enabled_by_default(self):
        adef = FRAMEWORK_AGENTS["facility-knowledge"]
        assert adef.default_enabled is True

    def test_is_not_custom(self):
        adef = FRAMEWORK_AGENTS["facility-knowledge"]
        assert adef.is_custom is False


# ---------------------------------------------------------------------------
# resolve_agents() output
# ---------------------------------------------------------------------------


class TestFacilityKnowledgeAgentResolved:
    """resolve_agents() includes facility-knowledge enabled in a default build."""

    def test_appears_in_resolved_agents(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        names = [a["name"] for a in agents]
        assert "facility-knowledge" in names

    def test_enabled_by_default(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["enabled"] is True

    def test_is_not_custom_in_resolved(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["is_custom"] is False

    def test_can_be_disabled_via_config(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"facility-knowledge": {"enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        agent = _get_agent(agents)
        assert agent["enabled"] is False

    def test_description_propagates(self):
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        agent = _get_agent(agents)
        assert agent["description"]


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestFacilityKnowledgeAgentTemplate:
    """The agent template renders correctly with all four required tools."""

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
        ctx["enabled_agents"] = {"facility-knowledge"} if enabled else set()
        return ctx

    def _render(self, tm, ctx: dict) -> str:
        return tm.jinja_env.get_template(
            "claude_code/claude/agents/facility-knowledge.md.j2"
        ).render(**ctx)

    def test_renders_when_enabled(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert rendered.strip()

    def test_renders_empty_when_disabled(self, template_manager):
        ctx = self._full_ctx(enabled=False)
        rendered = self._render(template_manager, ctx)
        assert not rendered.strip(), "Template must produce no output when agent is disabled"

    def test_contains_list_concepts_tool(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "mcp__osprey_facility_knowledge__list_concepts" in rendered

    def test_contains_read_concept_tool(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "mcp__osprey_facility_knowledge__read_concept" in rendered

    def test_contains_search_tool(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "mcp__osprey_facility_knowledge__search" in rendered

    def test_contains_submit_response_tool(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "mcp__osprey_workspace__submit_response" in rendered

    def test_disallowed_tools_present(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "disallowedTools" in rendered
        assert "Bash" in rendered

    def test_frontmatter_name_field(self, template_manager):
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "name: facility-knowledge" in rendered

    def test_no_capabilities_tool(self, template_manager):
        """capabilities is a server-level introspection tool, not exposed to the agent."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        assert "mcp__osprey_facility_knowledge__capabilities" not in rendered

    def test_no_draft_concept_tool(self, template_manager):
        """draft_concept is write-gated; the agent must not have it in tools:."""
        ctx = self._full_ctx(enabled=True)
        rendered = self._render(template_manager, ctx)
        # Must not appear in the tools: line (may appear in body text mentioning write gates)
        lines = rendered.splitlines()
        tools_line = next((line for line in lines if line.startswith("tools:")), "")
        assert "draft_concept" not in tools_line
