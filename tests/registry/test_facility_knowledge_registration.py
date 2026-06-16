"""Tests for the osprey_facility_knowledge ServerDefinition registration.

Verifies that:
- The server appears in FRAMEWORK_SERVERS and in resolve_servers() output.
- Read tools (list_concepts, read_concept, search) land in permissions_allow.
- draft_concept lands in permissions_ask.
- The approval hook fires only for draft_concept, not for read tools.
- The standard post-error hook covers all facility_knowledge tools.
- The server appears in the generated .mcp.json and settings.json templates.
"""

from __future__ import annotations

import json

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS, resolve_servers

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


def _get_fk(servers: list[dict]) -> dict:
    matches = [s for s in servers if s["name"] == "osprey_facility_knowledge"]
    assert len(matches) == 1, "Expected exactly one osprey_facility_knowledge server"
    return matches[0]


# ---------------------------------------------------------------------------
# FRAMEWORK_SERVERS catalog presence
# ---------------------------------------------------------------------------


class TestFacilityKnowledgeCatalog:
    """The ServerDefinition is correctly wired in FRAMEWORK_SERVERS."""

    def test_present_in_framework_servers(self):
        assert "osprey_facility_knowledge" in FRAMEWORK_SERVERS

    def test_module_path(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert sdef.module == "osprey.mcp_server.facility_knowledge"

    def test_read_tools_in_permissions_allow(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert "list_concepts" in sdef.permissions_allow
        assert "read_concept" in sdef.permissions_allow
        assert "search" in sdef.permissions_allow

    def test_draft_concept_in_permissions_ask(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert "draft_concept" in sdef.permissions_ask

    def test_draft_concept_not_in_permissions_allow(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert "draft_concept" not in sdef.permissions_allow

    def test_read_tools_not_in_permissions_ask(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        for tool in ("list_concepts", "read_concept", "search"):
            assert tool not in sdef.permissions_ask, f"{tool} must not be in permissions_ask"

    def test_approval_hook_on_draft_concept_only(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        pre_rules = sdef.hooks_pre
        # Exactly one pre rule
        assert len(pre_rules) == 1
        rule = pre_rules[0]
        assert rule.matcher == "mcp__osprey_facility_knowledge__draft_concept"
        # Contains the approval hook
        commands = [h.command for h in rule.hooks]
        assert any("osprey_approval.py" in c for c in commands)

    def test_post_error_hook_covers_all_tools(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        post_rules = sdef.hooks_post
        assert len(post_rules) >= 1
        matchers = [r.matcher for r in post_rules]
        assert any("osprey_facility_knowledge__.*" in m for m in matchers)

    def test_enabled_by_default(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert sdef.default_enabled is True

    def test_osprey_config_env_var(self):
        sdef = FRAMEWORK_SERVERS["osprey_facility_knowledge"]
        assert "OSPREY_CONFIG" in sdef.env


# ---------------------------------------------------------------------------
# resolve_servers() output
# ---------------------------------------------------------------------------


class TestFacilityKnowledgeResolved:
    """The resolved server dict has the correct shape for template rendering."""

    def test_appears_in_resolved_servers(self):
        servers = resolve_servers({}, _base_ctx())
        names = [s["name"] for s in servers]
        assert "osprey_facility_knowledge" in names

    def test_enabled_by_default(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        assert fk["enabled"] is True

    def test_command_uses_python_env(self):
        ctx = _base_ctx(current_python_env="/custom/python3")
        servers = resolve_servers({}, ctx)
        fk = _get_fk(servers)
        assert fk["command"] == "/custom/python3"
        assert fk["args"] == ["-m", "osprey.mcp_server.facility_knowledge"]

    def test_read_tools_in_permissions_allow(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        for tool in ("list_concepts", "read_concept", "search"):
            assert tool in fk["permissions_allow"], f"{tool} missing from permissions_allow"

    def test_draft_concept_in_permissions_ask(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        assert "draft_concept" in fk["permissions_ask"]

    def test_approval_hook_matcher_is_draft_only(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        pre_rules = fk["hooks_pre"]
        assert len(pre_rules) == 1
        assert pre_rules[0]["matcher"] == "mcp__osprey_facility_knowledge__draft_concept"

    def test_approval_hook_fires_for_draft_concept(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        draft_rule = fk["hooks_pre"][0]
        assert "osprey_approval.py" in draft_rule["hooks"][0]["command"]

    def test_no_pre_hook_for_read_tools(self):
        """list_concepts, read_concept, search must have no pre-tool-use hooks."""
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        pre_matchers = [r["matcher"] for r in fk["hooks_pre"]]
        for tool in ("list_concepts", "read_concept", "search"):
            assert not any(tool in m for m in pre_matchers), (
                f"Unexpected pre hook found for read tool {tool!r}"
            )

    def test_post_error_hook_present(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        post_matchers = [r["matcher"] for r in fk["hooks_post"]]
        assert any("osprey_facility_knowledge__.*" in m for m in post_matchers)

    def test_env_osprey_config_resolved(self):
        ctx = _base_ctx(project_root="/my/project")
        servers = resolve_servers({}, ctx)
        fk = _get_fk(servers)
        assert fk["env"]["OSPREY_CONFIG"] == "/my/project/config.yml"

    def test_can_be_disabled_via_override(self):
        ctx = _base_ctx()
        cfg = {"servers": {"osprey_facility_knowledge": {"enabled": False}}}
        servers = resolve_servers(cfg, ctx)
        fk = _get_fk(servers)
        assert fk["enabled"] is False

    def test_is_not_custom(self):
        servers = resolve_servers({}, _base_ctx())
        fk = _get_fk(servers)
        assert fk["is_custom"] is False


# ---------------------------------------------------------------------------
# Template rendering: .mcp.json and settings.json
# ---------------------------------------------------------------------------


class TestFacilityKnowledgeTemplates:
    """Facility-knowledge server appears correctly in rendered .mcp.json and settings.json."""

    @pytest.fixture()
    def template_manager(self):
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager()

    def _full_ctx(self, **overrides):
        ctx = _base_ctx(**overrides)
        claude_code_cfg = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers(claude_code_cfg, ctx)
        ctx["agents"] = []
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = set()
        return ctx

    def _render(self, tm, template_path: str, ctx: dict) -> str:
        return tm.jinja_env.get_template(template_path).render(**ctx)

    def test_appears_in_mcp_json(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(self._render(template_manager, "claude_code/mcp.json.j2", ctx))
        assert "osprey_facility_knowledge" in data["mcpServers"]
        entry = data["mcpServers"]["osprey_facility_knowledge"]
        assert entry["command"] == "/usr/bin/python3"
        assert "-m" in entry["args"]
        assert "osprey.mcp_server.facility_knowledge" in entry["args"]

    def test_mcp_json_has_osprey_config_env(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(self._render(template_manager, "claude_code/mcp.json.j2", ctx))
        env = data["mcpServers"]["osprey_facility_knowledge"].get("env", {})
        assert env.get("OSPREY_CONFIG") == "/tmp/test-project/config.yml"

    def test_settings_json_read_tools_in_allow(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        allow = data["permissions"]["allow"]
        for tool in (
            "mcp__osprey_facility_knowledge__list_concepts",
            "mcp__osprey_facility_knowledge__read_concept",
            "mcp__osprey_facility_knowledge__search",
        ):
            assert tool in allow, f"{tool} missing from settings.json allow"

    def test_settings_json_draft_concept_in_ask(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        ask = data["permissions"]["ask"]
        assert "mcp__osprey_facility_knowledge__draft_concept" in ask

    def test_settings_json_approval_hook_in_pre(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        pre_rules = data["hooks"]["PreToolUse"]
        fk_rules = [r for r in pre_rules if "osprey_facility_knowledge" in r.get("matcher", "")]
        assert len(fk_rules) == 1
        assert fk_rules[0]["matcher"] == "mcp__osprey_facility_knowledge__draft_concept"
        cmds = [h["command"] for h in fk_rules[0]["hooks"]]
        assert any("osprey_approval.py" in c for c in cmds)

    def test_settings_json_post_error_hook(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        post_rules = data["hooks"]["PostToolUse"]
        fk_post = [r for r in post_rules if "osprey_facility_knowledge" in r.get("matcher", "")]
        assert len(fk_post) >= 1
        assert any("osprey_facility_knowledge__.*" in r["matcher"] for r in fk_post)

    def test_hook_config_json_server_prefix(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(
            self._render(template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx)
        )
        assert "mcp__osprey_facility_knowledge__" in data["server_prefixes"]
        assert "mcp__osprey_facility_knowledge__" in data["approval_prefixes"]
