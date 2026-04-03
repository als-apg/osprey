"""Tests for the unified server and agent registry."""

import json
import logging

import pytest

from osprey.registry.mcp import HOOK_PRESETS, resolve_agents, resolve_servers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_ctx(**overrides):
    """Build a minimal template context for testing."""
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
    }
    ctx.update(overrides)
    return ctx


# ---------------------------------------------------------------------------
# Server resolution tests
# ---------------------------------------------------------------------------


class TestResolveServers:
    """Tests for resolve_servers()."""

    def test_resolve_default_config(self):
        """No overrides → core servers enabled, optional servers disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        enabled = {s["name"] for s in servers if s["enabled"]}
        disabled = {s["name"] for s in servers if not s["enabled"]}

        # Core servers always on
        assert {"controls", "workspace", "ariel"} <= enabled
        # Conditional servers off (conditions not in ctx)
        assert {"channel-finder"} <= disabled

    def test_resolve_disable_framework_server(self):
        """New format: servers: {ariel: {enabled: false}}."""
        ctx = _base_ctx()
        servers = resolve_servers({"servers": {"ariel": {"enabled": False}}}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["ariel"] is False
        # Other core servers still enabled
        assert names["controls"] is True

    def test_resolve_custom_server(self):
        """Custom server in new format appears with correct structure."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {"MY_KEY": "val"},
                    "permissions": {"allow": ["read_data"], "ask": ["write_data"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        custom = [s for s in servers if s["name"] == "my-server"]
        assert len(custom) == 1
        s = custom[0]
        assert s["enabled"] is True
        assert s["command"] == "node"
        assert s["args"] == ["server.js"]
        assert s["env"] == {"MY_KEY": "val"}
        assert s["permissions_allow"] == ["read_data"]
        assert s["permissions_ask"] == ["write_data"]
        assert s["is_custom"] is True

    def test_resolve_custom_server_url_transport(self):
        """Custom URL/SSE server resolves with url field, empty command."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "remote-api": {
                    "url": "http://remote:8001/sse",
                    "permissions": {"allow": ["search"], "ask": []},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        remote = [s for s in servers if s["name"] == "remote-api"]
        assert len(remote) == 1
        s = remote[0]
        assert s["url"] == "http://remote:8001/sse"
        assert s["command"] == ""
        assert s["args"] == []
        assert s["permissions_allow"] == ["search"]
        assert s["is_custom"] is True

    def test_resolve_conditional_server_enabled(self):
        """channel_finder_pipeline in ctx → channel-finder server enabled."""
        ctx = _base_ctx(channel_finder_pipeline="hierarchical")
        servers = resolve_servers({}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["channel-finder"] is True

    def test_resolve_conditional_server_disabled(self):
        """channel_finder_pipeline not in ctx → channel-finder server disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["channel-finder"] is False

    def test_env_resolution(self):
        """Placeholder {project_root} in env values is resolved."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        controls = [s for s in servers if s["name"] == "controls"][0]
        assert controls["env"]["OSPREY_CONFIG"] == "/tmp/test-project/config.yml"
        # Shell variables ${...} are preserved
        assert controls["env"]["EPICS_CA_ADDR_LIST"] == "${EPICS_CA_ADDR_LIST:-}"

    def test_controls_hook_structure(self):
        """controls server has 3 distinct PreToolUse hook rules."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        controls = [s for s in servers if s["name"] == "controls"][0]
        pre_matchers = [r["matcher"] for r in controls["hooks_pre"]]
        assert "mcp__controls__channel_write" in pre_matchers
        assert "mcp__controls__channel_read" in pre_matchers
        assert "mcp__controls__archiver_read" in pre_matchers
        # channel_write has 3 hooks
        cw = [r for r in controls["hooks_pre"] if r["matcher"] == "mcp__controls__channel_write"][0]
        assert len(cw["hooks"]) == 3

    def test_channel_finder_pipeline_resolution(self):
        """channel-finder server resolves {channel_finder_pipeline} in module."""
        ctx = _base_ctx(channel_finder_pipeline="hierarchical")
        servers = resolve_servers({}, ctx)
        cf = [s for s in servers if s["name"] == "channel-finder"][0]
        assert cf["enabled"] is True
        assert cf["args"] == ["-m", "osprey.mcp_server.channel_finder_hierarchical"]

    def test_custom_server_with_approval_preset(self):
        """Custom server with hooks.pre_tool_use: [approval] gets approval hook."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc_server"],
                    "hooks": {"pre_tool_use": ["approval"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        plc = [s for s in servers if s["name"] == "my-plc"][0]
        assert plc["enabled"] is True
        assert len(plc["hooks_pre"]) == 1
        rule = plc["hooks_pre"][0]
        assert rule["matcher"] == "mcp__my-plc__.*"
        assert len(rule["hooks"]) == 1
        assert "osprey_approval.py" in rule["hooks"][0]["command"]

    def test_custom_server_with_multiple_presets(self):
        """Custom server with multiple hook presets gets all hooks in one rule."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc_server"],
                    "hooks": {"pre_tool_use": ["approval", "writes_check"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        plc = [s for s in servers if s["name"] == "my-plc"][0]
        assert len(plc["hooks_pre"]) == 1
        rule = plc["hooks_pre"][0]
        assert len(rule["hooks"]) == 2
        commands = [h["command"] for h in rule["hooks"]]
        assert any("osprey_approval.py" in c for c in commands)
        assert any("osprey_writes_check.py" in c for c in commands)

    def test_custom_server_invalid_preset_warns(self, caplog):
        """Unknown hook preset logs a warning and is skipped."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc_server"],
                    "hooks": {"pre_tool_use": ["approval", "bogus"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        with caplog.at_level(logging.WARNING):
            servers = resolve_servers(cfg, ctx)
        plc = [s for s in servers if s["name"] == "my-plc"][0]
        # Only the valid preset should be in the hooks
        assert len(plc["hooks_pre"]) == 1
        assert len(plc["hooks_pre"][0]["hooks"]) == 1
        assert "osprey_approval.py" in plc["hooks_pre"][0]["hooks"][0]["command"]
        # Warning should have been logged
        assert any("bogus" in r.message for r in caplog.records)

    def test_custom_server_no_hooks_key(self):
        """Custom server without hooks key gets no pre-tool-use hooks."""
        ctx = _base_ctx()
        cfg = {
            "servers": {
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "permissions": {"allow": ["read_data"]},
                }
            }
        }
        servers = resolve_servers(cfg, ctx)
        custom = [s for s in servers if s["name"] == "my-server"][0]
        assert custom["hooks_pre"] == []

    def test_hook_presets_dict_contains_expected_keys(self):
        """HOOK_PRESETS has the three expected preset names."""
        assert set(HOOK_PRESETS.keys()) == {"approval", "writes_check", "limits"}


# ---------------------------------------------------------------------------
# Agent resolution tests
# ---------------------------------------------------------------------------


class TestResolveAgents:
    """Tests for resolve_agents()."""

    def test_default_agents(self):
        """Default config → core agents enabled, conditional ones disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        enabled = {a["name"] for a in agents if a["enabled"]}
        disabled = {a["name"] for a in agents if not a["enabled"]}

        assert {"logbook-search", "logbook-deep-research"} <= enabled
        assert {"channel-finder"} <= disabled
        assert "data-visualizer" in enabled

    def test_disable_framework_agent(self):
        """New format: agents: {logbook-search: {enabled: false}}."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"logbook-search": {"enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["logbook-search"] is False

    def test_custom_agent_from_config(self):
        """Custom agent defined in config appears with correct fields."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"my-db-agent": {"description": "Searches the facility DB."}}},
            ctx,
            resolved_servers=servers,
        )
        custom = [a for a in agents if a["name"] == "my-db-agent"]
        assert len(custom) == 1
        assert custom[0]["enabled"] is True
        assert custom[0]["is_custom"] is True
        assert custom[0]["description"] == "Searches the facility DB."

    def test_custom_agent_with_condition_disabled(self):
        """Custom agent with condition key not in ctx → disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {
                "agents": {
                    "cond-agent": {"description": "Needs feature.", "condition": "my_feature"}
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["cond-agent"] is False

    def test_custom_agent_with_condition_enabled(self):
        """Custom agent with condition key in ctx → enabled."""
        ctx = _base_ctx(my_feature=True)
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {
                "agents": {
                    "cond-agent": {"description": "Needs feature.", "condition": "my_feature"}
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["cond-agent"] is True

    def test_custom_agent_with_server_dependency_disabled(self):
        """Custom agent with server_dependency on inactive server → disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {
                "agents": {
                    "dep-agent": {
                        "description": "Depends on missing server.",
                        "server_dependency": "nonexistent-server",
                    }
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["dep-agent"] is False

    def test_custom_agent_with_server_dependency_enabled(self):
        """Custom agent with server_dependency on active server → enabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        # "controls" is enabled by default
        agents = resolve_agents(
            {
                "agents": {
                    "dep-agent": {
                        "description": "Depends on controls.",
                        "server_dependency": "controls",
                    }
                }
            },
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["dep-agent"] is True

    def test_custom_agent_explicitly_disabled(self):
        """Custom agent with enabled: false in config → not created."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents(
            {"agents": {"skip-agent": {"description": "Should not appear.", "enabled": False}}},
            ctx,
            resolved_servers=servers,
        )
        names = {a["name"] for a in agents}
        assert "skip-agent" not in names

    def test_auto_discovery(self, tmp_path):
        """Custom agent .md file in .claude/agents/ appears in resolved list."""
        agents_dir = tmp_path / ".claude" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "my-agent.md").write_text(
            '---\nname: my-agent\ndescription: "A custom agent"\n---\n\n# My Agent\n'
        )
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, project_dir=tmp_path, resolved_servers=servers)
        custom = [a for a in agents if a["name"] == "my-agent"]
        assert len(custom) == 1
        assert custom[0]["is_custom"] is True
        assert custom[0]["enabled"] is True
        assert custom[0]["description"] == "A custom agent"


# ---------------------------------------------------------------------------
# Template rendering tests
# ---------------------------------------------------------------------------


class TestTemplateRendering:
    """Tests that templates render valid JSON with the new data-driven approach."""

    @pytest.fixture()
    def template_manager(self):
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager()

    def _render(self, tm, template_path, ctx):
        template = tm.jinja_env.get_template(template_path)
        return template.render(**ctx)

    def _full_ctx(self, **overrides):
        """Build a context with servers and agents resolved."""
        ctx = _base_ctx(**overrides)
        claude_code_config = overrides.pop("_claude_code_config", {})
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers(claude_code_config, ctx)
        ctx["agents"] = resolve_agents(claude_code_config, ctx, resolved_servers=ctx["servers"])
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = {a["name"] for a in ctx["agents"] if a["enabled"]}
        return ctx

    def test_render_mcp_json(self, template_manager):
        """Rendered mcp.json is valid JSON with expected servers."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "mcpServers" in data
        assert "controls" in data["mcpServers"]
        assert "workspace" in data["mcpServers"]
        # Conditional servers not present
        assert "channel-finder" not in data["mcpServers"]

    def test_render_mcp_json_with_conditional(self, template_manager):
        """Conditional servers appear when conditions are met."""
        ctx = self._full_ctx(
            channel_finder_pipeline="hierarchical",
        )
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "channel-finder" in data["mcpServers"]

    def test_render_mcp_json_url_server(self, template_manager):
        """URL/SSE servers render as {type: sse, url: ...} entries."""
        ctx = self._full_ctx(
            _claude_code_config={
                "servers": {
                    "remote-api": {
                        "url": "http://remote:8001/sse",
                    }
                }
            }
        )
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "remote-api" in data["mcpServers"]
        remote = data["mcpServers"]["remote-api"]
        assert remote == {"type": "sse", "url": "http://remote:8001/sse"}
        # Framework servers still have command/args, no type
        controls = data["mcpServers"]["controls"]
        assert "command" in controls
        assert "type" not in controls

    def test_render_settings_json(self, template_manager):
        """Rendered settings.json is valid JSON with permissions and hooks."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        data = json.loads(rendered)
        assert "permissions" in data
        assert "allow" in data["permissions"]
        assert "deny" in data["permissions"]
        assert "ask" in data["permissions"]
        # Check a sample allow entry
        allow = data["permissions"]["allow"]
        assert '"Read(_agent_data/**)"' not in allow  # Not double-quoted
        assert "Read(_agent_data/**)" in allow
        assert "mcp__workspace__data_read" in allow
        # Check ask entries
        ask = data["permissions"]["ask"]
        assert "mcp__controls__channel_write" in ask
        # Check hooks structure
        assert "hooks" in data
        assert "PreToolUse" in data["hooks"]
        assert "PostToolUse" in data["hooks"]

    def test_render_settings_json_hooks(self, template_manager):
        """Hook rules from controls server appear in rendered output."""
        ctx = self._full_ctx()
        rendered = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        data = json.loads(rendered)
        pre_matchers = [r["matcher"] for r in data["hooks"]["PreToolUse"]]
        assert "Write" in pre_matchers  # Fixed memory guard hook
        assert "mcp__controls__channel_write" in pre_matchers
        post_matchers = [r["matcher"] for r in data["hooks"]["PostToolUse"]]
        assert "NotebookEdit" in post_matchers  # Fixed entry
        assert "mcp__controls__.*" in post_matchers

    def test_render_claude_md(self, template_manager):
        """Rendered CLAUDE.md includes enabled agents."""
        ctx = self._full_ctx(facility_name="Test Facility")
        rendered = self._render(template_manager, "claude_code/CLAUDE.md.j2", ctx)
        assert "logbook-search" in rendered
        assert "logbook-deep-research" in rendered
        # Logbook guidance sentence should appear
        assert "simple lookups" in rendered

    def test_render_hook_config_json(self, template_manager):
        """hook_config.json.j2 renders valid JSON with server/approval prefixes."""
        ctx = self._full_ctx()
        rendered = self._render(
            template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx
        )
        data = json.loads(rendered)
        assert "server_prefixes" in data
        assert "approval_prefixes" in data
        # Core servers should be in server_prefixes
        assert "mcp__controls__" in data["server_prefixes"]
        assert "mcp__workspace__" in data["server_prefixes"]
        # Controls, workspace, and ariel all have approval hooks
        assert "mcp__controls__" in data["approval_prefixes"]
        assert "mcp__workspace__" in data["approval_prefixes"]
        assert "mcp__ariel__" in data["approval_prefixes"]

    def test_render_hook_config_json_with_custom_server(self, template_manager):
        """Custom server with approval preset appears in both prefix lists."""
        cfg = {
            "servers": {
                "my-plc": {
                    "command": "python",
                    "args": ["-m", "my_plc"],
                    "hooks": {"pre_tool_use": ["approval"]},
                    "permissions": {"ask": ["set_output"]},
                }
            }
        }
        ctx = self._full_ctx(_claude_code_config=cfg)
        rendered = self._render(
            template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx
        )
        data = json.loads(rendered)
        assert "mcp__my-plc__" in data["server_prefixes"]
        assert "mcp__my-plc__" in data["approval_prefixes"]
