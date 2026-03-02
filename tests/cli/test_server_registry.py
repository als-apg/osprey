"""Tests for the unified server and agent registry."""

import json
import warnings

import pytest

from osprey.cli.server_registry import resolve_agents, resolve_servers

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
        assert {"controls", "workspace", "ariel", "accelpapers"} <= enabled
        # Conditional servers off (conditions not in ctx)
        assert {"matlab", "channel-finder", "confluence", "direct-channel-finder"} <= disabled

    def test_resolve_disable_framework_server(self):
        """New format: servers: {accelpapers: {enabled: false}}."""
        ctx = _base_ctx()
        servers = resolve_servers({"servers": {"accelpapers": {"enabled": False}}}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["accelpapers"] is False
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

    def test_resolve_legacy_format(self):
        """Legacy disable_servers emits deprecation warning and disables server."""
        ctx = _base_ctx()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            servers = resolve_servers({"disable_servers": ["ariel"]}, ctx)
            assert any("deprecated" in str(x.message).lower() for x in w)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["ariel"] is False

    def test_resolve_legacy_extra_servers(self):
        """Legacy extra_servers emits deprecation warning and adds server."""
        ctx = _base_ctx()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            servers = resolve_servers(
                {"extra_servers": {"foo": {"command": "cmd", "args": ["a"]}}}, ctx
            )
            assert any("deprecated" in str(x.message).lower() for x in w)
        foo = [s for s in servers if s["name"] == "foo"]
        assert len(foo) == 1
        assert foo[0]["enabled"] is True

    def test_resolve_conditional_server_enabled(self):
        """matlab in ctx → matlab server enabled."""
        ctx = _base_ctx(matlab={"path": "/opt/matlab"})
        servers = resolve_servers({}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["matlab"] is True

    def test_resolve_conditional_server_disabled(self):
        """matlab not in ctx → matlab server disabled."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        names = {s["name"]: s["enabled"] for s in servers}
        assert names["matlab"] is False

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

    def test_confluence_env_resolution(self):
        """Confluence server resolves {confluence_url} from ctx."""
        ctx = _base_ctx(confluence={"url": "https://wiki.example.com"})
        servers = resolve_servers({}, ctx)
        conf = [s for s in servers if s["name"] == "confluence"][0]
        assert conf["enabled"] is True
        assert conf["env"]["CONFLUENCE_URL"] == "https://wiki.example.com"
        assert conf["command"] == "uvx"
        assert conf["args"] == ["--python=3.12", "mcp-atlassian"]

    def test_direct_channel_finder_resolution(self):
        """direct_channel_finder in ctx → direct-channel-finder server enabled."""
        ctx = _base_ctx(direct_channel_finder=True)
        servers = resolve_servers({}, ctx)
        dcf = [s for s in servers if s["name"] == "direct-channel-finder"][0]
        assert dcf["enabled"] is True
        assert dcf["args"] == ["-m", "osprey.mcp_server.direct_channel_finder"]

    def test_channel_finder_pipeline_resolution(self):
        """channel-finder server resolves {channel_finder_pipeline} in module."""
        ctx = _base_ctx(channel_finder_pipeline="hierarchical")
        servers = resolve_servers({}, ctx)
        cf = [s for s in servers if s["name"] == "channel-finder"][0]
        assert cf["enabled"] is True
        assert cf["args"] == ["-m", "osprey.mcp_server.channel_finder_hierarchical"]


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

        assert {"logbook-search", "logbook-deep-research", "literature-search"} <= enabled
        assert {
            "wiki-search",
            "matlab-search",
            "graph-analyst",
            "channel-finder",
            "direct-channel-finder",
        } <= disabled
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

    def test_conditional_agent(self):
        """confluence in ctx → wiki-search enabled."""
        ctx = _base_ctx(confluence={"url": "https://wiki.example.com"})
        servers = resolve_servers({}, ctx)
        agents = resolve_agents({}, ctx, resolved_servers=servers)
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["wiki-search"] is True

    def test_legacy_disable_agents(self):
        """Legacy disable_agents emits deprecation warning."""
        ctx = _base_ctx()
        servers = resolve_servers({}, ctx)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agents = resolve_agents(
                {"disable_agents": ["literature-search"]},
                ctx,
                resolved_servers=servers,
            )
            assert any("deprecated" in str(x.message).lower() for x in w)
        names = {a["name"]: a["enabled"] for a in agents}
        assert names["literature-search"] is False

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
        from osprey.cli.templates import TemplateManager

        return TemplateManager()

    def _render(self, tm, template_path, ctx):
        template = tm.jinja_env.get_template(template_path)
        return template.render(**ctx)

    def _full_ctx(self, **overrides):
        """Build a context with servers and agents resolved."""
        ctx = _base_ctx(**overrides)
        claude_code_config = overrides.pop("_claude_code_config", {})
        ctx["servers"] = resolve_servers(claude_code_config, ctx)
        ctx["agents"] = resolve_agents(claude_code_config, ctx, resolved_servers=ctx["servers"])
        ctx["disable_servers"] = [s["name"] for s in ctx["servers"] if not s["enabled"]]
        ctx["disable_agents"] = [a["name"] for a in ctx["agents"] if not a["enabled"]]
        ctx["extra_servers"] = {}
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
        assert "matlab" not in data["mcpServers"]
        assert "confluence" not in data["mcpServers"]

    def test_render_mcp_json_with_conditional(self, template_manager):
        """Conditional servers appear when conditions are met."""
        ctx = self._full_ctx(
            matlab={"path": "/opt"},
            confluence={"url": "https://wiki.example.com"},
        )
        rendered = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        data = json.loads(rendered)
        assert "matlab" in data["mcpServers"]
        assert "confluence" in data["mcpServers"]
        assert data["mcpServers"]["confluence"]["command"] == "uvx"

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
        assert "literature-search" in rendered
        # Logbook guidance sentence should appear
        assert "simple lookups" in rendered
        # Conditional agents not present
        assert "wiki-search" not in rendered
        assert "matlab-search" not in rendered

    def test_render_claude_md_with_conditionals(self, template_manager):
        """Conditional agents appear in CLAUDE.md when enabled."""
        ctx = self._full_ctx(
            facility_name="Test Facility",
            confluence={"url": "https://wiki.example.com"},
        )
        rendered = self._render(template_manager, "claude_code/CLAUDE.md.j2", ctx)
        assert "wiki-search" in rendered
