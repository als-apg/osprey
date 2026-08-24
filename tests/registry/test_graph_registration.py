"""Tests for the ``graph`` ServerDefinition registration.

The graph MCP server is read-only knowledge-graph search over the ``graphdb``
service. Its registration differs from every other framework server in two ways
that these tests pin:

- It is **conditional** on ``graphdb_configured`` — a config-derived context key
  that is truthy only when the project declares ``services.graphdb``. A project
  without a graph store must not get a server whose every tool can only fail.
- It is **allow-only**: all four tools are auto-approved and ``permissions_ask``
  is empty, because nothing the agent passes at call time can mutate the store.
  A future write-capable tool must not be folded into this entry silently.

The rendering class additionally pins that the tool vocabulary in
``permissions_allow`` matches what the server package actually registers — a
stale name there renders an allow rule for a tool that does not exist, so the
real tool falls through to a prompt or a denial and nothing errors.
"""

from __future__ import annotations

import json

import pytest

from osprey.registry.mcp import FRAMEWORK_SERVERS, RENDERED_CONFIG_ENV_VALUE, resolve_servers
from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

#: The full graph tool vocabulary, spelled as a literal so a renamed or dropped
#: tool fails here rather than silently shrinking the allowlist.
_GRAPH_TOOLS = {"capabilities", "example_queries", "get_schema", "read_cypher"}

#: The context key that gates the server. Produced by ``config_derived_context``
#: in ``osprey.cli.templates.claude_code``; until that key lands the condition is
#: falsy and the server is left out of every render.
_CONDITION_KEY = "graphdb_configured"


def _base_ctx(**overrides):
    ctx = {
        "project_root": "/tmp/test-project",
        "current_python_env": "/usr/bin/python3",
        # Part of the render funnel's minimum: settings.json.j2 concatenates it
        # into the Read() globs, so a ctx without it renders filesystem-root
        # grants that no real render produces.
        "agent_data_root": DEFAULT_AGENT_DATA_BASE_DIR,
    }
    ctx.update(overrides)
    return ctx


def _configured_ctx(**overrides):
    """A context that declares a graph store, so the condition is truthy."""
    return _base_ctx(**{_CONDITION_KEY: True, **overrides})


def _get_graph(servers: list[dict]) -> dict:
    matches = [s for s in servers if s["name"] == "graph"]
    assert len(matches) == 1, "Expected exactly one graph server"
    return matches[0]


# ---------------------------------------------------------------------------
# FRAMEWORK_SERVERS catalog presence
# ---------------------------------------------------------------------------


class TestGraphCatalog:
    """The ServerDefinition is correctly wired in FRAMEWORK_SERVERS."""

    @pytest.fixture()
    def sdef(self):
        assert "graph" in FRAMEWORK_SERVERS, "graph server missing from FRAMEWORK_SERVERS"
        return FRAMEWORK_SERVERS["graph"]

    def test_catalog_key_matches_server_name(self, sdef):
        """The dict key and ``name`` must agree: the key selects the entry for an
        ``enabled`` override, while ``name`` is spliced into every permission
        string and hook matcher."""
        assert sdef.name == "graph"

    def test_module_path(self, sdef):
        assert sdef.module == "osprey.mcp_server.graph"

    def test_carries_both_config_env_vars(self, sdef):
        """``osprey.utils.config`` reads CONFIG_FILE; OSPREY_CONFIG only locates
        ``.env``. A server launched under a foreign WORKDIR without CONFIG_FILE
        falls back to ``CWD/config.yml``."""
        assert sdef.env == {
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        }

    def test_all_four_tools_are_auto_allowed(self, sdef):
        assert set(sdef.permissions_allow) == _GRAPH_TOOLS

    def test_permissions_ask_is_empty(self, sdef):
        """Read-only posture: nothing on this server is approval-gated. The ask
        set doubles as the headless side-effect classifier, so a tool landing
        here would mean the server acquired a write path."""
        assert sdef.permissions_ask == []

    def test_no_fixed_permission_entries(self, sdef):
        assert sdef.fixed_allow == []
        assert sdef.fixed_ask == []

    def test_no_pre_tool_use_hooks(self, sdef):
        """No approval / writes-check hook — every tool reads."""
        assert sdef.hooks_pre == []

    def test_post_error_hook_covers_all_tools(self, sdef):
        assert len(sdef.hooks_post) == 1
        rule = sdef.hooks_post[0]
        assert rule.matcher == "mcp__graph__.*"
        commands = [h.command for h in rule.hooks]
        assert any("osprey_error_guidance.py" in c for c in commands), commands

    def test_condition_is_graphdb_configured(self, sdef):
        assert sdef.condition == _CONDITION_KEY

    def test_default_enabled_so_the_condition_is_the_only_gate(self, sdef):
        """``default_enabled=False`` would additionally require an explicit
        ``claude_code.servers.graph.enabled`` override, making a configured
        graph store silently invisible to the agent."""
        assert sdef.default_enabled is True


# ---------------------------------------------------------------------------
# resolve_servers() output
# ---------------------------------------------------------------------------


class TestGraphResolved:
    """The resolved server dict has the correct shape for template rendering."""

    def test_disabled_without_a_configured_graph_store(self):
        graph = _get_graph(resolve_servers({}, _base_ctx()))
        assert graph["enabled"] is False

    def test_enabled_when_the_context_declares_a_graph_store(self):
        graph = _get_graph(resolve_servers({}, _configured_ctx()))
        assert graph["enabled"] is True

    def test_command_uses_python_env(self):
        ctx = _configured_ctx(current_python_env="/custom/python3")
        graph = _get_graph(resolve_servers({}, ctx))
        assert graph["command"] == "/custom/python3"
        assert graph["args"] == ["-m", "osprey.mcp_server.graph"]

    def test_env_placeholders_resolved(self):
        ctx = _configured_ctx(project_root="/my/project")
        graph = _get_graph(resolve_servers({}, ctx))
        expected = "/my/project/build/config.yml"
        assert graph["env"]["OSPREY_CONFIG"] == expected
        assert graph["env"]["CONFIG_FILE"] == expected

    def test_permissions_survive_resolution(self):
        graph = _get_graph(resolve_servers({}, _configured_ctx()))
        assert set(graph["permissions_allow"]) == _GRAPH_TOOLS
        assert graph["permissions_ask"] == []

    def test_hooks_survive_resolution(self):
        graph = _get_graph(resolve_servers({}, _configured_ctx()))
        assert graph["hooks_pre"] == []
        assert [r["matcher"] for r in graph["hooks_post"]] == ["mcp__graph__.*"]

    def test_can_be_disabled_via_override_even_when_configured(self):
        cfg = {"servers": {"graph": {"enabled": False}}}
        graph = _get_graph(resolve_servers(cfg, _configured_ctx()))
        assert graph["enabled"] is False

    def test_is_not_custom(self):
        graph = _get_graph(resolve_servers({}, _configured_ctx()))
        assert graph["is_custom"] is False


# ---------------------------------------------------------------------------
# Tool-vocabulary parity with the server package
# ---------------------------------------------------------------------------


class TestGraphToolVocabulary:
    """``permissions_allow`` names exactly the tools the package registers."""

    @pytest.fixture(scope="class")
    def framework_tool_names(self) -> dict[str, set[str]]:
        from tests.integration.test_preset_static import _collect_framework_tool_names

        return _collect_framework_tool_names()

    def test_package_scan_finds_the_graph_server(self, framework_tool_names):
        """The scanner only walks ``mcp_server/<server>/tools/`` — a tool defined
        inline in ``server.py`` is invisible to it (and to the preset-static
        reference check that consumes it)."""
        assert "graph" in framework_tool_names, (
            "graph tools are not discoverable; every tool must live in "
            f"src/osprey/mcp_server/graph/tools/. Found: {sorted(framework_tool_names)}"
        )

    def test_registered_tools_match_the_allowlist(self, framework_tool_names):
        assert framework_tool_names["graph"] == _GRAPH_TOOLS

    def test_no_alias_entry_needed(self, framework_tool_names):
        """The package directory name equals the FRAMEWORK_SERVERS name, so the
        ``mcp__graph__<tool>`` strings resolve without a hard-coded alias."""
        from tests.integration.test_preset_static import _server_alias_map

        aliases = _server_alias_map(framework_tool_names)
        assert aliases["graph"] == _GRAPH_TOOLS
        assert set(FRAMEWORK_SERVERS["graph"].permissions_allow) <= aliases["graph"]


# ---------------------------------------------------------------------------
# Template rendering: .mcp.json, settings.json, hook_config.json
# ---------------------------------------------------------------------------


class TestGraphTemplates:
    """The graph server renders correctly — and only when it is configured."""

    @pytest.fixture()
    def template_manager(self):
        from osprey.cli.templates.manager import TemplateManager

        return TemplateManager()

    def _full_ctx(self, **overrides):
        ctx = _base_ctx(**overrides)
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers({}, ctx)
        ctx["agents"] = []
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = set()
        return ctx

    def _render(self, tm, template_path: str, ctx: dict) -> str:
        return tm.jinja_env.get_template(template_path).render(**ctx)

    def test_absent_from_mcp_json_without_a_graph_store(self, template_manager):
        """No graph store ⇒ the render is byte-identical to one without this
        entry. This is what keeps the registration inert until the
        ``graphdb_configured`` context key is produced."""
        ctx = self._full_ctx()
        data = json.loads(self._render(template_manager, "claude_code/mcp.json.j2", ctx))
        assert "graph" not in data["mcpServers"]

    def test_absent_from_settings_json_without_a_graph_store(self, template_manager):
        ctx = self._full_ctx()
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        graph_entries = [e for e in data["permissions"]["allow"] if e.startswith("mcp__graph__")]
        assert graph_entries == []

    def test_appears_in_mcp_json_when_configured(self, template_manager):
        ctx = self._full_ctx(**{_CONDITION_KEY: True})
        data = json.loads(self._render(template_manager, "claude_code/mcp.json.j2", ctx))
        assert "graph" in data["mcpServers"]
        entry = data["mcpServers"]["graph"]
        assert entry["command"] == "/usr/bin/python3"
        assert entry["args"] == ["-m", "osprey.mcp_server.graph"]
        assert entry["env"]["OSPREY_CONFIG"] == "/tmp/test-project/build/config.yml"
        assert entry["env"]["CONFIG_FILE"] == "/tmp/test-project/build/config.yml"

    def test_settings_json_allows_every_tool(self, template_manager):
        ctx = self._full_ctx(**{_CONDITION_KEY: True})
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        allow = set(data["permissions"]["allow"])
        assert {f"mcp__graph__{tool}" for tool in _GRAPH_TOOLS} <= allow

    def test_settings_json_asks_for_nothing_and_gates_nothing(self, template_manager):
        ctx = self._full_ctx(**{_CONDITION_KEY: True})
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        assert [e for e in data["permissions"]["ask"] if e.startswith("mcp__graph__")] == []
        pre = [r for r in data["hooks"]["PreToolUse"] if "graph" in r.get("matcher", "")]
        assert pre == []

    def test_settings_json_post_error_hook(self, template_manager):
        ctx = self._full_ctx(**{_CONDITION_KEY: True})
        data = json.loads(
            self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        )
        matchers = [r["matcher"] for r in data["hooks"]["PostToolUse"]]
        assert "mcp__graph__.*" in matchers

    def test_hook_config_json_prefixes(self, template_manager):
        ctx = self._full_ctx(**{_CONDITION_KEY: True})
        data = json.loads(
            self._render(template_manager, "claude_code/claude/hooks/hook_config.json.j2", ctx)
        )
        assert "mcp__graph__" in data["server_prefixes"]
        # No approval hook ⇒ the prefix must NOT appear in approval_prefixes,
        # which would make the approval hook claim tools it never gates.
        assert "mcp__graph__" not in data["approval_prefixes"]
