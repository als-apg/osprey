"""Tests for the ``pv-directory`` ServerDefinition registration.

The PV directory MCP server reads the control system's own directory of channel
addresses. Its registration follows the graph server's read-only posture:

- It is **conditional** on ``pv_directory_configured``, the context key that is
  truthy only when the project configures a ``pv_directory`` section.
- It is **allow-only**: every tool is auto-approved, ``permissions_ask`` is
  empty and no pre hook runs, because every tool reads.
- Four tools are **optional**: each is offered only when its sub-block's
  context key holds, and an absent key withholds it.

The walk in ``test_tool_classification_drift.py`` imports the package and so
sees the unconditional tools only; the gated ones register through
``register(server)``, so this file registers them on a throwaway FastMCP and
checks that each is classified.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import pkgutil

import pytest

from osprey.registry.mcp import (
    FRAMEWORK_SERVERS,
    RENDERED_CONFIG_ENV_VALUE,
    OptionalTool,
    resolve_servers,
)
from tests.registry._ctx import render_ctx

_NAME = "pv-directory"
_CONDITION_KEY = "pv_directory_configured"

#: The tools registered on every configured server.
_UNCONDITIONAL = {"directory_search", "directory_properties", "pv_record", "ioc_record"}

#: Each gated tool and the context key that offers it.
_OPTIONAL_KEYS = {
    "alarm_state": "pv_directory_alarm_state",
    "alarm_history": "pv_directory_alarm_log",
    "alarm_config": "pv_directory_alarm_config",
    "put_history": "pv_directory_put_log",
}

_ALL_TOOLS = _UNCONDITIONAL | set(_OPTIONAL_KEYS)


def _configured_ctx(**overrides):
    return render_ctx(**{_CONDITION_KEY: True, **overrides})


def _get(servers: list[dict]) -> dict:
    matches = [s for s in servers if s["name"] == _NAME]
    assert len(matches) == 1, f"Expected exactly one {_NAME} server"
    return matches[0]


@pytest.fixture()
def sdef():
    assert _NAME in FRAMEWORK_SERVERS, f"{_NAME} server missing from FRAMEWORK_SERVERS"
    return FRAMEWORK_SERVERS[_NAME]


# ---------------------------------------------------------------------------
# FRAMEWORK_SERVERS catalog entry
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_catalog_key_matches_server_name(self, sdef):
        assert sdef.name == _NAME

    def test_module_path(self, sdef):
        assert sdef.module == "osprey.mcp_server.pv_directory"

    def test_module_is_runnable(self, sdef):
        main = importlib.import_module(f"{sdef.module}.__main__")
        assert callable(main.main)

    def test_carries_both_config_env_vars(self, sdef):
        assert sdef.env == {
            "OSPREY_CONFIG": RENDERED_CONFIG_ENV_VALUE,
            "CONFIG_FILE": RENDERED_CONFIG_ENV_VALUE,
        }

    def test_condition_is_pv_directory_configured(self, sdef):
        assert sdef.condition == _CONDITION_KEY
        assert sdef.default_enabled is True

    def test_every_tool_is_auto_allowed(self, sdef):
        assert set(sdef.permissions_allow) == _ALL_TOOLS

    def test_read_only_posture(self, sdef):
        assert sdef.permissions_ask == []
        assert sdef.fixed_allow == []
        assert sdef.fixed_ask == []
        assert sdef.hooks_pre == []
        assert sdef.writes is None

    def test_post_error_hook_covers_all_tools(self, sdef):
        assert len(sdef.hooks_post) == 1
        rule = sdef.hooks_post[0]
        assert rule.matcher == "mcp__pv-directory__.*"
        assert any("osprey_error_guidance.py" in h.command for h in rule.hooks)

    def test_optional_tools_follow_their_sub_blocks(self, sdef):
        assert sdef.optional_tools == {
            tool: OptionalTool(key) for tool, key in _OPTIONAL_KEYS.items()
        }

    def test_an_absent_key_withholds_every_optional_tool(self, sdef):
        assert sdef.withheld_optional_tools({}) == frozenset(_OPTIONAL_KEYS)


# ---------------------------------------------------------------------------
# resolve_servers() output
# ---------------------------------------------------------------------------


class TestResolved:
    def test_disabled_when_unconfigured(self):
        assert _get(resolve_servers({}, render_ctx()))["enabled"] is False

    def test_enabled_when_configured(self):
        assert _get(resolve_servers({}, _configured_ctx()))["enabled"] is True

    def test_module_and_env(self):
        server = _get(resolve_servers({}, _configured_ctx()))
        assert server["args"] == ["-m", "osprey.mcp_server.pv_directory"]
        assert server["env"]["OSPREY_CONFIG"] == "/tmp/test-project/build/config.yml"
        assert server["env"]["CONFIG_FILE"] == "/tmp/test-project/build/config.yml"

    def test_allow_rows_are_the_unconditional_tools_without_sub_blocks(self):
        server = _get(resolve_servers({}, _configured_ctx()))
        assert set(server["permissions_allow"]) == _UNCONDITIONAL
        assert server["permissions_ask"] == []

    @pytest.mark.parametrize("tool", sorted(_OPTIONAL_KEYS))
    def test_a_configured_sub_block_adds_exactly_its_tool(self, tool):
        server = _get(resolve_servers({}, _configured_ctx(**{_OPTIONAL_KEYS[tool]: True})))
        assert set(server["permissions_allow"]) == _UNCONDITIONAL | {tool}

    def test_every_sub_block_allows_every_tool(self):
        ctx = _configured_ctx(**dict.fromkeys(_OPTIONAL_KEYS.values(), True))
        assert set(_get(resolve_servers({}, ctx))["permissions_allow"]) == _ALL_TOOLS

    def test_can_be_disabled_via_override_even_when_configured(self):
        cfg = {"servers": {_NAME: {"enabled": False}}}
        assert _get(resolve_servers(cfg, _configured_ctx()))["enabled"] is False


# ---------------------------------------------------------------------------
# Tool classification, including the gated tools the drift walk cannot see
# ---------------------------------------------------------------------------


def _gated_tool_names() -> dict[str, str]:
    """Each module ``register(probe)`` adds tools from, by tool name to module name."""
    from fastmcp import FastMCP

    from osprey.mcp_server.pv_directory import tools

    found_tools: dict[str, str] = {}
    for found in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"{tools.__name__}.{found.name}")
        register = getattr(module, "register", None)
        if not callable(register) or getattr(register, "__module__", None) != module.__name__:
            continue
        probe = FastMCP(f"{found.name}-classification-probe")
        register(probe)
        for tool in asyncio.run(probe.list_tools()):
            found_tools[getattr(tool, "name", tool)] = found.name
    return found_tools


class TestGatedToolClassification:
    def test_every_gated_tool_is_allowed_and_optional_on_its_module(self, sdef):
        gated = _gated_tool_names()
        assert gated, "no gated tool registered"
        for tool, module in gated.items():
            assert tool in sdef.permissions_allow, tool
            assert tool == module, tool
            assert module in sdef.optional_tools, tool

    def test_the_optional_tools_are_exactly_the_gated_ones(self, sdef):
        assert set(_gated_tool_names()) == set(sdef.optional_tools)


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestTemplates:
    def _full_ctx(self, **overrides):
        ctx = render_ctx(**overrides)
        ctx.setdefault("facility_permissions", {})
        ctx["servers"] = resolve_servers({}, ctx)
        ctx["agents"] = []
        ctx["enabled_servers"] = {s["name"] for s in ctx["servers"] if s["enabled"]}
        ctx["enabled_agents"] = set()
        return ctx

    def _render(self, tm, template_path: str, ctx: dict) -> dict:
        return json.loads(tm.jinja_env.get_template(template_path).render(**ctx))

    def test_absent_from_the_render_when_unconfigured(self, template_manager):
        ctx = self._full_ctx()
        mcp = self._render(template_manager, "claude_code/mcp.json.j2", ctx)
        settings = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        assert _NAME not in mcp["mcpServers"]
        assert not [e for e in settings["permissions"]["allow"] if "pv-directory" in e]

    def test_settings_allow_exactly_the_configured_tools(self, template_manager):
        ctx = self._full_ctx(**{_CONDITION_KEY: True, "pv_directory_put_log": True})
        settings = self._render(template_manager, "claude_code/claude/settings.json.j2", ctx)
        prefix = "mcp__pv-directory__"
        allow = {e for e in settings["permissions"]["allow"] if e.startswith(prefix)}
        assert allow == {f"{prefix}{tool}" for tool in _UNCONDITIONAL | {"put_history"}}
        assert [e for e in settings["permissions"]["ask"] if e.startswith(prefix)] == []
        pre = [r for r in settings["hooks"]["PreToolUse"] if prefix in r.get("matcher", "")]
        assert pre == []
        post = [r["matcher"] for r in settings["hooks"]["PostToolUse"]]
        assert "mcp__pv-directory__.*" in post
