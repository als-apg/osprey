"""Unit tests for the dispatch permission policy (dispatch_worker.tool_policy).

Covers the two enforcement layers built per run:

* ``make_pretooluse_hook`` — the single authority. Deny-only: allowed calls
  return ``{}`` (no decision) so facility hooks and the permission flow still
  apply; everything else returns an explicit deny decision.
* ``make_backstop`` — context-aware ``can_use_tool`` for calls the CLI routes
  to the permission prompt (defense-in-depth under the hook).

Hook input dicts use the SDK wire format (snake_case): ``tool_name``,
``tool_input``, ``agent_id``/``agent_type`` (present only inside subagents).
"""

import logging
from types import SimpleNamespace

import pytest

from osprey.mcp_server.dispatch_worker.tool_policy import (
    DELEGATION_TOOLS,
    PASSTHROUGH_TOOLS,
    make_backstop,
    make_pretooluse_hook,
)

TRIGGER_TOOLS = ["mcp__controls__channel_read", "mcp__ariel__keyword_search"]
SURFACES = {
    "channel-finder": frozenset(
        {"mcp__channel-finder__search", "mcp__osprey_workspace__submit_response"}
    ),
    "wild": None,  # declared, but no explicit tools: list -> non-delegable
}
DENIED = ["Bash", "WebFetch", "mcp__plugin_playwright_playwright__*"]


def _hook(trigger=TRIGGER_TOOLS, surfaces=SURFACES, denied=DENIED):
    return make_pretooluse_hook(trigger, surfaces, denied)


def _main_input(tool, tool_input=None):
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input or {}}


def _subagent_input(tool, agent_type="channel-finder", tool_input=None):
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool,
        "tool_input": tool_input or {},
        "agent_id": "agent-123",
        "agent_type": agent_type,
    }


def _decision(result):
    return (result or {}).get("hookSpecificOutput", {}).get("permissionDecision")


def _reason(result):
    return (result or {}).get("hookSpecificOutput", {}).get("permissionDecisionReason", "")


# ---------------------------------------------------------------------------
# make_pretooluse_hook
# ---------------------------------------------------------------------------


class TestPretooluseHookContexts:
    async def test_hook_allows_trigger_tool_on_main_thread_with_no_decision(self):
        # Arrange
        hook = _hook()

        # Act
        result = await hook(_main_input("mcp__controls__channel_read"), "t1", None)

        # Assert — {} means "no decision": permission flow / facility hooks still apply
        assert result == {}

    async def test_hook_denies_non_trigger_tool_on_main_thread(self):
        # Arrange
        hook = _hook()

        # Act — settings-allowed tool that the trigger did not name
        result = await hook(_main_input("mcp__osprey_workspace__data_list"), "t1", None)

        # Assert
        assert _decision(result) == "deny"
        assert "mcp__osprey_workspace__data_list" in _reason(result)

    async def test_hook_denies_subagent_only_tool_on_main_thread(self):
        # Arrange — subagent surfaces must not leak into the main thread
        hook = _hook()

        # Act
        result = await hook(_main_input("mcp__channel-finder__search"), "t1", None)

        # Assert
        assert _decision(result) == "deny"

    async def test_hook_allows_declared_tool_inside_subagent(self):
        # Arrange
        hook = _hook()

        # Act
        result = await hook(_subagent_input("mcp__channel-finder__search"), "t1", None)

        # Assert
        assert result == {}

    async def test_hook_denies_tool_outside_subagent_surface(self):
        # Arrange — trigger tools do not leak into the subagent context
        hook = _hook()

        # Act
        result = await hook(_subagent_input("mcp__ariel__keyword_search"), "t1", None)

        # Assert
        assert _decision(result) == "deny"

    async def test_hook_denies_when_agent_id_present_but_agent_type_unknown(self):
        # Arrange
        hook = _hook()
        input_data = _subagent_input("mcp__channel-finder__search", agent_type="mystery")

        # Act
        result = await hook(input_data, "t1", None)

        # Assert — fail-closed on unknown context
        assert _decision(result) == "deny"

    async def test_hook_denies_when_agent_id_present_but_agent_type_missing(self):
        # Arrange
        hook = _hook()
        input_data = _subagent_input("mcp__channel-finder__search")
        del input_data["agent_type"]

        # Act
        result = await hook(input_data, "t1", None)

        # Assert
        assert _decision(result) == "deny"


class TestPretooluseHookDenylist:
    async def test_hook_denies_bash_even_when_in_trigger_and_surface(self):
        # Arrange — deny beats allow everywhere
        hook = make_pretooluse_hook(["Bash"], {"channel-finder": frozenset({"Bash"})}, DENIED)

        # Act
        main = await hook(_main_input("Bash"), "t1", None)
        sub = await hook(_subagent_input("Bash"), "t2", None)

        # Assert
        assert _decision(main) == "deny"
        assert _decision(sub) == "deny"

    async def test_hook_denies_prefix_denylist_entry(self):
        # Arrange
        hook = _hook()

        # Act
        result = await hook(_main_input("mcp__plugin_playwright_playwright__click"), "t1", None)

        # Assert
        assert _decision(result) == "deny"

    async def test_hook_denylist_beats_passthrough(self):
        # Arrange — a passthrough name placed on the denylist stays denied
        hook = make_pretooluse_hook(TRIGGER_TOOLS, SURFACES, ["TodoWrite"])

        # Act
        result = await hook(_main_input("TodoWrite"), "t1", None)

        # Assert
        assert _decision(result) == "deny"


class TestPretooluseHookPassthrough:
    async def test_hook_passthrough_set_is_exactly_pinned(self):
        # Assert — CF-2: permission-free harness tools only; Read/Glob/Grep excluded
        assert PASSTHROUGH_TOOLS == frozenset({"TodoWrite", "WaitForMcpServers"})

    @pytest.mark.parametrize("tool", sorted(PASSTHROUGH_TOOLS))
    async def test_hook_allows_passthrough_in_both_contexts(self, tool):
        # Arrange
        hook = _hook()

        # Act
        main = await hook(_main_input(tool), "t1", None)
        sub = await hook(_subagent_input(tool), "t2", None)

        # Assert
        assert main == {}
        assert sub == {}

    @pytest.mark.parametrize("tool", ["Read", "Glob", "Grep"])
    async def test_hook_does_not_passthrough_file_tools(self, tool):
        # Arrange — main-thread Read would expose config.yml provider config
        hook = _hook()

        # Act
        result = await hook(_main_input(tool), "t1", None)

        # Assert
        assert _decision(result) == "deny"


class TestPretooluseHookDelegation:
    @pytest.mark.parametrize("tool_name", DELEGATION_TOOLS)
    async def test_hook_allows_delegation_to_declared_tools_bearing_agent(self, tool_name):
        # Arrange
        hook = _hook()
        input_data = _main_input(tool_name, {"subagent_type": "channel-finder"})

        # Act
        result = await hook(input_data, "t1", None)

        # Assert — delegation granted by declaration, no trigger change needed
        assert result == {}

    @pytest.mark.parametrize("tool_name", DELEGATION_TOOLS)
    async def test_hook_denies_delegation_with_missing_subagent_type(self, tool_name):
        # Arrange — missing type would fall back to the general-purpose agent
        hook = _hook()

        # Act
        result = await hook(_main_input(tool_name, {}), "t1", None)

        # Assert
        assert _decision(result) == "deny"
        assert "subagent_type" in _reason(result)

    @pytest.mark.parametrize("tool_name", DELEGATION_TOOLS)
    async def test_hook_denies_delegation_to_undeclared_agent(self, tool_name):
        # Arrange
        hook = _hook()
        input_data = _main_input(tool_name, {"subagent_type": "nonexistent"})

        # Act
        result = await hook(input_data, "t1", None)

        # Assert
        assert _decision(result) == "deny"
        assert "nonexistent" in _reason(result)

    @pytest.mark.parametrize("tool_name", DELEGATION_TOOLS)
    async def test_hook_denies_delegation_to_tools_less_agent(self, tool_name):
        # Arrange — 'wild' is declared but has no explicit tools: list
        hook = _hook()
        input_data = _main_input(tool_name, {"subagent_type": "wild"})

        # Act
        result = await hook(input_data, "t1", None)

        # Assert
        assert _decision(result) == "deny"
        assert "wild" in _reason(result)
        assert "tools" in _reason(result)


class TestPretooluseHookFailClosed:
    async def test_hook_internal_error_denies_and_logs(self, caplog):
        # Arrange — a surfaces mapping that raises on lookup
        class Poisoned(dict):
            def get(self, *a, **kw):  # noqa: D401
                raise RuntimeError("boom")

        hook = make_pretooluse_hook(TRIGGER_TOOLS, Poisoned(), DENIED)

        # Act
        with caplog.at_level(logging.ERROR):
            result = await hook(_subagent_input("mcp__channel-finder__search"), "t1", None)

        # Assert — fail-closed, never fail-open
        assert _decision(result) == "deny"
        assert any("boom" in (r.message + str(r.exc_info)) for r in caplog.records)


# ---------------------------------------------------------------------------
# make_backstop
# ---------------------------------------------------------------------------


def _ctx(agent_id=None):
    return SimpleNamespace(agent_id=agent_id, suggestions=[])


def _is_allow(result) -> bool:
    return type(result).__name__ == "PermissionResultAllow"


def _is_deny(result) -> bool:
    return type(result).__name__ == "PermissionResultDeny"


class TestBackstop:
    async def test_backstop_allows_trigger_tool_on_main_thread(self):
        # Arrange
        backstop = make_backstop(TRIGGER_TOOLS, SURFACES, DENIED)

        # Act
        result = await backstop("mcp__controls__channel_read", {}, _ctx())

        # Assert
        assert _is_allow(result)

    async def test_backstop_denies_subagent_only_tool_on_main_thread(self):
        # Arrange — CC-2: no flat union on the main thread
        backstop = make_backstop(TRIGGER_TOOLS, SURFACES, DENIED)

        # Act
        result = await backstop("mcp__channel-finder__search", {}, _ctx())

        # Assert
        assert _is_deny(result)
        assert "mcp__channel-finder__search" in result.message

    async def test_backstop_allows_union_member_in_subagent_context(self):
        # Arrange — ToolPermissionContext has no agent_type, only agent_id
        backstop = make_backstop(TRIGGER_TOOLS, SURFACES, DENIED)

        # Act
        result = await backstop(
            "mcp__osprey_workspace__submit_response", {}, _ctx(agent_id="agent-1")
        )

        # Assert
        assert _is_allow(result)

    async def test_backstop_denies_non_member_in_subagent_context(self):
        # Arrange
        backstop = make_backstop(TRIGGER_TOOLS, SURFACES, DENIED)

        # Act
        result = await backstop("mcp__osprey_workspace__data_list", {}, _ctx(agent_id="a"))

        # Assert
        assert _is_deny(result)

    async def test_backstop_denylist_beats_union(self):
        # Arrange — Bash inside a declared surface is still denied
        backstop = make_backstop(TRIGGER_TOOLS, {"x": frozenset({"Bash"})}, DENIED)

        # Act
        result = await backstop("Bash", {}, _ctx(agent_id="a"))

        # Assert
        assert _is_deny(result)

    async def test_backstop_allows_passthrough_tools(self):
        # Arrange
        backstop = make_backstop(TRIGGER_TOOLS, SURFACES, DENIED)

        # Act
        main = await backstop("WaitForMcpServers", {}, _ctx())
        sub = await backstop("TodoWrite", {}, _ctx(agent_id="a"))

        # Assert
        assert _is_allow(main)
        assert _is_allow(sub)

    async def test_backstop_none_surfaces_contribute_nothing_to_union(self):
        # Arrange — only 'wild': None declared: union is empty
        backstop = make_backstop(TRIGGER_TOOLS, {"wild": None}, DENIED)

        # Act
        result = await backstop("AnyTool", {}, _ctx(agent_id="a"))

        # Assert
        assert _is_deny(result)
