"""Unit tests for the dispatch permission policy (osprey.agent_runner.tool_policy).

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

from osprey.agent_runner.tool_policy import (
    DELEGATION_TOOLS,
    PASSTHROUGH_TOOLS,
    make_backstop,
    make_pretooluse_hook,
    narrow_allowed_tools,
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
        result = await hook(_main_input("mcp__osprey_workspace__artifact_list"), "t1", None)

        # Assert
        assert _decision(result) == "deny"
        assert "mcp__osprey_workspace__artifact_list" in _reason(result)

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
        hook = make_pretooluse_hook(TRIGGER_TOOLS, SURFACES, ["TaskCreate"])

        # Act
        result = await hook(_main_input("TaskCreate"), "t1", None)

        # Assert
        assert _decision(result) == "deny"


class TestPretooluseHookPassthrough:
    async def test_hook_passthrough_set_is_exactly_pinned(self):
        # Assert — permission-free task-list tools only; Read/Glob/Grep excluded
        assert PASSTHROUGH_TOOLS == frozenset({"TaskCreate", "TaskGet", "TaskList", "TaskUpdate"})

    @pytest.mark.parametrize("tool", ["TodoWrite", "WaitForMcpServers"])
    async def test_hook_denies_names_outside_the_passthrough_set(self, tool):
        # Arrange — neither name passes without a trigger allow rule
        hook = _hook()

        # Act
        result = await hook(_main_input(tool), "t1", None)

        # Assert
        assert _decision(result) == "deny"

    def test_passthrough_set_never_names_a_floor_or_delegation_tool(self):
        # Arrange
        from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS

        # Assert — the Task-prefixed background-command tools never pass through
        assert not PASSTHROUGH_TOOLS & (DISPATCH_DENIED_TOOLS | set(DELEGATION_TOOLS))
        assert "TaskStop" not in PASSTHROUGH_TOOLS

    async def test_hook_floor_beats_passthrough_under_the_server_denylist(self):
        # Arrange — built the way a real dispatch run is wired
        from osprey.agent_runner import tool_names

        hook = make_pretooluse_hook(
            ["TaskStop"], SURFACES, sorted(tool_names.DISPATCH_DENIED_TOOLS)
        )

        # Act
        floored = await hook(_main_input("TaskStop"), "t1", None)
        passed = await hook(_main_input("TaskCreate"), "t2", None)

        # Assert
        assert _decision(floored) == "deny"
        assert passed == {}

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
            def get(self, *a, **kw):
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
        result = await backstop("mcp__osprey_workspace__artifact_list", {}, _ctx(agent_id="a"))

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
        main = await backstop("TaskList", {}, _ctx())
        sub = await backstop("TaskUpdate", {}, _ctx(agent_id="a"))

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


# ---------------------------------------------------------------------------
# narrow_allowed_tools
# ---------------------------------------------------------------------------


class TestNarrowAllowedTools:
    def test_narrows_allowed_tools_to_surface_intersection(self):
        # Arrange — surface_tools names a subset of the trigger's allow set
        trigger = ["A", "B", "C"]
        surface = ["A", "B"]

        # Act
        result = narrow_allowed_tools(trigger, surface)

        # Assert — C dropped, order/membership preserved from trigger
        assert result == ["A", "B"]

    def test_surface_tools_never_adds_a_tool_absent_from_trigger(self):
        # Arrange — surface_tools names a tool the trigger never granted
        trigger = ["A", "B"]
        surface = ["A", "Z"]

        # Act
        result = narrow_allowed_tools(trigger, surface)

        # Assert — Z is never added; narrowing can only remove
        assert result == ["A"]
        assert "Z" not in result

    def test_none_surface_tools_is_a_no_op(self):
        # Arrange
        trigger = ["A", "B", "C"]

        # Act
        result = narrow_allowed_tools(trigger, None)

        # Assert — unchanged trigger set
        assert result == trigger

    def test_empty_surface_tools_is_a_no_op(self):
        # Arrange
        trigger = ["A", "B", "C"]

        # Act
        result = narrow_allowed_tools(trigger, [])

        # Assert — falsy empty list also leaves trigger unchanged
        assert result == trigger

    async def test_denied_floor_survives_surface_narrowing_via_pretooluse_hook(self):
        # Arrange — surface_tools NAMES a denied tool ("Bash"); narrowing alone
        # would happily keep it (it's a plain intersection), but the deny floor
        # is enforced independently downstream by make_pretooluse_hook, so the
        # hook must still deny it.
        trigger = ["Bash", "mcp__controls__channel_read"]
        surface = ["Bash", "mcp__controls__channel_read"]
        narrowed = narrow_allowed_tools(trigger, surface)
        assert "Bash" in narrowed  # narrowing itself cannot know about the floor

        hook = make_pretooluse_hook(narrowed, SURFACES, DENIED)

        # Act
        result = await hook(_main_input("Bash"), "t1", None)

        # Assert — still denied: the denylist is checked first, independent of
        # whatever narrow_allowed_tools computed.
        assert _decision(result) == "deny"

    async def test_surface_narrowing_removes_a_tool_the_hook_would_otherwise_allow(self):
        # Arrange — without narrowing, the trigger tool is allowed on the main
        # thread; with surface_tools narrowing it away, the hook denies it.
        trigger = TRIGGER_TOOLS  # ["mcp__controls__channel_read", ...]
        surface = ["mcp__ariel__keyword_search"]  # keeps only the second tool
        narrowed = narrow_allowed_tools(trigger, surface)

        hook = make_pretooluse_hook(narrowed, SURFACES, DENIED)

        # Act
        result = await hook(_main_input("mcp__controls__channel_read"), "t1", None)

        # Assert — narrowed away, so now denied even though it's in TRIGGER_TOOLS
        assert _decision(result) == "deny"


class TestServerDenylistFloor:
    """The worker's own ``DISPATCH_DENIED_TOOLS`` floor, and the hook built from it.

    ``tool_policy`` takes the denylist as an argument; the list itself lives in
    ``osprey.agent_runner.tool_names``. These tests pin the two together for the dispatcher's
    firing tool, which must be unreachable from a dispatch job.
    """

    def test_the_dispatchers_firing_tool_is_on_the_server_denylist(self):
        # Arrange / Act
        from osprey.agent_runner import tool_names
        from osprey.mcp_server.dispatch_worker import dispatch_api

        # Assert — a job cannot fire jobs, whatever a trigger asks for
        assert "mcp__event_dispatcher__manual_fire" in tool_names.DISPATCH_DENIED_TOOLS
        assert dispatch_api._is_denied("mcp__event_dispatcher__manual_fire") is True

    async def test_the_hook_denies_the_firing_tool_under_the_server_denylist(self):
        # Arrange — the trigger names it and the run is built with the server's
        # own denylist, which is how a real dispatch run is wired.
        from osprey.agent_runner import tool_names

        tool = "mcp__event_dispatcher__manual_fire"
        hook = make_pretooluse_hook([tool], SURFACES, sorted(tool_names.DISPATCH_DENIED_TOOLS))

        # Act
        result = await hook(_main_input(tool), "t1", None)

        # Assert
        assert _decision(result) == "deny"


class TestRenderedAppToolDenylist:
    """The worker's job denylist: the floor plus the render's ``dispatch.denied_app_tools``.

    The list is read from the rendered config the worker was pointed at. Every case
    writes its own render and points ``CONFIG_FILE`` at it, so none depends on the
    host, and checks the result through the hook a real job is built with.
    """

    JOB_TOOLS = ["mcp__knob__set", "mcp__knob__get", "mcp__controls__channel_read"]

    @staticmethod
    def _render(tmp_path, monkeypatch, text):
        config = tmp_path / "build" / "config.yml"
        config.parent.mkdir(parents=True)
        config.write_text(text)
        monkeypatch.setenv("CONFIG_FILE", str(config))

    async def _decisions(self):
        from osprey.mcp_server.dispatch_worker import dispatch_api

        hook = make_pretooluse_hook(
            self.JOB_TOOLS, SURFACES, dispatch_api._denied_tools(self.JOB_TOOLS)
        )
        return {
            tool: _decision(await hook(_main_input(tool), "t1", None)) for tool in self.JOB_TOOLS
        }

    async def test_a_listed_tool_is_denied_and_another_tool_of_its_app_is_not(
        self, tmp_path, monkeypatch
    ):
        # Arrange
        self._render(
            tmp_path, monkeypatch, "dispatch:\n  denied_app_tools:\n    - mcp__knob__set\n"
        )

        # Act
        decisions = await self._decisions()

        # Assert — exactly the listed name, nothing else of the same app
        assert decisions == {
            "mcp__knob__set": "deny",
            "mcp__knob__get": None,
            "mcp__controls__channel_read": None,
        }

    async def test_a_missing_list_denies_every_tool_outside_the_framework_servers(
        self, tmp_path, monkeypatch
    ):
        # Arrange — a render whose config carries no dispatch section at all
        self._render(tmp_path, monkeypatch, "control_system:\n  type: mock\n")

        # Act
        decisions = await self._decisions()

        # Assert
        assert decisions["mcp__knob__set"] == "deny"
        assert decisions["mcp__knob__get"] == "deny"
        assert decisions["mcp__controls__channel_read"] is None

    @pytest.mark.parametrize(
        "text",
        [
            "dispatch: [unclosed\n",
            "dispatch: 3\n",
            "dispatch:\n  denied_app_tools: mcp__knob__set\n",
            "dispatch:\n  denied_app_tools:\n    - 7\n",
            "dispatch:\n  denied_app_tools:\n    - mcp__knob__*\n",
        ],
        ids=["unparseable", "not-a-mapping", "not-a-list", "not-a-name", "a-wildcard"],
    )
    async def test_an_unreadable_list_fails_closed_the_same_way(self, tmp_path, monkeypatch, text):
        # Arrange
        self._render(tmp_path, monkeypatch, text)

        # Act
        decisions = await self._decisions()

        # Assert
        assert decisions["mcp__knob__set"] == "deny"
        assert decisions["mcp__controls__channel_read"] is None

    def test_failing_closed_needs_no_render_at_all(self, tmp_path, monkeypatch):
        # Arrange — the worker was pointed at a config that does not exist
        from osprey.mcp_server.dispatch_worker import dispatch_api

        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "build" / "config.yml"))

        # Act / Assert
        assert dispatch_api._is_denied("mcp__knob__set") is True
        assert dispatch_api._is_denied("mcp__controls__channel_read") is False

    def test_failing_closed_denies_by_the_name_the_agent_cli_forms(self):
        # Arrange — server segments as they arrive in tool names, already sanitized
        from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS, dispatch_denied_tools

        # Act
        denied = dispatch_denied_tools(None, ["my_srv", "channel-finder", "controls"])

        # Assert — one prefix entry per server outside the framework's own
        assert denied - DISPATCH_DENIED_TOOLS == {"mcp__my_srv__*"}

    def test_a_present_list_extends_the_floor_and_never_replaces_it(self):
        # Arrange
        from osprey.agent_runner.tool_names import DISPATCH_DENIED_TOOLS, dispatch_denied_tools

        # Act
        denied = dispatch_denied_tools(["mcp__knob__set"], ["knob"])

        # Assert
        assert denied == DISPATCH_DENIED_TOOLS | {"mcp__knob__set"}
        assert dispatch_denied_tools([], ["knob"]) == DISPATCH_DENIED_TOOLS
