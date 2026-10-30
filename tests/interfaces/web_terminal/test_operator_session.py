"""Tests for operator session management."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from claude_agent_sdk import (
    ClaudeAgentOptions,
    CLIConnectionError,
    PermissionResultAllow,
    PermissionResultDeny,
    PermissionUpdate,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from claude_agent_sdk.types import PermissionRuleValue

from osprey.agent_runner import (
    ApiErrorEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    ToolUseEvent,
)
from osprey.agent_runner.clean_env import build_clean_env
from osprey.agent_runner.write_tools import read_only_disallowed_tools
from osprey.interfaces.web_terminal.chat_session_pool import (
    UNREAD_IDLE_FACTOR,
    ChatCapacityError,
    ChatSessionPool,
)
from osprey.interfaces.web_terminal.operator_session import (
    _NOT_ARMED_MESSAGE,
    OperatorRegistry,
    OperatorSession,
    TurnInProgressError,
    _event_to_wire,
    _is_side_effect_tool,
    _read_render,
    _standing_rules,
    format_tool_name,
    validate_project_directory,
)
from osprey.interfaces.web_terminal.process_tree import ProcessTableUnreadable
from osprey_connectors.refusals import refusal_phrase, refusal_text
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    state_dir_under,
    write_control_context,
)
from tests._agent_cli import STUB_AGENT_CLI
from tests.interfaces.web_terminal._fakes import (
    PRESET_SYSTEM_PROMPT,
    ArmedChat,
    armed_chat,
    assistant_message,
    collect_turn,
    mock_sdk_client,
    result_message,
    sdk_seam,
    start_turn,
    system_message,
    unnumbered,
    user_message,
)

_OS = "osprey.interfaces.web_terminal.operator_session."

# ---------------------------------------------------------------------------
# format_tool_name
# ---------------------------------------------------------------------------


class TestFormatToolName:
    @pytest.mark.parametrize(
        ("raw", "shown"),
        [
            ("mcp__osprey__channel_read", "Channel Read"),
            ("Read", "Read"),
            ("file_search", "File Search"),
            # Framework servers with underscores in the name strip cleanly too.
            ("mcp__osprey_workspace__submit_response", "Submit Response"),
            ("mcp__osprey_facility_knowledge__resolve_channel", "Resolve Channel"),
            # A facility-declared server name OSPREY never saw at authoring time.
            ("mcp__als_custom_srv__do_thing", "Do Thing"),
        ],
    )
    def test_formats_the_tool_name(self, raw, shown):
        assert format_tool_name(raw) == shown


# ---------------------------------------------------------------------------
# _event_to_wire
# ---------------------------------------------------------------------------


def _tool_use(name: str, tool_use_id: str, tool_input: dict) -> ToolUseEvent:
    return ToolUseEvent(
        tool_use_id=tool_use_id, name=name, input=tool_input, parent_tool_use_id=None
    )


def _result(
    *,
    is_error: bool = False,
    total_cost_usd: float | None = 0.01,
    duration_ms: int = 1200,
    num_turns: int = 1,
) -> ResultEvent:
    return ResultEvent(
        subtype="success",
        is_error=is_error,
        num_turns=num_turns,
        duration_ms=duration_ms,
        session_id="sdk-session",
        total_cost_usd=total_cost_usd,
        usage=None,
        result=None,
        api_error_status=None,
    )


class TestEventToWire:
    def test_text_block(self):
        ev = _event_to_wire(TextEvent(text="hello", parent_tool_use_id=None))
        assert ev == {"type": "text", "content": "hello", "parent_tool_use_id": None}

    def test_thinking_block(self):
        ev = _event_to_wire(ThinkingEvent(text="pondering..."))
        assert ev == {"type": "thinking", "content": "pondering...", "parent_tool_use_id": None}

    def test_tool_use_block(self):
        ev = _event_to_wire(_tool_use("mcp__osprey__channel_read", "tu_1", {"channels": ["X"]}))
        assert ev is not None
        assert ev["type"] == "tool_use"
        assert ev["tool_name"] == "Channel Read"
        assert ev["tool_name_raw"] == "mcp__osprey__channel_read"
        assert ev["tool_use_id"] == "tu_1"
        assert ev["input"] == {"channels": ["X"]}

    def test_a_tool_call_starting_reaches_the_stream_named_like_its_row(self):
        """The status line names the call while its input is still being written."""
        ev = _event_to_wire(
            ToolStartEvent(
                name="mcp__osprey__channel_read", tool_use_id="tu_1", parent_tool_use_id=None
            )
        )
        assert ev == {
            "type": "tool_start",
            "tool_name": "Channel Read",
            "tool_name_raw": "mcp__osprey__channel_read",
            "tool_use_id": "tu_1",
            "parent_tool_use_id": None,
        }

    @pytest.mark.parametrize("is_error", [False, True])
    def test_tool_results_reach_the_stream_as_a_preview(self, is_error):
        event = ToolResultEvent(
            tool_use_id="tu_1", content="42.0", is_error=is_error, parent_tool_use_id=None
        )
        assert _event_to_wire(event) == {
            "type": "tool_result",
            "tool_use_id": "tu_1",
            "is_error": is_error,
            "preview": "42.0",
            "truncated": False,
            "refusal": None,
        }

    def test_assistant_error(self):
        ev = _event_to_wire(ApiErrorEvent(error="overloaded"))
        assert ev is not None
        assert ev["type"] == "error"
        assert "API error" in ev["message"]
        # The one error type is_terminal_event lets a turn continue past.
        assert ev["error_type"] == "AssistantMessageError"

    def test_result_message(self):
        ev = _event_to_wire(
            _result(is_error=False, total_cost_usd=0.05, duration_ms=3000, num_turns=2)
        )
        assert ev is not None
        assert ev["type"] == "result"
        assert ev["is_error"] is False
        assert ev["total_cost_usd"] == 0.05
        assert ev["duration_ms"] == 3000
        assert ev["num_turns"] == 2

    def test_frames_keep_their_key_order(self):
        frames = [
            _event_to_wire(ThinkingEvent(text="think")),
            _event_to_wire(TextEvent(text="answer", parent_tool_use_id=None)),
            _event_to_wire(_tool_use("Read", "tu_x", {"file": "a.py"})),
        ]
        assert [list(f) for f in frames if f is not None] == [
            ["type", "content", "parent_tool_use_id"],
            ["type", "content", "parent_tool_use_id"],
            ["type", "tool_name", "tool_name_raw", "tool_use_id", "input", "parent_tool_use_id"],
        ]

    def test_a_compaction_notice_carries_its_trigger_and_size(self):
        ev = _event_to_wire(
            SystemEvent(
                subtype="compact_boundary",
                data={"compact_metadata": {"trigger": "auto", "pre_tokens": 183000}},
            )
        )
        assert ev == {
            "type": "system",
            "subtype": "compact_boundary",
            "trigger": "auto",
            "pre_tokens": 183000,
        }


class TestSystemEventCarriesResumeIdentity:
    """The init message is where the child names the transcript it writes to.

    A resume can be answered with a different id than the one asked for, so the
    id the child reports is the only authoritative one — it has to reach the
    consumer, not stop at the SDK boundary.
    """

    def test_the_init_session_id_reaches_the_system_event(self):
        ev = _event_to_wire(SystemEvent(subtype="init", data={"session_id": "abc-123"}))
        assert ev == {"type": "system", "subtype": "init", "session_id": "abc-123"}

    def test_a_system_message_without_one_carries_no_key(self):
        """Absent, not ``None``: a message that says nothing about identity
        must not read as one reporting a missing id."""
        ev = _event_to_wire(SystemEvent(subtype="compact_boundary", data={}))
        assert "session_id" not in ev
        assert ev == {
            "type": "system",
            "subtype": "compact_boundary",
            "trigger": None,
            "pre_tokens": None,
        }


# ---------------------------------------------------------------------------
# The chat's approval gate: the child's own render, and session allows
# ---------------------------------------------------------------------------


def _session_allow(tool: str, *, kind: str = "addRules") -> list:
    """A permission suggestion the CLI would offer for *tool*: an allow rule, aimed at
    the project's settings. ``_standing_rules`` must narrow it, or refuse it outright."""
    return [
        PermissionUpdate(
            type=kind,
            rules=[PermissionRuleValue(tool_name=tool)],
            behavior="allow",
            destination="localSettings",
        )
    ]


def _write_render(directory: Path, text: str) -> str:
    """Write a ``config.yml`` into *directory*; return the directory's path."""
    (directory / "config.yml").write_text(text, encoding="utf-8")
    return str(directory)


def _write_hook_config(
    project: Path, write_tools: list, approval_prefixes: list | None = None
) -> None:
    """Write the ``write_tools`` list the render's writes-check hook reads, and the
    ``approval_prefixes`` its approval hook reads when given."""
    hooks = project / ".claude" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    config: dict = {"write_tools": write_tools}
    if approval_prefixes is not None:
        config["approval_prefixes"] = approval_prefixes
    (hooks / "hook_config.json").write_text(json.dumps(config), encoding="utf-8")


class TestReadRender:
    """The chat reads the render its child runs in, and fails closed."""

    def test_a_render_whose_tier_allows_writes_allows_them(self, tmp_path):
        project = _write_render(tmp_path, "control_system:\n  writes: allowed\n")
        assert _read_render(project).tier_allows is True

    def test_the_tier_key_decides_over_the_old_leaf(self, tmp_path):
        project = _write_render(
            tmp_path, "control_system:\n  writes: never\n  writes_enabled: true\n"
        )
        assert _read_render(project).tier_allows is False

    @pytest.mark.parametrize(
        "render",
        [
            # D2: the retired leaf alone no longer arms (row tier-absent-old-armed-flat).
            pytest.param("control_system:\n  writes_enabled: true\n", id="retired-leaf-alone"),
            pytest.param("control_system:\n  writes: never\n", id="tier-never"),
            pytest.param("control_system: {writes: allowed\n", id="will-not-load"),
            pytest.param(None, id="missing"),
            pytest.param("project_name: demo\n", id="no-control-section"),
        ],
    )
    def test_writes_stay_refused_unless_the_tier_allows_them(self, tmp_path, render):
        project = str(tmp_path) if render is None else _write_render(tmp_path, render)
        assert _read_render(project).tier_allows is False

    def test_the_child_render_decides_not_the_server_process_config(self, tmp_path, monkeypatch):
        """The server's own working directory may hold an armed config; the child's does not."""
        child = tmp_path / "child"
        child.mkdir()
        project = _write_render(child, "control_system:\n  writes: never\n")
        server = tmp_path / "server"
        server.mkdir()
        _write_render(server, "control_system:\n  writes: allowed\n")
        monkeypatch.chdir(server)
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)
        assert _read_render(project).tier_allows is False

    def test_the_hook_config_is_read_with_the_render(self, tmp_path):
        project = _write_render(tmp_path, "control_system:\n  writes: allowed\n")
        _write_hook_config(tmp_path, ["mcp__controls__channel_write"])
        assert _read_render(project).hook_config == {
            "write_tools": ["mcp__controls__channel_write"]
        }

    def test_an_unreadable_hook_config_is_an_empty_one(self, tmp_path):
        project = _write_render(tmp_path, "control_system:\n  writes: allowed\n")
        hooks = tmp_path / ".claude" / "hooks"
        hooks.mkdir(parents=True)
        (hooks / "hook_config.json").write_text("{not json", encoding="utf-8")
        assert _read_render(project).hook_config == {}

    async def test_start_arms_the_session_from_its_own_render(self, tmp_path):
        _write_render(tmp_path, "control_system:\n  writes: allowed\n")
        _write_hook_config(tmp_path, ["mcp__controls__channel_write"])
        session = OperatorSession(cwd=str(tmp_path))
        with sdk_seam():
            await session.start()
        try:
            assert session._render.tier_allows is True
            assert "mcp__controls__channel_write" in session._side_effect_tools
        finally:
            await session.stop()

    async def test_a_tool_the_approval_hook_asks_about_gets_no_session_allow(self, tmp_path):
        """The hook asks on every call under its prefixes, so a standing allow for one
        would either do nothing or let the hook's ask be skipped: none is offered."""
        _write_render(tmp_path, "control_system:\n  writes: allowed\n")
        _write_hook_config(tmp_path, ["mcp__controls__channel_write"], ["mcp__controls__"])
        session = OperatorSession(cwd=str(tmp_path))
        with sdk_seam():
            await session.start()
        try:
            asked = "mcp__controls__channel_read"
            assert _standing_rules(asked, _session_allow(asked), session._side_effect_tools) == []
            other = "mcp__weather__forecast"
            kept = _standing_rules(other, _session_allow(other), session._side_effect_tools)
            assert [rule.destination for rule in kept] == ["session"]
        finally:
            await session.stop()


#: A switch-capable render whose tier allows writes, as YAML.
_TWO_TARGETS_RENDER = (
    "control_system:\n"
    "  type: epics\n"
    "  writes: {writes}\n"
    "  connector:\n"
    "    epics: {{prefix: 'RING:'}}\n"
    "    virtual_accelerator: {{prefix: 'VA:'}}\n"
)

#: The render's write list, a facility-custom server among it.
_CARD_HOOK_CONFIG = {
    "server_prefixes": ["mcp__controls__", "mcp__python__", "mcp__magnets__"],
    "write_tools": ["mcp__controls__channel_write", "mcp__python__execute", "mcp__magnets__.*"],
}


@contextlib.asynccontextmanager
async def _chat_on(project: Path, *, writes: str = "allowed"):
    """A session on the two-target render in *project*, with its hook_config."""
    _write_render(project, _TWO_TARGETS_RENDER.format(writes=writes))
    hooks = project / ".claude" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "hook_config.json").write_text(json.dumps(_CARD_HOOK_CONFIG), encoding="utf-8")
    session = OperatorSession(cwd=str(project))
    with sdk_seam() as captured:
        await session.start()
        try:
            feed, _hello = session.subscribe()
            yield ArmedChat(session, captured[0].can_use_tool, feed)
        finally:
            await session.stop()


class TestCardTarget:
    """C-FRAME: a write's card names the machine it lands on, placed as the hooks place it."""

    @pytest.fixture
    def root(self, tmp_path, monkeypatch) -> Path:
        root = tmp_path / "agent_data"
        root.mkdir()
        isolate_verdict_environment(monkeypatch, root)
        return root

    def _publish(self, root: Path, kind: str) -> None:
        """This process as a live controls server publishing *kind* for ``standin``."""
        directory = state_dir_under(root)
        directory.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        payload = {"server_pid": pid, "targets": {"standin": {"machine_kind": kind}}}
        (directory / f"server_{pid}.json").write_text(json.dumps(payload), encoding="utf-8")

    async def _card(self, project: Path, tool: str, tool_input: dict) -> dict:
        async with _chat_on(project) as chat:
            _answer, card = await chat.ask(tool, tool_input)
        return card

    @pytest.mark.parametrize(
        ("session_machine", "published", "expected"),
        [
            pytest.param(
                "live",
                None,
                {
                    "key": "live",
                    "machine_kind": "real",
                    "kind_words": "Real machine — writes move hardware",
                },
                id="real",
            ),
            pytest.param(
                "standin",
                "rehearsal",
                {
                    "key": "standin",
                    "machine_kind": "rehearsal",
                    "kind_words": "Rehearsal — nothing moves",
                },
                id="rehearsal",
            ),
            pytest.param(
                "standin",
                None,
                {
                    "key": "standin",
                    "machine_kind": "real",
                    "kind_words": "Real machine — writes move hardware",
                },
                id="standin-not-vouched-for",
            ),
            pytest.param(
                "va",
                None,
                {
                    "key": "va",
                    "machine_kind": "simulator",
                    "kind_words": "Simulator — nothing moves",
                },
                id="simulator",
            ),
        ],
    )
    async def test_a_write_card_names_its_machine(
        self, tmp_path, root, session_machine, published, expected
    ):
        write_control_context(root, target=session_machine, generation=2)
        if published:
            self._publish(root, published)

        card = await self._card(tmp_path, "mcp__controls__channel_write", {"value": 1})

        assert card["target"] == expected

    async def test_a_facility_write_names_the_real_machine_whatever_the_session_is_on(
        self, tmp_path, root
    ):
        write_control_context(root, target="va", generation=2)

        card = await self._card(tmp_path, "mcp__magnets__set_current", {})

        assert card["target"]["key"] == "live"
        assert card["target"]["machine_kind"] == "real"

    async def test_an_unplaceable_write_still_asks_and_names_no_machine(self, tmp_path, root):
        directory = state_dir_under(root)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "control_context.json").write_text('{"schema": ', encoding="utf-8")

        card = await self._card(tmp_path, "mcp__controls__channel_write", {"value": 1})

        assert card["type"] == "permission_request"
        assert card["target"] is None

    @pytest.mark.parametrize(
        ("tool", "tool_input"),
        [
            ("mcp__controls__channel_read", {}),
            ("mcp__python__execute", {"execution_mode": "readonly"}),
        ],
        ids=["a-read", "a-readonly-execute"],
    )
    async def test_a_call_that_writes_nothing_names_no_machine(
        self, tmp_path, root, tool, tool_input
    ):
        write_control_context(root, target="live", generation=2)

        card = await self._card(tmp_path, tool, tool_input)

        assert card["target"] is None

    @pytest.mark.usefixtures("root")
    async def test_question_and_plan_cards_name_no_machine(self, tmp_path):
        async with _chat_on(tmp_path) as chat:
            _a, question = await chat.ask("AskUserQuestion", {"questions": []})
            _b, plan = await chat.ask("ExitPlanMode", {"plan": "the plan"})

        assert question["target"] is None
        assert plan["target"] is None


class TestTierVerdictDeny:
    """C-FRAME: on a render whose tier allows no writes, a write is refused with its reason."""

    @pytest.fixture(autouse=True)
    def root(self, tmp_path, monkeypatch) -> Path:
        root = tmp_path / "agent_data"
        root.mkdir()
        isolate_verdict_environment(monkeypatch, root)
        write_control_context(root, target="live", generation=2)
        return root

    async def test_an_app_render_deny_carries_the_tier_refusal(self, tmp_path):
        async with _chat_on(tmp_path, writes="never") as chat:
            result = await chat.can_use_tool(
                "mcp__controls__channel_write", {"value": 1}, SimpleNamespace()
            )
            shown = chat.feed.qsize()

        assert isinstance(result, PermissionResultDeny)
        assert result.message == refusal_text("tier_never", "real")
        assert shown == 0, "a refused write was put in front of the operator"
        # The deny comes back to the stream as the call's error result, refusal and all.
        frame = _event_to_wire(
            ToolResultEvent(
                tool_use_id="tu_1", content=result.message, is_error=True, parent_tool_use_id=None
            )
        )
        assert frame["refusal"] == {
            "reason": "tier_never",
            "phrase": refusal_phrase("tier_never", "real"),
        }
        assert "[osprey-refusal:" not in frame["preview"]

    async def test_an_unplaceable_write_is_refused_without_machine_words(self, tmp_path, root):
        (state_dir_under(root) / "control_context.json").write_text("{", encoding="utf-8")
        async with _chat_on(tmp_path, writes="never") as chat:
            result = await chat.can_use_tool(
                "mcp__controls__channel_write", {"value": 1}, SimpleNamespace()
            )

        assert result.message == refusal_text("tier_never")

    async def test_an_ask_that_is_not_a_write_keeps_todays_deny(self, tmp_path):
        async with _chat_on(tmp_path, writes="never") as chat:
            result = await chat.can_use_tool("mcp__weather__forecast", {}, SimpleNamespace())

        assert isinstance(result, PermissionResultDeny)
        assert result.message == _NOT_ARMED_MESSAGE
        assert "[osprey-refusal:" not in result.message


class TestStandingRules:
    """A session allow is never offered for a tool on the side-effect surface.

    The surface is ``read_only_disallowed_tools``, the list the headless read-only run
    refuses, so whatever the CLI suggests, a tool that can act is approved per call.
    """

    def test_every_entry_of_the_side_effect_surface_is_refused_a_session_allow(self, tmp_path):
        """The render's own write list is read, so each entry is tried by a tool it covers."""
        _write_hook_config(
            tmp_path, ["mcp__phoebus__phoebus_drive", "mcp__facility__.*", "mcp__controls__x"]
        )
        surface = read_only_disallowed_tools(tmp_path)
        assert "mcp__phoebus__phoebus_drive" in surface
        for entry in surface:
            if entry.endswith(".*"):
                tool = entry[: -len(".*")] + "probe"
            elif entry.endswith("*"):
                tool = entry[:-1] + "probe"
            else:
                tool = entry
            assert _standing_rules(tool, _session_allow(tool), surface) == [], tool

    def test_a_declaring_servers_write_is_on_the_surface_and_its_read_is_not(self, tmp_path):
        """A server that declares ``writes:`` renders one exact write-list entry per write
        tool, so only that tool is approved per call; its reads keep a session allow."""
        (tmp_path / "config.yml").write_text(
            "claude_code:\n"
            "  servers:\n"
            "    plc:\n"
            "      command: node\n"
            "      writes: [set_value]\n"
            "      permissions: {allow: [get_value, set_value]}\n",
            encoding="utf-8",
        )
        _write_hook_config(tmp_path, ["mcp__controls__channel_write", "mcp__plc__set_value"])
        surface = read_only_disallowed_tools(tmp_path)
        write, read = "mcp__plc__set_value", "mcp__plc__get_value"

        assert _is_side_effect_tool(write, surface)
        assert not _is_side_effect_tool(read, surface)
        assert _standing_rules(write, _session_allow(write), surface) == []
        rules = _standing_rules(read, _session_allow(read), surface)
        assert [r.tool_name for rule in rules for r in rule.rules] == [read]

    def test_a_whole_server_facility_server_is_covered_whole(self, tmp_path):
        """A list-less server's whole-server matcher covers its reads too."""
        _write_hook_config(tmp_path, ["mcp__magnets__.*"])
        surface = read_only_disallowed_tools(tmp_path)

        assert _is_side_effect_tool("mcp__magnets__read_current", surface)
        assert not _is_side_effect_tool("mcp__magnetsx__read_current", surface)

    def test_control_target_set_is_never_offered_a_session_allow(self, tmp_path):
        """Regression pin: it moves the whole session to another machine. It is approval-gated
        but not writes-check gated, so only the side-effect surface keeps it per call."""
        surface = read_only_disallowed_tools(tmp_path)
        assert "mcp__controls__control_target_set" in surface
        tool = "mcp__controls__control_target_set"
        assert _standing_rules(tool, _session_allow(tool), surface) == []

    def test_phoebus_drive_is_never_offered_a_session_allow(self, tmp_path):
        """Regression pin: the phoebus family was outside the hand-written prefixes."""
        surface = read_only_disallowed_tools(tmp_path)
        tool = "mcp__phoebus__phoebus_drive"
        assert _standing_rules(tool, _session_allow(tool), surface) == []

    def test_a_tool_off_the_surface_keeps_its_session_allow(self, tmp_path):
        surface = read_only_disallowed_tools(tmp_path)
        tool = "mcp__osprey_workspace__channel_find"
        rules = _standing_rules(tool, _session_allow(tool), surface)
        assert [rule.destination for rule in rules] == ["session"]
        assert [r.tool_name for rule in rules for r in rule.rules] == [tool]

    def test_a_server_level_entry_covers_every_tool_of_that_server(self):
        """The disallow engine reads ``mcp__<server>`` as the whole server."""
        surface = ["mcp__facility"]
        assert (
            _standing_rules("mcp__facility__probe", _session_allow("mcp__facility__probe"), surface)
            == []
        )
        rules = _standing_rules(
            "mcp__facilityx__probe", _session_allow("mcp__facilityx__probe"), surface
        )
        assert [rule.destination for rule in rules] == ["session"]

    def test_a_malformed_entry_covers_nothing_and_does_not_raise(self):
        tool = "mcp__osprey_workspace__channel_find"
        rules = _standing_rules(tool, _session_allow(tool), [7, None])
        assert [rule.destination for rule in rules] == ["session"]

    def test_a_suggestion_wider_than_the_asked_tool_is_not_offered(self):
        """A server-wide rule, or one for another tool, would let a session allow cover
        calls the operator was never asked about; neither is kept."""
        tool = "mcp__controls__channel_read"
        assert _standing_rules(tool, _session_allow("mcp__controls"), []) == []
        assert _standing_rules(tool, _session_allow("mcp__controls__channel_write"), []) == []
        mixed = [
            PermissionUpdate(
                type="addRules",
                rules=[
                    PermissionRuleValue(tool_name="mcp__controls"),
                    PermissionRuleValue(tool_name=tool),
                ],
                behavior="allow",
                destination="projectSettings",
            )
        ]
        (kept,) = _standing_rules(tool, mixed, [])
        assert [r.tool_name for r in kept.rules] == [tool]
        assert kept.destination == "session"

    def test_a_replace_rules_suggestion_is_not_offered(self):
        """``replaceRules`` would swap the session's whole rule set, not add one allow."""
        tool = "mcp__controls__channel_read"
        assert _standing_rules(tool, _session_allow(tool, kind="replaceRules"), []) == []

    async def test_an_unreadable_surface_still_starts_and_offers_no_session_allow(
        self, tmp_path, monkeypatch
    ):
        def unreadable(_project_dir):
            raise RuntimeError("config.yml is not valid YAML")

        monkeypatch.setattr(
            "osprey.interfaces.web_terminal.operator_session.read_only_disallowed_tools",
            unreadable,
        )
        _write_render(tmp_path, "control_system:\n  writes: allowed\n")
        session = OperatorSession(cwd=str(tmp_path))
        with sdk_seam():
            await session.start()
        try:
            assert session._side_effect_tools == ("*",)
            tool = "mcp__controls__channel_read"
            rules = _standing_rules(tool, _session_allow(tool), session._side_effect_tools)
            assert rules == []
        finally:
            await session.stop()


# ---------------------------------------------------------------------------
# A card nobody answers expires as a decline
# ---------------------------------------------------------------------------


class TestAskExpiry:
    """An approval, question or plan card nobody answers is declined after ASK_EXPIRY_S."""

    @pytest.fixture(autouse=True)
    def _short_expiry(self, monkeypatch):
        """Expire an unanswered card in 10 ms instead of 30 minutes."""
        monkeypatch.setattr(_OS + "ASK_EXPIRY_S", 0.01)

    @pytest.mark.parametrize(
        ("tool", "tool_input", "told"),
        [
            pytest.param(
                "mcp__python__execute", {"code": "1"}, "No answer from the operator", id="tool"
            ),
            pytest.param("ExitPlanMode", {"plan": "the plan"}, "keep planning", id="plan"),
            pytest.param("AskUserQuestion", {"questions": []}, "not answered", id="question"),
        ],
    )
    async def test_an_unanswered_card_is_declined_and_settles_as_expired(
        self, tmp_path, tool, tool_input, told
    ):
        """The agent is told what the card's kind calls for, every tab settles the card as
        expired rather than as the operator's "No", a tab that opens later is shown none, and
        a late answer finds nothing waiting."""
        async with armed_chat(tmp_path) as chat:
            answer, card = await chat.ask(tool, tool_input)
            result = await asyncio.wait_for(answer, timeout=2.0)
            resolved = chat.feed.get_nowait()
            _feed, hello = chat.session.subscribe()
            late = await chat.session.resolve_permission(card["request_id"], allow=True)

        assert isinstance(result, PermissionResultDeny)
        assert told in result.message
        assert result.interrupt is False
        assert unnumbered(resolved) == {
            "type": "permission_resolved",
            "request_id": card["request_id"],
            "allow": False,
            "for_session": False,
            "answers": None,
            "expired": True,
        }
        assert hello["pending"] == []
        assert late is False

    async def test_an_answer_before_the_deadline_is_the_answer(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_OS + "ASK_EXPIRY_S", 5.0)
        async with armed_chat(tmp_path) as chat:
            answer, card = await chat.ask()
            assert await chat.session.resolve_permission(card["request_id"], allow=True)
            result = await asyncio.wait_for(answer, timeout=2.0)
            resolved = chat.feed.get_nowait()

        assert isinstance(result, PermissionResultAllow)
        assert resolved["allow"] is True
        assert resolved["expired"] is False

    async def test_a_stop_before_the_deadline_declines_with_an_interrupt_not_an_expiry(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(_OS + "ASK_EXPIRY_S", 5.0)
        async with armed_chat(tmp_path) as chat:
            answer, _card = await chat.ask()
            await chat.session.interrupt()
            result = await asyncio.wait_for(answer, timeout=2.0)
            _feed, hello = chat.session.subscribe()
            later = [chat.feed.get_nowait() for _ in range(chat.feed.qsize())]

        assert isinstance(result, PermissionResultDeny)
        assert result.interrupt is True
        assert hello["pending"] == []
        assert not [frame for frame in later if frame.get("expired")]


# ---------------------------------------------------------------------------
# build_clean_env
# ---------------------------------------------------------------------------


class TestBuildCleanEnv:
    def test_strips_claudecode_vars(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE_SESSION", "123")
        monkeypatch.setenv("CLAUDE_CODE_BETA", "1")
        monkeypatch.setenv("HOME", "/Users/test")

        env = build_clean_env()
        assert "CLAUDECODE_SESSION" not in env
        assert "CLAUDE_CODE_BETA" not in env
        assert env.get("HOME") == "/Users/test"

    def test_strips_api_key_when_auth_token(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok_abc")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-123")

        env = build_clean_env()
        assert "ANTHROPIC_API_KEY" not in env
        assert env["ANTHROPIC_AUTH_TOKEN"] == "tok_abc"

    def test_keeps_api_key_without_auth_token(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-123")

        env = build_clean_env()
        assert env["ANTHROPIC_API_KEY"] == "sk-123"

    def test_augments_path_with_user_bin_dirs(self, monkeypatch, tmp_path):
        """PATH includes user-local bin dirs not already on PATH."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        monkeypatch.setattr("osprey.utils.shell_resolver._user_bin_candidates", lambda: [bin_dir])
        monkeypatch.setenv("PATH", "/usr/bin")

        env = build_clean_env()
        assert str(bin_dir) in env["PATH"]


# ---------------------------------------------------------------------------
# OperatorSession
# ---------------------------------------------------------------------------


class TestOperatorSession:
    @pytest.mark.asyncio
    async def test_start_passes_setting_sources(self):
        """Verify SDK receives setting_sources=['project'] for config auto-discovery."""
        session = OperatorSession(cwd="/tmp")

        with sdk_seam() as captured:
            await session.start()

        assert captured[0].setting_sources == ["project"]
        await session.stop()

    @pytest.mark.asyncio
    async def test_start_requires_sdk(self):
        with patch(_OS + "HAS_SDK", False):
            session = OperatorSession(cwd="/tmp")
            with pytest.raises(RuntimeError, match="not installed"):
                await session.start()

    @pytest.mark.asyncio
    async def test_is_active_lifecycle(self):
        session = OperatorSession(cwd="/tmp")
        assert not session.is_active

        with sdk_seam():
            await session.start()
            assert session.is_active

            await session.stop()
            assert not session.is_active

    @pytest.mark.asyncio
    async def test_send_prompt_queues_events(self):
        """A prompt's response reaches the turn's consumer as frames, in stream order."""
        session = OperatorSession(cwd="/tmp")
        client = _ScriptedClient([assistant_message([TextBlock("hello")]), result_message()])

        with sdk_seam(client):
            await session.start()
            frames = [unnumbered(f) for f in await collect_turn(session, "test")]
            await session.stop()

        assert [f["type"] for f in frames] == ["turn_start", "text", "result"]
        assert frames[1] == {"type": "text", "content": "hello", "parent_tool_use_id": None}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt", "rows"),
        [
            pytest.param("/diagnose SR03", [("diagnose", "SR03")], id="a-skill"),
            pytest.param("/check SR03", [], id="a-command-file"),
            pytest.param("diagnose SR03", [], id="plain-words"),
        ],
    )
    async def test_a_skill_the_operator_runs_opens_its_turn_as_a_skill_row(
        self, tmp_path, prompt, rows
    ):
        """The CLI loads a typed skill with no tool call, so the turn draws its row itself."""
        (tmp_path / ".claude" / "skills" / "diagnose").mkdir(parents=True)
        (tmp_path / ".claude" / "skills" / "diagnose" / "SKILL.md").write_text("Body\n")
        (tmp_path / ".claude" / "commands").mkdir()
        (tmp_path / ".claude" / "commands" / "check.md").write_text("Check\n")
        session = OperatorSession(cwd=str(tmp_path))
        client = _ScriptedClient([assistant_message([TextBlock("on it")]), result_message()])

        with sdk_seam(client):
            await session.start()
            frames = [unnumbered(f) for f in await collect_turn(session, prompt)]
            await session.stop()

        calls = [f for f in frames if f["type"] == "tool_use"]
        assert [(c["input"]["skill"], c["input"].get("args")) for c in calls] == rows
        if rows:
            assert [f["type"] for f in frames][:3] == ["turn_start", "tool_use", "tool_result"]
            assert frames[2]["tool_use_id"] == calls[0]["tool_use_id"]


# ---------------------------------------------------------------------------
# OperatorSession start identity: resume a transcript, or open one
# ---------------------------------------------------------------------------


class TestOperatorSessionResumeOptions:
    """The shapes a chat child can be started in.

    Either it continues a transcript that already exists (``resume``) or it
    opens one under the session key (``session_id``). Never both: a resume
    names the transcript, and a session id alongside it would ask the SDK to
    write one conversation under two identities. A pool key outside the
    session-key grammar names neither, because the CLI would refuse it.
    """

    KEY = "11111111-2222-3333-4444-555555555555"

    @pytest.mark.asyncio
    async def test_a_resume_names_the_transcript_and_nothing_else(self):
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=self.KEY)

        with sdk_seam() as captured:
            await session.start(resume_id="transcript-9")

        (options,) = captured
        assert options.resume == "transcript-9"
        assert options.session_id is None
        # Everything else about the launch is unchanged by the resume.
        assert options.setting_sources == ["project"]
        assert options.env["OSPREY_WEB_UX"] == "simple"

    @pytest.mark.asyncio
    async def test_without_a_resume_the_child_opens_one_under_the_session_key(self):
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=self.KEY)

        with sdk_seam() as captured:
            await session.start()

        (options,) = captured
        assert options.session_id == self.KEY
        assert options.resume is None

    @pytest.mark.asyncio
    async def test_a_pool_key_the_cli_would_reject_names_no_session_id(self):
        """An embedder's own chat key never reaches the CLI as an identity.

        ``POST /api/chat`` accepts any string as ``chat_id`` and pools the chat
        under it, so the key can be ``"e2e"`` or ``"user-42-chat-3"``.
        ``--session-id`` takes a canonical UUID and the CLI exits non-zero on
        anything else, so naming the key there would fail the child on its
        first prompt. It mints its own id instead.
        """
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key="e2e")

        with sdk_seam() as captured:
            await session.start()

        (options,) = captured
        # None, not the key: the SDK omits the flag entirely for a falsy id.
        assert options.session_id is None
        assert options.resume is None
        # The key still names the session everywhere it is ours to spend.
        assert options.env["OSPREY_TELEMETRY_SESSION_ID"] == "e2e"

    @pytest.mark.asyncio
    async def test_the_telemetry_id_is_the_session_key_in_both_shapes(self):
        """Never a fresh uuid: an id re-drawn per start would split one
        conversation's traces across as many ids as it had surfaces."""

        with sdk_seam() as fresh:
            await OperatorSession(
                cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=self.KEY
            ).start()
        with sdk_seam() as resumed:
            await OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=self.KEY).start(
                resume_id="transcript-9"
            )

        assert fresh[0].env["OSPREY_TELEMETRY_SESSION_ID"] == self.KEY
        assert resumed[0].env["OSPREY_TELEMETRY_SESSION_ID"] == self.KEY

    @pytest.mark.asyncio
    async def test_a_session_given_no_key_holds_one_stable_across_starts(self):
        """The key belongs to the session object, not to a single start."""
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"})

        with sdk_seam() as first:
            await session.start()
        with sdk_seam() as second:
            await session.start()

        minted = first[0].env["OSPREY_TELEMETRY_SESSION_ID"]
        assert minted
        assert second[0].env["OSPREY_TELEMETRY_SESSION_ID"] == minted
        assert second[0].session_id == minted

    @pytest.mark.asyncio
    async def test_two_keyless_sessions_do_not_share_an_identity(self):
        with sdk_seam() as first:
            await OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}).start()
        with sdk_seam() as second:
            await OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}).start()

        assert first[0].session_id != second[0].session_id


# ---------------------------------------------------------------------------
# OperatorSession per-turn epoch guard
# ---------------------------------------------------------------------------


class TestTurnGuardEpoch:
    def test_acquire_mints_incrementing_token_and_sets_in_flight(self):
        session = OperatorSession(cwd="/tmp")
        token = session.acquire_turn()
        assert token == 1
        assert session.in_flight is True

    def test_acquire_while_active_raises(self):
        session = OperatorSession(cwd="/tmp")
        session.acquire_turn()
        with pytest.raises(TurnInProgressError):
            session.acquire_turn()

    def test_release_clears_and_returns_true(self):
        session = OperatorSession(cwd="/tmp")
        token = session.acquire_turn()
        assert session.release_turn(token) is True
        assert session.in_flight is False

    def test_reacquire_after_release_mints_next_epoch(self):
        session = OperatorSession(cwd="/tmp")
        t1 = session.acquire_turn()
        session.release_turn(t1)
        t2 = session.acquire_turn()
        assert t2 == 2
        assert t2 != t1
        assert session.in_flight is True

    def test_double_release_is_idempotent(self):
        session = OperatorSession(cwd="/tmp")
        token = session.acquire_turn()
        assert session.release_turn(token) is True
        # Second release of the same token does nothing.
        assert session.release_turn(token) is False
        assert session.in_flight is False

    def test_stale_token_release_is_noop(self):
        """A release with a token from an already-ended turn must not clear
        the turn a later acquire started."""
        session = OperatorSession(cwd="/tmp")
        t1 = session.acquire_turn()
        session.release_turn(t1)
        t2 = session.acquire_turn()

        # Late release from the first turn — must NOT clear t2's turn.
        assert session.release_turn(t1) is False
        assert session.in_flight is True

        # The current owner can still release cleanly.
        assert session.release_turn(t2) is True
        assert session.in_flight is False

    def test_release_when_idle_is_noop(self):
        session = OperatorSession(cwd="/tmp")
        assert session.release_turn(1) is False
        assert session.in_flight is False


# ---------------------------------------------------------------------------
# Tests of OperatorSession.cancel(), spawn_quiesce() and last_activity
# ---------------------------------------------------------------------------


class FakeStreamClient:
    """Controllable fake ``ClaudeSDKClient`` for cancel/quiesce tests.

    Each prompt sent gets a response: partial text, then, once interrupted, a
    terminal result. ``hang`` keeps the response from ever ending on its own.
    ``interrupt`` is a coroutine so that a caller which forgets to ``await`` it
    (the historical bug) never runs its body — letting a test assert the await
    happened via ``interrupt_calls``.
    """

    def __init__(self, *, hang: bool = False) -> None:
        self.hang = hang
        self.interrupt_calls = 0
        self.query_calls = 0
        self._interrupted = asyncio.Event()
        # As the SDK's, a response is only ever read for a prompt that was sent.
        self._prompts: asyncio.Queue[str] = asyncio.Queue()
        # Set once the reader has yielded its first (partial) message, so a
        # test can be sure the turn is genuinely in-flight before cancelling.
        self.first_yielded = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.query_calls += 1
        self._prompts.put_nowait(prompt)

    async def interrupt(self):
        self.interrupt_calls += 1
        self._interrupted.set()

    async def receive_response(self):
        await self._prompts.get()
        yield assistant_message([TextBlock("partial")])
        self.first_yielded.set()
        if self.hang:
            # Never terminate on its own — only a hard cancel stops us.
            while True:
                await asyncio.sleep(3600)
        else:
            # Drain toward a terminal message once interrupted.
            await self._interrupted.wait()
            await asyncio.sleep(0.001)
            yield result_message()


@contextlib.asynccontextmanager
async def _started_session(client):
    """Yield a started ``OperatorSession`` wired to ``client``."""
    session = OperatorSession(cwd="/tmp")
    with sdk_seam(client):
        await session.start()
        try:
            yield session
        finally:
            await session.stop()


class _LateWordsClient(FakeStreamClient):
    """Ignores the interrupt, and ends its response only once ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def receive_response(self):
        await self._prompts.get()
        yield assistant_message([TextBlock("partial")])
        self.first_yielded.set()
        await self.release.wait()
        yield assistant_message([TextBlock("late")])
        yield result_message()


class _OneTurnClient(FakeStreamClient):
    """Answers each prompt with one text block and a terminal result."""

    async def receive_response(self):
        await self._prompts.get()
        yield assistant_message([TextBlock("done")])
        yield result_message()


class TestRunTurnReleasesAtTheTerminalEvent:
    """The turn is free the moment its terminal event reaches the consumer.

    A client that sees the final frame may send its next prompt before the
    consumer resumes the generator; that prompt must find the guard released,
    not a ``turn_in_progress`` left over from a turn that has already ended.
    """

    @pytest.mark.asyncio
    async def test_a_follow_up_turn_can_start_as_the_terminal_event_is_seen(self):
        async with _started_session(_OneTurnClient()) as session:
            token = session.acquire_turn()
            turn = session.run_turn("hi", token, timeout_s=5.0)
            follow_up = None
            async for event in turn:
                if event["type"] == "result":
                    follow_up = session.acquire_turn()
                    break
            await turn.aclose()

            assert follow_up is not None
            # The ended turn's own release cannot clear the follow-up's guard.
            assert session.in_flight is True
            assert session.release_turn(follow_up) is True


class TestOperatorSessionCancel:
    @pytest.mark.asyncio
    async def test_idle_cancel_is_noop_no_interrupt(self):
        """No turn in flight → short-circuit, no hang."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            assert session.is_busy is False
            await asyncio.wait_for(session.cancel(), timeout=1.0)
            # Short-circuits before touching the client.
            assert client.interrupt_calls == 0

    @pytest.mark.asyncio
    async def test_cancel_noop_when_turn_already_ended(self):
        """A turn that has reached its result → cancel short-circuits without interrupt."""
        client = _OneTurnClient()
        async with _started_session(client) as session:
            await collect_turn(session, "hi")
            await asyncio.wait_for(session.cancel(), timeout=1.0)
            assert client.interrupt_calls == 0

    @pytest.mark.asyncio
    async def test_cancel_awaits_interrupt_and_drains(self):
        """In-flight turn: interrupt is awaited FIRST, then the turn drains to its result."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            consumer, frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)
            assert session.is_busy is True

            await asyncio.wait_for(session.cancel(), timeout=2.0)
            await asyncio.wait_for(consumer, timeout=2.0)

            # interrupt ran exactly once (proves it was awaited, not discarded).
            assert client.interrupt_calls == 1
            # Drained cleanly to its terminal message, not abandoned.
            assert frames[-1]["type"] == "result"
            assert session.is_busy is False

    @pytest.mark.asyncio
    async def test_cancel_abandons_a_turn_that_never_ends(self):
        """A turn that does not end once interrupted is abandoned after the bound.

        Its consumer is told so, and the turn closes.
        """
        client = FakeStreamClient(hang=True)
        async with _started_session(client) as session:
            consumer, frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            with patch(_OS + "_QUIESCE_TIMEOUT_S", 0.01):
                await asyncio.wait_for(session.cancel(), timeout=2.0)
            await asyncio.wait_for(consumer, timeout=2.0)

            assert client.interrupt_calls == 1
            assert frames[-1]["type"] == "error"
            assert frames[-1]["error_type"] == "TurnAbandoned"
            assert session.is_busy is False

    async def test_what_the_cli_says_after_an_abandon_still_reaches_the_feeds(self, monkeypatch):
        """The abandon closes the turn, not the reading: the CLI's later words reach every feed."""
        monkeypatch.setattr(_OS + "_QUIESCE_TIMEOUT_S", 0.01)
        client = _LateWordsClient()
        async with _started_session(client) as session:
            consumer, frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)
            await asyncio.wait_for(session.cancel(), timeout=2.0)
            await asyncio.wait_for(consumer, timeout=2.0)
            feed, _hello = session.subscribe()

            client.release.set()
            late = [await asyncio.wait_for(feed.get(), timeout=2.0) for _ in range(2)]

        assert frames[-1]["error_type"] == "TurnAbandoned"
        assert [(f["type"], f.get("content")) for f in late] == [("text", "late"), ("result", None)]

    @pytest.mark.asyncio
    async def test_spawn_quiesce_returns_stored_detached_task(self):
        """spawn_quiesce returns a task, stores it, and quiesces when awaited."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            consumer, _frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            task = session.spawn_quiesce()
            assert isinstance(task, asyncio.Task)
            assert session._quiesce_task is task

            await asyncio.wait_for(task, timeout=2.0)
            await asyncio.wait_for(consumer, timeout=2.0)
            assert client.interrupt_calls == 1
            assert session.is_busy is False

    @pytest.mark.asyncio
    async def test_last_activity_initialized_at_creation(self):
        """A new session's idle clock starts now, so the reaper cannot take it at once."""
        before = time.monotonic()
        session = OperatorSession(cwd="/tmp")
        after = time.monotonic()
        assert before <= session.last_activity <= after

    @pytest.mark.asyncio
    async def test_last_activity_restamped_on_turn_completion(self):
        client = _OneTurnClient()
        async with _started_session(client) as session:
            before = session.last_activity
            await collect_turn(session, "hi")
            assert session.last_activity > before


class TestOperatorSessionConnectedAgent:
    """The runner is reachable exactly while its scope is open.

    The handle outlives the scope on purpose, so a retained handle alone
    reaches nothing.
    """

    @pytest.mark.asyncio
    async def test_a_session_never_started_reaches_no_runner(self):
        client = FakeStreamClient()
        session = OperatorSession(cwd="/tmp")

        assert session._connected_agent is None
        await session.interrupt()
        with pytest.raises(RuntimeError, match="Session not started"):
            await session.send_prompt("hi")
        assert client.interrupt_calls == 0
        assert client.query_calls == 0

    @pytest.mark.asyncio
    async def test_a_started_session_reaches_its_runner(self):
        client = FakeStreamClient()
        async with _started_session(client) as session:
            assert session._connected_agent is not None
            assert session._connected_agent is session._agent
            # The signal-only interrupt reaches the runner while a turn is in flight.
            consumer, _frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)
            await session.interrupt()
            assert client.interrupt_calls == 1
            await asyncio.wait_for(consumer, timeout=2.0)

    async def test_an_interrupt_with_no_turn_in_flight_reaches_no_runner(self):
        client = FakeStreamClient()
        async with _started_session(client) as session:
            await session.interrupt()
        assert client.interrupt_calls == 0

    @pytest.mark.asyncio
    async def test_a_stopped_session_keeps_the_handle_and_reaches_no_runner(self):
        client = FakeStreamClient()
        session = OperatorSession(cwd="/tmp")
        with sdk_seam(client):
            await session.start()
            await session.stop()

            assert session._agent is not None
            assert session._connected_agent is None
            await session.interrupt()
            with pytest.raises(RuntimeError, match="Session not started"):
                await session.send_prompt("hi")
        assert client.interrupt_calls == 0
        assert client.query_calls == 0


class _ExitingClient(FakeStreamClient):
    """A child that exits when ``exited`` is set: its response stream then ends with no result.

    Until then each prompt is answered with one text block, and with a result
    too unless ``answers`` is False, so a turn can be left open when it exits.
    """

    def __init__(self, *, answers: bool = True) -> None:
        super().__init__()
        self.answers = answers
        self.exited = asyncio.Event()

    async def receive_response(self):
        prompt = asyncio.ensure_future(self._prompts.get())
        gone = asyncio.ensure_future(self.exited.wait())
        await asyncio.wait({prompt, gone}, return_when=asyncio.FIRST_COMPLETED)
        if self.exited.is_set():
            prompt.cancel()
            return
        yield assistant_message([TextBlock("done")])
        self.first_yielded.set()
        if self.answers:
            gone.cancel()
            yield result_message()
        else:
            await gone


class TestOperatorSessionChildExits:
    """A child that exits leaves the session dead, and says so to whoever is reading.

    A session that still read as active after its child was gone stayed pooled,
    and every later prompt failed until the operator started a new conversation.
    """

    @pytest.mark.asyncio
    async def test_an_idle_child_exit_ends_the_session_and_tells_every_feed(self):
        client = _ExitingClient()
        async with _started_session(client) as session:
            frames = await collect_turn(session, "hi")
            assert frames[-1]["type"] == "result"
            feed, _hello = session.subscribe()

            client.exited.set()
            frame = await asyncio.wait_for(feed.get(), timeout=2.0)

            assert frame["type"] == "error"
            assert frame["error_type"] == "AgentExited"
            assert session.is_active is False
            assert session.feed_open(feed) is False

    @pytest.mark.asyncio
    async def test_a_child_exit_mid_turn_ends_the_turn_and_the_session(self):
        client = _ExitingClient(answers=False)
        async with _started_session(client) as session:
            consumer, frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            client.exited.set()
            await asyncio.wait_for(consumer, timeout=2.0)

            assert frames[-1]["error_type"] == "AgentExited"
            assert session.in_flight is False
            assert session.is_active is False


class _UnpromptedClient(FakeStreamClient):
    """Starts a response of its own when ``speak`` is set, and ends it when ``finish`` is.

    As the CLI does after a background agent reports with no prompt behind it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.speak = asyncio.Event()
        self.finish = asyncio.Event()

    async def receive_response(self):
        await self.speak.wait()
        self.speak.clear()
        yield assistant_message([TextBlock("the report")])
        self.first_yielded.set()
        await self.finish.wait()
        yield result_message()


class TestHelloInFlight:
    """The ``hello`` snapshot says a response is under way whenever one is."""

    @pytest.mark.asyncio
    async def test_a_response_the_cli_started_on_its_own_is_in_flight(self):
        client = _UnpromptedClient()
        async with _started_session(client) as session:
            feed, before = session.subscribe()
            client.speak.set()
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            _feed, during = session.subscribe()
            client.finish.set()
            while (await asyncio.wait_for(feed.get(), timeout=1.0))["type"] != "result":
                pass
            _feed, after = session.subscribe()

        assert (before["in_flight"], during["in_flight"], after["in_flight"]) == (
            False,
            True,
            False,
        )


# ---------------------------------------------------------------------------
# OperatorRegistry
# ---------------------------------------------------------------------------


class TestOperatorRegistry:
    @pytest.mark.asyncio
    async def test_create_and_get(self):
        registry = OperatorRegistry()
        mock_session = AsyncMock(spec=OperatorSession)
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            return_value=mock_session,
        ):
            session = await registry.create_session("test-1", cwd="/tmp")
            assert session is mock_session
            assert registry.get_session("test-1") is mock_session
            mock_session.start.assert_awaited_once()

        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_create_replaces_existing(self):
        registry = OperatorRegistry()

        mock_s1 = AsyncMock(spec=OperatorSession)
        mock_s1.start = AsyncMock()
        mock_s1.stop = AsyncMock()

        mock_s2 = AsyncMock(spec=OperatorSession)
        mock_s2.start = AsyncMock()
        mock_s2.stop = AsyncMock()

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            side_effect=[mock_s1, mock_s2],
        ):
            await registry.create_session("default", cwd="/tmp")
            await registry.create_session("default", cwd="/tmp")

        # First session should have been stopped
        mock_s1.stop.assert_awaited_once()
        assert registry.get_session("default") is mock_s2

        await registry.cleanup_all()

    @pytest.mark.asyncio
    async def test_terminate_session(self):
        registry = OperatorRegistry()
        mock_session = AsyncMock(spec=OperatorSession)
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            return_value=mock_session,
        ):
            await registry.create_session("test-1", cwd="/tmp")

        await registry.terminate_session("test-1")
        assert registry.get_session("test-1") is None
        mock_session.stop.assert_awaited()

    @pytest.mark.asyncio
    async def test_stale_cleanup_does_not_kill_replacement(self):
        """Simulate page reload: WS1 creates session, WS2 replaces it,
        then WS1's finally block runs — must NOT kill WS2's session."""
        registry = OperatorRegistry()

        mock_s1 = AsyncMock(spec=OperatorSession)
        mock_s1.start = AsyncMock()
        mock_s1.stop = AsyncMock()

        mock_s2 = AsyncMock(spec=OperatorSession)
        mock_s2.start = AsyncMock()
        mock_s2.stop = AsyncMock()

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            side_effect=[mock_s1, mock_s2],
        ):
            await registry.create_session("default", cwd="/tmp")
            await registry.create_session("default", cwd="/tmp")

        # WS1's cleanup runs with the OLD session reference
        await registry.terminate_session_if_owner("default", mock_s1)

        # s2 must NOT have been stopped by the stale cleanup
        assert registry.get_session("default") is mock_s2
        mock_s2.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_owner_terminate_works(self):
        registry = OperatorRegistry()
        mock_session = AsyncMock(spec=OperatorSession)
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()

        with patch(
            "osprey.interfaces.web_terminal.operator_session.OperatorSession",
            return_value=mock_session,
        ):
            await registry.create_session("default", cwd="/tmp")

        await registry.terminate_session_if_owner("default", mock_session)
        assert registry.get_session("default") is None


class FakeTask:
    """Minimal stand-in for an asyncio.Task with a controllable done() state."""

    def __init__(self, done: bool = False):
        self._done = done

    def done(self) -> bool:
        return self._done


class FakeChatSession:
    """Lightweight OperatorSession double for registry pool tests."""

    def __init__(self, cwd: str = "/tmp", env=None, session_key=None):
        self.cwd = cwd
        self.env = env
        self.session_key = session_key
        self.resume_id = None
        self.is_active = True
        self.last_activity = time.monotonic()
        self.in_flight = False
        self._pump_task = None
        self._turn_open = False
        self._quiesce_task = None
        self.start_calls = 0
        self.stop_calls = 0
        self.start_delay = 0.0
        self.start_error: Exception | None = None
        # What ``started_commands`` answers (``None`` raises as a table that could
        # not be read), the threads it was asked on, and what it does when asked.
        self.running_commands: list | None = []
        self.look_threads: list[int] = []
        self.look_error: Exception | None = None
        self.on_look: Callable[[], None] | None = None

    def started_commands(self) -> list:
        self.look_threads.append(threading.get_ident())
        if self.on_look is not None:
            self.on_look()
        if self.look_error is not None:
            raise self.look_error
        if self.running_commands is None:
            raise ProcessTableUnreadable(OSError("process table"))
        return list(self.running_commands)

    async def start(self, *, resume_id=None):
        self.resume_id = resume_id
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        if self.start_error is not None:
            raise self.start_error
        self.start_calls += 1

    async def stop(self):
        self.stop_calls += 1
        self.is_active = False

    @property
    def is_busy(self) -> bool:
        # Mirrors OperatorSession.is_busy against this double's plain attrs.
        reader_running = self._pump_task is not None and not self._pump_task.done()
        quiesce_running = self._quiesce_task is not None and not self._quiesce_task.done()
        return self.in_flight and ((reader_running and self._turn_open) or quiesce_running)

    async def teardown(self):
        await self.stop()


class TestOperatorSessionBusyAndTeardown:
    """Pin the REAL OperatorSession.is_busy / teardown the pool drives.

    The pool tests below exercise FakeChatSession's mirror of is_busy; these
    cover the shipped property itself so the mirror cannot silently drift.
    """

    def test_not_in_flight_is_not_busy(self):
        session = OperatorSession(cwd="/tmp")
        assert session.is_busy is False

    async def test_a_turn_in_flight_on_a_live_reader_is_busy(self):
        client = FakeStreamClient()
        async with _started_session(client) as session:
            consumer, _frames = start_turn(session, "hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)
            assert session.is_busy is True

            await session.interrupt()
            await asyncio.wait_for(consumer, timeout=2.0)
            assert session.is_busy is False

    def test_guard_held_with_running_quiesce_is_busy(self):
        session = OperatorSession(cwd="/tmp")
        session.acquire_turn()
        session._quiesce_task = FakeTask(done=False)
        assert session.is_busy is True

    def test_zombie_guard_with_no_reader_or_quiesce_is_not_busy(self):
        session = OperatorSession(cwd="/tmp")
        session.acquire_turn()
        session._quiesce_task = FakeTask(done=True)
        assert session.is_busy is False

    @pytest.mark.asyncio
    async def test_teardown_awaits_pending_quiesce_then_stops(self):
        session = OperatorSession(cwd="/tmp")
        quiesce_ran = asyncio.Event()

        async def fake_quiesce():
            quiesce_ran.set()

        session._quiesce_task = asyncio.create_task(fake_quiesce())
        await session.teardown()
        assert quiesce_ran.is_set()
        assert session.is_active is False


class FakeChildProcess:
    """The slice of the SDK transport's process handle ``stop`` reads and signals."""

    def __init__(self, returncode: int | None = None, *, gone: bool = False, pid: int = 4242):
        self.returncode = returncode
        self.gone = gone
        self.pid = pid
        self.kills = 0

    def kill(self) -> None:
        if self.gone:
            raise ProcessLookupError
        self.kills += 1


def _client_over(process: FakeChildProcess) -> AsyncMock:
    client = mock_sdk_client()
    client._transport = SimpleNamespace(_process=process)
    return client


async def _started_on(client) -> OperatorSession:
    """A session started on *client* through the SDK seam."""
    session = OperatorSession(cwd="/tmp")
    with sdk_seam(client):
        await session.start()
    return session


async def _started_over(process: FakeChildProcess) -> OperatorSession:
    return await _started_on(_client_over(process))


class TestOperatorSessionStopKillsALingeringChild:
    """``stop`` sends SIGKILL to a retained child that has no return code yet.

    The SDK's own close escalates through SIGTERM to SIGKILL with bounded
    waits; this covers the child that outlived it, and — the case a hand-off
    relies on — the second ``stop`` on a session whose client is already gone.
    """

    @pytest.mark.asyncio
    async def test_a_child_still_running_after_the_client_closed_is_killed(self):
        process = FakeChildProcess(returncode=None)
        session = await _started_over(process)

        await session.stop()

        assert process.kills == 1
        assert session.pid == process.pid
        assert session.process_exited is False
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_a_child_that_exited_is_not_signalled(self):
        process = FakeChildProcess(returncode=0)
        session = await _started_over(process)

        await session.stop()

        assert process.kills == 0
        assert session.process_exited is True

    @pytest.mark.asyncio
    async def test_a_second_stop_with_no_client_signals_the_retained_child_again(self):
        process = FakeChildProcess(returncode=None)
        session = await _started_over(process)
        await session.stop()
        assert process.kills == 1

        await session.teardown()

        assert process.kills == 2
        assert session.pid == process.pid
        assert session.process_exited is False

    @pytest.mark.asyncio
    async def test_a_child_gone_between_looks_is_tolerated(self):
        process = FakeChildProcess(returncode=None, gone=True)
        session = await _started_over(process)

        await session.stop()

        assert process.kills == 0
        assert session.is_active is False

    @pytest.mark.asyncio
    async def test_a_session_with_no_handle_signals_nothing(self):
        session = OperatorSession(cwd="/tmp")
        await session.stop()
        assert session.pid is None
        assert session.process_exited is None


# ---------------------------------------------------------------------------
# OperatorSession.process_exited through a real start()
# ---------------------------------------------------------------------------


class _FakeProcess:
    """The child handle the SDK transport holds (``returncode`` is the signal)."""

    def __init__(self, returncode: int | None = None):
        self.returncode = returncode


class _FakeTransport:
    def __init__(self, process: _FakeProcess):
        self._process = process


class _FakeSDKClient:
    """Stand-in for ``ClaudeSDKClient`` with a controllable transport.

    ``transport=None`` models a client that never exposed one — the shape the
    property must answer ``None`` for rather than ``False``.
    """

    def __init__(self, transport: _FakeTransport | None = None):
        if transport is not None:
            self._transport = transport
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.exited = True
        return False


class TestProcessExited:
    """Three answers, and the difference between two of them matters."""

    @pytest.mark.asyncio
    async def test_a_child_still_running_after_the_close_reads_false(self):
        """Closing the client is a request, not a death certificate.

        The transport is gone but the process it held has no return code, so
        the child is still there — the case a handover must keep waiting on.
        """
        client = _FakeSDKClient(_FakeTransport(_FakeProcess(returncode=None)))
        session = await _started_on(client)

        await session.stop()

        assert client.exited is True
        assert session.process_exited is False

    @pytest.mark.parametrize("returncode", [0, 137])
    @pytest.mark.asyncio
    async def test_a_child_that_exited_reads_true(self, returncode):
        """The question is whether the child is gone, not how it went."""
        client = _FakeSDKClient(_FakeTransport(_FakeProcess(returncode=returncode)))
        session = await _started_on(client)

        await session.stop()

        assert session.process_exited is True

    @pytest.mark.asyncio
    async def test_a_client_with_no_transport_reads_none(self):
        """Nothing to observe is not the same as *not exited*."""
        client = _FakeSDKClient(transport=None)
        session = await _started_on(client)

        await session.stop()

        assert session.process_exited is None

    @pytest.mark.asyncio
    async def test_the_handle_is_read_before_the_client_is_closed(self):
        """Captured from the live transport, not from whatever survives close.

        The real client drops ``_transport`` on ``__aexit__``; a fake that does
        the same would answer ``None`` if the capture happened after.
        """
        process = _FakeProcess(returncode=None)
        client = _FakeSDKClient(_FakeTransport(process))

        async def _drop_transport(*exc):
            client._transport = None
            return False

        client.__aexit__ = _drop_transport  # type: ignore[method-assign]
        session = await _started_on(client)

        await session.stop()

        process.returncode = 0
        assert session.process_exited is True


@pytest.mark.asyncio
async def test_the_pid_names_the_running_child_and_nothing_else():
    """``pid`` is the chat child's pid while it runs, and ``None`` when it is not.

    The Simple view holds a session in this child, so anything addressed to
    the process behind a session key — a controls-server record, a
    diagnostic — has to be able to ask which process that is. A session that
    never started has no handle, and a handle carrying a return code names a
    process that is gone; both answer ``None`` rather than a pid nothing is
    running under. A client already closed still answers from the handle the
    session retained, which is what makes the question survive a teardown.
    """
    assert OperatorSession(cwd="/tmp").pid is None

    process = FakeChildProcess(returncode=None, pid=31337)
    session = await _started_over(process)
    assert session.pid == 31337

    await session.stop()
    assert session.pid == 31337

    process.returncode = 0
    assert session.pid is None


def _session_factory(start_delay: float = 0.0):
    """Return a side_effect callable that builds FakeChatSessions and records them."""
    created: list[FakeChatSession] = []

    def factory(cwd=None, env=None, session_key=None, **_sink):
        s = FakeChatSession(cwd=cwd, env=env, session_key=session_key)
        s.start_delay = start_delay
        created.append(s)
        return s

    factory.created = created  # type: ignore[attr-defined]
    return factory


@contextlib.contextmanager
def _patch_session(factory):
    with patch(
        "osprey.interfaces.web_terminal.operator_session.OperatorSession",
        side_effect=factory,
    ):
        yield


class TestOperatorRegistryChatPool:
    @pytest.mark.asyncio
    async def test_create_and_get_namespaced_key(self):
        registry = OperatorRegistry()
        factory = _session_factory()
        with _patch_session(factory):
            session, was_reused = await registry.get_or_create_chat_session("a", cwd="/tmp")

        assert was_reused is False
        assert len(factory.created) == 1
        assert session.start_calls == 1
        assert registry.get_chat_session("a") is session
        # Chat sessions live in the pool's own map, never in the operator map.
        assert "a" in registry.chats._sessions
        assert "a" not in registry._sessions
        assert registry.get_chat_session("missing") is None

    @pytest.mark.asyncio
    async def test_dead_session_is_replaced_and_torn_down(self):
        registry = OperatorRegistry()
        factory = _session_factory()
        with _patch_session(factory):
            s1, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            s1.is_active = False  # simulate a crashed client
            s2, r2 = await registry.get_or_create_chat_session("a", cwd="/tmp")

        assert s2 is not s1
        assert r2 is False
        assert s1.stop_calls == 1
        assert registry.get_chat_session("a") is s2

    @pytest.mark.asyncio
    async def test_double_submit_shares_one_creation(self):
        """Concurrent get_or_create for the same id must start only one session."""
        registry = OperatorRegistry()
        factory = _session_factory(start_delay=0.05)
        with _patch_session(factory):
            results = await asyncio.gather(
                registry.get_or_create_chat_session("a", cwd="/tmp"),
                registry.get_or_create_chat_session("a", cwd="/tmp"),
            )

        (s0, r0), (s1, r1) = results
        assert s0 is s1
        assert len(factory.created) == 1
        assert factory.created[0].start_calls == 1
        # Exactly one creator (was_reused False), one joiner (True).
        assert {r0, r1} == {True, False}

    @pytest.mark.asyncio
    async def test_capacity_evicts_lru_non_busy(self):
        registry = OperatorRegistry(chat_max_sessions=2)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            b, _ = await registry.get_or_create_chat_session("b", cwd="/tmp")
            c, _ = await registry.get_or_create_chat_session("c", cwd="/tmp")

        # 'a' was least-recently-used and not busy → evicted for 'c'.
        assert registry.get_chat_session("a") is None
        assert a.stop_calls == 1
        assert registry.get_chat_session("b") is b
        assert registry.get_chat_session("c") is c

    @pytest.mark.asyncio
    async def test_reuse_bumps_lru_order(self):
        registry = OperatorRegistry(chat_max_sessions=2)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            b, _ = await registry.get_or_create_chat_session("b", cwd="/tmp")
            # Touch 'a' so 'b' becomes the LRU.
            await registry.get_or_create_chat_session("a", cwd="/tmp")
            await registry.get_or_create_chat_session("c", cwd="/tmp")

        assert registry.get_chat_session("b") is None
        assert b.stop_calls == 1
        assert registry.get_chat_session("a") is a
        assert registry.get_chat_session("c") is not None

    @pytest.mark.asyncio
    async def test_all_busy_raises_capacity_error(self):
        registry = OperatorRegistry(chat_max_sessions=1)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            # Genuinely busy: guard held, and a turn open on a live reader.
            a.in_flight = True
            a._pump_task = FakeTask(done=False)
            a._turn_open = True

            with pytest.raises(ChatCapacityError):
                await registry.get_or_create_chat_session("b", cwd="/tmp")

        # 'a' untouched; nothing new started.
        assert registry.get_chat_session("a") is a
        assert a.stop_calls == 0

    @pytest.mark.asyncio
    async def test_zombie_busy_is_evictable(self):
        """Guard held but reader + quiesce both done → not busy → evictable."""
        registry = OperatorRegistry(chat_max_sessions=1)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            a.in_flight = True
            a._pump_task = FakeTask(done=True)
            a._quiesce_task = FakeTask(done=True)

            b, r = await registry.get_or_create_chat_session("b", cwd="/tmp")

        assert r is False
        assert registry.get_chat_session("a") is None
        assert a.stop_calls == 1
        assert registry.get_chat_session("b") is b

    @pytest.mark.asyncio
    async def test_reap_idle_reaps_stale_not_busy_only(self):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            b, _ = await registry.get_or_create_chat_session("b", cwd="/tmp")
            c, _ = await registry.get_or_create_chat_session("c", cwd="/tmp")

        now = time.monotonic()
        a.last_activity = now - 100  # stale, not busy → reaped
        b.last_activity = now  # fresh → kept
        c.last_activity = now - 100  # stale but busy → kept
        c.in_flight = True
        c._pump_task = FakeTask(done=False)
        c._turn_open = True

        reaped = await registry.reap_idle_chat_sessions()
        assert reaped == 1
        assert registry.get_chat_session("a") is None
        assert a.stop_calls == 1
        assert registry.get_chat_session("b") is b
        assert registry.get_chat_session("c") is c

    @pytest.mark.asyncio
    async def test_reap_idle_keeps_a_stale_chat_with_a_started_command_running(self):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
            b, _ = await registry.get_or_create_chat_session("b", cwd="/tmp")

        stale = time.monotonic() - 100
        a.last_activity = stale
        a.running_commands = [object()]
        b.last_activity = stale

        reaped = await registry.reap_idle_chat_sessions()
        assert reaped == 1
        assert registry.get_chat_session("a") is a
        assert a.stop_calls == 0
        assert a.last_activity > stale
        assert registry.get_chat_session("b") is None

    @pytest.mark.asyncio
    async def test_reap_idle_never_looks_at_a_busy_chat(self):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            c, _ = await registry.get_or_create_chat_session("c", cwd="/tmp")
        c.last_activity = time.monotonic() - 100
        c.in_flight = True
        c._pump_task = FakeTask(done=False)
        c._turn_open = True

        assert await registry.reap_idle_chat_sessions() == 0
        assert c.look_threads == []

    @pytest.mark.asyncio
    async def test_reap_idle_looks_off_the_loop_without_the_pool_lock(self):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
        a.last_activity = time.monotonic() - 100
        held: list[bool] = []
        a.on_look = lambda: held.append(registry.chats._lock.locked())

        await registry.reap_idle_chat_sessions()

        loop_thread = threading.get_ident()
        assert a.look_threads
        assert all(thread != loop_thread for thread in a.look_threads)
        assert held == [False] * len(a.look_threads)

    @pytest.mark.asyncio
    async def test_reap_idle_keeps_a_chat_whose_process_table_could_not_be_read(self, caplog):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("stale-chat", cwd="/tmp")
        # Past the idle timeout, within the grace an unknown look is given.
        stale = time.monotonic() - 15
        a.last_activity = stale
        a.running_commands = None

        with caplog.at_level(logging.WARNING):
            reaped = await registry.reap_idle_chat_sessions()

        assert reaped == 0
        assert registry.get_chat_session("stale-chat") is a
        assert a.stop_calls == 0
        assert a.last_activity == stale
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "stale-chat" in r.getMessage()
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_reap_idle_keeps_a_chat_whose_look_failed(self, caplog):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("stale-chat", cwd="/tmp")
        stale = time.monotonic() - 15
        a.last_activity = stale
        a.look_error = OSError("ps")

        with caplog.at_level(logging.WARNING):
            reaped = await registry.reap_idle_chat_sessions()

        assert reaped == 0
        assert registry.get_chat_session("stale-chat") is a
        assert a.stop_calls == 0
        assert a.last_activity == stale
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "stale-chat" in r.getMessage()
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_an_unreadable_process_table_does_not_keep_a_chat_open_for_good(self):
        """Every sweep that cannot look leaves the idle clock alone, so the reap comes."""
        # Whole seconds, as test_chat_process_tree.py's clock reads: on a raw
        # monotonic stamp ``last + 20.0`` can round to one step short of 20 s
        # after ``last``, and the sweep landing exactly on the timeout misses.
        last = 1000.0
        clock = [last]
        pool = ChatSessionPool(factory=_session_factory(), idle_seconds=10.0, now=lambda: clock[0])
        a, _ = await pool.get_or_create("stale-chat", "/tmp")
        a.running_commands = None
        a.last_activity = last

        for moment in (10.0, 15.0, 10.0 * UNREAD_IDLE_FACTOR - 1.0):
            clock[0] = last + moment
            assert await pool.reap_idle() == 0
            assert pool.get("stale-chat") is a
            assert a.last_activity == last

        clock[0] = last + 10.0 * UNREAD_IDLE_FACTOR
        assert await pool.reap_idle() == 1
        assert pool.get("stale-chat") is None
        assert a.stop_calls == 1

    @pytest.mark.asyncio
    async def test_reap_idle_leaves_a_chat_replaced_during_the_look(self):
        registry = OperatorRegistry(chat_idle_seconds=10.0)
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")
        a.last_activity = time.monotonic() - 100
        fresh = FakeChatSession()

        def replace() -> None:
            # A creation racing the sweep fills the key with a new session.
            registry.chats._sessions["a"] = fresh

        a.on_look = replace

        assert await registry.reap_idle_chat_sessions() == 0
        assert registry.get_chat_session("a") is fresh
        assert fresh.stop_calls == 0

    @pytest.mark.asyncio
    async def test_cleanup_all_tears_down_both_pools(self):
        registry = OperatorRegistry()
        factory = _session_factory()
        with _patch_session(factory):
            op = await registry.create_session("op-1", cwd="/tmp")
            chat, _ = await registry.get_or_create_chat_session("c", cwd="/tmp")

        await registry.cleanup_all()

        assert registry.get_session("op-1") is None
        assert registry.get_chat_session("c") is None
        assert op.stop_calls == 1
        assert chat.stop_calls == 1

    @pytest.mark.asyncio
    async def test_start_failure_clears_pending_and_allows_retry(self):
        registry = OperatorRegistry()

        created: list[FakeChatSession] = []

        def factory(cwd=None, env=None, session_key=None, **_sink):
            s = FakeChatSession(cwd=cwd, env=env, session_key=session_key)
            # First construction fails during start; later ones succeed.
            if not created:
                s.start_error = RuntimeError("boom")
            created.append(s)
            return s

        with _patch_session(factory):
            with pytest.raises(RuntimeError, match="boom"):
                await registry.get_or_create_chat_session("a", cwd="/tmp")

            # Pending marker cleared; map has no stale entry.
            assert "a" not in registry.chats._sessions
            assert "a" not in registry.chats._pending

            # A retry succeeds cleanly.
            s2, r2 = await registry.get_or_create_chat_session("a", cwd="/tmp")
            assert r2 is False
            assert s2.start_calls == 1
            assert registry.get_chat_session("a") is s2


# ---------------------------------------------------------------------------
# validate_project_directory
# ---------------------------------------------------------------------------


class TestValidateProjectDirectory:
    def test_all_files_present(self, tmp_path):
        (tmp_path / ".mcp.json").touch()
        (tmp_path / "CLAUDE.md").touch()
        (tmp_path / ".claude").mkdir()
        (tmp_path / "config.yml").touch()

        warnings = validate_project_directory(str(tmp_path))
        assert warnings == []

    @pytest.mark.parametrize(
        ("present", "missing"),
        [
            ((), (".mcp.json", "CLAUDE.md", ".claude", "config.yml")),
            (("config.yml", ".claude"), (".mcp.json", "CLAUDE.md")),
        ],
        ids=["none", "partial"],
    )
    def test_each_missing_file_is_named(self, tmp_path, present, missing):
        for name in present:
            if name == ".claude":
                (tmp_path / name).mkdir()
            else:
                (tmp_path / name).touch()

        warnings = validate_project_directory(str(tmp_path))
        assert len(warnings) == len(missing)
        for name in missing:
            assert any(name in w for w in warnings), name


# ---------------------------------------------------------------------------
# build_clean_env — project_cwd parameter
# ---------------------------------------------------------------------------


class TestBuildCleanEnvProjectCwd:
    def test_sets_osprey_config_when_config_exists(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yml"
        config_file.touch()
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert env["OSPREY_CONFIG"] == str(config_file)

    @pytest.mark.parametrize("cwd", ["empty-dir", None], ids=["no-config-file", "no-cwd"])
    def test_skips_when_no_config_file(self, tmp_path, monkeypatch, cwd):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path) if cwd else None)
        assert "OSPREY_CONFIG" not in env

    def test_does_not_override_existing_osprey_config(self, tmp_path, monkeypatch):
        (tmp_path / "config.yml").touch()
        monkeypatch.setenv("OSPREY_CONFIG", "/custom/config.yml")

        env = build_clean_env(project_cwd=str(tmp_path))
        assert env["OSPREY_CONFIG"] == "/custom/config.yml"


# ---------------------------------------------------------------------------
# Parity pins: launch options, stream frames, child handling
# ---------------------------------------------------------------------------

_KEY = TestOperatorSessionResumeOptions.KEY
_PRIMITIVES = "osprey.agent_runner.primitives."


class _ScriptedClient:
    """A client whose response to each prompt is *messages*, or *error* raised mid-turn.

    As the SDK's is, ``receive_response`` answers only a prompt that was sent.
    """

    def __init__(self, messages=(), *, error: Exception | None = None) -> None:
        self.messages = list(messages)
        self.error = error
        self.interrupt_calls = 0
        self._prompts: asyncio.Queue[str] = asyncio.Queue()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self._prompts.put_nowait(prompt)

    async def interrupt(self):
        self.interrupt_calls += 1

    async def receive_response(self):
        await self._prompts.get()
        for message in self.messages:
            yield message
        if self.error is not None:
            raise self.error


class TestOperatorChatLaunchParity:
    """The options the chat launches with are the ones it always built by hand."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", ["resume", "posture_key", "foreign_key"])
    async def test_the_launch_options_equal_the_hand_built_ones(self, shape):
        key = "e2e" if shape == "foreign_key" else _KEY
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=key)

        with sdk_seam() as captured:
            if shape == "resume":
                await session.start(resume_id="transcript-9")
            else:
                await session.start()

        (options,) = captured
        if shape == "resume":
            identity = {"resume": "transcript-9"}
        else:
            identity = {"session_id": _KEY if shape == "posture_key" else None}
        assert options == ClaudeAgentOptions(
            system_prompt=PRESET_SYSTEM_PROMPT,
            cwd="/tmp",
            env=options.env,
            setting_sources=["project"],
            mcp_servers="/tmp/.mcp.json",
            strict_mcp_config=True,
            cli_path=STUB_AGENT_CLI,
            include_partial_messages=True,
            can_use_tool=session._can_use_tool,
            # The stub CLI takes --thinking-display, so the launch passes the
            # default display.
            extra_args={"thinking-display": "summarized"},
            **identity,
        )
        assert options.env["PATH"] == "/usr/bin"
        assert options.env["OSPREY_TELEMETRY_SESSION_ID"] == key
        assert options.env["OSPREY_TELEMETRY_SESSION_START"]
        assert options.env["OSPREY_WEB_UX"] == "simple"
        await session.stop()

    @pytest.mark.asyncio
    async def test_the_launch_resolves_no_provider_and_waits_for_no_mcp_server(self):
        def refuse(*_args, **_kwargs):
            raise AssertionError("the chat launch must not route its own run")

        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=_KEY)
        with (
            patch(_PRIMITIVES + "sdk_env", side_effect=refuse),
            patch(_PRIMITIVES + "resolve_default_model", side_effect=refuse),
            patch(_PRIMITIVES + "_resolve_project_spec", side_effect=refuse),
            patch(_PRIMITIVES + "start_proxy_for", side_effect=refuse),
            patch(_PRIMITIVES + "await_mcp_ready", side_effect=refuse),
            sdk_seam(),
        ):
            await session.start()

        assert session.is_active
        await session.stop()

    @pytest.mark.asyncio
    async def test_chat_child_rules_come_from_the_project_and_its_callback_answers_asks(self):
        """What the chat child is launched with, and what its one SDK callback may do.

        The project's settings and hooks are the rules: ``setting_sources`` loads them, and the
        SDK is handed no hooks of its own. Nothing bypasses them: ``permission_mode`` stays
        unset, so the CLI's default applies and the approval hook is not skipped. The one SDK
        callback is the approval card, which answers a permission ask the CLI raises.

        The rest is behaviour, so this test pins only the launch. That a call a hook denies never
        reaches the callback, and that an allow runs the call, are proved on the real CLI by
        ``tests/e2e/claude_code/test_chat_approval_gates_scripted.py`` (scenarios a to d).
        """
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=_KEY)

        with sdk_seam() as captured:
            await session.start()

        (options,) = captured
        assert options.setting_sources == ["project"]
        assert options.hooks is None
        assert options.permission_mode in (None, "default")
        assert options.can_use_tool == session._can_use_tool
        await session.stop()

    @pytest.mark.asyncio
    async def test_a_session_built_without_an_env_inherits_the_process_env(self):
        session = OperatorSession(cwd="/tmp", session_key=_KEY)

        with (
            patch(_PRIMITIVES + "sdk_env") as sdk_env,
            sdk_seam() as captured,
        ):
            await session.start()

        (options,) = captured
        assert options.env == {"CLAUDE_CODE_ENABLE_TODO_TOOLS": "1"}
        sdk_env.assert_not_called()
        await session.stop()


class TestOperatorChatStreamParity:
    """A real agent stream reaches the queue as the frames the chat always sent."""

    @pytest.mark.asyncio
    async def test_a_real_agent_stream_reaches_the_queue_frame_for_frame(self):
        tool = "mcp__osprey_workspace__channel_read"
        client = _ScriptedClient(
            [
                assistant_message(
                    [
                        ThinkingBlock("pondering", "sig"),
                        ToolUseBlock("tu_1", tool, {"channel": "SR:BPM"}),
                        TextBlock("done"),
                    ],
                    error="rate_limit",
                ),
                user_message([ToolResultBlock("tu_1", "42.0", is_error=False)]),
                system_message("init", {"session_id": "abc"}),
                result_message(),
            ]
        )
        async with _started_session(client) as session:
            frames = await collect_turn(session, "read it")

        # The feed numbers every frame in the order it was emitted.
        assert [f["seq"] for f in frames] == list(range(1, len(frames) + 1))
        assert [unnumbered(f) for f in frames] == [
            {"type": "turn_start", "prompt": "read it"},
            {
                "type": "error",
                "message": "API error: rate_limit",
                "error_type": "AssistantMessageError",
            },
            {"type": "thinking", "content": "pondering", "parent_tool_use_id": None},
            {
                "type": "tool_use",
                "tool_name": "Channel Read",
                "tool_name_raw": tool,
                "tool_use_id": "tu_1",
                "input": {"channel": "SR:BPM"},
                "parent_tool_use_id": None,
            },
            {"type": "text", "content": "done", "parent_tool_use_id": None},
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "is_error": False,
                "preview": "42.0",
                "truncated": False,
                "refusal": None,
            },
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {
                "type": "result",
                "is_error": False,
                "total_cost_usd": 0.01,
                "duration_ms": 1200,
                "num_turns": 1,
            },
        ]

    @pytest.mark.asyncio
    async def test_an_agent_sdk_failure_mid_turn_is_named_for_its_class(self):
        client = _ScriptedClient(error=CLIConnectionError("gone"))
        async with _started_session(client) as session:
            frames = [unnumbered(f) for f in await collect_turn(session, "x")]

        assert frames[-1] == {
            "type": "error",
            "message": "gone",
            "error_type": "CLIConnectionError",
        }

    @pytest.mark.asyncio
    async def test_an_unexpected_failure_mid_turn_keeps_its_prefix(self):
        client = _ScriptedClient(error=ValueError("x"))
        async with _started_session(client) as session:
            frames = await collect_turn(session, "x")

        assert frames[-1]["type"] == "error"
        assert frames[-1]["message"] == "Unexpected error: x"
        assert frames[-1]["error_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_interrupt_after_stop_reaches_no_client(self):
        client = FakeStreamClient()
        async with _started_session(client) as session:
            pass
        calls = client.interrupt_calls

        await session.interrupt()
        await session.cancel()

        assert client.interrupt_calls == calls


def test_the_operator_session_module_imports_nothing_from_the_agent_sdk():
    """The chat reaches the agent SDK only through the agent runner."""
    root = Path(__file__).resolve().parents[3] / "src" / "osprey" / "interfaces" / "web_terminal"
    for path in (root / "operator_session.py", root / "routes" / "chat.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = []
            assert not any(name.split(".")[0] == "claude_agent_sdk" for name in names), path
            if isinstance(node, ast.Attribute):
                assert node.attr != "_transport", path
