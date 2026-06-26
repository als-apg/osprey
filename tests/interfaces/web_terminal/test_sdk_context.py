"""Tests for the SDK context primitives used by operator and dispatch paths."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from claude_agent_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)

from osprey.interfaces.web_terminal.sdk_context import (
    build_system_prompt,
    make_tool_allowlist,
)


class TestBuildSystemPrompt:
    def test_shape(self):
        prompt = build_system_prompt(ZoneInfo("Europe/Berlin"))

        assert prompt["type"] == "preset"
        assert prompt["preset"] == "claude_code"
        assert prompt["append"].startswith("Facility local time: ")

    def test_with_extra(self):
        prompt = build_system_prompt(
            ZoneInfo("Asia/Tokyo"),
            extra="Machine mode: User beam",
        )

        lines = prompt["append"].split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("Facility local time: ")
        assert lines[1] == "Machine mode: User beam"


class TestMakeToolAllowlist:
    async def test_allows_tool_in_set(self):
        callback = make_tool_allowlist(["Read", "mcp__ariel__keyword_search"])

        result = await callback("Read", {"path": "foo"}, ToolPermissionContext())

        assert isinstance(result, PermissionResultAllow)

    async def test_denies_tool_not_in_set(self):
        callback = make_tool_allowlist(["Read"])

        result = await callback("Write", {"path": "foo"}, ToolPermissionContext())

        assert isinstance(result, PermissionResultDeny)
        assert "Write" in result.message
