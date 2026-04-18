"""Tests for operator session management."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.operator_session import (
    OperatorRegistry,
    OperatorSession,
    _format_tool_name,
    _message_to_events,
    build_clean_env,
    validate_project_directory,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight fakes for SDK message types
# ---------------------------------------------------------------------------


class FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class FakeThinkingBlock:
    def __init__(self, thinking: str, signature: str = "sig"):
        self.thinking = thinking
        self.signature = signature


class FakeToolUseBlock:
    def __init__(self, name: str, id: str, input: dict):
        self.name = name
        self.id = id
        self.input = input


class FakeToolResultBlock:
    def __init__(self, tool_use_id: str, content: str, is_error: bool = False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class FakeAssistantMessage:
    """Mimics ``claude_agent_sdk.AssistantMessage``."""

    def __init__(self, content: list, error=None):
        self.content = content
        self.error = error


class FakeResultMessage:
    """Mimics ``claude_agent_sdk.ResultMessage``."""

    def __init__(
        self,
        is_error: bool = False,
        total_cost_usd: float = 0.01,
        duration_ms: int = 1200,
        num_turns: int = 1,
    ):
        self.is_error = is_error
        self.total_cost_usd = total_cost_usd
        self.duration_ms = duration_ms
        self.num_turns = num_turns


class FakeSystemMessage:
    """Mimics ``claude_agent_sdk.SystemMessage``."""

    def __init__(self, subtype: str = "init"):
        self.subtype = subtype


# ---------------------------------------------------------------------------
# _format_tool_name
# ---------------------------------------------------------------------------


class TestFormatToolName:
    def test_strips_mcp_prefix(self):
        assert _format_tool_name("mcp__osprey__channel_read") == "Channel Read"

    def test_leaves_plain_name(self):
        assert _format_tool_name("Read") == "Read"

    def test_title_cases_underscored(self):
        assert _format_tool_name("file_search") == "File Search"

    def test_multi_segment_mcp(self):
        assert _format_tool_name("mcp__ariel__entry_create") == "Entry Create"


# ---------------------------------------------------------------------------
# _message_to_events
# ---------------------------------------------------------------------------


class TestMessageToEvents:
    """Patch isinstance checks so our fakes work with the real converter."""

    @pytest.fixture(autouse=True)
    def _patch_sdk_types(self):
        """Replace SDK type references with our fakes so isinstance() works."""
        with (
            patch(
                "osprey.interfaces.web_terminal.operator_session.AssistantMessage",
                FakeAssistantMessage,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ResultMessage",
                FakeResultMessage,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.SystemMessage",
                FakeSystemMessage,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.TextBlock",
                FakeTextBlock,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ThinkingBlock",
                FakeThinkingBlock,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ToolUseBlock",
                FakeToolUseBlock,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ToolResultBlock",
                FakeToolResultBlock,
            ),
        ):
            yield

    def test_text_block(self):
        msg = FakeAssistantMessage([FakeTextBlock("hello")])
        events = _message_to_events(msg)
        assert len(events) == 1
        assert events[0] == {"type": "text", "content": "hello"}

    def test_thinking_block(self):
        msg = FakeAssistantMessage([FakeThinkingBlock("pondering...")])
        events = _message_to_events(msg)
        assert len(events) == 1
        assert events[0] == {"type": "thinking", "content": "pondering..."}

    def test_tool_use_block(self):
        msg = FakeAssistantMessage(
            [FakeToolUseBlock("mcp__osprey__channel_read", "tu_1", {"channels": ["X"]})]
        )
        events = _message_to_events(msg)
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "tool_use"
        assert ev["tool_name"] == "Channel Read"
        assert ev["tool_name_raw"] == "mcp__osprey__channel_read"
        assert ev["tool_use_id"] == "tu_1"
        assert ev["input"] == {"channels": ["X"]}

    def test_tool_result_block(self):
        msg = FakeAssistantMessage([FakeToolResultBlock("tu_1", "42.0", is_error=False)])
        events = _message_to_events(msg)
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "tool_result"
        assert ev["tool_use_id"] == "tu_1"
        assert ev["content"] == "42.0"
        assert ev["is_error"] is False

    def test_tool_result_error(self):
        msg = FakeAssistantMessage([FakeToolResultBlock("tu_2", "timeout", is_error=True)])
        events = _message_to_events(msg)
        assert events[0]["is_error"] is True

    def test_assistant_error(self):
        msg = FakeAssistantMessage([], error=SimpleNamespace(type="overloaded", message="busy"))
        events = _message_to_events(msg)
        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "API error" in events[0]["message"]

    def test_result_message(self):
        msg = FakeResultMessage(is_error=False, total_cost_usd=0.05, duration_ms=3000, num_turns=2)
        events = _message_to_events(msg)
        assert len(events) == 1
        ev = events[0]
        assert ev["type"] == "result"
        assert ev["is_error"] is False
        assert ev["total_cost_usd"] == 0.05
        assert ev["duration_ms"] == 3000
        assert ev["num_turns"] == 2

    def test_system_message(self):
        msg = FakeSystemMessage("init")
        events = _message_to_events(msg)
        assert len(events) == 1
        assert events[0] == {"type": "system", "subtype": "init"}

    def test_unknown_message_ignored(self):
        events = _message_to_events("something_unknown")
        assert events == []

    def test_multiple_blocks(self):
        msg = FakeAssistantMessage(
            [
                FakeThinkingBlock("think"),
                FakeTextBlock("answer"),
                FakeToolUseBlock("Read", "tu_x", {"file": "a.py"}),
            ]
        )
        events = _message_to_events(msg)
        assert len(events) == 3
        assert [e["type"] for e in events] == ["thinking", "text", "tool_use"]


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

        monkeypatch.setattr(
            "osprey.utils.shell_resolver._USER_BIN_CANDIDATES", [bin_dir]
        )
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
        captured_kwargs: dict = {}

        def capture_options(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()  # Return a mock ClaudeAgentOptions

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)

        with (
            patch("osprey.interfaces.web_terminal.operator_session.CLAUDE_SDK_AVAILABLE", True),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ClaudeAgentOptions",
                side_effect=capture_options,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ClaudeSDKClient",
                return_value=mock_client,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.validate_project_directory",
                return_value=[],
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.build_system_prompt",
                return_value={"type": "preset", "preset": "claude_code"},
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.get_facility_timezone",
                return_value=None,
            ),
        ):
            await session.start()

        assert captured_kwargs.get("setting_sources") == ["project"]
        await session.stop()

    @pytest.mark.asyncio
    async def test_start_requires_sdk(self):
        with patch("osprey.interfaces.web_terminal.operator_session.CLAUDE_SDK_AVAILABLE", False):
            session = OperatorSession(cwd="/tmp")
            with pytest.raises(RuntimeError, match="not installed"):
                await session.start()

    @pytest.mark.asyncio
    async def test_is_active_lifecycle(self):
        session = OperatorSession(cwd="/tmp")
        assert not session.is_active

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("osprey.interfaces.web_terminal.operator_session.CLAUDE_SDK_AVAILABLE", True),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ClaudeSDKClient",
                return_value=mock_client,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.build_system_prompt",
                return_value={"type": "preset", "preset": "claude_code"},
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.get_facility_timezone",
                return_value=None,
            ),
        ):
            await session.start()
            assert session.is_active

            await session.stop()
            assert not session.is_active

    @pytest.mark.asyncio
    async def test_send_prompt_queues_events(self):
        """Verify that send_prompt streams SDK messages into the queue."""
        session = OperatorSession(cwd="/tmp")

        # Build a mock client whose receive_response yields our fakes
        fake_messages = [
            FakeAssistantMessage([FakeTextBlock("hello")]),
            FakeResultMessage(),
        ]

        async def fake_receive():
            for m in fake_messages:
                yield m

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.query = AsyncMock()
        mock_client.receive_response = fake_receive

        with (
            patch("osprey.interfaces.web_terminal.operator_session.CLAUDE_SDK_AVAILABLE", True),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ClaudeSDKClient",
                return_value=mock_client,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.AssistantMessage",
                FakeAssistantMessage,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.ResultMessage",
                FakeResultMessage,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.TextBlock",
                FakeTextBlock,
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.build_system_prompt",
                return_value={"type": "preset", "preset": "claude_code"},
            ),
            patch(
                "osprey.interfaces.web_terminal.operator_session.get_facility_timezone",
                return_value=None,
            ),
        ):
            await session.start()
            await session.send_prompt("test")

            # Wait for the response task to complete
            await session._response_task

            events = []
            while not session._queue.empty():
                events.append(session._queue.get_nowait())

            assert len(events) == 2
            assert events[0]["type"] == "text"
            assert events[1]["type"] == "result"

            await session.stop()


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

    @pytest.mark.asyncio
    async def test_cleanup_all(self):
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
            await registry.create_session("a", cwd="/tmp")
            await registry.create_session("b", cwd="/tmp")

        await registry.cleanup_all()
        assert registry.get_session("a") is None
        assert registry.get_session("b") is None
        mock_s1.stop.assert_awaited()
        mock_s2.stop.assert_awaited()


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

    def test_all_files_missing(self, tmp_path):
        warnings = validate_project_directory(str(tmp_path))
        assert len(warnings) == 4
        assert any(".mcp.json" in w for w in warnings)
        assert any("CLAUDE.md" in w for w in warnings)
        assert any(".claude" in w for w in warnings)
        assert any("config.yml" in w for w in warnings)

    def test_partial_files(self, tmp_path):
        (tmp_path / "config.yml").touch()
        (tmp_path / ".claude").mkdir()

        warnings = validate_project_directory(str(tmp_path))
        assert len(warnings) == 2
        assert any(".mcp.json" in w for w in warnings)
        assert any("CLAUDE.md" in w for w in warnings)


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

    def test_skips_when_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        env = build_clean_env(project_cwd=str(tmp_path))
        assert "OSPREY_CONFIG" not in env

    def test_does_not_override_existing_osprey_config(self, tmp_path, monkeypatch):
        (tmp_path / "config.yml").touch()
        monkeypatch.setenv("OSPREY_CONFIG", "/custom/config.yml")

        env = build_clean_env(project_cwd=str(tmp_path))
        assert env["OSPREY_CONFIG"] == "/custom/config.yml"

    def test_no_project_cwd_is_noop(self, monkeypatch):
        monkeypatch.delenv("OSPREY_CONFIG", raising=False)

        env = build_clean_env()
        assert "OSPREY_CONFIG" not in env

        env2 = build_clean_env(project_cwd=None)
        assert "OSPREY_CONFIG" not in env2
