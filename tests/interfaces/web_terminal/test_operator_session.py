"""Tests for operator session management."""

from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.interfaces.web_terminal.chat_session_pool import ChatCapacityError
from osprey.interfaces.web_terminal.operator_session import (
    OperatorRegistry,
    OperatorSession,
    TurnInProgressError,
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

        monkeypatch.setattr("osprey.utils.shell_resolver._USER_BIN_CANDIDATES", [bin_dir])
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
# OperatorSession per-turn epoch guard
# ---------------------------------------------------------------------------


class TestTurnGuardEpoch:
    def test_fresh_session_is_not_in_flight(self):
        session = OperatorSession(cwd="/tmp")
        assert session.in_flight is False

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
# OperatorSession.cancel() / spawn_quiesce() / last_activity
# ---------------------------------------------------------------------------


class FakeStreamClient:
    """Controllable fake ``ClaudeSDKClient`` for cancel/quiesce tests.

    ``interrupt`` is a coroutine so that a caller which forgets to ``await`` it
    (the historical bug) never runs its body — letting a test assert the await
    happened via ``interrupt_calls``.
    """

    def __init__(self, *, hang: bool = False) -> None:
        self.hang = hang
        self.interrupt_calls = 0
        self.query_calls = 0
        self._interrupted = asyncio.Event()
        # Set once the reader has yielded its first (partial) message, so a
        # test can be sure the turn is genuinely in-flight before cancelling.
        self.first_yielded = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.query_calls += 1

    async def interrupt(self):
        self.interrupt_calls += 1
        self._interrupted.set()

    async def receive_response(self):
        yield FakeAssistantMessage([FakeTextBlock("partial")])
        self.first_yielded.set()
        if self.hang:
            # Never terminate on its own — only a hard cancel stops us.
            while True:
                await asyncio.sleep(3600)
        else:
            # Drain toward a terminal message once interrupted.
            await self._interrupted.wait()
            await asyncio.sleep(0.001)
            yield FakeResultMessage()


@contextlib.asynccontextmanager
async def _started_session(client):
    """Yield a started ``OperatorSession`` wired to ``client``."""
    session = OperatorSession(cwd="/tmp")
    with (
        patch("osprey.interfaces.web_terminal.operator_session.CLAUDE_SDK_AVAILABLE", True),
        patch(
            "osprey.interfaces.web_terminal.operator_session.ClaudeSDKClient",
            return_value=client,
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
        try:
            yield session
        finally:
            await session.stop()


class TestOperatorSessionCancel:
    @pytest.mark.asyncio
    async def test_idle_cancel_is_noop_no_interrupt(self):
        """No in-flight turn (never sent a prompt) → short-circuit, no hang."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            assert session._response_task is None
            await asyncio.wait_for(session.cancel(), timeout=1.0)
            # Short-circuits before touching the client.
            assert client.interrupt_calls == 0

    @pytest.mark.asyncio
    async def test_cancel_noop_when_reader_already_done(self):
        """A completed turn's reader → cancel short-circuits without interrupt."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            await session.send_prompt("hi")
            client._interrupted.set()  # let the reader reach its terminal message
            await asyncio.wait_for(session._response_task, timeout=1.0)
            assert session._response_task.done()

            await asyncio.wait_for(session.cancel(), timeout=1.0)
            assert client.interrupt_calls == 0

    @pytest.mark.asyncio
    async def test_cancel_awaits_interrupt_and_drains(self):
        """In-flight turn: interrupt is awaited FIRST, then reader drains done."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            await session.send_prompt("hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)
            assert not session._response_task.done()

            await asyncio.wait_for(session.cancel(), timeout=2.0)

            # interrupt ran exactly once (proves it was awaited, not discarded).
            assert client.interrupt_calls == 1
            assert session._response_task.done()
            # Drained cleanly to a terminal message — not force-cancelled.
            assert not session._response_task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_hard_cancels_on_drain_timeout(self):
        """A reader that never terminates is hard-cancelled after the bound."""
        client = FakeStreamClient(hang=True)
        async with _started_session(client) as session:
            await session.send_prompt("hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            with patch(
                "osprey.interfaces.web_terminal.operator_session._QUIESCE_TIMEOUT_S",
                0.05,
            ):
                await asyncio.wait_for(session.cancel(), timeout=2.0)

            assert client.interrupt_calls == 1
            assert session._response_task.done()
            assert session._response_task.cancelled()

    @pytest.mark.asyncio
    async def test_double_cancel_is_idempotent(self):
        """Cancelling twice: the second call short-circuits (task done)."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            await session.send_prompt("hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            await asyncio.wait_for(session.cancel(), timeout=2.0)
            assert client.interrupt_calls == 1

            # Second cancel is a no-op — reader already done.
            await asyncio.wait_for(session.cancel(), timeout=1.0)
            assert client.interrupt_calls == 1

    @pytest.mark.asyncio
    async def test_spawn_quiesce_returns_stored_detached_task(self):
        """spawn_quiesce returns a task, stores it, and quiesces when awaited."""
        client = FakeStreamClient()
        async with _started_session(client) as session:
            await session.send_prompt("hi")
            await asyncio.wait_for(client.first_yielded.wait(), timeout=1.0)

            task = session.spawn_quiesce()
            assert isinstance(task, asyncio.Task)
            assert session._quiesce_task is task

            await asyncio.wait_for(task, timeout=2.0)
            assert client.interrupt_calls == 1
            assert session._response_task.done()

    @pytest.mark.asyncio
    async def test_last_activity_initialized_at_creation(self):
        session = OperatorSession(cwd="/tmp")
        assert isinstance(session.last_activity, float)

    @pytest.mark.asyncio
    async def test_last_activity_restamped_on_turn_completion(self):
        client = FakeStreamClient()
        async with _started_session(client) as session:
            before = session.last_activity
            await session.send_prompt("hi")
            client._interrupted.set()  # let the reader reach its terminal message
            await asyncio.wait_for(session._response_task, timeout=1.0)
            assert session.last_activity > before


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
# OperatorRegistry chat pool
# ---------------------------------------------------------------------------


class FakeTask:
    """Minimal stand-in for an asyncio.Task with a controllable done() state."""

    def __init__(self, done: bool = False):
        self._done = done

    def done(self) -> bool:
        return self._done


class FakeChatSession:
    """Lightweight OperatorSession double for registry pool tests."""

    def __init__(self, cwd: str = "/tmp", env=None):
        self.cwd = cwd
        self.env = env
        self.is_active = True
        self.last_activity = time.monotonic()
        self.in_flight = False
        self._response_task = None
        self._quiesce_task = None
        self.start_calls = 0
        self.stop_calls = 0
        self.start_delay = 0.0
        self.start_error: Exception | None = None

    async def start(self):
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
        handler_running = self._response_task is not None and not self._response_task.done()
        quiesce_running = self._quiesce_task is not None and not self._quiesce_task.done()
        return self.in_flight and (handler_running or quiesce_running)

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

    def test_guard_held_with_running_reader_is_busy(self):
        session = OperatorSession(cwd="/tmp")
        session.acquire_turn()
        session._response_task = FakeTask(done=False)
        assert session.is_busy is True

    def test_guard_held_with_running_quiesce_is_busy(self):
        session = OperatorSession(cwd="/tmp")
        session.acquire_turn()
        session._response_task = FakeTask(done=True)
        session._quiesce_task = FakeTask(done=False)
        assert session.is_busy is True

    def test_zombie_guard_held_all_tasks_done_is_not_busy(self):
        session = OperatorSession(cwd="/tmp")
        session.acquire_turn()
        session._response_task = FakeTask(done=True)
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


def _session_factory(start_delay: float = 0.0):
    """Return a side_effect callable that builds FakeChatSessions and records them."""
    created: list[FakeChatSession] = []

    def factory(cwd=None, env=None):
        s = FakeChatSession(cwd=cwd, env=env)
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
    async def test_reuse_returns_same_live_session(self):
        registry = OperatorRegistry()
        factory = _session_factory()
        with _patch_session(factory):
            s1, r1 = await registry.get_or_create_chat_session("a", cwd="/tmp")
            s2, r2 = await registry.get_or_create_chat_session("a", cwd="/tmp")

        assert s1 is s2
        assert r1 is False
        assert r2 is True
        assert len(factory.created) == 1
        assert s1.start_calls == 1  # not restarted

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
            # Genuinely busy: guard held and reader still running.
            a.in_flight = True
            a._response_task = FakeTask(done=False)

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
            a._response_task = FakeTask(done=True)
            a._quiesce_task = FakeTask(done=True)

            b, r = await registry.get_or_create_chat_session("b", cwd="/tmp")

        assert r is False
        assert registry.get_chat_session("a") is None
        assert a.stop_calls == 1
        assert registry.get_chat_session("b") is b

    @pytest.mark.asyncio
    async def test_terminate_chat_session(self):
        registry = OperatorRegistry()
        factory = _session_factory()
        with _patch_session(factory):
            a, _ = await registry.get_or_create_chat_session("a", cwd="/tmp")

        await registry.terminate_chat_session("a")
        assert registry.get_chat_session("a") is None
        assert a.stop_calls == 1
        # Terminating a missing chat is a no-op.
        await registry.terminate_chat_session("a")

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
        c._response_task = FakeTask(done=False)

        reaped = await registry.reap_idle_chat_sessions()
        assert reaped == 1
        assert registry.get_chat_session("a") is None
        assert a.stop_calls == 1
        assert registry.get_chat_session("b") is b
        assert registry.get_chat_session("c") is c

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

        def factory(cwd=None, env=None):
            s = FakeChatSession(cwd=cwd, env=env)
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
