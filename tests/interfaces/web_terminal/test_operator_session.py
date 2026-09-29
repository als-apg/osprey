"""Tests for operator session management."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    CLIConnectionError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from osprey.agent_runner import (
    ApiErrorEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
)
from osprey.agent_runner.clean_env import build_clean_env
from osprey.interfaces.web_terminal.chat_session_pool import ChatCapacityError
from osprey.interfaces.web_terminal.operator_session import (
    OperatorRegistry,
    OperatorSession,
    TurnInProgressError,
    _event_to_wire,
    _format_tool_name,
    validate_project_directory,
)

# ---------------------------------------------------------------------------
# Helpers — real SDK messages, and the seam a started session connects through
# ---------------------------------------------------------------------------

_OS = "osprey.interfaces.web_terminal.operator_session."
#: The name the agent runner constructs its client under.
_CLIENT = "osprey.agent_runner.session.ClaudeSDKClient"
PRESET = {"type": "preset", "preset": "claude_code"}


def assistant_message(blocks, *, error=None) -> AssistantMessage:
    return AssistantMessage(content=list(blocks), model="claude-test", error=error)


def user_message(blocks) -> UserMessage:
    return UserMessage(content=list(blocks))


def result_message(
    *,
    is_error: bool = False,
    total_cost_usd: float = 0.01,
    duration_ms: int = 1200,
    num_turns: int = 1,
) -> ResultMessage:
    return ResultMessage(
        subtype="error_during_execution" if is_error else "success",
        duration_ms=duration_ms,
        duration_api_ms=duration_ms,
        is_error=is_error,
        num_turns=num_turns,
        session_id="sdk-session",
        total_cost_usd=total_cost_usd,
    )


def system_message(subtype: str = "init", data: dict | None = None) -> SystemMessage:
    return SystemMessage(subtype=subtype, data=data or {})


@contextlib.contextmanager
def _sdk_seam(client, captured: list | None = None):
    """Patch the runner's client so ``start()`` connects *client* and nothing else.

    Every ``ClaudeAgentOptions`` the client is constructed with is appended to
    *captured*.
    """

    def construct(**kwargs):
        if captured is not None:
            captured.append(kwargs["options"])
        return client

    with (
        patch(_OS + "HAS_SDK", True),
        patch(_CLIENT, side_effect=construct),
        patch(_OS + "validate_project_directory", return_value=[]),
        patch(_OS + "build_system_prompt", return_value=PRESET),
        patch(_OS + "get_facility_timezone", return_value=None),
    ):
        yield


def _mock_client() -> AsyncMock:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


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

    def test_underscored_server_name(self):
        """Framework servers with underscores in the name (osprey_workspace,
        osprey_facility_knowledge) must strip cleanly too."""
        assert _format_tool_name("mcp__osprey_workspace__submit_response") == "Submit Response"
        assert (
            _format_tool_name("mcp__osprey_facility_knowledge__resolve_channel")
            == "Resolve Channel"
        )

    def test_underscored_facility_custom_server(self):
        """A facility-declared server name OSPREY never saw at authoring time."""
        assert _format_tool_name("mcp__als_custom_srv__do_thing") == "Do Thing"


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
        assert ev == {"type": "text", "content": "hello"}

    def test_thinking_block(self):
        ev = _event_to_wire(ThinkingEvent(text="pondering..."))
        assert ev == {"type": "thinking", "content": "pondering..."}

    def test_tool_use_block(self):
        ev = _event_to_wire(_tool_use("mcp__osprey__channel_read", "tu_1", {"channels": ["X"]}))
        assert ev is not None
        assert ev["type"] == "tool_use"
        assert ev["tool_name"] == "Channel Read"
        assert ev["tool_name_raw"] == "mcp__osprey__channel_read"
        assert ev["tool_use_id"] == "tu_1"
        assert ev["input"] == {"channels": ["X"]}

    @pytest.mark.parametrize("is_error", [False, True])
    def test_tool_results_are_not_part_of_the_stream(self, is_error):
        event = ToolResultEvent(
            tool_use_id="tu_1", content="42.0", is_error=is_error, parent_tool_use_id=None
        )
        assert _event_to_wire(event) is None

    def test_assistant_error(self):
        ev = _event_to_wire(ApiErrorEvent(error="overloaded"))
        assert ev is not None
        assert ev["type"] == "error"
        assert "API error" in ev["message"]

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

    def test_system_message(self):
        ev = _event_to_wire(SystemEvent(subtype="init", data={}))
        assert ev == {"type": "system", "subtype": "init"}

    def test_frames_keep_their_key_order(self):
        frames = [
            _event_to_wire(ThinkingEvent(text="think")),
            _event_to_wire(TextEvent(text="answer", parent_tool_use_id=None)),
            _event_to_wire(_tool_use("Read", "tu_x", {"file": "a.py"})),
        ]
        assert [list(f) for f in frames if f is not None] == [
            ["type", "content"],
            ["type", "content"],
            ["type", "tool_name", "tool_name_raw", "tool_use_id", "input"],
        ]


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
        assert ev == {"type": "system", "subtype": "compact_boundary"}


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
        captured: list = []

        with _sdk_seam(_mock_client(), captured):
            await session.start()

        assert captured[0].setting_sources == ["project"]
        await session.stop()

    @pytest.mark.asyncio
    async def test_start_marks_simple_web_surface_in_env(self):
        """The operator chat IS the simple web UX — its sessions carry
        OSPREY_WEB_UX=simple (alongside the telemetry vars) so the
        panels-context SessionStart hook can tell the agent which UI the
        operator is looking at. Injection follows the telemetry pattern: only
        when an env dict was provided (None means inherit the process env)."""
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"})
        captured: list = []

        with _sdk_seam(_mock_client(), captured):
            await session.start()

        env = captured[0].env
        assert env is not None
        assert env["OSPREY_WEB_UX"] == "simple"
        assert "OSPREY_TELEMETRY_SESSION_ID" in env
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

        with _sdk_seam(_mock_client()):
            await session.start()
            assert session.is_active

            await session.stop()
            assert not session.is_active

    @pytest.mark.asyncio
    async def test_send_prompt_queues_events(self):
        """Verify that send_prompt streams SDK messages into the queue."""
        session = OperatorSession(cwd="/tmp")

        # Build a mock client whose receive_response yields real messages
        fake_messages = [
            assistant_message([TextBlock("hello")]),
            result_message(),
        ]

        async def fake_receive():
            for m in fake_messages:
                yield m

        mock_client = _mock_client()
        mock_client.query = AsyncMock()
        mock_client.receive_response = fake_receive

        with _sdk_seam(mock_client):
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
# OperatorSession start identity: resume a transcript, or open one
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _capture_start_options(captured: list):
    """Patch the runner's client and record the ``ClaudeAgentOptions`` it gets.

    Hoisted so the identity tests read as the one assertion each is making.
    """
    with _sdk_seam(_mock_client(), captured):
        yield


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
        captured: list = []

        with _capture_start_options(captured):
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
        captured: list = []

        with _capture_start_options(captured):
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
        captured: list = []

        with _capture_start_options(captured):
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
        fresh: list = []
        resumed: list = []

        with _capture_start_options(fresh):
            await OperatorSession(
                cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=self.KEY
            ).start()
        with _capture_start_options(resumed):
            await OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=self.KEY).start(
                resume_id="transcript-9"
            )

        assert fresh[0].env["OSPREY_TELEMETRY_SESSION_ID"] == self.KEY
        assert resumed[0].env["OSPREY_TELEMETRY_SESSION_ID"] == self.KEY

    @pytest.mark.asyncio
    async def test_a_session_given_no_key_holds_one_stable_across_starts(self):
        """The key belongs to the session object, not to a single start."""
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"})
        first: list = []
        second: list = []

        with _capture_start_options(first):
            await session.start()
        with _capture_start_options(second):
            await session.start()

        minted = first[0].env["OSPREY_TELEMETRY_SESSION_ID"]
        assert minted
        assert second[0].env["OSPREY_TELEMETRY_SESSION_ID"] == minted
        assert second[0].session_id == minted

    @pytest.mark.asyncio
    async def test_two_keyless_sessions_do_not_share_an_identity(self):
        first: list = []
        second: list = []

        with _capture_start_options(first):
            await OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}).start()
        with _capture_start_options(second):
            await OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}).start()

        assert first[0].session_id != second[0].session_id


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

    async def query(self, _prompt):
        self.query_calls += 1

    async def interrupt(self):
        self.interrupt_calls += 1
        self._interrupted.set()

    async def receive_response(self):
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
    with _sdk_seam(client):
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

    def __init__(self, cwd: str = "/tmp", env=None, session_key=None):
        self.cwd = cwd
        self.env = env
        self.session_key = session_key
        self.resume_id = None
        self.is_active = True
        self.last_activity = time.monotonic()
        self.in_flight = False
        self._response_task = None
        self._quiesce_task = None
        self.start_calls = 0
        self.stop_calls = 0
        self.start_delay = 0.0
        self.start_error: Exception | None = None

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
    client = _mock_client()
    client._transport = SimpleNamespace(_process=process)
    return client


async def _started_over(process: FakeChildProcess) -> OperatorSession:
    session = OperatorSession(cwd="/tmp")
    with _sdk_seam(_client_over(process)):
        await session.start()
    return session


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

    def factory(cwd=None, env=None, session_key=None):
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

        def factory(cwd=None, env=None, session_key=None):
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


# ---------------------------------------------------------------------------
# Parity pins: launch options, stream frames, child handling
# ---------------------------------------------------------------------------

_KEY = TestOperatorSessionResumeOptions.KEY
_PRIMITIVES = "osprey.agent_runner.primitives."


class _ScriptedClient:
    """A client whose one response is *messages*, or *error* raised mid-turn."""

    def __init__(self, messages=(), *, error: Exception | None = None) -> None:
        self.messages = list(messages)
        self.error = error
        self.interrupt_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, _prompt):
        return None

    async def interrupt(self):
        self.interrupt_calls += 1

    async def receive_response(self):
        for message in self.messages:
            yield message
        if self.error is not None:
            raise self.error


async def _drain(session: OperatorSession) -> list[dict]:
    """Run the turn's reader to its end and return every frame it queued."""
    assert session._response_task is not None
    await asyncio.wait_for(session._response_task, timeout=2.0)
    frames = []
    while not session._queue.empty():
        frames.append(session._queue.get_nowait())
    return frames


class TestOperatorChatLaunchParity:
    """The options the chat launches with are the ones it always built by hand."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("shape", ["resume", "posture_key", "foreign_key"])
    async def test_the_launch_options_equal_the_hand_built_ones(self, shape):
        key = "e2e" if shape == "foreign_key" else _KEY
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=key)
        captured: list = []

        with _sdk_seam(_mock_client(), captured):
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
            system_prompt=PRESET,
            cwd="/tmp",
            env=options.env,
            setting_sources=["project"],
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
            patch(_PRIMITIVES + "start_proxy", side_effect=refuse),
            patch(_PRIMITIVES + "await_mcp_ready", side_effect=refuse),
            _sdk_seam(_mock_client()),
        ):
            await session.start()

        assert session.is_active
        await session.stop()

    @pytest.mark.asyncio
    async def test_no_permission_callback_hook_or_bypass_reaches_the_child(self):
        session = OperatorSession(cwd="/tmp", env={"PATH": "/usr/bin"}, session_key=_KEY)
        captured: list = []

        with _sdk_seam(_mock_client(), captured):
            await session.start()

        (options,) = captured
        assert options.can_use_tool is None
        assert options.hooks is None
        assert options.permission_mode is None
        await session.stop()

    @pytest.mark.asyncio
    async def test_a_session_built_without_an_env_inherits_the_process_env(self):
        session = OperatorSession(cwd="/tmp", session_key=_KEY)
        captured: list = []

        with (
            patch(_PRIMITIVES + "sdk_env") as sdk_env,
            _sdk_seam(_mock_client(), captured),
        ):
            await session.start()

        (options,) = captured
        assert options.env == {}
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
            await session.send_prompt("read it")
            frames = await _drain(session)

        assert frames == [
            {
                "type": "error",
                "message": "API error: rate_limit",
                "error_type": "AssistantMessageError",
            },
            {"type": "thinking", "content": "pondering"},
            {
                "type": "tool_use",
                "tool_name": "Channel Read",
                "tool_name_raw": tool,
                "tool_use_id": "tu_1",
                "input": {"channel": "SR:BPM"},
            },
            {"type": "text", "content": "done"},
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
            await session.send_prompt("x")
            frames = await _drain(session)

        assert frames[-1] == {
            "type": "error",
            "message": "gone",
            "error_type": "CLIConnectionError",
        }

    @pytest.mark.asyncio
    async def test_an_unexpected_failure_mid_turn_keeps_its_prefix(self):
        client = _ScriptedClient(error=ValueError("x"))
        async with _started_session(client) as session:
            await session.send_prompt("x")
            frames = await _drain(session)

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
