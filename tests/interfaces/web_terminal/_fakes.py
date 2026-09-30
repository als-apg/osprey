"""Fakes shared by the web-terminal session tests.

``FakePtySession`` is the PTY the registry hands out when ``_spawn_session``
is patched: alive until told otherwise, silent unless ``emit`` queues bytes.
``FakeClock``, ``FakeChatSession`` and ``FakeChatPool`` are the slice of the
hand-off door's collaborators that its phase tests drive.

``PoolChatSession`` is the ``OperatorSession`` double the *real*
``ChatSessionPool`` drives, and ``created_first`` waits for a pool's factory
to build and start its first one. The ``Fake*Block`` and ``Fake*Message``
classes stand in for the Claude SDK's message types, and ``sdk_seam`` patches
the whole SDK boundary of ``operator_session`` so ``OperatorSession.start``
connects a test's client and nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


class FakePtySession:
    """Minimal PtySession substitute — alive until told otherwise.

    ``emit`` queues bytes the output loop will forward; ``exit`` ends the
    child with a code. Queued output is drained before the loop sees the exit,
    which is what a real PTY does too (:meth:`PtySession.read_output`).
    """

    def __init__(self):
        self._alive = True
        self._exit_code: int | None = None
        self._chunks: list[bytes] = []
        self._last_rows = 24
        self._last_cols = 80
        self._command_list = ["fake"]

    @property
    def is_alive(self):
        return self._alive

    @property
    def exit_code(self):
        if self._alive:
            return None
        return 0 if self._exit_code is None else self._exit_code

    # ``PtySession.start``'s signature: the manager names every argument it passes.
    def start(self, initial_rows=24, initial_cols=80, extra_env=None, cwd=None):  # noqa: ARG002
        self._last_rows = initial_rows
        self._last_cols = initial_cols

    def resize(self, rows, cols):
        self._last_rows = rows
        self._last_cols = cols

    def write_input(self, data):
        pass

    def terminate(self):
        self._alive = False

    def emit(self, data: bytes) -> None:
        self._chunks.append(data)

    def exit(self, code: int) -> None:
        self._exit_code = code
        self._alive = False

    async def read_output(self):
        try:
            while self._alive or self._chunks:
                if self._chunks:
                    yield self._chunks.pop(0)
                else:
                    await asyncio.sleep(0.05)
        except (asyncio.CancelledError, GeneratorExit):
            return


class FakeClock:
    """A monotonic clock that only moves when ``sleep`` is awaited."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self.on_sleep: list = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        for hook in list(self.on_sleep):
            hook(len(self.sleeps))
        # Yield once so a concurrent task can run between looks.
        await asyncio.sleep(0)


class FakeChatSession:
    """The slice of ``OperatorSession`` the pools and phase (a) read."""

    def __init__(self, *, active: bool = True, busy: bool = False) -> None:
        self._active = active
        self.is_busy = busy
        self.process_exited: bool | None = None
        self.teardowns = 0

    @property
    def is_active(self) -> bool:
        return self._active

    async def teardown(self) -> None:
        self.teardowns += 1
        self._active = False
        self.process_exited = True


class FakeChatPool:
    """``ChatSessionPool`` as phase (a) sees it: ``get``, ``has_key``, ``terminate``."""

    def __init__(self) -> None:
        self.sessions: dict[str, FakeChatSession] = {}
        self.starting: set[str] = set()
        self.terminated: list[str] = []
        self.get_calls = 0
        # When set, ``terminate`` waits on it — used to hold the lock open.
        self.terminate_gate: asyncio.Event | None = None

    def get(self, chat_id: str) -> FakeChatSession | None:
        self.get_calls += 1
        return self.sessions.get(chat_id)

    def has_key(self, chat_id: str) -> bool:
        return chat_id in self.sessions or chat_id in self.starting

    async def terminate(self, chat_id: str) -> FakeChatSession | None:
        if self.terminate_gate is not None:
            await self.terminate_gate.wait()
        self.terminated.append(chat_id)
        session = self.sessions.pop(chat_id, None)
        if session is not None:
            await session.teardown()
        return session


class PoolChatSession:
    """An ``OperatorSession`` double the real ``ChatSessionPool`` can drive.

    The surface the pool asks for (``start``/``is_active``/``is_busy``/
    ``last_activity``/``teardown``), ``acquire_turn`` for the turn-guard
    callers, the ``process_exited`` the hand-off's death check reads, and the
    launch facts worth asserting: the ``session_key`` it was built under and
    the ``resume_id`` it was started on. ``start_delay`` holds ``start`` open
    after ``started`` is set, so a test can act while a creation is in flight.
    """

    def __init__(self, cwd: str = "/tmp", env=None, session_key: str | None = None) -> None:
        self.cwd = cwd
        self.env = env
        self.session_key = session_key
        self.resume_id: str | None = None
        self.is_active = True
        self.in_flight = False
        self.last_activity = time.monotonic()
        self.process_exited: bool | None = False
        self.start_calls = 0
        self.stop_calls = 0
        self.turns = 0
        self.start_delay = 0.0
        self.started = asyncio.Event()

    @property
    def is_busy(self) -> bool:
        return self.in_flight

    async def start(self, *, resume_id: str | None = None) -> None:
        self.resume_id = resume_id
        self.started.set()
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        self.start_calls += 1

    def acquire_turn(self) -> int:
        self.turns += 1
        return self.turns

    async def teardown(self) -> None:
        self.stop_calls += 1
        self.is_active = False
        self.process_exited = True


async def created_first(created: list, timeout: float = 1.0) -> None:
    """Wait until a pool's factory has built its first session and entered ``start()``.

    Waiting on the session's own ``started`` event rather than a sleep keeps a
    race against a creation in flight deterministic.
    """
    deadline = time.monotonic() + timeout
    while not created:
        if time.monotonic() > deadline:  # pragma: no cover - guards a hang
            raise AssertionError("factory never ran")
        await asyncio.sleep(0)
    await asyncio.wait_for(created[0].started.wait(), timeout=timeout)


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

    def __init__(self, subtype: str = "init", data: dict | None = None):
        self.subtype = subtype
        self.data = data or {}


#: The SDK names ``operator_session`` binds at import, and the double for each.
_SDK_TYPES = {
    "AssistantMessage": FakeAssistantMessage,
    "ResultMessage": FakeResultMessage,
    "SystemMessage": FakeSystemMessage,
    "TextBlock": FakeTextBlock,
    "ThinkingBlock": FakeThinkingBlock,
    "ToolUseBlock": FakeToolUseBlock,
    "ToolResultBlock": FakeToolResultBlock,
}


@contextlib.contextmanager
def sdk_seam(client_or_factory: Any = None) -> Iterator[dict[str, Any]]:
    """Patch ``operator_session``'s SDK boundary; yield the captured options kwargs.

    Inside the block the SDK reads as available, ``ClaudeSDKClient`` answers
    the test's client, the seven SDK message and block types are the ``Fake*``
    doubles above (so ``isinstance`` inside ``_message_to_events`` matches what
    a fake client yields), and the project check, system prompt and facility
    timezone are fixed. The yielded dict fills with the keyword arguments of
    every ``ClaudeAgentOptions`` built inside the block.

    Args:
        client_or_factory: A class or plain function is a factory, called with
            what ``ClaudeSDKClient`` is called with; anything else is the client
            itself. ``None`` is an ``AsyncMock`` client whose context manager
            enters and exits cleanly.
    """
    seam = "osprey.interfaces.web_terminal.operator_session."
    captured: dict[str, Any] = {}

    def capture_options(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    if client_or_factory is None:
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client_patch = patch(seam + "ClaudeSDKClient", return_value=client)
    elif inspect.isclass(client_or_factory) or inspect.isfunction(client_or_factory):
        client_patch = patch(seam + "ClaudeSDKClient", side_effect=client_or_factory)
    else:
        client_patch = patch(seam + "ClaudeSDKClient", return_value=client_or_factory)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(seam + "CLAUDE_SDK_AVAILABLE", True))
        stack.enter_context(patch(seam + "ClaudeAgentOptions", side_effect=capture_options))
        stack.enter_context(client_patch)
        for name, double in _SDK_TYPES.items():
            stack.enter_context(patch(seam + name, double))
        stack.enter_context(patch(seam + "validate_project_directory", return_value=[]))
        stack.enter_context(
            patch(
                seam + "build_system_prompt",
                return_value={"type": "preset", "preset": "claude_code"},
            )
        )
        stack.enter_context(patch(seam + "get_facility_timezone", return_value=None))
        yield captured
