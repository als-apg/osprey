"""Fakes shared by the web-terminal session tests.

``FakePtySession`` is the PTY the registry hands out when ``_spawn_session``
is patched: alive until told otherwise, silent unless ``emit`` queues bytes.
``FakeClock``, ``FakeChatSession`` and ``FakeChatPool`` are the slice of the
hand-off door's collaborators that its phase tests drive.

``PoolChatSession`` is the ``OperatorSession`` double the *real*
``ChatSessionPool`` drives, and ``created_first`` waits for a pool's factory
to build and start its first one. ``assistant_message``, ``user_message``,
``result_message`` and ``system_message`` build the real Claude SDK messages a
fake client yields, and ``sdk_seam`` patches the SDK boundary under
``operator_session`` so ``OperatorSession.start`` connects a test's client
through the agent runner and nothing else.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    UserMessage,
)


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


#: The system prompt ``sdk_seam`` has ``operator_session`` build.
PRESET_SYSTEM_PROMPT = {"type": "preset", "preset": "claude_code"}


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


def mock_sdk_client() -> AsyncMock:
    """An ``AsyncMock`` client whose context manager enters and exits cleanly."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@contextlib.contextmanager
def sdk_seam(client_or_factory: Any = None) -> Iterator[list[ClaudeAgentOptions]]:
    """Patch the SDK boundary under ``operator_session``; yield the captured options.

    Inside the block the SDK reads as available, the ``ClaudeSDKClient`` the
    agent runner constructs answers the test's client, and the project check,
    system prompt (``PRESET_SYSTEM_PROMPT``) and facility timezone are fixed.
    The yielded list fills with the ``ClaudeAgentOptions`` of every client
    constructed inside the block.

    Args:
        client_or_factory: A class or plain function is a factory, called with
            what ``ClaudeSDKClient`` is called with; anything else is the client
            itself. ``None`` is a ``mock_sdk_client()``.
    """
    seam = "osprey.interfaces.web_terminal.operator_session."
    captured: list[ClaudeAgentOptions] = []

    if client_or_factory is None:
        client_or_factory = mock_sdk_client()
    is_factory = inspect.isclass(client_or_factory) or inspect.isfunction(client_or_factory)

    def construct(**kwargs: Any) -> Any:
        captured.append(kwargs["options"])
        return client_or_factory(**kwargs) if is_factory else client_or_factory

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch(seam + "HAS_SDK", True))
        stack.enter_context(
            patch("osprey.agent_runner.session.ClaudeSDKClient", side_effect=construct)
        )
        stack.enter_context(patch(seam + "validate_project_directory", return_value=[]))
        stack.enter_context(patch(seam + "build_system_prompt", return_value=PRESET_SYSTEM_PROMPT))
        stack.enter_context(patch(seam + "get_facility_timezone", return_value=None))
        yield captured
