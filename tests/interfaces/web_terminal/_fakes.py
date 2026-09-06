"""Fakes shared by the web-terminal session tests.

``FakePtySession`` is the PTY the registry hands out when ``_spawn_session``
is patched: alive until told otherwise, silent unless ``emit`` queues bytes.
``FakeClock``, ``FakeChatSession`` and ``FakeChatPool`` are the slice of the
hand-off door's collaborators that its phase tests drive.
"""

from __future__ import annotations

import asyncio


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

    def start(self, initial_rows=24, initial_cols=80, extra_env=None, cwd=None):
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
