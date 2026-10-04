"""Shared harness for the hand-off door's phase and invariant suites.

``acquire_surface`` is driven here on a ``SimpleNamespace`` app: a real
``PtyRegistry``, a fake chat pool, a ``HandoffState`` on a fake clock, and the
turn-state and transcript stores phase (b) reads, present and empty. This
module centralizes that app, the fake PTYs and chats the phases act on, the
spawn callbacks phase (c) calls, the transcript builders the idle wait reads,
the pending-slot assertions, and the one ``disk`` fixture that stands in for
the transcripts on disk. The fakes it builds on live in ``_fakes.py``.

:func:`acquire` is the door as a test calls it when the outcome must not
start a process: it passes :func:`must_not_spawn` unless the test hands in a
spawn callback of its own.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal import session_handoff
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.session_handoff import (
    AcquirePlan,
    AcquireResult,
    HandoffState,
    SpawnCallback,
    SpawnRequest,
    Surface,
    WaitOutcome,
    acquire_surface,
    get_state,
)
from tests.interfaces.web_terminal._fakes import (
    FakeChatPool,
    FakeChatSession,
    FakeClock,
    FakePtySession,
)

KEY = "11111111-2222-3333-4444-555555555555"
OTHER_KEY = "22222222-3333-4444-5555-666666666666"
THIRD_KEY = "33333333-4444-5555-6666-777777777777"
TRANSCRIPT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingPty(FakePtySession):
    """The fake PTY, remembering what the hand-off does to it."""

    def __init__(self, *, terminate_blocks_s: float = 0.0) -> None:
        super().__init__()
        self.writes: list[bytes] = []
        self.terminates = 0
        self._terminate_blocks_s = terminate_blocks_s

    def write_input(self, data: bytes) -> None:
        self.writes.append(data)

    def terminate(self) -> None:
        self.terminates += 1
        if self._terminate_blocks_s:
            # A real terminate blocks its thread for seconds; this one for a
            # little, so a test can watch the loop stay responsive meanwhile.
            time.sleep(self._terminate_blocks_s)
        self._alive = False


class SurvivorPty(RecordingPty):
    """A PTY that ``terminate`` cannot kill — until ``die`` is called."""

    def terminate(self) -> None:
        self.terminates += 1

    def die(self) -> None:
        self._alive = False


class ChatPool(FakeChatPool):
    """The fake chat pool plus the ``reinsert`` phase (c) asks of it."""

    def __init__(self) -> None:
        super().__init__()
        self.reinserted: list[str] = []

    async def reinsert(self, chat_id: str, session: FakeChatSession) -> bool:
        self.reinserted.append(chat_id)
        if chat_id in self.sessions or chat_id in self.starting:
            return False
        self.sessions[chat_id] = session
        return True


class Chat(FakeChatSession):
    """A fake chat whose child's death is under the test's control.

    ``exits`` is what ``process_exited`` reads after ``teardown``: True (the
    ordinary case), False (a survivor), or None (no handle was ever captured).
    ``dies_after`` sleeps lets a survivor become a corpse while (c) polls.
    """

    def __init__(
        self,
        *,
        busy: bool = False,
        exits: bool | None = True,
        dies_after: int | None = None,
    ) -> None:
        super().__init__(busy=busy)
        self._exits = exits
        self._dies_after = dies_after
        self.cancels = 0

    async def teardown(self) -> None:
        await super().teardown()
        self.process_exited = self._exits

    def let_die(self) -> None:
        """Arm the next ``teardown`` to take: the child exits when signalled again."""
        self._exits = True

    async def cancel(self) -> None:
        self.cancels += 1
        self.is_busy = False

    def dying(self, app: SimpleNamespace) -> None:
        """Arm ``dies_after``: the child exits once that many polls have slept."""
        assert self._dies_after is not None
        target = len(app.clock.sleeps) + self._dies_after

        def maybe_exit(count: int) -> None:
            if count >= target:
                self.process_exited = True

        app.clock.on_sleep.append(maybe_exit)


class Recorder:
    """Stands in for phase (c): keeps what it was handed and hands the plan back."""

    def __init__(self) -> None:
        self.calls: list[tuple[AcquirePlan, WaitOutcome]] = []

    async def __call__(self, _app, plan: AcquirePlan, outcome: WaitOutcome) -> AcquirePlan:
        self.calls.append((plan, outcome))
        return plan


# ---------------------------------------------------------------------------
# The app and its pools
# ---------------------------------------------------------------------------


def make_app(*, hook: bool = True, clock: FakeClock | None = None) -> SimpleNamespace:
    """An app with both pools, a fake-clock ``HandoffState`` and empty turn stores.

    Args:
        hook: Whether the PTY's turns are visible through the turn hook.
        clock: The fake clock the state sleeps on; a fresh one when omitted.
    """
    clock = clock or FakeClock()
    state = HandoffState(clock=clock, sleep=clock.sleep)
    return SimpleNamespace(
        state=SimpleNamespace(
            pty_registry=PtyRegistry(max_background=5),
            operator_registry=SimpleNamespace(chats=ChatPool()),
            handoff=state,
            turn_hook_present=hook,
            turn_state={},
            transcript_map={},
            transcript_map_provisional=False,
        ),
        clock=clock,
    )


def chats(app: SimpleNamespace) -> FakeChatPool:
    return app.state.operator_registry.chats


def registry(app: SimpleNamespace) -> PtyRegistry:
    return app.state.pty_registry


def pool_pty(app: SimpleNamespace, fake: RecordingPty | None = None) -> RecordingPty:
    """Put a live fake PTY into the registry under :data:`KEY` through the pool path."""
    fake = fake or RecordingPty()
    reg = registry(app)
    with patch.object(reg, "_spawn_session", return_value=fake):
        session, reused = reg.get_or_create_session(KEY, ["fake"])
    assert session is fake and not reused
    return fake


def set_store(app: SimpleNamespace, state: str, ts: float, key: str = KEY) -> None:
    app.state.turn_state[key] = {"state": state, "ts": ts, "transcript_id": key}


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def write_transcript(path: Path, entries: list[dict], mtime: float) -> Path:
    """Put *entries* at *path* with *mtime*, as one change.

    Written next to the target and moved into place with the stamp already
    set, so a wait polling the file in a worker thread never sees the write
    and the stamp as two changes (and reads the tail twice for one edit).
    """
    staging = path.with_name(path.name + ".staging")
    staging.write_text("".join(json.dumps(e) + "\n" for e in entries))
    os.utime(staging, (mtime, mtime))
    os.replace(staging, path)
    return path


def user_entry(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def assistant_entry(text: str) -> dict:
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


INTERRUPT_ENTRY = user_entry("[Request interrupted by user]")


@pytest.fixture
def disk():
    """The transcripts on disk, as phase (c) sees them; tests add ids to it."""
    ids: set[str] = set()
    with patch.object(session_handoff, "_transcripts_on_disk", lambda _app: ids):
        yield ids


# ---------------------------------------------------------------------------
# Spawning and the door
# ---------------------------------------------------------------------------


def pty_spawner(
    app: SimpleNamespace, *, pty: RecordingPty | None = None, gate: asyncio.Event | None = None
):
    """A spawn callback that pools a fake PTY the way the terminal handler does."""
    calls: list[SpawnRequest] = []

    async def spawn(request: SpawnRequest):
        calls.append(request)
        if gate is not None:
            await gate.wait()
        return pool_pty(app, pty or RecordingPty())

    spawn.calls = calls  # type: ignore[attr-defined]
    return spawn


def chat_spawner(
    app: SimpleNamespace, *, chat: Chat | None = None, gate: asyncio.Event | None = None
):
    """A spawn callback that pools a fake chat the way ``get_or_create`` does."""
    calls: list[SpawnRequest] = []

    async def spawn(request: SpawnRequest):
        calls.append(request)
        if gate is not None:
            await gate.wait()
        session = chat or Chat()
        chats(app).starting.discard(request.key)
        chats(app).sessions[request.key] = session
        return session

    spawn.calls = calls  # type: ignore[attr-defined]
    return spawn


async def must_not_spawn(request: SpawnRequest):
    """The spawn callback of an acquire whose outcome starts no process."""
    raise AssertionError(f"unexpected spawn for {request.key}")


async def acquire(
    app: SimpleNamespace,
    key: str,
    surface: Surface,
    channel: object,
    *,
    interrupt: bool = False,
    spawn: SpawnCallback | None = None,
    end_started: bool = False,
) -> AcquireResult:
    """``acquire_surface`` with :func:`must_not_spawn` unless *spawn* is given."""
    return await acquire_surface(
        app,
        key,
        surface,
        channel,
        interrupt=interrupt,
        spawn=spawn or must_not_spawn,
        end_started=end_started,
    )


def wait_returns(reason: str, *, before=None):
    """Stub phase (b) to report *reason*, optionally after running *before*."""

    async def wait(_app, _plan):
        if before is not None:
            before()
        return WaitOutcome(reason)

    return patch.object(session_handoff, "_wait_for_idle", wait)


# ---------------------------------------------------------------------------
# Waiting and asserting
# ---------------------------------------------------------------------------


async def until(predicate, *, timeout: float = 5.0) -> None:
    """Spin the loop (real time) until *predicate* holds."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.001)


async def ticks(app: SimpleNamespace, n: int) -> None:
    """Let the wait take *n* more looks (each look ends in one fake sleep)."""
    target = len(app.clock.sleeps) + n
    await until(lambda: len(app.clock.sleeps) >= target)


def assert_registered(app: SimpleNamespace, plan: AcquirePlan, channel: object) -> None:
    """The pending slot and the reservation an accepted plan leaves behind."""
    pending = get_state(app).pending[plan.key]
    assert pending.channel is channel
    assert pending.surface == plan.surface
    assert pending.task is not None
    assert registry(app).is_reserved(plan.key)
    assert plan.channel is channel


def assert_released(app: SimpleNamespace) -> None:
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)


def assert_held(app: SimpleNamespace, channel: object) -> None:
    assert get_state(app).pending[KEY].channel is channel
    assert registry(app).is_reserved(KEY)
