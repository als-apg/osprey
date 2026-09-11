"""Phase (b) of a surface acquire: the wait for the outgoing turn to end.

Phase (b) runs outside the per-key lock and is the only cancellable phase. It
waits — without a time bound — for the entry phase (a) found under the key to
finish its turn, and ends in a ``WaitOutcome`` that phase (c) is handed. These
tests pin the invariants around that wait:

- a running turn is never torn down, however long it runs, while the caller
  is still there;
- the caller's channel is polled on every tick, and its closing cancels the
  wait, releasing the pending slot and the reservation with no teardown;
- an interrupt is Escape, a bounded grace, then ``terminate`` — and the
  terminate runs off the loop;
- a PTY whose turns are invisible (no turn-state hook) is refused unless the
  caller interrupts;
- of the turn-state store and the transcript tail, the newer witness wins;
- an entry that leaves its pool mid-wait is an error, not a spawn.

Time is faked through the state's injected ``clock``/``sleep``, as in the
phase (a) tests, whose fakes this module reuses. The PTY is a recording fake
so writes and terminates can be asserted; transcripts are real files under
``tmp_path`` with their modification time set by hand.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anyio
import pytest

from osprey.interfaces.web_terminal import session_handoff
from osprey.interfaces.web_terminal.session_handoff import (
    ACTION_HANDOFF,
    ACTION_REUSE,
    ATTACH_POLL_S,
    ERROR_HANDOFF_NEEDS_INTERRUPT,
    ERROR_HANDOFF_SUPERSEDED,
    ERROR_OUTGOING_VANISHED,
    ERROR_SESSION_ATTACHED_ELSEWHERE,
    IDLE_POLL_S,
    INTERRUPT_GRACE_S,
    REASON_EXITED,
    REASON_FORCED,
    REASON_IDLE,
    REASON_INTERRUPTED,
    REASON_NONE,
    AcquireChannel,
    AcquirePlan,
    ChannelClosed,
    ChannelToken,
    HandoffError,
    HandoffNeedsInterrupt,
    HandoffRefused,
    HandoffSuperseded,
    WaitOutcome,
    acquire_surface,
    channel_closed,
    get_state,
)
from tests.interfaces.web_terminal._fakes import FakeChatSession, FakeClock, FakePtySession
from tests.interfaces.web_terminal.test_handoff_phase_a import KEY, chats, make_app, registry

ESC = b"\x1b"


# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class RecordingPty(FakePtySession):
    """The phase (a) fake PTY, remembering what phase (b) does to it."""

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


def pool_pty(app: SimpleNamespace, fake: RecordingPty | None = None) -> RecordingPty:
    fake = fake or RecordingPty()
    reg = registry(app)
    with patch.object(reg, "_spawn_session", return_value=fake):
        session, reused = reg.get_or_create_session(KEY, ["fake"])
    assert session is fake and not reused
    return fake


def make_pty_app(*, hook: bool = True, clock: FakeClock | None = None) -> SimpleNamespace:
    """An app whose stores phase (b) reads are present and empty."""
    app = make_app(clock)
    app.state.turn_hook_present = hook
    app.state.turn_state = {}
    app.state.transcript_map = {}
    app.state.transcript_map_provisional = False
    return app


def set_store(app: SimpleNamespace, state: str, ts: float, key: str = KEY) -> None:
    app.state.turn_state[key] = {"state": state, "ts": ts, "transcript_id": key}


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


class Recorder:
    """Stands in for phase (c): keeps what it was handed and hands the plan back."""

    def __init__(self) -> None:
        self.calls: list[tuple[AcquirePlan, WaitOutcome]] = []

    async def __call__(self, app, plan: AcquirePlan, outcome: WaitOutcome) -> AcquirePlan:
        self.calls.append((plan, outcome))
        return plan


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


def assert_released(app: SimpleNamespace) -> None:
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)


def assert_held(app: SimpleNamespace, channel: object) -> None:
    assert get_state(app).pending[KEY].channel is channel
    assert registry(app).is_reserved(KEY)


# ---------------------------------------------------------------------------
# Constants and the channel protocol
# ---------------------------------------------------------------------------


def test_the_channel_is_polled_at_least_every_200ms():
    assert IDLE_POLL_S <= 0.2
    assert INTERRUPT_GRACE_S == 5.0


async def test_a_plain_token_never_closes():
    """Phase (a) callers pass ``object()``; that must keep working."""
    assert await channel_closed(object()) is False
    assert await channel_closed(ChannelToken()) is False
    assert not isinstance(object(), AcquireChannel)
    assert isinstance(ChannelToken(), AcquireChannel)


async def test_channel_token_accepts_sync_and_async_probes():
    event = asyncio.Event()
    sync_token = ChannelToken(event.is_set)
    assert await channel_closed(sync_token) is False
    event.set()
    assert await channel_closed(sync_token) is True

    answers = iter([False, True])

    async def is_disconnected() -> bool:
        return next(answers)

    post_token = ChannelToken(is_disconnected)
    assert await channel_closed(post_token) is False
    assert await channel_closed(post_token) is True


async def test_channel_closed_is_a_cancellation():
    assert issubclass(ChannelClosed, asyncio.CancelledError)
    assert issubclass(HandoffNeedsInterrupt, HandoffRefused)
    assert not issubclass(HandoffError, HandoffRefused)


# ---------------------------------------------------------------------------
# Nothing to wait on
# ---------------------------------------------------------------------------


async def test_no_outgoing_or_no_wait_hands_c_reason_none():
    # spawn: nothing live
    app = make_pty_app()
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        await acquire_surface(app, KEY, "simple", object())
    assert recorder.calls[-1][1] == WaitOutcome(REASON_NONE)
    assert app.clock.sleeps == []

    # Expert reattaching to its own live PTY: no wait even if the turn runs.
    app = make_pty_app()
    pool_pty(app)
    set_store(app, "busy", time.time())
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        plan = await acquire_surface(app, KEY, "expert", object())
    assert plan.action == ACTION_REUSE and plan.wait_for_idle is False
    assert recorder.calls[-1][1] == WaitOutcome(REASON_NONE)
    assert app.clock.sleeps == []


async def test_acquire_surface_hands_plan_and_outcome_to_phase_c():
    app = make_pty_app()
    chats(app).sessions[KEY] = FakeChatSession(busy=False)
    recorder = Recorder()
    channel = object()
    with patch.object(session_handoff, "_phase_c", recorder):
        plan = await acquire_surface(app, KEY, "expert", channel)
    assert plan.action == ACTION_HANDOFF
    (handed_plan, outcome), *_ = recorder.calls
    assert handed_plan is plan
    assert outcome == WaitOutcome(REASON_IDLE)
    # Phase (c) is a stub: the slot it will release still stands.
    assert_held(app, channel)


# ---------------------------------------------------------------------------
# Chat-held keys
# ---------------------------------------------------------------------------


async def test_chat_held_key_waits_on_is_busy_then_ends_idle():
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    channel = object()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "expert", channel))
        await ticks(app, 20)
        assert not task.done()
        assert_held(app, channel)
        assert chats(app).terminated == []
        chat.is_busy = False
        plan = await task
    assert plan.action == ACTION_HANDOFF and plan.teardown is True
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)
    assert set(app.clock.sleeps) == {IDLE_POLL_S}
    assert chats(app).terminated == []


async def test_a_chat_held_key_needs_no_hook():
    """Without the turn-state hook an Expert still waits on ``is_busy``."""
    app = make_pty_app(hook=False)
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "expert", object()))
        await ticks(app, 5)
        assert not task.done()
        chat.is_busy = False
        plan = await task
    assert plan.action == ACTION_HANDOFF
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)
    assert chats(app).terminated == []


async def test_simple_reusing_a_busy_chat_waits_too():
    """Same surface, chat held: the route cannot take a turn on a busy chat."""
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
        await ticks(app, 5)
        assert not task.done()
        chat.is_busy = False
        plan = await task
    assert plan.action == ACTION_REUSE and plan.wait_for_idle is True
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)


async def test_chat_held_key_with_interrupt_skips_the_wait():
    app = make_pty_app()
    chats(app).sessions[KEY] = FakeChatSession(busy=True)
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        await acquire_surface(app, KEY, "expert", object(), interrupt=True)
    assert recorder.calls[-1][1] == WaitOutcome(REASON_INTERRUPTED)
    assert app.clock.sleeps == []
    # Phase (b) never tears anything down; the pool's terminate is (c)'s.
    assert chats(app).terminated == []


async def test_simple_reusing_a_busy_chat_with_interrupt_leaves_the_cancel_to_c():
    """``reuse`` + ``interrupted``: phase (c) owes ``session.cancel()``; (b) touched nothing."""
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        plan = await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert (plan.action, plan.teardown, plan.wait_for_idle) == (ACTION_REUSE, False, True)
    assert recorder.calls[-1] == (plan, WaitOutcome(REASON_INTERRUPTED))
    assert chat.is_busy is True
    assert chats(app).terminated == []
    assert app.clock.sleeps == []


async def test_chat_vanishing_mid_wait_is_an_error_and_releases_the_slot():
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    channel = object()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "expert", channel))
        await ticks(app, 3)
        del chats(app).sessions[KEY]
        with pytest.raises(HandoffError) as excinfo:
            await task
    assert excinfo.value.error == ERROR_OUTGOING_VANISHED
    assert recorder.calls == []
    assert_released(app)


async def test_chat_dying_while_pooled_ends_the_wait_as_exited():
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "expert", object()))
        await ticks(app, 3)
        chat._active = False
        await task
    assert recorder.calls[-1][1] == WaitOutcome(REASON_EXITED)


# ---------------------------------------------------------------------------
# PTY-held keys: the wait has no time bound
# ---------------------------------------------------------------------------


async def test_a_running_turn_is_never_terminated_while_the_channel_is_open():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    recorder = Recorder()
    channel = ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", channel))
        # Hours of fake time, a few hundred looks.
        await ticks(app, 300)
        app.clock.now += 6 * 3600
        await ticks(app, 50)
        assert not task.done()
        assert pty.terminates == 0
        assert pty.writes == []
        assert pty.is_alive
        assert registry(app).get_session(KEY) is pty
        assert_held(app, channel)
        assert recorder.calls == []
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert_released(app)
    assert pty.terminates == 0


async def test_pty_going_idle_in_the_store_ends_the_wait():
    app = make_pty_app()
    pool_pty(app)
    set_store(app, "busy", time.time())
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
        await ticks(app, 5)
        assert not task.done()
        set_store(app, "idle", time.time())
        plan = await task
    assert plan.action == ACTION_HANDOFF
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)


async def test_pty_already_idle_hands_c_idle_without_a_sleep():
    app = make_pty_app()
    pool_pty(app)
    set_store(app, "idle", time.time())
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        await acquire_surface(app, KEY, "simple", object())
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)
    assert app.clock.sleeps == []


async def test_pty_vanishing_mid_wait_is_an_error_and_releases_the_slot():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    recorder = Recorder()
    channel = object()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", channel))
        await ticks(app, 3)
        assert registry(app).pop_session(KEY) is pty
        with pytest.raises(HandoffError) as excinfo:
            await task
    assert excinfo.value.error == ERROR_OUTGOING_VANISHED
    assert recorder.calls == []
    assert pty.terminates == 0
    assert_released(app)


async def test_pty_dying_while_pooled_ends_the_wait_as_exited():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
        await ticks(app, 3)
        pty._alive = False
        await task
    assert recorder.calls[-1][1] == WaitOutcome(REASON_EXITED)
    assert pty.terminates == 0


# ---------------------------------------------------------------------------
# The caller's channel closing
# ---------------------------------------------------------------------------


async def test_channel_closing_mid_wait_cancels_releases_and_touches_nothing():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    closed = asyncio.Event()
    channel = ChannelToken(closed.is_set)
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", channel))
        await ticks(app, 4)
        assert_held(app, channel)
        closed.set()
        with pytest.raises(ChannelClosed):
            await task
    assert task.cancelled()
    assert_released(app)
    assert recorder.calls == []
    assert pty.terminates == 0
    assert pty.writes == []
    assert registry(app).get_session(KEY) is pty
    # The key is free for the next acquire.
    result = await acquire_surface(app, KEY, "expert", object())
    assert result.plan.action == ACTION_REUSE


async def test_post_style_awaitable_probe_closes_the_wait_too():
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    disconnected = False

    async def is_disconnected() -> bool:
        return disconnected

    channel = ChannelToken(is_disconnected)
    task = asyncio.create_task(acquire_surface(app, KEY, "expert", channel))
    await ticks(app, 3)
    disconnected = True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_released(app)
    assert chats(app).terminated == []


async def test_the_channel_is_asked_on_every_look():
    app = make_pty_app()
    chats(app).sessions[KEY] = FakeChatSession(busy=True)
    probes = 0

    def is_closed() -> bool:
        nonlocal probes
        probes += 1
        return False

    task = asyncio.create_task(acquire_surface(app, KEY, "expert", ChannelToken(is_closed)))
    await ticks(app, 10)
    looks = len(app.clock.sleeps)
    assert probes >= looks
    assert all(s <= 0.2 for s in app.clock.sleeps)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_released(app)


async def test_a_closed_channel_is_seen_on_the_first_look():
    """A caller already gone when (b) starts never sees a tick of waiting."""
    app = make_pty_app()
    chats(app).sessions[KEY] = FakeChatSession(busy=True)
    with pytest.raises(ChannelClosed):
        await acquire_surface(app, KEY, "expert", ChannelToken(lambda: True))
    assert app.clock.sleeps == []
    assert_released(app)


# ---------------------------------------------------------------------------
# Hook-less PTY
# ---------------------------------------------------------------------------


async def test_hookless_pty_held_key_is_refused_without_an_interrupt():
    app = make_pty_app(hook=False)
    pty = pool_pty(app)
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        with pytest.raises(HandoffNeedsInterrupt) as excinfo:
            await acquire_surface(app, KEY, "simple", object())
    refused = excinfo.value
    assert isinstance(refused, HandoffRefused)
    assert (refused.status, refused.error) == (409, ERROR_HANDOFF_NEEDS_INTERRUPT)
    assert refused.ws_close_code is None
    assert HandoffRefused.needs_interrupt(KEY).error == ERROR_HANDOFF_NEEDS_INTERRUPT
    assert recorder.calls == []
    assert pty.writes == [] and pty.terminates == 0
    assert app.clock.sleeps == []
    assert_released(app)
    # The PTY is untouched and still the holder.
    assert registry(app).get_session(KEY) is pty


async def test_hookless_pty_with_interrupt_is_escaped_and_ends_on_the_marker(tmp_path):
    # The marker lands on a named look for the reason spelt out on the
    # store-edge test below: a wait whose sleeps are free takes an unbounded
    # number of looks while the test body waits in real time.
    app = make_pty_app(hook=False)
    pty = pool_pty(app)
    transcript = write_transcript(tmp_path / f"{KEY}.jsonl", [user_entry("do it")], time.time())
    looks_before_marker = 4

    def marker_on_the_fourth_look(count: int) -> None:
        assert pty.writes == [ESC], "Escape is written on the first look, before any sleep"
        if count == looks_before_marker:
            write_transcript(transcript, [user_entry("do it"), INTERRUPT_ENTRY], time.time())

    app.clock.on_sleep.append(marker_on_the_fourth_look)
    recorder = Recorder()
    with (
        patch.object(session_handoff, "_transcript_path", lambda _app, _tid: transcript),
        patch.object(session_handoff, "_phase_c", recorder),
    ):
        await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert recorder.calls[-1][1] == WaitOutcome(REASON_INTERRUPTED)
    assert pty.writes == [ESC]
    assert pty.terminates == 0
    assert len(app.clock.sleeps) == looks_before_marker
    assert sum(app.clock.sleeps) < INTERRUPT_GRACE_S


async def test_hookless_pty_without_a_transcript_is_not_assumed_idle():
    """No store entry and no file: nothing witnessed the turn end."""
    app = make_pty_app(hook=False)
    pty = pool_pty(app)
    task = asyncio.create_task(acquire_surface(app, KEY, "simple", object(), interrupt=True))
    await ticks(app, 5)
    assert pty.writes == [ESC]
    assert not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Interrupt with the hook present
# ---------------------------------------------------------------------------


async def test_interrupt_writes_escape_then_proceeds_on_the_store_idle_edge():
    # The flip to idle is driven from the fake clock rather than from the test
    # body, because the wait's sleeps cost no real time: every real-time poll
    # the body makes lets the wait take tens of further looks, and past the
    # fiftieth the grace has expired and the outcome is `forced`. Flipping on a
    # named look keeps "several looks pass while the store says busy" exact.
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    looks_before_idle = 4

    def idle_on_the_fourth_look(count: int) -> None:
        assert pty.writes == [ESC], "Escape is written on the first look, before any sleep"
        if count >= looks_before_idle:
            set_store(app, "idle", time.time())

    app.clock.on_sleep.append(idle_on_the_fourth_look)
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        plan = await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert pty.writes == [ESC]
    assert plan.interrupt is True
    assert recorder.calls[-1][1] == WaitOutcome(REASON_INTERRUPTED)
    assert pty.terminates == 0
    # The wait held for every look the store said busy, and ended well inside
    # the grace rather than by running it out.
    assert len(app.clock.sleeps) == looks_before_idle
    assert sum(app.clock.sleeps) < INTERRUPT_GRACE_S


async def test_interrupt_on_an_idle_pty_writes_nothing():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "idle", time.time())
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)
    assert pty.writes == []


async def test_interrupt_that_never_lands_terminates_after_the_grace():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    recorder = Recorder()
    channel = object()
    with patch.object(session_handoff, "_phase_c", recorder):
        plan = await acquire_surface(app, KEY, "simple", channel, interrupt=True)
    assert pty.writes == [ESC]
    assert pty.terminates == 1
    assert recorder.calls[-1] == (plan, WaitOutcome(REASON_FORCED))
    # The grace is spent in full and by no more than one extra look.
    assert sum(app.clock.sleeps) == pytest.approx(INTERRUPT_GRACE_S, abs=2 * IDLE_POLL_S)
    assert len(app.clock.sleeps) >= round(INTERRUPT_GRACE_S / IDLE_POLL_S)
    # Terminated, not popped: (c) runs its ordinary teardown and death check.
    assert registry(app).get_session(KEY) is pty
    assert_held(app, channel)


async def test_the_loop_stays_responsive_during_a_slow_terminate():
    app = make_pty_app()
    pty = pool_pty(app, RecordingPty(terminate_blocks_s=0.3))
    set_store(app, "busy", time.time())
    heartbeats = 0
    stop = asyncio.Event()

    async def heartbeat() -> None:
        nonlocal heartbeats
        while not stop.is_set():
            heartbeats += 1
            await asyncio.sleep(0.01)

    beat = asyncio.create_task(heartbeat())
    with patch.object(session_handoff, "_phase_c", Recorder()):
        started = time.monotonic()
        await acquire_surface(app, KEY, "simple", object(), interrupt=True)
        wall = time.monotonic() - started
    stop.set()
    await beat
    assert pty.terminates == 1
    assert wall >= 0.3
    # A blocked loop would have counted one or two beats across 300 ms; the
    # bar is kept low so a loaded machine does not fail it.
    assert heartbeats >= 3


async def test_a_failed_escape_write_is_classified_by_the_next_look():
    """The master fd closing between the liveness look and the write is an exit, not a crash."""
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())

    def write_input(_data: bytes) -> None:
        pty._alive = False
        raise OSError("Input/output error")

    pty.write_input = write_input  # type: ignore[method-assign]
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert recorder.calls[-1][1] == WaitOutcome(REASON_EXITED)
    assert pty.terminates == 0


async def test_a_failed_escape_write_on_a_live_pty_still_runs_the_grace():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())

    def write_input(_data: bytes) -> None:
        raise OSError("Bad file descriptor")

    pty.write_input = write_input  # type: ignore[method-assign]
    recorder = Recorder()
    with patch.object(session_handoff, "_phase_c", recorder):
        await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert recorder.calls[-1][1] == WaitOutcome(REASON_FORCED)
    assert pty.terminates == 1
    assert sum(app.clock.sleeps) == pytest.approx(INTERRUPT_GRACE_S, abs=2 * IDLE_POLL_S)


async def test_channel_closing_during_the_interrupt_grace_still_cancels():
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    closed = asyncio.Event()
    channel = ChannelToken(closed.is_set)
    task = asyncio.create_task(acquire_surface(app, KEY, "simple", channel, interrupt=True))
    await until(lambda: pty.writes == [ESC])
    await ticks(app, 2)
    closed.set()
    with pytest.raises(ChannelClosed):
        await task
    assert pty.terminates == 0
    assert_released(app)


# ---------------------------------------------------------------------------
# Store versus tail: the newer witness wins
# ---------------------------------------------------------------------------


def judge(app, store, transcript: Path | None) -> bool:
    with patch.object(session_handoff, "_transcript_path", lambda _app, _tid: transcript):
        return session_handoff._judge_pty_idle(app, store, KEY)


def test_store_newer_than_tail_wins(tmp_path):
    app = make_pty_app()
    now = time.time()
    # An unanswered prompt at the tail, written before the store's idle edge.
    transcript = write_transcript(tmp_path / "t.jsonl", [user_entry("prompt")], now - 10)
    assert judge(app, {"state": "idle", "ts": now}, transcript) is True
    # The same tail, older than a busy edge: busy.
    assert judge(app, {"state": "busy", "ts": now}, transcript) is False
    # An interrupt marker older than a busy edge cannot end the newer turn.
    interrupted = write_transcript(tmp_path / "i.jsonl", [INTERRUPT_ENTRY], now - 10)
    assert judge(app, {"state": "busy", "ts": now}, interrupted) is False
    # A tie goes to the store.
    tied = write_transcript(tmp_path / "e.jsonl", [user_entry("prompt")], now)
    assert judge(app, {"state": "idle", "ts": now}, tied) is True


def test_tail_newer_than_store_wins_when_it_has_evidence(tmp_path):
    app = make_pty_app()
    now = time.time()
    # Store idle, then a prompt was submitted: busy.
    prompt = write_transcript(tmp_path / "p.jsonl", [user_entry("prompt")], now + 10)
    assert judge(app, {"state": "idle", "ts": now}, prompt) is False
    # Store busy, then the turn was interrupted: idle.
    interrupted = write_transcript(tmp_path / "i.jsonl", [INTERRUPT_ENTRY], now + 10)
    assert judge(app, {"state": "busy", "ts": now}, interrupted) is True
    # Store busy, then slash-command output newer than the busy stamp: idle.
    entry = user_entry("<local-command-stdout>ok</local-command-stdout>")
    entry["timestamp"] = "2099-01-01T00:00:00Z"
    command = write_transcript(tmp_path / "c.jsonl", [entry], now + 10)
    assert judge(app, {"state": "busy", "ts": now}, command) is True


def test_a_newer_tail_without_evidence_hands_back_to_the_store(tmp_path):
    app = make_pty_app()
    now = time.time()
    reply = write_transcript(tmp_path / "a.jsonl", [assistant_entry("done")], now + 10)
    assert judge(app, {"state": "idle", "ts": now}, reply) is True
    assert judge(app, {"state": "busy", "ts": now}, reply) is False


def test_without_a_store_entry_only_an_explicit_idle_shape_counts(tmp_path):
    app = make_pty_app()
    now = time.time()
    assert judge(app, None, None) is False
    reply = write_transcript(tmp_path / "a.jsonl", [assistant_entry("done")], now)
    assert judge(app, None, reply) is False
    prompt = write_transcript(tmp_path / "p.jsonl", [user_entry("prompt")], now)
    assert judge(app, None, prompt) is False
    interrupted = write_transcript(tmp_path / "i.jsonl", [INTERRUPT_ENTRY], now)
    assert judge(app, None, interrupted) is True


def test_a_missing_transcript_leaves_the_store_as_the_only_witness(tmp_path):
    app = make_pty_app()
    now = time.time()
    assert judge(app, {"state": "idle", "ts": now}, None) is True
    assert judge(app, {"state": "busy", "ts": now}, None) is False
    gone = tmp_path / "gone.jsonl"
    assert judge(app, {"state": "idle", "ts": now}, gone) is True


async def test_the_tail_is_read_for_the_keys_current_transcript(tmp_path):
    """After a ``/clear`` the key points at a new transcript; that is the one read."""
    app = make_pty_app()
    pool_pty(app)
    moved = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    app.state.transcript_map[KEY] = moved
    now = time.time()
    set_store(app, "busy", now)
    files = {
        KEY: write_transcript(tmp_path / f"{KEY}.jsonl", [INTERRUPT_ENTRY], now + 10),
        moved: write_transcript(tmp_path / f"{moved}.jsonl", [user_entry("prompt")], now + 10),
    }
    asked: list[str] = []

    def transcript_path(_app, transcript_id: str):
        asked.append(transcript_id)
        return files.get(transcript_id)

    with patch.object(session_handoff, "_transcript_path", transcript_path):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
        await ticks(app, 3)
        assert not task.done(), "the old transcript's marker must not end the wait"
        assert set(asked) == {moved}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_an_unchanged_transcript_is_read_once_per_wait(tmp_path):
    """Every look stats the file; only a changed file is read again."""
    app = make_pty_app()
    pool_pty(app)
    now = time.time()
    set_store(app, "busy", now)
    # Newer than the store, so the tail rule is consulted on every look.
    transcript = write_transcript(tmp_path / f"{KEY}.jsonl", [user_entry("prompt")], now + 10)
    reads = 0
    real_tail_state = session_handoff.tail_state

    def counting_tail_state(path, busy_since):
        nonlocal reads
        reads += 1
        return real_tail_state(path, busy_since)

    with (
        patch.object(session_handoff, "_transcript_path", lambda _app, _tid: transcript),
        patch.object(session_handoff, "tail_state", counting_tail_state),
        patch.object(session_handoff, "_phase_c", Recorder()),
    ):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
        await ticks(app, 30)
        assert not task.done()
        assert reads == 1
        # The file moves: one more read, and its marker ends the wait.
        write_transcript(transcript, [user_entry("prompt"), INTERRUPT_ENTRY], now + 20)
        await task
    assert reads == 2


def test_the_tail_memo_keys_on_size_mtime_and_busy_stamp(tmp_path):
    memo = session_handoff._TailMemo()
    transcript = write_transcript(tmp_path / "t.jsonl", [INTERRUPT_ENTRY], 1_700_000_000)
    reads = 0
    real_tail_state = session_handoff.tail_state

    def counting_tail_state(path, busy_since):
        nonlocal reads
        reads += 1
        return real_tail_state(path, busy_since)

    with patch.object(session_handoff, "tail_state", counting_tail_state):
        stat = transcript.stat()
        assert memo.verdict(transcript, stat, None) == "idle"
        assert memo.verdict(transcript, stat, None) == "idle"
        assert reads == 1
        assert memo.verdict(transcript, stat, 1.0) == "idle"
        assert reads == 2
        write_transcript(transcript, [INTERRUPT_ENTRY, user_entry("again")], 1_700_000_100)
        assert memo.verdict(transcript, transcript.stat(), 1.0) == "busy"
        assert reads == 3


def test_transcript_path_needs_a_project_cwd():
    app = make_pty_app()
    assert session_handoff._transcript_path(app, KEY) is None


# ---------------------------------------------------------------------------
# A newer Simple acquire with an interrupt supersedes a pending Simple wait
# ---------------------------------------------------------------------------


def busy_pty_app() -> tuple[SimpleNamespace, RecordingPty]:
    """A hooked terminal in the middle of a turn: the wait a Simple acquire meets."""
    app = make_pty_app()
    pty = pool_pty(app)
    set_store(app, "busy", time.time())
    return app, pty


async def start_wait(app: SimpleNamespace, surface: str, channel: object) -> asyncio.Task:
    """Start an acquire and see it into its phase (b) wait."""
    task = asyncio.create_task(acquire_surface(app, KEY, surface, channel))
    await ticks(app, 3)
    assert not task.done()
    assert_held(app, channel)
    return task


def idle_once_escaped(app: SimpleNamespace, pty: RecordingPty) -> None:
    """Report the terminal idle on the first look after Escape was written."""

    def hook(_count: int) -> None:
        if pty.writes:
            set_store(app, "idle", time.time())

    app.clock.on_sleep.append(hook)


async def test_a_simple_interrupt_supersedes_a_pending_simple_wait():
    """The first wait ends refused and released; the second carries the interrupt through."""
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel, second_channel = ChannelToken(), ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        first = await start_wait(app, "simple", first_channel)
        idle_once_escaped(app, pty)
        second = asyncio.create_task(
            acquire_surface(app, KEY, "simple", second_channel, interrupt=True)
        )

        with pytest.raises(HandoffSuperseded) as excinfo:
            await first
        assert excinfo.value.status == 409
        assert excinfo.value.error == ERROR_HANDOFF_SUPERSEDED
        assert excinfo.value.ws_close_code is None
        # A refusal, not a cancellation: the route answers it like any other.
        assert not first.cancelled()

        plan = await second

    assert plan.channel is second_channel
    assert plan.action == ACTION_HANDOFF and plan.interrupt
    assert recorder.calls == [(plan, WaitOutcome(REASON_INTERRUPTED))]
    assert pty.writes == [ESC]
    assert pty.terminates == 0
    assert registry(app).get_session(KEY) is pty
    assert_held(app, second_channel)


async def test_the_superseded_mark_lands_before_the_waiter_runs():
    """The mark is what the waiter reads; it is set on the look that cancels, once."""
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel = ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        first = await start_wait(app, "simple", first_channel)
        second = asyncio.create_task(
            acquire_surface(app, KEY, "simple", ChannelToken(), interrupt=True)
        )
        # One loop step: the second has looked and cancelled; the first has not run yet.
        await asyncio.sleep(0)
        pending = get_state(app).pending[KEY]
        assert pending.channel is first_channel
        assert pending.superseded
        assert not first.done()

        with pytest.raises(HandoffSuperseded):
            await first
        idle_once_escaped(app, pty)
        await second
    assert pty.writes == [ESC]


async def test_a_cancellation_from_elsewhere_arriving_with_the_supersede_is_honoured():
    """Only the one cancellation the supersede delivered is withdrawn."""
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel = ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        first = await start_wait(app, "simple", first_channel)
        second = asyncio.create_task(
            acquire_surface(app, KEY, "simple", ChannelToken(), interrupt=True)
        )
        await asyncio.sleep(0)
        assert get_state(app).pending[KEY].superseded
        first.cancel()

        with pytest.raises(asyncio.CancelledError):
            await first
        assert first.cancelled()

        idle_once_escaped(app, pty)
        await second
    assert pty.writes == [ESC]


async def test_without_an_interrupt_a_second_simple_acquire_waits_out_the_grace():
    """No interrupt, no supersede: blocked through the attach grace, then 409."""
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel = ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        first = await start_wait(app, "simple", first_channel)

        with pytest.raises(HandoffRefused) as excinfo:
            await acquire_surface(app, KEY, "simple", ChannelToken())
        assert excinfo.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE
        assert not isinstance(excinfo.value, HandoffSuperseded)

        assert not first.done()
        assert_held(app, first_channel)
        assert not get_state(app).pending[KEY].superseded
        assert pty.writes == []
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    assert recorder.calls == []
    assert_released(app)


async def test_an_expert_acquire_never_supersedes_a_pending_simple_wait():
    """The terminal handshake carries no interrupt; even asked for one it waits like today."""
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel = ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        first = await start_wait(app, "simple", first_channel)

        with pytest.raises(HandoffRefused) as excinfo:
            await acquire_surface(app, KEY, "expert", ChannelToken(), interrupt=True)
        assert excinfo.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE

        assert not first.done()
        assert_held(app, first_channel)
        assert not get_state(app).pending[KEY].superseded
        assert pty.writes == []
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    assert recorder.calls == []
    assert_released(app)


async def test_a_pending_expert_wait_is_not_superseded_by_a_simple_interrupt():
    """The interrupt is a gesture at the Simple view's own wait, not at the other view's."""
    app = make_pty_app()
    chat = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = chat
    recorder = Recorder()
    expert_channel = ChannelToken()
    with patch.object(session_handoff, "_phase_c", recorder):
        first = await start_wait(app, "expert", expert_channel)

        with pytest.raises(HandoffRefused) as excinfo:
            await acquire_surface(app, KEY, "simple", ChannelToken(), interrupt=True)
        assert excinfo.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE

        assert not first.done()
        assert get_state(app).pending[KEY].channel is expert_channel
        assert not get_state(app).pending[KEY].superseded
        chat.is_busy = False
        plan = await first
    assert plan.surface == "expert"
    assert recorder.calls[-1][1] == WaitOutcome(REASON_IDLE)
    assert chats(app).get(KEY) is chat


async def test_three_requests_settle_on_one_supersede_and_one_completion():
    """Two interrupts behind one wait: the first cancels it, the second is plainly blocked.

    While the first wait is marked, a further interrupt is an ordinary
    blocked look — the mark is never re-applied. Once the wait releases,
    the interrupt that registers is superseded by the other, and that one
    completes: exactly one ``HandoffSuperseded`` among the two, one plan
    handed to phase (c), one Escape, and no ping-pong between them.

    The interleaving is pinned, not observed: the PTY idle judgement is
    replaced by one that reads the store and yields once, so no worker
    thread decides which task runs next and the loop's own order does —
    the second and third acquires are created in one step, so the third
    looks while the first is marked; after the first releases, the second
    registers and yields at its first look, and the third supersedes it
    there, before any Escape is written.
    """
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel, second_channel, third_channel = ChannelToken(), ChannelToken(), ChannelToken()

    async def judge_from_the_store(_app, key: str, _memo=None) -> bool:
        await asyncio.sleep(0)
        return app.state.turn_state[key]["state"] == "idle"

    with (
        patch.object(session_handoff, "_phase_c", recorder),
        patch.object(session_handoff, "_pty_turn_idle", judge_from_the_store),
    ):
        first = await start_wait(app, "simple", first_channel)
        idle_once_escaped(app, pty)
        second = asyncio.create_task(
            acquire_surface(app, KEY, "simple", second_channel, interrupt=True)
        )
        third = asyncio.create_task(
            acquire_surface(app, KEY, "simple", third_channel, interrupt=True)
        )
        # One step: the second has marked the first, the third has met the
        # mark and been plainly blocked, and the first has not run yet.
        await asyncio.sleep(0)
        pending = get_state(app).pending[KEY]
        assert pending.channel is first_channel
        assert pending.superseded
        assert not first.done() and not second.done() and not third.done()

        # One cancellation, not two: the first ends refused, not cancelled.
        with pytest.raises(HandoffSuperseded):
            await first
        with pytest.raises(HandoffSuperseded):
            await second
        plan = await third

    assert plan.channel is third_channel and plan.interrupt
    assert recorder.calls == [(plan, WaitOutcome(REASON_INTERRUPTED))]
    assert pty.writes == [ESC]
    assert pty.terminates == 0
    assert_held(app, third_channel)


# ---------------------------------------------------------------------------
# The mark ends the wait even when the cancellation is lost
# ---------------------------------------------------------------------------


def store_judge(app: SimpleNamespace):
    """A PTY idle judgement that reads the store and yields once, so no thread decides the order."""

    async def judge(_app, key: str, _memo=None) -> bool:
        await asyncio.sleep(0)
        return app.state.turn_state[key]["state"] == "idle"

    return judge


class AbsorbingProbe:
    """A channel probe shaped like Starlette's ``is_disconnected``, with a hook inside the scope.

    The probe awaits under an anyio cancel scope it has already cancelled,
    exactly as the route's channel does. On look *on_look* it calls *spawn*
    either just before the scope is entered or just after the scope was
    cancelled, so a task *spawn* starts has its first step land on either side
    of anyio's own cancellation delivery — the two orderings in which an
    external ``task.cancel()`` can meet the scope.
    """

    def __init__(self, *, on_look: int, spawn, before_scope: bool) -> None:
        self.calls = 0
        self._on_look = on_look
        self._spawn = spawn
        self._before = before_scope

    async def is_closed(self) -> bool:
        self.calls += 1
        if self.calls == self._on_look and self._before:
            self._spawn()
        with anyio.CancelScope() as scope:
            scope.cancel()
            if self.calls == self._on_look and not self._before:
                self._spawn()
            await asyncio.Event().wait()
        return False


@pytest.mark.parametrize("before_scope", [True, False], ids=["cancel-first", "scope-first"])
async def test_a_supersede_meeting_the_channel_probes_scope_still_ends_the_wait(before_scope):
    """Whether the cancel is delivered or absorbed by the probe's scope, the wait ends superseded.

    The second acquire is started from inside the first wait's channel probe,
    so its supersede — mark and ``task.cancel()`` — lands while the first is
    suspended in the probe's anyio scope. However anyio resolves that, the
    first ends ``handoff_superseded`` with no cancellation left counted on its
    task, its slot released, and the second registered on its next looks.
    """
    app, pty = busy_pty_app()
    recorder = Recorder()
    second_channel = ChannelToken()
    started: list[asyncio.Task] = []

    def spawn() -> None:
        started.append(
            asyncio.create_task(acquire_surface(app, KEY, "simple", second_channel, interrupt=True))
        )

    first_channel = AbsorbingProbe(on_look=3, spawn=spawn, before_scope=before_scope)
    with (
        patch.object(session_handoff, "_phase_c", recorder),
        patch.object(session_handoff, "_pty_turn_idle", store_judge(app)),
    ):
        idle_once_escaped(app, pty)
        first = asyncio.create_task(acquire_surface(app, KEY, "simple", first_channel))

        with pytest.raises(HandoffSuperseded):
            await first
        assert first.cancelling() == 0
        assert first_channel.calls >= 3

        (second,) = started
        plan = await second

    assert plan.channel is second_channel and plan.interrupt
    assert recorder.calls == [(plan, WaitOutcome(REASON_INTERRUPTED))]
    assert pty.writes == [ESC]
    assert pty.terminates == 0
    # The superseder was blocked for the marking look and at most the two
    # steps the first needed to reach its next look; never the attach grace.
    assert app.clock.sleeps.count(ATTACH_POLL_S) <= 3
    assert_held(app, second_channel)


async def test_the_mark_alone_ends_the_wait_with_no_cancellation_counted():
    """A supersede whose cancellation never arrives is honoured from the mark on the next look."""
    app, pty = busy_pty_app()
    recorder = Recorder()
    first_channel = ChannelToken()
    with (
        patch.object(session_handoff, "_phase_c", recorder),
        patch.object(session_handoff, "_pty_turn_idle", store_judge(app)),
    ):
        first = await start_wait(app, "simple", first_channel)
        looks = len(app.clock.sleeps)
        get_state(app).pending[KEY].superseded = True

        with pytest.raises(HandoffSuperseded) as excinfo:
            await first

    assert excinfo.value.error == ERROR_HANDOFF_SUPERSEDED
    assert not first.cancelled()
    assert first.cancelling() == 0
    assert len(app.clock.sleeps) - looks <= 1
    assert pty.writes == [] and pty.terminates == 0
    assert recorder.calls == []
    assert_released(app)


class MarkingProbe:
    """A channel probe that supersedes the wait it is polled by, on its *on_look*-th look.

    Sets the mark between the look's first read of it and its verdict — the
    shape of a supersede absorbed on the probe — with no cancellation at all.
    """

    def __init__(self, app: SimpleNamespace, *, on_look: int) -> None:
        self._app = app
        self._on_look = on_look
        self.calls = 0

    async def is_closed(self) -> bool:
        self.calls += 1
        if self.calls == self._on_look:
            get_state(self._app).pending[KEY].superseded = True
        return False


async def test_a_mark_set_between_the_probe_and_an_idle_verdict_still_ends_the_wait():
    """The look that would return idle re-reads the mark before returning, and refuses instead."""
    app, pty = busy_pty_app()
    set_store(app, "idle", time.time())
    recorder = Recorder()
    channel = MarkingProbe(app, on_look=1)
    with (
        patch.object(session_handoff, "_phase_c", recorder),
        patch.object(session_handoff, "_pty_turn_idle", store_judge(app)),
    ):
        first = asyncio.create_task(acquire_surface(app, KEY, "simple", channel))
        with pytest.raises(HandoffSuperseded):
            await first

    assert channel.calls == 1
    assert first.cancelling() == 0
    assert recorder.calls == []
    assert pty.writes == [] and pty.terminates == 0
    assert registry(app).get_session(KEY) is pty
    assert_released(app)
