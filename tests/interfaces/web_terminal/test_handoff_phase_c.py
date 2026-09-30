"""Phase (c) of a surface acquire: teardown, death check, spawn, attach, release.

Phase (c) runs under the per-key lock inside a shielded task and turns the
plan phase (a) made and the outcome phase (b) reported into a live process
under the key. These tests pin the invariants around that:

- the outgoing entry is popped from its pool before it is killed, the kill
  runs off the loop, and the child is *observed* dead before anything is
  spawned — whatever reason phase (b) gave;
- a survivor goes back into its own pool unattached and the acquire is
  refused with 503, spawning nothing;
- the spawn callback is told what to resume from a fresh look at the
  transcript map and the transcripts on disk, and is not called at all when
  the incoming surface's own entry is already pooled;
- an Expert caller is attached with its channel token before the pending
  slot and the reservation are released, and a displaced Expert connection
  is closed and detached first;
- a cancellation delivered mid-phase does not stop the phase.

Fakes and the faked clock come from ``_handoff_harness``. Phase (b)
is real unless a test says otherwise; where a test is about how (c) answers a
particular outcome, the wait is stubbed to return that outcome.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from osprey.interfaces.web_terminal import session_handoff
from osprey.interfaces.web_terminal.session_handoff import (
    ACTION_HANDOFF,
    ACTION_REUSE,
    ACTION_SPAWN,
    ACTION_TAKEOVER,
    DEATH_GRACE_S,
    DEATH_POLL_S,
    ERROR_OUTGOING_STILL_RUNNING,
    ERROR_OUTGOING_VANISHED,
    ERROR_SESSION_ATTACHED_ELSEWHERE,
    ERROR_SPAWN_NOT_POOLED,
    REASON_EXITED,
    REASON_FORCED,
    REASON_IDLE,
    REASON_INTERRUPTED,
    REASON_NONE,
    WS_CLOSE_OUTGOING_RUNNING,
    AcquireResult,
    HandoffError,
    HandoffRefused,
    SpawnRequest,
    WaitOutcome,
    acquire_surface,
    get_state,
)
from osprey.interfaces.web_terminal.turn_state import BUSY, IDLE, get_turn_state
from tests.interfaces.web_terminal._handoff_harness import (
    KEY,
    TRANSCRIPT,
    Chat,
    RecordingPty,
    SurvivorPty,
    acquire,
    assert_released,
    chat_spawner,
    chats,
    disk,  # noqa: F401  (fixture, requested by name)
    make_app,
    pool_pty,
    pty_spawner,
    registry,
    set_store,
    until,
    wait_returns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def turn_state(app: SimpleNamespace) -> str | None:
    entry = get_turn_state(app, KEY)
    return None if entry is None else entry["state"]


# ---------------------------------------------------------------------------
# Spawning onto a free key
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_free_key_expert_spawns_attaches_and_releases():
    app = make_app()
    channel = object()
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", channel, spawn=spawn)

    assert isinstance(result, AcquireResult)
    assert result.plan.action == ACTION_SPAWN
    assert result.outcome == WaitOutcome(REASON_NONE)
    assert result.spawned is True
    assert result.resume_id is None
    assert spawn.calls == [
        SpawnRequest(key=KEY, surface="expert", resume_id=None, transcript_id=KEY)
    ]
    assert registry(app).get_session(KEY) is result.session
    assert registry(app).attached_owner(KEY) is channel
    # A fresh TUI has no turn in flight: the store says so explicitly.
    assert turn_state(app) == IDLE
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_free_key_simple_spawns_the_chat_and_releases():
    app = make_app()
    spawn = chat_spawner(app)
    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert result.plan.action == ACTION_SPAWN
    assert result.spawned is True
    assert spawn.calls == [
        SpawnRequest(key=KEY, surface="simple", resume_id=None, transcript_id=KEY)
    ]
    assert chats(app).get(KEY) is result.session
    # Nothing to attach for a chat, and nothing of the PTY's turn store touched.
    assert not registry(app).is_attached(KEY)
    assert turn_state(app) is None
    assert_released(app)


@pytest.mark.parametrize(
    ("mapped", "on_disk", "expected"),
    [
        (TRANSCRIPT, {TRANSCRIPT}, TRANSCRIPT),
        (TRANSCRIPT, {TRANSCRIPT, KEY}, TRANSCRIPT),
        (TRANSCRIPT, {KEY}, KEY),
        (TRANSCRIPT, set(), None),
        (None, {KEY}, KEY),
        (None, set(), None),
    ],
)
async def test_the_spawn_resumes_the_current_transcript_when_it_is_on_disk(
    disk,  # noqa: F811
    mapped,
    on_disk,
    expected,
):
    app = make_app()
    if mapped is not None:
        app.state.transcript_map[KEY] = mapped
    disk.update(on_disk)
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    (request,) = spawn.calls
    assert request.resume_id == expected
    assert request.transcript_id == (mapped or KEY)
    assert result.resume_id == expected


# ---------------------------------------------------------------------------
# The incoming surface already holds the key
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_expert_reuse_attaches_without_spawning():
    app = make_app()
    pty = pool_pty(app)
    set_store(app, BUSY, 1.0)
    channel = object()
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", channel, spawn=spawn)

    assert result.plan.action == ACTION_REUSE
    # A running TUI is reattached to, not waited on.
    assert result.outcome == WaitOutcome(REASON_NONE)
    assert app.clock.sleeps == []
    assert result.session is pty
    assert result.spawned is False
    assert result.resume_id is None
    assert spawn.calls == []
    assert registry(app).attached_owner(KEY) is channel
    # A reattach is not a spawn edge: the running turn's state stands.
    assert turn_state(app) == BUSY
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_simple_reuse_hands_the_idle_chat_back_without_spawning():
    app = make_app()
    chat = Chat(busy=False)
    chats(app).sessions[KEY] = chat
    spawn = chat_spawner(app)
    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert result.plan.action == ACTION_REUSE
    assert result.outcome == WaitOutcome(REASON_IDLE)
    assert result.session is chat and result.spawned is False
    assert spawn.calls == []
    assert chat.cancels == 0 and chat.teardowns == 0
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_simple_reuse_with_interrupt_cancels_the_turn():
    app = make_app()
    chat = Chat(busy=True)
    chats(app).sessions[KEY] = chat
    spawn = chat_spawner(app)
    result = await acquire_surface(app, KEY, "simple", object(), interrupt=True, spawn=spawn)

    assert result.outcome == WaitOutcome(REASON_INTERRUPTED)
    assert result.session is chat
    assert chat.cancels == 1
    assert chat.teardowns == 0
    assert chats(app).terminated == []
    assert spawn.calls == []
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_simple_reuse_of_a_chat_that_died_discards_it_and_spawns():
    app = make_app()
    corpse = Chat(busy=True)
    chats(app).sessions[KEY] = corpse
    spawn = chat_spawner(app)
    with wait_returns(REASON_EXITED):
        result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert chats(app).terminated == [KEY]
    assert result.spawned is True
    assert result.session is not corpse
    assert chats(app).get(KEY) is result.session
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_simple_joining_a_creation_in_flight_goes_through_the_spawn():
    app = make_app()
    chats(app).starting.add(KEY)
    spawn = chat_spawner(app)
    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert result.plan.action == ACTION_REUSE and result.plan.outgoing is None
    assert len(spawn.calls) == 1
    assert chats(app).get(KEY) is result.session
    assert_released(app)


# ---------------------------------------------------------------------------
# Takeover
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_takeover_closes_the_displaced_owner_then_attaches_the_newcomer():
    app = make_app()
    pty = pool_pty(app)
    older, newer = object(), object()
    assert registry(app).attach_session(KEY, older)
    closed: list[str] = []

    async def close_older() -> None:
        closed.append("closed")
        # The displaced handler is still holding the key when its socket
        # closes; only (c)'s own detach releases it.
        assert registry(app).attached_owner(KEY) is older

    get_state(app).closers[older] = close_older
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", newer, spawn=spawn)

    assert result.plan.action == ACTION_TAKEOVER
    # Immediate: a takeover is not a wait for the older owner to leave.
    assert app.clock.sleeps == []
    assert closed == ["closed"]
    assert registry(app).attached_owner(KEY) is newer
    assert result.session is pty and result.spawned is False
    assert spawn.calls == []
    assert pty.terminates == 0
    assert_released(app)
    # The displaced handler's own late detach is a no-op against the new owner.
    registry(app).detach_session(KEY, older)
    assert registry(app).attached_owner(KEY) is newer


@pytest.mark.usefixtures("disk")
async def test_takeover_tolerates_a_missing_or_failing_closer():
    for closer in (None, "raises"):
        app = make_app()
        pool_pty(app)
        older, newer = object(), object()
        assert registry(app).attach_session(KEY, older)
        if closer == "raises":

            async def close_older() -> None:
                raise OSError("socket already gone")

            get_state(app).closers[older] = close_older
        result = await acquire(app, KEY, "expert", newer)
        assert registry(app).attached_owner(KEY) is newer
        assert result.spawned is False
        assert_released(app)


# ---------------------------------------------------------------------------
# Hand-off: chat out, PTY in
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_handoff_tears_the_chat_down_and_spawns_the_pty():
    app = make_app()
    chat = Chat(busy=False)
    chats(app).sessions[KEY] = chat
    channel = object()
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", channel, spawn=spawn)

    assert result.plan.action == ACTION_HANDOFF
    assert result.outcome == WaitOutcome(REASON_IDLE)
    assert chats(app).terminated == [KEY]
    assert chat.teardowns == 1
    assert chats(app).get(KEY) is None
    assert len(spawn.calls) == 1
    assert registry(app).get_session(KEY) is result.session
    assert registry(app).attached_owner(KEY) is channel
    assert result.spawned is True
    assert turn_state(app) == IDLE
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_an_interrupted_chat_handoff_cancels_the_turn_through_the_teardown():
    app = make_app()
    chat = Chat(busy=True)
    chats(app).sessions[KEY] = chat
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", object(), interrupt=True, spawn=spawn)
    assert result.outcome == WaitOutcome(REASON_INTERRUPTED)
    assert chats(app).terminated == [KEY]
    assert chat.teardowns == 1
    assert result.spawned is True


@pytest.mark.usefixtures("disk")
async def test_the_chat_child_is_observed_dead_before_the_spawn():
    app = make_app()
    chat = Chat(busy=False, exits=False, dies_after=3)
    chats(app).sessions[KEY] = chat
    chat.dying(app)
    spawn = pty_spawner(app)
    result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert result.spawned is True
    assert chat.process_exited is True
    polls = [s for s in app.clock.sleeps if s <= DEATH_POLL_S]
    assert len(polls) >= 3


@pytest.mark.usefixtures("disk")
async def test_a_chat_child_that_survives_is_reinserted_and_refused_503():
    app = make_app()
    chat = Chat(busy=False, exits=False)
    chats(app).sessions[KEY] = chat
    spawn = pty_spawner(app)
    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)

    assert refused.value.status == 503
    assert refused.value.error == ERROR_OUTGOING_STILL_RUNNING
    assert refused.value.ws_close_code == WS_CLOSE_OUTGOING_RUNNING
    assert chats(app).terminated == [KEY]
    assert chats(app).reinserted == [KEY]
    assert chats(app).get(KEY) is chat
    assert spawn.calls == []
    assert registry(app).get_session(KEY) is None
    # The full grace was spent looking.
    assert sum(app.clock.sleeps) == pytest.approx(DEATH_GRACE_S, abs=DEATH_POLL_S)
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_the_next_acquire_meets_the_chat_survivor_and_kills_it_again():
    """A reinserted chat survivor is a holder, not a corpse: it is torn down again."""
    app = make_app()
    chat = Chat(busy=False, exits=False)
    chats(app).sessions[KEY] = chat
    spawn = pty_spawner(app)
    with pytest.raises(HandoffRefused):
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert chat.teardowns == 1 and chats(app).get(KEY) is chat

    chat.let_die()
    result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    # Handed off, not discarded: popped and torn down a second time, and
    # observed dead before the spawn.
    assert result.plan.action == ACTION_HANDOFF
    assert result.plan.outgoing is not None and result.plan.outgoing.session is chat
    assert chats(app).terminated == [KEY, KEY]
    assert chat.teardowns == 2
    assert chat.process_exited is True
    assert result.spawned is True
    assert chats(app).get(KEY) is None
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_chat_survivor_is_never_handed_back_to_the_simple_view():
    """Its client is closed; a Simple caller gets a fresh chat after the re-kill."""
    app = make_app()
    chat = Chat(busy=False, exits=False)
    chats(app).sessions[KEY] = chat
    with pytest.raises(HandoffRefused):
        await acquire_surface(app, KEY, "expert", object(), spawn=pty_spawner(app))
    assert chats(app).get(KEY) is chat

    spawn = chat_spawner(app)
    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "simple", object(), spawn=spawn)
    assert refused.value.error == ERROR_OUTGOING_STILL_RUNNING
    assert chat.teardowns == 2
    assert spawn.calls == []

    chat.let_die()
    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)
    assert result.plan.action == ACTION_HANDOFF
    assert chat.teardowns == 3
    assert result.spawned is True
    assert result.session is not chat
    assert chats(app).get(KEY) is result.session
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_chat_survivor_whose_child_exited_meanwhile_is_a_corpse():
    """Once the child is gone the inactive entry is discarded by phase (a), no wait."""
    app = make_app()
    chat = Chat(busy=False, exits=False)
    chats(app).sessions[KEY] = chat
    spawn = pty_spawner(app)
    with pytest.raises(HandoffRefused):
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)

    chat.process_exited = True
    app.clock.sleeps.clear()
    result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert result.plan.action == ACTION_SPAWN
    assert chat.teardowns == 2
    assert app.clock.sleeps == []
    assert result.spawned is True


@pytest.mark.usefixtures("disk")
async def test_an_outgoing_chat_already_gone_is_still_torn_down_and_observed():
    """The reference (c) kept is torn down again — the kill re-signals — then watched."""
    app = make_app()
    chat = Chat(busy=False, exits=True)
    chats(app).sessions[KEY] = chat

    def gone():
        chats(app).sessions.pop(KEY)

    spawn = pty_spawner(app)
    with wait_returns(REASON_IDLE, before=gone):
        result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert chats(app).terminated == []
    assert chat.teardowns == 1
    assert chat.process_exited is True
    assert result.spawned is True
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_chat_without_a_process_handle_is_taken_as_gone_with_a_warning(caplog):
    app = make_app()
    chat = Chat(busy=False, exits=None)
    chats(app).sessions[KEY] = chat
    spawn = pty_spawner(app)
    with caplog.at_level(logging.WARNING, logger=session_handoff.__name__):
        result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert result.spawned is True
    assert "cannot be observed dead" in caplog.text
    assert app.clock.sleeps == []


# ---------------------------------------------------------------------------
# Hand-off: PTY out, chat in
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_handoff_pops_terminates_off_the_loop_and_resets_the_turn_state():
    app = make_app()
    pty = pool_pty(app)
    set_store(app, IDLE, 1.0)
    spawn = chat_spawner(app)
    terminate_threads: list[str] = []
    real_terminate = pty.terminate

    def terminate() -> None:
        import threading

        terminate_threads.append(threading.current_thread().name)
        real_terminate()

    pty.terminate = terminate  # type: ignore[method-assign]
    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert result.plan.action == ACTION_HANDOFF
    assert result.outcome == WaitOutcome(REASON_IDLE)
    assert pty.terminates == 1
    assert terminate_threads and "MainThread" not in terminate_threads
    assert registry(app).get_session(KEY) is None
    assert not registry(app).is_attached(KEY)
    assert turn_state(app) == IDLE
    assert len(spawn.calls) == 1
    assert chats(app).get(KEY) is result.session
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_forced_outcome_still_pops_terminates_and_checks_death():
    """Phase (b) terminated once on the grace expiring; (c) does not trust it."""
    app = make_app(hook=False)
    pty = pool_pty(app)
    set_store(app, BUSY, 1.0)
    spawn = chat_spawner(app)
    result = await acquire_surface(app, KEY, "simple", object(), interrupt=True, spawn=spawn)

    assert result.outcome == WaitOutcome(REASON_FORCED)
    assert pty.terminates == 2
    assert registry(app).get_session(KEY) is None
    assert turn_state(app) == IDLE
    assert result.spawned is True


@pytest.mark.usefixtures("disk")
async def test_an_exited_outcome_reaps_the_corpse_and_spawns():
    app = make_app()
    pty = pool_pty(app)
    set_store(app, BUSY, 1.0)

    def dies():
        pty._alive = False

    spawn = chat_spawner(app)
    with wait_returns(REASON_EXITED, before=dies):
        result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert pty.terminates == 1
    assert registry(app).get_session(KEY) is None
    assert turn_state(app) == IDLE
    assert result.spawned is True
    assert app.clock.sleeps == []


@pytest.mark.usefixtures("disk")
async def test_a_pty_that_survives_is_reinserted_unattached_and_refused_503():
    app = make_app()
    pty = SurvivorPty()
    pool_pty(app, pty)
    set_store(app, IDLE, 1.0)
    set_store(app, BUSY, 2.0)
    set_store(app, IDLE, 3.0)
    spawn = chat_spawner(app)
    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert refused.value.status == 503
    assert refused.value.error == ERROR_OUTGOING_STILL_RUNNING
    assert pty.terminates == 1
    assert registry(app).get_session(KEY) is pty
    assert not registry(app).is_attached(KEY)
    assert not registry(app).is_reserved(KEY)
    assert spawn.calls == []
    assert chats(app).get(KEY) is None
    # Not torn down, so not a teardown edge: the store is left alone.
    assert turn_state(app) == IDLE and get_turn_state(app, KEY)["ts"] == 3.0
    assert_released(app)


# ---------------------------------------------------------------------------
# The premise no longer holds
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_an_outgoing_pty_replaced_mid_wait_is_an_error_and_kills_nothing():
    app = make_app()
    pty = pool_pty(app)
    set_store(app, IDLE, 1.0)
    impostor = RecordingPty()

    def replace():
        registry(app).pop_session(KEY)
        pool_pty(app, impostor)

    spawn = chat_spawner(app)
    with wait_returns(REASON_IDLE, before=replace), pytest.raises(HandoffError) as failed:
        await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert failed.value.error == ERROR_OUTGOING_VANISHED
    assert pty.terminates == 0 and impostor.terminates == 0
    assert registry(app).get_session(KEY) is impostor
    assert spawn.calls == []
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_an_outgoing_chat_replaced_mid_wait_is_an_error():
    app = make_app()
    chats(app).sessions[KEY] = Chat(busy=False)
    impostor = Chat(busy=False)

    def replace():
        chats(app).sessions[KEY] = impostor

    spawn = pty_spawner(app)
    with wait_returns(REASON_IDLE, before=replace), pytest.raises(HandoffError) as failed:
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert failed.value.error == ERROR_OUTGOING_VANISHED
    assert chats(app).terminated == [] and impostor.teardowns == 0
    assert spawn.calls == []
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_an_outgoing_entry_already_gone_leaves_nothing_to_tear_down():
    app = make_app()
    pty = pool_pty(app)
    set_store(app, IDLE, 1.0)

    def gone():
        registry(app).pop_session(KEY)

    spawn = chat_spawner(app)
    with wait_returns(REASON_IDLE, before=gone):
        result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)
    # The reference (c) kept is still reaped; the key was free for the spawn.
    assert pty.terminates == 1
    assert result.spawned is True
    assert_released(app)


# ---------------------------------------------------------------------------
# The spawn callback
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_other_spawn_failures_escape_unchanged_after_the_release():
    app = make_app()
    chat = Chat(busy=False)
    chats(app).sessions[KEY] = chat

    async def spawn(_request):
        raise OSError("fork failed")

    with pytest.raises(OSError, match="fork failed"):
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    # The outgoing chat was already torn down; the key is empty and free.
    assert chat.teardowns == 1
    assert chats(app).get(KEY) is None
    assert registry(app).get_session(KEY) is None
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_spawn_that_pools_nothing_is_an_error():
    app = make_app()

    async def spawn(_request):
        return RecordingPty()

    with pytest.raises(HandoffError) as failed:
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert failed.value.error == ERROR_SPAWN_NOT_POOLED
    assert not registry(app).is_attached(KEY)
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_an_attachment_taken_meanwhile_is_the_409_refusal():
    app = make_app()
    intruder = object()

    async def spawn(request):
        pty = pool_pty(app)
        registry(app).attach_session(request.key, intruder)
        return pty

    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)
    assert refused.value.status == 409
    assert refused.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE
    assert registry(app).attached_owner(KEY) is intruder
    assert_released(app)


# ---------------------------------------------------------------------------
# Shield and lock
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_a_cancellation_mid_phase_lets_the_phase_finish():
    app = make_app()
    chat = Chat(busy=False)
    chats(app).sessions[KEY] = chat
    gate = asyncio.Event()
    channel = object()
    spawn = pty_spawner(app, gate=gate)
    task = asyncio.create_task(acquire_surface(app, KEY, "expert", channel, spawn=spawn))
    await until(lambda: len(spawn.calls) == 1)
    assert chat.teardowns == 1

    task.cancel()
    # Cancelled, but not done: the phase is still behind the gate.
    await asyncio.sleep(0.01)
    assert not task.done()
    assert registry(app).get_session(KEY) is None

    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    # The spawn landed, the PTY is pooled and attached, the slot is released.
    session = registry(app).get_session(KEY)
    assert session is not None
    assert registry(app).attached_owner(KEY) is channel
    assert_released(app)
    # The caller's own cleanup detaches with its token, as a handler would.
    registry(app).detach_session(KEY, channel)
    assert not registry(app).is_attached(KEY)


@pytest.mark.usefixtures("disk")
async def test_a_failure_after_the_caller_was_cancelled_is_logged_not_lost(caplog):
    """The phase's exception is retrieved by its own callback and logged."""
    app = make_app()
    chats(app).sessions[KEY] = Chat(busy=False)
    gate = asyncio.Event()
    calls: list[SpawnRequest] = []

    async def spawn(request: SpawnRequest):
        calls.append(request)
        await gate.wait()
        raise OSError("fork failed")

    task = asyncio.create_task(acquire_surface(app, KEY, "expert", object(), spawn=spawn))
    await until(lambda: len(calls) == 1)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()

    with caplog.at_level(logging.WARNING, logger=session_handoff.__name__):
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
    assert "failed after its caller was cancelled: fork failed" in caplog.text
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_the_phase_holds_the_key_lock_so_a_second_acquire_waits():
    app = make_app()
    gate = asyncio.Event()
    first, second = object(), object()
    spawn = pty_spawner(app, gate=gate)
    task = asyncio.create_task(acquire_surface(app, KEY, "expert", first, spawn=spawn))
    await until(lambda: len(spawn.calls) == 1)
    assert get_state(app).lock_for(KEY).locked()

    # The second blocks on the lock itself: not one look at the pools yet.
    other = asyncio.create_task(acquire(app, KEY, "expert", second))
    await asyncio.sleep(0.02)
    assert not other.done()
    assert app.clock.sleeps == []

    gate.set()
    await task
    result = await other
    # ...and then takes the PTY over from the first, as the newer connection.
    assert result.plan.action == ACTION_TAKEOVER
    assert registry(app).attached_owner(KEY) is second
    assert_released(app)
