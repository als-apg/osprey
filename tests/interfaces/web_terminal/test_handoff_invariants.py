"""The invariants ``acquire_surface`` keeps, put under adversarial pressure.

The phase suites next to this one each pin one phase's own behaviour. This one
pins the properties that have to survive *across* the phases, when a caller
goes away half way, a kill does not take, two connections want the same key at
once, or the pools are pushed past their capacity while a hand-off is in
flight:

- **one live process per key** — however the surfaces are interleaved, and
  whatever the eviction pass and the dead-entry discard do meanwhile, the two
  pools never both hold a live entry under the key;
- **pending slot held ⇔ key reserved** — the two are taken together and
  dropped together, so no look at the key ever sees one without the other;
- **the slot is released exactly once on every exit path** — a return, each
  refusal, a channel that closes, a spawn that raises, a cancellation;
- **attach before release** — an Expert caller owns the PTY before the
  reservation that protected it is dropped, so the eviction pass never meets
  the entry unheld;
- **no spawn after a refusal** — a refused acquire starts no process;
- **a survivor is re-pooled unattached** — a kill that did not take leaves an
  ordinary holder the next acquire meets and kills again;
- **the turn-state store is reset on both PTY edges**, and the resume id is
  chosen from a snapshot taken at spawn time, not at plan time.

Time is faked through the state's injected ``clock``/``sleep`` as in the phase
suites, whose fakes this module reuses; the few tests that are about the event
loop staying responsive spend real milliseconds and say so.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal import session_handoff
from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.session_handoff import (
    ACTION_HANDOFF,
    ACTION_REUSE,
    ACTION_SPAWN,
    ACTION_TAKEOVER,
    ATTACH_POLL_S,
    ERROR_HANDOFF_SUPERSEDED,
    ERROR_OUTGOING_STILL_RUNNING,
    ERROR_SESSION_ATTACHED_ELSEWHERE,
    IDLE_POLL_S,
    REASON_IDLE,
    REASON_INTERRUPTED,
    ChannelClosed,
    ChannelToken,
    HandoffNeedsInterrupt,
    HandoffRefused,
    WaitOutcome,
    acquire_surface,
    get_state,
)
from osprey.interfaces.web_terminal.turn_state import BUSY, IDLE, get_turn_state
from tests.interfaces.web_terminal._fakes import PoolChatSession
from tests.interfaces.web_terminal._handoff_harness import (
    KEY,
    OTHER_KEY,
    THIRD_KEY,
    TRANSCRIPT,
    Chat,
    RecordingPty,
    SurvivorPty,
    assert_held,
    assert_released,
    chat_spawner,
    chats,
    disk,  # noqa: F401  (fixture, requested by name)
    make_app,
    pool_pty,
    pty_spawner,
    registry,
    set_store,
    ticks,
    wait_returns,
)

# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class StubbornPty(RecordingPty):
    """Survives its first kill; the second one takes.

    The shape of a ``terminate`` whose signal escalation ran out before the
    child was reaped: the acquire is refused with 503 and the next one meets
    the survivor as an ordinary holder.
    """

    def terminate(self) -> None:
        self.terminates += 1
        if self.terminates > 1:
            self._alive = False


def live_entries(app: SimpleNamespace, key: str = KEY) -> int:
    """How many pools hold a live entry under *key* — the number this module guards."""
    pty = registry(app).get_session(key)
    chat = chats(app).get(key)
    return int(pty is not None) + int(chat is not None)


def pool_pty_under(app: SimpleNamespace, key: str, pty: RecordingPty) -> RecordingPty:
    """``pool_pty`` for a key other than the one under test."""
    reg = registry(app)
    with patch.object(reg, "_spawn_session", return_value=pty):
        session, reused = reg.get_or_create_session(key, ["fake"])
    assert session is pty and not reused
    return pty


def watch_slot_invariant(app: SimpleNamespace) -> list[str]:
    """Check "slot held ⇔ key reserved" on every look the fake clock takes."""
    seen: list[str] = []

    def probe(_count: int) -> None:
        pending = KEY in get_state(app).pending
        reserved = registry(app).is_reserved(KEY)
        if pending != reserved:
            seen.append(f"pending={pending} reserved={reserved}")

    app.clock.on_sleep.append(probe)
    return seen


def watch_attach(app: SimpleNamespace) -> list[bool]:
    """Whether the pending slot still stood at each ``attach_session``."""
    reg = registry(app)
    real = reg.attach_session
    held: list[bool] = []

    def attach(key: str, owner: object) -> bool:
        held.append(key in get_state(app).pending)
        return real(key, owner)

    reg.attach_session = attach  # type: ignore[method-assign]
    return held


@contextlib.contextmanager
def releases():
    """Collect the keys whose pending slot was actually released."""
    real = session_handoff.release_pending
    taken: list[str] = []

    def release(app, key: str, channel: object) -> bool:
        done = real(app, key, channel)
        if done:
            taken.append(key)
        return done

    with patch.object(session_handoff, "release_pending", release):
        yield taken


def real_chat_pool(app: SimpleNamespace) -> tuple[ChatSessionPool, list[PoolChatSession]]:
    """Put a real ``ChatSessionPool`` on *app*, over a fake session factory."""
    created: list[PoolChatSession] = []

    def factory(cwd: str, env, session_key: str) -> PoolChatSession:
        session = PoolChatSession(cwd=cwd, env=env, session_key=session_key)
        created.append(session)
        return session

    pool = ChatSessionPool(factory=factory, max_sessions=5)
    app.state.operator_registry.chats = pool
    return pool, created


def turn_state(app: SimpleNamespace) -> dict | None:
    return get_turn_state(app, KEY)


# ---------------------------------------------------------------------------
# One live process per key
# ---------------------------------------------------------------------------


async def test_interleaved_acquires_never_leave_two_live_entries(disk):  # noqa: F811
    """Six flips in a row, on a key whose transcript a ``/clear`` already moved."""
    app = make_app()
    app.state.transcript_map[KEY] = TRANSCRIPT
    disk.add(TRANSCRIPT)
    pty_spawn = pty_spawner(app)
    chat_spawn = chat_spawner(app)
    flips = [
        ("expert", pty_spawn),
        ("simple", chat_spawn),
        ("simple", chat_spawn),
        ("expert", pty_spawn),
        ("expert", pty_spawn),
        ("simple", chat_spawn),
    ]

    spawned: list[bool] = []
    for surface, spawn in flips:
        channel = object()
        result = await acquire_surface(app, KEY, surface, channel, spawn=spawn)
        assert live_entries(app) == 1
        assert_released(app)
        spawned.append(result.spawned)
        if surface == "expert":
            # What a terminal handler does in its own cleanup.
            registry(app).detach_session(KEY, channel)

    assert spawned == [True, True, False, True, False, True]
    # Every spawn resumed the key's current transcript, not the key itself.
    assert {call.resume_id for call in pty_spawn.calls} == {TRANSCRIPT}
    assert {call.resume_id for call in chat_spawn.calls} == {TRANSCRIPT}
    assert {call.transcript_id for call in chat_spawn.calls} == {TRANSCRIPT}


@pytest.mark.usefixtures("disk")
async def test_a_dead_entry_is_reaped_rather_than_handed_off():
    app = make_app()
    corpse = pool_pty(app)
    corpse.exit(1)
    spawn = chat_spawner(app)

    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert corpse.terminates == 1
    assert result.plan.outgoing is None
    assert chats(app).get(KEY) is result.session
    assert live_entries(app) == 1
    assert_released(app)


@pytest.mark.usefixtures("disk")
@pytest.mark.parametrize("surface", ["expert", "simple"])
async def test_a_key_live_in_both_pools_is_reduced_to_one(surface):
    """The state this module exists to make unreachable, met head on.

    Whichever view acquires, the other view's entry is the one torn down and
    the acquiring view's own entry is what it is handed.
    """
    app = make_app()
    pty = pool_pty(app)
    chat = Chat(busy=False)
    chats(app).sessions[KEY] = chat
    set_store(app, IDLE, 1.0)
    assert live_entries(app) == 2
    spawn = pty_spawner(app) if surface == "expert" else chat_spawner(app)

    result = await acquire_surface(app, KEY, surface, object(), spawn=spawn)

    assert result.plan.action == ACTION_HANDOFF
    assert result.spawned is False and spawn.calls == []
    if surface == "expert":
        assert result.plan.outgoing is not None and result.plan.outgoing.session is chat
        assert chat.teardowns == 1 and chats(app).get(KEY) is None
        assert result.session is pty
    else:
        assert result.plan.outgoing is not None and result.plan.outgoing.session is pty
        assert pty.terminates == 1 and registry(app).get_session(KEY) is None
        assert result.session is chat and chat.teardowns == 0
    assert live_entries(app) == 1
    assert_released(app)


# ---------------------------------------------------------------------------
# Slot held ⇔ key reserved, and the eviction pass
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_the_slot_and_the_reservation_are_taken_and_dropped_together():
    app = make_app()
    chat = Chat(busy=True)
    chats(app).sessions[KEY] = chat
    violations = watch_slot_invariant(app)
    channel = object()

    task = asyncio.create_task(acquire_surface(app, KEY, "expert", channel, spawn=pty_spawner(app)))
    await ticks(app, 10)
    assert_held(app, channel)
    chat.is_busy = False
    await task

    assert violations == []
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_the_eviction_pass_steps_over_the_key_a_handoff_reserved():
    app = make_app()
    app.state.pty_registry = PtyRegistry(max_background=2)
    outgoing = pool_pty(app)
    spare = pool_pty_under(app, OTHER_KEY, RecordingPty())
    set_store(app, BUSY, 1.0)

    task = asyncio.create_task(
        acquire_surface(app, KEY, "simple", object(), spawn=chat_spawner(app))
    )
    await ticks(app, 3)
    assert registry(app).is_reserved(KEY)

    # A third session takes the pool past capacity while the wait runs.
    pool_pty_under(app, THIRD_KEY, RecordingPty())
    assert registry(app).get_session(KEY) is outgoing
    assert outgoing.terminates == 0
    assert registry(app).get_session(OTHER_KEY) is None
    assert spare.terminates == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_released(app)


# ---------------------------------------------------------------------------
# Every exit path releases the slot exactly once
# ---------------------------------------------------------------------------


async def _exit_returns(app: SimpleNamespace) -> None:
    chats(app).sessions[KEY] = Chat(busy=False)
    await acquire_surface(app, KEY, "expert", object(), spawn=pty_spawner(app))


async def _exit_refused_409(app: SimpleNamespace) -> None:
    pool_pty(app)
    registry(app).attach_session(KEY, object())
    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "simple", object(), spawn=chat_spawner(app))
    assert refused.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE


async def _exit_refused_503(app: SimpleNamespace) -> None:
    pool_pty(app, SurvivorPty())
    set_store(app, IDLE, 1.0)
    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "simple", object(), spawn=chat_spawner(app))
    assert refused.value.error == ERROR_OUTGOING_STILL_RUNNING


async def _exit_needs_interrupt(app: SimpleNamespace) -> None:
    pool_pty(app)
    with pytest.raises(HandoffNeedsInterrupt):
        await acquire_surface(app, KEY, "simple", object(), spawn=chat_spawner(app))


async def _exit_channel_closed(app: SimpleNamespace) -> None:
    chats(app).sessions[KEY] = Chat(busy=True)
    closed = asyncio.Event()
    token = ChannelToken(closed.is_set)
    task = asyncio.create_task(acquire_surface(app, KEY, "expert", token, spawn=pty_spawner(app)))
    await ticks(app, 3)
    closed.set()
    with pytest.raises(ChannelClosed):
        await task


async def _exit_cancelled(app: SimpleNamespace) -> None:
    chats(app).sessions[KEY] = Chat(busy=True)
    task = asyncio.create_task(
        acquire_surface(app, KEY, "expert", object(), spawn=pty_spawner(app))
    )
    await ticks(app, 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _exit_spawn_raised(app: SimpleNamespace) -> None:
    chats(app).sessions[KEY] = Chat(busy=False)

    async def spawn(_request):
        raise OSError("fork failed")

    with pytest.raises(OSError, match="fork failed"):
        await acquire_surface(app, KEY, "expert", object(), spawn=spawn)


#: Each exit path and how many slots it should release. One, for every exit
#: that got as far as taking the slot — and none for the 409 the attach grace
#: ends in, which is refused before phase (a) registers anything at all.
EXIT_PATHS = {
    "returns": (_exit_returns, 1),
    "refused_409": (_exit_refused_409, 0),
    "refused_503": (_exit_refused_503, 1),
    "needs_interrupt": (_exit_needs_interrupt, 1),
    "channel_closed": (_exit_channel_closed, 1),
    "cancelled": (_exit_cancelled, 1),
    "spawn_raised": (_exit_spawn_raised, 1),
}


@pytest.mark.usefixtures("disk")
@pytest.mark.parametrize("name", list(EXIT_PATHS))
async def test_the_pending_slot_is_released_exactly_once_on_every_exit(name):
    run, expected = EXIT_PATHS[name]
    app = make_app(hook=name != "needs_interrupt")
    violations = watch_slot_invariant(app)
    with releases() as taken:
        await run(app)
    assert taken == [KEY] * expected
    assert violations == []
    assert_released(app)


# ---------------------------------------------------------------------------
# No spawn after a refusal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
@pytest.mark.parametrize("name", ["refused_409", "refused_503", "needs_interrupt"])
async def test_a_refused_acquire_starts_no_process(name):
    app = make_app(hook=name != "needs_interrupt")
    if name == "refused_409":
        pool_pty(app)
        registry(app).attach_session(KEY, object())
    elif name == "refused_503":
        pool_pty(app, SurvivorPty())
        set_store(app, IDLE, 1.0)
    else:
        pool_pty(app)
    spawn = chat_spawner(app)

    with pytest.raises(HandoffRefused):
        await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert spawn.calls == []
    assert chats(app).get(KEY) is None
    assert live_entries(app) == 1
    assert_released(app)


# ---------------------------------------------------------------------------
# A kill that did not take
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_a_pty_that_survives_once_is_repooled_and_the_retry_leaves_one_child():
    app = make_app()
    pty = pool_pty(app, StubbornPty())
    set_store(app, IDLE, 1.0)
    spawn = chat_spawner(app)

    with pytest.raises(HandoffRefused) as refused:
        await acquire_surface(app, KEY, "simple", object(), spawn=spawn)
    assert refused.value.status == 503
    assert registry(app).get_session(KEY) is pty
    assert not registry(app).is_attached(KEY)
    assert not registry(app).is_reserved(KEY)
    assert spawn.calls == []

    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert pty.terminates == 2
    assert not pty.is_alive
    assert len(spawn.calls) == 1
    assert chats(app).get(KEY) is result.session
    assert live_entries(app) == 1
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_the_spawn_waits_for_the_kill_to_return_and_the_child_to_be_gone():
    """The order a hand-off depends on: pop, kill, see it dead, only then spawn."""
    app = make_app()
    order: list[str] = []
    pty = SurvivorPty()
    pool_pty(app, pty)
    set_store(app, IDLE, 1.0)
    real_terminate = pty.terminate

    def terminate() -> None:
        order.append("terminate")
        real_terminate()

    pty.terminate = terminate  # type: ignore[method-assign]

    def maybe_die(count: int) -> None:
        if count >= 3 and pty.is_alive:
            order.append("dead")
            pty.die()

    app.clock.on_sleep.append(maybe_die)

    async def spawn(request):
        order.append("spawn")
        assert registry(app).get_session(request.key) is None
        assert not pty.is_alive
        session = Chat(busy=False)
        chats(app).sessions[request.key] = session
        return session

    await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert order == ["terminate", "dead", "spawn"]


# ---------------------------------------------------------------------------
# Attachment: cleared, never cleared, and taken over
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_an_attachment_that_clears_within_the_grace_becomes_a_handoff():
    app = make_app()
    pty = pool_pty(app)
    owner = object()
    assert registry(app).attach_session(KEY, owner)
    set_store(app, IDLE, 1.0)
    spawn = chat_spawner(app)

    # Six looks at 50 ms is the 300 ms a socket takes to notice it has gone.
    def detach_on_the_sixth_look(count: int) -> None:
        if count == 6:
            registry(app).detach_session(KEY, owner)

    app.clock.on_sleep.append(detach_on_the_sixth_look)
    result = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    # Six polls of the grace, and nothing else: the key was idle by then.
    assert app.clock.sleeps == [ATTACH_POLL_S] * 6
    assert result.plan.action == ACTION_HANDOFF
    assert pty.terminates == 1
    assert chats(app).get(KEY) is result.session
    assert live_entries(app) == 1
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_the_expert_owns_the_pty_before_the_reservation_is_dropped():
    app = make_app()
    chats(app).sessions[KEY] = Chat(busy=False)
    held = watch_attach(app)
    channel = object()

    result = await acquire_surface(app, KEY, "expert", channel, spawn=pty_spawner(app))

    assert held == [True]
    assert registry(app).attached_owner(KEY) is channel
    assert result.spawned is True
    assert_released(app)


# ---------------------------------------------------------------------------
# The same surface already holds the key
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_a_hookless_deployment_still_waits_on_a_busy_chat_for_the_expert():
    """The hook only speaks for a PTY; ``is_busy`` needs no hook at all."""
    app = make_app(hook=False)
    chat = Chat(busy=True)
    chats(app).sessions[KEY] = chat
    spawn = pty_spawner(app)

    task = asyncio.create_task(acquire_surface(app, KEY, "expert", object(), spawn=spawn))
    await ticks(app, 20)
    assert not task.done()
    assert chat.teardowns == 0 and spawn.calls == []
    assert set(app.clock.sleeps) == {IDLE_POLL_S}

    chat.is_busy = False
    result = await task

    assert result.plan.action == ACTION_HANDOFF
    assert chat.teardowns == 1 and result.spawned is True
    assert live_entries(app) == 1
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_channel_that_closes_mid_wait_tears_nothing_down():
    """The chat wait's own channel probe ends it: the terminal socket suite cannot see this.

    A socket that leaves is also cancelled by its server, so the websocket
    suite passes without the probe; this is the proof the chat wait asks the
    caller's channel on its own.
    """
    app = make_app()
    chat = Chat(busy=True)
    chats(app).sessions[KEY] = chat
    closed = asyncio.Event()
    token = ChannelToken(closed.is_set)
    spawn = pty_spawner(app)

    task = asyncio.create_task(acquire_surface(app, KEY, "expert", token, spawn=spawn))
    await ticks(app, 5)
    closed.set()
    with pytest.raises(ChannelClosed):
        await task

    assert chat.teardowns == 0 and chat.cancels == 0
    assert chats(app).terminated == []
    assert spawn.calls == []
    assert chats(app).get(KEY) is chat
    assert live_entries(app) == 1
    assert_released(app)


# ---------------------------------------------------------------------------
# The turn-state store and the resume id
# ---------------------------------------------------------------------------


async def test_both_pty_edges_reset_the_turn_state_under_the_moved_transcript(disk):  # noqa: F811
    app = make_app()
    app.state.transcript_map[KEY] = TRANSCRIPT
    disk.add(TRANSCRIPT)

    # Spawn edge: a fresh TUI has no turn in flight.
    channel = object()
    await acquire_surface(app, KEY, "expert", channel, spawn=pty_spawner(app))
    spawned = turn_state(app)
    assert spawned is not None
    assert spawned["state"] == IDLE and spawned["transcript_id"] == TRANSCRIPT
    registry(app).detach_session(KEY, channel)

    # The TUI runs a turn, then the Simple view takes the key.
    set_store(app, BUSY, spawned["ts"] + 1.0)
    with wait_returns(REASON_IDLE):
        await acquire_surface(app, KEY, "simple", object(), spawn=chat_spawner(app))

    torn_down = turn_state(app)
    assert torn_down is not None
    assert torn_down["state"] == IDLE and torn_down["transcript_id"] == TRANSCRIPT


async def test_the_resume_id_comes_from_a_snapshot_taken_at_spawn_time(disk):  # noqa: F811
    """A ``/clear`` that lands while the wait runs is what the new process opens."""
    app = make_app()
    chats(app).sessions[KEY] = Chat(busy=False)
    disk.add(KEY)

    def clears():
        app.state.transcript_map[KEY] = TRANSCRIPT
        disk.add(TRANSCRIPT)

    spawn = pty_spawner(app)
    with wait_returns(REASON_IDLE, before=clears):
        result = await acquire_surface(app, KEY, "expert", object(), spawn=spawn)

    assert result.resume_id == TRANSCRIPT
    assert spawn.calls[0].transcript_id == TRANSCRIPT


# ---------------------------------------------------------------------------
# Through the real chat pool
# ---------------------------------------------------------------------------


async def test_a_simple_round_trip_through_the_real_chat_pool(disk):  # noqa: F811
    """The Simple side end to end: the acquire door onto the real pool.

    Everything else here drives a fake pool, so the seam between the two — the
    spawn callback handing ``request.resume_id`` to
    ``ChatSessionPool.get_or_create`` and returning what the pool then holds —
    is only exercised by this test. The child under it is a fake; the pool,
    its reuse check and its teardown are the real ones.
    """
    app = make_app()
    pool, created = real_chat_pool(app)
    app.state.transcript_map[KEY] = TRANSCRIPT
    disk.add(TRANSCRIPT)
    spawns: list[str | None] = []

    async def spawn(request):
        spawns.append(request.resume_id)
        session, _reused = await pool.get_or_create(
            request.key, "/tmp", resume_id=request.resume_id
        )
        return session

    first = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)
    chat = first.session

    assert first.plan.action == ACTION_SPAWN
    assert first.spawned is True and first.resume_id == TRANSCRIPT
    assert spawns == [TRANSCRIPT]
    assert pool.get(KEY) is chat
    # The key is what the child is pooled and audited under; the transcript is
    # only which conversation it continues.
    assert chat.session_key == KEY
    assert chat.resume_id == TRANSCRIPT
    assert created == [chat]
    assert_released(app)

    # The next turn is handed the same child, and the pool is never asked.
    second = await acquire_surface(app, KEY, "simple", object(), spawn=spawn)

    assert second.session is chat
    assert second.spawned is False and second.resume_id is None
    assert second.outcome == WaitOutcome(REASON_IDLE)
    assert spawns == [TRANSCRIPT]
    assert chat.start_calls == 1 and chat.stop_calls == 0
    assert len(created) == 1
    assert_released(app)

    # The Expert view takes the key: the real pool tears the child down.
    third = await acquire_surface(app, KEY, "expert", object(), spawn=pty_spawner(app))

    assert third.plan.action == ACTION_HANDOFF
    assert chat.stop_calls == 1 and chat.process_exited is True
    assert pool.get(KEY) is None and not pool.has_key(KEY)
    assert registry(app).get_session(KEY) is third.session
    assert third.resume_id == TRANSCRIPT
    assert live_entries(app) == 1
    assert_released(app)


# ---------------------------------------------------------------------------
# Property-style: two acquirers, and a cancel at every await point
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
@pytest.mark.parametrize("stagger", range(6))
async def test_two_expert_acquirers_leave_one_child_and_one_owner(stagger):
    app = make_app()
    spawn = pty_spawner(app)
    first, second = object(), object()

    one = asyncio.create_task(acquire_surface(app, KEY, "expert", first, spawn=spawn))
    for _ in range(stagger):
        await asyncio.sleep(0)
    two = asyncio.create_task(acquire_surface(app, KEY, "expert", second, spawn=spawn))
    results = await asyncio.gather(one, two)

    # One child was started, both callers were handed it, and the attachment
    # belongs to whichever of them looked at the key last.
    assert len(spawn.calls) == 1
    assert live_entries(app) == 1
    assert {r.session for r in results} == {registry(app).get_session(KEY)}
    assert sum(r.spawned for r in results) == 1
    assert registry(app).attached_owner(KEY) in (first, second)
    assert [r.plan.action for r in results].count(ACTION_TAKEOVER) == 1
    assert_released(app)


@pytest.mark.usefixtures("disk")
@pytest.mark.parametrize("yields", range(26))
async def test_a_cancel_at_any_point_leaves_the_key_consistent(yields):
    app = make_app()
    chat = Chat(busy=False)
    chats(app).sessions[KEY] = chat
    channel = object()
    spawn = pty_spawner(app)

    task = asyncio.create_task(acquire_surface(app, KEY, "expert", channel, spawn=spawn))
    for _ in range(yields):
        await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    assert_released(app)
    assert live_entries(app) <= 1
    # The outgoing chat is never torn down without a replacement in the key.
    if chat.teardowns:
        assert registry(app).get_session(KEY) is not None
        assert registry(app).attached_owner(KEY) is channel
    else:
        assert chats(app).get(KEY) is chat
    # Nothing was started that the pool does not hold.
    assert len(spawn.calls) == int(registry(app).get_session(KEY) is not None)


@pytest.mark.usefixtures("disk")
async def test_a_cancelled_acquire_leaves_the_key_free_for_the_next_one():
    app = make_app()
    chat = Chat(busy=True)
    chats(app).sessions[KEY] = chat
    doomed = asyncio.create_task(
        acquire_surface(app, KEY, "expert", object(), spawn=pty_spawner(app))
    )
    await ticks(app, 3)
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed

    chat.is_busy = False
    result = await acquire_surface(app, KEY, "expert", object(), spawn=pty_spawner(app))

    assert result.plan.action == ACTION_HANDOFF
    assert result.outcome == WaitOutcome(REASON_IDLE)
    assert live_entries(app) == 1
    assert_released(app)


# ---------------------------------------------------------------------------
# A Simple interrupt supersedes the Simple wait ahead of it
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("disk")
async def test_a_simple_interrupt_supersedes_the_pending_simple_wait_and_lands():
    """The wait the operator wants to end is ended; the interrupt's own acquire completes.

    Through the real phase (c): the superseded wait releases its slot exactly
    once and starts nothing, the superseding acquire cuts the turn short,
    tears the terminal down and spawns the chat, and at no look are there two
    pending records or two live entries under the key.
    """
    app = make_app()
    pty = pool_pty(app)
    set_store(app, BUSY, 1.0)
    violations = watch_slot_invariant(app)
    slots: list[int] = []
    entries: list[int] = []

    def count(_n: int) -> None:
        slots.append(len(get_state(app).pending))
        entries.append(live_entries(app))
        if pty.writes:
            set_store(app, IDLE, 2.0)

    app.clock.on_sleep.append(count)
    first_channel, second_channel = ChannelToken(), ChannelToken()
    first_spawn, second_spawn = chat_spawner(app), chat_spawner(app)

    with releases() as taken:
        first = asyncio.create_task(
            acquire_surface(app, KEY, "simple", first_channel, spawn=first_spawn)
        )
        await ticks(app, 5)
        assert_held(app, first_channel)
        second = asyncio.create_task(
            acquire_surface(app, KEY, "simple", second_channel, interrupt=True, spawn=second_spawn)
        )
        with pytest.raises(HandoffRefused) as refused:
            await first
        assert refused.value.error == ERROR_HANDOFF_SUPERSEDED
        assert first_spawn.calls == []
        result = await second

    assert result.spawned and result.plan.interrupt
    assert result.outcome.reason == REASON_INTERRUPTED
    assert len(second_spawn.calls) == 1
    assert pty.writes == [b"\x1b"]
    assert pty.terminates == 1
    assert registry(app).get_session(KEY) is None
    assert chats(app).get(KEY) is result.session
    assert live_entries(app) == 1
    assert taken == [KEY, KEY]
    assert violations == []
    assert max(slots) <= 1
    assert max(entries) <= 1
    assert_released(app)


@pytest.mark.usefixtures("disk")
async def test_a_supersede_arriving_during_phase_c_leaves_the_carrier_to_answer():
    """A call already carrying the key out is not cancelled; the interrupt reuses what it pooled.

    The first acquire has left its wait and is spawning the chat when the
    interrupt looks at the key — the look lands in the step between phase
    (b) returning and the shielded phase (c) task taking the lock. It is
    plainly blocked, not a supersede: the first caller is answered, the
    second finds the pooled idle chat once the lock is free and is handed it
    back without a spawn of its own.
    """
    app = make_app()
    violations = watch_slot_invariant(app)
    gate = asyncio.Event()
    first_channel, second_channel = ChannelToken(), ChannelToken()
    first_spawn = chat_spawner(app, gate=gate)
    second_spawn = chat_spawner(app)

    with releases() as taken:
        first = asyncio.create_task(
            acquire_surface(app, KEY, "simple", first_channel, spawn=first_spawn)
        )
        second = asyncio.create_task(
            acquire_surface(app, KEY, "simple", second_channel, interrupt=True, spawn=second_spawn)
        )
        # One step: the first is at its shield, the spawn is waiting on the
        # gate, and the second has looked — and found a record it may not touch.
        await asyncio.sleep(0)
        pending = get_state(app).pending[KEY]
        assert pending.channel is first_channel
        assert not pending.waiting
        assert not pending.superseded
        assert not first.done() and not second.done()

        gate.set()
        landed = await first
        reused = await second

    assert landed.spawned and landed.resume_id is None
    assert not reused.spawned
    assert reused.session is landed.session
    assert reused.plan.action == ACTION_REUSE
    assert len(first_spawn.calls) == 1 and second_spawn.calls == []
    assert chats(app).get(KEY) is landed.session
    assert live_entries(app) == 1
    assert taken == [KEY, KEY]
    assert violations == []
    assert_released(app)
