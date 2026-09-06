"""Phase (a) of a surface acquire: inspect, decide, register.

``acquire_surface`` is the one door every spawn walks through, and phase (a)
is where it reads the two pools and the pending slot under the per-key lock
and decides what the incoming surface has to do. These tests pin that
decision table and the two invariants around it:

- one look at a key is atomic (the lock serialises concurrent acquires), and
- a key somebody else is consuming is re-inspected for the attach grace with
  the lock *released* between looks, then refused with 409.

Time is faked: the state's ``clock``/``sleep`` are injected, so the two-second
grace is exercised without being spent. The PTY pool is a real
``PtyRegistry`` holding the fake PTY from ``test_ws_resume_confirm``; the chat
pool is a fake that answers the three questions phase (a) asks it.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from osprey.interfaces.web_terminal import session_handoff
from osprey.interfaces.web_terminal.pty_manager import PtyRegistry
from osprey.interfaces.web_terminal.session_handoff import (
    ACTION_HANDOFF,
    ACTION_REUSE,
    ACTION_SPAWN,
    ACTION_TAKEOVER,
    ATTACH_GRACE_S,
    ATTACH_POLL_S,
    ERROR_SESSION_ATTACHED_ELSEWHERE,
    REASON_NONE,
    WS_CLOSE_OUTGOING_RUNNING,
    WS_CLOSE_SESSION_ATTACHED,
    AcquirePlan,
    HandoffRefused,
    HandoffState,
    WaitOutcome,
    acquire_surface,
    get_state,
    release_pending,
)
from tests.interfaces.web_terminal._fakes import (
    FakeChatPool,
    FakeChatSession,
    FakeClock,
    FakePtySession,
)

KEY = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _phase_b_does_not_wait():
    """These tests are about the decision; the wait it leads to is phase (b)'s own suite.

    Several plans here name a busy outgoing entry — a real phase (b) would wait
    on it — so the wait is stubbed out to return at once. A test that needs a
    particular wait patches ``_wait_for_idle`` itself inside this stub.
    """

    async def no_wait(_app, _plan):
        return WaitOutcome(REASON_NONE)

    with patch.object(session_handoff, "_wait_for_idle", no_wait):
        yield


@pytest.fixture(autouse=True)
def _phase_c_hands_the_plan_back():
    """Phase (c) would carry the plan out — spawn, attach, release the slot.

    These tests assert on the plan and on what phase (a) registered, so (c)
    is stubbed to hand the plan straight back and leave the slot standing.
    Its own suite exercises the real thing.
    """

    async def hand_back(_app, plan, _outcome):
        return plan

    with patch.object(session_handoff, "_phase_c", hand_back):
        yield


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_app(clock: FakeClock | None = None) -> SimpleNamespace:
    clock = clock or FakeClock()
    state = HandoffState(clock=clock, sleep=clock.sleep)
    return SimpleNamespace(
        state=SimpleNamespace(
            pty_registry=PtyRegistry(max_background=5),
            operator_registry=SimpleNamespace(chats=FakeChatPool()),
            handoff=state,
        ),
        clock=clock,
    )


def pool_pty(app: SimpleNamespace, key: str = KEY) -> FakePtySession:
    """Put a live fake PTY into the registry under *key* through the pool path."""
    fake = FakePtySession()
    registry: PtyRegistry = app.state.pty_registry
    with patch.object(registry, "_spawn_session", return_value=fake):
        session, reused = registry.get_or_create_session(key, ["fake"])
    assert session is fake and not reused
    return fake


def chats(app: SimpleNamespace) -> FakeChatPool:
    return app.state.operator_registry.chats


def registry(app: SimpleNamespace) -> PtyRegistry:
    return app.state.pty_registry


def assert_registered(app: SimpleNamespace, plan: AcquirePlan, channel: object) -> None:
    """The pending slot and the reservation an accepted plan leaves behind."""
    pending = get_state(app).pending[plan.key]
    assert pending.channel is channel
    assert pending.surface == plan.surface
    assert pending.task is not None
    assert registry(app).is_reserved(plan.key)
    assert plan.channel is channel


# ---------------------------------------------------------------------------
# Nothing held
# ---------------------------------------------------------------------------


async def test_a_free_key_is_a_spawn_for_either_surface():
    for surface in ("expert", "simple"):
        app = make_app()
        channel = object()
        plan = await acquire_surface(app, KEY, surface, channel)
        assert plan.action == ACTION_SPAWN
        assert plan.outgoing is None
        assert plan.teardown is False
        assert plan.wait_for_idle is False
        assert plan.displaced_owner is None
        assert plan.surface == surface
        assert plan.interrupt is False
        assert_registered(app, plan, channel)
        assert app.clock.sleeps == []


async def test_interrupt_is_carried_into_the_plan():
    app = make_app()
    plan = await acquire_surface(app, KEY, "simple", object(), interrupt=True)
    assert plan.interrupt is True


async def test_state_is_created_on_first_use():
    app = SimpleNamespace(
        state=SimpleNamespace(
            pty_registry=PtyRegistry(max_background=5),
            operator_registry=SimpleNamespace(chats=FakeChatPool()),
        )
    )
    state = get_state(app)
    assert isinstance(state, HandoffState)
    assert get_state(app) is state
    assert app.state.handoff is state
    assert state.lock_for(KEY) is state.lock_for(KEY)


# ---------------------------------------------------------------------------
# Dead entries are discarded, not handed off
# ---------------------------------------------------------------------------


async def test_a_dead_pty_is_popped_and_reaped_and_the_key_is_free():
    app = make_app()
    fake = pool_pty(app)
    fake.exit(1)
    terminated = []
    fake.terminate = lambda: terminated.append(True)  # type: ignore[method-assign]

    plan = await acquire_surface(app, KEY, "simple", object())

    assert plan.action == ACTION_SPAWN
    assert plan.outgoing is None
    assert registry(app).get_session(KEY) is None
    assert terminated == [True]


async def test_a_dead_pty_that_is_still_attached_is_discarded_too():
    """A socket that has not yet noticed its child died holds nothing."""
    app = make_app()
    fake = pool_pty(app)
    stale_token = object()
    assert registry(app).attach_session(KEY, stale_token)
    fake.exit(0)

    plan = await acquire_surface(app, KEY, "simple", object())

    assert plan.action == ACTION_SPAWN
    assert not registry(app).is_attached(KEY)
    assert app.clock.sleeps == []


async def test_a_dead_chat_is_terminated_and_the_key_is_free():
    app = make_app()
    dead = FakeChatSession(active=False)
    chats(app).sessions[KEY] = dead

    plan = await acquire_surface(app, KEY, "expert", object())

    assert plan.action == ACTION_SPAWN
    assert chats(app).terminated == [KEY]
    assert dead.teardowns == 1
    assert chats(app).get(KEY) is None


# ---------------------------------------------------------------------------
# Live entries: the decision table
# ---------------------------------------------------------------------------


async def test_a_live_unattached_pty_is_the_outgoing_entry_for_a_chat():
    app = make_app()
    fake = pool_pty(app)
    channel = object()

    plan = await acquire_surface(app, KEY, "simple", channel)

    assert plan.action == ACTION_HANDOFF
    assert plan.outgoing is not None
    assert plan.outgoing.surface == "expert"
    assert plan.outgoing.session is fake
    assert plan.teardown is True
    assert plan.wait_for_idle is True
    assert plan.displaced_owner is None
    # Phase (a) never touches the outgoing entry: still pooled, still alive.
    assert registry(app).get_session(KEY) is fake
    assert fake.is_alive
    assert_registered(app, plan, channel)


async def test_a_live_chat_is_the_outgoing_entry_for_an_expert():
    app = make_app()
    live = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = live

    plan = await acquire_surface(app, KEY, "expert", object())

    assert plan.action == ACTION_HANDOFF
    assert plan.outgoing is not None
    assert plan.outgoing.surface == "simple"
    assert plan.outgoing.session is live
    assert plan.teardown is True
    assert plan.wait_for_idle is True
    assert chats(app).terminated == []
    assert live.teardowns == 0


async def test_an_expert_reattaching_to_a_free_pty_reuses_it_without_waiting():
    app = make_app()
    fake = pool_pty(app)

    plan = await acquire_surface(app, KEY, "expert", object())

    assert plan.action == ACTION_REUSE
    assert plan.outgoing is not None
    assert plan.outgoing.session is fake
    assert plan.teardown is False
    assert plan.wait_for_idle is False
    assert plan.displaced_owner is None


async def test_a_chat_acquiring_a_chat_held_key_is_ready_with_no_teardown():
    app = make_app()
    live = FakeChatSession(busy=True)
    chats(app).sessions[KEY] = live

    plan = await acquire_surface(app, KEY, "simple", object())

    assert plan.action == ACTION_REUSE
    assert plan.outgoing is not None
    assert plan.outgoing.surface == "simple"
    assert plan.outgoing.session is live
    assert plan.teardown is False
    # The chat route cannot take a turn on a busy session, so (b) waits.
    assert plan.wait_for_idle is True
    assert chats(app).terminated == []


async def test_a_chat_joins_a_chat_creation_still_in_flight():
    app = make_app()
    chats(app).starting.add(KEY)

    plan = await acquire_surface(app, KEY, "simple", object())

    assert plan.action == ACTION_REUSE
    assert plan.outgoing is None
    assert plan.teardown is False
    assert plan.wait_for_idle is False
    assert app.clock.sleeps == []


async def test_an_expert_waits_for_a_chat_creation_to_land_then_hands_off():
    app = make_app()
    chats(app).starting.add(KEY)
    landed = FakeChatSession()

    def land(nth_sleep: int) -> None:
        if nth_sleep == 3:
            chats(app).starting.discard(KEY)
            chats(app).sessions[KEY] = landed

    app.clock.on_sleep.append(land)

    plan = await acquire_surface(app, KEY, "expert", object())

    assert plan.action == ACTION_HANDOFF
    assert plan.outgoing is not None and plan.outgoing.session is landed
    assert len(app.clock.sleeps) == 3


async def test_an_expert_is_refused_while_a_chat_creation_never_lands():
    """A creation still in flight for the whole grace is a holder, not a gap."""
    app = make_app()
    chats(app).starting.add(KEY)

    with pytest.raises(HandoffRefused) as excinfo:
        await acquire_surface(app, KEY, "expert", object())

    assert excinfo.value.status == 409
    assert excinfo.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE
    assert sum(app.clock.sleeps) == pytest.approx(ATTACH_GRACE_S)
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)
    # The creation itself is left to its creator.
    assert chats(app).terminated == []
    assert KEY in chats(app).starting


async def test_a_refusal_after_a_dead_discard_leaves_the_key_clean():
    """The discard in a look stands even when that look ends in a refusal."""
    app = make_app()
    dead = pool_pty(app)
    dead.exit(1)
    chats(app).starting.add(KEY)  # blocks an Expert for the whole grace

    with pytest.raises(HandoffRefused) as excinfo:
        await acquire_surface(app, KEY, "expert", object())

    assert excinfo.value.status == 409
    # The corpse was popped and reaped in the first look, not left for later.
    assert registry(app).get_session(KEY) is None
    assert not registry(app).is_attached(KEY)
    assert not dead.is_alive
    # Nothing of the refused call remains.
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)


async def test_both_pools_live_hands_off_the_other_surface(caplog):
    app = make_app()
    fake = pool_pty(app)
    live = FakeChatSession()
    chats(app).sessions[KEY] = live

    with caplog.at_level("WARNING"):
        as_expert = await acquire_surface(app, KEY, "expert", object())
    assert as_expert.action == ACTION_HANDOFF
    assert as_expert.outgoing is not None and as_expert.outgoing.session is live
    assert "live in both pools" in caplog.text

    release_pending(app, KEY, as_expert.channel)
    as_chat = await acquire_surface(app, KEY, "simple", object())
    assert as_chat.action == ACTION_HANDOFF
    assert as_chat.outgoing is not None and as_chat.outgoing.session is fake


# ---------------------------------------------------------------------------
# Attached PTY: takeover for an Expert, grace-then-409 for a chat
# ---------------------------------------------------------------------------


async def test_same_surface_expert_takes_over_an_attached_pty():
    app = make_app()
    fake = pool_pty(app)
    older = object()
    assert registry(app).attach_session(KEY, older)
    newer = object()

    plan = await acquire_surface(app, KEY, "expert", newer)

    assert plan.action == ACTION_TAKEOVER
    assert plan.displaced_owner is older
    assert plan.outgoing is not None and plan.outgoing.session is fake
    assert plan.teardown is False
    assert plan.wait_for_idle is False
    # Immediate: a takeover is not a wait for the older owner to leave.
    assert app.clock.sleeps == []
    # Phase (a) marks; phase (c) closes. The attachment is still the older one's.
    assert registry(app).attached_owner(KEY) is older
    assert fake.is_alive
    assert_registered(app, plan, newer)


async def test_a_chat_polls_an_attached_pty_every_50ms_for_2s_then_409():
    app = make_app()
    pool_pty(app)
    assert registry(app).attach_session(KEY, object())

    with pytest.raises(HandoffRefused) as excinfo:
        await acquire_surface(app, KEY, "simple", object())

    refused = excinfo.value
    assert refused.status == 409
    assert refused.error == ERROR_SESSION_ATTACHED_ELSEWHERE
    assert refused.ws_close_code == WS_CLOSE_SESSION_ATTACHED
    sleeps = app.clock.sleeps
    assert sum(sleeps) == pytest.approx(ATTACH_GRACE_S)
    assert all(0 < s <= ATTACH_POLL_S for s in sleeps)
    assert len(sleeps) >= round(ATTACH_GRACE_S / ATTACH_POLL_S)
    # Refused means nothing was registered.
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)


async def test_a_chat_proceeds_when_the_pty_detaches_within_the_grace():
    app = make_app()
    fake = pool_pty(app)
    token = object()
    assert registry(app).attach_session(KEY, token)

    def detach(nth_sleep: int) -> None:
        if nth_sleep == 5:
            registry(app).detach_session(KEY, token)

    app.clock.on_sleep.append(detach)

    plan = await acquire_surface(app, KEY, "simple", object())

    assert plan.action == ACTION_HANDOFF
    assert plan.outgoing is not None and plan.outgoing.session is fake
    assert len(app.clock.sleeps) == 5
    assert app.clock.now == pytest.approx(1000.0 + 5 * ATTACH_POLL_S)


# ---------------------------------------------------------------------------
# The pending slot
# ---------------------------------------------------------------------------


async def test_a_pending_acquire_from_another_connection_is_a_holder():
    app = make_app()
    first = object()
    await acquire_surface(app, KEY, "simple", first)

    with pytest.raises(HandoffRefused) as excinfo:
        await acquire_surface(app, KEY, "expert", object())

    assert excinfo.value.status == 409
    assert excinfo.value.error == ERROR_SESSION_ATTACHED_ELSEWHERE
    assert sum(app.clock.sleeps) == pytest.approx(ATTACH_GRACE_S)
    # The first call's slot survives the refusal untouched.
    assert get_state(app).pending[KEY].channel is first
    assert registry(app).is_reserved(KEY)


async def test_a_second_connection_proceeds_once_the_pending_slot_is_released():
    app = make_app()
    first = object()
    await acquire_surface(app, KEY, "simple", first)
    second = object()

    def finish_first(nth_sleep: int) -> None:
        if nth_sleep == 2:
            assert release_pending(app, KEY, first)

    app.clock.on_sleep.append(finish_first)

    plan = await acquire_surface(app, KEY, "expert", second)

    assert plan.action == ACTION_SPAWN
    assert len(app.clock.sleeps) == 2
    assert_registered(app, plan, second)


async def test_the_same_connection_replaces_its_own_pending_slot():
    app = make_app()
    channel = object()
    await acquire_surface(app, KEY, "simple", channel)

    plan = await acquire_surface(app, KEY, "expert", channel)

    assert plan.action == ACTION_SPAWN
    assert app.clock.sleeps == []
    pending = get_state(app).pending[KEY]
    assert pending.channel is channel
    assert pending.surface == "expert"


async def test_release_pending_is_owner_checked_and_drops_the_reservation():
    app = make_app()
    owner = object()
    await acquire_surface(app, KEY, "simple", owner)
    assert registry(app).is_reserved(KEY)

    assert release_pending(app, KEY, object()) is False
    assert KEY in get_state(app).pending
    assert registry(app).is_reserved(KEY)

    assert release_pending(app, KEY, owner) is True
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)

    assert release_pending(app, KEY, owner) is False


async def test_cancelling_the_phase_b_wait_releases_the_slot_and_reservation():
    """Phase (b) owns the pending slot while it waits: cancelled, it lets go."""
    app = make_app()
    pool_pty(app)
    channel = object()
    entered = asyncio.Event()

    async def wait_forever(_app, _plan):
        entered.set()
        await asyncio.Event().wait()

    with patch.object(session_handoff, "_wait_for_idle", wait_forever):
        task = asyncio.create_task(acquire_surface(app, KEY, "simple", channel))
        await entered.wait()
        # Phase (a) is done: slot and reservation stand.
        assert get_state(app).pending[KEY].channel is channel
        assert registry(app).is_reserved(KEY)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)
    # The key is free for the next acquire.
    plan = await acquire_surface(app, KEY, "expert", object())
    assert plan.action == ACTION_REUSE


async def test_waiting_for_the_lock_is_not_charged_to_the_attach_grace():
    """The grace starts at the first blocked look, not at the call."""
    app = make_app()
    pool_pty(app)
    assert registry(app).attach_session(KEY, object())
    lock = get_state(app).lock_for(KEY)

    await lock.acquire()
    task = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
    for _ in range(5):
        await asyncio.sleep(0)
    assert app.clock.sleeps == []
    # Far longer than the grace passes while the acquire waits for the lock.
    app.clock.now += 5 * ATTACH_GRACE_S
    lock.release()

    with pytest.raises(HandoffRefused) as excinfo:
        await task
    assert excinfo.value.status == 409
    # The full grace was still granted after the lock was obtained.
    assert sum(app.clock.sleeps) == pytest.approx(ATTACH_GRACE_S)
    assert len(app.clock.sleeps) >= round(ATTACH_GRACE_S / ATTACH_POLL_S)


# ---------------------------------------------------------------------------
# The per-key lock
# ---------------------------------------------------------------------------


async def test_the_per_key_lock_serialises_concurrent_inspections():
    """While one acquire is inside its look at the key, another cannot look.

    The first acquire finds a dead chat and awaits its teardown under the
    lock; the second must not read the pools until that hold ends.
    """
    app = make_app()
    chats(app).sessions[KEY] = FakeChatSession(active=False)
    gate = asyncio.Event()
    chats(app).terminate_gate = gate

    first = asyncio.create_task(acquire_surface(app, KEY, "expert", object()))
    for _ in range(5):
        await asyncio.sleep(0)
    assert get_state(app).lock_for(KEY).locked()
    assert chats(app).get_calls == 1

    second = asyncio.create_task(acquire_surface(app, KEY, "simple", object()))
    for _ in range(5):
        await asyncio.sleep(0)
    assert chats(app).get_calls == 1, "second acquire inspected under the first's hold"
    assert not second.done()

    gate.set()
    first_plan = await first
    assert first_plan.action == ACTION_SPAWN

    # The second now sees the first's pending slot (checked before the pools
    # are read), waits its grace, and is refused — never having looked while
    # the first was inside the lock.
    with pytest.raises(HandoffRefused):
        await second
    assert get_state(app).pending[KEY].channel is first_plan.channel


async def test_locks_are_per_key():
    app = make_app()
    other = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    chats(app).sessions[KEY] = FakeChatSession(active=False)
    gate = asyncio.Event()
    chats(app).terminate_gate = gate

    blocked = asyncio.create_task(acquire_surface(app, KEY, "expert", object()))
    for _ in range(5):
        await asyncio.sleep(0)

    plan = await acquire_surface(app, other, "simple", object())
    assert plan.action == ACTION_SPAWN

    gate.set()
    await blocked
    state = get_state(app)
    assert state.lock_for(KEY) is not state.lock_for(other)


# ---------------------------------------------------------------------------
# The refusal type
# ---------------------------------------------------------------------------


def test_refusals_carry_status_slug_and_close_code():
    attached = HandoffRefused.attached_elsewhere(KEY)
    assert (attached.status, attached.error) == (409, "session_attached_elsewhere")
    assert attached.ws_close_code == WS_CLOSE_SESSION_ATTACHED == 4409

    running = HandoffRefused.outgoing_still_running(KEY)
    assert (running.status, running.error) == (503, "outgoing_still_running")
    assert running.ws_close_code == WS_CLOSE_OUTGOING_RUNNING == 4503

    capacity = HandoffRefused.chat_capacity(KEY)
    assert (capacity.status, capacity.error) == (429, "chat_capacity")
    assert capacity.ws_close_code is None

    assert issubclass(HandoffRefused, Exception)
    assert session_handoff.ERROR_CHAT_CAPACITY == "chat_capacity"
