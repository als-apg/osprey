"""Phase (a) of a surface acquire: inspect, decide, register.

``acquire_surface`` is the one door every spawn walks through, and phase (a)
is where it reads the two pools and the pending slot under the per-key lock
and decides what the incoming surface has to do. These tests pin that
decision table and the two invariants around it:

- one look at a key is atomic (the lock serialises concurrent acquires), and
- a key somebody else is consuming is re-inspected for the attach grace with
  the lock *released* between looks, then refused with 409.

Time is faked: the state's ``clock``/``sleep`` are injected, so the two-second
grace is exercised without being spent. The app, the real ``PtyRegistry``
and the fake chat pool come from ``_handoff_harness``. Phases (b) and (c) are
stubbed here, so each test reads the plan phase (a) returned and what it
registered; the full-stack outcomes of the same decisions are pinned in the
phase (c) and invariant suites.
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
    ACTION_SPAWN,
    ATTACH_GRACE_S,
    ATTACH_POLL_S,
    ERROR_SESSION_ATTACHED_ELSEWHERE,
    REASON_NONE,
    WS_CLOSE_SESSION_ATTACHED,
    HandoffRefused,
    HandoffState,
    WaitOutcome,
    get_state,
    release_pending,
)
from tests.interfaces.web_terminal._fakes import FakeChatPool, FakeChatSession
from tests.interfaces.web_terminal._handoff_harness import (
    KEY,
    acquire,
    chats,
    make_app,
    pool_pty,
    registry,
)


@pytest.fixture(autouse=True)
def _phase_b_does_not_wait():
    """These tests are about the decision; the wait it leads to is phase (b)'s own suite.

    Several plans here name a busy outgoing entry — a real phase (b) would wait
    on it — so the wait is stubbed out to return at once.
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
# The state
# ---------------------------------------------------------------------------


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


async def test_a_dead_pty_that_is_still_attached_is_discarded_too():
    """A socket that has not yet noticed its child died holds nothing."""
    app = make_app()
    fake = pool_pty(app)
    stale_token = object()
    assert registry(app).attach_session(KEY, stale_token)
    fake.exit(0)

    plan = await acquire(app, KEY, "simple", object())

    assert plan.action == ACTION_SPAWN
    assert not registry(app).is_attached(KEY)
    assert app.clock.sleeps == []


# ---------------------------------------------------------------------------
# Live entries: the decision table
# ---------------------------------------------------------------------------


async def test_an_expert_waits_for_a_chat_creation_to_land_then_hands_off():
    app = make_app()
    chats(app).starting.add(KEY)
    landed = FakeChatSession()

    def land(nth_sleep: int) -> None:
        if nth_sleep == 3:
            chats(app).starting.discard(KEY)
            chats(app).sessions[KEY] = landed

    app.clock.on_sleep.append(land)

    plan = await acquire(app, KEY, "expert", object())

    assert plan.action == ACTION_HANDOFF
    assert plan.outgoing is not None and plan.outgoing.session is landed
    assert len(app.clock.sleeps) == 3


async def test_an_expert_is_refused_while_a_chat_creation_never_lands():
    """A creation still in flight for the whole grace is a holder, not a gap."""
    app = make_app()
    chats(app).starting.add(KEY)

    with pytest.raises(HandoffRefused) as excinfo:
        await acquire(app, KEY, "expert", object())

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
        await acquire(app, KEY, "expert", object())

    assert excinfo.value.status == 409
    # The corpse was popped and reaped in the first look, not left for later.
    assert registry(app).get_session(KEY) is None
    assert not registry(app).is_attached(KEY)
    assert not dead.is_alive
    # Nothing of the refused call remains.
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)


# ---------------------------------------------------------------------------
# Attached PTY: takeover for an Expert, grace-then-409 for a chat
# ---------------------------------------------------------------------------


async def test_a_chat_polls_an_attached_pty_every_50ms_for_2s_then_409():
    app = make_app()
    pool_pty(app)
    assert registry(app).attach_session(KEY, object())

    with pytest.raises(HandoffRefused) as excinfo:
        await acquire(app, KEY, "simple", object())

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


# ---------------------------------------------------------------------------
# The pending slot
# ---------------------------------------------------------------------------


async def test_release_pending_is_owner_checked_and_drops_the_reservation():
    app = make_app()
    owner = object()
    await acquire(app, KEY, "simple", owner)
    assert registry(app).is_reserved(KEY)

    assert release_pending(app, KEY, object()) is False
    assert KEY in get_state(app).pending
    assert registry(app).is_reserved(KEY)

    assert release_pending(app, KEY, owner) is True
    assert KEY not in get_state(app).pending
    assert not registry(app).is_reserved(KEY)

    assert release_pending(app, KEY, owner) is False


async def test_waiting_for_the_lock_is_not_charged_to_the_attach_grace():
    """The grace starts at the first blocked look, not at the call."""
    app = make_app()
    pool_pty(app)
    assert registry(app).attach_session(KEY, object())
    lock = get_state(app).lock_for(KEY)

    await lock.acquire()
    task = asyncio.create_task(acquire(app, KEY, "simple", object()))
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

    first = asyncio.create_task(acquire(app, KEY, "expert", object()))
    for _ in range(5):
        await asyncio.sleep(0)
    assert get_state(app).lock_for(KEY).locked()
    assert chats(app).get_calls == 1

    second = asyncio.create_task(acquire(app, KEY, "simple", object()))
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

    blocked = asyncio.create_task(acquire(app, KEY, "expert", object()))
    for _ in range(5):
        await asyncio.sleep(0)

    plan = await acquire(app, other, "simple", object())
    assert plan.action == ACTION_SPAWN

    gate.set()
    await blocked
    state = get_state(app)
    assert state.lock_for(KEY) is not state.lock_for(other)


# ---------------------------------------------------------------------------
# The refusal type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("refusal", "status", "slug", "close_code"),
    [
        (HandoffRefused.attached_elsewhere(KEY), 409, "session_attached_elsewhere", 4409),
        (HandoffRefused.outgoing_still_running(KEY), 503, "outgoing_still_running", 4503),
        (HandoffRefused.chat_capacity(KEY), 429, "chat_capacity", None),
        (HandoffRefused.needs_interrupt(KEY), 409, "handoff_needs_interrupt", None),
        (HandoffRefused.superseded(KEY), 409, "handoff_superseded", None),
    ],
    ids=["attached", "still_running", "capacity", "needs_interrupt", "superseded"],
)
def test_refusals_carry_status_slug_and_close_code(refusal, status, slug, close_code):
    """The wire bytes the browser branches on, compared against literals.

    ``chat.js`` keys its refusal table on these slugs and ``api.js`` treats
    exactly these close codes as terminal; every behavioural test compares
    against the module constants, so only a literal catches a renamed one.
    """
    assert (refusal.status, refusal.error, refusal.ws_close_code) == (status, slug, close_code)
