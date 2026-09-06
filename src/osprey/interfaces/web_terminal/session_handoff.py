"""Hand a session key from one surface to the other, one live process at a time.

The Expert view (a PTY-hosted TUI) and the Simple view (an SDK chat) are two
windows onto one OSPREY session. Both pools key their entries on the same
session key ``K``, and the rule that makes them one session is that **at most
one process is live under ``K`` at any moment**. A view flip therefore hands
the key over: the surface that holds it lets go, its process is torn down and
observed dead, and only then does the incoming surface resume the key's
transcript. :func:`acquire_surface` is the one door every spawn walks through
— the terminal websocket, the chat route and the hand-off route alike — so the
rule is enforced in one place.

**State.** Deliberately almost none. Which surface holds ``K`` is read off the
two pools themselves (``PtyRegistry`` membership, ``ChatSessionPool``
membership), never mirrored here, so there is no second registry to drift. The
only state this module adds is :class:`HandoffState` on ``app.state.handoff``:
one :class:`asyncio.Lock` per key, and one *pending acquire* per key — the
call that has decided to take the key but has not filled it yet. A survivor of
a failed kill is put back into its own pool unattached rather than parked in
an orphan list, so the next acquire meets it as an ordinary holder and re-runs
the kill.

**Phases.** An acquire runs in three phases; this module lays out all three,
and the entry point sequences them.

(a) *Under the per-key lock* — :func:`_phase_a`. Inspect both pools and the
    pending slot, discard a held entry whose process is already dead, and
    decide what the incoming surface has to do about whatever is live. A
    chat whose client is closed but whose child still runs — the survivor of
    a failed kill, put back by (c) — is live in the sense that matters: it
    is torn down and observed dead again, whichever surface is acquiring. The
    decision is an :class:`AcquirePlan`. A key that somebody else is actively
    consuming — an attached PTY when a chat wants it, a pending acquire from
    another connection, a chat still starting — is a *transient* blocker: it
    is re-inspected every :data:`ATTACH_POLL_S` seconds for up to
    :data:`ATTACH_GRACE_S`, with the lock released between looks so the
    holder can finish, and only then refused with 409. A same-surface Expert
    connection meeting an attached PTY does not wait: it takes the key over,
    and the plan names the displaced owner for a 4409 close. A Simple
    acquire carrying ``interrupt=True`` that meets another Simple acquire's
    pending wait does not wait for it either: it supersedes it — the waiter's
    task is cancelled, its wait ends in :class:`HandoffSuperseded`, and the
    next look finds the slot free — because that wait is the one the
    operator is asking to end, whether it is their own abandoned request or
    a second tab's. An Expert acquire never supersedes anything. The phase ends by
    registering the call as the key's pending acquire and reserving the key in
    the PTY pool, so the outgoing entry cannot be evicted from under the wait
    that follows.

(b) *Outside the lock* — :func:`_phase_b`. Wait for the outgoing process to
    finish its turn. The only cancellable phase, and the only one with no
    time bound: a turn takes as long as it takes. A chat is idle when its
    ``is_busy`` drops; a PTY is idle when the turn-state store the TUI's hooks
    feed says so, or — when the store has nothing newer to say — when the
    tail of the key's current transcript does. Every :data:`IDLE_POLL_S` the
    wait also asks the caller's channel whether it is still open and raises
    :class:`ChannelClosed` when it is not, so a tab that navigates away
    mid-wait does not leave a hand-off running for nobody. With
    ``interrupt=True`` the wait is cut short: the PTY is sent Escape and
    given :data:`INTERRUPT_GRACE_S` to show the interrupt marker or an idle
    edge, after which it is terminated — the operator asked for exactly that.
    The phase ends in a :class:`WaitOutcome` naming *why* the wait ended, and
    that outcome is what (c) is handed.

(c) *Under the lock again, shielded* — :func:`_phase_c`. Tear the outgoing
    process down, confirm it is dead, spawn the incoming surface's process,
    and release the pending slot. The whole phase runs as one task behind
    ``asyncio.shield``: a caller cancelled in the middle of it — a socket that
    dropped while the new child was starting — gets its cancellation only
    once the phase has finished, so the pools are never left with a popped
    entry and no replacement, a spawned child nobody inserted, or a pending
    slot nobody will release. The outgoing entry is popped from its pool
    before it is killed and the kill runs off the loop; the child is then
    *observed* dead (:attr:`PtySession.is_alive`,
    :attr:`OperatorSession.process_exited`) for up to :data:`DEATH_GRACE_S`.
    A survivor is put back into its pool unattached and the acquire refused
    with 503 — the next one meets it as an ordinary holder and kills it
    again: a PTY survivor is still alive and so still a holder; a chat
    survivor's client is gone but its child is not, and ``OperatorSession.stop``
    sends SIGKILL to a retained child that has no return code yet, so the
    second teardown signals it again before the death check re-runs. Only
    then is the incoming process started, through the caller's
    :data:`SpawnCallback`, told which transcript to resume by a fresh look at
    the transcript map and the transcripts on disk. An Expert caller is
    attached to the PTY (its channel token is the owner token) before the
    slot is released, so the eviction pass never sees the entry unheld. The
    phase ends in an :class:`AcquireResult`.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from osprey.interfaces.web_terminal import transcript_map
from osprey.interfaces.web_terminal.chat_session_pool import ChatCapacityError
from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery
from osprey.interfaces.web_terminal.turn_state import IDLE, get_turn_state, reset_turn_state
from osprey.mcp_server.workspace.transcript_reader import TranscriptReader, tail_state

if TYPE_CHECKING:
    from osprey.interfaces.web_terminal.chat_session_pool import ChatSessionPool
    from osprey.interfaces.web_terminal.operator_session import OperatorSession
    from osprey.interfaces.web_terminal.pty_manager import PtyRegistry, PtySession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

Surface = Literal["expert", "simple"]
"""Which view is acquiring: the PTY-hosted TUI or the SDK chat."""

SURFACE_EXPERT: Surface = "expert"
SURFACE_SIMPLE: Surface = "simple"

AcquireAction = Literal["spawn", "reuse", "takeover", "handoff"]
"""What phase (c) has to do once phase (b) has let the outgoing entry go idle.

``spawn``
    Nothing live holds the key. Spawn the incoming surface's process.
``reuse``
    The incoming surface already holds the key and nobody else is reading it.
    No teardown: an Expert reattaches to the pooled PTY, a chat proceeds to
    the pooled session (or joins the creation still in flight).
``takeover``
    An Expert connection met a PTY another Expert connection is attached to.
    The older owner is closed with 4409 and the newcomer attaches to the same
    PTY. No teardown.
``handoff``
    The key is held by a process the incoming surface cannot use: the other
    surface's entry, or a chat survivor whose client is already closed. It
    is torn down once idle and the incoming surface resumes the transcript.
"""

ACTION_SPAWN: AcquireAction = "spawn"
ACTION_REUSE: AcquireAction = "reuse"
ACTION_TAKEOVER: AcquireAction = "takeover"
ACTION_HANDOFF: AcquireAction = "handoff"

#: How long phase (a) keeps re-inspecting a key somebody else is consuming
#: before it gives up with 409. Long enough for a tab that closed its socket a
#: moment ago to be seen detaching; short enough that a key held for real is
#: refused before the operator wonders what the view is doing.
ATTACH_GRACE_S = 2.0
#: How often phase (a) re-inspects during that grace.
ATTACH_POLL_S = 0.05

#: How often phase (b) looks at the outgoing entry — and, on the same tick, at
#: the caller's channel. The channel look is what bounds how long a hand-off
#: can run for a caller that has already gone; it must stay within 0.2 s.
IDLE_POLL_S = 0.1
#: How long an interrupted PTY is given to show the interrupt marker or an
#: idle edge after Escape before phase (b) terminates it outright.
INTERRUPT_GRACE_S = 5.0
#: The keystroke that interrupts a running turn in the TUI.
_INTERRUPT_KEY = b"\x1b"

#: How long phase (c) keeps looking for the outgoing child to be gone after
#: its kill has returned. ``PtySession.terminate`` has already waited through
#: its own signal escalation by then and a chat teardown through the SDK's;
#: this covers the moment between the kill landing and the exit being
#: collected. A child still alive at the end is a survivor, not a slow one.
DEATH_GRACE_S = 2.0
#: How often phase (c) looks during that grace.
DEATH_POLL_S = 0.05

IdleReason = Literal["none", "idle", "interrupted", "forced", "exited"]
"""Why phase (b) stopped waiting; see :class:`WaitOutcome`."""

REASON_NONE: IdleReason = "none"
REASON_IDLE: IdleReason = "idle"
REASON_INTERRUPTED: IdleReason = "interrupted"
REASON_FORCED: IdleReason = "forced"
REASON_EXITED: IdleReason = "exited"

ERROR_SESSION_ATTACHED_ELSEWHERE = "session_attached_elsewhere"
ERROR_OUTGOING_STILL_RUNNING = "outgoing_still_running"
ERROR_CHAT_CAPACITY = "chat_capacity"
ERROR_HANDOFF_NEEDS_INTERRUPT = "handoff_needs_interrupt"
ERROR_HANDOFF_SUPERSEDED = "handoff_superseded"
ERROR_OUTGOING_VANISHED = "outgoing_vanished"
ERROR_SPAWN_NOT_POOLED = "spawn_not_pooled"

#: Websocket close codes for the refusals a terminal socket can meet. The
#: browser side treats both as terminal (no reconnect); see ``api.js``.
WS_CLOSE_SESSION_ATTACHED = 4409
WS_CLOSE_OUTGOING_RUNNING = 4503

_WS_CLOSE_BY_STATUS: dict[int, int] = {
    409: WS_CLOSE_SESSION_ATTACHED,
    503: WS_CLOSE_OUTGOING_RUNNING,
}


class HandoffRefused(Exception):
    """An acquire that must not proceed, with the status each channel reports.

    Routes send ``status`` with a body of ``{"detail": {"error": <error>}}``;
    the terminal websocket closes with :attr:`ws_close_code`. Every refusal
    is a classmethod, so a caller never assembles a status/slug pair by hand.
    """

    def __init__(self, status: int, error: str, message: str | None = None) -> None:
        super().__init__(message or error)
        self.status = status
        self.error = error

    @property
    def ws_close_code(self) -> int | None:
        """The websocket close code for this refusal, or ``None`` when the
        refusal has no websocket form (429 is only ever answered to a POST)."""
        return _WS_CLOSE_BY_STATUS.get(self.status)

    @classmethod
    def attached_elsewhere(cls, key: str) -> HandoffRefused:
        """409 — another connection or view is consuming *key* right now."""
        return cls(
            409,
            ERROR_SESSION_ATTACHED_ELSEWHERE,
            f"session {key!r} is attached to another connection or view",
        )

    @classmethod
    def outgoing_still_running(cls, key: str) -> HandoffRefused:
        """503 — the outgoing process survived its kill; retry re-runs it."""
        return cls(
            503,
            ERROR_OUTGOING_STILL_RUNNING,
            f"the previous agent for session {key!r} is still shutting down",
        )

    @classmethod
    def chat_capacity(cls, key: str) -> HandoffRefused:
        """429 — the chat pool is full and every chat is busy."""
        return cls(
            429,
            ERROR_CHAT_CAPACITY,
            f"no chat capacity to resume session {key!r}",
        )

    @classmethod
    def needs_interrupt(cls, key: str) -> HandoffNeedsInterrupt:
        """409 — a PTY holds *key* and nothing can tell when its turn ends.

        See :class:`HandoffNeedsInterrupt`.
        """
        return HandoffNeedsInterrupt(key)

    @classmethod
    def superseded(cls, key: str) -> HandoffSuperseded:
        """409 — a newer Simple acquire with an interrupt took *key* over.

        See :class:`HandoffSuperseded`.
        """
        return HandoffSuperseded(key)


class HandoffNeedsInterrupt(HandoffRefused):
    """The key is held by a PTY whose turns are invisible; only an interrupt can take it.

    Phase (b) cannot wait for a PTY to go idle without the turn-state hook
    installed in the TUI (``app.state.turn_hook_present``): the transcript
    tail alone cannot tell an answered prompt from one still being worked on.
    So a Simple view acquiring such a key is refused with this, and offers the
    operator *Stop and switch now* — the same acquire again with
    ``interrupt=True``, which phase (b) does honour without the hook.

    Only ever answered to a POST: an Expert acquire never waits on a PTY, so
    the refusal has no websocket form.
    """

    def __init__(self, key: str) -> None:
        super().__init__(
            409,
            ERROR_HANDOFF_NEEDS_INTERRUPT,
            f"session {key!r} is held by a terminal whose turn state is not reported",
        )

    @property
    def ws_close_code(self) -> int | None:
        return None


class HandoffSuperseded(HandoffRefused):
    """The wait was ended by a newer Simple acquire of the same key carrying an interrupt.

    The Simple view's *Stop and switch now* is a second request for the key
    while its first still waits in phase (b) — the browser abandons the first
    but the server cannot rely on seeing that, and a second tab has no
    request to abandon at all. Phase (a) therefore lets a Simple acquire with
    ``interrupt=True`` supersede a pending Simple acquire: the waiter's task
    is cancelled, its pending slot released, and its wait ends in this
    refusal, so the route it was made from answers 409 like any other. An
    abandoned request's answer is discarded by the client that abandoned it;
    a second tab's shows the operator what happened.

    Only ever answered to a POST: an Expert acquire neither supersedes nor is
    superseded, so the refusal has no websocket form.
    """

    def __init__(self, key: str) -> None:
        super().__init__(
            409,
            ERROR_HANDOFF_SUPERSEDED,
            f"another request took over session {key!r}",
        )

    @property
    def ws_close_code(self) -> int | None:
        return None


class HandoffError(Exception):
    """A hand-off that cannot be carried out because its premise no longer holds.

    Unlike :class:`HandoffRefused` this is not a state the caller can expect
    and retry against: the outgoing entry phase (a) built the plan around has
    been replaced in its pool by something other than this hand-off while
    phase (b) waited on it, or the caller's spawn callback returned a session
    its pool does not hold under the key. The pending slot and the
    reservation are released before this reaches the caller.
    """

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error

    @classmethod
    def vanished(cls, key: str, surface: Surface) -> HandoffError:
        return cls(
            ERROR_OUTGOING_VANISHED,
            f"the {surface} entry for session {key!r} left its pool during the hand-off",
        )

    @classmethod
    def not_pooled(cls, key: str, surface: Surface) -> HandoffError:
        return cls(
            ERROR_SPAWN_NOT_POOLED,
            f"the {surface} spawn for session {key!r} returned a session its pool does not hold",
        )


class ChannelClosed(asyncio.CancelledError):
    """The caller's channel closed while phase (b) waited.

    A cancellation like any other — ``except asyncio.CancelledError`` catches
    it, and the wait releases the pending slot on the way out exactly as it
    does for a ``task.cancel()`` — but distinguishable, so a route that wants
    to log *why* its acquire ended can. It is raised from inside the coroutine
    rather than delivered by ``task.cancel()``, so an acquire must not run
    under ``asyncio.timeout``, ``wait_for`` or a ``TaskGroup``: their uncancel
    accounting turns it into a ``TimeoutError`` or drops it silently. Callers
    catch :class:`ChannelClosed` directly. The reverse hazard exists too: an
    external ``task.cancel()`` that lands while the wait is suspended inside
    the channel probe's own cancel scope (Starlette's ``is_disconnected``)
    can be absorbed by that scope and never thrown — which is why a supersede
    does not depend on its cancellation but on the mark it leaves on the
    pending record (see :func:`_raise_if_superseded`).
    """


# ---------------------------------------------------------------------------
# The caller's channel
# ---------------------------------------------------------------------------


@runtime_checkable
class AcquireChannel(Protocol):
    """What phase (b) asks of the channel token: is the caller still there?

    A channel is any object passed as the ``channel`` argument of
    :func:`acquire_surface`; it is compared by identity for the pending slot.
    One that also implements ``is_closed`` is asked, every :data:`IDLE_POLL_S`
    during the wait, whether the caller has gone away. The answer may be a
    bool or an awaitable of one — a websocket handler answers from an event
    its receive loop sets on ``websocket.disconnect``; a POST route answers
    with ``request.is_disconnected()``. A token without ``is_closed`` never
    closes.
    """

    def is_closed(self) -> bool | Awaitable[bool]: ...


class ChannelToken:
    """A channel token with an injectable closed probe.

    The identity phase (a) registers and the probe phase (b) polls, in one
    object, so a caller need not define its own class::

        ChannelToken(request.is_disconnected)   # POST: awaitable probe
        ChannelToken(closed_event.is_set)        # websocket: sync probe
        ChannelToken()                           # never closes
    """

    def __init__(self, is_closed: Callable[[], bool | Awaitable[bool]] | None = None) -> None:
        self._probe = is_closed

    def is_closed(self) -> bool | Awaitable[bool]:
        if self._probe is None:
            return False
        return self._probe()


async def channel_closed(channel: object) -> bool:
    """Whether *channel* reports itself closed; False for a token without a probe."""
    probe = getattr(channel, "is_closed", None)
    if probe is None:
        return False
    answer = probe()
    if inspect.isawaitable(answer):
        answer = await answer
    return bool(answer)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class PendingAcquire:
    """The one call per key that has decided to take the key and not yet filled it.

    Registered at the end of phase (a), released at the end of phase (c) — or
    by phase (b) itself when its wait is cancelled, so (c) never runs. Callers
    never release the slot; the phases own it. While it stands, any other
    connection's acquire of the same key sees a holder.

    Attributes:
        channel: The token identifying the connection that made the call —
            the terminal socket's attach token or a per-request object for a
            POST. Compared by identity; the same channel re-acquiring replaces
            its own slot rather than blocking on it.
        surface: Which surface the call is acquiring for.
        task: The task running the acquire, so a diagnostic or a shutdown can
            see what is in flight — and so a superseding acquire can cancel
            it. ``None`` outside a task.
        superseded: A newer Simple acquire with an interrupt has cancelled
            this call's wait. Set before the cancel is delivered, so the
            waiter can tell it from any other cancellation and end in
            :class:`HandoffSuperseded`; also what stops a second look from
            cancelling the same task twice.
        waiting: The call is still in phase (b), where a cancellation ends a
            wait and nothing else. Cleared by phase (c) before its shielded
            task exists, so a supersede never reaches a call that is already
            carrying the key out: that call is left to finish and answer, and
            the superseder's next look meets what it pooled.
    """

    channel: object
    surface: Surface
    task: asyncio.Task[Any] | None = None
    superseded: bool = False
    waiting: bool = True


@dataclass
class HandoffState:
    """Everything this module keeps beyond the pools themselves.

    Lives on ``app.state.handoff``; :func:`get_state` builds it lazily so an
    app assembled without it (tests) still works.

    Attributes:
        locks: One lock per session key, created on first use and kept for the
            life of the process. A lock is never removed: a caller that has
            already looked its lock up and is waiting on it would otherwise be
            left holding an object nobody else can see, and two acquires of
            the same key would stop excluding each other. The table is bounded
            by the number of distinct keys the process has ever seen.
        pending: The pending acquire per key, if any. See :class:`PendingAcquire`.
        closers: How to close an attached terminal socket, by its attach
            token. The terminal handler registers its close coroutine here
            when it attaches and removes it when it detaches; phase (c) looks
            a plan's ``displaced_owner`` up here to send the 4409. Absent
            entries are tolerated — a socket that has already gone away has
            nothing to close.
        clock: The monotonic clock the attach grace is measured on.
        sleep: How phase (a) waits between re-inspections. Both are
            injectable so the grace can be exercised without spending it.
    """

    locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    pending: dict[str, PendingAcquire] = field(default_factory=dict)
    closers: dict[object, Callable[[], Awaitable[None]]] = field(default_factory=dict)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def lock_for(self, key: str) -> asyncio.Lock:
        """The lock serialising every acquire of *key*."""
        lock = self.locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[key] = lock
        return lock


def get_state(app: Any) -> HandoffState:
    """The :class:`HandoffState` on *app*, created on first use."""
    state = getattr(app.state, "handoff", None)
    if state is None:
        state = HandoffState()
        app.state.handoff = state
    return state


def _pty_registry(app: Any) -> PtyRegistry:
    registry: PtyRegistry = app.state.pty_registry
    return registry


def _chat_pool(app: Any) -> ChatSessionPool:
    pool: ChatSessionPool = app.state.operator_registry.chats
    return pool


# ---------------------------------------------------------------------------
# The contract between the phases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutgoingEntry:
    """The live entry phase (a) found under the key, and which pool it is in."""

    surface: Surface
    session: PtySession | OperatorSession


@dataclass(frozen=True)
class SpawnRequest:
    """What phase (c) hands the caller's spawn callback.

    Attributes:
        key: The session key the new process runs under — its pool key, its
            ``OSPREY_SESSION_ID``, and its own session id when nothing is
            resumed.
        surface: Which surface's process to start.
        resume_id: The transcript the process continues, or ``None`` when the
            key has no transcript on disk yet — the process then starts a
            conversation of its own under *key* (``claude --session-id K``,
            SDK ``session_id=K``). Chosen at spawn time from the transcript
            map and the transcripts on disk, so a transcript that moved while
            the other surface held the key is what the new process opens.
        transcript_id: The key's current transcript per the map (*key* itself
            when it never moved), whether or not it is on disk.
    """

    key: str
    surface: Surface
    resume_id: str | None
    transcript_id: str


SpawnCallback = Callable[[SpawnRequest], Awaitable["PtySession | OperatorSession"]]
"""How the incoming surface starts its process; see :func:`acquire_surface`.

Called by phase (c), inside its shielded task and under the key's lock, once
the key holds nothing of the other surface. It must start the process *and*
put it in its pool under ``request.key`` — the terminal handler through
``PtyRegistry.get_or_create_session`` with the resume argument built from
``request.resume_id``, the chat route through
``OperatorRegistry.get_or_create_chat_session(key, cwd, env,
resume_id=request.resume_id)`` — and return the pooled session. A chat pool
at capacity may raise ``ChatCapacityError``; phase (c) turns it into the 429
refusal. Any other exception escapes :func:`acquire_surface` unchanged after
the pending slot has been released.
"""


@dataclass(frozen=True)
class AcquirePlan:
    """What phase (a) decided; the input to phases (b) and (c).

    Attributes:
        key: The session key being acquired.
        surface: The surface acquiring it.
        channel: The caller's channel token, as registered in the pending slot.
        interrupt: The caller asked for the outgoing turn to be aborted rather
            than awaited. Carried through for phase (b).
        action: See :data:`AcquireAction`.
        outgoing: The live entry the plan is about, or ``None`` when nothing
            live holds the key (``spawn``), or when a chat creation is still
            in flight and the incoming chat will join it (``reuse`` for the
            Simple surface only).
        teardown: Phase (c) must tear ``outgoing`` down before spawning. True
            exactly for ``handoff``.
        wait_for_idle: Phase (b) must wait for ``outgoing`` to finish its
            turn. True for every ``handoff`` and for a chat reusing a chat —
            the chat route cannot take a turn on a busy session, so the
            hand-off is what tells the Simple view when the session is free.
            False for an Expert reattaching or taking over: a TUI keeps
            running and the newcomer simply sees its output.
        displaced_owner: On ``takeover``, the attach token of the Expert
            connection currently holding the PTY, to be closed with 4409 by
            phase (c). ``None`` otherwise.
        spawn: The caller's :data:`SpawnCallback`, carried through for phase
            (c). ``None`` when the caller has nothing to start — legal only
            for a plan whose incoming surface's entry is already pooled.
    """

    key: str
    surface: Surface
    channel: object
    interrupt: bool
    action: AcquireAction
    outgoing: OutgoingEntry | None
    teardown: bool
    wait_for_idle: bool
    displaced_owner: object | None = None
    spawn: SpawnCallback | None = None


@dataclass(frozen=True)
class WaitOutcome:
    """How phase (b) ended; the second input to phase (c).

    Attributes:
        reason: Why the wait stopped.

            ``none``
                There was nothing to wait on: the plan has no outgoing entry,
                or does not ask for idleness (an Expert reattaching to or
                taking over a running TUI).
            ``idle``
                The outgoing entry was observed idle — a chat's ``is_busy``
                False, a PTY's turn-state store or transcript tail at rest.
            ``interrupted``
                The caller asked for the turn to be aborted. For a PTY, Escape
                was written and the interrupt marker or an idle edge was seen
                within :data:`INTERRUPT_GRACE_S`; the turn is over. For a
                chat, no wait was made and the turn may still be running:
                ``interrupted`` obliges phase (c) to cancel it — through
                ``pool.terminate(key)`` when ``plan.teardown`` is True, and
                through ``await session.cancel()`` on the ``OperatorSession``
                when the plan is ``reuse`` (``cancel`` interrupts the client
                and quiesces the reader; ``interrupt`` alone does neither).
            ``forced``
                The caller asked for an interrupt and the PTY did not go idle
                within the grace, so phase (b) has already called its
                ``terminate`` (off the loop). The entry is still in its pool
                and ``terminate`` is idempotent, so phase (c) runs its
                ordinary teardown and death check; it must not expect the
                child to be gone yet.
            ``exited``
                The outgoing process died on its own while still pooled. There
                is no turn left to wait for; phase (c) discards the corpse.
    """

    reason: IdleReason


@dataclass(frozen=True)
class AcquireResult:
    """What :func:`acquire_surface` returns: the key is the caller's.

    Attributes:
        plan: What phase (a) decided.
        outcome: How phase (b)'s wait ended.
        session: The incoming surface's live entry under the key — the
            ``PtySession`` an Expert caller is now attached to (its channel
            token is the owner token; it detaches with the same token when it
            is done), or the ``OperatorSession`` a Simple caller takes its
            turn on (through ``acquire_turn``, which is where a concurrent
            turn is still refused).
        spawned: The spawn callback ran for this acquire. False when the
            caller was handed the surface's already-pooled entry without a
            call. True also when the callback joined a chat creation already
            in flight (the Simple ``reuse`` with no outgoing entry): the
            callback ran, whether or not it started the process itself.
        resume_id: What the spawn was asked to resume, ``None`` for a fresh
            start under the key — and ``None`` whenever nothing was spawned.
            The terminal handler watches a spawned ``--resume`` child for the
            CLI's own "no conversation" verdict; this is the id to watch for.
    """

    plan: AcquirePlan
    outcome: WaitOutcome
    session: PtySession | OperatorSession
    spawned: bool
    resume_id: str | None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def acquire_surface(
    app: Any,
    key: str,
    surface: Surface,
    channel: object,
    *,
    interrupt: bool = False,
    spawn: SpawnCallback | None = None,
) -> AcquireResult:
    """Take session key *key* for *surface* on behalf of *channel*.

    Runs the three phases and returns once the incoming surface's process is
    live under the key: for an Expert caller, attached to the returned
    ``PtySession`` with *channel* as the owner token; for a Simple caller,
    with the returned ``OperatorSession`` pooled and free of any other
    surface. The caller never releases the pending slot: the phases own it.

    Raises :class:`HandoffRefused` when the key cannot be taken — 409 when
    another connection or view is consuming it (or, as
    :class:`HandoffNeedsInterrupt`, when a hook-less PTY holds it and no
    interrupt was asked for; or, as :class:`HandoffSuperseded`, when a newer
    Simple acquire with an interrupt ended this call's wait), 503 when the
    outgoing process survived its kill, 429 when the chat pool is full; the
    caller maps ``.status`` and
    ``.error`` to its channel (HTTP status with a ``detail.error`` body, or
    ``.ws_close_code``). Raises :class:`ChannelClosed` (a cancellation) when
    *channel* reports itself closed during the wait: the caller sends
    nothing and returns. Raises :class:`HandoffError` when the outgoing
    entry was replaced in its pool mid-wait or the spawn callback returned a
    session its pool does not hold; the route answers 5xx with its
    ``.error``. Anything else *spawn* raises escapes unchanged. Because
    :class:`ChannelClosed` is raised from inside the coroutine, do not run
    this call under ``asyncio.timeout``, ``wait_for`` or a ``TaskGroup``;
    catch it directly.

    A cancellation delivered while phase (c) runs is honoured only once the
    phase has finished; the outgoing process is torn down and the incoming
    one started, pooled and (Expert) attached regardless. An Expert caller
    therefore detaches its token in its own cleanup whether or not this call
    returned — ``detach_session`` is owner-checked and a no-op for a token
    that never attached.

    Args:
        app: The application; its ``state`` carries the two pools and the
            :class:`HandoffState`.
        key: The session key.
        surface: ``"expert"`` or ``"simple"``.
        channel: A token identifying the calling connection, compared by
            identity. The terminal handler passes its attach token. A token
            implementing :class:`AcquireChannel` is polled for closure during
            the wait; see :class:`ChannelToken`.
        interrupt: Abort the outgoing turn instead of waiting for it.
        spawn: How the incoming surface starts its process; see
            :data:`SpawnCallback`. Called only when the key holds no live
            entry of the incoming surface, so a Simple caller that is merely
            taking its next turn on its own pooled chat is handed that chat
            back without a spawn. Omitted, the call can only hand back an
            entry that is already pooled.
    """
    plan = await _phase_a(app, key, surface, channel, interrupt=interrupt, spawn=spawn)
    outcome = await _phase_b(app, plan)
    return await _phase_c(app, plan, outcome)


def release_pending(app: Any, key: str, channel: object) -> bool:
    """Drop *channel*'s pending acquire of *key* and the reservation that came with it.

    Owner-checked: a slot held by another channel is left alone, so a caller
    cleaning up after its own cancelled wait cannot release the acquire that
    took the key after it. Safe to call when nothing is pending. The phases
    call it themselves; a caller of :func:`acquire_surface` never has to.

    Returns:
        True when a slot was released.
    """
    state = get_state(app)
    pending = state.pending.get(key)
    if pending is None or pending.channel is not channel:
        return False
    del state.pending[key]
    _pty_registry(app).unreserve(key)
    return True


# ---------------------------------------------------------------------------
# Phase (a): inspect, decide, register — under the per-key lock
# ---------------------------------------------------------------------------


class _Blocked(Exception):
    """Internal: the key is held in a way that may clear within the grace."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


async def _phase_a(
    app: Any,
    key: str,
    surface: Surface,
    channel: object,
    *,
    interrupt: bool,
    spawn: SpawnCallback | None = None,
) -> AcquirePlan:
    """Decide what acquiring *key* for *surface* means right now.

    Each look at the key — inspection, discard of dead entries, decision and
    registration — happens inside one hold of the key's lock, so two acquires
    of the same key can never both conclude the key is free. A transient
    blocker releases the lock, sleeps :data:`ATTACH_POLL_S`, and looks again,
    until :data:`ATTACH_GRACE_S` has elapsed since the *first* blocked look;
    time spent waiting for the lock itself is not charged to the grace. The
    lock is released between looks precisely so the holder (whose own phase
    (c) needs this lock to release its pending slot) can finish.
    """
    state = get_state(app)
    lock = state.lock_for(key)
    deadline: float | None = None
    while True:
        async with lock:
            try:
                return await _inspect_and_register(
                    app, state, key, surface, channel, interrupt=interrupt, spawn=spawn
                )
            except _Blocked as blocked:
                reason = blocked.reason
        now = state.clock()
        if deadline is None:
            deadline = now + ATTACH_GRACE_S
        if now >= deadline:
            logger.info(
                "Refusing %s acquire of session %s: %s after %.1fs",
                surface,
                key,
                reason,
                ATTACH_GRACE_S,
            )
            raise HandoffRefused.attached_elsewhere(key)
        await state.sleep(min(ATTACH_POLL_S, deadline - now))


async def _inspect_and_register(
    app: Any,
    state: HandoffState,
    key: str,
    surface: Surface,
    channel: object,
    *,
    interrupt: bool,
    spawn: SpawnCallback | None = None,
) -> AcquirePlan:
    """One look at the key. Caller holds the key's lock.

    Raises :class:`_Blocked` for a holder that may clear within the grace.
    Returns the plan otherwise, having registered the pending acquire and
    reserved the key.
    """
    registry = _pty_registry(app)
    pool = _chat_pool(app)

    pending = state.pending.get(key)
    if pending is not None and pending.channel is not channel:
        if _supersedes(pending, surface, interrupt=interrupt):
            _supersede(key, pending)
            # Blocked for this look only: the cancelled waiter releases the
            # slot as soon as it runs, and the next look finds the key free.
            raise _Blocked("a simple acquire from another connection is being superseded")
        raise _Blocked(f"a {pending.surface} acquire from another connection is pending")

    pty = registry.get_session(key)
    if pty is not None and not pty.is_alive:
        # A corpse holds nothing. Pop it on the loop, reap it off the loop —
        # ``terminate`` on a dead child only closes the fd and collects the
        # exit status, but it is still the blocking half.
        registry.pop_session(key)
        await asyncio.to_thread(pty.terminate)
        logger.info("Discarded dead PTY under session %s", key)
        pty = None

    chat = pool.get(key)
    survivor = chat is not None and _chat_survivor(chat)
    if chat is not None and not chat.is_active and not survivor:
        await pool.terminate(key)
        logger.info("Discarded dead chat session under session %s", key)
        chat = None
    chat_starting = chat is None and pool.has_key(key)

    if pty is not None and registry.is_attached(key):
        if surface != SURFACE_EXPERT:
            raise _Blocked("the PTY is attached to a terminal connection")
        # Same surface, live PTY, somebody reading it: the newer connection
        # wins. The older one is closed with 4409 by phase (c); the PTY is
        # never torn down for this.
        plan = AcquirePlan(
            key=key,
            surface=surface,
            channel=channel,
            interrupt=interrupt,
            action=ACTION_TAKEOVER,
            outgoing=OutgoingEntry(SURFACE_EXPERT, pty),
            teardown=False,
            wait_for_idle=False,
            displaced_owner=registry.attached_owner(key),
            spawn=spawn,
        )
        return _register(state, registry, plan)

    if chat_starting:
        if surface != SURFACE_SIMPLE:
            raise _Blocked("a chat session is still starting")
        # The chat route's get_or_create joins the in-flight creation; there
        # is no session object to wait on yet and nothing to tear down.
        plan = AcquirePlan(
            key=key,
            surface=surface,
            channel=channel,
            interrupt=interrupt,
            action=ACTION_REUSE,
            outgoing=None,
            teardown=False,
            wait_for_idle=False,
            spawn=spawn,
        )
        return _register(state, registry, plan)

    if pty is not None and chat is not None:
        # Both pools hold a live process under one key — the invariant this
        # module exists to keep. It cannot arise through this door; it can
        # through an older spawn path that has not been routed here. The
        # other surface's entry is the one handed off; the incoming surface's
        # own entry is what its pool reuses afterwards.
        logger.warning(
            "Session %s is live in both pools; handing off the %s entry",
            key,
            SURFACE_SIMPLE if surface == SURFACE_EXPERT else SURFACE_EXPERT,
        )

    outgoing: OutgoingEntry | None = None
    if surface == SURFACE_EXPERT:
        if chat is not None:
            outgoing = OutgoingEntry(SURFACE_SIMPLE, chat)
        elif pty is not None:
            outgoing = OutgoingEntry(SURFACE_EXPERT, pty)
    else:
        if pty is not None:
            outgoing = OutgoingEntry(SURFACE_EXPERT, pty)
        elif chat is not None:
            outgoing = OutgoingEntry(SURFACE_SIMPLE, chat)

    if outgoing is None:
        action: AcquireAction = ACTION_SPAWN
        teardown = False
        wait_for_idle = False
    elif outgoing.surface == surface and not survivor:
        action = ACTION_REUSE
        teardown = False
        wait_for_idle = surface == SURFACE_SIMPLE
    else:
        # The other surface's entry — or a chat survivor, which no surface
        # can use: its client is closed, only its child remains to be killed.
        action = ACTION_HANDOFF
        teardown = True
        wait_for_idle = True

    plan = AcquirePlan(
        key=key,
        surface=surface,
        channel=channel,
        interrupt=interrupt,
        action=action,
        outgoing=outgoing,
        teardown=teardown,
        wait_for_idle=wait_for_idle,
        spawn=spawn,
    )
    return _register(state, registry, plan)


def _chat_survivor(chat: OperatorSession) -> bool:
    """A chat whose client is closed while its child still runs.

    What phase (c) puts back after a teardown whose death check failed. It
    is not dead — ``process_exited`` is False, not True or None — and not
    usable either, so phase (a) hands it off for another teardown instead of
    discarding it as a corpse or reusing it as a chat.
    """
    return not chat.is_active and chat.process_exited is False


def _register(state: HandoffState, registry: PtyRegistry, plan: AcquirePlan) -> AcquirePlan:
    """Record the call as the key's pending acquire and reserve the key.

    Caller holds the key's lock. The reservation keeps the outgoing PTY — now
    attached to nobody — off the eviction pass for as long as the slot
    stands; :func:`release_pending` drops both together.
    """
    state.pending[plan.key] = PendingAcquire(
        channel=plan.channel,
        surface=plan.surface,
        task=asyncio.current_task(),
    )
    registry.reserve(plan.key)
    return plan


def _supersedes(pending: PendingAcquire, surface: Surface, *, interrupt: bool) -> bool:
    """Whether an acquire for *surface* may end *pending*'s wait instead of waiting on it.

    Only a Simple acquire that asks for an interrupt, meeting a Simple call
    that is still waiting in phase (b), has a task to cancel and has not been
    superseded already. A call already in phase (c) is carrying the key out
    and is left alone: it answers its caller, and the superseder's next look
    meets the chat it pooled. An Expert acquire never supersedes: its
    handshake carries no interrupt, and a terminal that wants a key a chat is
    waiting for is the ordinary blocked case. A pending Expert wait is never
    superseded either — the interrupt is a gesture at the Simple view's own
    wait, not at the other view's.
    """
    return (
        interrupt
        and surface == SURFACE_SIMPLE
        and pending.surface == SURFACE_SIMPLE
        and pending.waiting
        and pending.task is not None
        and not pending.superseded
    )


def _supersede(key: str, pending: PendingAcquire) -> None:
    """Cancel *pending*'s wait, marking it so the waiter ends in :class:`HandoffSuperseded`.

    Caller holds the key's lock. The mark lands before the cancel so the
    waiter, which reads it under no lock, cannot see the cancellation first;
    the slot itself is released by the waiter's phase (b) on its way out,
    never here, so the "slot held ⇔ key reserved" invariant is kept by the
    one owner it has.
    """
    assert pending.task is not None
    pending.superseded = True
    pending.task.cancel()
    logger.info("Superseding the pending simple acquire of session %s on an interrupt", key)


# ---------------------------------------------------------------------------
# Phase (b): wait for the outgoing turn to end — outside the lock
# ---------------------------------------------------------------------------


async def _phase_b(app: Any, plan: AcquirePlan) -> WaitOutcome:
    """Wait until ``plan.outgoing`` is idle, when ``plan.wait_for_idle`` says to.

    The only phase a caller may cancel, and the owner of the pending slot for
    as long as it runs. Every exit that does not reach phase (c) — a
    ``task.cancel()``, the caller's channel closing, a refusal, the outgoing
    entry vanishing — releases the slot and the reservation here, so the key
    is free for the next acquire and a caller never has to know which of
    those ended its wait. The wait itself is :func:`_wait_for_idle`.

    One cancellation is not passed on as one: a wait that a newer Simple
    acquire superseded (see :func:`_supersede`) ends in
    :class:`HandoffSuperseded` instead, so the route that made it answers a
    refusal rather than dying cancelled. The mark is read off this call's own
    pending record before the record is released — a record another channel
    holds by then says nothing about this wait. The task's cancellation
    request is withdrawn with ``uncancel`` — safe because an acquire never
    runs under ``asyncio.timeout``, ``wait_for`` or a ``TaskGroup``, whose
    accounting the withdrawal would otherwise confuse — unless a further
    cancellation is outstanding, in which case that one is honoured as
    usual. A :class:`ChannelClosed` itself is never converted. The
    cancellation is only the prompt form of the supersede: the wait reads
    the mark at the start of every look and again before it returns, and
    ends itself from it (:func:`_raise_if_superseded`), because a cancel
    landing inside the channel probe's anyio scope can be absorbed there —
    so a marked call is answered the refusal from the look that finds the
    mark, whichever of the two exits the cancellation would have taken.
    """
    try:
        return await _wait_for_idle(app, plan)
    except asyncio.CancelledError as cancelled:
        superseded = _superseded(get_state(app), plan) and not isinstance(cancelled, ChannelClosed)
        release_pending(app, plan.key, plan.channel)
        if superseded and _withdraw_cancellation():
            logger.info(
                "The %s acquire of session %s was superseded by a newer request",
                plan.surface,
                plan.key,
            )
            raise HandoffRefused.superseded(plan.key) from None
        raise
    except BaseException:
        release_pending(app, plan.key, plan.channel)
        raise


def _superseded(state: HandoffState, plan: AcquirePlan) -> bool:
    """Whether *plan*'s own pending record — still registered — carries the superseded mark."""
    pending = state.pending.get(plan.key)
    return pending is not None and pending.channel is plan.channel and pending.superseded


def _withdraw_cancellation() -> bool:
    """Take back the one cancellation a supersede delivered to the current task.

    Returns False — leaving the cancellation to propagate — when there is no
    task, or when a further cancellation request is still outstanding after
    the one withdrawn here: that one was somebody else's and is honoured.
    """
    task = asyncio.current_task()
    if task is None:
        return False
    task.uncancel()
    return task.cancelling() == 0


async def _wait_for_idle(app: Any, plan: AcquirePlan) -> WaitOutcome:
    """Block until the outgoing entry has finished its turn, and say how it ended.

    No time bound. Ends when the entry is idle, when the caller's channel
    closes (:class:`ChannelClosed`), or — with ``plan.interrupt`` — once the
    turn has been aborted. An entry that leaves its pool while the wait runs
    is a :class:`HandoffError`; one whose process dies while still pooled ends
    the wait with ``exited``.
    """
    if plan.outgoing is None or not plan.wait_for_idle:
        return WaitOutcome(REASON_NONE)
    state = get_state(app)
    if plan.outgoing.surface == SURFACE_SIMPLE:
        return await _wait_for_chat_idle(app, state, plan)
    return await _wait_for_pty_idle(app, state, plan)


async def _wait_for_chat_idle(app: Any, state: HandoffState, plan: AcquirePlan) -> WaitOutcome:
    """A chat is idle when ``is_busy`` is False; no hook is involved.

    With ``plan.interrupt`` nothing is waited for and nothing is cancelled
    here: the turn is left to phase (c), which on ``interrupted`` must cancel
    it — ``pool.terminate(key)`` when ``plan.teardown`` is True, ``await
    session.cancel()`` on the ``OperatorSession`` when the plan is ``reuse``.
    """
    assert plan.outgoing is not None
    session = cast("OperatorSession", plan.outgoing.session)
    pool = _chat_pool(app)
    if plan.interrupt:
        return WaitOutcome(REASON_INTERRUPTED)
    while True:
        _raise_if_superseded(state, plan)
        await _raise_if_channel_closed(plan)
        if pool.get(plan.key) is not session:
            raise HandoffError.vanished(plan.key, SURFACE_SIMPLE)
        if not session.is_active:
            _raise_if_superseded(state, plan)
            return WaitOutcome(REASON_EXITED)
        if not session.is_busy:
            _raise_if_superseded(state, plan)
            return WaitOutcome(REASON_IDLE)
        await state.sleep(IDLE_POLL_S)


async def _wait_for_pty_idle(app: Any, state: HandoffState, plan: AcquirePlan) -> WaitOutcome:
    """A PTY is idle when the turn-state store, or failing that the transcript tail, says so.

    Without the turn-state hook in the TUI the store never speaks and the tail
    alone cannot tell an answered prompt from one in progress, so a plain wait
    is refused with :class:`HandoffNeedsInterrupt`. An interrupt is honoured
    either way: Escape is written once the PTY is seen busy, and the wait
    then runs for at most :data:`INTERRUPT_GRACE_S` — the interrupt marker in
    the transcript, or an idle edge, ends it as ``interrupted``; the grace
    expiring ends it as ``forced`` after ``terminate`` has been called on the
    PTY in a worker thread. A PTY already idle when an interrupt arrives is
    not sent anything.
    """
    assert plan.outgoing is not None
    session = cast("PtySession", plan.outgoing.session)
    registry = _pty_registry(app)
    if not plan.interrupt and not getattr(app.state, "turn_hook_present", False):
        raise HandoffRefused.needs_interrupt(plan.key)

    interrupt_sent = False
    deadline: float | None = None
    memo = _TailMemo()
    while True:
        _raise_if_superseded(state, plan)
        await _raise_if_channel_closed(plan)
        if registry.get_session(plan.key) is not session:
            raise HandoffError.vanished(plan.key, SURFACE_EXPERT)
        if not session.is_alive:
            _raise_if_superseded(state, plan)
            return WaitOutcome(REASON_EXITED)
        if await _pty_turn_idle(app, plan.key, memo):
            _raise_if_superseded(state, plan)
            return WaitOutcome(REASON_INTERRUPTED if interrupt_sent else REASON_IDLE)
        if plan.interrupt and not interrupt_sent:
            try:
                session.write_input(_INTERRUPT_KEY)
            except OSError as exc:
                # The master fd went away between the liveness look and the
                # write: the child is exiting. The next look classifies it.
                logger.info("Interrupt for session %s not written: %s", plan.key, exc)
            else:
                logger.info("Interrupting the terminal turn for session %s", plan.key)
            interrupt_sent = True
            deadline = state.clock() + INTERRUPT_GRACE_S
        elif deadline is not None and state.clock() >= deadline:
            logger.warning(
                "Terminal turn for session %s did not end %.0fs after the interrupt; terminating",
                plan.key,
                INTERRUPT_GRACE_S,
            )
            await asyncio.to_thread(session.terminate)
            _raise_if_superseded(state, plan)
            return WaitOutcome(REASON_FORCED)
        await state.sleep(IDLE_POLL_S)


async def _raise_if_channel_closed(plan: AcquirePlan) -> None:
    if await channel_closed(plan.channel):
        logger.info("Channel closed during the %s acquire of session %s", plan.surface, plan.key)
        raise ChannelClosed(f"channel closed during the acquire of session {plan.key!r}")


def _raise_if_superseded(state: HandoffState, plan: AcquirePlan) -> None:
    """End the wait from the superseded mark on its own record, cancellation or not.

    The supersede cancels the waiter's task for promptness, but the mark is
    what the wait is ended by: a ``task.cancel()`` that lands while the
    waiter is suspended inside the channel probe's cancel scope (Starlette's
    ``is_disconnected``) can be absorbed there as if it were the scope's own,
    and an absorbed cancellation stays counted on the task without ever
    being thrown again. So every look reads the mark first, and a marked
    wait ends here — withdrawing whatever cancel requests are still counted,
    since none of them will be delivered — with the same refusal the
    delivered cancellation turns into.
    """
    if not _superseded(state, plan):
        return
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling() > 0:
            task.uncancel()
    logger.info(
        "The %s acquire of session %s was superseded by a newer request",
        plan.surface,
        plan.key,
    )
    raise HandoffRefused.superseded(plan.key)


class _TailMemo:
    """The last tail verdict, kept while the transcript file is unchanged.

    One wait looks at the tail every :data:`IDLE_POLL_S`; a turn that runs
    for minutes would otherwise re-read the file's last 256 KB hundreds of
    times for the same answer. The file is still stat'ed on every look — that
    is what tells the two witnesses apart — and re-read only when its size or
    modification time moved, or the busy stamp the rule is asked against did.
    """

    def __init__(self) -> None:
        self._key: tuple[Path, int, int, float | None] | None = None
        self._verdict: str = "unknown"

    def verdict(self, path: Path, stat: os.stat_result, busy_since: float | None) -> str:
        key = (path, stat.st_size, stat.st_mtime_ns, busy_since)
        if key != self._key:
            self._verdict = tail_state(path, busy_since)
            self._key = key
        return self._verdict


async def _pty_turn_idle(app: Any, key: str, memo: _TailMemo | None = None) -> bool:
    """Whether the PTY under *key* is between turns, judged off the loop.

    The two witnesses are gathered here — the turn-state store entry for the
    key and the key's *current* transcript — and weighed in a worker thread,
    since the tail rule reads the transcript file. *memo* is the wait's
    :class:`_TailMemo`, so an unchanged file is not read twice.
    """
    store = get_turn_state(app, key)
    return await asyncio.to_thread(_judge_pty_idle, app, store, key, memo)


def _judge_pty_idle(
    app: Any,
    store: dict[str, Any] | None,
    key: str,
    memo: _TailMemo | None = None,
) -> bool:
    """Weigh the turn-state store against the transcript tail; the newer witness wins.

    Runs in a worker thread: resolving *key* to its current transcript may
    re-read the transcript map from disk while the map has no settled
    location, and the tail rule reads the transcript itself.

    The store entry carries the moment it was written; the transcript's
    modification time is the moment its last entry was appended. Whichever is
    newer has seen the later event and is believed, ties going to the store.
    A tail that is newer but says ``unknown`` — an assistant entry, a tool
    result — carries no evidence and hands the question back to the store.
    Without a store entry the tail is the only witness, and only an explicit
    idle shape (the interrupt marker, slash-command output) counts: a key
    nothing has reported on is not assumed idle.
    """
    path = _transcript_path(app, transcript_map.get(app, key))
    stat: os.stat_result | None = None
    if path is not None:
        try:
            stat = path.stat()
        except OSError:
            path = None
    memo = memo or _TailMemo()

    if store is None:
        return path is not None and stat is not None and memo.verdict(path, stat, None) == "idle"

    store_idle = store.get("state") == IDLE
    store_ts = float(store.get("ts") or 0.0)
    if path is None or stat is None or stat.st_mtime <= store_ts:
        return store_idle
    verdict = memo.verdict(path, stat, None if store_idle else store_ts)
    if verdict == "unknown":
        return store_idle
    return verdict == "idle"


def _transcript_path(app: Any, transcript_id: str) -> Path | None:
    """The ``.jsonl`` for *transcript_id* under the app's project, or ``None``."""
    cwd = getattr(app.state, "project_cwd", None)
    if not cwd:
        return None
    return TranscriptReader(cwd).find_transcript_by_id(transcript_id)


# ---------------------------------------------------------------------------
# Phase (c): tear down, confirm dead, spawn, release — under the lock, shielded
# ---------------------------------------------------------------------------


async def _phase_c(app: Any, plan: AcquirePlan, outcome: WaitOutcome) -> AcquireResult:
    """Carry the plan out: teardown, death check, spawn, attach, release of the slot.

    *outcome* says how the wait ended — see :class:`WaitOutcome` for what each
    reason obliges this phase to do. The work runs in its own task behind
    ``asyncio.shield``, so a cancellation of the caller cannot stop it half
    way: the caller is made to wait for the task to finish and only then
    receives its cancellation. The task itself is never cancelled — a pool
    left with a popped entry and no replacement, or a spawned child nobody
    inserted, is exactly what the shield exists to make unreachable.

    A failure of the phase after its caller was cancelled reaches nobody, so
    it is logged from a done-callback on the task rather than by the waiter:
    a second cancellation can take the waiter away from its
    ``asyncio.wait``, and the callback is what still retrieves the exception.
    """
    # From here the call is carrying the key out, not waiting: a supersede
    # must not reach it. Cleared before the shielded task exists, in the same
    # loop step that left phase (b), so no look at the key sees a waiting
    # record with a phase (c) behind it.
    pending = get_state(app).pending.get(plan.key)
    if pending is not None and pending.channel is plan.channel:
        pending.waiting = False
    task = asyncio.ensure_future(_carry_out(app, plan, outcome))
    abandoned = False

    def retrieve(done: asyncio.Future[AcquireResult]) -> None:
        if done.cancelled():
            return
        failure = done.exception()
        if failure is not None and abandoned:
            logger.warning(
                "Hand-off for session %s failed after its caller was cancelled: %s",
                plan.key,
                failure,
            )

    task.add_done_callback(retrieve)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Let the phase finish. ``asyncio.wait`` does not cancel what it waits
        # on when the waiter is cancelled, so a second cancellation delivered
        # here still leaves the task running to its end on its own.
        abandoned = True
        await asyncio.wait({task})
        raise


async def _carry_out(app: Any, plan: AcquirePlan, outcome: WaitOutcome) -> AcquireResult:
    """The body of phase (c), under the key's lock; releases the slot on every exit."""
    state = get_state(app)
    async with state.lock_for(plan.key):
        try:
            return await _teardown_and_spawn(app, state, plan, outcome)
        finally:
            release_pending(app, plan.key, plan.channel)


async def _teardown_and_spawn(
    app: Any, state: HandoffState, plan: AcquirePlan, outcome: WaitOutcome
) -> AcquireResult:
    """Clear the key of the other surface, then fill it for the incoming one.

    What clearing means depends on the plan: a ``takeover`` closes the
    displaced Expert connection and takes its attachment; a ``handoff`` tears
    the outgoing entry down and confirms its death; a Simple ``reuse`` of a
    busy chat the caller interrupted cancels that turn, and of a chat that
    died discards the corpse. Filling the key is the incoming surface's own
    pooled entry when it has one, else the caller's spawn callback. An Expert
    caller is attached last, before the slot is released by the caller of
    this function.
    """
    key = plan.key
    registry = _pty_registry(app)
    pool = _chat_pool(app)
    outgoing = plan.outgoing

    if plan.action == ACTION_TAKEOVER:
        await _displace(state, registry, plan)
    elif outgoing is not None and plan.teardown:
        await _tear_down(app, state, plan, outcome)
    elif outgoing is not None and outgoing.surface == SURFACE_SIMPLE:
        # A Simple caller on its own chat. The wait left a running turn only
        # when the caller asked for it to be cut short; a chat that died while
        # pooled is discarded here so the spawn below replaces it.
        chat = cast("OperatorSession", outgoing.session)
        if outcome.reason == REASON_INTERRUPTED:
            logger.info("Cancelling the chat turn for session %s", key)
            await chat.cancel()
        elif outcome.reason == REASON_EXITED and pool.get(key) is chat:
            await pool.terminate(key)
            logger.info("Discarded dead chat session under session %s", key)

    if plan.surface == SURFACE_EXPERT:
        pooled: PtySession | OperatorSession | None = registry.get_session(key)
    else:
        pooled = pool.get(key)

    resume_id: str | None = None
    if pooled is not None:
        session = pooled
    else:
        if plan.spawn is None:
            raise RuntimeError(
                f"acquire_surface for session {key!r} was given no spawn callback "
                f"and nothing of the {plan.surface} surface is pooled"
            )
        request = await _spawn_request(app, key, plan.surface)
        resume_id = request.resume_id
        logger.info(
            "Starting the %s process for session %s (%s)",
            plan.surface,
            key,
            f"resuming {resume_id}" if resume_id else "fresh under the key",
        )
        try:
            session = await plan.spawn(request)
        except ChatCapacityError:
            raise HandoffRefused.chat_capacity(key) from None
        now_pooled = registry.get_session(key) if plan.surface == SURFACE_EXPERT else pool.get(key)
        if now_pooled is not session:
            raise HandoffError.not_pooled(key, plan.surface)
        if plan.surface == SURFACE_EXPERT:
            reset_turn_state(app, key)

    if plan.surface == SURFACE_EXPERT and not registry.attach_session(key, plan.channel):
        raise HandoffRefused.attached_elsewhere(key)

    return AcquireResult(
        plan=plan,
        outcome=outcome,
        session=session,
        spawned=pooled is None,
        resume_id=resume_id,
    )


async def _displace(state: HandoffState, registry: PtyRegistry, plan: AcquirePlan) -> None:
    """Close the Expert connection holding the PTY and take its attachment.

    The displaced handler registered how to close its socket under its attach
    token; a token with no closer belongs to a handler already on its way
    out. The detach here carries the displaced token, so the registry
    releases the key from its side too, and the handler's own later detach —
    with a token that no longer holds the key — is a no-op.
    """
    owner = plan.displaced_owner
    closer = state.closers.get(owner) if owner is not None else None
    if closer is not None:
        logger.info("Closing the terminal connection displaced from session %s", plan.key)
        try:
            await closer()
        except Exception:
            logger.warning(
                "Closing the displaced terminal connection for session %s failed",
                plan.key,
                exc_info=True,
            )
    if owner is not None:
        registry.detach_session(plan.key, owner)


async def _tear_down(
    app: Any, state: HandoffState, plan: AcquirePlan, outcome: WaitOutcome
) -> None:
    """Pop the outgoing entry, kill it off the loop, and see it dead.

    Whatever phase (b) reported, the same sequence runs: a ``forced`` wait has
    already called ``terminate`` once but popped nothing and proved nothing;
    an ``exited`` child still has an exit status to collect. The entry is
    identity-checked against its pool first — one already gone leaves nothing
    to do; one replaced by something else is a :class:`HandoffError`. A
    survivor is put back where it came from, unattached, and the acquire
    refused with 503.
    """
    assert plan.outgoing is not None
    key = plan.key
    if plan.outgoing.surface == SURFACE_EXPERT:
        pty = cast("PtySession", plan.outgoing.session)
        registry = _pty_registry(app)
        pooled = registry.get_session(key)
        if pooled is None:
            logger.info("Outgoing PTY for session %s already left its pool", key)
        elif pooled is not pty:
            raise HandoffError.vanished(key, SURFACE_EXPERT)
        else:
            registry.pop_session(key)
        # ``terminate`` on a dead child only closes the fd and collects the
        # exit status; on a live one it is the blocking signal escalation.
        await asyncio.to_thread(pty.terminate)
        if not await _observed_dead(state, lambda: not pty.is_alive):
            logger.warning(
                "PTY for session %s survived its kill; refusing the %s acquire",
                key,
                plan.surface,
            )
            if not registry.reinsert(key, pty):
                logger.error("PTY survivor for session %s could not be pooled again", key)
            raise HandoffRefused.outgoing_still_running(key)
        reset_turn_state(app, key)
        logger.info("Terminal for session %s torn down (%s)", key, outcome.reason)
        return

    chat = cast("OperatorSession", plan.outgoing.session)
    pool = _chat_pool(app)
    pooled_chat = pool.get(key)
    if pooled_chat is None:
        # Whoever popped it also tore it down; ``teardown`` is idempotent
        # and re-signals a child that is still running.
        logger.info("Outgoing chat for session %s already left its pool", key)
        await chat.teardown()
    elif pooled_chat is not chat:
        raise HandoffError.vanished(key, SURFACE_SIMPLE)
    else:
        # Pops and tears down; an ``interrupted`` outcome's running turn is
        # cancelled by the teardown itself.
        await pool.terminate(key)
    if chat.process_exited is None:
        logger.warning(
            "Chat session %s exposes no process handle; its child cannot be observed dead",
            key,
        )
    elif not await _observed_dead(state, lambda: chat.process_exited is not False):
        logger.warning(
            "Chat child for session %s survived its teardown; refusing the %s acquire",
            key,
            plan.surface,
        )
        if not await pool.reinsert(key, chat):
            logger.error("Chat survivor for session %s could not be pooled again", key)
        raise HandoffRefused.outgoing_still_running(key)
    logger.info("Chat for session %s torn down (%s)", key, outcome.reason)


async def _observed_dead(state: HandoffState, dead: Callable[[], bool]) -> bool:
    """Poll *dead* every :data:`DEATH_POLL_S` for up to :data:`DEATH_GRACE_S`."""
    deadline = state.clock() + DEATH_GRACE_S
    while True:
        if dead():
            return True
        now = state.clock()
        if now >= deadline:
            return False
        await state.sleep(min(DEATH_POLL_S, deadline - now))


async def _spawn_request(app: Any, key: str, surface: Surface) -> SpawnRequest:
    """Decide what the new process resumes, from the map and the disk as of now.

    The key's current transcript is resumed when its file exists. A map
    entry whose file is gone falls back to the key's own transcript when
    *that* exists — the conversation the key started with is better than
    none — and a key with no transcript on disk at all starts fresh under
    the key. The map look-up (a file read when the map is provisional) and
    the directory listing run off the loop together.
    """
    transcript_id, on_disk = await asyncio.to_thread(_transcript_and_disk, app, key)
    if transcript_id in on_disk:
        resume_id: str | None = transcript_id
    elif key in on_disk:
        resume_id = key
    else:
        resume_id = None
    return SpawnRequest(key=key, surface=surface, resume_id=resume_id, transcript_id=transcript_id)


def _transcript_and_disk(app: Any, key: str) -> tuple[str, set[str]]:
    """The key's current transcript id and the transcripts on disk, as of now."""
    return transcript_map.get(app, key), _transcripts_on_disk(app)


def _transcripts_on_disk(app: Any) -> set[str]:
    """The ids of every transcript under the app's project; empty without a project."""
    cwd = getattr(app.state, "project_cwd", None)
    if not cwd:
        return set()
    return SessionDiscovery(cwd).snapshot_session_ids()
