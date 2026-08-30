"""LRU pool of Simple-mode chat sessions.

The pool owns the chat-keyed session lifecycle: get-or-create with a per-key
pending Future (concurrent double-submits share one creation), LRU capacity
eviction, idle reaping, and busy-safe teardown. It builds sessions through an
injected ``factory`` and drives them only through the public
:class:`~osprey.interfaces.web_terminal.operator_session.OperatorSession`
surface (``start``/``is_active``/``is_busy``/``last_activity``/``teardown``),
so any conforming double can stand in.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

from osprey.interfaces.web_terminal.pty_manager import EMPTY_ENV_FINGERPRINT, env_fingerprint

if TYPE_CHECKING:
    from osprey.interfaces.web_terminal.operator_session import OperatorSession

logger = logging.getLogger(__name__)


class ChatCapacityError(RuntimeError):
    """Raised when a new chat session is requested at capacity and every existing
    chat is busy (no evictable session). Routes map this to HTTP 429.
    """


class ChatSessionTerminatedError(RuntimeError):
    """Raised when a chat session is terminated while it was still starting.

    ``terminate``/``drain_all`` cannot pop a session that has not been
    registered yet, so they mark the in-flight creation *superseded* instead;
    the creator then tears the started session down rather than registering it,
    and raises this. Routes map it to HTTP 409.

    Without the marker a terminate racing a first prompt would be silently
    undone by the creation that outlived it. On the posture path that is the
    failure that matters: the operator is told the flip applied while a child
    built from the pre-flip environment lands in the pool a moment later and
    keeps running the posture they just stepped out of.
    """


class ChatSessionPool:
    """LRU-ordered pool of chat sessions with capacity eviction and idle reaping.

    The lock is held only for map inspection/mutation, never across
    ``session.start()`` or teardown. ``factory(cwd, env)`` returns an unstarted
    session; the pool starts it outside the lock.
    """

    def __init__(
        self,
        factory: Callable[[str, dict[str, str] | None], OperatorSession],
        max_sessions: int = 5,
        idle_seconds: float = 900.0,
    ) -> None:
        self._factory = factory
        # LRU-ordered; newest at the end.
        self._sessions: OrderedDict[str, OperatorSession] = OrderedDict()
        self._lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[OperatorSession]] = {}
        # Fingerprint of the env each key's child was built from — written when
        # a creation is registered and kept as that creation becomes a session,
        # so a pending and a pooled entry answer the same question. See
        # get_or_create's reuse check. The digest, never the mapping: a chat
        # child's env carries the panel token.
        self._env_fingerprints: dict[str, str] = {}
        # Creations that a terminate/drain overtook. Keyed on the Future rather
        # than the chat id, so a *later* creation under the same id is not
        # collateral damage of an earlier terminate.
        self._superseded: set[asyncio.Future[OperatorSession]] = set()
        self._max_sessions = max_sessions
        self._idle_seconds = idle_seconds

    async def get_or_create(
        self,
        chat_id: str,
        cwd: str,
        env: dict[str, str] | Callable[[], dict[str, str] | None] | None = None,
    ) -> tuple[OperatorSession, bool]:
        """Return a live session for ``chat_id``, creating it if needed.

        Returns ``(session, was_reused)`` — ``was_reused`` is True when an
        existing live session is returned or when this call joined an in-flight
        creation started by a concurrent double-submit.

        The lock is held only to inspect/mutate the map and elect a creator; it
        is released before ``session.start()``. Concurrent callers for the same
        ``chat_id`` await a shared pending Future rather than starting duplicate
        SDK subprocesses. At capacity the least-recently-used non-busy session
        is evicted; if every chat is busy, :class:`ChatCapacityError` is raised.

        *env* may be a ready-made mapping **or a zero-arg builder** that
        produces one. The builder form exists for callers whose environment is
        derived from mutable state a concurrent request can change — the chat
        route's runtime posture is the live case — and it is what makes the
        read and this creation's registration **atomic**: a builder is called
        inside the same uninterrupted ``_lock`` hold that registers
        ``_pending``, so a :meth:`terminate` racing it either lands *before*
        the read (and the child is built from the new state) or *after* the
        registration (and the creation is superseded). There is no third
        interleaving in which a child built from stale state reaches the pool.
        A caller that resolves its own mapping first reopens exactly that gap
        the moment anything awaits between the read and this call, and no test
        would notice; hand over the builder instead. Builders must be
        synchronous — the lock is held across the call, so it must not await.
        They are also called on **every** invocation, reuse included, because
        the reuse check below compares against what they produce.

        **A live entry is only reused when its environment still matches.** A
        child's environment is fixed when it is spawned and cannot be
        amended. A runtime posture flip is not such a change: it lands in
        the per-target posture store and is read at write time, never
        stamped into the child's environment, so the control-target chip in
        the header reflects it without any rebuild. The comparison is on
        :func:`~osprey.interfaces.web_terminal.pty_manager.env_fingerprint`,
        the same digest and the same deny list the PTY pool uses, so the two
        topologies cannot drift on what "the same environment" means.

        A posture change never reaches this check at all: it lands in the
        per-target posture store, and the running chat child reads it live at
        every write-time gate, so the same live entry keeps serving the
        conversation right through a narrowing or a widen. The reuse rule
        above is unaffected — it exists for other env changes only.
        """
        to_stop: list[OperatorSession] = []
        builder_error: BaseException | None = None
        creator = False
        pending = None

        async with self._lock:
            # Resolve the environment HERE, under the same lock hold that
            # inspects the map and registers ``_pending`` below, and before
            # anything can await — that adjacency is the atomicity (see the
            # docstring). Resolved before the reuse check as well as the
            # capacity check: the reuse check now compares against it, a
            # raising builder must not strand an evicted victim nobody tears
            # down, and its error is held rather than raised so a dead entry
            # popped below still gets its teardown before the error leaves.
            try:
                resolved_env = env() if callable(env) else env
            except BaseException as exc:
                builder_error = exc
                fingerprint = None
            else:
                fingerprint = env_fingerprint(resolved_env)

            existing = self._sessions.get(chat_id)
            if existing is not None and not existing.is_active:
                # Dead entry — drop and tear down below (outside the lock).
                # Unconditional on the builder's outcome: a builder that raised
                # must not leave this popped-but-unreaped, and a corpse is
                # nothing to compare an environment against anyway.
                self._sessions.pop(chat_id, None)
                self._env_fingerprints.pop(chat_id, None)
                to_stop.append(existing)
            elif existing is not None and builder_error is None:
                recorded = self._env_fingerprints.get(chat_id, EMPTY_ENV_FINGERPRINT)
                if recorded == fingerprint:
                    self._sessions.move_to_end(chat_id)  # LRU bump
                    return existing, True
                # The env changed under a live child. A child's environment is
                # fixed at spawn and cannot be amended, so the only way to
                # deliver the change is to tear it down and build a new one —
                # busy or not, exactly as :meth:`terminate` does, because a
                # turn running under a posture the operator has revoked is the
                # case this exists for. Values are never logged: the env
                # carries the panel token.
                logger.info(
                    "Launch env changed for chat session %r — tearing the warm "
                    "session down so the new environment reaches a fresh child",
                    chat_id,
                )
                self._sessions.pop(chat_id, None)
                self._env_fingerprints.pop(chat_id, None)
                to_stop.append(existing)
            # A live entry with a builder that raised is left exactly as it is:
            # a failed read of the posture store is no reason to kill a child.

            if builder_error is None:
                pending = self._pending.get(chat_id)
                if pending is not None and self._env_fingerprints.get(chat_id) != fingerprint:
                    # A creation is in flight that was built from a different
                    # environment. Joining it would hand this caller the very
                    # child the change was meant to replace, so it is overtaken
                    # instead and this call becomes the creator. The marker is
                    # keyed on the Future, so the creation registered below is
                    # not collateral damage of this supersede.
                    self._supersede_pending(chat_id)
                    pending = None

                if pending is None:
                    # We will create — reserve a slot, evicting if at capacity.
                    # Count live sessions plus in-flight creations toward the
                    # cap. The capacity raise below cannot strand anything: the
                    # only entries in ``to_stop`` at this point are ones this
                    # call already removed from the map, and those pops are
                    # what keep the count under the cap here.
                    if len(self._sessions) + len(self._pending) >= self._max_sessions:
                        victim = self._pick_evictable_victim()
                        if victim is None:
                            raise ChatCapacityError(
                                "all chat sessions are busy; cannot create a new one"
                            )
                        to_stop.append(victim)
                    pending = asyncio.get_running_loop().create_future()
                    self._pending[chat_id] = pending
                    self._env_fingerprints[chat_id] = fingerprint
                    creator = True

        # Teardowns happen outside the lock (never block map access on stop()).
        # Teardown failures are logged, never raised (a cancellation of THIS
        # task still propagates): this runs *after* ``_pending`` is registered
        # and outside the ``except`` that clears it, so a raising
        # teardown would leave the key pending forever — every later
        # ``get_or_create`` joining a Future nobody will ever settle, and a
        # terminate adding that orphan to ``_superseded``, which nothing
        # discards. The sessions here are already out of the map and are dead
        # or evicted, so a failed stop costs a leaked child, not a wedged key.
        if to_stop:
            results = await asyncio.gather(*(s.teardown() for s in to_stop), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    logger.warning(
                        "Chat session teardown failed while creating %r; continuing",
                        chat_id,
                        exc_info=result,
                    )

        if builder_error is not None:
            raise builder_error

        if not creator:
            # Join the in-flight creation; propagate its outcome.
            session = await pending
            return session, True

        try:
            session = self._factory(cwd, resolved_env)
            await session.start()  # deliberately outside the lock
        except BaseException as exc:
            async with self._lock:
                if self._pending.get(chat_id) is pending:
                    # Ours to clear — and the fingerprint with it. Guarded,
                    # because a later creation may already own both.
                    del self._pending[chat_id]
                    self._env_fingerprints.pop(chat_id, None)
                self._superseded.discard(pending)
            if not pending.done():
                pending.set_exception(exc)
            # Consume the future's exception so a creator-only failure (no
            # concurrent joiner to await it) doesn't warn at GC. Joiners that
            # do await it still receive the exception — this only clears the
            # "exception was never retrieved" flag.
            if pending.done() and not pending.cancelled():
                pending.exception()
            raise

        async with self._lock:
            mine = self._pending.get(chat_id) is pending
            if mine:
                del self._pending[chat_id]
            superseded = pending in self._superseded
            self._superseded.discard(pending)
            if not superseded:
                # The fingerprint registered with this creation carries over
                # unchanged: it already describes the child now landing here.
                self._sessions[chat_id] = session
            elif mine:
                # Nothing of this creation survives, so neither does its
                # fingerprint — unless a later creation has already claimed
                # the key, which ``mine`` is what tells us.
                self._env_fingerprints.pop(chat_id, None)

        if superseded:
            # A terminate/drain arrived while this session was starting. It had
            # nothing to pop then; honouring the marker here is what makes
            # "terminated" mean there is no live child under this key
            # afterwards. Joiners see the same refusal — handing one of them a
            # session that is about to be torn down would be worse than an
            # error they can act on.
            await session.teardown()
            terminated = ChatSessionTerminatedError(
                f"chat session {chat_id!r} was terminated while it was starting"
            )
            if not pending.done():
                pending.set_exception(terminated)
            if pending.done() and not pending.cancelled():
                # Consume the flag; awaiting joiners still receive the exception.
                pending.exception()
            raise terminated

        if not pending.done():
            pending.set_result(session)
        return session, False

    def get(self, chat_id: str) -> OperatorSession | None:
        return self._sessions.get(chat_id)

    def has_key(self, chat_id: str) -> bool:
        """Whether the pool would answer to *chat_id* — pooled **or starting**.

        The membership question :meth:`get` cannot answer. ``get`` reads the
        session map only, and a creation still inside ``start()`` lives in
        ``_pending`` instead; a caller asking "is there a chat here?" during
        that window is asking about the very child a posture flip most needs to
        catch, and ``get`` would tell it "no".

        Read-only in the strong sense: no ``move_to_end``, no ``last_activity``
        touch, no creation — an existence probe must not refresh an idle clock
        or elect an eviction victim. Both maps are mutated only under
        ``_lock``, and every mutation is a plain assignment or pop with no
        ``await`` between the two maps, so this synchronous read sees a
        consistent snapshot without taking the lock. It deliberately does not
        take it: the callers are synchronous gates, and a probe that could
        block behind a creation would make the toggle wait on the child it is
        trying to replace.

        Liveness is not part of the answer, exactly as in :meth:`get`: a
        dead-but-unreaped entry still names a chat the operator can address,
        and terminating it evicts the corpse.
        """
        return chat_id in self._sessions or chat_id in self._pending

    async def terminate(self, chat_id: str) -> None:
        """Evict and tear down a chat session (busy-safe, idempotent).

        Eviction is the half that makes a respawn possible: with the entry
        gone, the next :meth:`get_or_create` builds a fresh child from the
        environment it is handed then — which is how a runtime posture change
        reaches an SDK agent at all.

        A creation still in flight cannot be popped, so it is marked superseded
        and torn down by its own creator the moment ``start()`` returns (see
        :class:`ChatSessionTerminatedError`). This call does not wait for that:
        a hung ``start()`` must not hang the operator's toggle.
        """
        async with self._lock:
            session = self._sessions.pop(chat_id, None)
            self._supersede_pending(chat_id)
            self._env_fingerprints.pop(chat_id, None)
        if session is not None:
            await session.teardown()

    async def reap_idle(self) -> int:
        """Tear down every idle chat session; return how many were reaped.

        Idle = not busy (no in-flight turn, or a zombie whose reader and quiesce
        are both done) AND ``last_activity`` older than ``idle_seconds``.
        """
        now = time.monotonic()
        async with self._lock:
            idle_keys = [k for k, s in self._sessions.items() if self._is_idle(s, now)]
            victims = [self._sessions.pop(k) for k in idle_keys]
            for key in idle_keys:
                self._env_fingerprints.pop(key, None)
        if victims:
            await asyncio.gather(*(s.teardown() for s in victims))
        return len(victims)

    async def drain_all(self) -> None:
        """Tear down every session concurrently (shutdown path)."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            # Clearing ``_pending`` alone would let a creation that is still
            # starting register itself into the drained pool behind us.
            self._superseded.update(self._pending.values())
            self._pending.clear()
            self._env_fingerprints.clear()
        if sessions:
            await asyncio.gather(*(s.teardown() for s in sessions))

    # ---- Internals ---- #

    def _supersede_pending(self, chat_id: str) -> None:
        """Mark any in-flight creation for *chat_id* as overtaken.

        Caller must hold ``_lock``.
        """
        pending = self._pending.get(chat_id)
        if pending is not None:
            self._superseded.add(pending)

    def _is_idle(self, session: OperatorSession, now: float) -> bool:
        """Not busy AND ``last_activity`` older than ``idle_seconds``."""
        if session.is_busy:
            return False
        return (now - session.last_activity) >= self._idle_seconds

    def _pick_evictable_victim(self) -> OperatorSession | None:
        """Pop and return the least-recently-used non-busy chat, or None.

        Caller must hold ``_lock``.
        """
        victim_key = None
        for candidate in self._sessions:  # OrderedDict iterates LRU-first
            if not self._sessions[candidate].is_busy:
                victim_key = candidate
                break
        if victim_key is None:
            return None
        self._env_fingerprints.pop(victim_key, None)
        return self._sessions.pop(victim_key)
