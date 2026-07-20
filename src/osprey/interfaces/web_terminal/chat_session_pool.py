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

if TYPE_CHECKING:
    from osprey.interfaces.web_terminal.operator_session import OperatorSession

logger = logging.getLogger(__name__)


class ChatCapacityError(RuntimeError):
    """Raised when a new chat session is requested at capacity and every existing
    chat is busy (no evictable session). Routes map this to HTTP 429.
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
        self._max_sessions = max_sessions
        self._idle_seconds = idle_seconds

    async def get_or_create(
        self, chat_id: str, cwd: str, env: dict[str, str] | None = None
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
        """
        to_stop: list[OperatorSession] = []

        async with self._lock:
            existing = self._sessions.get(chat_id)
            if existing is not None and existing.is_active:
                self._sessions.move_to_end(chat_id)  # LRU bump
                return existing, True
            if existing is not None:
                # Dead entry — drop and tear down below (outside the lock).
                self._sessions.pop(chat_id, None)
                to_stop.append(existing)

            pending = self._pending.get(chat_id)
            if pending is not None:
                creator = False
            else:
                # We will create — reserve a slot, evicting if at capacity.
                # Count live sessions plus in-flight creations toward the cap.
                if len(self._sessions) + len(self._pending) >= self._max_sessions:
                    victim = self._pick_evictable_victim()
                    if victim is None:
                        raise ChatCapacityError(
                            "all chat sessions are busy; cannot create a new one"
                        )
                    to_stop.append(victim)
                pending = asyncio.get_running_loop().create_future()
                self._pending[chat_id] = pending
                creator = True

        # Teardowns happen outside the lock (never block map access on stop()).
        if to_stop:
            await asyncio.gather(*(s.teardown() for s in to_stop))

        if not creator:
            # Join the in-flight creation; propagate its outcome.
            session = await pending
            return session, True

        try:
            session = self._factory(cwd, env)
            await session.start()  # deliberately outside the lock
        except BaseException as exc:
            async with self._lock:
                if self._pending.get(chat_id) is pending:
                    del self._pending[chat_id]
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
            self._sessions[chat_id] = session
            if self._pending.get(chat_id) is pending:
                del self._pending[chat_id]
        if not pending.done():
            pending.set_result(session)
        return session, False

    def get(self, chat_id: str) -> OperatorSession | None:
        return self._sessions.get(chat_id)

    async def terminate(self, chat_id: str) -> None:
        """Explicitly tear down a chat session (busy-safe, idempotent)."""
        async with self._lock:
            session = self._sessions.pop(chat_id, None)
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
        if victims:
            await asyncio.gather(*(s.teardown() for s in victims))
        return len(victims)

    async def drain_all(self) -> None:
        """Tear down every session concurrently (shutdown path)."""
        async with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
            self._pending.clear()
        if sessions:
            await asyncio.gather(*(s.teardown() for s in sessions))

    # ---- Internals ---- #

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
        return self._sessions.pop(victim_key)
