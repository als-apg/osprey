"""The one event loop every Channel Access call in this process runs on.

aioca is asyncio-only and binds each channel, and each monitor, to the event
loop that opened it: it keeps one channel cache per loop and never purges the
cache of a loop that has finished. The connector is called from many loops —
the MCP server's, ``osprey.runtime``'s ``asyncio.run`` per call (a fresh
thread *and* loop per call under Jupyter), the connector host's, the
RunEngine's. Calling aioca on the caller's loop would reconnect every channel
on every call, leak a cache per call, and end a subscription with the loop
that opened it.

So every aioca call is marshalled onto one long-lived loop on one daemon
thread, shared by everything in the process. aioca's CA context is created on
that thread the first time it is used, and any other thread aioca later runs
on (ophyd-async on the RunEngine's, say) attaches to the same context.

At exit, aioca's own ``atexit`` hook purges its channels and, when it created
the context, destroys it. Nothing may still be running on the loop by then: a
channel created or read on the loop thread while the context is destroyed
under it crashes the process. So the loop is stopped, and its thread joined,
from ``threading``'s shutdown hook -- the one ``concurrent.futures`` stops its
workers from -- which runs before every ``atexit`` hook, whichever of the two
was registered first.

The context itself is then left for the process's exit to reclaim rather
than destroyed: ``ca_context_destroy`` waits out a TCP circuit still trying to
reach a name server nobody answers (an unreachable
``EPICS_CA_NAME_SERVERS`` gateway), which held a process's exit for half a
minute. aioca still clears its channels first, and makes no CA call after.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import logging
import os
import sys
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

#: Runs at interpreter shutdown before any ``atexit`` hook; private, but the
#: stdlib's own ``concurrent.futures`` relies on it for the same reason.
#: CPython has had it since 3.9.
_register_before_atexit: Callable[[Callable[[], None]], Any] = getattr(
    threading, "_register_atexit", atexit.register
)

_THREAD_NAME = "osprey-ca-loop"

#: The environment a Channel Access context is created from.
CA_ENV_KEYS = (
    "EPICS_CA_ADDR_LIST",
    "EPICS_CA_SERVER_PORT",
    "EPICS_CA_NAME_SERVERS",
    "EPICS_CA_AUTO_ADDR_LIST",
)


def current_ca_env() -> dict[str, str | None]:
    """The ``EPICS_CA_*`` settings a context created now would run with."""
    return {key: os.environ.get(key) for key in CA_ENV_KEYS}


def _drop_ca_context() -> None:
    """Close every aioca channel and let the next CA call create a fresh context.

    Called on the loop thread, the one that created the context and is
    attached to it. The context is detached rather than destroyed; any other
    thread still attached to it is moved to the new one by aioca itself the
    next time it opens a channel cache.
    """
    catools = sys.modules.get("aioca._catools")
    if catools is None:
        return
    context = catools._Context
    context.purge_channel_caches()
    if context._ca_context:
        catools.cadef.ca_detach_context()
    context._ca_context = None
    context._should_destroy = False


class CaLoop:
    """An event loop running forever on a daemon thread of its own."""

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
        self._thread.start()
        self._env: dict[str, str | None] | None = None
        self._used = False
        self._env_lock = threading.Lock()
        # What re-opens each long-lived monitor after a repoint. A dict for its
        # insertion order, so monitors come back in the order they were opened.
        # Read and written on the loop thread only, like everything aioca does.
        self._reopeners: dict[Callable[[], None], None] = {}

    async def point_at(self, env: dict[str, str | None]) -> None:
        """Make Channel Access run with these ``EPICS_CA_*`` settings from now on.

        CA reads its environment once, when the context is created on the
        first call. So the settings are written to ``os.environ`` and, when a
        context already serves different ones, that context is dropped for the
        next call to create a fresh one from them -- what pyepics'
        ``clear_cache`` did on every connect. A notebook kernel that follows a
        target switch relies on it, and so does a read-only connect after a
        write one: it must not ride the earlier connector's write gateway.

        Every channel is closed, monitors included, and everything else in the
        process that uses CA follows the new settings: there is one context
        per process. The old context is detached and left, not destroyed:
        ``ca_context_destroy`` can wait half a minute on a name server nobody
        answers (pyepics left it the same way).

        aioca closes a purged monitor without a word to its callback, so a
        monitor that must outlive a repoint registers a re-opener with
        :meth:`add_reopener`; each one is run on the new context right after
        the old one is dropped, before any other call can reach the loop.

        Runs on the loop thread, so it cannot interleave with a CA call.
        """
        if self.on_loop_thread():
            self._point_at(env)
            return

        async def on_loop() -> None:
            self._point_at(env)

        self._refuse_in_a_forked_child()
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(on_loop(), self._loop))

    def _point_at(self, env: dict[str, str | None]) -> None:
        with self._env_lock:
            for key, setting in env.items():
                if setting is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = setting
            dropped = self._used and env != self._env
            if dropped:
                _drop_ca_context()
                self._used = False
            self._env = dict(env)
        if dropped and self._reopeners:
            # Outside the lock: re-opening is a CA call, which marks the loop
            # used again under that same lock.
            self._mark_used()
            for reopen in list(self._reopeners):
                try:
                    reopen()
                except Exception:
                    # One channel that cannot be re-opened must not keep the
                    # rest of the process's monitors closed.
                    logger.warning("A Channel Access monitor was not re-opened", exc_info=True)

    def add_reopener(self, reopen: Callable[[], None]) -> None:
        """Run ``reopen`` on the loop thread each time a repoint drops the context.

        For a monitor that must outlive a repoint: ``reopen`` opens it again on
        the fresh context. Called on the loop thread, as is
        :meth:`discard_reopener`; a monitor's open and its close run there
        too, so whichever of them comes second sees the other.
        """
        self._reopeners[reopen] = None

    def discard_reopener(self, reopen: Callable[[], None]) -> None:
        """Stop re-opening this monitor; a no-op for one never added. Loop thread only."""
        self._reopeners.pop(reopen, None)

    def _mark_used(self) -> None:
        """Record that CA has been called, pinning the environment it runs with."""
        with self._env_lock:
            if not self._used:
                self._used = True
                if self._env is None:
                    self._env = current_ca_env()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The loop itself, for callers that schedule plain callbacks on it."""
        return self._loop

    def on_loop_thread(self) -> bool:
        """True when the calling thread is this loop's own thread."""
        return threading.current_thread() is self._thread

    def _refuse_in_a_forked_child(self, coro: Coroutine[Any, Any, Any] | None = None) -> None:
        """Raise in a process forked from the one that started the loop.

        A child inherits the loop but not its thread, so work queued to it
        would wait forever; nor can it use the parent's libca context.
        """
        if os.getpid() != self._pid:
            if coro is not None:
                coro.close()
            raise RuntimeError(
                "Channel Access was started in the parent of this forked process and "
                "cannot be used here; start the child with 'spawn' or 'forkserver'."
            )

    def submit(self, coro: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        """Schedule ``coro`` on the loop, from any thread; return its future."""
        self._refuse_in_a_forked_child(coro)
        self._mark_used()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Await ``coro`` on the CA loop from whichever loop the caller is on."""
        if self.on_loop_thread():
            self._mark_used()
            return await coro
        return await asyncio.wrap_future(self.submit(coro))

    def run_sync(self, coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        """Run ``coro`` on the CA loop and block the calling thread for its result.

        Blocking the loop's own thread on work queued to that same loop would
        never return, so that is refused outright rather than left to hang.
        """
        if self.on_loop_thread():
            coro.close()
            raise RuntimeError("blocking on the Channel Access loop from its own thread")
        return self.submit(coro).result(timeout)

    def call_soon(self, callback: Callable[..., Any], *args: Any) -> None:
        """Run a plain callback on the loop thread, from any thread."""
        self._refuse_in_a_forked_child()
        self._loop.call_soon_threadsafe(callback, *args)

    def stop(self) -> None:
        """Stop the loop and wait briefly for its thread. Idempotent."""
        if self._loop.is_closed() or os.getpid() != self._pid:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)


_ca_loop: CaLoop | None = None
_ca_loop_lock = threading.Lock()


def _shut_down() -> None:
    """Stop the process's CA loop; leave the CA context for the exit to reclaim."""
    if _ca_loop is not None:
        _ca_loop.stop()
    catools = sys.modules.get("aioca._catools")
    if catools is not None:
        catools._Context._should_destroy = False


def get_ca_loop() -> CaLoop:
    """The process's Channel Access loop, started on first use."""
    global _ca_loop
    with _ca_loop_lock:
        if _ca_loop is None:
            _ca_loop = CaLoop()
            try:
                _register_before_atexit(_shut_down)
            except RuntimeError:
                # First used while the interpreter is already shutting down.
                atexit.register(_shut_down)
        return _ca_loop
