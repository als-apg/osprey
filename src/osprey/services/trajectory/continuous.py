"""Continuous mode: the holder's chain, run again and again until something ends it.

A :class:`ContinuousRun` belongs to the service's
:class:`~osprey.services.trajectory.sessions.Sessions`, which starts it for the
holder and keeps at most one: the run is the only occupant of the run slot, and
a second start while one runs is refused. Its driver task submits one chain
action per iteration through :meth:`Sessions.shot`, each built from the chain
body the run was started with, so every iteration runs the same steps and its
own setup re-check before it injects, and the post-shot hook recomputes the
step after every stored shot. Each iteration is its own turn of the action
layer: the run lock is taken for the iteration and released before the next is
submitted, which lets any other queued action run in between. The run fires
chains only; it never applies a step.

Two iterations start at least ``min_period_s`` apart, measured start to start;
the wait between them ends early on the run's end.

The run ends, first cause wins, when:

- the holder stop event current at its start is set (``stopped``): a Stop, a
  take-over or a release set it, and an iteration queued after it is refused
  by the layer or stopped mid-shot;
- its lease lapsed (``lease_expired``): only a continuous run holds a lease,
  moved forward by each heartbeat, and a lease no heartbeat renewed within
  ``lease_s`` ends the run, voids the pending step and renews the holder stop
  event, so a shot in flight stops at its next checkpoint;
- an iteration raised anything else while the stop event was clear, or every
  injection of an iteration's shot failed (``failed``);
- the owner of the sessions ended it by name (:meth:`ContinuousRun.end`:
  ``taken_over``, ``released``) or cancelled it (:meth:`ContinuousRun.cancel`,
  ``shutdown``).

A zero step ends nothing, and no count of shots does. The end is published once
as a :class:`~.state.ContinuousEndedEvent`, handed to the ``on_end`` callback
and kept on :attr:`ContinuousRun.ended`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from osprey.services.trajectory.actions import ACTION_STOPPED, Action, ActionRefused
from osprey.services.trajectory.constants import DEFAULT_CONTINUOUS_MIN_PERIOD_S
from osprey.services.trajectory.state import (
    ChainResult,
    ContinuousEnd,
    ContinuousEndedEvent,
    ContinuousStatus,
    ShotResult,
)

if TYPE_CHECKING:
    from osprey.services.trajectory.acquisition import PostShotHook
    from osprey.services.trajectory.sessions import Sessions
    from osprey.services.trajectory.store import TrajectoryStore

__all__ = [
    "DEFAULT_HEARTBEAT_S",
    "DEFAULT_LEASE_S",
    "ContinuousRun",
]

#: Seconds between two heartbeats a client of a continuous run sends.
DEFAULT_HEARTBEAT_S = 5.0
#: Seconds a continuous run's lease lasts without a heartbeat.
DEFAULT_LEASE_S = 10.0


class ContinuousRun:
    """One continuous run of the holder's chain.

    Attributes:
        owner: The holder's owner the shots are submitted for.
        kind: The holder's door.
        client_id: The holder's session token.
        min_period_s: Seconds at least between the starts of two iterations.
        heartbeat_s: Seconds between two heartbeats a client sends.
        lease_s: Seconds the lease lasts without a heartbeat.
        shots: Iterations submitted so far.
        ended: The ended event, once the run ended.
        error: What an iteration raised when the run ended ``failed``, else
            ``None``.
    """

    def __init__(
        self,
        sessions: Sessions,
        owner: Any,
        kind: str,
        client_id: str,
        store: TrajectoryStore,
        build: Callable[[PostShotHook], Action[Any]],
        *,
        min_period_s: float = DEFAULT_CONTINUOUS_MIN_PERIOD_S,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
        lease_s: float = DEFAULT_LEASE_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        on_end: Callable[[ContinuousEndedEvent], Any] | None = None,
    ) -> None:
        if not min_period_s >= 0:
            raise ValueError(f"the minimum period must be at least 0 s, not {min_period_s}")
        if not 0 < heartbeat_s < lease_s:
            raise ValueError(
                f"the heartbeat interval ({heartbeat_s} s) must be positive and shorter "
                f"than the lease ({lease_s} s)"
            )
        self.owner = owner
        self.kind = kind
        self.client_id = client_id
        self.min_period_s = min_period_s
        self.heartbeat_s = heartbeat_s
        self.lease_s = lease_s
        self.shots = 0
        self.ended: ContinuousEndedEvent | None = None
        self.error: BaseException | None = None
        self._sessions = sessions
        self._store = store
        self._build = build
        self._clock = clock
        self._sleep = sleep
        self._on_end = on_end
        self._end: ContinuousEnd | None = None
        self._deadline = 0.0
        self._stop: asyncio.Event | None = None
        self._driver: asyncio.Task[None] | None = None
        self._lease: asyncio.Task[None] | None = None

    # -- life ----------------------------------------------------------------

    def start(self) -> None:
        """Capture the holder stop event, start the lease and the driver task."""
        if self._driver is not None:
            raise RuntimeError("a continuous run starts once")
        self._stop = self._sessions.layer.stop_event
        self._deadline = self._clock() + self.lease_s
        loop = asyncio.get_running_loop()
        self._driver = loop.create_task(self._drive(), name="trajectory-continuous-driver")
        self._lease = loop.create_task(self._watch_lease(), name="trajectory-continuous-lease")

    @property
    def running(self) -> bool:
        """Whether the run started and has not ended."""
        return self._driver is not None and self.ended is None

    def heartbeat(self) -> ContinuousStatus:
        """Move the lease deadline ``lease_s`` from now; nothing else."""
        if self.running and self._end is None:
            self._deadline = self._clock() + self.lease_s
        return self.status()

    def status(self) -> ContinuousStatus:
        """The run's iterations so far and the seconds its lease has left."""
        expires_in = max(0.0, self._deadline - self._clock()) if self.running else None
        return ContinuousStatus(shots=self.shots, lease_expires_in_s=expires_in)

    def end(self, end: ContinuousEnd) -> bool:
        """Name the run's end, unless one is named already.

        The caller then stops the run, normally by renewing the holder stop
        event; the driver ends with the end named here.

        Returns:
            Whether *end* is the run's end.
        """
        if self._end is not None or self.ended is not None:
            return False
        self._end = end
        return True

    async def wait(self) -> ContinuousEndedEvent:
        """Await the run's end and answer its ended event."""
        if self._driver is None:
            raise RuntimeError("the continuous run was never started")
        await asyncio.wait({self._driver})
        if self.ended is None:  # cancelled before its first step: it never ran
            self.end("shutdown")
            self._finish()
        assert self.ended is not None
        return self.ended

    async def cancel(self, end: ContinuousEnd = "shutdown") -> ContinuousEndedEvent:
        """End the run as *end* unless it already has an end, cancel the driver and await it."""
        self.end(end)
        if self._driver is None:
            raise RuntimeError("the continuous run was never started")
        self._driver.cancel()
        return await self.wait()

    # -- the driver ----------------------------------------------------------

    async def _drive(self) -> None:
        stop = self._stop
        assert stop is not None
        sessions = self._sessions
        try:
            while self._end is None:
                if stop.is_set():
                    self.end("stopped")
                    break
                started = self._clock()
                self.shots += 1
                try:
                    result = await sessions.shot(
                        self.owner, self.kind, self.client_id, self._store, self._build
                    )
                except ActionRefused as exc:
                    if exc.code != ACTION_STOPPED:
                        self._fail(exc, stop)
                        break
                    self.end("stopped")
                    break
                except Exception as exc:  # a stopped shot, or one that failed
                    self._fail(exc, stop)
                    break
                if _every_injection_failed(result):
                    self.end("failed")
                    break
                await self._pace(started, stop)
        except asyncio.CancelledError:
            self.end("shutdown")
            raise
        finally:
            self._finish()

    async def _pace(self, started: float, stop: asyncio.Event) -> None:
        """Wait out the rest of the minimum period from *started*, or until the run ends."""
        while self._end is None and not stop.is_set():
            remaining = started + self.min_period_s - self._clock()
            if remaining <= 0:
                return
            pause = asyncio.ensure_future(self._sleep(remaining))
            stopped = asyncio.ensure_future(stop.wait())
            try:
                await asyncio.wait({pause, stopped}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                pause.cancel()
                stopped.cancel()

    def _fail(self, exc: BaseException, stop: asyncio.Event) -> None:
        from osprey.services.trajectory.acquisition import ShotStopped

        if stop.is_set() or isinstance(exc, ShotStopped):
            self.end("stopped")
            return
        if self.end("failed"):
            self.error = exc

    def _finish(self) -> None:
        if self.ended is not None:
            return
        if self._lease is not None:
            self._lease.cancel()
        end = self._end or "failed"
        sessions = self._sessions
        if end == "lease_expired":
            # A shot that finished after the expiry may have installed a step.
            sessions.step.void()
        self.ended = ContinuousEndedEvent(end=end, shots=self.shots)
        sessions.continuous_ended(self)
        if self._on_end is not None:
            self._on_end(self.ended)

    # -- the lease -----------------------------------------------------------

    async def _watch_lease(self) -> None:
        while self._end is None:
            remaining = self._deadline - self._clock()
            if remaining <= 0:
                if self.end("lease_expired"):
                    self._sessions.expire_lease(self)
                return
            await self._sleep(min(remaining, self.heartbeat_s))


def _every_injection_failed(result: Any) -> bool:
    """Whether an iteration's stored shot included no injection at all."""
    shot = result.shot if isinstance(result, ChainResult) else result
    return isinstance(shot, ShotResult) and shot.shots_included == 0
