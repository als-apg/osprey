"""Continuous mode: the holder's shots, one after another, until something ends them.

A :class:`ContinuousRun` belongs to the service's
:class:`~osprey.services.trajectory.sessions.Sessions`, which starts it for the
holder and keeps at most one. Its driver task submits one shot action per
iteration through :meth:`Sessions.shot`, so the post-shot hook recomputes the
step after every shot, and each shot is its own turn of the action layer: the
run lock is taken for the shot and released before the next is submitted,
which lets any other queued action run in between. The run fires shots only;
it never applies a step.

The run ends, first cause wins, when:

- the holder stop event current at its start is set (``stopped``): a Stop, a
  take-over or a release set it, and an iteration queued after it is refused
  by the layer or stopped mid-shot;
- it submitted ``max_shots`` shot actions (``max_shots``);
- the step the latest shot left pending is all zero (``zero_step``), the
  ended event then carrying that step's reason and status;
- its lease lapsed (``lease_expired``): only a continuous run holds a lease,
  moved forward by each heartbeat, and a lease no heartbeat renewed within
  ``lease_s`` ends the run, voids the pending step and renews the holder stop
  event, so a shot in flight stops at its next checkpoint;
- a shot raised anything else while the stop event was clear (``failed``);
- the owner of the sessions ended it by name (:meth:`ContinuousRun.end`) or
  cancelled it (:meth:`ContinuousRun.cancel`).

Its end is published once as a :class:`~.state.ContinuousEndedEvent`, handed
to the ``on_end`` callback and kept on :attr:`ContinuousRun.ended`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from osprey.services.trajectory.actions import ACTION_STOPPED, Action, ActionRefused
from osprey.services.trajectory.state import (
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
    """One continuous run of the holder's shots.

    Attributes:
        owner: The holder's owner the shots are submitted for.
        kind: The holder's door.
        client_id: The holder's session token.
        max_shots: The most shot actions the run submits.
        heartbeat_s: Seconds between two heartbeats a client sends.
        lease_s: Seconds the lease lasts without a heartbeat.
        shots: Shot actions submitted so far.
        ended: The ended event, once the run ended.
        error: What a shot raised when the run ended ``failed``.
    """

    def __init__(
        self,
        sessions: Sessions,
        owner: Any,
        kind: str,
        client_id: str,
        store: TrajectoryStore,
        build: Callable[[PostShotHook], Action[ShotResult]],
        *,
        max_shots: int,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
        lease_s: float = DEFAULT_LEASE_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        on_end: Callable[[ContinuousEndedEvent], Any] | None = None,
    ) -> None:
        if max_shots < 1:
            raise ValueError(f"a continuous run takes at least one shot, not {max_shots}")
        if not 0 < heartbeat_s < lease_s:
            raise ValueError(
                f"the heartbeat interval ({heartbeat_s} s) must be positive and shorter "
                f"than the lease ({lease_s} s)"
            )
        self.owner = owner
        self.kind = kind
        self.client_id = client_id
        self.max_shots = max_shots
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
        """The run's shots so far, its limit, and the seconds its lease has left."""
        expires_in = max(0.0, self._deadline - self._clock()) if self.running else None
        return ContinuousStatus(
            shots=self.shots, max_shots=self.max_shots, lease_expires_in_s=expires_in
        )

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
                self.shots += 1
                try:
                    await sessions.shot(
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
                step = sessions.step.step
                if step is not None and step.all_zero:
                    self.end("zero_step")
                elif self.shots >= self.max_shots:
                    self.end("max_shots")
        except asyncio.CancelledError:
            self.end("shutdown")
            raise
        finally:
            self._finish()

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
        step = sessions.step.step if end == "zero_step" else None
        self.ended = ContinuousEndedEvent(
            end=end,
            shots=self.shots,
            step_reason=step.reason if step is not None else None,
            step_status=step.status if step is not None else None,
        )
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
