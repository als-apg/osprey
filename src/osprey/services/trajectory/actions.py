"""The action layer: one worker task runs every writing action, in order.

An action submitted through :meth:`ActionLayer.submit` waits its turn in one
queue and, when the worker reaches it, passes these steps in this order; the
first that refuses decides and nothing after it runs:

1. :meth:`~osprey.services.trajectory.admit.AdmitGate.admit` for the owner and
   the door, read as the record and the ceiling are *now* — an action queued
   behind a narrowing is refused, not run on the posture it was submitted on;
2. the caller's precondition (sessions pass their client and holder checks);
3. the reachability probe (503 ``target_unreachable`` /
   ``trajectory_unavailable``);
4. a limits validator rebuilt for the service target and assigned onto the
   connector: none, or a fail-safe one, is 503 ``limits_unavailable`` whatever
   the limits posture says, because a command fires without confirmation only
   when a limits record says so;
5. the owner bound with :func:`~osprey_connectors.posture_store.bind_owner` and
   the target's run lock taken with an
   :class:`~osprey.runtime.guarded_run.ActionContext` naming that owner: a lock
   another run holds is 409 ``run_busy``, a lock directory that cannot be used
   is 503 ``guarded_run_unavailable``, a read-only process is 403
   ``writes_disabled``.

An action belongs to the holder stop event current when it was submitted: if
that event is set by the time the action would start — a take-over or a Stop
landed while it was queued or between its checks — it is refused 409
``action_stopped`` and never runs. Otherwise the action is handed that event.

The action is then awaited directly in the worker task, under both, so the
connector's per-write check reads the owner the lock names. It never hops to
an executor: the owner lives in the worker task's context and would not follow.

The **control path**, :meth:`ActionLayer.control`, acts at admission instead:
outside the queue, never taking the run lock, never probing, never writing —
so a take-over or a release is answered while an action is in flight, and an
unreachable target never blocks one. The layer keeps the current holder's
stop event; renewing it on the control path sets the old one, which the
in-flight action was handed, and installs a clear one for the next holder.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from osprey.runtime.guarded_run import (
    ActionContext,
    GuardedRunDirError,
    OspreyRunBusy,
    OspreyWriteRefused,
    lock,
)
from osprey.services.trajectory import reach
from osprey.services.trajectory.admit import WRITES_DISABLED, Admission, AdmitGate
from osprey.services.trajectory.reach import TARGET_UNREACHABLE, TRAJECTORY_UNAVAILABLE
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.startup import TrajectoryStartup
from osprey_connectors.control_system.limits_validator import LimitsValidator
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG, bind_owner

__all__ = [
    "ACTION_STOPPED",
    "GUARDED_RUN_UNAVAILABLE",
    "LIMITS_UNAVAILABLE",
    "RUN_BUSY",
    "Action",
    "ActionLayer",
    "ActionRefused",
    "Precondition",
]

RUN_BUSY = "run_busy"
GUARDED_RUN_UNAVAILABLE = "guarded_run_unavailable"
LIMITS_UNAVAILABLE = "limits_unavailable"
ACTION_STOPPED = "action_stopped"

#: The HTTP status each refusal raised here answers with.
_STATUS = {
    RUN_BUSY: 409,
    ACTION_STOPPED: 409,
    GUARDED_RUN_UNAVAILABLE: 503,
    LIMITS_UNAVAILABLE: 503,
    TARGET_UNREACHABLE: 503,
    TRAJECTORY_UNAVAILABLE: 503,
    WRITES_DISABLED: 403,
}

T = TypeVar("T")

#: A worker action: handed its admission and the stop event of the holder it
#: was submitted under, awaited in the worker task.
Action = Callable[[Admission, asyncio.Event], Awaitable[T]]
#: A caller's check, run in the worker after ``admit``; it refuses by raising.
Precondition = Callable[[Admission], None]


class ActionRefused(Refused):
    """An action the worker refused before it ran."""

    statuses = _STATUS


@dataclass
class _Job:
    owner: Any
    kind: str
    action: Action[Any]
    precondition: Precondition | None
    stop: asyncio.Event
    future: asyncio.Future[Any]


_CLOSE = object()


class ActionLayer:
    """The one worker and the control path of one started service.

    Attributes:
        startup: The started service: target, connector and command set.
        gate: The admission gate for the service target.
    """

    def __init__(self, startup: TrajectoryStartup, gate: AdmitGate) -> None:
        self.startup = startup
        self.gate = gate
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._stop = asyncio.Event()

    # -- the stop event ------------------------------------------------------

    @property
    def stop_event(self) -> asyncio.Event:
        """The current holder's stop event, captured by each action as it is submitted."""
        return self._stop

    def renew_stop_event(self) -> asyncio.Event:
        """Set the current holder's stop event and install a clear one.

        Called on the control path when the holder changes or is told to stop;
        an action in flight keeps the event it was handed and sees it set, and
        one submitted before the renewal that has not started is refused.

        Returns:
            The new holder's stop event.
        """
        self._stop.set()
        self._stop = asyncio.Event()
        return self._stop

    # -- the control path ----------------------------------------------------

    def control(
        self,
        owner: Any,
        kind: str,
        act: Callable[[Admission | None], T],
        *,
        admit: bool = True,
    ) -> T:
        """Act now, at admission, outside the queue.

        Takes no run lock, probes nothing and writes nothing; *act* must not
        either. With *admit* the owner passes :meth:`AdmitGate.admit` first and
        *act* is handed the admission. Without it *act* is handed ``None``: a
        release checks only that the caller is the holder, so a narrowed holder
        or an unreachable target never blocks one.

        Returns:
            What *act* returned.

        Raises:
            AdmitRefused: ``admit`` refused the owner; *act* was not called.
        """
        return act(self.gate.admit(owner, kind) if admit else None)

    # -- the worker ----------------------------------------------------------

    @property
    def worker_task(self) -> asyncio.Task[None] | None:
        """The worker task, once started."""
        return self._worker

    def start(self) -> None:
        """Start the worker in the running loop unless one is running.

        A worker that ended (its task was cancelled) is replaced.
        """
        if self._closed:
            raise RuntimeError("the action layer is closed")
        if self._worker is None or self._worker.done():
            self._worker = asyncio.get_running_loop().create_task(
                self._run(), name="trajectory-action-worker"
            )

    async def close(self) -> None:
        """Refuse new actions, run those already queued, and end the worker."""
        self._closed = True
        if self._worker is None or self._worker.done():
            _fail_pending(self._queue)
            return
        self._queue.put_nowait(_CLOSE)
        await self._worker

    async def submit(
        self,
        owner: Any,
        kind: str,
        action: Action[T],
        *,
        precondition: Precondition | None = None,
    ) -> T:
        """Queue *action* for *owner* through door *kind* and await its result.

        A caller that gives up while its action is still queued cancels it; one
        that gives up after it started leaves it to finish. The action is bound
        to the holder stop event current now (:attr:`stop_event`).

        Returns:
            What the action returned.

        Raises:
            AdmitRefused: ``admit`` refused the owner.
            ActionRefused: The target, the limits or the lock refused, or the
                action's stop event was set before it started.
            RuntimeError: The layer is closed, or the worker ended before
                the action ran.
            Exception: Whatever the precondition or the action raised.
        """
        if self._closed:
            raise RuntimeError("the action layer is closed")
        self.start()
        future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
        self._queue.put_nowait(_Job(owner, kind, action, precondition, self._stop, future))
        return await future

    async def _run(self) -> None:
        worker = asyncio.current_task()
        try:
            while True:
                job = await self._queue.get()
                if job is _CLOSE:
                    return
                if job.future.done():  # its caller gave up while it was queued
                    continue
                try:
                    result = await self._turn(job)
                except asyncio.CancelledError:
                    if not job.future.done():
                        job.future.cancel()
                    # Only the worker's own cancellation ends it; a
                    # cancellation the action raised belongs to its caller.
                    if worker is not None and worker.cancelling():
                        raise
                except BaseException as exc:  # every failure belongs to the caller
                    if not job.future.done():
                        job.future.set_exception(exc)
                else:
                    if not job.future.done():
                        job.future.set_result(result)
        finally:
            _fail_pending(self._queue)

    async def _turn(self, job: _Job) -> Any:
        _refuse_if_stopped(job)
        admission = self.gate.admit(job.owner, job.kind)
        if job.precondition is not None:
            job.precondition(admission)
        code = await reach.probe(self.startup)
        if code is not None:
            raise ActionRefused(code, _unreachable_detail(code, self.startup.target))
        target = self.startup.target
        validator = LimitsValidator.from_config(target=target)
        if validator is None or getattr(validator, "failsafe_reason", None) is not None:
            why = "limits checking is off" if validator is None else validator.failsafe_reason
            raise ActionRefused(
                LIMITS_UNAVAILABLE,
                f"no limits record can be read for target {target!r} ({why}); "
                "a command fires only when a limits record allows it",
            )
        self.startup.connector._limits_validator = validator
        context = ActionContext(target=target, owner=admission.owner, approved_target=target)
        with bind_owner({RESERVED_OWNER_KWARG: admission.owner}), contextlib.ExitStack() as held:
            try:
                held.enter_context(lock(target, context=context))
            except OspreyRunBusy as exc:
                raise ActionRefused(RUN_BUSY, str(exc)) from exc
            except OspreyWriteRefused as exc:
                raise ActionRefused(WRITES_DISABLED, exc.reason) from exc
            except (GuardedRunDirError, OSError) as exc:
                raise ActionRefused(
                    GUARDED_RUN_UNAVAILABLE,
                    f"the run lock of target {target!r} cannot be taken: {exc}",
                ) from exc
            # Nothing is awaited between this check and the action's start.
            _refuse_if_stopped(job)
            return await job.action(admission, job.stop)


def _unreachable_detail(code: str, target: str) -> str:
    if code == TRAJECTORY_UNAVAILABLE:
        return "the build gives this service no trajectory model, so there is nothing to command"
    return f"target {target!r} did not answer a read of its injection flag"


def _refuse_if_stopped(job: _Job) -> None:
    if job.stop.is_set():
        raise ActionRefused(
            ACTION_STOPPED,
            "the holder this action was submitted under was stopped or taken over "
            "before it started, so it did not run",
        )


def _fail_pending(queue: asyncio.Queue[Any]) -> None:
    while not queue.empty():
        job = queue.get_nowait()
        if job is not _CLOSE and not job.future.done():
            job.future.set_exception(
                RuntimeError("the action worker ended before running this action")
            )
