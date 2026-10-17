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
   connector. A fail-safe one is 503 ``limits_unavailable``. When none is
   built, the limits posture of the service target decides: a readable config
   whose posture is complete and states ``enabled: false`` lets the action run
   with no validator and an :class:`~osprey.runtime.guarded_run.ActionContext`
   marked ``limits_off``; an unstated ``enabled`` or a config that cannot be
   read is 503 ``limits_unavailable``, because a command fires unchecked only
   where the deployment says so;
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
so a Stop, a take-over or a release is answered while an action is in flight,
and an unreachable target never blocks one. The layer keeps the current
holder's stop event; a Stop, a take-over, a release and a continuous lease's
expiry renew it on the control path, which sets the old one — the in-flight
action was handed it — and installs a clear one for what is submitted next.
Nothing on the control path awaits the action it stops.
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
    "SERVICE_STOPPING",
    "Action",
    "ActionLayer",
    "ActionRefused",
    "Precondition",
]

RUN_BUSY = "run_busy"
GUARDED_RUN_UNAVAILABLE = "guarded_run_unavailable"
LIMITS_UNAVAILABLE = "limits_unavailable"
ACTION_STOPPED = "action_stopped"
SERVICE_STOPPING = "service_stopping"

#: The HTTP status each refusal raised here answers with.
_STATUS = {
    RUN_BUSY: 409,
    ACTION_STOPPED: 409,
    GUARDED_RUN_UNAVAILABLE: 503,
    LIMITS_UNAVAILABLE: 503,
    SERVICE_STOPPING: 503,
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

    @property
    def queued(self) -> int:
        """How many submitted actions wait behind the one the worker runs."""
        return self._queue.qsize()

    # -- the stop event ------------------------------------------------------

    @property
    def stop_event(self) -> asyncio.Event:
        """The current holder's stop event, captured by each action as it is submitted."""
        return self._stop

    def renew_stop_event(self) -> asyncio.Event:
        """Set the current holder's stop event and install a clear one.

        Called on the control path by a Stop, a take-over, a release and a
        continuous lease's expiry; an action in flight keeps the event it was
        handed and sees it set, and one submitted before the renewal that has
        not started is refused.

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

    @property
    def closed(self) -> bool:
        """Whether the layer refuses new actions (``service_stopping``)."""
        return self._closed

    def refuse_new(self) -> None:
        """Refuse every later :meth:`submit` with ``service_stopping``; queued actions stay."""
        self._closed = True

    async def close(self) -> None:
        """Refuse new actions, fail the queued ones, and end the worker.

        Every action still queued fails with ``service_stopping``; the one in
        flight, if any, runs to its end and its caller gets its result. Closing
        a closed layer again is a no-op beyond awaiting the worker's end.
        """
        self._closed = True
        _fail_pending(self._queue, _stopping)
        if self._worker is None or self._worker.done():
            return
        self._queue.put_nowait(_CLOSE)
        await asyncio.shield(self._worker)

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
            ActionRefused: The target, the limits or the lock refused, the
                action's stop event was set before it started, or the layer
                is closed or closed while the action was queued
                (``service_stopping``).
            RuntimeError: The worker ended before the action ran.
            Exception: Whatever the precondition or the action raised.
        """
        if self._closed:
            raise _stopping()
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
        failsafe = getattr(validator, "failsafe_reason", None)
        if failsafe is not None:
            raise ActionRefused(
                LIMITS_UNAVAILABLE,
                f"limits checking for target {target!r} blocks every write ({failsafe}); "
                "a command fires only under an enforcing limits validator or with "
                "limits checking stated off",
            )
        if validator is None:
            _refuse_unless_limits_off(target)
        self.startup.connector._limits_validator = validator
        context = ActionContext(
            target=target,
            owner=admission.owner,
            approved_target=target,
            limits_off=validator is None,
        )
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


def _refuse_unless_limits_off(target: str) -> None:
    """Refuse unless the config states limits checking off for *target*.

    Only a readable config whose posture for *target* is complete and says
    ``enabled: false`` is off; silence and an unreadable config are not
    permission to fire a command unchecked.
    """
    from osprey_connectors.config import get_config_value
    from osprey_connectors.types import target_limits_posture

    try:
        section = get_config_value("control_system", {})
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        raise ActionRefused(
            LIMITS_UNAVAILABLE,
            f"the deployment config cannot be read ({exc.__class__.__name__}), so no limits "
            f"posture holds for target {target!r}; a command fires only under an enforcing "
            "limits validator or with limits checking stated off",
        ) from exc
    posture = target_limits_posture(section, target)
    if not posture.incomplete and posture.enabled is False:
        return
    raise ActionRefused(
        LIMITS_UNAVAILABLE,
        f"limits checking for target {target!r} is neither on nor stated off "
        f"({posture.key('enabled')} is unstated); a command fires only under an "
        "enforcing limits validator or with limits checking stated off",
    )


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


def _stopping() -> ActionRefused:
    return ActionRefused(SERVICE_STOPPING, "the service is shutting down; this action did not run")


def _worker_ended() -> RuntimeError:
    return RuntimeError("the action worker ended before running this action")


def _fail_pending(
    queue: asyncio.Queue[Any], error: Callable[[], BaseException] = _worker_ended
) -> None:
    """Fail every job still queued with a fresh *error*."""
    while not queue.empty():
        job = queue.get_nowait()
        if job is not _CLOSE and not job.future.done():
            job.future.set_exception(error())
