"""Sessions: one holder per service target, on the action layer.

A holder is an owner, the door it came through, a ``client_id`` bound to that
``(owner, kind)``, and the stop event the action layer keeps for it — read
through to the layer, so a holder has one stop source. Sessions live in memory
only.

- **Open** is a worker action (:meth:`ActionLayer.submit`): it is admitted,
  probed and run under the lock like any other. Its precondition re-checks in
  the worker that the target has no holder, so of two queued opens the second
  is 409 ``session_held``, naming the holder's door and owner. Open is the only
  way to acquire a holder.
- **Stop** acts on the control path after ``admit``, on either door, for any
  admitted owner, and carries no ``client_id``: it renews the holder stop
  event, so the action in flight ends at its next stop checkpoint, and records
  the stopper's owner on that action. The holder stays. Stop answers at once;
  the action ends within the stop budget — the module's longest pause, one
  connector call and one poll of the stop event.
- **Take-over** acts on the control path (:meth:`ActionLayer.control`) after
  ``admit``, in one in-memory step with nothing awaited: it sets the old
  holder's stop event, voids its ``client_id`` and installs the new holder
  with a clear stop event. It never probes and never takes the run lock, so it
  is answered while an action is in flight and on an unreachable target. With
  no holder it is 409 ``no_session``; while an apply or a put-back is writing
  it is 409 ``apply_in_progress``, and nothing changes.
- **Release** acts on the control path without ``admit`` and checks only that
  the caller's owner and ``client_id`` are the holder's, so a narrowed holder
  or the holder of an unreachable target can release. It renews the stop
  event too, so an action of the released holder in flight stops.
- **A call** is a worker action whose precondition checks the caller's
  ``client_id`` in the worker: a voided one is 409 ``session_taken_over``, an
  unknown one, or one bound to another owner or door, is 409
  ``not_session_holder``. A call may add its own precondition, which runs
  only after the holder check passed, so a non-holder never reaches it.

Each started service keeps one :class:`~osprey.services.trajectory.step.StepState`
on its sessions. A holder's shot computes the step from the averaged shot and
the current reference after the shot is stored, and a shot whose every shot
failed voids it; the sessions keep the id of the latest shot, so an apply can
tell a step of an older shot; selecting a reference is a holder call that recomputes it
against the step's shot. A step computed for a ``client_id`` that no longer
holds the target is never installed. A take-over voids the step in the same
in-memory step that swaps the holder, and a release in the one that drops it.

The sessions own the holder's one :class:`~.continuous.ContinuousRun`. It is
started on the control path for the holder (409 ``continuous_running`` while
one runs); only it holds a lease, which a heartbeat from the holder moves and
whose expiry ends the run, voids the step and renews the stop event; a
heartbeat moves the lease deadline and nothing else. A Stop ends the run as
``stopped``, a take-over as ``taken_over`` and a release as ``released``, each
named before the stop event is renewed. While it runs, an apply and a setup
are refused ``continuous_running`` (409).

The state view publishes the stop and action budgets
(:func:`~osprey.services.trajectory.constants.budgets`) and the action in
flight, by name, with whether a Stop reached it.

The door passes ``kind``; it is never read from a request body. A body that
carries any key is 400: ``owner_in_body`` for ``owner``, ``bad_request`` for
anything else. The state view names the holder's owner and door and never its
``client_id``.
"""

from __future__ import annotations

import asyncio
import functools
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from osprey.services.trajectory.actions import (
    SERVICE_STOPPING,
    Action,
    ActionLayer,
    ActionRefused,
    Precondition,
)
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.constants import MAX_PAUSE_CEILING_S, Budgets, budgets
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.state import (
    ContinuousEnd,
    ContinuousStatus,
    LastAction,
    RunningAction,
    ShotResult,
    Step,
)
from osprey.services.trajectory.step import StepState, model_of, step_for_shot
from osprey_connectors.control_system.call_timeout import DEFAULT_TIMEOUT_S

if TYPE_CHECKING:
    from osprey.services.trajectory.acquisition import PostShotHook
    from osprey.services.trajectory.continuous import ContinuousRun
    from osprey.services.trajectory.store import TrajectoryStore

__all__ = [
    "APPLY_IN_PROGRESS",
    "BAD_REQUEST",
    "CONTINUOUS_RUNNING",
    "NOT_SESSION_HOLDER",
    "NO_SESSION",
    "OWNER_IN_BODY",
    "SESSION_HELD",
    "SESSION_TAKEN_OVER",
    "Holder",
    "SessionGrant",
    "SessionRefused",
    "SessionRelease",
    "SessionStop",
    "Sessions",
    "check_body",
]

SESSION_HELD = "session_held"
NO_SESSION = "no_session"
NOT_SESSION_HOLDER = "not_session_holder"
SESSION_TAKEN_OVER = "session_taken_over"
OWNER_IN_BODY = "owner_in_body"
BAD_REQUEST = "bad_request"
CONTINUOUS_RUNNING = "continuous_running"
APPLY_IN_PROGRESS = "apply_in_progress"

#: The names of the actions a take-over waits out: they write settings one by
#: one, and a holder swapped mid-way would inherit a half-made change.
_APPLY_NAMES = frozenset({"apply", "put_back"})

#: The HTTP status each refusal raised here answers with.
_STATUS = {
    SESSION_HELD: 409,
    NO_SESSION: 409,
    NOT_SESSION_HOLDER: 409,
    SESSION_TAKEN_OVER: 409,
    OWNER_IN_BODY: 400,
    BAD_REQUEST: 400,
    CONTINUOUS_RUNNING: 409,
    APPLY_IN_PROGRESS: 409,
}

T = TypeVar("T")


class SessionRefused(Refused):
    """A session request refused; its code is one of this module's constants."""

    statuses = _STATUS


@dataclass(frozen=True)
class Holder:
    """The one holder of a service target.

    Attributes:
        owner: Who holds it, as admitted.
        kind: The door it was opened or taken over through.
        client_id: The token later calls carry, bound to ``(owner, kind)``.
        layer: The action layer whose stop event is this holder's.
    """

    owner: str
    kind: str
    client_id: str
    layer: Any = field(repr=False, compare=False)

    @property
    def stop(self) -> asyncio.Event:
        """The holder stop event: the action layer's current one, read through."""
        event: asyncio.Event = self.layer.stop_event
        return event


@dataclass(frozen=True)
class SessionGrant:
    """What an open or a take-over answers with."""

    client_id: str
    owner: str
    target: str

    def as_body(self) -> dict[str, str]:
        """The response body: ``client_id``, ``owner`` and ``target``."""
        return {"client_id": self.client_id, "owner": self.owner, "target": self.target}


@dataclass(frozen=True)
class SessionRelease:
    """What a release's audit record carries: ``owner``, ``target`` and ``client_id``."""

    owner: str
    target: str
    client_id: str

    def as_record(self) -> dict[str, str]:
        """The audit record of the release."""
        return {"owner": self.owner, "target": self.target, "client_id": self.client_id}


@dataclass(frozen=True)
class SessionStop:
    """What a Stop's audit record carries: the stopper and the action it reached.

    Attributes:
        owner: The stopper, as admitted.
        kind: The door the Stop came through.
        target: The service target.
        action: The name of the action in flight, ``None`` when none was.
    """

    owner: str
    kind: str
    target: str
    action: str | None

    def as_record(self) -> dict[str, str | None]:
        """The audit record of the Stop."""
        return {
            "owner": self.owner,
            "kind": self.kind,
            "target": self.target,
            "action": self.action,
        }


@dataclass
class _Running:
    """The action in flight: its name, the stop event it was handed, its stopper."""

    name: str | None
    stop: asyncio.Event
    stopper: str | None = None


def check_body(body: Mapping[str, Any] | None) -> None:
    """Refuse a session request body that carries any key.

    The owner comes from the door's identity and the door from its route, so a
    session request has nothing to say in its body.

    Raises:
        SessionRefused: ``owner_in_body`` when the body names an owner,
            ``bad_request`` for any other key or a body that is not a mapping.
    """
    if body is None:
        return
    if not isinstance(body, Mapping):
        raise SessionRefused(BAD_REQUEST, "a session request carries no body")
    if "owner" in body:
        raise SessionRefused(
            OWNER_IN_BODY,
            "the owner is the identity the request arrived with, never a body field",
        )
    if body:
        keys = ", ".join(sorted(repr(key) for key in body))
        raise SessionRefused(BAD_REQUEST, f"a session request carries no body fields; got {keys}")


class Sessions:
    """The sessions of one started service.

    Attributes:
        layer: The action layer the sessions act through.
        step: The pending step.
        latest_shot_id: The id of the latest shot a holder's shot action
            returned or handed its post-shot hook, ``None`` before the first.
        call_timeout_s: The service connector's call timeout, one term of the
            stop budget.
        shutdown_signal: The service-shutdown signal, set once by
            :meth:`shutdown`; it ends every state stream.
    """

    def __init__(self, layer: ActionLayer, *, call_timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.layer = layer
        self.call_timeout_s = call_timeout_s
        self.shutdown_signal = asyncio.Event()
        self._shutdown: asyncio.Future[None] | None = None
        self.step = StepState()
        self.latest_shot_id: str | None = None
        self._holder: Holder | None = None
        self._voided: set[str] = set()
        self._continuous: ContinuousRun | None = None
        self._running: _Running | None = None
        self._ended: _Running | None = None

    @property
    def target(self) -> str:
        """The service target the one holder holds."""
        return self.layer.startup.target

    @property
    def holder(self) -> Holder | None:
        """The current holder, if any."""
        return self._holder

    # -- open ----------------------------------------------------------------

    async def open(
        self, owner: Any, kind: str, body: Mapping[str, Any] | None = None
    ) -> SessionGrant:
        """Open a session for *owner* through door *kind*, as a worker action.

        Raises:
            SessionRefused: The body carries a key, or the target has a holder.
            AdmitRefused: ``admit`` refused the owner.
            ActionRefused: The worker refused the open.
        """
        check_body(body)

        async def install(admission: Admission, stop: asyncio.Event) -> SessionGrant:
            return self._install(admission, stop)

        return await self.layer.submit(owner, kind, install, precondition=self._unheld)

    def _unheld(self, _admission: Admission) -> None:
        holder = self._holder
        if holder is not None:
            raise SessionRefused(
                SESSION_HELD,
                f"target {self.target!r} is held by {holder.owner!r} through the "
                f"{holder.kind!r} door; take it over to act here",
            )

    # -- take-over -----------------------------------------------------------

    def take_over(
        self, owner: Any, kind: str, body: Mapping[str, Any] | None = None
    ) -> SessionGrant:
        """Take the target over for *owner* through door *kind*, on the control path.

        The old holder's step and continuous lease are void: a continuous run
        ends as ``taken_over``.

        Raises:
            SessionRefused: The body carries a key, the target has no holder,
                or an apply is writing (``apply_in_progress``).
            AdmitRefused: ``admit`` refused the owner.
        """
        check_body(body)
        return self.layer.control(owner, kind, self._swap)

    def _swap(self, admission: Admission | None) -> SessionGrant:
        if admission is None:  # the control path admits every take-over
            raise RuntimeError("a take-over runs only after admit")
        old = self._holder
        if old is None:
            raise SessionRefused(
                NO_SESSION,
                f"target {self.target!r} has no holder to take over; open a session instead",
            )
        running = self._running
        if running is not None and running.name in _APPLY_NAMES:
            raise SessionRefused(
                APPLY_IN_PROGRESS,
                f"an {running.name} by {old.owner!r} is writing; Stop it or wait for it "
                "to end, then take over",
            )
        self._end_continuous("taken_over")
        stop = self.layer.renew_stop_event()
        self._voided.add(old.client_id)
        self.step.void()
        return self._install(admission, stop)

    def _install(self, admission: Admission, _stop: asyncio.Event) -> SessionGrant:
        client_id = secrets.token_urlsafe(24)
        self._holder = Holder(
            owner=admission.owner, kind=admission.kind, client_id=client_id, layer=self.layer
        )
        return SessionGrant(client_id=client_id, owner=admission.owner, target=admission.target)

    # -- release -------------------------------------------------------------

    def release(
        self,
        owner: Any,
        kind: str,
        client_id: str | None,
        body: Mapping[str, Any] | None = None,
    ) -> SessionRelease:
        """Release the session, on the control path and without ``admit``.

        The release voids the step, ends a continuous run as ``released`` and
        renews the stop event, so an action of the holder in flight stops.

        Raises:
            SessionRefused: The body carries a key, or *owner* and *client_id*
                are not the holder's.
        """
        check_body(body)

        def drop(_admission: Admission | None) -> SessionRelease:
            holder = self._holder
            if holder is None or holder.owner != owner or holder.client_id != client_id:
                raise _non_holder(client_id, self._voided)
            self._end_continuous("released")
            self.layer.renew_stop_event()
            self.step.void()
            self._holder = None
            return SessionRelease(owner=holder.owner, target=self.target, client_id=client_id)

        return self.layer.control(owner, kind, drop, admit=False)

    # -- stop ----------------------------------------------------------------

    def stop(self, owner: Any, kind: str, body: Mapping[str, Any] | None = None) -> SessionStop:
        """Stop the action in flight for *owner* through door *kind*, on the control path.

        Any admitted owner may stop, on either door, without a ``client_id``;
        the holder stays. The stop event is renewed, a continuous run ends as
        ``stopped``, and the action in flight records *owner* as its stopper.
        Nothing is awaited: the action ends on its own at its next stop
        checkpoint, within the stop budget.

        Raises:
            SessionRefused: The body carries a key.
            AdmitRefused: ``admit`` refused the owner.
        """
        check_body(body)

        def halt(admission: Admission | None) -> SessionStop:
            if admission is None:  # the control path admits every Stop
                raise RuntimeError("a Stop runs only after admit")
            running = self._running
            if running is not None and running.stopper is None and not running.stop.is_set():
                running.stopper = admission.owner
            self._end_continuous("stopped")
            self.layer.renew_stop_event()
            return SessionStop(
                owner=admission.owner,
                kind=admission.kind,
                target=self.target,
                action=running.name if running is not None else None,
            )

        return self.layer.control(owner, kind, halt)

    @property
    def stopper(self) -> str | None:
        """The owner whose Stop reached the latest action, ``None`` when none did.

        The latest action is the one in flight or, when none is, the one that
        ended last; the stopper stays readable after the stopped action has
        returned, until the next action starts.
        """
        latest = self._running if self._running is not None else self._ended
        return latest.stopper if latest is not None else None

    def running_action(self) -> RunningAction | None:
        """The action in flight by name, and whether its stop event is set."""
        running = self._running
        if running is None or running.name is None:
            return None
        return RunningAction(name=running.name, stopping=running.stop.is_set())

    def _end_continuous(self, end: ContinuousEnd) -> None:
        run = self._continuous
        if run is not None and run.running:
            run.end(end)

    # -- calls ---------------------------------------------------------------

    def precondition(self, client_id: str | None) -> Precondition:
        """The worker check a holder's call passes, for the caller's *client_id*."""

        def check(admission: Admission) -> None:
            holder = self._holder
            if (
                holder is None
                or holder.client_id != client_id
                or holder.owner != admission.owner
                or holder.kind != admission.kind
            ):
                raise _non_holder(client_id, self._voided)

        return check

    async def call(
        self,
        owner: Any,
        kind: str,
        client_id: str | None,
        action: Action[T],
        *,
        precondition: Precondition | None = None,
        name: str | None = None,
    ) -> T:
        """Run *action* for the holder as a worker action.

        Args:
            owner: The admitted caller.
            kind: The door the call came through.
            client_id: The caller's session token.
            action: The worker action.
            precondition: The call's own worker check, run after the holder
                check passed and before the probe, the limits and the lock.
            name: The action's name while it runs (``shot``, ``setup``,
                ``apply``, ``put_back``, ``reference``); a take-over is
                refused while an ``apply`` or a ``put_back`` runs.

        Raises:
            SessionRefused: ``session_taken_over`` or ``not_session_holder``,
                checked in the worker.
            AdmitRefused: ``admit`` refused the owner.
            ActionRefused: The worker refused the action.
            Exception: Whatever *precondition* raised.
        """
        holder = self.precondition(client_id)
        tracked = self._tracked(name, action)
        if precondition is None:
            return await self.layer.submit(owner, kind, tracked, precondition=holder)
        extra = precondition

        def check(admission: Admission) -> None:
            holder(admission)
            extra(admission)

        return await self.layer.submit(owner, kind, tracked, precondition=check)

    def _tracked(self, name: str | None, action: Action[T]) -> Action[T]:
        """*action*, known to the sessions as the action in flight while it runs."""

        @functools.wraps(action)
        async def run(admission: Admission, stop: asyncio.Event) -> T:
            running = _Running(name=name, stop=stop)
            self._running = running
            self._ended = None
            try:
                return await action(admission, stop)
            finally:
                if self._running is running:
                    self._running = None
                    self._ended = running

        return run

    # -- the step ------------------------------------------------------------

    def post_shot(self, store: TrajectoryStore, client_id: str) -> PostShotHook:
        """The post-shot hook of a shot the holder *client_id* submits.

        It voids the pending step, computes the step for the shot, and installs
        it only while *client_id* still holds the target.
        """

        async def hook(shot: ShotResult) -> Step | None:
            self.latest_shot_id = shot.shot_id
            return await self._recompute(store, shot, client_id)

        return hook

    async def _recompute(
        self, store: TrajectoryStore, shot: ShotResult, client_id: str
    ) -> Step | None:
        self.step.void()
        step = await step_for_shot(self.layer.startup, store, shot, client_id=client_id)
        holder = self._holder
        if holder is None or holder.client_id != client_id:
            return None
        self.step.replace(step, shot)
        return step

    async def shot(
        self,
        owner: Any,
        kind: str,
        client_id: str | None,
        store: TrajectoryStore,
        build: Callable[[PostShotHook], Action[ShotResult]],
    ) -> ShotResult:
        """Run the holder's shot action, with the post-shot hook bound.

        *build* makes the shot action from the hook it must call. A shot action
        whose every shot failed voids the pending step.

        Raises:
            As :meth:`call`; and :class:`~osprey.services.trajectory.step.StepRefused`
            when the step cannot be computed for the stored shot.
        """
        shot_action = build(self.post_shot(store, client_id or ""))

        async def action(admission: Admission, stop: asyncio.Event) -> ShotResult:
            result = await shot_action(admission, stop)
            self.latest_shot_id = result.shot_id
            if result.shots_included == 0:
                self.step.void()
            return result

        return await self.call(owner, kind, client_id, action, name="shot")

    async def select_reference(
        self,
        owner: Any,
        kind: str,
        client_id: str | None,
        store: TrajectoryStore,
        name: str | None,
    ) -> Step | None:
        """Make *name* the current reference and recompute the step, as a holder call.

        ``None`` clears the current reference, so the step solves to zero. With
        no pending step there is no shot to recompute from, and none is made.

        Raises:
            As :meth:`call`; ``ReferenceNotFound`` or ``InvalidName`` from the
            store; :class:`~osprey.services.trajectory.step.StepRefused` when
            the step cannot be computed.
        """

        async def action(_admission: Admission, _stop: asyncio.Event) -> Step | None:
            model = model_of(self.layer.startup)
            await store.set_current(model, name)
            shot = self.step.shot
            if shot is None:
                return None
            return await self._recompute(store, shot, client_id or "")

        return await self.call(owner, kind, client_id, action, name="reference")

    async def setup(
        self, owner: Any, kind: str, client_id: str | None, **kwargs: Any
    ) -> LastAction:
        """Run the setup action for the holder, refused while a continuous run runs.

        *kwargs* are :func:`~osprey.services.trajectory.acquisition.setup_action`'s.
        The setup voids the pending step as its first statement.

        Raises:
            As :meth:`call`; and
            :class:`~osprey.services.trajectory.acquisition.AcquisitionRefused`
            as ``setup_action`` refuses, ``continuous_running`` included.
        """
        from osprey.services.trajectory.acquisition import AcquisitionRefused, setup_action

        def running() -> bool:
            return self.continuous_running

        setup = setup_action(self.layer.startup, continuous_running=running, **kwargs)

        def idle(_admission: Admission) -> None:
            if running():
                raise AcquisitionRefused(
                    CONTINUOUS_RUNNING,
                    "a continuous run is in progress; stop it before a setup",
                )

        async def action(admission: Admission, stop: asyncio.Event) -> LastAction:
            self.step.void()
            return await setup(admission, stop)

        return await self.call(owner, kind, client_id, action, precondition=idle, name="setup")

    # -- continuous ----------------------------------------------------------

    @property
    def continuous(self) -> ContinuousRun | None:
        """The continuous run in progress, if any."""
        return self._continuous

    @property
    def continuous_running(self) -> bool:
        """Whether a continuous run is in progress."""
        run = self._continuous
        return run is not None and run.running

    def start_continuous(
        self,
        owner: Any,
        kind: str,
        client_id: str | None,
        store: TrajectoryStore,
        build: Callable[[PostShotHook], Action[ShotResult]],
        *,
        max_shots: int | None = None,
        **kwargs: Any,
    ) -> ContinuousRun:
        """Start a continuous run of the holder's shots, on the control path.

        Args:
            owner: The admitted caller.
            kind: The door the call came through.
            client_id: The caller's session token.
            store: The store each shot is saved to.
            build: Makes one shot action from the post-shot hook, per shot.
            max_shots: The most shot actions the run submits; the service's
                ``continuous_max_shots`` when not given.
            **kwargs: :class:`~.continuous.ContinuousRun`'s keywords.

        Raises:
            SessionRefused: ``continuous_running`` while one runs;
                ``session_taken_over`` or ``not_session_holder`` for a caller
                that does not hold the target.
            ActionRefused: ``service_stopping`` once the service is shutting down.
            AdmitRefused: ``admit`` refused the owner.
        """
        from osprey.services.trajectory.continuous import ContinuousRun

        holder = self.precondition(client_id)

        def begin(admission: Admission | None) -> ContinuousRun:
            if admission is None:  # the control path admits every start
                raise RuntimeError("a continuous start runs only after admit")
            holder(admission)
            if self.continuous_running:
                raise SessionRefused(
                    CONTINUOUS_RUNNING, "a continuous run is in progress; stop it first"
                )
            if self._shutdown is not None:
                raise ActionRefused(
                    SERVICE_STOPPING, "the service is shutting down; no continuous run starts"
                )
            limit = max_shots
            if limit is None:
                limit = self.layer.startup.settings.continuous_max_shots
            run = ContinuousRun(
                self,
                admission.owner,
                admission.kind,
                client_id or "",
                store,
                build,
                max_shots=limit,
                **kwargs,
            )
            self._continuous = run
            run.start()
            return run

        return self.layer.control(owner, kind, begin)

    def heartbeat(self, owner: Any, kind: str, client_id: str | None) -> ContinuousStatus | None:
        """Move the continuous run's lease, on the control path; nothing else.

        Returns:
            The run's status, ``None`` when no continuous run is in progress.

        Raises:
            SessionRefused: ``session_taken_over`` or ``not_session_holder``
                when the caller is not the run's holder.
        """

        def beat(_admission: Admission | None) -> ContinuousStatus | None:
            holder = self._holder
            if (
                holder is None
                or holder.owner != owner
                or holder.kind != kind
                or holder.client_id != client_id
            ):
                raise _non_holder(client_id, self._voided)
            run = self._continuous
            if run is None or not run.running or run.client_id != client_id:
                return None
            return run.heartbeat()

        return self.layer.control(owner, kind, beat, admit=False)

    def expire_lease(self, run: ContinuousRun) -> None:
        """A continuous run's lease lapsed: void the step and renew the stop event."""

        def expire(_admission: Admission | None) -> None:
            self.step.void()
            self.layer.renew_stop_event()

        self.layer.control(run.owner, run.kind, expire, admit=False)

    def continuous_ended(self, run: ContinuousRun) -> None:
        """Forget *run*, which ended."""
        if self._continuous is run:
            self._continuous = None

    # -- shutdown ------------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop the service's work, in order; a second call awaits the first.

        The layer refuses every new action (``service_stopping``), the holder
        stop event is set without a fresh one installed, so the action in
        flight stops at its next checkpoint, the shutdown signal ends every
        state stream, a continuous run ends as ``shutdown`` with its driver
        cancelled and awaited, and :meth:`ActionLayer.close` fails every queued
        action with ``service_stopping`` and awaits the one in flight.
        """
        if self._shutdown is None:
            self._shutdown = asyncio.ensure_future(self._shut_down())
        await asyncio.shield(self._shutdown)

    async def _shut_down(self) -> None:
        self.layer.refuse_new()
        run = self._continuous
        self._end_continuous("shutdown")
        self.layer.stop_event.set()
        self.shutdown_signal.set()
        if run is not None:
            await run.cancel("shutdown")
        await self.layer.close()

    # -- state ---------------------------------------------------------------

    def budgets(self) -> Budgets:
        """The stop and action budgets of this service.

        The module's ``max_pause_s`` is the longest pause; with no module
        loaded it is :data:`~.constants.MAX_PAUSE_CEILING_S`, the longest any
        loadable module may declare.
        """
        startup = self.layer.startup
        sequence = startup.sequence
        max_pause_s = sequence.metadata.max_pause_s if sequence is not None else MAX_PAUSE_CEILING_S
        return budgets(
            max_pause_s=max_pause_s,
            call_timeout_s=self.call_timeout_s,
            action_timeout_s=startup.settings.action_timeout_s,
        )

    def state(self) -> dict[str, Any]:
        """The holder, the action in flight and the budgets, for a state view.

        The holder is named by owner and door, never by ``client_id``.
        """
        holder = self._holder
        action = self.running_action()
        limits = self.budgets()
        return {
            "target": self.target,
            "owner": holder.owner if holder is not None else None,
            "kind": holder.kind if holder is not None else None,
            "action": action.model_dump() if action is not None else None,
            "action_timeout_s": self.layer.startup.settings.action_timeout_s,
            "action_budget_s": limits.action_budget_s,
            "stop_budget_s": limits.stop_budget_s,
        }


def _non_holder(client_id: str | None, voided: set[str]) -> SessionRefused:
    if client_id is not None and client_id in voided:
        return SessionRefused(
            SESSION_TAKEN_OVER,
            "this session was taken over; open or take over a session to act again",
        )
    return SessionRefused(NOT_SESSION_HOLDER, "the caller does not hold this target's session")
