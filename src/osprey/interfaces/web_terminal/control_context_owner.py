"""The web terminal's one way to change the control-context record.

The record at ``<agent-data root>/control_target/control_context.json`` has a
single writer — its owner — and when a web terminal is running, that owner is
the web terminal. Three parts of the server need to change the record: the
posture route, the target route, and the owner task that claims the record and
consumes switch requests. All three go through :meth:`ControlContextOwner.mutate_record`,
and this module exists so that there is exactly one implementation of what
"changing the record" means.

**Why a primitive at all.** Every change is a read-modify-write: what gets
written depends on what is already there. Two of those interleaving is a lost
update — the second writer computes its candidate from the record as it was
before the first one wrote, and ``os.replace`` silently discards the first
change. The posture store this record replaces avoided that by doing load,
write and memory update synchronously on the event loop with no ``await``
between them: the loop itself was the lock. That worked, and it cost the loop a
disk round-trip on every toggle — acceptable for a few hundred bytes, not
acceptable for this record, whose owner also stats a directory of per-server
reports and probes liveness on the same path.

So the two properties are separated. An :class:`asyncio.Lock` provides the
serialisation the loop used to provide, and the read, the caller's change and
the write happen together inside **one** ``asyncio.to_thread`` job so none of
it runs on the loop. Both halves are load-bearing: a lock around the write
alone would still lose updates, because the losing read has already happened by
then; a thread hop without the lock would lose them for the same reason.

**Who may write.** The record names its owner, and a process that is no longer
the owner must not write it — a second web terminal may have started, or this
one's claim may have been taken over while it was idle. The check is made
inside the job, against the record just read, rather than against anything held
in memory: that is the point of re-reading. A caller that is *establishing*
ownership rather than exercising it (the claim) opts the check out with
``verify_owner=False``, and every write through this primitive stamps
``owner`` back to this process either way, so a caller cannot give ownership
away by handing back a record it read before the check.

The ownership *rules* — when a terminal may claim, when it must follow — are
not in the primitive. They belong to the owner task at the bottom of this
module, which is the only place that knows whether the current owner is alive.
The primitive knows only "am I the owner the record names", which is the
question a write has to answer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

from osprey_connectors.control_context import (
    OWNER_CONTROLS_SERVER,
    OWNER_WEB_TERMINAL,
    SWITCH_APPLIED,
    SWITCH_REFUSED,
    ControlContext,
    Owner,
    applied_detail,
    converged,
    file_signature,
    live_owner,
    live_reports,
    owned_here,
    read_record,
    record_path,
    report_paths,
    terminus,
    unchanged_detail,
    write_record,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CONTROL_CONTEXT_FRAME",
    "ContextOwnedElsewhere",
    "ContextOwnerError",
    "ContextStoreUnavailable",
    "ControlContextOwner",
    "ControlContextOwnerTask",
    "Mutation",
    "Mutator",
    "owned_elsewhere_message",
    "start_control_context_owner",
    "terminal_identity",
]

T = TypeVar("T")


@dataclass(frozen=True)
class Mutation(Generic[T]):
    """What a mutation callable hands back: what to store, and what to return.

    Attributes:
        record: The record to write, or ``None`` to write nothing. ``None`` is
            an ordinary outcome rather than a failure — a switch the gate
            refuses on facts read inside the job, a posture toggle that asks
            for the posture already stored, an owner tick with no request to
            consume. Nothing changed, so nothing is written, and the file's
            signature does not move for the readers watching it.
        result: Returned to the caller of :meth:`ControlContextOwner.mutate_record`
            once the write has landed. This is how a route gets its response
            body — the stored posture, the gate's verdict — out of the worker
            thread.
    """

    record: ControlContext | None
    result: T

    @classmethod
    def unchanged(cls, result: T) -> Mutation[T]:
        """A mutation that writes nothing and answers *result*."""
        return cls(record=None, result=result)


#: A mutation: given the record as it is on disk (``None`` when there is no
#: readable record), decide what to store. Runs in a worker thread, so it must
#: be synchronous and must not touch the event loop or anything guarded by it.
Mutator: TypeAlias = Callable[[ControlContext | None], Mutation[T]]


class ContextOwnerError(RuntimeError):
    """A mutation did not happen. Nothing was written.

    Carries an *error* code alongside the message because the routes answer
    these as HTTP failures and the code is what the response body and the log
    line agree on — the same shape the retired posture store's
    ``PostureStoreUnavailable`` established.
    """

    def __init__(self, error: str, message: str) -> None:
        super().__init__(message)
        self.error = error
        self.message = message


class ContextStoreUnavailable(ContextOwnerError):
    """There is nowhere to write, or the write itself failed.

    Both spellings of "the record could not be stored" share one class because
    the operator-facing answer is the same either way: the gesture was refused
    and nothing changed. :attr:`ContextOwnerError.error` tells them apart —
    ``store_unavailable`` for an agent-data root that does not resolve,
    ``store_write_failed`` for a write that raised.
    """


#: How each owner kind is named in a sentence an operator reads. The record
#: stores slugs; a refusal that printed ``web_terminal`` would be showing the
#: field name where the thing it names belongs. Unknown kinds keep their slug,
#: which is still better than dropping the only clue about what holds the file.
_OWNER_KIND_WORDS = {
    OWNER_WEB_TERMINAL: "the web terminal",
    OWNER_CONTROLS_SERVER: "the controls server",
}


def owned_elsewhere_message(owner: Owner | None) -> str:
    """The sentence a process reads when the record is not its to write.

    Spelled here rather than at each refusal because two surfaces say it: the
    mutation primitive, when a write loses the record to a takeover, and the
    web terminal's write routes, which refuse a follower before they reach for
    the record at all. Those are the same refusal a tick apart, and an operator
    who sees both must not be told two different things about where to go.

    Args:
        owner: The owner the record names, or ``None`` when there is no
            readable record to name one.
    """
    if owner is None:
        return (
            "The control-context record no longer names an owner this process can write as, "
            "so the change was not applied. It will be claimed again on the next owner tick."
        )
    whose = _OWNER_KIND_WORDS.get(owner.kind, owner.kind)
    where = f" on port {owner.port}" if owner.port else ""
    return (
        f"The control context is owned by {whose} at pid {owner.pid}{where}, not by this "
        "process, so the change was not applied. Make the change there."
    )


class ContextOwnedElsewhere(ContextOwnerError):
    """This process is not the owner the record names, so it may not write.

    Attributes:
        owner: The owner the record names, so a refusal can say where the
            operator should go. ``None`` when there is no readable record at
            all — this process's claim is simply gone, and the owner task will
            try to take it again on its next tick.
    """

    def __init__(self, owner: Owner | None, message: str) -> None:
        super().__init__("context_owned_elsewhere", message)
        self.owner = owner


class ControlContextOwner:
    """One process's claim on the record, and the only way it changes it.

    Built once per server (at lifespan start, by the owner task) and held on
    ``app.state``. The lock is an instance attribute rather than a module
    global so that it belongs to the same event loop as the app that owns it —
    an ``asyncio.Lock`` binds to the loop it is first awaited on, and a
    module-level one would be shared across every app in a process.

    Args:
        identity: Who this process is in the record's ``owner`` field. Every
            write stamps it.
        path: Write this file instead of the one :func:`record_path` resolves.
            For a caller that has already resolved a root, and for tests. When
            omitted the path is resolved on **each** mutation, so an owner
            built before the agent-data root existed starts working as soon as
            it does.
    """

    def __init__(self, identity: Owner, *, path: Path | None = None) -> None:
        self._identity = identity
        self._path = path
        self._lock = asyncio.Lock()

    @property
    def identity(self) -> Owner:
        """Who this process is in the record it writes."""
        return self._identity

    async def mutate_record(self, fn: Mutator[T], *, verify_owner: bool = True) -> T:
        """Read the record, apply *fn* to it, and store what *fn* returns.

        The read, the call and the write are one unit: serialised against every
        other mutation on this owner, and run together in a worker thread. A
        mutation therefore sees the result of the mutation before it, and no
        part of the sequence touches the event loop.

        Args:
            fn: The change to make. Receives the record as it is on disk —
                ``None`` when there is none, or none that parses — and returns
                a :class:`Mutation`. Runs in a worker thread.
            verify_owner: Refuse unless the record already names this process
                as its owner. ``False`` for a claim, which is how a process
                *becomes* the owner. The write stamps this owner either way.

        Returns:
            The mutation's ``result``, once any write has landed.

        Raises:
            ContextStoreUnavailable: The agent-data root does not resolve, or
                the write failed. Nothing changed.
            ContextOwnedElsewhere: *verify_owner* was set and the record names
                a different owner, or no record could be read. *fn* was not
                called and nothing changed.
            Exception: Whatever *fn* raises, unchanged. Nothing is written.
        """
        async with self._lock:
            return await asyncio.to_thread(self._apply, fn, verify_owner)

    def _apply(self, fn: Mutator[T], verify_owner: bool) -> T:
        """The whole read-modify-write, in one worker thread. Never on the loop."""
        path = self._resolve_path()
        record = read_record(path=path)

        if verify_owner:
            self._require_ownership(record)

        mutation = fn(record)
        if mutation.record is None:
            return mutation.result

        try:
            write_record(replace(mutation.record, owner=self._identity), path=path)
        except Exception as exc:  # noqa: BLE001 — reported to the operator as a 503
            logger.warning(
                "Could not write the control-context record to %s; nothing was changed",
                path,
                exc_info=True,
            )
            raise ContextStoreUnavailable(
                "store_write_failed",
                "The control-context record could not be written, so the change was not "
                "applied. Check the server's write access to the agent-data root and try "
                "again.",
            ) from exc
        return mutation.result

    def _resolve_path(self) -> Path:
        path = self._path if self._path is not None else record_path()
        if path is None:
            raise ContextStoreUnavailable(
                "store_unavailable",
                "This deployment's agent-data root does not resolve, so there is nowhere to "
                "record a control context that the agent would read back. Nothing was changed.",
            )
        return path

    def _require_ownership(self, record: ControlContext | None) -> None:
        owner = None if record is None else record.owner
        if owner == self._identity:
            return
        raise ContextOwnedElsewhere(owner, owned_elsewhere_message(owner))


# ---------------------------------------------------------------------------
# The ownership rules, and the task that applies them
# ---------------------------------------------------------------------------

#: How often the owner task re-examines the record. One second, which is what
#: bounds how long a deployment goes on carrying a dead owner's name — and it
#: is the controls server's reconciler period too, so the two owners take at
#: most one tick to agree on which of them is following.
OWNER_TICK_S = 1.0

#: The environment variable naming the port this terminal actually listens on.
#: ``osprey web`` publishes it once the bind has settled, the fallback port it
#: picks when the configured one is busy included, so it is the only place in
#: the serving process that knows the real number: ``create_app`` is reached
#: through uvicorn's factory with no arguments under ``--reload``, so a port
#: threaded through the app factory would be absent in exactly the mode that
#: runs it. Spelled literally rather than imported from the CLI, which keeps
#: its ``osprey`` imports function-local and would drag Click into the lifespan.
WEB_PORT_ENV = "OSPREY_WEB_PORT"

#: The refusal reason for a gate that raised while judging a request. The same
#: literal ``control_target`` and ``session_control`` carry, restated for the
#: reason they restate each other: a requester reads one vocabulary whichever
#: owner happened to answer it.
REASON_INTERNAL_ERROR = "internal_error"

#: The push frame the header chip refetches on. Deliberately payload-free:
#: every reader of the control context reads the record, and a frame carrying
#: a copy of it would be a second answer to "what is the context now".
CONTROL_CONTEXT_FRAME = {"type": "control_context"}


class _Unobserved:
    """Sentinel: this task has not yet seen who owns the record."""


def terminal_identity() -> Owner:
    """Who this web terminal is in the record's ``owner`` field.

    A port that is unset or unreadable becomes ``None`` rather than a guess:
    the port is what a refusal sends an operator to, and sending them to a
    socket nothing is listening on is worse than not naming one.
    """
    declared = (os.environ.get(WEB_PORT_ENV) or "").strip()
    try:
        port = int(declared) if declared else None
    except ValueError:
        logger.warning(
            "%s is %r, which is not a port; the record names none", WEB_PORT_ENV, declared
        )
        port = None
    return Owner(kind=OWNER_WEB_TERMINAL, pid=os.getpid(), port=port)


@dataclass(frozen=True)
class _Answerable:
    """One switch request this owner may answer, and the facts to judge it by.

    Assembled in a single worker thread — the glob, the request body, the
    fleet's reports, the in-flight markers and the rendered config are all
    disk — so that the answer itself needs nothing but the record it is about
    to write.
    """

    path: Path
    request_id: str
    target: str
    requested_by_pid: int
    requested_by: str
    requested_at: Any
    config: Any
    baseline: str
    reports: list[Any] = field(default_factory=list)
    in_flight: list[Any] = field(default_factory=list)


class ControlContextOwnerTask:
    """This terminal's claim on the control-context record, renewed once a second.

    The rules it applies are the PROPOSAL's: a web terminal claims over an
    owner that is **absent, dead, or a controls server**, and is a follower
    only behind a **live other web terminal**. It never displaces one, and it
    is never displaced by a server — a server claims only over an absent or
    dead owner, so the two rules compose to "the terminal wins, once it is
    there", which is right because the terminal is what an operator is
    actually looking at.

    The decision is made **inside** the mutation primitive's job rather than
    before it, against the record that same job is about to write. Deciding
    first and writing second would be the lost update the primitive exists to
    prevent, one rung up: a second terminal that started in between would be
    claimed over by a decision taken when it was not yet there.

    Every tick's blocking work — the glob, the stats, the reads, and above all
    :func:`~osprey_connectors.control_context.is_process_alive`, a syscall
    against a process this deployment does not control — happens in a worker
    thread. A liveness probe that never returns parks a thread and stops this
    task; it does not stop the terminal serving.

    Args:
        app: The application whose ``state`` this task publishes ownership on
            and whose ``broadcaster`` it pushes the context frame to.
        identity: Who to claim as. Defaults to :func:`terminal_identity`.
        interval_s: Seconds between ticks.
    """

    def __init__(
        self,
        app: Any,
        *,
        identity: Owner | None = None,
        interval_s: float = OWNER_TICK_S,
    ) -> None:
        self._app = app
        self._identity = identity if identity is not None else terminal_identity()
        self._owner = ControlContextOwner(self._identity)
        self._interval_s = float(interval_s)
        self._task: asyncio.Task[None] | None = None
        self._follows: Any = _Unobserved
        self._signatures: tuple[Any, ...] | None = None

    @property
    def identity(self) -> Owner:
        """Who this task claims as."""
        return self._identity

    @property
    def owner(self) -> ControlContextOwner:
        """The mutation primitive this task claims through."""
        return self._owner

    # -- the loop ----------------------------------------------------------

    def start(self) -> None:
        """Begin ticking. Needs a running loop, so it is not callable from ``create_app``."""
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop ticking and wait for the tick in flight to unwind."""
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self.tick_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — one bad tick must not end the claim
                logger.warning("The control-context owner tick failed", exc_info=True)

    async def tick_once(self) -> None:
        """Claim or follow, answer at most one request, push what moved.

        Public so a test drives the passes itself: everything this task
        decides is decided here, and the loop around it only handles the clock
        and the failures. Raises whatever it cannot handle — the loop logs it
        and ticks again, and at startup that raise is what leaves ``app.state``
        without an owner.
        """
        follows = await self._owner.mutate_record(self._claim_or_follow, verify_owner=False)
        self._publish(follows)
        if follows is None:
            await self._consume_one_request()
        await self._push_if_moved()

    # -- claim or follow ---------------------------------------------------

    def _claim_or_follow(self, record: ControlContext | None) -> Mutation[Owner | None]:
        """Take the record, or name the live terminal that already has it.

        Runs in the primitive's worker thread, so the liveness probe behind
        :func:`~osprey_connectors.control_context.live_owner` is off the loop.

        A claim is a **merge**: the target, generation and posture belong to
        the deployment and outlive every process that reads them, so taking
        the file over changes who may write it and nothing else. Only a record
        that is absent — which is also what an unparseable one reads as —
        starts a fresh one, at the deployment's baseline, generation 0,
        narrowing nothing.
        """
        owner = live_owner(record)
        if owner is not None and owner.pid == self._identity.pid:
            # Already ours. Writing an unchanged record here would move the
            # file's signature once a second and wake every reader watching it.
            return Mutation.unchanged(None)
        if owner is not None and owner.kind == OWNER_WEB_TERMINAL:
            return Mutation.unchanged(owner)
        if record is None:
            return Mutation(
                record=ControlContext(target=self._baseline(), generation=0), result=None
            )
        return Mutation(record=record, result=None)

    def _publish(self, follows: Owner | None) -> None:
        """Say on ``app.state`` what this terminal may do with the record.

        ``control_context_owner`` is how a route reaches the record at all —
        its **absence** is what makes the roster read-only and the write routes
        answer ``503 store_unavailable`` — and it appears only once a tick has
        got far enough to know the record is reachable.
        ``control_context_follows`` is the owner this terminal is behind, and
        ``None`` means this terminal is the owner.

        The snapshot can be a tick stale, which is why it is not the only
        guard: a write that races a takeover is refused inside the mutation
        primitive with :class:`ContextOwnedElsewhere`, carrying the owner the
        record actually names.
        """
        self._app.state.control_context_owner = self._owner
        self._app.state.control_context_follows = follows

        if follows == self._follows:
            return
        if follows is None:
            logger.info("This web terminal (pid %s) owns the control context", self._identity.pid)
        else:
            where = f" on port {follows.port}" if follows.port else ""
            logger.info(
                "Following the control context owned by web terminal pid %s%s; this terminal "
                "serves it read-only",
                follows.pid,
                where,
            )
        self._follows = follows

    def _baseline(self) -> str:
        """The target this deployment's own config selects.

        Imported where it is used: this module is reached on every web-terminal
        import, and the controls-server package is a heavier dependency than a
        record writer should carry.
        """
        from osprey.mcp_server.control_system.connector_host_manager import baseline_target
        from osprey.utils.workspace import load_osprey_config

        return baseline_target(load_osprey_config())

    # -- answering a switch request ----------------------------------------

    async def _consume_one_request(self) -> None:
        """Answer at most one switch request, while this terminal owns the record.

        One per tick for the reason the controls server also answers one per
        pass: a second answer written before the fleet has reported the first
        would overwrite a terminus somebody is still waiting on. That is the
        same reason the whole half is gated on
        :func:`~osprey_connectors.control_context.converged`.
        """
        request = await asyncio.to_thread(self._next_answerable)
        if request is None:
            return
        try:
            answered = await self._owner.mutate_record(lambda record: self._answer(request, record))
        except ContextOwnedElsewhere:
            # The record changed hands between the survey and the write. The
            # request is untouched and belongs to whoever owns it now.
            logger.info(
                "Leaving switch request %r: this terminal no longer owns the control context",
                request.request_id,
            )
            return
        if answered:
            await asyncio.to_thread(self._consume_file, request)

    def _next_answerable(self) -> _Answerable | None:
        """The first request this owner is in a position to answer. All disk work.

        Requests are named for the process that **asked**, not for a server, so
        any of them is this owner's to answer and they are taken in sorted
        order. A file nobody is left to read an answer for — unreadable,
        nameless, from a process that has gone, or past the request TTL — is
        removed here without a terminus, because the record's ``last_switch``
        is how a requester learns what happened and a block written for a
        requester that has gone would only overwrite the answer to a gesture
        somebody IS watching.
        """
        from osprey.mcp_server.control_system import target_state

        try:
            paths = sorted(target_state.state_dir().glob(target_state.REQUEST_FILE_GLOB))
        except Exception:  # noqa: BLE001 — an unreachable state dir is "no requests"
            logger.debug("Could not list the control-context directory", exc_info=True)
            return None
        if not paths:
            return None

        record = read_record()
        if record is None or not owned_here(record):
            return None

        reports = live_reports()
        if not converged(record, reports, None):
            logger.debug(
                "Leaving %d switch request(s): the deployment has not settled on generation %s",
                len(paths),
                record.generation,
            )
            return None

        config = self._rendered_config()
        baseline = self._baseline()
        for path in paths:
            request = self._triage(path, config, baseline, reports)
            if request is not None:
                return request
        return None

    def _rendered_config(self) -> Any:
        """The whole rendered config, which is what the switch gate reads."""
        from osprey.utils.workspace import load_osprey_config

        return load_osprey_config()

    def _triage(
        self, path: Path, config: Any, baseline: str, reports: list[Any]
    ) -> _Answerable | None:
        """One request file as something to answer, or ``None`` once it is removed.

        The removal cases are
        :func:`~osprey.mcp_server.control_system.target_state.triage_request`'s
        — spelled once, for both consumers of this directory.
        """
        from osprey.mcp_server.control_system import target_state

        triaged = target_state.triage_request(path)
        if triaged is None:
            return None
        body, pid, request_id = triaged

        session = body.get("session")
        requested_by = (
            session.strip() if isinstance(session, str) and session.strip() else f"pid:{pid}"
        )
        return _Answerable(
            path=path,
            request_id=request_id,
            target=str(body.get("target") or "").strip(),
            requested_by_pid=pid,
            requested_by=requested_by,
            requested_at=body.get("requested_at"),
            config=config,
            baseline=baseline,
            reports=reports,
            in_flight=target_state.in_flight_executions(),
        )

    def _answer(self, request: _Answerable, record: ControlContext | None) -> Mutation[bool]:
        """The terminus for *request*, judged against the record as it is now.

        ``True`` means the request has been dealt with and its file is ready to
        go — the case where an earlier tick already answered it and only the
        removal was outstanding included, which is safe precisely because the
        answer is in the record: reaching a terminus twice writes nothing twice.
        """
        if record is None:
            return Mutation.unchanged(False)
        if (record.last_switch or {}).get("request_id") == request.request_id:
            logger.debug("Switch request %r was already answered", request.request_id)
            return Mutation.unchanged(True)

        if request.target == record.target:
            # No mint. The deployment is where the request asked for it to be,
            # and a generation bumped for a switch that did not happen would
            # refuse every write bound to the old one for nothing.
            return self._terminus(
                record,
                request,
                status=SWITCH_APPLIED,
                reason=None,
                detail=unchanged_detail(request.target, record.generation),
                generation=record.generation,
            )

        try:
            verdict = self._gate(record, request)
        except Exception as exc:  # noqa: BLE001 — reported to the requester as a refusal
            logger.exception("Switch request %r could not be judged", request.request_id)
            return self._terminus(
                record,
                request,
                status=SWITCH_REFUSED,
                reason=REASON_INTERNAL_ERROR,
                detail=f"{type(exc).__name__}: {exc}",
                generation=None,
            )

        if not verdict.allowed:
            return self._terminus(
                record,
                request,
                status=SWITCH_REFUSED,
                reason=str(verdict.reason or ""),
                detail=verdict.detail,
                generation=None,
            )
        return self._terminus(
            record,
            request,
            status=SWITCH_APPLIED,
            reason=None,
            detail=applied_detail(request.target, record.generation + 1),
            generation=record.generation + 1,
        )

    def _terminus(
        self,
        record: ControlContext,
        request: _Answerable,
        *,
        status: str,
        reason: str | None,
        detail: str,
        generation: int | None,
    ) -> Mutation[bool]:
        return Mutation(
            record=terminus(
                record,
                request_id=request.request_id,
                target=request.target,
                requested_at=request.requested_at,
                requested_by=request.requested_by,
                status=status,
                reason=reason,
                detail=detail,
                generation=generation,
            ),
            result=True,
        )

    def _gate(self, record: ControlContext, request: _Answerable) -> Any:
        """The switch verdict, in the words every other surface would get.

        Asked here rather than when the request was filed: an execution that
        started in between, a posture that narrowed, a target that stopped
        being eligible all have to refuse this switch, and a verdict taken
        earlier would be a verdict about a different moment. ``current_target``
        is the **record's**, because the record is what this answer is written
        into.
        """
        from osprey.mcp_server.control_system import target_eligibility

        config = request.config
        section = config.get("control_system") if isinstance(config, dict) else None
        return target_eligibility.evaluate_switch(
            config,
            request.target,
            current_target=record.target,
            baseline=request.baseline,
            in_flight=request.in_flight,
            reports=request.reports,
            writes_enabled=target_eligibility.effective_writes_for_target(section, request.target),
        )

    def _consume_file(self, request: _Answerable) -> None:
        """Remove the request file, once the answer to it has been read back.

        The read-verify is the controls server's, for its reason: a requester
        polls the record for its own id and nothing else, so a request removed
        on the strength of a write that did not stick would leave it with
        neither an outcome nor a pending gesture to wait on.
        """
        from osprey.mcp_server.control_system import target_state

        record = read_record()
        if record is None or not owned_here(record):
            answered = False
        else:
            answered = (record.last_switch or {}).get("request_id") == request.request_id
        if not answered:
            logger.warning(
                "The answer to switch request %r did not stick; leaving the request for "
                "whichever process owns the control context now",
                request.request_id,
            )
            return
        target_state.remove_request(requested_by_pid=request.requested_by_pid)

    # -- the push ----------------------------------------------------------

    async def _push_if_moved(self) -> None:
        """Broadcast the context frame when the record or a report has changed.

        Watched by signature rather than pushed at each write, so that a record
        written by a route — the posture toggle, the target switch — reaches
        the browsers without every writer having to remember to push. The cost
        is up to one tick of latency on a change this task did not make itself.

        The first observation establishes the baseline and pushes nothing:
        there is nothing to tell a browser at startup, because there is no
        browser yet.
        """
        signatures = await asyncio.to_thread(self._observe)
        if signatures == self._signatures:
            return
        first = self._signatures is None
        self._signatures = signatures
        if first:
            return
        broadcaster = getattr(self._app.state, "broadcaster", None)
        if broadcaster is not None:
            broadcaster.broadcast(dict(CONTROL_CONTEXT_FRAME))

    def _observe(self) -> tuple[Any, ...]:
        """The record's signature and every report's, in one worker thread."""
        return (
            file_signature(record_path()),
            tuple(file_signature(path) for path in report_paths()),
        )


async def start_control_context_owner(
    app: Any, *, interval_s: float = OWNER_TICK_S
) -> ControlContextOwnerTask:
    """Claim the control context for this terminal, and keep the claim renewed.

    Never raises. The startup claim is fail-open: when it cannot be made — an
    agent-data root that does not resolve, a directory that cannot be written —
    it is logged, ``app.state`` is left without ``control_context_owner`` so
    the roster renders read-only and the write routes answer their
    ``503 store_unavailable``, and the task is started anyway so the next tick
    tries again. A terminal that cannot own the control context is still a
    terminal.
    """
    task = ControlContextOwnerTask(app, interval_s=interval_s)
    try:
        await task.tick_once()
    except Exception:  # noqa: BLE001 — never let the claim block startup
        logger.warning(
            "Could not claim the control context at startup; this terminal serves the control "
            "roster read-only until an owner tick takes it",
            exc_info=True,
        )
    task.start()
    return task
