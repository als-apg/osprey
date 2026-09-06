"""The reconciler that brings this server into line with the control context.

The control target, its generation and the write posture are one record per
deployment — ``control_target/control_context.json`` — written only by whichever
process owns it: the web terminal when there is one, else a controls server.
This server is one reader of that record among many, and this task is what
makes the record true of the connector this process holds.

Three things happen once a second, in this order, and the order is the design.

**Claim or follow.** A controls server is the fallback owner: it claims the
record when nothing alive owns it and follows on every tick where a live web
terminal (or another server that got there first) does. The claim itself is
:func:`~osprey.mcp_server.control_system.server_context.claim_control_context`
— one implementation, shared with server start — and it is asked for only when
this tick has already seen that the owner is absent or dead, so a follower
neither writes nor logs its way through the day.

**Reconcile to the record.** The record's ``(target, generation)`` is what this
server's connector host must be on. Before any await, this pass compares them
with what the supervisor intends; when they differ **and a child is live**, it
publishes ``last_switch{status: applying}`` into this server's own report
first, so that no other session's launch is admitted into the window where this
process is between two targets. The move itself is
:meth:`~osprey.mcp_server.control_system.connector_host_manager.ConnectorHostManager.reconcile`,
which adopts silently when no child is running (nothing is bound, so nothing
can be bound wrongly, and the first launch comes up on the adopted values) and
otherwise adopts the generation against the running child or spawns the new
target. It publishes its own ``applied`` terminus, which is what releases the
``applying`` block written here; a failed swap files ``failed`` at the
generation it kept. This reconcile runs BEFORE the posture half deliberately: a
same-target adoption leaves a pending realignment pending, and the rebuild that
answers it happens in the same pass.

**Consume switch requests.** A surface that cannot call this server writes down
what it wants: a per-requester ``switch_request_<pid>.json``, consumed by the
record's owner. There is no addressee — the guard is ownership, and a follower
leaves every request alone. The owner consumes only while
:func:`~osprey_connectors.control_context.converged` reports the fleet settled,
so ``last_switch`` moves at most once per convergence cycle and no terminus is
clobbered by a swap that is still landing.

Two vocabularies, and they are not the same word twice
------------------------------------------------------
A **report** (``server_<pid>.json``) carries this server's PROGRESS:
``applying`` / ``applied`` / ``failed``, published by
:func:`~osprey.mcp_server.control_system.target_state.publish_last_switch` and
read by every other process to decide whether the deployment has settled. The
**record** carries a REQUEST'S TERMINUS: :data:`STATUS_APPLIED` or
:data:`STATUS_REFUSED`, and nothing else — a refusal is a record write that
moves neither target nor generation. Reporting progress is this server's; the
terminus is the owner's.

How a request is answered, and why in that order
------------------------------------------------
1. The record is re-read IMMEDIATELY before acting. A request whose id is
   already the record's ``last_switch.request_id`` was answered by an earlier
   pass — or by an earlier process — and the only thing left to do with it is
   unlink it. That is what makes consumption idempotent: the answer is the
   record, not the disappearance of a file.
2. A request naming the target the record is already on is answered
   :data:`STATUS_APPLIED` at the CURRENT generation. Nothing is minted: the
   deployment is where it was asked to be, and a generation bumped for a switch
   that did not happen would refuse every write in flight for nothing.
3. Otherwise the gate runs — :func:`target_eligibility.evaluate_switch`, the
   same verdict the terminal route and the agent's tool get — and either the
   refusal or the move is written into the record atomically: target, ``+1``
   generation, and the terminus in one replace.
4. The write is read-verified before the request file is unlinked. Two owners
   can believe they own one record for as long as it takes the loser to find
   out; unlinking a request whose answer did not stick would leave the operator
   with neither an outcome nor a pending gesture.
5. Only then are the ledger and the operator's activity feed told, ledger
   first: every await is a place this task can be cancelled, and a shutdown
   between the two must not lose the record of a switch that happened.

The posture half
----------------
A narrowing recorded for the target this deployment is currently ON does not
take effect by itself: writes are refused per call from the record already, but
the connector-host child *connected* on a gateway role chosen under the old
posture. Realignment is a rebuild of that child through
:meth:`~osprey.mcp_server.control_system.server_context.ControlSystemContext.invalidate_connector`,
which owns its own lock — this task takes none, deliberately, because a second
lock around the same operation is how two things that must agree stop agreeing.
A record that moved also republishes the ``targets`` block through
:meth:`~osprey.mcp_server.control_system.connector_host_manager.ConnectorHostManager.publish_display`:
display metadata names the gateway a posture chose, and a narrowing that lands
without a switch to carry it has no other writer to restate that name.

The rebuild waits for any execution in flight. A python execution is stamped
with the target and generation it launched under, and retiring its child
mid-run would break a promise the executor made rather than enforce a posture
the operator set. So the realignment is *deferred*, and
:func:`~osprey.mcp_server.control_system.target_state.publish_posture_realign`
says ``pending`` while it waits — which is what lets the popover say "read-only
applies after the running execution finishes" instead of showing a toggle that
appears to have done nothing.

A narrowing on a target this deployment is NOT on realigns nothing: the
connector has nothing to do with that machine, and the record is read again the
moment a switch lands there. A switch that happens while a realignment is
pending clears it for the same reason — the child the switch built read the
record on the way up.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import Any

from osprey.audit import posture
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.http import (
    SWITCH_OUTCOME_FAILURE,
    SWITCH_OUTCOME_SUCCESS,
    TARGET_SWITCH_TOOL,
    notify_target_switch_async,
)
from osprey_connectors import control_context, posture_store

logger = logging.getLogger("osprey.mcp_server.control_system.session_control")

#: How often the loop looks. A second is the whole budget an operator will wait
#: for a button to answer, and a settled deployment costs one cached record
#: read, one ``stat`` and one glob of a small directory.
POLL_INTERVAL_S = 1.0

# -- terminus statuses ------------------------------------------------------
#
# The record's own vocabulary, restated here rather than respelled: a switch
# request ends ``applied`` or ``refused`` and there is no third answer. A
# failure to reach the new target is not a terminus of the request — the
# request landed, the deployment moved, and the server that could not follow
# says so in its own report.

#: The record moved (or was already where the request asked for).
STATUS_APPLIED = control_context.SWITCH_APPLIED
#: The gate said no. Neither target nor generation moved.
STATUS_REFUSED = control_context.SWITCH_REFUSED

#: An exception the consumption did not classify. Restated from
#: :data:`~osprey.mcp_server.control_system.tools.control_target.REASON_INTERNAL_ERROR`
#: rather than imported — that module imports the server module this task's
#: lifespan lives in — and pinned equal to it by a test.
REASON_INTERNAL_ERROR = "internal_error"

#: Realignment states, in :func:`target_state.publish_posture_realign`'s terms.
REALIGN_PENDING = "pending"
REALIGN_DONE = "done"

#: What the ledger calls a completed switch. A refusal files the gate's own
#: reason instead, so the two surfaces' records can be matched on it.
REASON_TARGET_SWITCHED = "target_switched"

#: The audit subject for a gesture that moves the deployment's control target —
#: the same word the agent's tool and the web route record under, so an
#: operator reading the ledger sees one kind of event whichever surface asked.
AUDIT_SUBJECT_TARGET_SET = TARGET_SWITCH_TOOL

#: The env var the registry stamps with this server's rendered name, and the
#: fallback surface when it is unset. Both restated from
#: :mod:`osprey.mcp_server.audit_middleware` rather than imported, because
#: importing that module here would pull the middleware's clamp machinery into
#: a background task that decides nothing about tool calls.
TOOL_PREFIX_ENV = "OSPREY_MCP_TOOL_PREFIX"
SURFACE_UNPREFIXED = "mcp"

__all__ = [
    "AUDIT_SUBJECT_TARGET_SET",
    "POLL_INTERVAL_S",
    "REALIGN_DONE",
    "REALIGN_PENDING",
    "REASON_INTERNAL_ERROR",
    "REASON_TARGET_SWITCHED",
    "STATUS_APPLIED",
    "STATUS_REFUSED",
    "SessionControlReconciler",
]


def _requester(record: dict[str, Any], pid: int) -> str:
    """Who to name in the terminus and the ledger for a request from *pid*.

    The requester's audit session when it has one, so a ledger reader can match
    the gesture to the session that made it, and its PID otherwise — a bare
    ``claude`` reports no session and still has to be nameable.
    """
    session = record.get("session")
    if isinstance(session, str) and session.strip():
        return session.strip()
    return f"pid:{pid}"


class SessionControlReconciler:
    """Reconciles this server to the deployment's control context, once a second.

    Owned by the controls server's lifespan beside
    :class:`~osprey.mcp_server.control_system.endpoint_prober.EndpointProber`,
    because a task needs a running loop and ``create_server()`` is called before
    one exists.

    ``poll_once`` is public so a test can drive the passes itself: everything
    this class decides is decided there, and the loop around it only handles
    the clock and the failures.
    """

    def __init__(self, *, interval_s: float = POLL_INTERVAL_S) -> None:
        self._interval_s = float(interval_s)
        self._task: asyncio.Task[None] | None = None

        # What the last pass saw of the record's posture. All three start
        # unset, so the first pass *baselines* rather than acting: a narrowing
        # already recorded when this server started was read by the child it
        # started, and replaying it as a change would rebuild a connector
        # nobody narrowed.
        self._record_signature: tuple[int, int, int] | None = None
        self._active_target: str | None = None
        self._active_posture: str | None = None
        self._realign_pending = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Start the poll loop. Idempotent; does not wait for a pass."""
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="session-control-reconciler")

    async def stop(self) -> None:
        """Cancel the poll loop and wait for it to finish. Idempotent."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self) -> None:
        """Poll until cancelled, surviving anything one pass can raise.

        A reconciler that died on one bad pass would strand every later
        gesture with nothing an operator could see — the chip would keep
        accepting clicks and nothing would ever answer them. So a failed pass
        is logged and the next one happens.
        """
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Session-control reconcile pass failed; continuing")
            await asyncio.sleep(self._interval_s)

    # -- one pass ----------------------------------------------------------

    async def poll_once(self) -> None:
        """One pass: claim or follow, reconcile to the record, posture, requests.

        The reconcile runs before the posture half so that a realignment left
        pending by a previous pass is answered against the child this pass
        settled on, and the request half runs last so that a request consumed
        here is reconciled to on the next tick with the record already written.
        """
        context = self._context()
        if context is None:
            return
        record = self._own_or_follow(context)
        if record is not None:
            await self._reconcile_to_record(context, record)
        await self._reconcile_posture(context)
        if record is not None:
            await self._consume_requests(context)

    def _context(self) -> Any:
        """The server context, or ``None`` when there is not one yet.

        A poll before ``initialize_server_context()`` has run is not an error:
        the lifespan starts this task, and a context that cannot be read has no
        connector to reconcile.
        """
        from osprey.mcp_server.control_system.server_context import get_server_context

        try:
            return get_server_context()
        except RuntimeError:
            return None

    # -- ownership ---------------------------------------------------------

    def _own_or_follow(self, context: Any) -> control_context.ControlContext | None:
        """The record this pass works from, claimed if nothing alive owns it.

        The liveness question is asked here and the claim is made there:
        :func:`~osprey.mcp_server.control_system.server_context.claim_control_context`
        is the single claim implementation — merge, read-verify, never raises —
        but it announces what it found, and a follower that called it once a
        second would say so once a second. So it is reached only on the tick
        where this server might actually take the record over.

        Returns:
            The record to reconcile to — this server's, somebody else's, or the
            one that beat it to the claim — and ``None`` when there is no
            readable record and none could be written.
        """
        record = control_context.read_record()
        if control_context.live_owner(record) is not None:
            return record

        from osprey.mcp_server.control_system.server_context import claim_control_context

        try:
            baseline = context.baseline
        except Exception:
            logger.debug("No deployment baseline to claim the control context with")
            return record
        return claim_control_context(baseline=baseline) or record

    # -- the record --------------------------------------------------------

    async def _reconcile_to_record(
        self, context: Any, record: control_context.ControlContext
    ) -> None:
        """Bring the connector host onto the record's ``(target, generation)``.

        The comparison and the ``applying`` block both happen before the first
        await, so a live server holding a connector is never silent about a
        generation it has not reached: every other reader learns from the
        report that this process is between two targets, within one tick of the
        record moving.

        With no live child nothing is published at all. Nothing is bound, so
        nothing can be bound to the wrong generation, and an ``applying`` block
        written here would never be released — the silent adoption publishes no
        terminus to release it with.
        """
        try:
            hosts = context.connector_hosts
            on_target = hosts.active_target()
            on_generation = hosts.active_generation()
        except Exception:
            logger.debug("No connector-host supervisor to reconcile to the control context")
            return
        if record.target == on_target and record.generation == on_generation:
            return

        if hosts.has_child():
            self._publish_applying(hosts, record.generation)

        # Imported inside the call: the manager module imports this server's
        # context, which imports enough of the server that a module-level
        # import here would close a cycle.
        from osprey.mcp_server.control_system.connector_host_manager import SwitchError

        try:
            result = await hosts.reconcile(record.target, record.generation)
        except SwitchError as exc:
            # Already reported ``failed`` at the generation it kept, by the
            # supervisor itself. Publishing a second verdict here would be this
            # task's opinion about a swap it did not run; the next pass finds
            # the record still naming another target and tries again.
            logger.warning(
                "Could not reconcile to the control context's target %r at stage %r: %s",
                record.target,
                exc.stage,
                exc.detail,
            )
            return

        if result["target_changed"] or result["generation_changed"]:
            logger.info(
                "Reconciled to the control context: target %r, generation %s (respawned=%s)",
                result["target"],
                result["generation"],
                result["respawned"],
            )

    def _publish_applying(self, hosts: Any, generation: int) -> None:
        """Say this server is mid-swap, with the deadline only it can compute.

        The bound is this process's own spawn, probe and drain timeouts; every
        other reader sees the report and nothing else, so the publisher writes
        the deadline it will be judged against. Never costs the reconcile: a
        block that could not be written leaves readers judging this server by
        its binding, which is stricter rather than looser.
        """
        try:
            target_state.publish_last_switch(
                {"generation": generation, "status": target_state.SWITCH_APPLYING},
                expires_in_s=hosts.applying_bound_s(),
            )
        except Exception:
            logger.warning(
                "Could not publish the in-progress switch block for generation %s",
                generation,
                exc_info=True,
            )

    # -- the posture -------------------------------------------------------

    async def _reconcile_posture(self, context: Any) -> None:
        """Republish on any move of the record; realign on the ACTIVE target's.

        The two halves answer different questions. Display metadata names the
        gateway a posture chose, so it is stale the moment the record moves for
        ANY target, and it is republished on every move. The connector is only
        wrong when the narrowing for the target this deployment is ON moved, so
        that is the only case that rebuilds a child.
        """
        signature = control_context.file_signature(control_context.record_path())
        try:
            target = context.connector_hosts.active_target()
        except Exception:
            logger.debug("No connector-host supervisor to read the active target from")
            return

        record_moved = signature != self._record_signature
        if record_moved:
            # Only on a move of the record. A change of TARGET republished
            # already, inside the switch that made it.
            self._publish_display(context)
        if record_moved or target != self._active_target:
            self._record_signature = signature
            self._observe(target)

        if self._realign_pending:
            await self._realign(context)

    def _observe(self, target: str) -> None:
        """Record what the context now says about *target*, and whether it moved.

        A change of TARGET is not a change of posture: the child a switch built
        read the record on the way up, so the deployment is already aligned and
        a realignment left pending from the previous target is moot. A change
        of the narrowing for the target this server is still on is the one case
        that owes the operator a rebuild.
        """
        entry = posture_store.target_posture(target)
        if target != self._active_target:
            self._active_target = target
            self._active_posture = entry
            if self._realign_pending:
                # The deployment left the target that narrowing was about, on a
                # child that read the record itself. Nothing is outstanding.
                self._realign_pending = False
                self._publish_realign(REALIGN_DONE)
            return
        if entry == self._active_posture:
            return
        self._active_posture = entry
        self._realign_pending = True
        # Published before the wait, not after it: "pending" is the answer to
        # "why has my toggle not taken effect", and it is only useful while the
        # operator is still asking.
        self._publish_realign(REALIGN_PENDING)

    async def _realign(self, context: Any) -> None:
        """Rebuild the control-system connector, once nothing is running.

        Two ways the rebuild does not happen, and both leave the realignment
        pending rather than reporting it done: an exception out of
        ``invalidate_connector``, and its ``False`` — which is the ordinary
        one. A connector-host respawn that fails is *caught inside* that method
        (spawn-then-swap: the old child keeps serving rather than being torn
        down for a replacement that will not come up), so the only evidence a
        caller ever gets is the returned answer. Publishing ``done`` on it
        would tell an operator their read-only toggle had taken effect on a
        child still connected under the old posture.

        A test double that answers ``None`` counts as a rebuild: only an
        explicit ``False`` is the refusal this reads.

        Neither wait re-publishes: ``pending`` was written the moment the flip
        was seen and it is still the true answer, so a retry that restamped it
        would rewrite the report once a second for the length of a run to say
        nothing new.
        """
        if target_state.in_flight_executions():
            return
        try:
            rebuilt = await context.invalidate_connector("control_system")
        except Exception:
            logger.warning(
                "Could not realign the control-system connector to the recorded posture; "
                "retrying on the next pass",
                exc_info=True,
            )
            return
        if rebuilt is False:
            logger.warning(
                "The connector host refused to respawn, so the recorded posture is not "
                "realigned yet; retrying on the next pass"
            )
            return
        self._realign_pending = False
        self._publish_realign(REALIGN_DONE)

    def _publish_display(self, context: Any) -> None:
        """Restate the target display metadata. Never costs the pass that ran it.

        The writer renders identity from the posture it is actually in, so a
        narrowing recorded for ANY target moves what the block should say —
        including one this server is not on, where no switch runs and nothing
        else would republish. A republication that fails is logged and left:
        readers keep rendering the endpoint the last successful one named,
        which is a stale identity line rather than a reconcile pass that
        stopped before the realignment it was really there for.

        Args:
            context: The server context whose supervisor owns the block.
        """
        try:
            context.connector_hosts.publish_display()
        except Exception:
            logger.warning("Could not republish the target display metadata", exc_info=True)

    def _publish_realign(self, state: str) -> None:
        """Publish a realignment note. Never costs the reconcile that made it."""
        try:
            target_state.publish_posture_realign({"state": state})
        except Exception:
            logger.warning("Could not publish the posture realignment note", exc_info=True)

    # -- switch requests ---------------------------------------------------

    async def _consume_requests(self, context: Any) -> None:
        """Answer one switch request, while this process owns the record.

        Globbed rather than read from a slot of this server's own: a request is
        named for the process that ASKED, and any of them is this owner's to
        answer. The directory is small and the read is one ``os.scandir``, so
        there is no signature to keep — and keeping one would be a second
        answer to "has this been dealt with", which the record's
        ``last_switch.request_id`` already answers correctly.

        At most one request is answered per pass. Consuming a second while the
        fleet has not yet reported the first would clobber a terminus somebody
        is still waiting for, which is the same reason the whole half is gated
        on :func:`~osprey_connectors.control_context.converged`.
        """
        try:
            paths = sorted(target_state.state_dir().glob(target_state.REQUEST_FILE_GLOB))
        except OSError:
            logger.debug("Could not list the control-context directory", exc_info=True)
            return
        if not paths:
            return

        record = control_context.read_record()
        if record is None or not control_context.owned_here(record):
            # A follower answers nothing. It files requests of its own and
            # waits for the owner exactly as every other surface does.
            return

        reports = control_context.live_reports()
        if not control_context.converged(record, reports, None):
            logger.debug(
                "Leaving %d switch request(s): the deployment has not settled on generation %s",
                len(paths),
                record.generation,
            )
            return

        for path in paths:
            if await self._consume_one(context, path, reports):
                return

    async def _consume_one(self, context: Any, path: Path, reports: list[Any]) -> bool:
        """Deal with one request file. ``True`` when it was actually answered.

        A file this owner will never act on is removed by
        :func:`~osprey.mcp_server.control_system.target_state.triage_request`
        — which is where the four ways that happens are spelled, once, for both
        consumers of this directory — and reports ``False``, so the pass goes on
        to look at the next one.
        """
        triaged = target_state.triage_request(path)
        if triaged is None:
            return False
        body, pid, request_id = triaged

        # Re-read immediately before acting. Between the glob and here the
        # record can have been answered by an earlier pass of this same loop, or
        # taken over by a process that outranks this one.
        record = control_context.read_record()
        if record is None or not control_context.owned_here(record):
            return True
        if (record.last_switch or {}).get("request_id") == request_id:
            # Answered already. The unlink is all that was outstanding, and it
            # is safe precisely because the answer is in the record: reaching
            # this terminus twice writes nothing twice.
            logger.debug("Switch request %r was already answered; removing it", request_id)
            target_state.remove_request(requested_by_pid=pid)
            return False

        wanted = str(body.get("target") or "").strip()
        if wanted == record.target:
            # No mint. The deployment is where the request asked for it to be,
            # and a generation bumped for a switch that did not happen would
            # refuse every write bound to the old one for nothing.
            await self._answer(
                record,
                body,
                pid=pid,
                status=STATUS_APPLIED,
                reason=None,
                detail=control_context.unchanged_detail(wanted, record.generation),
                generation=record.generation,
            )
            return True

        try:
            verdict = self._gate(context, record, wanted, reports)
        except Exception as exc:
            logger.exception("Switch request %r could not be judged", request_id)
            await self._answer(
                record,
                body,
                pid=pid,
                status=STATUS_REFUSED,
                reason=REASON_INTERNAL_ERROR,
                detail=f"{type(exc).__name__}: {exc}",
                generation=None,
            )
            return True

        if not verdict.allowed:
            await self._answer(
                record,
                body,
                pid=pid,
                status=STATUS_REFUSED,
                reason=str(verdict.reason or ""),
                detail=verdict.detail,
                generation=None,
            )
            return True

        await self._answer(
            record,
            body,
            pid=pid,
            status=STATUS_APPLIED,
            reason=None,
            detail=control_context.applied_detail(wanted, record.generation + 1),
            generation=record.generation + 1,
        )
        return True

    def _gate(
        self,
        context: Any,
        record: control_context.ControlContext,
        wanted: str,
        reports: list[Any],
    ) -> Any:
        """The switch verdict, in the words every other surface would get.

        Asked HERE rather than when the request was written: an execution that
        started in between, a posture that narrowed, a target that stopped
        being eligible all have to refuse this switch, and a verdict taken
        earlier would be a verdict about a different moment.

        The gate opens no file, so its inputs are gathered by its caller — the
        record's own target rather than the supervisor's, because this pass has
        already reconciled to the record and it is the record the answer is
        written into.
        """
        from osprey.mcp_server.control_system import target_eligibility

        config = context.config.raw
        section = config.get("control_system") if isinstance(config, dict) else None
        return target_eligibility.evaluate_switch(
            config,
            wanted,
            current_target=record.target,
            baseline=context.baseline,
            in_flight=target_state.in_flight_executions(),
            reports=reports,
            writes_enabled=target_eligibility.effective_writes_for_target(section, wanted),
        )

    # -- the terminus ------------------------------------------------------

    async def _answer(
        self,
        record: control_context.ControlContext,
        body: dict[str, Any],
        *,
        pid: int,
        status: str,
        reason: str | None,
        detail: str,
        generation: int | None,
    ) -> None:
        """End one request: write the record, verify, consume, report, record.

        *generation* is the record's generation AFTER this answer, and ``None``
        is what makes it a refusal — a refusal moves neither target nor
        generation, and its terminus carries a null generation because there is
        no binding to name.

        The record is written before the request file is removed, and only
        removed once the write has been read back: the requester polls the
        record for its own id and nothing else, so an unlinked request whose
        answer did not stick would leave it with neither an outcome nor a
        pending gesture to wait on.
        """
        request_id = str(body.get("request_id") or "")
        wanted = str(body.get("target") or "").strip()
        requested_by = _requester(body, pid)

        updated = control_context.terminus(
            record,
            request_id=request_id,
            target=wanted,
            requested_at=body.get("requested_at"),
            requested_by=requested_by,
            status=status,
            reason=reason,
            detail=detail,
            generation=generation,
        )

        if not control_context.write_terminus(updated, request_id):
            logger.warning(
                "The answer to switch request %r did not stick; leaving the request for "
                "whichever process owns the control context now",
                request_id,
            )
            return

        target_state.remove_request(requested_by_pid=pid)

        # The ledger record is filed BEFORE the awaited emit, and not after it:
        # every await is a place this task can be cancelled, and a shutdown
        # landing between the two would leave a switch that happened with no
        # record that it did. The activity emit is a convenience for whoever is
        # watching; the ledger is the trail.
        self._record_gesture(
            status=status,
            reason=reason,
            request_id=request_id,
            target=wanted,
            requested_by=requested_by,
        )
        await self._notify(
            from_target=record.target,
            to_target=wanted,
            status=status,
            reason=reason,
            generation=generation,
        )

    async def _notify(
        self,
        *,
        from_target: str,
        to_target: str,
        status: str,
        reason: str | None,
        generation: int | None,
    ) -> None:
        """Report the attempt on the operator's activity feed. Never raises."""
        try:
            if status == STATUS_APPLIED:
                await notify_target_switch_async(
                    from_target=from_target,
                    to_target=to_target,
                    outcome=SWITCH_OUTCOME_SUCCESS,
                    generation=generation,
                )
            else:
                await notify_target_switch_async(
                    from_target=from_target,
                    to_target=to_target,
                    outcome=SWITCH_OUTCOME_FAILURE,
                    reason=reason,
                )
        except Exception:  # pragma: no cover - the emitter swallows its own
            logger.debug("Could not report the switch attempt (ignored)", exc_info=True)

    def _record_gesture(
        self,
        *,
        status: str,
        reason: str | None,
        request_id: str,
        target: str,
        requested_by: str,
    ) -> None:
        """File one ledger record for the operator's gesture. Never raises.

        Through :func:`~osprey.audit.dedup.record_and_mark` for the reason the
        web routes use it: this is the innermost layer that decided, and a
        record filed here claims the decision so no outer recorder files a
        second, blander one for the same event.

        ``detail`` carries identifiers only — the request id, the target name,
        and who asked — never a config value or a payload.
        """
        try:
            from osprey.audit.dedup import record_and_mark
            from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED

            record_and_mark(
                decision=DECISION_ALLOWED if status == STATUS_APPLIED else DECISION_REFUSED,
                reason=reason or REASON_TARGET_SWITCHED,
                surface=(os.environ.get(TOOL_PREFIX_ENV) or "").strip() or SURFACE_UNPREFIXED,
                posture=posture.posture(),
                posture_source=posture.posture_source(),
                session=posture.posture_session(),
                subject=AUDIT_SUBJECT_TARGET_SET,
                detail=(
                    f"target={target} status={status} "
                    f"requested_by={requested_by} request_id={request_id}"
                ),
            )
        except Exception:  # noqa: BLE001 — the audit trail degrades; the switch does not
            logger.warning("Could not record the target-switch gesture for audit", exc_info=True)
