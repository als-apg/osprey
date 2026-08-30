"""The reconciler that turns desired session control state into what is true.

The web terminal cannot call this server. The controls MCP server speaks
JSON-RPC over stdio to the Claude Code process that spawned it, has no inbound
channel of its own, and is the single writer of the control-target state file.
So the header chip does not ask it for anything: it *writes down what an
operator wants* under the shared agent-data root — a per-(session, target)
posture entry in :mod:`osprey_connectors.session_store`, a switch request in
``request_<server_pid>.json`` — and this task is what makes either of them
happen to the connector this process owns.

That asymmetry is the design, not a workaround. Desired state and true state
live in different files with different writers: a request says what somebody
asked for and can be stale, refused or ignored, while the state file says what
IS, and only this server writes it. Nothing on the web side can move a session
by writing a file; it can only ask.

What the loop does, once a second
---------------------------------
Both halves are driven by a ``(st_mtime_ns, st_size, st_ino)`` signature — the
same rule :mod:`osprey_connectors.session_store` and :mod:`osprey.health.signatures`
follow, with the inode third because both files are replaced atomically and two
gestures inside one filesystem clock tick would otherwise differ in nothing
else. A poll whose signatures have not moved costs two ``stat`` calls.

**The posture store.** A narrowing recorded for the target this session is
currently ON does not take effect by itself: writes are refused per call from
the store already, but the connector-host child *connected* on a gateway role
chosen under the old posture, so a narrowed session keeps a write gateway it
may no longer use. Realignment is a rebuild of that child through
:meth:`~osprey.mcp_server.control_system.server_context.ControlSystemContext.invalidate_connector`,
which owns its own lock — this task takes none, deliberately, because a second
lock around the same operation is how two things that must agree stop agreeing.

The rebuild waits for any execution in flight. A python execution is stamped
with the target and generation it launched under, and retiring its child
mid-run would break a promise the executor made rather than enforce a posture
the operator set. So the realignment is *deferred*, and
:func:`~osprey.mcp_server.control_system.target_state.publish_posture_realign`
says ``pending`` while it waits — which is what lets the popover say "read-only
applies after the running execution finishes" instead of showing a toggle that
appears to have done nothing.

A narrowing on a target the session is NOT on realigns nothing: the session's
connector has nothing to do with that machine, and the store is read again the
moment a switch lands there. A switch that happens while a realignment is
pending clears it for the same reason — the child the switch built read the
store on the way up.

**The switch request.** Addressed, not broadcast: the file is named for a
server PID, and its *body* names one too. A body naming another process is
dropped without acting on it, because a PID can be reused and honouring a
request written for a server that has since died would move a session on a
gesture nobody made in it. A request older than
:data:`~osprey.mcp_server.control_system.target_state.REQUEST_TTL_S` ends as
``request_expired``: the operator who clicked Switch is no longer watching, and
a switch that lands minutes after the gesture is a surprise rather than a
service. A request whose addressee *died* is never answered from here at all —
there is no process left to answer it — so the chip synthesises
``request_expired`` locally once the TTL has passed with no outcome, and
:func:`~osprey.mcp_server.control_system.target_state.sweep_stale` unlinks the
file when some later server starts.

What a status means to the chip
-------------------------------
``success`` renders as a tick and nothing else; ``refused`` and ``failed``
render as a cross beside their ``reason`` — the gate's own word, or the stage
the switch stopped at — and ``expired`` is the one whose reason is always
:data:`REASON_REQUEST_EXPIRED`, which is also the word the chip writes for
itself when nobody ever answered. So a reader has one rule: tick on
``success``, otherwise show ``reason``.

Everything else is the agent's own path, reached through the same functions:
:func:`~osprey.mcp_server.control_system.tools.control_target.switch_gate` for
the three refusals, evaluated IMMEDIATELY before the switch so the world it
read is the world the switch happens in, and
:meth:`~osprey.mcp_server.control_system.connector_host_manager.ConnectorHostManager.switch`
for the move itself — the same lock and the same equality guard the tool goes
through, so a reconciler and an agent aiming at the same target spawn one child
between them rather than two.

Every terminus, in one order
----------------------------
Success, each refusal, a failed switch and an expiry all end the same way:

1. :func:`~osprey.mcp_server.control_system.target_state.publish_last_switch`
   records the outcome under the ``request_id`` the chip is holding, and names
   the ``target`` it was aimed at — the chip matches on the request, but the
   popover's roster is one row per machine and renders the outcome on the row
   the request named;
2. the request file is removed;
3. the operator's activity feed is told (on everything that was *attempted* —
   an expired request never became an attempt);
4. one audit record is filed under the subject the web route and the agent's
   tool both use, so the ledger shows one kind of event whichever surface asked.

The order matters at step 2: the route reads the request file's presence to
refuse a second click, so publishing after removing it would leave a window in
which the chip finds neither a pending request nor an outcome.
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
from osprey_connectors import session_store

logger = logging.getLogger("osprey.mcp_server.control_system.session_control")

#: How often the loop looks. A second is the whole budget an operator will wait
#: for a button to answer, and two ``stat`` calls a second is what an unchanged
#: pair of files costs.
POLL_INTERVAL_S = 1.0

# -- terminus statuses ------------------------------------------------------
#
# The vocabulary the chip renders: ``success`` is a tick, everything else is
# the ``reason`` beside a cross. They are spelled here rather than in the state
# file's module because they are this reconciler's answers — the state file
# records what it is told and arbitrates none of it.

#: The session moved.
STATUS_SUCCESS = "success"
#: A gate said no; nothing was attempted.
STATUS_REFUSED = "refused"
#: An attempt was made and did not complete; the session is where it was.
STATUS_FAILED = "failed"
#: The request was reached later than its TTL and was never acted on.
STATUS_EXPIRED = "expired"

#: The reason an expired request carries — the word the chip renders and the
#: word the route's own refusal vocabulary already contains.
REASON_REQUEST_EXPIRED = "request_expired"

#: An exception the switch did not classify. Restated from
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

#: The audit subject for a gesture that moves the session's control target —
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
    "REASON_REQUEST_EXPIRED",
    "REASON_TARGET_SWITCHED",
    "STATUS_EXPIRED",
    "STATUS_FAILED",
    "STATUS_REFUSED",
    "STATUS_SUCCESS",
    "SessionControlReconciler",
]


def _signature(path: Path | None) -> tuple[int, int, int] | None:
    """``(mtime_ns, size, inode)`` for *path*, or ``None`` when it is absent.

    The inode is the third element on purpose: both files this task watches are
    replaced atomically, so two gestures inside one filesystem clock tick
    differ by inode even when mtime and size do not.
    """
    if path is None:
        return None
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


class SessionControlReconciler:
    """Reconciles the operator's desired session control state, once a second.

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

        # What the last pass saw. All four start unset, so the first pass
        # *baselines* rather than acting: a narrowing already in the store when
        # this server started was read by the child it started, and replaying
        # it as a change would rebuild a connector nobody narrowed.
        self._store_signature: tuple[int, int, int] | None = None
        self._request_signature: tuple[int, int, int] | None = None
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
        """One reconcile pass: the posture store, then the switch request.

        The posture half runs first so that a switch landing in the same pass
        is judged against a connector that already reflects the store.
        """
        context = self._context()
        if context is None:
            return
        await self._reconcile_posture(context)
        await self._reconcile_request(context)

    def _context(self) -> Any:
        """The server context, or ``None`` when there is not one yet.

        A poll before ``initialize_server_context()`` has run is not an error:
        the lifespan starts this task, and a context that cannot be read has no
        session to reconcile.
        """
        from osprey.mcp_server.control_system.server_context import get_server_context

        try:
            return get_server_context()
        except RuntimeError:
            return None

    # -- the posture store -------------------------------------------------

    async def _reconcile_posture(self, context: Any) -> None:
        """Realign the connector when the ACTIVE target's posture moved."""
        signature = _signature(session_store.store_path())
        try:
            target = context.connector_hosts.active_target()
        except Exception:
            logger.debug("No connector-host supervisor to read the session target from")
            return

        if signature != self._store_signature or target != self._active_target:
            self._store_signature = signature
            self._observe(target)

        if self._realign_pending:
            await self._realign(context)

    def _observe(self, target: str) -> None:
        """Record what the store now says about *target*, and whether it moved.

        A change of TARGET is not a change of posture: the child a switch built
        read the store on the way up, so the session is already aligned and a
        realignment left pending from the previous target is moot. A change of
        the entry for the target the session is still on is the one case that
        owes the operator a rebuild.
        """
        entry = session_store.target_posture(posture.posture_session(), target)
        if target != self._active_target:
            self._active_target = target
            self._active_posture = entry
            if self._realign_pending:
                # The session left the target that narrowing was about, on a
                # child that read the store itself. Nothing is outstanding.
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
        would rewrite the state file once a second for the length of a run to
        say nothing new.
        """
        if target_state.in_flight_executions():
            return
        try:
            rebuilt = await context.invalidate_connector("control_system")
        except Exception:
            logger.warning(
                "Could not realign the control-system connector to the session posture; "
                "retrying on the next pass",
                exc_info=True,
            )
            return
        if rebuilt is False:
            logger.warning(
                "The connector host refused to respawn, so the session posture is not "
                "realigned yet; retrying on the next pass"
            )
            return
        self._realign_pending = False
        self._publish_realign(REALIGN_DONE)

    def _publish_realign(self, state: str) -> None:
        """Publish a realignment note. Never costs the reconcile that made it."""
        try:
            target_state.publish_posture_realign({"state": state})
        except Exception:
            logger.warning("Could not publish the posture realignment note", exc_info=True)

    # -- the switch request ------------------------------------------------

    async def _reconcile_request(self, context: Any) -> None:
        """Consume the switch request addressed to this server, if there is one."""
        path = target_state.request_file_path()
        signature = _signature(path)
        if signature == self._request_signature:
            return
        self._request_signature = signature
        if signature is None:
            return

        record = target_state.read_request()
        if not isinstance(record, dict):
            # Named for this server and unreadable: this process's residue to
            # clear, and residue that would otherwise make the route refuse
            # every later click as ``request_pending``.
            logger.warning("Unreadable switch request at %s; removing it", path)
            self._forget_request()
            return

        if record.get("server_pid") != os.getpid():
            # Addressed to another process. Dropped without a terminus: the
            # outcome block belongs to whoever the request was written for, and
            # answering on their behalf would be this server's second opinion
            # about a gesture it never received.
            logger.warning(
                "Switch request %r names server_pid %r, not this server; dropping it",
                record.get("request_id"),
                record.get("server_pid"),
            )
            self._forget_request()
            return

        if not target_state.is_request_fresh(record):
            await self._finish(
                record,
                status=STATUS_EXPIRED,
                reason=REASON_REQUEST_EXPIRED,
                detail=(
                    f"The switch request was written more than {target_state.REQUEST_TTL_S}s "
                    "ago and was not acted on."
                ),
            )
            return

        # A request this server has decided to act on is a request it owes an
        # answer to, whatever happens next. The signature above has already
        # moved, so an exception escaping here would leave the file on disk and
        # never look at it again: no outcome published, the route refusing
        # every later click as ``request_pending``, and a chip spinning until
        # the operator gives up. The loop's own guard is too late to answer a
        # request it cannot see.
        try:
            await self._switch(context, record)
        except Exception as exc:
            logger.exception("Switch request %r failed unexpectedly", record.get("request_id"))
            await self._finish(
                record,
                status=STATUS_FAILED,
                reason=REASON_INTERNAL_ERROR,
                detail=f"{type(exc).__name__}: {exc}",
            )

    async def _switch(self, context: Any, record: dict[str, Any]) -> None:
        """Gate, then switch, then report — the agent's path with no agent.

        The gate is asked HERE rather than when the request was written: an
        execution that started in between, a posture that narrowed, a target
        that stopped being eligible all have to refuse this switch, and a
        verdict taken earlier would be a verdict about a different moment.
        """
        # Imported inside the call: ``tools.control_target`` imports the server
        # module this task's own lifespan lives in, so a module-level import
        # here would close an import cycle.
        from osprey.mcp_server.control_system.connector_host_manager import SwitchError
        from osprey.mcp_server.control_system.tools.control_target import switch_gate

        wanted = str(record.get("target") or "").strip()
        hosts = context.connector_hosts
        session_target = hosts.active_target()

        verdict = await switch_gate(context, wanted)
        if verdict is not None:
            await self._finish(
                record,
                status=STATUS_REFUSED,
                reason=verdict.reason,
                detail=verdict.detail,
                from_target=session_target,
            )
            return

        try:
            result = await hosts.switch(wanted)
        except SwitchError as exc:
            logger.warning(
                "Operator-requested switch to %r failed at stage %r: %s",
                wanted,
                exc.stage,
                exc.detail,
            )
            await self._finish(
                record,
                status=STATUS_FAILED,
                reason=exc.reason,
                detail=exc.detail,
                from_target=session_target,
            )
            return

        logger.info(
            "Operator-requested switch: session target is now %r (generation %s)",
            result["target"],
            result["generation"],
        )
        await self._finish(
            record,
            status=STATUS_SUCCESS,
            reason=None,
            detail=(
                f"Control-system target is now {result['target']!r} "
                f"(generation {result['generation']})."
            ),
            from_target=result["previous_target"],
            to_target=result["target"],
            generation=result["generation"],
        )

    # -- the terminus ------------------------------------------------------

    async def _finish(
        self,
        record: dict[str, Any],
        *,
        status: str,
        reason: str | None,
        detail: str,
        from_target: str | None = None,
        to_target: str | None = None,
        generation: int | None = None,
    ) -> None:
        """End one request: publish, consume, report, record. In that order.

        Publication comes first because the route reads the request file's
        presence to refuse a second click — removing it before the outcome
        existed would leave a window in which the chip finds neither a pending
        request nor an answer to the one it sent.
        """
        request_id = str(record.get("request_id") or "")
        wanted = to_target or str(record.get("target") or "")
        requested_by = str(record.get("requested_by") or "")

        try:
            target_state.publish_last_switch(
                {
                    "request_id": request_id,
                    "target": wanted,
                    "status": status,
                    "reason": reason,
                    "detail": detail,
                }
            )
        except Exception:
            logger.warning("Could not publish the switch outcome for %r", request_id, exc_info=True)

        self._forget_request()

        # The ledger record is filed BEFORE the awaited emit, and not after it:
        # every await is a place this task can be cancelled, and a shutdown
        # landing between the two would leave a switch that happened with no
        # record that it did. The activity emit is a convenience for whoever is
        # watching the session; the ledger is the trail.
        self._record_gesture(
            status=status,
            reason=reason,
            request_id=request_id,
            target=wanted,
            requested_by=requested_by,
        )

        # Nothing was attempted for an expired request, so there is no switch
        # for the operator's activity feed to report. Every other terminus is
        # an attempt somebody watching the session needs to see the end of.
        if status != STATUS_EXPIRED:
            await self._notify(
                from_target=from_target or "unknown",
                to_target=wanted,
                status=status,
                reason=reason,
                generation=generation,
            )

    def _forget_request(self) -> None:
        """Consume the request file and forget its signature."""
        target_state.remove_request()
        self._request_signature = _signature(target_state.request_file_path())

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
            if status == STATUS_SUCCESS:
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
                decision=DECISION_ALLOWED if status == STATUS_SUCCESS else DECISION_REFUSED,
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
