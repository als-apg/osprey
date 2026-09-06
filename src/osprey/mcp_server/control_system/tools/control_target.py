"""MCP tools: the control-system target roster, and the switch that moves it.

``control_target`` reports; ``control_target_set`` acts. They live together
because they answer the same question from the same functions — a roster that
said a target was available while the switch refused it, or that named a
different reason, would be worse than no roster at all — so the eligibility
call, the display metadata and the manager accessors are shared here rather
than restated twice.

The roster
----------
Side-effect-free by construction: it derives endpoints from config, reads the
endpoint prober's cache, and asks the manager what it already knows. It opens
no socket, spawns no child and writes no state. That matters beyond tidiness —
a tool that had to act in order to report would make "what would happen" and
"make it happen" the same call, and the roster exists precisely so an operator
can ask the first without the second.

It is therefore correct BEFORE anything has been switched: a target nobody has
ever activated is judged from configuration alone, and reports
``available_now`` on that basis. Reachability is the part it cannot know from
config, so a row carries ``endpoint_tcp`` only where the background prober has
actually measured one; a deployment whose prober never started reports rows
without it rather than a guess.

The switch
----------
The target is the DEPLOYMENT's, held in one control-context record, and only
that record's owner writes it. So this tool no longer switches anything: it
asks for a switch and reports what the record says came of it.

Two paths, one answer. When this server owns the record it takes the verdict
from :func:`~osprey.mcp_server.control_system.target_eligibility.evaluate_switch`
and writes the terminus — a refusal, or the new target with the minted
generation — in one atomic replace. When it does not, it files a request file
named for this process and polls the record until ``last_switch.request_id`` is
its own; the owner runs the very same gate and writes the very same terminus.
Neither path mints a generation for a request naming the target of record: that
is answered where it stands, and a generation bumped for a switch that did not
happen would refuse every write bound to the old one for nothing.

The gate itself lives in :mod:`target_eligibility`, context-free, because the
operator's Switch on the web terminal has to get the same answer from a process
that owns no manager. Its ladder — a read-only run, an execution in flight,
eligibility, then reachability — is documented there, and the tool's whole job
with a refusal is to report it in the words the gate chose.

What is left for this tool to wait for is its own connector host. The record
moving is the deployment's answer; the reconcile loop is what brings this
server's child to it, and until that lands the session's tools are still
talking to the previous target. So an applied terminus is followed by a bounded
wait on this server's own report — bounded by the spawn, probe and drain
timeouts this process holds, which is why the bound is computed here and never
guessed by a reader.

A deployment that has not settled refuses rather than queues: while any live
server is still applying a generation, a second terminus would overwrite the
answer to the gesture somebody is still watching for. That refusal names the
pids so an operator can see which server to wait for.

The in-flight marker contract
-----------------------------
The executor is a separate MCP server process, so "is an execution running" has
to be asked across a process boundary. It is asked through the file system, in
the directory the two already share — :func:`target_state.state_dir` — because
the executor is already a reader of that directory and needs no new path rule.

A marker is ``exec_inflight_<pid>_<run id>.json``, written before the sandbox
subprocess starts and removed in a ``finally``. It carries the writing
process's PID, so a marker left behind by an executor that was killed is
ignored (and swept) rather than wedging every future switch: a PID that names
no live process cannot be running anything.

The reader's half — the constants and :func:`in_flight_executions` — lives in
:mod:`osprey.mcp_server.control_system.target_state`, beside the directory it
names, because this tool is no longer its only reader: the session-control
reconciler asks the same question before an operator-initiated switch, and the
posture route asks it before widening a posture out from under a running
execution. The names are re-exported here so every existing importer keeps
working.

The constants and the record shape are still stated **twice** — in
``target_state`` and in :mod:`osprey.mcp_server.python_executor.executor` — and
pinned equal by ``tests/mcp_server/test_control_target_set.py``. That is the
replica pattern this repository already uses for the deployed hooks: the
alternative is for one MCP server process to import the other's module at run
time, which would drag the whole controls server into the executor (or the
reverse) for two string constants.

"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any, NoReturn

from osprey.audit.posture import posture_session
from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import target_display_metadata
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.control_system.target_eligibility import (
    REASON_EXECUTION_IN_FLIGHT,
    REASON_READONLY_RUN,
    GateVerdict,
    derive_endpoints,
    effective_writes_for_target,
    evaluate_switch,
    target_availability,
)
from osprey.mcp_server.control_system.target_state import (
    INFLIGHT_FILE_GLOB,
    INFLIGHT_FILE_PREFIX,
    INFLIGHT_FILE_SUFFIX,
    in_flight_executions,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import (
    SWITCH_OUTCOME_FAILURE,
    SWITCH_OUTCOME_SUCCESS,
    notify_target_switch_async,
)
from osprey_connectors import control_context
from osprey_connectors.types import configured_targets, target_limits_posture

logger = logging.getLogger("osprey.mcp_server.tools.control_target")

__all__ = [
    "INFLIGHT_FILE_GLOB",
    "INFLIGHT_FILE_PREFIX",
    "INFLIGHT_FILE_SUFFIX",
    "REASON_EXECUTION_IN_FLIGHT",
    "REASON_READONLY_RUN",
    "GateVerdict",
    "control_target",
    "control_target_set",
    "in_flight_executions",
    "target_rows",
]

# The in-flight marker names and reader are re-exported from ``target_state``
# (imported above and named in ``__all__``); see the module docstring for why
# they moved. Importers of this module — including the drift guard that pins the
# spelling against the executor's replica — are unaffected. ``GateVerdict`` and
# the two refusal reasons the gate adds to eligibility's own are re-exported
# from :mod:`target_eligibility` for the same reason: the gate moved out of this
# module, and its vocabulary moved with it.

# -- machine-readable refusal reasons ---------------------------------------

#: The server context this tool reads its deployment from is not initialized.
REASON_CONTEXT_UNAVAILABLE = "context_unavailable"
#: No control-context record: nothing holds this deployment's target.
REASON_RECORD_UNAVAILABLE = "control_context_unavailable"
#: A live server is still applying the generation the deployment is on, so no
#: answer can be written without overwriting the one it will produce.
REASON_SWITCH_IN_PROGRESS = "switch_in_progress"
#: A request was filed and no owner took it. The record's owner is named, so an
#: operator can tell "nobody owns this deployment" from "the owner is wedged".
REASON_REQUEST_NOT_CONSUMED = "request_not_consumed"
#: The switch was granted and this server's connector host did not reach it
#: within the bound its own spawn, probe and drain timeouts imply.
REASON_SWAP_INCOMPLETE = "swap_incomplete"
#: A second request from this same process occupies its request slot.
REASON_REQUEST_PENDING = "request_pending"
#: The control-target directory could not be written at all.
REASON_STORE_UNAVAILABLE = "store_unavailable"
#: An exception the switch did not classify. Reported as a failure rather than
#: swallowed: the operator approved an attempt, and an attempt that ended in a
#: way nobody anticipated is still an attempt that ended.
REASON_INTERNAL_ERROR = "internal_error"

#: Stands in for the control target when it cannot be read at all — the context
#: is what holds it, so a context failure is exactly the case that has no answer.
#: Spelled rather than omitted: the operator's line still has to say something.
UNKNOWN_TARGET = "unknown"

#: Error envelopes. A refusal is a gate saying no; a failure is a switch that
#: was attempted and did not complete, with the previous target still active.
ERROR_REFUSED = "target_switch_refused"
ERROR_FAILED = "target_switch_failed"
ERROR_UNAVAILABLE = "target_switch_unavailable"


# ---------------------------------------------------------------------------
# Shared helpers (the roster tool lands in this module too)
# ---------------------------------------------------------------------------


def _server_context() -> Any:
    """The controls server's context singleton, or ``None`` if it has none.

    Returns rather than refuses, because the two tools in this module owe the
    operator different things for the same failure: the switch reports a
    declined attempt, the roster reports a session it cannot describe.
    """
    from osprey.mcp_server.control_system.server_context import get_server_context

    try:
        return get_server_context()
    except RuntimeError as exc:
        logger.warning("The control-system server context is not initialized: %s", exc)
        return None


def _context_unavailable_message() -> str:
    return (
        "The control-system server context is not initialized, so this deployment has no "
        "target of record to read or change."
    )


async def _emit_failure(from_target: str, to_target: str, reason: str) -> None:
    """Tell the operator that an approved switch attempt did not happen.

    Every way this tool can decline or fail goes through here, because the
    operator's view has to be the same shape whichever check declined it: an
    approved attempt that then does not happen is exactly the event somebody
    watching the session needs to see. The emitter never raises and does not
    block, so it sits inline ahead of the error.
    """
    await notify_target_switch_async(
        from_target=from_target,
        to_target=to_target,
        outcome=SWITCH_OUTCOME_FAILURE,
        reason=reason,
    )


async def _refuse(
    *,
    from_target: str,
    to_target: str,
    message: str,
    suggestions: list[str],
    details: dict[str, Any],
    error_type: str = ERROR_REFUSED,
    notify: bool = True,
) -> NoReturn:
    """Report the refusal to the operator, then raise it to the agent.

    *notify* is false for exactly one case: a refusal this process is relaying
    rather than making. The owner that wrote the terminus into the record
    already emitted the operator's line, and a second one for the same gesture
    would read as a second gesture.
    """
    if notify:
        await _emit_failure(from_target, to_target, str(details.get("reason") or ""))
    make_error(error_type, message, suggestions, details=details)


# ---------------------------------------------------------------------------
# The roster
# ---------------------------------------------------------------------------


def _endpoint_rows(derivation: Any, probe_rows: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """One row per configured gateway role: where it points, and how it answered.

    The derived half (host, port, mode) is always present — it is config, and
    config is knowable without touching anything. The measured half
    (``endpoint_tcp``, ``probed_at``, staleness) is folded in ONLY for a role
    the prober has actually measured. A role with no measurement carries no
    ``endpoint_tcp`` key at all rather than a placeholder: "not measured" and
    "measured as down" are different claims, and a roster that spelled them the
    same way would be the roster lying about the one thing it is for.
    """
    rows: dict[str, dict[str, Any]] = {}
    for role, endpoint in derivation.endpoints.items():
        row: dict[str, Any] = dict(endpoint.as_dict())
        measured = probe_rows.get(role)
        if isinstance(measured, dict):
            row.update(measured)
        rows[role] = row
    return rows


def target_rows(
    config: Any,
    *,
    control_target: str,
    baseline: str,
    probe_snapshot: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """The per-target roster rows, from config and measurements alone.

    Pure: no process is spawned, no socket is opened, nothing is written. Every
    verdict comes from :mod:`target_eligibility` — the same functions the switch
    consults — so a row's ``reason`` is character-for-character the reason a
    refusal would carry.

    Args:
        config: The full rendered config mapping.
        control_target: The target the deployment is on right now.
        baseline: The target this deployment's own config selects.
        probe_snapshot: :meth:`EndpointProber.snapshot`'s output, or ``None``
            when no prober is running — in which case rows carry the derived
            endpoints and no reachability at all.

    Returns:
        ``{target: row}`` for every target this deployment configures
        (:func:`~osprey_connectors.types.configured_targets`), in that
        function's order. A target this config never described — most often
        ``standin`` on a deployment that stands up no soft IOC — has no row at
        all, rather than a row saying a machine nobody deployed is unavailable.
    """
    metadata = target_display_metadata(config)
    snapshot = probe_snapshot or {}

    rows: dict[str, dict[str, Any]] = {}
    # The configured targets and never CONTROL_TARGETS: the constant is the
    # vocabulary of machines that can exist, and looping it would hand a
    # deployment with no `control_system.connector.live_standin` block a
    # `standin` row describing a soft IOC nobody stood up. `configured_targets`
    # takes the section rather than the whole config, and answers a missing or
    # malformed one with the baseline alone.
    section = config.get("control_system") if isinstance(config, dict) else None
    for target in configured_targets(section):
        # The deployment's real posture for the target, read ONCE and then used
        # for every answer this row carries. The eligibility verdict, the
        # gateway role and the `writes_permitted` flag are three views of the
        # same fact, and a row that read the store separately for each could
        # report a role the operator narrowed away beside a flag saying they
        # had not.
        writes_permitted = _writes_permitted(config, target)
        availability = target_availability(
            config, target, control_target, baseline, writes_enabled=writes_permitted
        )
        display = metadata.get(target, {})
        row: dict[str, Any] = {
            "target": target,
            "active": target == control_target,
            "is_baseline": target == baseline,
            "label": display.get("label", ""),
            "real_machine": bool(display.get("real_machine", False)),
            "available_now": availability.available_now,
            "reason": availability.reason,
            "detail": availability.detail,
            "eligible": availability.eligible,
            "eligible_from_baseline": availability.eligible_from_baseline,
            # This target's own write posture, carried on every row so a reader
            # never has to pair a row with a flag from somewhere else. Rows of
            # one deployment may differ: posture is per connector type, so a
            # simulator can be armed beside a live machine that is not. The
            # gateway those writes would leave by is `selected_role`.
            "writes_permitted": writes_permitted,
            # This target's own limits posture, per connector type for the same
            # reason the write posture is: a deployment may relax unlisted
            # channels on its simulator while its live machine refuses them.
            # Strict means limits checking on and unlisted channels explicitly
            # refused; a target whose config states neither is not strict,
            # because a deployment that stated nothing has refused nothing.
            # Unlike `writes_permitted`, this is a deployment fact, not a
            # session one: the store narrows what a session may write, never
            # which channels a target's limits database governs.
            "limits_strict": target_limits_posture(section, target).strict,
        }
        probe_channel = display.get("probe_channel") or ""
        if probe_channel:
            row["probe_channel"] = probe_channel

        try:
            derivation = derive_endpoints(config, target, writes_enabled=writes_permitted)
        except ValueError:
            # An underivable target has no connector type and no endpoints —
            # the availability verdict above already says so, with the reason.
            row["endpoints"] = {}
            rows[target] = row
            continue

        row["connector_type"] = derivation.connector_type
        row["selected_role"] = derivation.selected_role
        row["endpoints"] = _endpoint_rows(derivation, snapshot.get(target) or {})
        rows[target] = row
    return rows


def _writes_permitted(config: Any, target: str) -> bool:
    """Whether a write to *target* would be permitted on this deployment, now.

    Three things decide it: that target's own posture
    (``control_system.connector.<type>.writes_enabled``, falling back to
    ``control_system.writes_enabled`` where its type states none), this run's
    own claim (``OSPREY_EXECUTION_MODE``), and the operator's narrowing from
    the header chip. All three are combined by
    :func:`~osprey.mcp_server.control_system.target_eligibility.effective_writes_for_target`,
    which is also the value the roster hands
    :func:`~osprey.mcp_server.control_system.target_eligibility.derive_endpoints`
    — so the flag a row reports and the gateway that row names are the same
    answer rather than two readings that could drift apart.
    """
    section = config.get("control_system") if isinstance(config, dict) else None
    return effective_writes_for_target(section, target)


@mcp.tool()
async def control_target() -> str:
    """Report which control system this deployment is pointed at, and what else it could be.

    Read-only and side-effect-free: nothing is spawned, connected to or
    written. Ask this before proposing a switch — it says, per target, whether
    the session may move there right now and why not if it may not, in the same
    words the switch itself would use.

    Each target row carries: the connector type and the gateway role this
    deployment would select; a per-role endpoint table with the host, port and
    routing mode derived from config, plus the background prober's last
    reachability observation where it has one (``endpoint_tcp``, ``probed_at``,
    and ``stale`` once an observation is too old to stand); whether writes are
    permitted on that target, which is a per-target answer and not one flag for
    the deployment; whether that target's limits posture is strict
    (``limits_strict`` — limits checking on and channels the limits database
    does not list refused), which is per-target for the same reason; whether
    the target is the real machine; and the channel a switch would read to
    prove the target is reachable.

    A target nobody has activated yet is described from configuration alone —
    that is what makes this answer correct before any switch has happened.
    ``endpoint_tcp`` is absent where nothing has measured it, and
    ``not_applicable`` where the gateway is reached over an address list (CA
    search is UDP there, so a TCP probe could prove nothing).

    The roster is the targets this deployment *configures*, not every target
    OSPREY names. A machine this config never described — most often
    ``standin`` on a deployment that stands no soft IOC up — has no row here at
    all, rather than a row reporting a machine nobody deployed as unavailable.
    So an absent row means "no such target here", and only a row that is
    present carries a reason a switch would refuse with.

    Returns:
        JSON with the active target and generation, and one row per configured
        target.
    """
    context = _server_context()
    if context is None:
        return make_error(
            ERROR_UNAVAILABLE,
            _context_unavailable_message(),
            ["Restart the controls MCP server; no target state can be read until it starts."],
            details={"reason": REASON_CONTEXT_UNAVAILABLE},
        )

    hosts = context.connector_hosts
    status = hosts.status()

    from osprey.mcp_server.control_system.server import get_endpoint_prober

    prober = get_endpoint_prober()
    snapshot = prober.snapshot() if prober is not None else None

    rows = target_rows(
        context.config.raw,
        control_target=status["target"],
        baseline=status["baseline_target"],
        probe_snapshot=snapshot,
    )

    return json.dumps(
        {
            "status": "success",
            "description": (
                f"The deployment is on the {status['target']!r} target "
                f"(generation {status['generation']}); "
                f"deployment baseline is {status['baseline_target']!r}."
            ),
            "summary": {
                "target": status["target"],
                "generation": status["generation"],
                "baseline_target": status["baseline_target"],
                "connector_host_alive": status["child_alive"],
                "switchable_targets": sorted(
                    name for name, row in rows.items() if row["available_now"]
                ),
            },
            "access_details": {
                "targets": rows,
                "endpoint_probe": {
                    "running": prober is not None,
                    "probe_interval_s": getattr(prober, "probe_interval_s", None),
                    "staleness_threshold_s": getattr(prober, "staleness_threshold_s", None),
                    # Said plainly rather than left to be inferred from absent
                    # keys: with no prober, every row is config-only.
                    "detail": (
                        ""
                        if prober is not None
                        else "No endpoint prober is running, so no row carries a measured "
                        "reachability status."
                    ),
                },
            },
        },
        default=str,
    )


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

#: One switch per server process, held from the moment a request is filed to
#: the moment its outcome is known. Two agent calls in one process would
#: otherwise write into the same request slot — the second replacing the first,
#: leaving the first watching for a ``request_id`` no file carries any more —
#: and would then poll one record for two different answers. The second caller
#: waits and re-reads: by the time it runs the deployment may already be where
#: it wanted to go, which is the no-mint answer rather than a second switch.
_SWITCH_LOCK = asyncio.Lock()

#: How often the record is re-read while an answer is outstanding. The owner
#: reconciles once a second, so a request cannot be answered more often than
#: that and a tighter poll would only re-read the same bytes.
POLL_INTERVAL_S = 1.0

#: How many owner ticks a filed request may go unanswered before this tool
#: stops waiting. Well inside the request TTL, so a request withdrawn here was
#: never going to be swept for staleness first.
OWNER_TICKS_BEFORE_UNCONSUMED = 5


def _now_iso() -> str:
    """Wall clock, ISO-8601, UTC — the stamp a request carries."""
    return datetime.now(UTC).isoformat()


def _no_record_message() -> str:
    return (
        "This deployment has no control-context record, so there is no target of record to "
        "read or change."
    )


def _gate(
    context: Any, record: control_context.ControlContext, wanted: str, reports: Any
) -> GateVerdict:
    """The switch verdict, from the one gate every surface asks.

    The inputs are gathered here because the gate opens no file: the record's
    target rather than this manager's, because the answer is written into the
    record and it is the record the deployment routes by; and the deployment's
    effective write posture, so a target the operator narrowed is not refused
    for a gateway role the child would never have selected.
    """
    return evaluate_switch(
        context.config.raw,
        wanted,
        current_target=record.target,
        baseline=context.connector_hosts.baseline,
        in_flight=in_flight_executions(),
        reports=reports,
        writes_enabled=_writes_permitted(context.config.raw, wanted),
    )


async def _refuse_switch_in_progress(
    record: control_context.ControlContext, wanted: str, reports: Any
) -> NoReturn:
    """Today's refusal shape for a deployment that has not settled yet.

    No record is written and no request is left behind: a terminus written now
    would overwrite the answer to the gesture still in flight, which is the one
    somebody is actually waiting for.
    """
    pids = control_context.blocking_pids(record, reports, None)
    named = ",".join(str(pid) for pid in pids) or "an unnamed server"
    await _refuse(
        from_target=record.target,
        to_target=wanted,
        message=(
            f"switch_in_progress:{named}. A control-target switch is already in flight on "
            f"pid {named}, so this one was not filed."
        ),
        suggestions=[
            "Nothing was changed: the deployment is still on the target it was on.",
            "Wait for the control-target chip to settle, then ask for the switch again.",
        ],
        details={
            "target": wanted,
            "reason": REASON_SWITCH_IN_PROGRESS,
            "pids": list(pids),
        },
    )


def _terminus(
    record: control_context.ControlContext,
    *,
    request_id: str,
    wanted: str,
    requested_at: str,
    status: str,
    reason: str | None,
    detail: str,
    generation: int | None,
) -> control_context.ControlContext:
    """The record as it will be once this request has been answered.

    The block itself is
    :func:`~osprey_connectors.control_context.terminus`; what this server adds
    is who asked. It is answering its own agent, so the requester is this
    session — or this process, for a run that reports no session.
    """
    return control_context.terminus(
        record,
        request_id=request_id,
        target=wanted,
        requested_at=requested_at,
        requested_by=posture_session() or f"pid:{os.getpid()}",
        status=status,
        reason=reason,
        detail=detail,
        generation=generation,
    )


async def _record_write_failed(from_target: str, wanted: str) -> NoReturn:
    """The answer when the record would not take this server's write."""
    await _refuse(
        from_target=from_target,
        to_target=wanted,
        message="The control-context record could not be updated, so the target was not changed.",
        suggestions=[
            "Nothing was changed: the deployment is still on the target it was on.",
            "Check that the agent-data directory is writable, then ask for the switch again.",
        ],
        details={"target": wanted, "reason": REASON_INTERNAL_ERROR},
        error_type=ERROR_FAILED,
    )


async def _apply_here(
    context: Any, record: control_context.ControlContext, wanted: str, reports: Any
) -> str:
    """The owning server's own answer: gate it, then write the terminus.

    This server owns the record, so there is nobody to ask and nothing to wait
    for — the verdict is taken and the answer written in one place, in exactly
    the shape a consumed request file would have produced. The wait that
    follows is for this server's *own* connector host to reach the generation
    the record now names, which is a different question from whether the switch
    was granted.
    """
    if not control_context.converged(record, reports, None):
        return await _refuse_switch_in_progress(record, wanted, reports)

    request_id = uuid.uuid4().hex
    requested_at = _now_iso()
    verdict = _gate(context, record, wanted, reports)

    if not verdict.allowed:
        control_context.write_terminus(
            _terminus(
                record,
                request_id=request_id,
                wanted=wanted,
                requested_at=requested_at,
                status=control_context.SWITCH_REFUSED,
                reason=str(verdict.reason or ""),
                detail=verdict.detail,
                generation=None,
            ),
            request_id,
        )
        return await _refuse(
            from_target=record.target,
            to_target=wanted,
            message=verdict.detail,
            suggestions=verdict.suggestions,
            details=verdict.details,
        )

    generation = record.generation + 1
    if not control_context.write_terminus(
        _terminus(
            record,
            request_id=request_id,
            wanted=wanted,
            requested_at=requested_at,
            status=control_context.SWITCH_APPLIED,
            reason=None,
            detail=control_context.applied_detail(wanted, generation),
            generation=generation,
        ),
        request_id,
    ):
        return await _record_write_failed(record.target, wanted)

    return await _await_the_swap(context, wanted, generation, previous_target=record.target)


async def _file_request(record: control_context.ControlContext, wanted: str) -> str:
    """Ask the owner for the switch, and return the id it was filed under.

    The body is the whole of what a consumer reads, and the file is named for
    THIS process: a request survives exactly as long as the session waiting for
    its answer, and is swept with it.
    """
    request_id = uuid.uuid4().hex
    try:
        target_state.write_request(
            {
                "request_id": request_id,
                "target": wanted,
                "requested_at": _now_iso(),
                "session": posture_session(),
                "requested_by_pid": os.getpid(),
            }
        )
    except target_state.RequestSuperseded:
        return await _refuse(
            from_target=record.target,
            to_target=wanted,
            message=(
                "Another switch request from this server is already pending, so this one was "
                "not filed."
            ),
            suggestions=["Wait for the pending switch to be answered, then ask again."],
            details={"target": wanted, "reason": REASON_REQUEST_PENDING},
        )
    except OSError as exc:
        logger.warning("Could not file a switch request for %r: %s", wanted, exc)
        return await _refuse(
            from_target=record.target,
            to_target=wanted,
            message=(
                "The control-target directory could not be written, so the switch could not be "
                "asked for."
            ),
            suggestions=["Check that the agent-data directory is writable, then ask again."],
            details={"target": wanted, "reason": REASON_STORE_UNAVAILABLE},
            error_type=ERROR_UNAVAILABLE,
        )
    return request_id


async def _request_and_wait(
    context: Any, record: control_context.ControlContext, wanted: str, reports: Any
) -> str:
    """File a request with the owner, then wait for the record to answer it.

    The gate is asked here as well as by the owner, and deliberately: the
    read-only rung is a claim about THIS run, and the process that made it is
    the only one that can see it — an owner in another process would evaluate
    its own execution mode and let the switch through. Everything refused here
    is refused again by the owner a moment later; what the local call buys is
    that the answer is about the run that asked.
    """
    verdict = _gate(context, record, wanted, reports)
    if not verdict.allowed:
        # A follower writes nothing to the record: this is this process's
        # answer to its own agent, not a terminus for a gesture the owner never
        # saw.
        return await _refuse(
            from_target=record.target,
            to_target=wanted,
            message=verdict.detail,
            suggestions=verdict.suggestions,
            details=verdict.details,
        )

    return await _poll_for_the_terminus(
        context, record, wanted, await _file_request(record, wanted)
    )


async def _poll_for_the_terminus(
    context: Any, filed_against: control_context.ControlContext, wanted: str, request_id: str
) -> str:
    """Wait for the record to carry this request's own answer.

    The record is polled and the request file is not: the owner unlinks a
    request only after its record write has been read back, so an absent file
    with no terminus means the request was dropped, not that it was applied.

    Two ways this ends without an answer, and both withdraw the request rather
    than leave it for an owner to apply into a session that has stopped
    watching. The deployment stops being converged — a switch somebody else
    asked for landed first — or no owner takes this one within
    :data:`OWNER_TICKS_BEFORE_UNCONSUMED` ticks, which names the process that
    should have consumed it.
    """
    previous_target = filed_against.target
    for _ in range(OWNER_TICKS_BEFORE_UNCONSUMED):
        record = control_context.read_record() or filed_against
        answer = await _answered(context, record, wanted, request_id, previous_target)
        if answer is not None:
            return answer

        reports = control_context.live_reports()
        if not control_context.converged(record, reports, None):
            target_state.remove_request()
            return await _refuse_switch_in_progress(record, wanted, reports)

        await asyncio.sleep(POLL_INTERVAL_S)

    record = control_context.read_record() or filed_against
    answer = await _answered(context, record, wanted, request_id, previous_target)
    if answer is not None:
        return answer

    target_state.remove_request()
    owner_pid = record.owner.pid if record.owner is not None else None
    named = "no process" if owner_pid is None else f"pid {owner_pid}"
    return await _refuse(
        from_target=record.target,
        to_target=wanted,
        message=(
            f"no owner consumed the request within {OWNER_TICKS_BEFORE_UNCONSUMED} ticks; the "
            f"control context is owned by {named}, and the target was not changed."
        ),
        suggestions=[
            "Nothing was changed: the deployment is still on the target it was on.",
            f"Check that {named} is still serving this deployment, then ask for the switch again.",
        ],
        details={
            "target": wanted,
            "reason": REASON_REQUEST_NOT_CONSUMED,
            "owner_pid": owner_pid,
        },
    )


async def _answered(
    context: Any,
    record: control_context.ControlContext,
    wanted: str,
    request_id: str,
    previous_target: str,
) -> str | None:
    """This request's terminus, reported — or ``None`` while there is none yet.

    The refusal is the owner's gate verdict relayed rather than re-derived: it
    was taken at the moment the switch would have happened, and this process's
    own facts are a moment older. It reaches the agent without an activity line
    of its own, because the owner that wrote the terminus already emitted one
    and two lines for one gesture read as two gestures.
    """
    last_switch = record.last_switch or {}
    if last_switch.get("request_id") != request_id:
        return None

    status = str(last_switch.get("status") or "")
    if status == control_context.SWITCH_REFUSED:
        reason = str(last_switch.get("reason") or "")
        detail = str(last_switch.get("detail") or "")
        return await _refuse(
            from_target=record.target,
            to_target=wanted,
            message=detail or f"the switch to {wanted!r} was refused.",
            suggestions=[
                "Ask for the target roster to see what each target would need to become usable."
            ],
            details={"target": wanted, "reason": reason},
            notify=False,
        )

    generation = last_switch.get("generation")
    if not isinstance(generation, int):
        generation = record.generation
    return await _await_the_swap(
        context,
        wanted,
        generation,
        previous_target=previous_target,
        notify_success=False,
    )


def _report_block(generation: int) -> dict[str, Any]:
    """This server's own progress through the generation it is reaching for.

    ``{}`` for a report that says nothing about this generation: a block for an
    older one is not evidence about this swap, and a reader that treated it as
    one would report the previous switch's outcome for this one.
    """
    try:
        report = control_context.read_report(target_state.report_file_path())
    except Exception:
        logger.debug("Could not read this server's report", exc_info=True)
        return {}
    block = getattr(report, "last_switch", None)
    if not isinstance(block, dict) or block.get("generation") != generation:
        return {}
    return block


async def _await_the_swap(
    context: Any,
    wanted: str,
    generation: int,
    *,
    previous_target: str,
    notify_success: bool = True,
) -> str:
    """Wait for THIS server to reach the generation the record now names.

    The record moving is the deployment's answer; it is not yet this server's
    connector. The reconcile loop picks the new generation up within a tick and
    publishes ``applying`` before its first await, and the swap that follows is
    bounded by the spawn, probe and drain timeouts this process holds — which
    is why the bound is computed here rather than guessed by a reader.

    A server with no live child adopts silently and publishes nothing at all,
    so "the manager is on the wanted binding and nothing says otherwise" is a
    landing too. Without that, a deployment that has never launched a connector
    host would wait out the whole bound for a report that was never coming.
    """
    hosts = context.connector_hosts
    deadline = time.monotonic() + hosts.applying_bound_s()
    while True:
        block = _report_block(generation)
        status = str(block.get("status") or "")
        if status == target_state.SWITCH_APPLIED:
            break
        if status == target_state.SWITCH_FAILED:
            return await _swap_failed(previous_target, wanted, generation, block)
        if (
            status != target_state.SWITCH_APPLYING
            and hosts.active_target() == wanted
            and hosts.active_generation() >= generation
        ):
            break
        if time.monotonic() >= deadline:
            return await _swap_failed(previous_target, wanted, generation, block, incomplete=True)
        await asyncio.sleep(POLL_INTERVAL_S)

    if notify_success:
        await notify_target_switch_async(
            from_target=previous_target,
            to_target=wanted,
            outcome=SWITCH_OUTCOME_SUCCESS,
            generation=generation,
        )
    logger.info("Control target is now %r (generation %s)", wanted, generation)
    return _binding_payload(
        hosts,
        wanted,
        generation,
        previous_target,
        description=f"Control-system target is now {wanted!r} (generation {generation}).",
    )


async def _swap_failed(
    previous_target: str,
    wanted: str,
    generation: int,
    block: dict[str, Any],
    *,
    incomplete: bool = False,
) -> NoReturn:
    """This server did not reach the generation the deployment is on.

    Reported as a failure and not as a refusal: the switch was granted, the
    record moved, and it is this server's connector host that did not follow.
    The record is left exactly as it is — the deployment's target of record is
    not this server's to revoke, and the next reconcile tick tries again.
    """
    reason = str(block.get("reason") or "") or REASON_SWAP_INCOMPLETE
    detail = str(block.get("detail") or "")
    if incomplete or not detail:
        detail = (
            f"the swap did not complete: this server has not reached target {wanted!r} "
            f"(generation {generation})."
        )
    await _emit_failure(previous_target, wanted, reason)
    make_error(
        ERROR_FAILED,
        detail,
        [
            "The deployment's target of record has moved; this server's connector host has "
            "not followed it.",
            "Check this server's log for the launch failure, then ask for the switch again.",
        ],
        details={"target": wanted, "reason": reason, "generation": generation},
    )


def _binding_payload(
    hosts: Any, target: str, generation: int, previous_target: str, *, description: str
) -> str:
    """The success answer, from what this server can honestly report.

    Everything here is read after the swap rather than returned by it: this
    tool no longer performs the switch, so the child, its connector type and
    the channel that proved it are the running host's own report of itself.
    """
    status = hosts.status()
    return json.dumps(
        {
            "status": "success",
            "description": description,
            "summary": {
                "target": target,
                "generation": generation,
                "previous_target": previous_target,
                "target_changed": target != previous_target,
                "connector_type": status["connector_type"],
                "probe_channel": status["probe_channel"],
            },
            "access_details": {
                "selected_role": status["selected_role"],
                "baseline_target": status["baseline_target"],
                "child_pid": status["child_pid"],
                "connector_host_alive": status["child_alive"],
                "drain_timeout_s": status["drain_timeout_s"],
            },
        },
        default=str,
    )


def _already_there(record: control_context.ControlContext, hosts: Any) -> str:
    """The no-mint answer: the deployment is already where the caller asked.

    The same answer the owner writes for a request naming the target of record,
    and it has to be the same one: a generation bumped for a switch that did
    not happen would refuse every write bound to the old one, for nothing.
    Nothing is emitted to the operator's feed either — a line for a switch that
    did not happen is a switch in the feed that did not happen.
    """
    return _binding_payload(
        hosts,
        record.target,
        record.generation,
        record.target,
        description=(
            f"Control-system target is already {record.target!r} "
            f"(generation {record.generation}); nothing was switched."
        ),
    )


@mcp.tool()
async def control_target_set(target: str) -> str:
    """Point this deployment's control-system tools at a different target.

    Targets are ``live`` (the machine this deployment's facility authored),
    ``va`` (the virtual accelerator it deploys) and ``standin`` (the live
    stand-in: a soft IOC this deployment runs for itself, which behaves like
    hardware and is a machine of its own rather than a mode of ``live``). A
    deployment has the ones its config describes, and ``control_target`` is the
    authority on which: a target absent from that roster is not switchable
    here, whatever this list names.

    The target belongs to the DEPLOYMENT and not to this session: every window,
    notebook kernel and sandbox on it moves too. The control-context record's
    owner is the only process that changes it, so this tool either writes the
    answer itself — when this server owns the record — or asks the owner for it
    and waits for the answer to appear in the record.

    Refused when: the run is read-only; an execution is in flight anywhere on
    this deployment, in which case the refusal names the busy client; the
    destination is not available — already active, unconfigured, or short of the
    posture that target requires; or the live servers have measured the
    destination unreachable. A deployment that has not settled on its current
    generation answers ``switch_in_progress`` naming the servers still applying,
    and nothing is filed. Asking for the target the deployment is already on is
    not a switch at all: it is answered where it stands, and mints no
    generation.

    Args:
        target: The target to switch to — ``live``, ``va`` or ``standin``.

    Returns:
        JSON naming the target, the generation it is on, and the connector host
        now serving it.
    """
    wanted = str(target or "").strip()
    async with _SWITCH_LOCK:
        return await _switch(wanted)


async def _switch(wanted: str) -> str:
    """The switch itself, with the one-at-a-time lock already held."""
    context = _server_context()
    if context is None:
        return await _refuse(
            from_target=UNKNOWN_TARGET,
            to_target=wanted,
            message=_context_unavailable_message(),
            suggestions=[
                "Restart the controls MCP server; no target can be read or changed until it "
                "has started."
            ],
            details={"target": wanted, "reason": REASON_CONTEXT_UNAVAILABLE},
            error_type=ERROR_UNAVAILABLE,
        )

    # Read after the lock, not before it: the caller that waited answers about
    # the deployment as it is now, and the switch it was queued behind may be
    # the very one it wanted.
    record = control_context.read_record()
    if record is None:
        return await _refuse(
            from_target=UNKNOWN_TARGET,
            to_target=wanted,
            message=_no_record_message(),
            suggestions=[
                "Wait for the controls MCP server to claim the control context, then ask again."
            ],
            details={"target": wanted, "reason": REASON_RECORD_UNAVAILABLE},
            error_type=ERROR_UNAVAILABLE,
        )

    if wanted == record.target:
        return _already_there(record, context.connector_hosts)

    reports = control_context.live_reports()
    if control_context.owned_here(record):
        return await _apply_here(context, record, wanted, reports)
    return await _request_and_wait(context, record, wanted, reports)
