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
The switch itself lives in
:class:`~osprey.mcp_server.control_system.connector_host_manager.ConnectorHostManager`,
which owns the child process, the generation counter and the spawn-then-swap
order. This module is the *gate* in front of it: three refusals, in a fixed
order, and then a delegation.

The three refusals are :func:`switch_gate`, a function rather than tool-body
code, because the agent's tool is no longer the only way a session moves: the
session-control reconciler asks the same question immediately before
``hosts.switch()``, on a path that has no MCP request behind it. The gate
therefore takes the server context and a target name and nothing else, and the
tool's only remaining job with a refusal is to report it. Two implementations
of "may this session move there" would eventually disagree, and the surface an
operator happened to use would decide which answer they got.

Why the order is fixed
----------------------
Each refusal answers a different question, and the order runs from "this
session may never switch at all" to "this particular destination is not usable
right now". Reporting the narrowest true reason last means the answer always
names the nearest thing an operator can act on:

1. **A read-only run** (``OSPREY_EXECUTION_MODE=readonly``) mutates nothing.
   Agent code submitted read-only is a claim about the whole run, and a switch
   is session state — so the claim is enforced here rather than being
   negotiated per target. It is checked first precisely because everything else
   might also refuse: a read-only session that asks for an ineligible target
   should be told it cannot switch, not sent off to fix a config key that would
   still leave it unable to switch.
2. **An execution in flight.** The python executor's sandbox is stamped with
   the target and generation at launch, so a switch under a running execution
   retires the connector host that run was promised. The executor records a
   marker file for the duration of every run (see the marker contract below)
   and this tool refuses while one is live.
3. **Eligibility**, from :mod:`target_eligibility` and nowhere else. The reason
   string a refusal carries is the module's own, so the roster and this tool
   can never disagree about why a target is unusable — including the honesty
   rule (a simulated present with an invented past) and the FR-8 posture the
   live machine requires.

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

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from osprey.mcp_server.control_system.connector_host_manager import (
    SwitchError,
    target_display_metadata,
)
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.control_system.target_eligibility import (
    derive_endpoints,
    effective_writes_for_target,
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
from osprey_connectors.control_system.base import is_readonly_run
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
    "switch_gate",
    "target_rows",
]

# The in-flight marker names and reader are re-exported from ``target_state``
# (imported above and named in ``__all__``); see the module docstring for why
# they moved. Importers of this module — including the drift guard that pins the
# spelling against the executor's replica — are unaffected.

# -- machine-readable refusal reasons ---------------------------------------

#: Reasons this gate adds to the eligibility module's own. Eligibility reasons
#: travel through verbatim, so a refusal here and a roster row can be matched.
REASON_READONLY_RUN = "readonly_run"
REASON_EXECUTION_IN_FLIGHT = "execution_in_flight"
REASON_CONTEXT_UNAVAILABLE = "context_unavailable"
#: An exception the switch did not classify. Reported as a failure rather than
#: swallowed: the operator approved an attempt, and an attempt that ended in a
#: way nobody anticipated is still an attempt that ended.
REASON_INTERNAL_ERROR = "internal_error"

#: Stands in for the session target when it cannot be read at all — the context
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
        "The control-system server context is not initialized, so this session has no "
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
) -> Any:
    """Report the refusal to the operator, then raise it to the agent."""
    await _emit_failure(from_target, to_target, str(details.get("reason") or ""))
    return make_error(error_type, message, suggestions, details=details)


def _in_flight_detail(record: dict[str, Any], target: str) -> tuple[str, list[str], dict[str, Any]]:
    """The refusal a running execution earns: message, suggestions, details."""
    running_on = str(record.get("target") or "unknown")
    whose = (
        "this session"
        if record.get("owner_ppid") == os.getppid()
        else "another session sharing this deployment"
    )
    return (
        f"execution in flight on target {running_on!r}; wait or stop it.",
        [
            f"The running execution belongs to {whose} and was launched against target "
            f"{running_on!r}.",
            "Wait for it to finish, or stop it, then switch again.",
        ],
        {
            "target": target,
            "reason": REASON_EXECUTION_IN_FLIGHT,
            "executing_target": running_on,
            "executor_pid": record.get("pid"),
            "started_at": record.get("started_at"),
        },
    )


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
    session_target: str,
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
        session_target: The target this session is on right now.
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
        # This session's real posture for the target, read ONCE and then used
        # for every answer this row carries. The eligibility verdict, the
        # gateway role and the `writes_permitted` flag are three views of the
        # same fact, and a row that read the store separately for each could
        # report a role the operator narrowed away beside a flag saying they
        # had not.
        writes_permitted = _writes_permitted(config, target)
        availability = target_availability(
            config, target, session_target, baseline, writes_enabled=writes_permitted
        )
        display = metadata.get(target, {})
        row: dict[str, Any] = {
            "target": target,
            "active": target == session_target,
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
    """Whether a write to *target* would be permitted on this session, now.

    Three things decide it: that target's own posture
    (``control_system.connector.<type>.writes_enabled``, falling back to
    ``control_system.writes_enabled`` where its type states none), this run's
    own claim (``OSPREY_EXECUTION_MODE``), and the operator's narrowing for
    this session from the header chip. All three are combined by
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
    """Report which control system this session is pointed at, and what else it could be.

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
        session_target=status["target"],
        baseline=status["baseline_target"],
        probe_snapshot=snapshot,
    )

    return json.dumps(
        {
            "status": "success",
            "description": (
                f"Session target is {status['target']!r} (generation {status['generation']}); "
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


@dataclass(frozen=True)
class GateVerdict:
    """One refusal, in the exact words the operator would be told.

    Carries everything a caller needs to report the refusal on its own surface:
    the machine-readable ``reason``, the human-readable ``detail`` and the
    suggestions, plus the structured ``details`` payload that goes into the
    error envelope (and whose ``reason`` key is the same string). A verdict is
    a refusal — "may proceed" is spelled ``None`` by :func:`switch_gate`, not
    as a fourth verdict, so no caller can mistake an allowed switch for a
    refusal it forgot to check the flag on.
    """

    reason: str
    detail: str
    suggestions: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


async def switch_gate(context: Any, wanted: str) -> GateVerdict | None:
    """Whether this session may switch to *wanted* right now.

    The three refusals, in the fixed order the module docstring explains: a
    read-only run, an execution in flight, then eligibility — which is where
    ``already_active`` arrives, since "you are already there" is the roster's
    answer and not a check of this gate's own.

    It takes the server context and a target name and nothing else: the second
    caller is the session-control reconciler, which asks this immediately
    before :meth:`ConnectorHostManager.switch` on a path that has no MCP
    request behind it. The session's target of record and the deployment
    baseline are therefore read from the manager the context holds, not passed
    in — two callers that supplied their own would eventually supply different
    ones.

    Args:
        context: The controls server context (already resolved; a context that
            could not be read is its caller's problem to report, because the
            two surfaces owe the operator different things for that failure).
        wanted: The target the session wants to move to.

    Returns:
        The refusal, or ``None`` when nothing refuses and the caller may switch.
    """
    hosts = context.connector_hosts

    # (1) A read-only run mutates nothing, whatever the destination.
    if is_readonly_run():
        return GateVerdict(
            reason=REASON_READONLY_RUN,
            detail=(
                "a switch mutates session state; read-only sessions stay on the "
                "deployment baseline."
            ),
            suggestions=[
                "Re-run this without the read-only execution mode to change the "
                "control-system target."
            ],
            details={"target": wanted, "reason": REASON_READONLY_RUN},
        )

    # (2) A run in flight was stamped with the target it started on. This reads
    # another process's files, so there is a window: an execution that starts
    # between this check and the swap is not seen here. That window is closed
    # elsewhere and not by widening this check — the sandbox is pinned to the
    # generation it launched under, and its writes refuse once the session moves
    # past it. What this check buys is the refusal arriving as an answer to the
    # switch rather than as a failure inside a running script.
    running = in_flight_executions()
    if running:
        message, suggestions, details = _in_flight_detail(running[0], wanted)
        return GateVerdict(
            reason=REASON_EXECUTION_IN_FLIGHT,
            detail=message,
            suggestions=suggestions,
            details=details,
        )

    # (3) Eligibility, session-relative, in the eligibility module's own words.
    availability = target_availability(
        context.config.raw,
        wanted,
        hosts.active_target(),
        hosts.baseline,
    )
    if not availability.available_now:
        return GateVerdict(
            reason=str(availability.reason or ""),
            detail=availability.detail,
            suggestions=[
                "Ask for the target roster to see what each target would need to become usable."
            ],
            details=availability.as_dict(),
        )

    return None


@mcp.tool()
async def control_target_set(target: str) -> str:
    """Point this session's control-system tools at a different target.

    Targets are ``live`` (the machine this deployment's facility authored),
    ``va`` (the virtual accelerator it deploys) and ``standin`` (the live
    stand-in: a soft IOC this deployment runs for itself, which behaves like
    hardware and is a machine of its own rather than a mode of ``live``). A
    deployment has the ones its config describes, and ``control_target`` is the
    authority on which: a target absent from that roster is not switchable
    here, whatever this list names. The switch replaces the process that talks
    to the control system: the destination is spawned and proven reachable
    BEFORE the current one is retired, so a switch that fails leaves the
    session exactly where it was.

    Refused, in this order, when: the run is read-only; an execution is in
    flight; or the destination is not available to this session — already
    active, unconfigured, or short of the posture that target requires. The
    strict limits posture guards ``live`` and ``standin`` alike; the operator
    acknowledgment is the live machine's alone, since standing the soft IOC up
    was itself the deployment saying what it is. A ``standin`` this deployment
    has not actually stood up is refused ``standin_not_deployed``, so the
    switch cannot be talked onto hardware under a soft label. The refusal names
    the reason the target roster reports.

    Args:
        target: The session target to switch to — ``live``, ``va`` or
            ``standin``.

    Returns:
        JSON naming the new target, the generation it is on, the connector type
        and gateway the child came up against, and the channel that proved it.
    """
    wanted = str(target or "").strip()

    # Resolved before the first refusal, not as one: every outcome is reported
    # to the operator as a move *from* somewhere, and that is the session's
    # target of record. The order of the three refusals below is unaffected.
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
    hosts = context.connector_hosts
    session_target = hosts.active_target()

    # The three refusals, in their fixed order, from the one function the
    # operator-initiated switch consults too. Read them in :func:`switch_gate`;
    # this tool's job with a verdict is only to report it the way it always has.
    verdict = await switch_gate(context, wanted)
    if verdict is not None:
        return await _refuse(
            from_target=session_target,
            to_target=wanted,
            message=verdict.detail,
            suggestions=verdict.suggestions,
            details=verdict.details,
        )

    try:
        result = await hosts.switch(wanted)
    except SwitchError as exc:
        logger.warning("Target switch to %r failed at stage %r: %s", wanted, exc.stage, exc.detail)
        # A failed switch left the session where it was, and the operator who
        # approved the attempt is told so rather than left to infer it.
        await _emit_failure(session_target, wanted, exc.reason)
        suggestions = [
            f"The session is still on target {hosts.active_target()!r}; nothing was switched.",
        ]
        if exc.gateway:
            # Both roles usually share a hostname and differ only by port, so
            # without this line the failure reads as "the control system is
            # down" when only one gateway beside a healthy one is unserved.
            suggestions.append(
                f"The readiness probe ran through the {exc.gateway['role']!r} gateway at "
                f"{exc.gateway['host']}:{exc.gateway['port']}. Check that a gateway process "
                "is actually serving that host and port — the control system itself, and "
                "this target's other gateway roles, may be healthy."
            )
        suggestions.append("Fix what the detail names, then ask for the switch again.")
        return make_error(
            ERROR_FAILED,
            exc.detail,
            suggestions,
            details=exc.as_dict(),
        )
    except BaseException:
        # Anything the switch did not classify — a cancellation, a bug, an
        # error from a layer below. The operator watching this session saw an
        # attempt begin and must see that it ended, whatever ended it; the
        # exception itself still travels, unchanged, to the agent.
        await _emit_failure(session_target, wanted, REASON_INTERNAL_ERROR)
        raise

    await notify_target_switch_async(
        from_target=result["previous_target"],
        to_target=result["target"],
        outcome=SWITCH_OUTCOME_SUCCESS,
        generation=result["generation"],
    )

    status = hosts.status()
    logger.info("Session target is now %r (generation %s)", result["target"], result["generation"])
    description = (
        f"Control-system target is now {result['target']!r} (generation {result['generation']})."
    )
    access_details = {
        "endpoint": result["endpoint"],
        "selected_role": result["selected_role"],
        "baseline_target": status["baseline_target"],
        "child_pid": result["child_pid"],
        "previous_drained": result["previous_drained"],
        "drain_timeout_s": result["drain_timeout_s"],
    }
    # A landing that could not use the write gateway is a success with a
    # warning, not a silent success: the operator has a gateway to fix.
    fallback = result.get("write_gateway_fallback")
    if fallback:
        description += f" WARNING: {fallback['detail']}"
        access_details["write_gateway_fallback"] = fallback
    return json.dumps(
        {
            "status": "success",
            "description": description,
            "summary": {
                "target": result["target"],
                "generation": result["generation"],
                "previous_target": result["previous_target"],
                "target_changed": result["target_changed"],
                "connector_type": result["connector_type"],
                "probe_channel": result["probe_channel"],
            },
            "access_details": access_details,
        },
        default=str,
    )
