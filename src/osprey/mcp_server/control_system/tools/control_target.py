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

The constants and the record shape are stated **twice** — here and in
:mod:`osprey.mcp_server.python_executor.executor` — and pinned equal by
``tests/mcp_server/test_control_target_set.py``. That is the replica pattern
this repository already uses for the deployed hooks: the alternative is for one
MCP server process to import the other's module at run time, which would drag
the whole controls server into the executor (or the reverse) for two string
constants.

"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from typing import Any

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import (
    SwitchError,
    target_display_metadata,
)
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.control_system.target_eligibility import (
    derive_endpoints,
    target_availability,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import (
    SWITCH_OUTCOME_FAILURE,
    SWITCH_OUTCOME_SUCCESS,
    notify_target_switch_async,
)
from osprey_connectors.control_system.base import is_readonly_run
from osprey_connectors.types import CONTROL_TARGETS, target_writes_enabled

logger = logging.getLogger("osprey.mcp_server.tools.control_target")

__all__ = [
    "INFLIGHT_FILE_GLOB",
    "INFLIGHT_FILE_PREFIX",
    "INFLIGHT_FILE_SUFFIX",
    "REASON_EXECUTION_IN_FLIGHT",
    "REASON_READONLY_RUN",
    "control_target",
    "control_target_set",
    "in_flight_executions",
    "target_rows",
]

# -- the in-flight marker contract (replica; see the module docstring) -------

#: One marker per execution, named for the process that will remove it.
INFLIGHT_FILE_PREFIX = "exec_inflight_"
INFLIGHT_FILE_SUFFIX = ".json"
INFLIGHT_FILE_GLOB = f"{INFLIGHT_FILE_PREFIX}*{INFLIGHT_FILE_SUFFIX}"

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


def in_flight_executions() -> list[dict[str, Any]]:
    """Every live execution marker, oldest first. Never raises.

    A marker whose writing process is gone is swept: it is the residue of a
    killed executor, and treating it as live would make every later switch
    impossible with nothing an operator could stop. A marker that cannot be
    read is neither reported nor removed — it says nothing, and it is not this
    reader's file to delete.

    The answer is ADVISORY. It is a best-effort observation of another
    process's files, so a marker can be missing (the executor could not write
    it) or can appear a moment after this reader looked. Nothing about
    correctness rests on it: the guarantee that a run cannot be moved onto a
    machine nobody selected is the generation pin — an execution stamped at
    generation *n* has its writes refused once the session moves past it — and
    this check exists to turn that refusal into a question asked before the
    switch rather than an error discovered after it.
    """
    try:
        entries = sorted(target_state.state_dir().glob(INFLIGHT_FILE_GLOB))
    except OSError:  # pragma: no cover - unreadable state dir
        return []

    live: list[dict[str, Any]] = []
    for entry in entries:
        record = target_state.read_file(entry)
        if record is None:
            logger.debug("Unreadable execution marker %s; ignoring it", entry.name)
            continue
        pid = record.get("pid")
        if not isinstance(pid, int) or not target_state.is_process_alive(pid):
            with contextlib.suppress(OSError):
                entry.unlink(missing_ok=True)
            logger.debug("Swept execution marker %s (writer pid %r gone)", entry.name, pid)
            continue
        live.append(record)
    return live


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
        ``{target: row}`` for every target the connectors package defines.
    """
    metadata = target_display_metadata(config)
    snapshot = probe_snapshot or {}

    rows: dict[str, dict[str, Any]] = {}
    for target in CONTROL_TARGETS:
        availability = target_availability(config, target, session_target, baseline)
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
            "writes_permitted": _writes_permitted(config, target),
        }
        probe_channel = display.get("probe_channel") or ""
        if probe_channel:
            row["probe_channel"] = probe_channel

        try:
            derivation = derive_endpoints(config, target)
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
    """Whether a write to *target* would be permitted at all.

    The two things that decide it are that target's own posture
    (``control_system.connector.<type>.writes_enabled``, falling back to
    ``control_system.writes_enabled`` where its type states none) and this run's
    own claim (``OSPREY_EXECUTION_MODE``) — exactly the pair
    :func:`~osprey.mcp_server.control_system.target_eligibility.derive_endpoints`
    takes, resolved through the same helper it resolves it with, so the roster
    and the gateway selection cannot disagree about the posture they describe.
    """
    section = config.get("control_system") if isinstance(config, dict) else None
    return target_writes_enabled(section, target) and not is_readonly_run()


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
    the deployment; whether the target is the real machine; and the channel a
    switch would read to prove the target is reachable.

    A target nobody has activated yet is described from configuration alone —
    that is what makes this answer correct before any switch has happened.
    ``endpoint_tcp`` is absent where nothing has measured it, and
    ``not_applicable`` where the gateway is reached over an address list (CA
    search is UDP there, so a TCP probe could prove nothing).

    Returns:
        JSON with the active target and generation, and one row per target.
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


@mcp.tool()
async def control_target_set(target: str) -> str:
    """Point this session's control-system tools at a different target.

    Targets are ``live`` (the real machine this deployment describes) and
    ``va`` (the virtual accelerator it deploys). The switch replaces the
    process that talks to the control system: the destination is spawned and
    proven reachable BEFORE the current one is retired, so a switch that fails
    leaves the session exactly where it was.

    Refused, in this order, when: the run is read-only; an execution is in
    flight; or the destination is not available to this session — already
    active, unconfigured, or short of the posture switching to the live machine
    requires. The refusal names the reason the target roster reports.

    Args:
        target: The session target to switch to — ``live`` or ``va``.

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

    # (1) A read-only run mutates nothing, whatever the destination.
    if is_readonly_run():
        return await _refuse(
            from_target=session_target,
            to_target=wanted,
            message=(
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
    # between this check and the swap below is not seen here. That window is
    # closed elsewhere and not by widening this check — the sandbox is pinned to
    # the generation it launched under, and its writes refuse once the session
    # moves past it. What this check buys is the refusal arriving as an answer
    # to the operator's switch rather than as a failure inside their running
    # script.
    running = in_flight_executions()
    if running:
        message, suggestions, details = _in_flight_detail(running[0], wanted)
        return await _refuse(
            from_target=session_target,
            to_target=wanted,
            message=message,
            suggestions=suggestions,
            details=details,
        )

    # (3) Eligibility, session-relative, in the eligibility module's own words.
    availability = target_availability(
        context.config.raw,
        wanted,
        session_target,
        hosts.baseline,
    )
    if not availability.available_now:
        return await _refuse(
            from_target=session_target,
            to_target=wanted,
            message=availability.detail,
            suggestions=[
                "Ask for the target roster to see what each target would need to become usable."
            ],
            details=availability.as_dict(),
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
