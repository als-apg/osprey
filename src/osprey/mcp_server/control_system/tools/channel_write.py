"""MCP tool: channel_write — write values to control-system channels.

Safety: PreToolUse hooks enforce human approval before this tool runs.
The tool docstring is the static prompt the agent sees.
"""

import json
import logging

from osprey.errors import ChannelWriteBlockedError
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async

logger = logging.getLogger("osprey.mcp_server.tools.channel_write")

#: Severity at or above which a *verified* readback is still reported as an
#: alarm state. EPICS grades MINOR=1, MAJOR=2, INVALID=3; a MINOR alarm on a
#: channel that read back the value it was given is routine, MAJOR is not.
VERIFIED_ALARM_SEVERITY = 2

#: The closed set of ``write_state`` words, in the order the predicates below
#: test them. The generated safety rules name this key, so the spelling here and
#: the spelling there have to stay one string.
WRITE_STATE_KEY = "write_state"


def _alarm_severity(verification: object) -> int | None:
    """Readback alarm severity as an int, or None when it was not reported.

    Read through ``getattr`` and type-checked: a custom connector may return any
    duck-typed verification object, and a missing or non-numeric severity must
    degrade to "not reported" rather than raise inside the result projection.
    """
    severity = getattr(verification, "readback_alarm_severity", None)
    if isinstance(severity, bool) or not isinstance(severity, int):
        return None
    return severity


def _derive_write_state(result: object) -> str:
    """Classify one write result into a single closed-set word.

    Derived only from structured fields — never from ``notes``, which is
    display-only text — so the state an agent keys on cannot drift with wording.
    The predicates are ordered: the first one that matches wins, which is what
    keeps a refused write from also being reported as unverified.
    """
    if getattr(result, "blocked", None):
        return "blocked"
    if not getattr(result, "success", False):
        return "write_failed"

    verification = getattr(result, "verification", None)
    if verification is None:
        return "verification_not_reported"
    if getattr(verification, "level", None) == "none":
        return "verification_not_requested"

    severity = _alarm_severity(verification)
    if getattr(verification, "verified", False):
        if severity is not None and severity >= VERIFIED_ALARM_SEVERITY:
            return "verified_with_alarm"
        return "verified"

    if getattr(verification, "failure_kind", None) == "readback_failed":
        return "readback_failed"
    if severity is not None and severity > 0:
        return "unverified_alarm"
    return "verification_failed"


@mcp.tool()
async def channel_write(
    operations: list[dict],
    verification_level: str | None = None,
) -> str:
    """Write values to one or more control-system channels.

    Each operation is a dict with keys: channel (str), value (any), notes (str, optional).
    PreToolUse hooks handle human approval BEFORE this code runs.

    Every result carries one `write_state` word. Report the outcome its
    `write_state` names and nothing stronger — an unverified write is not a
    successful write, and you must not describe it as one. The states, in the
    order they are decided:

    - `blocked` — refused on policy or limits grounds; never sent to the machine.
    - `write_failed` — sent to the machine and failed.
    - `verification_not_reported` — the connector reported no verification at all.
    - `verification_not_requested` — level `none`: the write was sent, nothing
      confirmed it took effect.
    - `verified_with_alarm` — the readback matched, but the channel is in a MAJOR
      or worse alarm.
    - `verified` — the write was confirmed.
    - `readback_failed` — the readback itself could not be read, so the write is
      unconfirmed.
    - `unverified_alarm` — unconfirmed, and the channel reports an alarm.
    - `verification_failed` — unconfirmed: the readback disagreed with the setpoint.

    `summary.verification_failed` counts executed writes that asked for
    verification and did not get it — never a refused or unrequested one.

    Leave `verification_level` unset unless the operator names one. It is then
    resolved by the deployment, per write: the channel's limits entry, then the
    limits database `defaults.verification`, then the connector config, then
    `callback`. This tool does not raise on an unverified write — it reports it in
    `write_state`. (`osprey.runtime.write_channel`, on the Python path, raises.)

    Args:
        operations: List of write operations, each with "channel", "value", and optional "notes".
        verification_level: Optional override — "none", "callback", or "readback".
            Omit it to let the deployment resolve the level per write.

    Returns:
        JSON with per-operation results. Each `summary.results[]` entry carries
        `write_state`; its `verification` carries the readback value, the alarm
        name and severity, and `failure_kind` when the connector reported them.
        `summary.verification_failed` counts executed writes that asked for
        verification and did not get it.
    """
    if not operations:
        return make_error(
            "validation_error",
            "No write operations provided.",
            ["Provide at least one operation with 'channel' and 'value'."],
        )

    # Limits validation (additional safety layer inside the tool)
    try:
        from osprey.connectors.control_system.limits_validator import LimitsValidator
    except ImportError:
        LimitsValidator = None  # type: ignore[assignment,misc]

    validator = None
    if LimitsValidator is not None:
        validator = LimitsValidator.from_config()

    violations: list[dict] = []
    for op in operations:
        channel = op.get("channel")
        value = op.get("value")
        if not channel:
            return make_error(
                "validation_error",
                "Each operation must include a 'channel' key.",
                ["Ensure every entry in operations has 'channel' and 'value'."],
            )
        if validator:
            try:
                validator.validate(channel, value)
            except Exception as exc:
                violation = {
                    "channel": channel,
                    "attempted_value": value,
                    "violation_type": getattr(exc, "violation_type", "unknown"),
                    "reason": getattr(exc, "violation_reason", str(exc)),
                }
                if getattr(exc, "min_value", None) is not None:
                    violation["min_value"] = exc.min_value
                if getattr(exc, "max_value", None) is not None:
                    violation["max_value"] = exc.max_value
                if getattr(exc, "max_step", None) is not None:
                    violation["max_step"] = exc.max_step
                if getattr(exc, "current_value", None) is not None:
                    violation["current_value"] = exc.current_value
                violations.append(violation)

    if violations:
        # Build a clear message with limits info for each violation
        parts = []
        for v in violations:
            part = f"{v['channel']}={v['attempted_value']}: {v['reason']}"
            if "min_value" in v or "max_value" in v:
                part += f" (allowed range: [{v.get('min_value')}, {v.get('max_value')}])"
            if "max_step" in v:
                part += f" (max step: {v['max_step']})"
            parts.append(part)

        return make_error(
            "limits_violation",
            f"Channel limits violated: {'; '.join(parts)}",
            [
                "Do NOT attempt to work around this limit.",
                "Report the violation to the operator with the allowed range.",
                "The operator may adjust the limits database if the value is appropriate.",
            ],
            details=violations,
        )

    # Execute writes
    async with connector_error_handler("channel_write"):
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        connector = await registry.control_system()

        # Determine per-channel verification level and tolerance
        connector_results = []  # Raw connector results for bridge
        results_serialised = []  # Serialised dicts for the data file

        if len(operations) == 1:
            op = operations[0]
            channel, value = op["channel"], op["value"]
            # An explicit level is the caller's decision and is forwarded
            # unchanged; the limits database only fills in a level the caller
            # left out. The tolerance is independent of that: a per-channel
            # tolerance applies whether or not the level was named, or an
            # explicit "readback" would silently fall back to the connector's
            # absolute default.
            level = verification_level
            tolerance = None
            if validator:
                cfg_level, cfg_tol = validator.get_verification_config(channel, value)
                if level is None and cfg_level:
                    level = cfg_level
                if cfg_tol is not None:
                    tolerance = cfg_tol
            # Omission is a sentinel, not a value: forwarding None would override
            # a legacy custom connector's own declared default.
            write_kwargs: dict = {}
            if level is not None:
                write_kwargs["verification_level"] = level
            if tolerance is not None:
                write_kwargs["tolerance"] = tolerance
            wr = await connector.write_channel(channel, value, **write_kwargs)
            connector_results.append(wr)
        else:
            # A batch carries one scalar level for every channel in it, so an
            # omitted level is not resolved here: leaving the keyword off lets
            # each channel resolve its own entry exactly as a single write does.
            write_ops = [(op["channel"], op["value"]) for op in operations]
            batch_kwargs: dict = {}
            if verification_level is not None:
                batch_kwargs["verification_level"] = verification_level
            connector_results = await connector.write_multiple_channels(write_ops, **batch_kwargs)

        for op, wr in zip(operations, connector_results, strict=True):
            result_entry = {
                "channel": wr.channel_address,
                "value_written": wr.value_written,
                "success": wr.success,
                "error_message": wr.error_message,
                "blocked": wr.blocked,
                "refusal_reason": wr.refusal_reason,
            }
            result_entry[WRITE_STATE_KEY] = _derive_write_state(wr)
            # "is not None", matching _derive_write_state: absence of a
            # verification object is what "verification_not_reported" means, and
            # the two must not disagree about which results have one.
            verification = getattr(wr, "verification", None)
            if verification is not None:
                result_entry["verification"] = {
                    "level": getattr(verification, "level", None),
                    "verified": getattr(verification, "verified", None),
                    "readback_value": getattr(verification, "readback_value", None),
                    "tolerance_used": getattr(verification, "tolerance_used", None),
                    # Machine-readable half of the verification. Absent fields
                    # stay null: "not reported" is deliberately distinct from a
                    # reported healthy severity of 0.
                    "readback_alarm_status": getattr(verification, "readback_alarm_status", None),
                    "readback_alarm_severity": getattr(
                        verification, "readback_alarm_severity", None
                    ),
                    "failure_kind": getattr(verification, "failure_kind", None),
                    "notes": getattr(verification, "notes", None),
                }
            if op.get("notes"):
                result_entry["notes"] = op["notes"]
            results_serialised.append(result_entry)

        # Build compact summary inline
        successful = sum(1 for r in results_serialised if r["success"])
        refused = sum(1 for r in results_serialised if r.get("blocked"))
        # A write that never executed is not a verification failure: only count
        # writes that succeeded, asked for verification, and did not get it.
        verification_failed = sum(
            1
            for r in results_serialised
            if r["success"]
            and (r.get("verification") or {}).get("level") in ("callback", "readback")
            and not (r.get("verification") or {}).get("verified")
        )
        summary = {
            "total_writes": len(results_serialised),
            "successful": successful,
            "failed": len(results_serialised) - successful,
            "refused": refused,
            "verification_failed": verification_failed,
            "results": [
                {
                    "channel": r["channel"],
                    "value": r["value_written"],
                    "success": r["success"],
                    WRITE_STATE_KEY: r[WRITE_STATE_KEY],
                    "error": r.get("error_message"),
                    "blocked": r.get("blocked"),
                    "refusal_reason": r.get("refusal_reason"),
                    "verification": r.get("verification"),
                }
                for r in results_serialised
            ],
        }
        # The caller's value, or null when they left the level to the deployment.
        access_details = {"verification_level": verification_level}

        if results_serialised and successful == 0:
            failures = [wr for wr in connector_results if not wr.success]
            if failures and all(wr.blocked for wr in failures):
                # Every failed op was a policy refusal (never sent to the control
                # system): surface a typed write-refusal envelope, not internal_error.
                refused_channels = [wr.channel_address for wr in failures]
                first = failures[0]
                raise ChannelWriteBlockedError(
                    first.channel_address,
                    first.refusal_reason or "WRITES_DISABLED",
                    message=(
                        f"All {len(failures)} write(s) refused by the reference monitor: "
                        f"{', '.join(refused_channels)}"
                    ),
                )
            # At least one failure was an attempted caput that failed (I/O failure,
            # not a policy refusal): preserve the internal_error classification.
            errors = sorted(
                {r["error_message"] for r in results_serialised if r.get("error_message")}
            )
            raise RuntimeError(
                f"All {len(results_serialised)} write(s) rejected: {'; '.join(errors)}"
            )

        # Agent-activity highlight (purely additive, after the fact): name only
        # the channels whose writes actually executed. Every refusal path —
        # limits_violation validation, all-blocked, all-failed — returns or
        # raises before this point, so those emit nothing. notify_agent_activity
        # never raises; the blocking call runs off the event loop.
        executed_channels = [r["channel"] for r in results_serialised if r["success"]]
        if executed_channels:
            await notify_agent_activity_async(
                "channel_write", "channel", detail=", ".join(executed_channels)
            )

        # Return ephemeral result (no persistent storage for channel writes)
        return json.dumps(
            {
                "status": "success",
                "description": f"Wrote {len(results_serialised)} channel(s)",
                "summary": summary,
                "access_details": access_details,
            },
            default=str,
        )
