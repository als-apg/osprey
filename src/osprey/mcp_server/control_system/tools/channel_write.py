"""MCP tool: channel_write — write values to control-system channels.

Safety: PreToolUse hooks enforce human approval before this tool runs.
Tool docstring is the static prompt visible to Claude Code.
"""

import json
import logging

from osprey.mcp_server.control_system.error_handling import connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.channel_write")


@mcp.tool()
async def channel_write(
    operations: list[dict],
    verification_level: str = "callback",
) -> str:
    """Write values to one or more control-system channels.

    Each operation is a dict with keys: channel (str), value (any), notes (str, optional).
    PreToolUse hooks handle human approval BEFORE this code runs.

    Args:
        operations: List of write operations, each with "channel", "value", and optional "notes".
        verification_level: Verification after write — "none", "callback", or "readback".

    Returns:
        JSON with per-operation results including verification status.
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
            level = verification_level
            tolerance = None
            if validator:
                cfg_level, cfg_tol = validator.get_verification_config(channel, value)
                if cfg_level:
                    level = cfg_level
                if cfg_tol is not None:
                    tolerance = cfg_tol
            wr = await connector.write_channel(
                channel, value, verification_level=level, tolerance=tolerance
            )
            connector_results.append(wr)
        else:
            write_ops = [(op["channel"], op["value"]) for op in operations]
            connector_results = await connector.write_multiple_channels(
                write_ops,
                verification_level=verification_level,
            )

        for op, wr in zip(operations, connector_results, strict=True):
            result_entry = {
                "channel": wr.channel_address,
                "value_written": wr.value_written,
                "success": wr.success,
                "error_message": wr.error_message,
            }
            if wr.verification:
                result_entry["verification"] = {
                    "level": wr.verification.level,
                    "verified": wr.verification.verified,
                    "readback_value": wr.verification.readback_value,
                    "tolerance_used": wr.verification.tolerance_used,
                    "notes": wr.verification.notes,
                }
            if op.get("notes"):
                result_entry["notes"] = op["notes"]
            results_serialised.append(result_entry)

        # Build compact summary inline
        successful = sum(1 for r in results_serialised if r["success"])
        summary = {
            "total_writes": len(results_serialised),
            "successful": successful,
            "failed": len(results_serialised) - successful,
            "results": [
                {
                    "channel": r["channel"],
                    "value": r["value_written"],
                    "success": r["success"],
                    "error": r.get("error_message"),
                    "verification": r.get("verification"),
                }
                for r in results_serialised
            ],
        }
        access_details = {"verification_level": verification_level}

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
