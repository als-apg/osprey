"""MCP tool: channel_write — write values to control-system channels.

Safety: PreToolUse hooks enforce human approval before this tool runs.
Tool docstring is the static prompt visible to Claude Code.
"""

import json
import logging

from osprey.mcp_server.control_system.error_handling import ToolError, connector_error_handler
from osprey.mcp_server.control_system.server import mcp
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.channel_write")


@mcp.tool()
async def channel_write(
    operations: list[dict],
    verification_level: str = "callback",
) -> str:
    """Write values to one or more EPICS control-system channels.

    Each operation is a dict with keys: channel (str), value (any), notes (str, optional).
    PreToolUse hooks handle human approval BEFORE this code runs.

    Args:
        operations: List of write operations, each with "channel", "value", and optional "notes".
        verification_level: Verification after write — "none", "callback", or "readback".

    Returns:
        JSON with per-operation results including verification status.
    """
    if not operations:
        return json.dumps(
            make_error(
                "validation_error",
                "No write operations provided.",
                ["Provide at least one operation with 'channel' and 'value'."],
            )
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
            return json.dumps(
                make_error(
                    "validation_error",
                    "Each operation must include a 'channel' key.",
                    ["Ensure every entry in operations has 'channel' and 'value'."],
                )
            )
        if validator:
            try:
                validator.validate(channel, value)
            except Exception as exc:
                violations.append(
                    {
                        "channel": channel,
                        "value": value,
                        "violation": str(exc),
                    }
                )

    if violations:
        return json.dumps(
            make_error(
                "limits_violation",
                f"Channel limits violated for {len(violations)} operation(s).",
                [v["violation"] for v in violations],
            )
        )

    # Execute writes
    try:
        async with connector_error_handler("channel_write"):
            from osprey.mcp_server.control_system.registry import get_mcp_registry

            registry = get_mcp_registry()
            connector = await registry.control_system()

            # Determine per-channel verification level and tolerance
            connector_results = []  # Raw connector results for bridge
            results_serialised = []  # Serialised dicts for the data file
            for op in operations:
                channel = op["channel"]
                value = op["value"]

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

    except ToolError as exc:
        return exc.response
