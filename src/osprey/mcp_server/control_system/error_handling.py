"""Shared error handling for control-system MCP tools.

Provides an async context manager that wraps the common
ConnectionError / TimeoutError / Exception pattern used by
channel_read, channel_write, and archiver_read.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastmcp.exceptions import ToolError

from osprey.errors import ChannelLimitsViolationError, ChannelWriteBlockedError
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.control_system.error_handling")


__all__ = [
    "ToolError",
    "connector_error_handler",
    "describe_active_target",
    "invalidate_active_connector",
]


def describe_active_target() -> dict[str, str] | None:
    """Identify the control-system target the session is pointed at, or ``None``.

    A failure envelope that says "the write to ``SR:...:SP`` timed out" is a
    materially different situation on the live machine than on the simulator,
    and the operator should not have to reconstruct which one it was from
    session memory. This resolves ``{name, label, endpoint}`` for the active
    target from the same authorities the roster uses: the supervisor's target
    of record and the per-target display metadata rendered from config.

    Fail-soft by construction: an envelope that raises while describing a
    failure is worse than one that omits the target, so every problem here —
    no server context yet, a config the deriver rejects, anything — collapses
    to ``None`` and the envelope renders exactly as it did before this existed.
    """
    try:
        from osprey.mcp_server.control_system.connector_host_manager import (
            target_display_metadata,
        )
        from osprey.mcp_server.control_system.server_context import get_server_context

        context = get_server_context()
        target = context.connector_hosts.active_target()
        meta = target_display_metadata(context.config.raw).get(target) or {}
        identity = {"name": target, "label": str(meta.get("label") or target)}
        endpoint = str(meta.get("endpoint") or "")
        if endpoint:
            identity["endpoint"] = endpoint
        return identity
    except Exception:  # noqa: BLE001 — see docstring: never fail an envelope
        return None


def _target_clause(identity: dict[str, str] | None) -> str:
    """The human half: `` (active target: LIVE MACHINE at 127.0.0.1:5064)``."""
    if not identity:
        return ""
    where = f" at {identity['endpoint']}" if identity.get("endpoint") else ""
    return f" (active target: {identity['label']}{where})"


async def invalidate_active_connector(connector_name: str) -> None:
    """Drop the connector a failed call was using, so the next call rebuilds it.

    This is the single seam between the ``ConnectionError`` branch below and
    whatever owns the connector's lifetime. Today that owner is the server
    context, and "invalidate" means dropping the cached instance so the next
    ``control_system()`` call recreates it — exactly what the branch did inline
    before, with the same call and the same failure behaviour.

    It is a named function rather than an inline call because the owner is
    about to change. When the connector lives in a connector-host child, a
    connection failure has to mean *respawn the child for the target it was
    already serving* — the same target, no generation bump, since a dead child
    is a lost process and not a target change. Re-pointing this one function is
    then the whole of that change; the envelope rendering below never has to
    learn where connectors live.
    """
    from osprey.mcp_server.control_system.server_context import get_server_context

    await get_server_context().invalidate_connector(connector_name)


@asynccontextmanager
async def connector_error_handler(
    tool_name: str,
    connector_name: str = "control_system",
) -> AsyncIterator[None]:
    """Wrap a control-system tool body with standardized error handling.

    Usage::

        async with connector_error_handler("channel_read"):
            registry = get_server_context()
            connector = await registry.control_system()
            # ... tool logic ...
            return CallToolResult(...)

    On known errors (Connection / Timeout / limits violation / unhandled
    exception), ``make_error()`` raises a ``fastmcp.ToolError`` carrying the
    structured envelope. fastmcp converts that into a wire-level
    ``CallToolResult(isError=True)``.

    For the ``control_system`` connector, every envelope rendered here also
    carries the active target's identity — ``details["active_target"]`` with
    ``name``/``label`` (and ``endpoint`` where config knows one), plus a
    human clause in the message where the machine matters — so a dead-IOC
    timeout on the live machine can never be mistaken for one on the
    simulator. The archiver has no live/VA axis, so its envelopes carry none.
    """
    try:
        yield
    except ToolError:
        raise  # Already a formatted envelope — propagate as-is
    except ConnectionError as exc:
        await invalidate_active_connector(connector_name)
        identity = describe_active_target() if connector_name == "control_system" else None
        if identity:
            where = f" at {identity['endpoint']}" if identity.get("endpoint") else ""
            reach_suggestion = f"Check that the {identity['label']}{where} is reachable."
        else:
            reach_suggestion = f"Check that the {connector_name.replace('_', ' ')} is running."
        make_error(
            "connection_error",
            f"Failed to connect to the {connector_name.replace('_', ' ')}"
            f"{_target_clause(identity)}: {exc}",
            [
                reach_suggestion,
                f"If the {connector_name} settings are wrong, fix them in the build profile "
                "(profile.yml on the host), then rebuild and redeploy.",
            ],
            details={"active_target": identity} if identity else None,
        )
    except TimeoutError as exc:
        identity = describe_active_target() if connector_name == "control_system" else None
        make_error(
            "timeout_error",
            f"{tool_name} timed out{_target_clause(identity)}: {exc}",
            ["Check network connectivity.", "Try a smaller request."],
            details={"active_target": identity} if identity else None,
        )
    except ChannelLimitsViolationError as exc:
        violation = {
            "channel": exc.channel_address,
            "attempted_value": exc.attempted_value,
            "violation_type": exc.violation_type,
            "reason": exc.violation_reason,
        }
        if exc.min_value is not None:
            violation["min_value"] = exc.min_value
        if exc.max_value is not None:
            violation["max_value"] = exc.max_value
        if exc.max_step is not None:
            violation["max_step"] = exc.max_step
        if exc.current_value is not None:
            violation["current_value"] = exc.current_value
        identity = describe_active_target() if connector_name == "control_system" else None
        if identity:
            violation["active_target"] = identity

        msg = f"Channel limits violated during {tool_name}: {exc.violation_reason}"
        if exc.min_value is not None or exc.max_value is not None:
            msg += f" (allowed range: [{exc.min_value}, {exc.max_value}])"

        make_error(
            "limits_violation",
            msg,
            [
                "Do NOT attempt to work around this limit.",
                "Report the violation to the operator with the allowed range.",
            ],
            details=violation,
        )
    except ChannelWriteBlockedError as exc:
        # Both branches describe a write that put no value on the channel, but
        # they must not name the same refuser. A control-system denial reaches
        # here only after OSPREY sent the write and the control system said no;
        # telling the operator it "was never sent" would point them at OSPREY's
        # own policy settings for something only the control system can grant.
        identity = describe_active_target() if connector_name == "control_system" else None
        if exc.reason == "CONTROL_SYSTEM_REFUSED":
            message = (
                f"Write refused by the control system during {tool_name}"
                f"{_target_clause(identity)}: {exc}"
            )
            suggestions = [
                "The control system itself denied this write (access security); "
                "no value was written.",
                "Do NOT attempt to work around the refusal.",
                "Report it to the operator: write access for this channel is granted "
                "by the control system, not by OSPREY.",
            ]
        else:
            message = f"Write refused by the reference monitor during {tool_name}: {exc}"
            suggestions = [
                "This write was refused on policy grounds; it was never sent to the control system.",
                "Do NOT attempt to work around the refusal.",
            ]
        refusal_details: dict = {"channel": exc.channel_address, "reason": exc.reason}
        if identity:
            refusal_details["active_target"] = identity
        make_error(
            "write_refused",
            message,
            suggestions,
            details=refusal_details,
        )
    except Exception as exc:
        logger.exception("%s failed", tool_name)
        identity = describe_active_target() if connector_name == "control_system" else None
        make_error(
            "internal_error",
            f"Unexpected error during {tool_name}{_target_clause(identity)}: {exc}",
            ["Check the MCP server logs for details."],
            details={"active_target": identity} if identity else None,
        )
