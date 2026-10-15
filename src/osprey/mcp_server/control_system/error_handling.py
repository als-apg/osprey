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

from osprey.errors import (
    ChannelLimitsViolationError,
    ChannelNotFoundError,
    ChannelWriteBlockedError,
)
from osprey.mcp_server.errors import make_error
from osprey_connectors.ipc.pool import DEFAULT_PING_TIMEOUT_S
from osprey_connectors.ipc.proxy import ChildUnresponsiveError

logger = logging.getLogger("osprey.mcp_server.control_system.error_handling")


__all__ = [
    "ToolError",
    "connector_error_handler",
    "describe_active_target",
    "invalidate_active_connector",
]


def describe_active_target() -> dict[str, str] | None:
    """Identify the control-system target the deployment is on, or ``None``.

    A failure envelope that says "the write to ``SR:...:SP`` timed out" is a
    materially different situation on the live machine than on the simulator,
    and the operator should not have to reconstruct which one it was from
    session memory. This resolves ``{name, label, endpoint}`` for the active
    target from the supervisor alone: its target of record, and the endpoint
    the writer last rendered for that target. Re-deriving that endpoint from
    config here would name the write gateway no matter which one the session
    is actually talking through, so the envelope reads the writer's rendering
    instead of forming a second opinion beside it.

    Fail-soft by construction: an envelope that raises while describing a
    failure is worse than one that omits the target, so every problem here —
    no server context yet, a config the deriver rejects, anything — collapses
    to ``None`` and the envelope renders exactly as it did before this existed.
    """
    try:
        from osprey.mcp_server.control_system.server_context import get_server_context

        context = get_server_context()
        target = context.connector_hosts.active_target()
        meta = context.connector_hosts.display_metadata().get(target) or {}
        identity = {"name": target, "label": str(meta.get("label") or target)}
        endpoint = str(meta.get("endpoint") or "")
        if endpoint:
            identity["endpoint"] = endpoint
        return identity
    except Exception:  # see docstring: never fail an envelope
        return None


#: What an answer for an address outside the facility tells the agent first.
NOT_READ_OR_WRITTEN = "Nothing was read from or written to this address."

#: Where the agent finds a real address instead.
LOOK_IT_UP = (
    "Look the channel up among the facility's channels (the channel finder, where this "
    "deployment has one) instead of guessing an address."
)


def _target_clause(identity: dict[str, str] | None) -> str:
    """The human half: `` (active target: LIVE MACHINE at 127.0.0.1:5064)``."""
    if not identity:
        return ""
    where = f" at {identity['endpoint']}" if identity.get("endpoint") else ""
    return f" (active target: {identity['label']}{where})"


#: One rendered envelope: ``(error_type, error_message, suggestions, details)``.
_Envelope = tuple[str, str, list[str], dict | None]


def _not_a_channel(tool_name: str, address: str, connector_name: str) -> _Envelope:
    """Answer an address that is not a channel of the facility as ``not_found``.

    The one rendering for both triggers: a read that raised
    :class:`ChannelNotFoundError` and a write refused with its reason. Nothing
    was denied, so the answer names no refuser.
    """
    identity = describe_active_target() if connector_name == "control_system" else None
    error_type = "not_found"
    return (
        error_type,
        f"Unknown channel during {tool_name}{_target_clause(identity)}: "
        f"{ChannelNotFoundError(address)}",
        [NOT_READ_OR_WRITTEN, LOOK_IT_UP],
        {"channel": address, **({"active_target": identity} if identity else {})},
    )


async def invalidate_active_connector(connector_name: str) -> None:
    """Drop the connector a failed call was using, so the next call rebuilds it.

    This is the one seam between the ``ConnectionError`` branch below and
    whoever owns the connector's lifetime. It delegates to
    :meth:`ServerContext.invalidate_connector`, which recreates an in-process
    connector, or respawns a connector-host child on the same target, so the
    envelope rendering never has to know where connectors live.
    """
    from osprey.mcp_server.control_system.server_context import get_server_context

    await get_server_context().invalidate_connector(connector_name)


async def _answer_unresponsive_child(connector_name: str) -> str:
    """Ping the active child after a call missed its deadline; replace it if silent.

    A child that answers the ping is alive and merely slow, so it is kept: the
    call may yet complete in it. A child that answers nothing is wedged, and it
    is invalidated, which respawns the connector host on the same target with
    the generation unchanged. The ping goes to the child active now, which may
    not be the one that missed the deadline if a switch ran in between; a
    child that answers is never replaced, so that ping harms nothing. Nothing
    is retried.

    Returns:
        The sentence the timeout envelope appends to say which happened, or
        ``""`` when the control system is not served from a child.
    """
    if connector_name != "control_system":
        return ""
    try:
        from osprey.mcp_server.control_system.server_context import get_server_context

        context = get_server_context()
        if not context.switch_capable:
            return ""
        proxy = context.connector_hosts.active_proxy()
    except Exception:  # no server context: there is no child to answer for
        return ""
    if proxy is None:
        return ""
    try:
        await proxy.supervisor_request("ping", {}, DEFAULT_PING_TIMEOUT_S)
    except Exception:
        await invalidate_active_connector(connector_name)
        return (
            " The connector-host child did not answer a ping and was replaced on the same "
            "target; the call was not retried."
        )
    return (
        " The connector-host child still answers a ping, so it was kept; the call may yet "
        "complete in it and was not retried."
    )


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

    On known errors (Connection / Timeout / unknown channel / limits
    violation / write refusal / unhandled exception), ``make_error()`` raises a ``fastmcp.ToolError`` carrying the
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
    except Exception as exc:
        error_type, message, suggestions, details = await _envelope(exc, tool_name, connector_name)
        make_error(error_type, message, suggestions, details=details)


async def _envelope(exc: BaseException, tool_name: str, connector_name: str) -> _Envelope:
    """Render ``exc`` as the envelope its class is answered with, running its side effects.

    A ``ConnectionError`` drops the connector it came from, and a timed-out call
    into the connector-host child pings that child, before the envelope is
    built; each runs once per rendered envelope.
    """
    if isinstance(exc, ConnectionError):
        await invalidate_active_connector(connector_name)
        identity = describe_active_target() if connector_name == "control_system" else None
        if identity:
            where = f" at {identity['endpoint']}" if identity.get("endpoint") else ""
            reach_suggestion = f"Check that the {identity['label']}{where} is reachable."
        else:
            reach_suggestion = f"Check that the {connector_name.replace('_', ' ')} is running."
        error_type = "connection_error"
        return (
            error_type,
            f"Failed to connect to the {connector_name.replace('_', ' ')}"
            f"{_target_clause(identity)}: {exc}",
            [
                reach_suggestion,
                f"If the {connector_name} settings are wrong, fix them in the build profile "
                "(profile.yml on the host), then rebuild and redeploy.",
            ],
            {"active_target": identity} if identity else None,
        )
    if isinstance(exc, ChildUnresponsiveError):
        outcome = await _answer_unresponsive_child(connector_name)
        identity = describe_active_target() if connector_name == "control_system" else None
        error_type = "timeout_error"
        return (
            error_type,
            f"{tool_name} timed out{_target_clause(identity)}: {exc}"
            + (f".{outcome}" if outcome else ""),
            ["Check network connectivity.", "Try a smaller request."],
            {"active_target": identity} if identity else None,
        )
    if isinstance(exc, TimeoutError):
        identity = describe_active_target() if connector_name == "control_system" else None
        error_type = "timeout_error"
        return (
            error_type,
            f"{tool_name} timed out{_target_clause(identity)}: {exc}",
            ["Check network connectivity.", "Try a smaller request."],
            {"active_target": identity} if identity else None,
        )
    if isinstance(exc, ChannelNotFoundError):
        return _not_a_channel(tool_name, exc.channel_address, connector_name)
    if isinstance(exc, ChannelLimitsViolationError):
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

        error_type = "limits_violation"
        return (
            error_type,
            msg,
            [
                "Do NOT attempt to work around this limit.",
                "Report the violation to the operator with the allowed range.",
            ],
            violation,
        )
    if isinstance(exc, ChannelWriteBlockedError):
        if exc.reason == ChannelNotFoundError.reason:
            return _not_a_channel(tool_name, exc.channel_address, connector_name)
        # Every refusal put no value on the channel, but who refused decides
        # where the operator goes next. A control-system denial reaches here
        # only after OSPREY sent the write and the control system said no;
        # telling the operator it "was never sent" would point them at OSPREY's
        # own policy settings for something only the control system can grant.
        # A reason that names no refuser is reported without one.
        identity = describe_active_target() if connector_name == "control_system" else None
        refuser = ChannelWriteBlockedError.refuser(exc.reason)
        where = ""
        if exc.reason == "CONTROL_SYSTEM_REFUSED":
            where = _target_clause(identity)
            suggestions = [
                "The control system itself denied this write (access security); "
                "no value was written.",
                "Do NOT attempt to work around the refusal.",
                "Report it to the operator: write access for this channel is granted "
                "by the control system, not by OSPREY.",
            ]
        elif refuser is None:
            suggestions = ["Do NOT attempt to work around the refusal."]
        else:
            suggestions = [
                "This write was refused on policy grounds; it was never sent to the control system.",
                "Do NOT attempt to work around the refusal.",
            ]
        by_refuser = f" by {refuser}" if refuser else ""
        message = f"Write refused{by_refuser} during {tool_name}{where}: {exc}"
        refusal_details: dict = {"channel": exc.channel_address, "reason": exc.reason}
        if identity:
            refusal_details["active_target"] = identity
        error_type = "write_refused"
        return error_type, message, suggestions, refusal_details
    logger.exception("%s failed", tool_name, exc_info=exc)
    identity = describe_active_target() if connector_name == "control_system" else None
    error_type = "internal_error"
    return (
        error_type,
        f"Unexpected error during {tool_name}{_target_clause(identity)}: {exc}",
        ["Check the MCP server logs for details."],
        {"active_target": identity} if identity else None,
    )
