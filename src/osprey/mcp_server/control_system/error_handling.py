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

from osprey.errors import ChannelLimitsViolationError
from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.control_system.error_handling")


__all__ = ["ToolError", "connector_error_handler"]


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
    """
    try:
        yield
    except ToolError:
        raise  # Already a formatted envelope — propagate as-is
    except ConnectionError as exc:
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        await registry.invalidate_connector(connector_name)
        make_error(
            "connection_error",
            f"Failed to connect to the {connector_name.replace('_', ' ')}: {exc}",
            [
                f"Check that the {connector_name.replace('_', ' ')} is running.",
                f"Verify config.yml {connector_name} settings.",
            ],
        )
    except TimeoutError as exc:
        make_error(
            "timeout_error",
            f"{tool_name} timed out: {exc}",
            ["Check network connectivity.", "Try a smaller request."],
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
    except Exception as exc:
        logger.exception("%s failed", tool_name)
        make_error(
            "internal_error",
            f"Unexpected error during {tool_name}: {exc}",
            ["Check the MCP server logs for details."],
        )
