"""Shared error handling for control-system MCP tools.

Provides an async context manager that wraps the common
ConnectionError / TimeoutError / Exception pattern used by
channel_read, channel_write, and archiver_read.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.control_system.error_handling")


class ToolError(Exception):
    """Raised inside the context manager to short-circuit with an error response."""

    def __init__(self, response: str) -> None:
        self.response = response
        super().__init__(response)


@asynccontextmanager
async def connector_error_handler(
    tool_name: str,
    connector_name: str = "control_system",
) -> AsyncIterator[None]:
    """Wrap a control-system tool body with standardized error handling.

    Usage::

        async with connector_error_handler("channel_read") as _:
            registry = get_server_context()
            connector = await registry.control_system()
            # ... tool logic ...
            return json.dumps(result)

    On ConnectionError, the connector is invalidated and a structured
    error response is raised as a ``ToolError``.
    """
    try:
        yield
    except ToolError:
        raise  # Already a formatted response — re-raise as-is
    except ConnectionError as exc:
        from osprey.mcp_server.control_system.server_context import get_server_context

        registry = get_server_context()
        await registry.invalidate_connector(connector_name)
        raise ToolError(
            json.dumps(
                make_error(
                    "connection_error",
                    f"Failed to connect to the {connector_name.replace('_', ' ')}: {exc}",
                    [
                        f"Check that the {connector_name.replace('_', ' ')} is running.",
                        f"Verify config.yml {connector_name} settings.",
                    ],
                )
            )
        ) from exc
    except TimeoutError as exc:
        raise ToolError(
            json.dumps(
                make_error(
                    "timeout_error",
                    f"{tool_name} timed out: {exc}",
                    ["Check network connectivity.", "Try a smaller request."],
                )
            )
        ) from exc
    except Exception as exc:
        logger.exception("%s failed", tool_name)
        raise ToolError(
            json.dumps(
                make_error(
                    "internal_error",
                    f"Unexpected error during {tool_name}: {exc}",
                    ["Check the MCP server logs for details."],
                )
            )
        ) from exc
