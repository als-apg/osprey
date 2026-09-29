"""Errors the agent runner raises to its callers."""

from __future__ import annotations

from typing import Any


class AgentRunError(RuntimeError):
    """The agent SDK failed while a run was in flight.

    ``str()`` is the SDK error's own message and ``__cause__`` is that error, so
    a classifier that walks the exception chain by type name still sees it.

    Attributes:
        error_type: The SDK error's class name (``"CLIConnectionError"``,
            ``"ProcessError"``, …).
    """

    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class McpNotReadyError(RuntimeError):
    """An MCP server the run requires was not connected before the first turn,
    so the run was refused rather than started without its tools.

    Attributes:
        servers: The MCP status snapshot the readiness barrier ended on.
        missing: Names of the expected servers that were not connected, sorted.
    """

    def __init__(self, message: str, *, servers: list[dict[str, Any]], missing: list[str]) -> None:
        super().__init__(message)
        self.servers = servers
        self.missing = missing
