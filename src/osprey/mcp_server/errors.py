"""Structured error envelope for MCP tool responses."""

from __future__ import annotations

import json
from typing import NoReturn

from fastmcp.exceptions import ToolError
from mcp.types import CallToolResult


def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
    details: dict | list | None = None,
) -> NoReturn:
    """Raise a ``fastmcp.ToolError`` carrying the cross-team standard envelope.

    fastmcp converts a raised ``ToolError`` into a ``CallToolResult`` with
    ``isError=True`` and the exception message verbatim as a ``TextContent``
    block. Returning a ``CallToolResult`` directly does *not* set ``isError``
    correctly under fastmcp's structured-output wrapping (see PR for the
    reproduction); raising is the only path that produces a clean
    error response on the wire.

    Args:
        error_type: Machine-readable error category (e.g. "limits_violation").
        error_message: Human-readable summary of the error.
        suggestions: Actionable next steps for the operator/agent.
        details: Structured data about the error (e.g. violation specifics,
                 channel limits). Included only if provided.
    """
    envelope: dict = {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }
    if details is not None:
        envelope["details"] = details
    raise ToolError(json.dumps(envelope))


def extract_error_envelope(result: CallToolResult) -> dict | None:
    """Pull the structured envelope out of a wire-form error CallToolResult.

    Returns ``None`` when the result is not an error or its content cannot be
    parsed as the envelope dict. Useful for tests and hooks that need to
    inspect the structured payload behind ``isError=True``.
    """
    if not isinstance(result, CallToolResult):
        return None
    if result.isError is not True:
        return None
    for block in result.content or []:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("error") is True:
            return parsed
    return None
