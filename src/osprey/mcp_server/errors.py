"""Structured error envelope for MCP tool responses."""

from __future__ import annotations

import json

from mcp.types import CallToolResult, TextContent


def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
    details: dict | list | None = None,
) -> CallToolResult:
    """Build a CallToolResult error with the cross-team standard envelope.

    All MCP tools must return this on failure so that the SDK's ``isError``
    flag is meaningful AND the model still receives the structured envelope
    (``error_type``, ``error_message``, ``suggestions``, ``details``) in the
    text content for downstream guidance hooks.

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
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(envelope))],
        isError=True,
    )


def extract_error_envelope(result: CallToolResult) -> dict | None:
    """Reverse of make_error: pull the structured envelope out of a CallToolResult.

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
