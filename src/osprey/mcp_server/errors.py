"""Structured error envelope for MCP tool responses."""


def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
    details: dict | list | None = None,
) -> dict:
    """Build the cross-team standard error envelope.

    All MCP tools must return this shape on failure so that Claude Code
    can reliably detect and display errors.

    Args:
        error_type: Machine-readable error category (e.g. "limits_violation").
        error_message: Human-readable summary of the error.
        suggestions: Actionable next steps for the operator/agent.
        details: Structured data about the error (e.g. violation specifics,
                 channel limits). Included only if provided.
    """
    result: dict = {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }
    if details is not None:
        result["details"] = details
    return result
