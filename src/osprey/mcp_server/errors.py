"""Structured error envelope for MCP tool responses."""


def make_error(
    error_type: str,
    error_message: str,
    suggestions: list[str] | None = None,
) -> dict:
    """Build the cross-team standard error envelope.

    All MCP tools must return this shape on failure so that Claude Code
    can reliably detect and display errors.
    """
    return {
        "error": True,
        "error_type": error_type,
        "error_message": error_message,
        "suggestions": suggestions or [],
    }
