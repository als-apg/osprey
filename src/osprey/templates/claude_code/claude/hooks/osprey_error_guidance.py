#!/usr/bin/env python3
"""PostToolUse hook: Error-handling guidance injection.

When an OSPREY MCP tool returns a structured error (the cross-team standard
``{"error": true, "error_type": ...}`` envelope), this hook injects a short
reminder into Claude's context pointing it to the error-handling protocol.

This hook NEVER blocks — it only adds ``additionalContext`` on error.
On success (no error detected), it produces no output at all.
"""

import json
import sys

OSPREY_PREFIXES = (
    "mcp__osprey-control-system__",
    "mcp__osprey-python-executor__",
    "mcp__osprey-workspace__",
    "mcp__ariel__",
)

# Map OSPREY error_type values to short human-readable classes.
# Matches the taxonomy in .claude/rules/error-handling.md.
ERROR_CLASS_MAP = {
    "connection_error": "Connection",
    "timeout_error": "Connection",
    "validation_error": "Validation",
    "limits_violation": "Validation",
    "not_found": "Data",
    "no_results": "Data",
    "execution_error": "Execution",
    "internal_error": "Internal",
    "platform_error": "Internal",
}


def _detect_error(tool_response: str | dict | None) -> tuple[str | None, str | None]:
    """Detect a structured error in the tool response.

    Returns (error_class, error_message) or (None, None) if no error detected.
    """
    if tool_response is None:
        return None, None

    # tool_response may be a JSON string or an already-parsed dict
    if isinstance(tool_response, str):
        try:
            parsed = json.loads(tool_response)
        except (json.JSONDecodeError, ValueError):
            # Not JSON — check for common error substrings as a fallback
            lower = tool_response.lower()
            if any(kw in lower for kw in ("error", "failed", "traceback", "exception")):
                return "Internal", tool_response[:200]
            return None, None
    elif isinstance(tool_response, dict):
        parsed = tool_response
    else:
        return None, None

    # Standard OSPREY error envelope: {"error": true, "error_type": ..., "error_message": ...}
    if parsed.get("error") is True:
        error_type = parsed.get("error_type", "unknown")
        error_class = ERROR_CLASS_MAP.get(error_type, "Internal")
        error_message = parsed.get("error_message", str(parsed))
        return error_class, error_message

    return None, None


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect OSPREY tools
    if not any(tool_name.startswith(p) for p in OSPREY_PREFIXES):
        sys.exit(0)

    tool_response = hook_input.get("tool_response")
    error_class, error_message = _detect_error(tool_response)

    if error_class is None:
        # No error — silent exit, no output
        sys.exit(0)

    # Inject guidance reminder
    guidance = (
        f"ERROR DETECTED [{error_class}]: {error_message}\n\n"
        "Follow the error-handling protocol (.claude/rules/error-handling.md):\n"
        "- Report the error clearly to the user with actionable next steps.\n"
        "- Do NOT debug infrastructure, write mock data, or work around the failure.\n"
        "- Do NOT retry unless the error explicitly indicates a transient condition."
    )

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": guidance,
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
