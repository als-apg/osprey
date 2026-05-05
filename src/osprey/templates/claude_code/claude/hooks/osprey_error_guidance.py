#!/usr/bin/env python3
"""
---
name: Error Guidance
description: Injects error-handling protocol guidance when an OSPREY tool returns a structured error
summary: Injects error-handling guidance into tool error responses
event: PostToolUse
tools: all OSPREY MCP tools
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Is OSPREY tool?  ──NO──► EXIT (silent)
              │
             YES
              │
              ▼
         Parse tool_response
              │
              ▼
         Detect error envelope
         {"error": true, ...}
              │
              ▼
         Error found?  ──NO──► EXIT (silent)
              │
             YES
              │
              ▼
         Map error_type to class
         (Connection/Validation/
          Data/Execution/Internal)
              │
              ▼
         Inject additionalContext:
         error class + protocol ref
```

## Details

Never blocks execution — only adds `additionalContext` pointing Claude to
the error-handling protocol in `.claude/rules/error-handling.md`. Detects
the standard OSPREY error envelope (`{"error": true, "error_type": ...}`)
and falls back to keyword detection for non-JSON responses.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, load_hook_config, log_hook

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

    OSPREY tools now return ``CallToolResult(isError=True, ...)`` for every
    failure. The PostToolUse hook input surfaces this as a dict with
    ``isError: true`` and a ``content`` list of text blocks; the first text
    block carries the structured envelope JSON. Returns
    ``(error_class, error_message)`` or ``(None, None)`` if no error detected.
    """
    if not isinstance(tool_response, dict) or tool_response.get("isError") is not True:
        return None, None

    for block in tool_response.get("content", []) or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        try:
            parsed = json.loads(block.get("text", ""))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("error") is True:
            error_type = parsed.get("error_type", "unknown")
            return (
                ERROR_CLASS_MAP.get(error_type, "Internal"),
                parsed.get("error_message", str(parsed)),
            )

    # isError=True but no structured envelope (shouldn't happen for OSPREY
    # tools, but cover the case so guidance still fires).
    return "Internal", "Tool returned an error"


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect OSPREY tools (prefixes loaded from hook_config.json)
    _prefixes = load_hook_config().get("server_prefixes", [])
    if not any(tool_name.startswith(p) for p in _prefixes):
        sys.exit(0)

    tool_response = hook_input.get("tool_response")
    error_class, error_message = _detect_error(tool_response)

    if error_class is None:
        log_hook("error-guidance", hook_input, status="no-error")
        sys.exit(0)

    # Inject guidance reminder
    guidance = (
        f"ERROR DETECTED [{error_class}]: {error_message}\n\n"
        "Follow the error-handling protocol (.claude/rules/error-handling.md):\n"
        "- Report the error clearly to the user with actionable next steps.\n"
        "- Do NOT debug infrastructure, write mock data, or work around the failure.\n"
        "- Do NOT retry unless the error explicitly indicates a transient condition."
    )

    log_hook("error-guidance", hook_input, status="error", detail=f"class={error_class}")

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
