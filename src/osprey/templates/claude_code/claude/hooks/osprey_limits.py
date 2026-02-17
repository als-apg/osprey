#!/usr/bin/env python3
"""PreToolUse hook: Channel limits validation.

Validates channel write values against the limits database before execution.

PROMPT-PROVIDER: This hook contains facility-customizable static text:
  - Limits violation denial message (section=limits_violation_message)
  Future: source from FrameworkPromptProvider.get_limits_violation_message()
  Facility-customizable: violation message wording, severity/tone,
  additional instructions (e.g., "contact shift supervisor")
"""

import json
import sys


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only validate channel_write
    if tool_name != "mcp__osprey-control-system__channel_write":
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})

    # Try to import LimitsValidator; if unavailable, allow
    try:
        from osprey.services.python_executor.execution.limits_validator import (
            LimitsValidator,
        )
    except ImportError:
        # osprey not installed — allow and let the MCP tool handle it
        sys.exit(0)

    validator = LimitsValidator.from_config()
    if validator is None:
        # Limits validation disabled in config
        sys.exit(0)

    # Collect operations to validate. Support both single and batch writes.
    operations = tool_input.get("operations", [])
    if not operations:
        # Single-write form
        channel = tool_input.get("channel")
        value = tool_input.get("value")
        if channel is not None and value is not None:
            operations = [{"channel": channel, "value": value}]

    if not operations:
        sys.exit(0)

    violations = []
    for op in operations:
        channel = op.get("channel", "")
        value = op.get("value")
        try:
            validator.validate(channel, value)
        except Exception as exc:
            violations.append(f"  {channel}={value}: {exc}")

    if not violations:
        sys.exit(0)

    # Deny — limits violated
    # PROMPT-PROVIDER: section=limits_violation_message
    # Future: source from FrameworkPromptProvider.get_limits_violation_message()
    # Facility-customizable: header, detail format, footer instructions
    violation_text = "\n".join(violations)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "\U0001f6ab CHANNEL LIMITS VIOLATION\n\n"
                "The following operations violate configured limits:\n"
                f"{violation_text}\n\n"
                "These writes have been BLOCKED for safety."
            ),
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
