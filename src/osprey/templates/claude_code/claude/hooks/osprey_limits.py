#!/usr/bin/env python3
"""
---
name: Channel Limits Validator
description: Validates channel write values against the limits database before execution
summary: Validates channel write values against the limits database
event: PreToolUse
tools: channel_write
safety_layer: 3
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Is channel_write? ──NO──► EXIT (allow)
              │
             YES
              │
              ▼
         Import LimitsValidator
              │
              ▼
         validator = from_config()
              │
              ▼
         validator exists? ──NO──► EXIT (allow)
              │
             YES
              │
              ▼
         Collect operations
         (single or batch)
              │
              ▼
         Validate each op
              │
              ▼
         Violations found? ──NO──► EXIT (allow)
              │
             YES
              │
              ▼
         DENY: limits violated
```

## Details

Validates every channel write against min/max/step/writable constraints
from the limits database. Supports both single-write and batch-write forms.
If `LimitsValidator` is not importable or not configured, allows the write
through (fail-open for environments without limits).

PROMPT-PROVIDER: This hook contains facility-customizable static text:
  - Limits violation denial message (section=limits_violation_message)
  Future: source from FrameworkPromptProvider.get_limits_violation_message()
  Facility-customizable: violation message wording, severity/tone,
  additional instructions (e.g., "contact shift supervisor")
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, log_hook


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only validate channel_write
    if tool_name != "mcp__controls__channel_write":
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})

    # Try to import LimitsValidator; if unavailable, allow
    try:
        from osprey.connectors.control_system.limits_validator import (
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
        log_hook("limits", hook_input, status="allow")
        sys.exit(0)

    # Deny — limits violated
    log_hook("limits", hook_input, status="deny", detail=f"violations={len(violations)}")
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
