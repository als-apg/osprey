#!/usr/bin/env python3
"""PreToolUse hook: Master writes kill switch.

Blocks ALL write operations when control_system.writes_enabled is false in config.yml.

PROMPT-PROVIDER: This hook contains facility-customizable static text:
  - Writes-disabled denial message (section=writes_disabled_message)
  Future: source from FrameworkPromptProvider.get_writes_disabled_message()
  Facility-customizable: message wording, instructions for enabling writes
"""

import json
import os
import sys
from pathlib import Path

import yaml


def load_osprey_config():
    config_path = Path(os.path.expandvars(
        os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml"))
    ))
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


WRITE_TOOLS = {
    "mcp__osprey-control-system__channel_write",
    "mcp__osprey-python-executor__python_execute",
}


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect write tools
    if tool_name not in WRITE_TOOLS:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})

    # For python_execute: allow readonly even when writes disabled
    if tool_name == "mcp__osprey-python-executor__python_execute":
        if tool_input.get("execution_mode") == "readonly":
            sys.exit(0)

    config = load_osprey_config()
    writes_enabled = config.get("control_system", {}).get("writes_enabled", False)

    if writes_enabled:
        sys.exit(0)

    # Deny — writes are disabled
    # PROMPT-PROVIDER: section=writes_disabled_message
    # Future: source from FrameworkPromptProvider.get_writes_disabled_message()
    # Facility-customizable: header, instructions, who to contact
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                "\U0001f512 WRITES DISABLED\n\n"
                "Control system writes are disabled in config.yml.\n"
                "Set control_system.writes_enabled: true to enable."
            ),
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
