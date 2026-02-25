#!/usr/bin/env python3
"""
---
name: Writes Kill Switch
description: Blocks ALL write operations when control_system.writes_enabled is false
summary: Blocks write operations when writes are disabled
event: PreToolUse
tools: channel_write, execute
safety_layer: 1
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Is write tool?  ──NO──► EXIT (allow)
              │
             YES
              │
              ▼
         execute          ──YES──► readonly mode? ──YES──► EXIT (allow)
         tool?                          │
              │                        NO
             NO                         │
              │◄────────────────────────┘
              ▼
         Load config.yml
              │
              ▼
         writes_enabled?  ──YES──► EXIT (allow)
              │
             NO
              │
              ▼
         DENY: writes disabled
```

## Details

First gate in the PreToolUse chain. Checks `control_system.writes_enabled`
in `config.yml`. When false, **all** channel writes and non-readonly Python
executions are blocked before any other hook runs.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, get_project_dir, log_hook


def load_osprey_config(project_dir=""):
    default = (
        str(Path(project_dir) / "config.yml") if project_dir else str(Path.cwd() / "config.yml")
    )
    config_path = Path(os.path.expandvars(os.environ.get("OSPREY_CONFIG", default)))
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


WRITE_TOOLS = {
    "mcp__controls__channel_write",
    "mcp__python__execute",
}


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect write tools
    if tool_name not in WRITE_TOOLS:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})

    # For execute: allow readonly even when writes disabled
    if tool_name == "mcp__python__execute":
        if tool_input.get("execution_mode") == "readonly":
            sys.exit(0)

    project_dir = get_project_dir(hook_input)
    config = load_osprey_config(project_dir)
    writes_enabled = config.get("control_system", {}).get("writes_enabled", False)

    if writes_enabled:
        log_hook("writes-check", hook_input, status="allow")
        sys.exit(0)

    # Deny — writes are disabled
    log_hook("writes-check", hook_input, status="deny")
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
