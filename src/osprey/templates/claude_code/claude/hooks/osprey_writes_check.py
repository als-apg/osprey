#!/usr/bin/env python3
"""
---
name: Writes Kill Switch
description: Blocks ALL write operations under a readonly session posture or writes kill switch
summary: Blocks write operations when the session is sandboxed or writes are disabled
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
         OSPREY_EXECUTION_MODE
         == "readonly"?   ──YES──► DENY: sandbox posture
              │
             NO
              │
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

First gate in the PreToolUse chain. Two independent reasons to refuse, in
order:

1. **Session posture.** `OSPREY_EXECUTION_MODE=readonly` means *this terminal
   session* was switched to the sandbox posture, whatever the deployment
   allows. Answered from the environment alone, ahead of any config read.
2. **Deployment kill switch.** `control_system.writes_enabled` in `config.yml`.
   When false, **all** channel writes and non-readonly Python executions are
   blocked before any other hook runs.

The two keep separate vocabularies: a posture refusal never points the operator
at `writes_enabled`, because flipping that key would not lift it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    get_hook_input,
    load_hook_config,
    load_osprey_config,
    log_hook,
)

_FALLBACK_WRITE_TOOLS = [
    "mcp__controls__channel_write",
    "mcp__python__execute",
]


def _get_write_tools():
    """Return the set of tool names subject to the writes kill switch.

    Loaded from hook_config.json (generated at deploy time).
    Falls back to framework defaults if missing — fail-closed.
    """
    tools = load_hook_config().get("write_tools", _FALLBACK_WRITE_TOOLS)
    return set(tools)


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect write tools
    if tool_name not in _get_write_tools():
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})

    # For execute: allow readonly even when writes disabled.
    # The server defaults execution_mode to "readonly", so treat missing as readonly.
    if tool_name == "mcp__python__execute":
        if tool_input.get("execution_mode", "readonly") == "readonly":
            sys.exit(0)

    # -- Session posture -------------------------------------------------
    # A web terminal session switched to the sandbox posture launches its agent
    # with OSPREY_EXECUTION_MODE=readonly, and this hook inherits it. The
    # posture belongs to *this session*, not to the deployment, so it is
    # answered from the environment alone — deliberately ahead of
    # load_osprey_config() so the answer never depends on config I/O, on the
    # config being parseable, or on PyYAML being importable at all.
    #
    # Value comparison, never a presence check (same semantics as the
    # executor's posture clamp and osprey_connectors' is_readonly_run): only
    # the exact "readonly" string sandboxes a session. "readwrite" is the
    # writes posture, and a presence check would sandbox on it.
    #
    # It sits *after* the execute-readonly exit above: a readonly execution is
    # exactly what a sandboxed session is for.
    #
    # Nothing here may raise. This hook fails OPEN — an uncaught exception
    # exits non-zero with no JSON and the tool proceeds — and for a mixed
    # read/write kernel this deny is the primary layer, since the renderer
    # re-grants those tools via `allow` and leans on the hard deny. So the one
    # call that touches the filesystem (the debug logger, which reads config
    # and appends to a JSONL file) is wrapped: a broken config costs a log
    # line, never the decision.
    if os.environ.get("OSPREY_EXECUTION_MODE") == "readonly":
        try:
            log_hook("writes-check", hook_input, status="deny", detail="reason=posture")
        except Exception:
            pass  # logging must never cost the deny
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "\U0001f512 SANDBOX POSTURE — this terminal session refuses "
                    "control-system writes.\n\n"
                    "Switch the session to writes posture from the terminal card; "
                    "config.yml is not the gate here."
                ),
            }
        }
        json.dump(output, sys.stdout)
        sys.exit(0)

    config = load_osprey_config(hook_input)
    writes_enabled = config.get("control_system", {}).get("writes_enabled", False)

    if writes_enabled:
        log_hook("writes-check", hook_input, status="allow")
        sys.exit(0)

    # Deny — writes are disabled. Emit a JSON `permissionDecision: deny`. This
    # is the canonical PreToolUse deny mechanism. Empirically: in the 2-hook
    # chain (`mcp__python__execute`: writes_check + approval), the deny here
    # combined with approval's defer (see osprey_approval.py) is enough to
    # suppress Claude Code's `permissions.ask` → `can_use_tool` callback. In
    # the 3-hook chain (`mcp__controls__channel_write`: writes_check + limits
    # + approval), this deny is NOT sufficient — the channel_write kill switch
    # is enforced by the renderer's `permissions.deny` augmentation in
    # `src/osprey/cli/templates/claude_code.py`. The deny emitted here is
    # defense-in-depth for that case (and primary for the execute case).
    log_hook("writes-check", hook_input, status="deny")
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
