#!/usr/bin/env python3
"""
---
name: Plan Mode Guard
description: Refuses every OSPREY tool that acts while the session is in plan mode
summary: Refuses the tools that act while the session is in plan mode
event: PreToolUse
tools: mcp__.*
safety_layer: 0
wiring: standalone
timeout: 5
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         permission_mode
         exactly "plan"?  ──NO──► EXIT (no opinion)
              │
             YES
              │
              ▼
         plan_mode_denied_tools
         a readable list?  ──NO──► EXIT (no opinion)
              │
             YES
              │
              ▼
         Tool on the list? ──NO──► EXIT (no opinion)
              │
             YES
              │
              ▼
         DENY: off while planning
```

## Details

In plan mode the session reads and plans; it does not act. Which tools act is
rendered data: `plan_mode_denied_tools` in `hook_config.json`, derived at build
time from the registry's side-effect classification of every resolved server,
so this file names no tool. Entries are exact tool names plus the
`mcp__<server>__.*` form a custom server that gates its own writes contributes.

Every case this hook cannot read with certainty — another mode, a missing or
malformed list, an unreadable config, a tool name that is not a string — leaves
the call to the rest of the chain, exactly as if this hook were absent.

The approval hook imports `plan_mode_denies` and stays silent for exactly the
calls refused here, because an approval prompt would win over this deny.

A refusal is written to the audit trail (`emit_audit`, surface
`hook_plan_mode`) before the deny is emitted, and can never cost it.
"""

import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    AUDIT_DECISION_REFUSED,
    emit_audit,
    get_hook_input,
    is_write_tool,
    load_hook_config,
    log_hook,
    short_tool_name,
)

#: The one ``permission_mode`` value this hook acts on, compared exactly.
PLAN_MODE = "plan"

#: The ``hook_config.json`` key holding the tools plan mode refuses.
PLAN_MODE_DENIED_KEY = "plan_mode_denied_tools"

#: What the agent is told, filled with the tool's short name.
PLAN_MODE_DENY_REASON = "Plan mode: {tool} is off while planning. Leave plan mode to run it."

#: The audit reason word for a plan-mode refusal.
_AUDIT_REASON = "plan_mode"


def plan_mode_denies(hook_input):
    """True when the session is in plan mode and *hook_input*'s tool is on the list.

    True only when *hook_input* is a mapping, its ``permission_mode`` is exactly
    :data:`PLAN_MODE`, its ``tool_name`` is a string, the rendered
    :data:`PLAN_MODE_DENIED_KEY` is a list, and that list covers the tool (an
    exact name, or an ``mcp__<server>__.*`` entry). Every other case is False:
    the hook then says nothing and the rest of the chain decides, exactly as
    it would without this hook.
    """
    if not isinstance(hook_input, dict):
        return False
    if hook_input.get("permission_mode") != PLAN_MODE:
        return False
    tool_name = hook_input.get("tool_name")
    if not isinstance(tool_name, str):
        return False
    config = load_hook_config()
    if not isinstance(config, dict):
        return False
    denied = config.get(PLAN_MODE_DENIED_KEY)
    if not isinstance(denied, list):
        return False
    return is_write_tool(tool_name, denied)


def main():
    hook_input = get_hook_input()
    if not hook_input or not plan_mode_denies(hook_input):
        sys.exit(0)

    tool_name = hook_input["tool_name"]
    prefixes = load_hook_config().get("server_prefixes", [])
    short_name = short_tool_name(tool_name, prefixes if isinstance(prefixes, list) else [])

    # logging must never cost the deny
    with contextlib.suppress(Exception):
        log_hook("plan_mode", hook_input, status="deny", detail=f"tool={short_name}")
    # the audit trail must never cost the deny
    with contextlib.suppress(Exception):
        emit_audit(
            "plan_mode",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=tool_name,
            reason=_AUDIT_REASON,
        )
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": PLAN_MODE_DENY_REASON.format(tool=short_name),
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
