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
         Is write tool?  ──NO──► readonly mode? ──NO──► EXIT (allow)
              │                        │
             YES                      YES
              │                        │
              │                        ▼
              │                   write_servers
              │                   prefix hit?   ──NO──► EXIT (allow)
              │                        │
              │                       YES
              │                        │
              │                        ▼
              │                   DENY: sandbox posture
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

## Server-level coverage (`write_servers`)

A facility-custom server that opts into `writes_check` has tool names the render
cannot know, so the registry gives it a REGEX matcher (`mcp__<name>__.*`). That
matcher reaches `write_tools` like any other, where the exact-name membership
test above can never match a real call against it. The render therefore also
names the server in `write_servers`, and this hook covers it by prefix.

Only on the posture path. An exact-name miss whose `mcp__<server>__` prefix is
listed refuses under `OSPREY_EXECUTION_MODE=readonly` — before any config read —
and exits 0 on every other posture. Two reasons for that asymmetry, both
deliberate:

- Nothing in the render knows which of a custom server's tools write, so
  server-level coverage necessarily takes the reads with it. A sandboxed session
  is the one posture where refusing the whole server is the honest answer.
- Letting the prefix reach the writes-off kill switch would turn a deployment
  that merely disabled writes into one whose custom-server *reads* all fail. So
  that branch never sees these servers, and their reads keep working exactly as
  they do today.

Degrade, deliberately different from `write_tools`: an absent or malformed
`write_servers` yields the empty list — no coverage, pre-feature behavior —
because server names are deployment-specific and no floor shipped here could
name the right ones. A `hook_config.json` that parsed but says nothing about
the key is a stale render, so that branch warns once on stderr and names
`osprey build` as the remedy.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    AUDIT_DECISION_REFUSED,
    emit_audit,
    get_hook_input,
    load_hook_config,
    load_osprey_config,
    log_hook,
)

#: The write tools refused when hook_config.json cannot be read. Covers EVERY
#: write-gated tool the framework ships, not the ones a default render happens
#: to enable: the bluesky plan-queue arming pair used to be left out because
#: bluesky is opt-in, which has it backwards — a deployment that opted in is
#: precisely the one whose sandboxed session can reach those tools, and a
#: degraded render is exactly when nothing else is left to refuse them.
#: Deliberately the same literal as the MCP audit middleware's floor of the
#: same name (the two layers must not disagree about what a degraded deployment
#: refuses), and both are pinned against registry.mcp.framework_write_tools()
#: by tests/registry/test_mixed_floor_driftguard.py so registry growth cannot
#: strand them. Not imported from either: this file ships as a copied-in
#: project template run by a bare python3 and imports nothing from osprey.
_FALLBACK_WRITE_TOOLS = [
    "mcp__bluesky__queue_add",
    "mcp__bluesky__queue_start",
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


#: One WARNING per process, and a hook process handles exactly one tool call —
#: so this is "once per stale call", not once per session. Cheap, and it keeps
#: the message in front of the operator until the render is refreshed.
_warned_write_servers = False


def _warn_stale_write_servers():
    """Say that this render predates server-level coverage, once, on stderr.

    stderr, never stdout: Claude Code parses a PreToolUse decision off stdout,
    and a warning printed there would make a deny unreadable — reporting a stale
    render by failing the call open.
    """
    global _warned_write_servers
    if _warned_write_servers:
        return
    _warned_write_servers = True
    try:
        print(
            "WARNING: [writes-check] hook_config.json names no usable "
            "'write_servers'. Facility-custom MCP servers get no server-level "
            "coverage under the sandbox posture. Re-render with `osprey build`.",
            file=sys.stderr,
        )
    except Exception:
        pass  # a warning must never cost a decision


def _get_write_servers():
    """Bare server names whose tools the sandbox posture refuses wholesale.

    No fallback floor, unlike :func:`_get_write_tools`: server names are
    deployment-specific, so the only honest answer for a render that does not
    supply them is the empty list.

    The warning fires on a hook_config that *parsed to something* yet has no
    usable list — a stale render, which `osprey build` fixes. A config the hook
    could not read at all (missing or malformed file) is a different failure:
    ``load_hook_config`` answers ``{}`` there, ``write_tools`` already covers it
    with its fail-closed floor and no warning, and both keys are emitted by the
    same render — so it says nothing about this key specifically and warning on
    it would fire on every framework write call too.
    """
    config = load_hook_config()
    names = config.get("write_servers")
    if isinstance(names, list):
        return [name for name in names if isinstance(name, str) and name]
    if config:
        _warn_stale_write_servers()
    return []


def _write_server_for(tool_name):
    """The ``write_servers`` entry owning *tool_name*, or ``None``.

    Prefix match on the composed ``mcp__<server>__``. The trailing ``__`` is what
    keeps a listed ``sitectl`` from swallowing a neighbouring ``sitectl_extra``.

    Wrapped: this runs on the posture path, which must reach its deny without
    raising (an uncaught exception exits non-zero, prints no JSON, and the tool
    proceeds). A failure here costs the coverage this key adds — the same place
    an unrendered key leaves it — never a traceback.
    """
    try:
        for server in _get_write_servers():
            if tool_name.startswith("mcp__" + server + "__"):
                return server
    except Exception:
        return None
    return None


#: The posture refusal, said once. Both posture branches — exact-name write tool
#: and server-level prefix — are lifted by the same action, so they say the same
#: thing rather than teaching the operator two dialects.
_POSTURE_DENY_REASON = (
    "\U0001f512 SANDBOX POSTURE — this terminal session refuses "
    "control-system writes.\n\n"
    "Switch the session to writes posture from the terminal card; "
    "config.yml is not the gate here."
)


#: The machine-ish reason both posture refusals record. Deliberately the same
#: word the MCP audit middleware and the python executor's in-tool session
#: clamp record for the same refusal, so a sandboxed session's records join
#: across all three layers on one spelling. A cross-layer test pins them
#: together, reading this literal by AST — the hook imports nothing from
#: osprey, so it cannot share the constant itself.
_POSTURE_DENY_AUDIT_REASON = "posture"

#: What the kill-switch refusal records. A different word from the posture one,
#: because a different action lifts it — the same separation the two
#: operator-facing messages keep.
_WRITES_DISABLED_AUDIT_REASON = "writes_disabled"


def _deny_posture(hook_input, server=None):
    """Emit the sandbox-posture deny and exit 0. Does not return.

    The one seam both posture refusals go through — the exact-name write tool
    and the server-level prefix — so the debug line and the audit record are
    written once for two branches. *server* is the ``write_servers`` entry that
    matched, or ``None`` for the exact-name branch: the only thing that
    distinguishes them, and the only thing either record adds.

    Nothing here may raise. This hook fails OPEN — an uncaught exception exits
    non-zero with no JSON and the tool proceeds — and for a mixed read/write
    kernel this deny is the primary layer, since the renderer re-grants those
    tools via ``allow`` and leans on the hard deny. So both calls that touch the
    filesystem (the debug logger, which reads config and appends to a JSONL
    file, and the audit record) are wrapped, and both happen BEFORE the exit: a
    broken config or an unwritable audit zone costs a line, never the decision.
    """
    detail = "server=" + server if server else None
    try:
        log_hook(
            "writes-check",
            hook_input,
            status="deny",
            detail="reason=posture" + (" " + detail if detail else ""),
        )
    except Exception:
        pass  # logging must never cost the deny
    try:
        emit_audit(
            "writes-check",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=hook_input.get("tool_name", ""),
            reason=_POSTURE_DENY_AUDIT_REASON,
            detail=detail,
        )
    except Exception:
        pass  # the audit trail must never cost the deny
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": _POSTURE_DENY_REASON,
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect write tools -- by exact name. A miss is not automatically a
    # read: a facility-custom server's write tools arrive as an unmatchable
    # regex, and `write_servers` is how the render says so. That coverage is
    # confined to the sandbox posture (see the module docstring); every other
    # posture exits here exactly as it did before the key existed.
    if tool_name not in _get_write_tools():
        if os.environ.get("OSPREY_EXECUTION_MODE") == "readonly":
            server = _write_server_for(tool_name)
            if server:
                _deny_posture(hook_input, server=server)
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
    # `_deny_posture` states the fail-open rule this branch has to survive.
    if os.environ.get("OSPREY_EXECUTION_MODE") == "readonly":
        _deny_posture(hook_input)

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
    try:
        emit_audit(
            "writes-check",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=tool_name,
            reason=_WRITES_DISABLED_AUDIT_REASON,
        )
    except Exception:
        pass  # the audit trail must never cost the deny
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
