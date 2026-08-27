#!/usr/bin/env python3
"""
---
name: Writes Kill Switch
description: Blocks ALL write operations under a readonly session posture or an unarmed target
summary: Blocks write operations when the session is sandboxed or the target is not armed
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
  STAGE 1  OSPREY_EXECUTION_MODE
           == "readonly"?   ──YES──► DENY: sandbox posture
              │
             NO
              │
              ▼
         Lane-addressed
         queue tool?      ──YES──► EXIT (allow)
              │
             NO
              │
              ▼
  STAGE 2  Load config.yml
           Read session target
              │
              ▼
         Armed for that
         target?          ──YES──► EXIT (allow)
              │
             NO
              │
              ▼
         DENY: writes not armed
```

## Details

First gate in the PreToolUse chain. Two independent reasons to refuse, in
order:

1. **Session posture.** `OSPREY_EXECUTION_MODE=readonly` means *this terminal
   session* was switched to the sandbox posture, whatever the deployment
   allows. Answered from the environment alone, ahead of any config read, and
   it covers every write call — a queue operation staged on a writes-armed
   deployment included.
2. **Deployment posture, per target.** Write posture is a property of the
   machine a call would reach, not of the deployment as a whole: a facility can
   arm its virtual accelerator and leave its ring unarmed. So the question is
   only answerable once the call has been pointed at a target — the session's,
   from the state file — and the answer comes from
   `control_system.connector.<type>.writes_enabled` over
   `control_system.writes_enabled`.

The two keep separate vocabularies: a posture refusal never points the operator
at a config key, because flipping one would not lift it. A stage-2 refusal names
the key the posture was actually read from, which on a per-target deployment is
the connector block rather than the deployment-wide flag.

Which tools are writes is rendered data (`write_tools` in `hook_config.json`),
read through the shared `write_tools()` accessor with its fail-closed floor. A
facility-custom server that opts into `writes_check` arrives there as a
whole-server matcher (`mcp__<name>__.*`), and `is_write_tool` honours it — so
every tool on such a server is gated by BOTH stages, reads included: nothing in
the render can tell a custom server's reads from its writes, and the server
asked for the gate at server level.

Lane-bound queue tools skip stage 2. A queue operation is addressed by *lane*,
and the lane's own bridge refuses what it must; the session target says nothing
about it. Which tools those are is rendered into `hook_config.json` off the tool
registry rather than spelled here, so renaming one never touches this file.
Stage 1 still applies to them in full.

The rules themselves live in `osprey_target_state`, shared with
`osprey_approval` so a deny here and a missing prompt there can never disagree
about one deployment.

Both refusals are also written to the audit trail (`emit_audit`, surface
`hook_writes-check`): the posture refusal under the reason word `posture` — the
same word the MCP audit middleware and the python executor's session clamp
record for the same refusal, so one grep finds all three layers — and the
stage-2 refusal under `writes_disabled`, because a different action lifts it.
A record is written before the deny is emitted and can never cost it.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    AUDIT_DECISION_REFUSED,
    emit_audit,
    get_hook_input,
    is_write_call,
    is_write_tool,
    load_hook_config,
    load_osprey_config,
    log_hook,
    short_tool_name,
    write_tools,
)

# The shared, stdlib-only reader for the control-system target state, and the
# home of the write-posture rules. Imported while this file's own directory is
# still the first `sys.path` entry the line above put there, because it is a
# sibling script rather than an installed module.
#
# Guarded exactly as `osprey_approval` guards it: a project rendered before the
# reader existed has no such sibling. Stage 2 treats that as not armed (see
# `_deployment_posture`).
try:
    import osprey_target_state as _target_state
except Exception:  # pragma: no cover - older render without the reader
    _target_state = None

#: The deployment-wide posture key, dotted as an operator spells it. Named in a
#: refusal only when no per-type block answered — an unidentifiable target, or a
#: `live` this deployment never described.
_GLOBAL_WRITES_KEY = "control_system.writes_enabled"

#: The hook_config key naming the tools stage 2 leaves to their own lane gate.
#: Short names, so an `extends` clone of the server — which renames only the
#: prefix — keeps the carve-out.
_LANE_ADDRESSED_KEY = "lane_addressed_tools"

#: The posture refusal, said once: it is lifted by one action, so it teaches the
#: operator one dialect.
_POSTURE_DENY_REASON = (
    "\U0001f512 SANDBOX POSTURE — this terminal session refuses "
    "control-system writes.\n\n"
    "Switch the session to writes posture from the terminal card; "
    "config.yml is not the gate here."
)

#: The machine-ish reason the posture refusal records. Deliberately the same
#: word the MCP audit middleware and the python executor's in-tool session
#: clamp record for the same refusal, so a sandboxed session's records join
#: across all three layers on one spelling. A cross-layer test pins them
#: together, reading this literal by AST — the hook imports nothing from
#: osprey, so it cannot share the constant itself.
_POSTURE_DENY_AUDIT_REASON = "posture"

#: What the stage-2 refusal records. A different word from the posture one,
#: because a different action lifts it — the same separation the two
#: operator-facing messages keep.
_WRITES_DISABLED_AUDIT_REASON = "writes_disabled"


def _server_prefixes():
    """Every MCP server prefix this render generated into hook_config.

    `server_prefixes` is the full list; `approval_prefixes` — the subset of
    servers that also wired the approval hook — is appended so a hook_config
    carrying only that one still resolves a short name.

    Never raises: a hook_config carrying something other than a list under those
    keys must not cost the gate a short name, and `short_tool_name` still
    resolves one from the `mcp__<server>__<tool>` shape without any prefixes.
    """
    try:
        hook_config = load_hook_config()
        return [
            prefix
            for key in ("server_prefixes", "approval_prefixes")
            for prefix in hook_config.get(key) or ()
        ]
    except Exception:
        return []


def _is_lane_addressed(short_name):
    """Whether stage 2 leaves a tool alone because a lane addresses it.

    A queue operation binds to one plan lane, not to the session target: the
    lane-bound tools name their lane and refuse in-tool, and the one that only
    stages work composes tokenless by contract, so nothing it stages reaches a
    machine on its own. Gating them on the session target would refuse a plan
    queued for the simulator because the session happens to point at the ring.

    The set is data, read from this render's hook_config, never a name spelled
    in this file — a renamed tool would otherwise detach its carve-out here
    while still looking gated.

    Fails *towards* gating, twice over: a hook_config that lists no such tools
    leaves every write tool subject to the target posture, and one whose
    prefixes could not be read leaves *short_name* as the full tool name, which
    no short name matches. Both are the more restrictive answer.
    """
    try:
        listed = load_hook_config().get(_LANE_ADDRESSED_KEY) or ()
        return short_name in tuple(listed)
    except Exception:
        return False


def _key_for_type(connector_type):
    """The config key that arms one connector type, spelled as an operator does.

    The deployment-wide key when there is no type: an unknown target and an
    underivable `live` have no block to name, and that key is the one their
    posture was read from anyway.

    A deliberate restatement of `osprey_connectors.types.writes_enabled_key`.
    A hook runs on the operator's stdlib Python with none of OSPREY installed,
    so it cannot import the framework spelling; the parity test is what keeps
    the two readings from drifting apart.
    """
    if connector_type is None:
        return _GLOBAL_WRITES_KEY
    return f"control_system.connector.{connector_type}.writes_enabled"


def _refusal_keys(section, target):
    """The config keys an operator would edit to lift this refusal, in order.

    The blocks the posture was actually READ from, never the deployment-wide key
    by default: a refusal naming the global key on a deployment whose live block
    says `false` would send the operator to flip a key that changes nothing.

    A `None` *target* is the session whose target could not be identified. It is
    refused unless EVERY target it could reach is armed, so the keys are the
    unarmed ones among those — naming a target whose key already says `true`
    would be the same wrong instruction reached by a different route.

    The types come from the reader's own mappings rather than being re-derived
    here, so the keys named and the postures read cannot drift apart while both
    look right.
    """
    if target is not None:
        return [_key_for_type(_target_state.target_type(section, target))]
    keys = []
    for connector_type in _target_state.session_types(section).values():
        if _target_state.type_posture(section, connector_type) is True:
            continue
        key = _key_for_type(connector_type)
        if key not in keys:
            keys.append(key)
    return keys or [_GLOBAL_WRITES_KEY]


def _deployment_posture(hook_input):
    """Whether this deployment arms writes for the target this session acts on.

    Returns ``(armed, keys, target)`` — the decision, the config keys a refusal
    should name, and the target it was answered for (``None`` when the session's
    target could not be identified).

    Everything stage 2 touches sits inside one ``try``, and every failure
    resolves to NOT ARMED. That is the deliberate exception to this hook's
    fail-open rule: elsewhere an uncaught exception exits non-zero with no JSON
    and the tool proceeds, which is right for a hook that only enriches, and
    wrong for the one that decides whether a write reaches a machine. Stage 1
    runs before this and is unaffected either way.
    """
    try:
        config = load_osprey_config(hook_input)
        section = config.get("control_system")

        if _target_state is None:
            # The reader is also where the posture RULES live, so a render
            # without it cannot answer stage 2 at all — and "cannot answer" is
            # not armed. The sibling is rendered beside this file by the same
            # build, so this is theoretical rather than an upgrade path.
            return False, [_GLOBAL_WRITES_KEY], None

        result = _target_state.read_session_target(hook_input)
        target = None if _target_state.is_baseline(result) else result.get("target")

        if target is None:
            # A baseline fallback still NAMES the deployment's baseline target,
            # and answering for it would state a posture for a session that may
            # have switched away from it. The posture every target this session
            # could reach agrees on is the only one that cannot become a guess
            # in favour of hardware.
            posture = _target_state.most_restrictive_posture(section)
            return posture is True, _refusal_keys(section, None), None

        posture = _target_state.writes_posture(section, target)
        # `None` — a config that expresses no posture anywhere — refuses here.
        # This hook has always denied a config with no `control_system` block,
        # and a deployment that says nothing must not become one that writes.
        # It is also the one shape where this hook and `osprey_approval`
        # deliberately disagree: approval falls through to its normal prompt,
        # this hook still refuses.
        return posture is True, _refusal_keys(section, target), target
    except Exception:
        return False, [_GLOBAL_WRITES_KEY], None


def _record_refusal(hook_input, tool_name, reason, detail=None):
    """Write one refusal to the audit trail. Never raises.

    Wrapped because this hook's stage 1 fails OPEN — an uncaught exception exits
    non-zero with no JSON and the tool proceeds — and on a mixed read/write
    render the deny it accompanies is the primary layer, since the renderer
    re-grants those tools via ``allow`` and leans on the hard deny. An
    unwritable audit zone costs a record, never the decision.
    """
    try:
        emit_audit(
            "writes-check",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=tool_name,
            reason=reason,
            detail=detail,
        )
    except Exception:
        pass  # the audit trail must never cost the deny


def _deny_posture(hook_input, tool_name):
    """Emit the sandbox-posture deny and exit 0. Does not return.

    Nothing here may raise (see :func:`_record_refusal` for why). Both calls that
    touch the filesystem — the debug logger, which reads config and appends to
    a JSONL file, and the audit record — are wrapped, and both happen BEFORE
    the exit: a broken config or an unwritable audit zone costs a line, never
    the decision.
    """
    try:
        log_hook("writes-check", hook_input, status="deny", detail="reason=posture")
    except Exception:
        pass  # logging must never cost the deny
    _record_refusal(hook_input, tool_name, _POSTURE_DENY_AUDIT_REASON)
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

    # Only inspect write tools
    if not is_write_tool(tool_name, write_tools()):
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})

    # Resolved once, against this render's own server prefixes: both exits below
    # are carve-outs for a TOOL rather than for a server, and an `extends` clone
    # renames only the prefix — a carve-out matched on the full name would miss
    # every clone.
    short_name = short_tool_name(tool_name, _server_prefixes())

    # For execute: allow readonly even when writes are not armed.
    # The server defaults execution_mode to "readonly", so treat missing as readonly.
    if not is_write_call(tool_name, tool_input, short_name):
        sys.exit(0)

    # -- Stage 1: session posture ----------------------------------------
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
    # exactly what a sandboxed session is for. It sits *before* the queue exit
    # below, so a sandboxed session cannot arm a Bluesky lane either.
    #
    # `_deny_posture` states the fail-open rule this branch has to survive.
    if os.environ.get("OSPREY_EXECUTION_MODE") == "readonly":
        _deny_posture(hook_input, tool_name)

    # -- Stage 2: deployment posture, for this session's target ----------
    if _is_lane_addressed(short_name):
        log_hook("writes-check", hook_input, status="allow", detail="lane_addressed")
        sys.exit(0)

    armed, refusal_keys, target = _deployment_posture(hook_input)

    if armed:
        log_hook("writes-check", hook_input, status="allow", detail=f"target={target}")
        sys.exit(0)

    # Deny — this deployment is not armed for what the call would touch. Emit a
    # JSON `permissionDecision: deny`, the canonical PreToolUse deny mechanism.
    # Empirically: PreToolUse decision aggregation does not honour this deny if
    # another hook in the same chain returns an "ask", so on a mixed
    # read/write render — where the renderer no longer emits a static ask for
    # the write tools — this deny together with approval's defer (see
    # osprey_approval.py) is the whole gate. On a render that hard-blocks the
    # tool through `permissions.deny` in
    # `src/osprey/cli/templates/claude_code.py`, this is defense-in-depth.
    log_hook("writes-check", hook_input, status="deny", detail=f"target={target}")
    _record_refusal(
        hook_input,
        tool_name,
        _WRITES_DISABLED_AUDIT_REASON,
        detail=f"target={target}" if target else None,
    )
    if target:
        scope = f"Control system writes are not armed for the active target ({target})."
    else:
        # Named as a refusal ABOUT the missing identity, not about one target: the
        # posture was the intersection over every target this session could reach,
        # and a line naming one of them would describe a decision nobody made.
        scope = (
            "This session's control target could not be identified, so writes "
            "are refused unless every target it could reach is armed."
        )
    arm = "Arm them in config.yml: " + ", ".join(f"{key}: true" for key in refusal_keys)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (f"\U0001f512 WRITES DISABLED\n\n{scope}\n{arm}"),
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
