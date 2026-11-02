#!/usr/bin/env python3
"""
---
name: Writes Kill Switch
description: Blocks write operations the write verdict refuses, and every write in a read-only run
summary: Blocks write operations in a read-only run or when the write switch is off for their machine
event: PreToolUse
tools: channel_write, phoebus_drive, execute, execute_file, pyaml_measure, queue_add, queue_start
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
  STAGE 1  Deployment-wide
           read-only run?   ──YES──► DENY: readonly_run
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
  STAGE 2  Which machine?   facility tool ──► live
           (hook_write_target) framework tool ──► session target
              │
              ▼
         execution_verdict
         allows?          ──YES──► EXIT (allow)
              │
             NO
              │
              ▼
         DENY: the verdict's phrase
```

## Details

First gate in the PreToolUse chain. Two independent reasons to refuse, in
order:

1. **Deployment-wide read-only run.** `OSPREY_EXECUTION_MODE=readonly` means
   *the whole deployment* was launched read-only, and nothing a session or a
   switch does can override it. Answered from the environment alone, ahead of
   any config read, and it covers every write call, queue operations included.
2. **The write verdict, for the machine the call writes to.** The stdlib twin's
   `hook_write_target` places the call: a framework write tool writes to the
   session's target, and a facility write tool (a whole-server `writes_check`
   custom server, or a `control_system.write_tools` entry) writes to `live`,
   because only the real machine's switch covers a tool the deployment
   declared. The twin's `execution_verdict` then decides: the session's hold,
   and `write_verdict` term for term with the connector's reference monitor. A whole-server facility server is gated
   whole, reads included: nothing in the render tells its reads from its
   writes.

Every refusal is the twin's `refusal_text` — the phrase, which names no config
key and ends with who can lift it, then the trailer line
`[osprey-refusal: <reason>]`. The posture-unknown deny, for an unstamped
process whose guessed directory holds no record, has no verdict to word: it
speaks the `record_unavailable` refusal, because a switch nobody can read is
exactly that.

Lane-bound queue tools skip stage 2. A queue operation is addressed by *lane*,
and the tool decides on its lane's machine; the session's target says nothing
about it. Which tools those are is rendered into `hook_config.json` off the
tool registry. Stage 1 still applies to them in full.

Every refusal is written to the audit trail (`emit_audit`, surface
`hook_writes-check`) under the reason word `posture`, the word the MCP audit
middleware and the python executor's gates record for the same refusals, with
detail `reason=<r> target=<t>`. A record is written before the deny is emitted
and can never cost it.
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
# `_stage_two`).
try:
    import osprey_target_state as _target_state
except Exception:  # pragma: no cover - older render without the reader
    _target_state = None

#: The read-only run, restated from ``osprey_connectors.posture_env``: the
#: variable and the one value of it that sandboxes. Spelled here rather than
#: taken from the reader above, because stage 1 must answer even where that
#: sibling is missing. ``tests/hooks/test_posture_store_parity.py`` holds both
#: equal to the connector module's.
EXECUTION_MODE_ENV_VAR = "OSPREY_EXECUTION_MODE"
SANDBOX_MODE = "readonly"

#: The hook_config key naming the tools stage 2 leaves to their own lane gate.
#: Short names, so an `extends` clone of the server — which renames only the
#: prefix — keeps the carve-out.
_LANE_ADDRESSED_KEY = "lane_addressed_tools"

#: What every deny opens with; the twin's refusal text follows it.
_DENY_HEADER = "\U0001f512 WRITES OFF — "

#: The reason spoken for the one cell where the switch cannot be READ: the
#: agent-data root was not stamped on this process, and the directory this hook
#: derived instead holds no readable control-context record. An empty answer
#: read there proves nothing, and an unreadable state is not a permissive one.
#: See ``osprey_target_state.posture_unknown``.
_POSTURE_UNKNOWN_REASON = "record_unavailable"

#: The machine-ish reason every refusal here records. Deliberately the same
#: word the MCP audit middleware and the python executor's gates record, so a
#: refusal joins across all three layers on one spelling. A cross-layer test
#: pins them together, reading this literal by AST — the hook imports nothing
#: from osprey, so it cannot share the constant itself.
_POSTURE_DENY_AUDIT_REASON = "posture"

#: The audit DETAIL of the unreadable-state refusal. The reason stays
#: :data:`_POSTURE_DENY_AUDIT_REASON`; only the detail says which happened.
_POSTURE_UNKNOWN_DETAIL = "posture unknown"

#: The two phrases this hook needs when the twin is missing (a render older
#: than the twin). Copies of the twin's ``readonly_run`` and
#: ``record_unavailable`` phrases, pinned equal to them by the hook tests.
_READONLY_RUN_PHRASE = (
    "Writes are refused because the deployment is running read-only; "
    "the chip cannot lift this — ask the deployment admin."
)
_RECORD_UNAVAILABLE_PHRASE = (
    "Writes are refused because the write switch cannot be read; "
    "retry, then ask the deployment admin."
)

#: What :func:`_stage_two` answers for the unreadable-state cell.
_POSTURE_UNKNOWN = "posture unknown"


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

    A queue operation binds to one plan lane, not to the deployment's target:
    the lane-bound tools name their lane and refuse in-tool, and the one that
    only stages work composes tokenless by contract, so nothing it stages reaches
    a machine on its own. Gating them on the deployment's target would refuse a
    plan queued for the simulator because the deployment points at the real machine.

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


def _record_refusal(hook_input, tool_name, reason, detail=None):
    """Write one refusal to the audit trail. Never raises.

    Wrapped because this hook's stage 1 fails OPEN — an uncaught exception exits
    non-zero with no JSON and the tool proceeds — and on a mixed read/write
    render the deny it accompanies is the primary layer, since the renderer
    re-grants those tools via ``allow`` and leans on the hard deny. An
    unwritable audit zone costs a record, never the decision.
    """
    # the audit trail must never cost the deny
    with contextlib.suppress(Exception):
        emit_audit(
            "writes-check",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=tool_name,
            reason=reason,
            detail=detail,
        )


def _fallback_text(reason):
    """This file's copy of the twin's refusal text, for a render without the twin.

    Only ``readonly_run`` is answerable without the twin; anything else is the
    twin's unanswered stage 2, which refuses ``record_unavailable``.
    """
    if reason == "readonly_run":
        return f"{_READONLY_RUN_PHRASE}\n[osprey-refusal: readonly_run]"
    return f"{_RECORD_UNAVAILABLE_PHRASE}\n[osprey-refusal: record_unavailable]"


def _text(reason, kind=None):
    """The twin's refusal text for *reason*, or this file's copy when the twin is missing."""
    if _target_state is not None:
        return _target_state.refusal_text(reason, kind)
    return _fallback_text(reason)


def _stage_two(hook_input, tool_name, short_name):
    """The verdict on this call's machine, as ``(allowed, reason, target, kind)``.

    ``None`` for a call the classifier says writes nothing, and
    :data:`_POSTURE_UNKNOWN` for the unreadable-state cell.

    Everything stage 2 touches sits inside one ``try``, and every failure
    resolves to a refusal (``record_unavailable``). That is the deliberate
    exception to this hook's fail-open rule: elsewhere an uncaught exception
    exits non-zero with no JSON and the tool proceeds, which is right for a
    hook that only enriches, and wrong for the one that decides whether a
    write reaches a machine. A render without the twin cannot answer at all,
    and that refuses the same way.
    """
    refused = (False, "record_unavailable", None, None)
    if _target_state is None:
        return refused
    try:
        config = load_osprey_config(hook_input)
        if not isinstance(config, dict):
            config = {}
        section = config.get("control_system")
        target = _target_state.hook_write_target(config, tool_name, short_name, hook_input)
        if target is None:
            return None
        if target is _target_state.UNRESOLVED:
            if _target_state.posture_unknown(hook_input):
                return _POSTURE_UNKNOWN
            # The verdict on no machine refuses: `target_unknown`, unless an
            # earlier term (the tier) refuses first.
            target = None
        verdict = _target_state.execution_verdict(section, target, hook_input=hook_input)
        return verdict.allowed, verdict.reason, verdict.target, verdict.kind
    except Exception:
        return refused


def _lane_tool_held(hook_input):
    """Whether the session's hold refuses a lane-addressed tool at this hook.

    A lane says which machine a call reaches, and a hold covers every machine
    of the session, so the lane carve-out does not apply to it. This answer only
    spares the operator a prompt for a write that cannot run: the audit clamp and
    the tool's own gate refuse it regardless, so a failure here answers ``False``.
    """
    if _target_state is None:
        return False
    try:
        config = load_osprey_config(hook_input)
        section = config.get("control_system") if isinstance(config, dict) else None
        return _target_state.hold_applies(section)
    except Exception:
        return False


def _emit_deny(reason, kind=None):
    """Write the PreToolUse deny worded for *reason* and exit 0. Does not return.

    A twin that raises while wording it falls back to this file's
    ``record_unavailable`` text: the deny is never lost to its words.
    """
    try:
        text = _text(reason, kind)
    except Exception:
        text = _fallback_text("record_unavailable")
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": _DENY_HEADER + text,
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


def _deny(hook_input, tool_name, reason, target=None, kind=None):
    """Emit the verdict deny for *reason* and exit 0. Does not return.

    Worded by the twin's refusal text for *reason* (the phrase, naming the
    machine *kind* when it is known and never a config key, then the
    ``[osprey-refusal: <reason>]`` trailer). Recorded under
    :data:`_POSTURE_DENY_AUDIT_REASON` with detail ``reason=<r> target=<t>``.

    Nothing here may raise (see :func:`_record_refusal` for why). Both calls that
    touch the filesystem — the debug logger and the audit record — are wrapped,
    and both happen BEFORE the exit.
    """
    detail = f"reason={reason} target={target}"
    # logging must never cost the deny
    with contextlib.suppress(Exception):
        log_hook("writes-check", hook_input, status="deny", detail=detail)
    _record_refusal(hook_input, tool_name, _POSTURE_DENY_AUDIT_REASON, detail=detail)
    _emit_deny(reason, kind)


def _deny_posture_unknown(hook_input, tool_name):
    """Emit the unreadable-posture deny and exit 0. Does not return.

    Records the posture reason with :data:`_POSTURE_UNKNOWN_DETAIL`, so
    refusals under the sandbox posture still join across the three layers on one
    spelling while the ledger keeps "could not be read" apart from "was
    turned off". Worded as the ``record_unavailable`` refusal. Same no-raise
    rule as :func:`_deny`.
    """
    # logging must never cost the deny
    with contextlib.suppress(Exception):
        log_hook(
            "writes-check",
            hook_input,
            status="deny",
            detail=f"reason={_POSTURE_UNKNOWN_DETAIL}",
        )
    _record_refusal(
        hook_input, tool_name, _POSTURE_DENY_AUDIT_REASON, detail=_POSTURE_UNKNOWN_DETAIL
    )
    _emit_deny(_POSTURE_UNKNOWN_REASON)


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

    # For execute: allow readonly whatever the switch says.
    # The server defaults execution_mode to "readonly", so treat missing as readonly.
    if not is_write_call(tool_name, tool_input, short_name):
        sys.exit(0)

    # -- Stage 1: deployment-wide read-only run ---------------------------
    # Answered from the environment alone, ahead of load_osprey_config(), so
    # the answer never depends on config I/O, on the config being parseable,
    # or on PyYAML being importable at all. A value comparison, never a
    # presence check: only the exact "readonly" string is a read-only run.
    #
    # It sits *after* the execute-readonly exit above: a readonly execution is
    # exactly what a read-only run is for. It sits *before* the queue exit
    # below, so a read-only run cannot arm a Bluesky lane either.
    if os.environ.get(EXECUTION_MODE_ENV_VAR) == SANDBOX_MODE:
        _deny(hook_input, tool_name, "readonly_run")

    # -- Stage 2: the write verdict, for the machine the call writes to ---
    if _is_lane_addressed(short_name):
        if _lane_tool_held(hook_input):
            _deny(hook_input, tool_name, "session_held")
        log_hook("writes-check", hook_input, status="allow", detail="lane_addressed")
        sys.exit(0)

    answer = _stage_two(hook_input, tool_name, short_name)
    if answer == _POSTURE_UNKNOWN:
        _deny_posture_unknown(hook_input, tool_name)
    if answer is None:
        log_hook("writes-check", hook_input, status="allow", detail="not a write")
        sys.exit(0)

    allowed, reason, target, kind = answer
    if allowed:
        log_hook("writes-check", hook_input, status="allow", detail=f"target={target}")
        sys.exit(0)

    # Deny — a JSON `permissionDecision: deny`, the canonical PreToolUse deny.
    # On the pinned Claude Code this deny holds when another hook in the same
    # chain asks: the approval card is not shown and the call does not run.
    _deny(hook_input, tool_name, reason, target=target, kind=kind)


if __name__ == "__main__":
    main()
