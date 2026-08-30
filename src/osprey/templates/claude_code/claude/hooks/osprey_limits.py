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
         Import LimitsValidator ──FAILS──► EXIT (allow)
              │
              ▼
         Per-target API?  ──NO──► from_config()  ──┐
              │                   (older framework)│
             YES                                   │
              │                                    │
              ▼                                    │
         Session target                            │
         named?                                    │
              │                                    │
        ┌─────┴─────┐                              │
       YES          NO                             │
        │            │                             │
        ▼            ▼                             │
   from_config    from_config_                     │
   (target=…)     most_restrictive()               │
        │            │                             │
        └─────┬──────┴─────────────────────────────┘
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

The posture that database is applied under is a property of the machine a write
would reach, not of the deployment as a whole:
`control_system.connector.<type>.limits_checking` overrides the deployment-wide
`control_system.limits_checking` block whole, so a facility can refuse unlisted
channels on its ring and allow them on its virtual accelerator. Three branches
decide which posture this hook validates under, and a fourth case leaves it with
no posture to apply:

1. **The session names a target** — `from_config(target=…)`, the posture of
   the machine this write would actually reach. Every refusal names the key
   that answered, which on a per-target deployment is the connector block
   rather than the deployment-wide one. A target the config resolves to no
   connector type — `live` on a deployment that never named its real machine —
   is still an identified target, and the resolver answers it with the
   deployment-wide block, exactly as the write posture does.
2. **It names none** — no state file yet, an unreadable or ambiguous state
   directory, or a render without the sibling reader —
   `from_config_most_restrictive()`: limits checking on where any reachable
   target has it on, unlisted channels allowed only where every reachable
   target allows them. Naming a target here would be a guess, and a guess
   between a simulator and a ring is a guess in favour of hardware.
3. **The framework is older than this render** — hooks are rendered by
   `osprey build` on the host, while a service keeps the `osprey` its image was
   built with until the next `osprey up --build`, so a validator with no
   `target` argument and no `from_config_most_restrictive` is reachable. It is
   probed for, never caught, and answered with the deployment-wide block: what
   that framework did when the deployment-wide block was the whole story.
4. **There is no validator to consult** — `LimitsValidator` is not importable
   (`osprey` off the interpreter's path), or the resolved posture switches
   limits checking off — and the write is allowed through for the MCP tool to
   decide. This is the hook's only fail-open direction. A *failsafe* validator
   is not one of them: an incomplete per-type block or an unreadable database
   still builds a validator, and it blocks every write.

The target itself comes from `osprey_target_state`, the same stdlib reader the
writes kill switch and the approval prompt use, so one session cannot be
described as pointing at two machines.
"""

import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import AUDIT_DECISION_REFUSED, emit_audit, get_hook_input, log_hook

# The shared, stdlib-only reader for the control-system target state, imported
# while this file's own directory is still the first `sys.path` entry the line
# above put there, because it is a sibling script rather than an installed
# module.
#
# Guarded the way `osprey_writes_check` guards it: a project rendered before the
# reader existed has no such sibling, and a hook that cannot learn its target
# validates under the posture every reachable target agrees on.
try:
    import osprey_target_state as _target_state
except Exception:  # pragma: no cover - older render without the reader
    _target_state = None


def _session_target(hook_input):
    """The control target this session is pointed at, or ``None``.

    ``None`` means unidentifiable rather than absent, and every route to it —
    a render without the sibling reader, the reader's own baseline fallback (no
    state file, an unreadable or ambiguous state directory), or an exception on
    the way — is the same answer, because the caller does the same thing with
    all of them: validate under the most restrictive posture. The reader is
    documented never to raise, so the guard here is belt-and-braces; what it
    guarantees is that no failure of target IDENTITY can turn into a write that
    was never checked.
    """
    if _target_state is None:
        return None
    try:
        result = _target_state.read_session_target(hook_input)
        if _target_state.is_baseline(result):
            return None
        return result.get("target")
    except Exception:
        return None


def _supports_per_target_postures(validator_cls):
    """Whether the installed framework can resolve a limits posture per target.

    A version probe, not a feature flag. Hooks are rendered into the repo by
    ``osprey build``, which runs on the host, while the services that carry
    ``osprey`` keep the image they were built with until the next
    ``osprey up --build``. So this render can meet an older validator, one whose
    ``from_config`` takes no ``target`` and which has no
    ``from_config_most_restrictive`` at all. Calling either there raises
    straight past this hook, and an exception here exits non-zero with no
    decision — which the agent runtime reads as no opinion, so the write would
    reach the machine with no limits applied at all.

    Probed rather than caught, deliberately: a blanket ``except (AttributeError,
    TypeError)`` around the call would also swallow those raised *inside* a
    current resolver or database load, and silently answer a real fault with the
    deployment-wide posture.
    """
    if getattr(validator_cls, "from_config_most_restrictive", None) is None:
        return False
    try:
        return "target" in inspect.signature(validator_cls.from_config).parameters
    except (TypeError, ValueError):  # pragma: no cover - unintrospectable callable
        return False


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

    if not _supports_per_target_postures(LimitsValidator):
        # Framework older than this render (see `_supports_per_target_postures`):
        # ask the deployment-wide question, which is exactly what that framework
        # did when it was the whole story. Falling back to allowing the write
        # instead would take limits off a machine that still enforces them.
        validator = LimitsValidator.from_config()
    else:
        target = _session_target(hook_input)
        if target is None:
            # The posture every target a session here could reach agrees on. A
            # baseline fallback still NAMES the deployment's baseline target,
            # and validating under it would apply one machine's posture to a
            # session that may have switched away from it.
            validator = LimitsValidator.from_config_most_restrictive()
        else:
            validator = LimitsValidator.from_config(target=target)
    if validator is None:
        # Limits checking is off for this posture
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
    try:
        # Count only. The violation lines carry the offending channel VALUES,
        # and a value is exactly what an audit record never holds.
        emit_audit(
            "limits",
            hook_input,
            decision=AUDIT_DECISION_REFUSED,
            subject=tool_name,
            reason="limits_violation",
            detail=f"violations={len(violations)}",
        )
    except Exception:
        pass  # the audit trail must never cost the deny
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
