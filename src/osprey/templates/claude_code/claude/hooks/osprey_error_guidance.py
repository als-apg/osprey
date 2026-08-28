#!/usr/bin/env python3
"""
---
name: Error Guidance
description: Injects error-handling protocol guidance when an OSPREY tool returns a structured error
summary: Injects error-handling guidance into tool error responses
event: PostToolUse
tools: all OSPREY MCP tools
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Is OSPREY tool?  ──NO──► EXIT (silent)
              │
             YES
              │
              ▼
         Parse tool_response
              │
              ▼
         Detect error envelope
         {"error": true, ...}
              │
              ▼
         Error found?  ──NO──► EXIT (silent)
              │
             YES
              │
              ▼
         Map error_type to class
         (Connection/Validation/
          Data/Execution/Safety/
          Internal)
              │
              ▼
         Inject additionalContext:
         error class + protocol ref
```

## Details

Never blocks execution — only adds `additionalContext` pointing Claude to
the error-handling protocol in `.claude/rules/error-handling.md`. Detects
the standard OSPREY error envelope (`{"error": true, "error_type": ...}`)
and falls back to keyword detection for non-JSON responses.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, load_hook_config, log_hook

# Map OSPREY error_type values to short human-readable classes.
# Matches the taxonomy in .claude/rules/error-handling.md.
#
# Every error_type an MCP tool can emit has to be a key here: a value that is
# not falls through to "Internal" — "report verbatim, have an operator check
# the server logs" — which is the wrong protocol for a bridge that is merely
# unreachable, a lookup that merely missed, or a refusal the agent must not
# work around. tests/mcp_server/test_error_type_conformance.py scans every
# emitter under src/osprey/mcp_server and fails on a value missing here;
# tests/hooks/test_error_guidance_hook.py pins this table's exact contents.
#
# The class picks the agent's PROTOCOL; WHICH subsystem failed is in the
# envelope itself — the error_type, the message, and details (for example
# details.subsystem, details.active_target, details.kind).
ERROR_CLASS_MAP = {
    # ---- Connection: something OSPREY talks to did not answer, or would not
    # serve the request. The service is named by the error_type or the
    # envelope; "control system" only when the controls server says so.
    "connection_error": "Connection",
    "timeout_error": "Connection",
    "service_unavailable": "Connection",
    # A switch that was attempted and did not complete: the destination did
    # not answer, and the previous target is still active.
    "target_switch_failed": "Connection",
    "bluesky_bridge_unreachable": "Connection",
    "phoebus_unreachable": "Connection",
    "gallery_unreachable": "Connection",
    "web_terminal_unreachable": "Connection",
    # Bluesky bridge refusals relayed verbatim: the queue manager or worker
    # environment did not come up, or did not answer a pause in time.
    "manager_unreachable": "Connection",
    "environment_unavailable": "Connection",
    "abort_pause_timeout": "Connection",
    # The health checker's own worker is wedged; the report will come back.
    "health_suppressed": "Connection",
    # The ARIEL statement ran past its deadline, like timeout_error.
    "search_timeout": "Connection",
    # The logbook was reached and wants a credential the deployment does not
    # hold — a Connection-class fix (configure the credential), as the graph
    # store's connection_error for a rejected credential already is.
    "auth_required": "Connection",
    # ---- Validation: the request as made cannot be served, and the agent
    # can change it — a parameter, a query, a reference, a precondition.
    "validation_error": "Validation",
    "limits_violation": "Validation",
    # This deployment simply has no such target to switch to.
    "target_switch_unavailable": "Validation",
    "invalid_query": "Validation",
    "invalid_pattern": "Validation",
    "sql_error": "Validation",
    "file_too_large": "Validation",
    "conversion_not_supported": "Validation",
    "arrange_rejected": "Validation",
    "set_draft_no_argument": "Validation",
    "draft_conflict": "Validation",
    "plan_write_rejected": "Validation",
    "phoebus_handle_required": "Validation",
    "phoebus_rejected": "Validation",
    # These name the valid set in the message: pick from it.
    "unknown_category": "Validation",
    "unknown_panel": "Validation",
    "unknown_bluesky_lane": "Validation",
    "unknown_device": "Validation",
    # A two-lane deployment needs the lane named; guessing is the failure.
    "lane_required": "Validation",
    # Bluesky bridge refusals about the request's preconditions.
    "stale_draft_revision": "Validation",
    "draft_revision_already_launched": "Validation",
    "session_plan_unvalidated": "Validation",
    "session_plan_not_in_namespace": "Validation",
    "manager_not_idle": "Validation",
    "queue_request_rejected": "Validation",
    # ---- Data: a lookup that missed. Report what was asked for and that
    # nothing was found; suggest refining.
    "not_found": "Data",
    "no_results": "Data",
    "file_not_found": "Data",
    "no_draft": "Data",
    "unknown_plan": "Data",
    "unknown_run": "Data",
    "unknown_session_plan": "Data",
    "window_not_found": "Data",
    # The bridge answered: it has no source file for that plan name.
    "plan_source_unavailable": "Data",
    # Nothing was stopped because nothing was running — the answer, not a fault.
    "nothing_running": "Data",
    # ---- Execution: the user's own code (or document source) is wrong.
    "execution_error": "Execution",
    "lattice_error": "Execution",
    "compilation_error": "Execution",
    # ---- Safety: a gate said no. Explain the constraint; never work around
    # it, never touch its configuration. Falling through to Internal used to
    # tell the agent to "check the server logs" for all of these.
    "safety_error": "Safety",
    "write_refused": "Safety",
    "target_switched": "Safety",
    "target_switch_refused": "Safety",
    "target_changed": "Safety",
    "writes_disabled": "Safety",
    "launch_token_required": "Safety",
    "path_traversal": "Safety",
    "protected_key": "Safety",
    "session_target_mismatch": "Safety",
    "lane_mismatch": "Safety",
    # This deployment cannot do what was asked (a connector that cannot
    # execute plans, a logbook adapter that cannot write): explain, do not
    # look for another path.
    "browse_only_connector": "Safety",
    "unsupported_connector": "Safety",
    "not_supported": "Safety",
    # An interrupted plan is back at the head of the queue; a human decides
    # whether to remove it (approval-gated), never the agent by retrying.
    "interrupted_item_in_queue": "Safety",
    # ---- Internal: an OSPREY-side fault an operator has to fix — missing
    # config, missing dependency, a bridge or gallery that answered with an
    # error of its own. Report verbatim; name the service from the envelope.
    "internal_error": "Internal",
    "platform_error": "Internal",
    "configuration_error": "Internal",
    "not_configured": "Internal",
    "server_not_initialised": "Internal",
    "dependency_missing": "Internal",
    "conversion_error": "Internal",
    "capture_error": "Internal",
    # The process lacks filesystem access it needs — an ownership fix for
    # the operator, not a policy gate the agent should explain.
    "permission_denied": "Internal",
    "bluesky_bridge_error": "Internal",
    "phoebus_error": "Internal",
    "phoebus_open_failed": "Internal",
    "gallery_error": "Internal",
    # Bluesky bridge refusals that need an operator.
    "config_unreadable": "Internal",
    "manager_not_configured": "Internal",
}


def _detect_error(tool_response: str | dict | None) -> tuple[str | None, str | None]:
    """Detect a structured error in the tool response.

    OSPREY tools now return ``CallToolResult(isError=True, ...)`` for every
    failure. The PostToolUse hook input surfaces this as a dict with
    ``isError: true`` and a ``content`` list of text blocks; the first text
    block carries the structured envelope JSON. Returns
    ``(error_class, error_message)`` or ``(None, None)`` if no error detected.
    """
    if not isinstance(tool_response, dict) or tool_response.get("isError") is not True:
        return None, None

    for block in tool_response.get("content", []) or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        try:
            parsed = json.loads(block.get("text", ""))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and parsed.get("error") is True:
            error_type = parsed.get("error_type", "unknown")
            return (
                ERROR_CLASS_MAP.get(error_type, "Internal"),
                parsed.get("error_message", str(parsed)),
            )

    # isError=True but no structured envelope (shouldn't happen for OSPREY
    # tools, but cover the case so guidance still fires).
    return "Internal", "Tool returned an error"


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect OSPREY tools (prefixes loaded from hook_config.json)
    _prefixes = load_hook_config().get("server_prefixes", [])
    if not any(tool_name.startswith(p) for p in _prefixes):
        sys.exit(0)

    tool_response = hook_input.get("tool_response")
    error_class, error_message = _detect_error(tool_response)

    if error_class is None:
        log_hook("error-guidance", hook_input, status="no-error")
        sys.exit(0)

    # Inject guidance reminder
    guidance = (
        f"ERROR DETECTED [{error_class}]: {error_message}\n\n"
        "Follow the error-handling protocol (.claude/rules/error-handling.md):\n"
        "- Report the error clearly to the user with actionable next steps.\n"
        "- Do NOT debug infrastructure, write mock data, or work around the failure.\n"
        "- Do NOT retry unless the error explicitly indicates a transient condition."
    )

    log_hook("error-guidance", hook_input, status="error", detail=f"class={error_class}")

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": guidance,
        }
    }
    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
