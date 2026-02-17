#!/usr/bin/env python3
"""PostToolUse hook: Audit logging.

Logs all OSPREY tool invocations to audit files.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OSPREY_PREFIXES = (
    "mcp__osprey-control-system__",
    "mcp__osprey-python-executor__",
    "mcp__osprey-workspace__",
)
MAX_RESULT_LENGTH = 500


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only log OSPREY tools
    matched_prefix = None
    for prefix in OSPREY_PREFIXES:
        if tool_name.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        sys.exit(0)

    short_name = tool_name[len(matched_prefix):]
    tool_input = hook_input.get("tool_input", {})
    tool_result = hook_input.get("tool_result", "")

    # Truncate result summary
    if isinstance(tool_result, dict):
        result_summary = json.dumps(tool_result)
    else:
        result_summary = str(tool_result)
    if len(result_summary) > MAX_RESULT_LENGTH:
        result_summary = result_summary[:MAX_RESULT_LENGTH] + "..."

    now = datetime.now(timezone.utc)
    audit_entry = {
        "timestamp": now.isoformat(),
        "tool": short_name,
        "session_id": os.environ.get("OSPREY_SESSION_ID"),
        "arguments": tool_input,
        "result_summary": result_summary,
    }

    # Write to audit file
    audit_dir = Path.cwd() / "osprey-workspace" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    audit_file = audit_dir / f"audit_{now.strftime('%Y%m%d')}.jsonl"
    with open(audit_file, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

    # Always exit 0 — never block
    sys.exit(0)


if __name__ == "__main__":
    main()
