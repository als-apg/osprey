#!/usr/bin/env python3
"""
---
name: Memory Write Guard
description: Allows Write tool only for Claude Code native memory files, denies all other writes
summary: Restricts Write tool to Claude memory directory only
event: PreToolUse
tools: Write
safety_layer: 0
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         tool_name == "Write"?  ──NO──► EXIT (no opinion)
              │
             YES
              │
              ▼
         Resolve file_path
         to absolute path
              │
              ▼
         Inside Claude memory  ──YES──► ALLOW
         directory? (.md only)
              │
             NO
              │
              ▼
         DENY: not a memory file
```

## Details

Gate for the Write tool. Computes the Claude Code memory directory for the
current project (``~/.claude/projects/<encoded>/memory/``) and only allows
writes targeting ``.md`` files inside it. All other Write operations are
denied silently — the agent never sees a user prompt for non-memory writes.

This lets the agent use Claude Code's native memory system while preventing
arbitrary file creation. The memory gallery frontend reads from the same
directory, so saved memories appear in the UI automatically.

PROMPT-PROVIDER: This hook contains facility-customizable static text:
  - Denial message (section=write_denied_message)
  Future: source from FrameworkPromptProvider.get_write_guard_messages()
  Facility-customizable: allowed paths, denial message wording
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, get_project_dir, log_hook


def resolve_memory_dir(project_dir: str) -> Path:
    """Compute the Claude Code memory directory for a project.

    Claude Code stores memory files in
    ``~/.claude/projects/<encoded>/memory/``
    where ``<encoded>`` is the absolute project path with ``/`` replaced
    by ``-``.
    """
    abs_project = str(Path(project_dir).resolve())
    encoded = abs_project.replace("/", "-")
    return Path.home() / ".claude" / "projects" / encoded / "memory"


def is_allowed_memory_write(file_path: str, memory_dir: Path) -> bool:
    """Check whether a file path targets a .md file inside the memory directory."""
    try:
        target = Path(file_path).expanduser().resolve()
    except (OSError, ValueError):
        return False

    if target.suffix.lower() != ".md":
        return False

    # Must be a direct child of the memory dir (no subdirectories / traversal)
    try:
        target.relative_to(memory_dir)
    except ValueError:
        return False

    if target.parent != memory_dir:
        return False

    # Reject path traversal components in the original input
    if ".." in file_path:
        return False

    return True


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    if tool_name != "Write":
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        log_hook("memory-guard", hook_input, status="deny", detail="no-path")
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "\U0001f512 WRITE DENIED\n\n"
                    "No file path provided."
                ),
            }
        }
        json.dump(output, sys.stdout)
        sys.exit(0)

    project_dir = get_project_dir(hook_input) or os.getcwd()
    memory_dir = resolve_memory_dir(project_dir)

    if is_allowed_memory_write(file_path, memory_dir):
        log_hook("memory-guard", hook_input, status="allow")
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
        }
    else:
        log_hook("memory-guard", hook_input, status="deny", detail=f"path={file_path}")
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "\U0001f512 WRITE DENIED\n\n"
                    "File writes are restricted to Claude Code memory files only.\n"
                    f"Allowed directory: {memory_dir}/\n"
                    "Allowed extension: .md\n\n"
                    "Use the memory_save MCP tool for OSPREY session memory,\n"
                    "or write .md files to your Claude memory directory."
                ),
            }
        }

    json.dump(output, sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
