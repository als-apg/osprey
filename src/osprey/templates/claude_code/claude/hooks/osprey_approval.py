#!/usr/bin/env python3
"""
---
name: Human Approval Gate
description: Requires human approval for dangerous operations based on config approval mode
summary: Requires human approval for dangerous operations
event: PreToolUse
tools: channel_write, execute
safety_layer: 2
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         Is OSPREY tool?  ──NO──► EXIT (allow)
              │
             YES
              │
              ▼
         Load config.yml
         approval.global_mode
              │
              ▼
     ┌────────┼────────────┐
     │        │            │
  disabled  selective  all_capabilities
     │        │            │
     ▼        ▼            ▼
   EXIT    channel_write?  ASK (approve
  (allow)     │           every tool)
              ▼
          ──YES──► ASK (channel details)
              │
             NO
              │
              ▼
          execute?
              │
          ──YES──► write mode OR  ──YES──► ASK (with
              │    write patterns?         notebook link)
             NO         │
              │        NO
              ▼         │
           EXIT ◄───────┘
          (allow)
```

## Details

Supports three modes from `approval.global_mode` in config:
- **disabled**: all tools allowed without prompt
- **all_capabilities**: every OSPREY tool call requires approval
- **selective** (default): only `channel_write` and write-mode `execute`
  need approval; includes pattern detection for EPICS write calls

Creates a pre-execution notebook artifact for code review when approving
`execute` with write patterns.

PROMPT-PROVIDER: This hook contains facility-customizable static text:
  - build_approval_output(): Approval prompt message (section=approval_prompt)
  - Approval reason messages for channel_write and execute (section=approval_reasons)
  Future: source from FrameworkPromptProvider.get_approval_messages()
  Facility-customizable: approval prompt wording,
  reason detail format, severity/tone of approval messages
"""

import json
import os
import sys
from pathlib import Path

import yaml

OSPREY_PREFIXES = (
    "mcp__controls__",
    "mcp__python__",
    "mcp__workspace__",
)

# Pattern detection: prefer framework module (regex-based, config-driven, 11+ patterns)
# with graceful fallback to basic substring matching if osprey not installed
try:
    from osprey.services.python_executor.analysis.pattern_detection import (
        detect_control_system_operations,
    )

    def has_write_patterns(code: str) -> bool:
        """Check if code contains control system write patterns (framework detection)."""
        return detect_control_system_operations(code)["has_writes"]

except ImportError:
    _FALLBACK_WRITE_PATTERNS = ["caput(", "write_channel(", ".put("]

    def has_write_patterns(code: str) -> bool:  # type: ignore[misc]
        """Check if code contains control system write patterns (fallback)."""
        return any(p in code for p in _FALLBACK_WRITE_PATTERNS)


def load_osprey_config():
    config_path = Path(
        os.path.expandvars(os.environ.get("OSPREY_CONFIG", str(Path.cwd() / "config.yml")))
    )
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def build_approval_output(reason_detail: str) -> dict:
    # PROMPT-PROVIDER: section=approval_prompt
    # Future: source approval message template from FrameworkPromptProvider
    # Facility-customizable: header text, instructions, severity/tone
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "ask",
            "permissionDecisionReason": (
                "\u26a0\ufe0f  OSPREY APPROVAL REQUIRED\n\n"
                f"{reason_detail}\n\n"
                "Review the operation above and approve to proceed."
            ),
        }
    }


def build_allow_output() -> dict:
    """Explicit allow decision — overrides static permission lists."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


def _create_pre_execution_notebook(code: str, exec_mode: str, config: dict) -> str | None:
    """Create a pre-execution notebook artifact for code review.

    Returns the gallery URL if successful, None otherwise.
    Failures are silently swallowed — notebook creation must never break approval.
    """
    try:
        import nbformat

        from osprey.mcp_server.artifact_store import ArtifactStore

        # Build a minimal notebook with the code to be reviewed
        cells = []
        cells.append(
            nbformat.v4.new_markdown_cell(
                f"# Pre-Execution Review\n\n"
                f"**Mode:** `{exec_mode or 'unspecified'}`  \n"
                f"**Status:** Pending approval  \n"
            )
        )
        cells.append(nbformat.v4.new_code_cell(code))
        nb = nbformat.v4.new_notebook()
        nb.cells = cells

        nb_bytes = nbformat.writes(nb).encode()

        store = ArtifactStore()
        store.save_file(
            file_content=nb_bytes,
            filename="pre_execution_review.ipynb",
            artifact_type="notebook",
            title="Pre-Execution Review",
            description=f"Code pending approval (mode: {exec_mode or 'unspecified'})",
            mime_type="application/x-ipynb+json",
            tool_source="osprey_approval",
        )

        # Build gallery URL
        art_config = config.get("artifact_server", {})
        host = art_config.get("host", "127.0.0.1")
        port = art_config.get("port", 8086)
        return f"http://{host}:{port}#focus"
    except Exception:
        return None


def main():
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect OSPREY tools
    matched_prefix = None
    for prefix in OSPREY_PREFIXES:
        if tool_name.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    short_name = tool_name[len(matched_prefix) :]

    config = load_osprey_config()
    approval_config = config.get("approval", {})
    mode = approval_config.get("global_mode", "selective")

    # Disabled — explicitly allow everything
    if mode == "disabled":
        json.dump(build_allow_output(), sys.stdout)
        sys.exit(0)

    # All-capabilities — approve every OSPREY tool
    if mode == "all_capabilities":
        reason = f"Tool: {short_name}\nApproval mode: all_capabilities\nAll OSPREY tool calls require approval."
        json.dump(build_approval_output(reason), sys.stdout)
        sys.exit(0)

    # Selective mode
    if mode == "selective":
        # channel_write always needs approval
        if short_name == "channel_write":
            channels = tool_input.get("operations", [])
            if not channels:
                ch = tool_input.get("channel")
                val = tool_input.get("value")
                if ch is not None:
                    channels = [{"channel": ch, "value": val}]
            channel_list = ", ".join(f"{op.get('channel')}={op.get('value')}" for op in channels)
            reason = f"Channel write: {channel_list or 'unknown'}"
            json.dump(build_approval_output(reason), sys.stdout)
            sys.exit(0)

        # execute: approve if mode=="write" or code has write patterns
        if short_name == "execute":
            exec_mode = tool_input.get("execution_mode", "")
            code = tool_input.get("code", "")

            needs_approval = exec_mode == "write" or has_write_patterns(code)
            if needs_approval:
                reason_parts = [f"Python execution (mode: {exec_mode or 'unspecified'})"]
                if has_write_patterns(code):
                    reason_parts.append("Code contains control system write patterns.")

                # Create pre-execution notebook for review
                gallery_link = _create_pre_execution_notebook(code, exec_mode, config)
                if gallery_link:
                    reason_parts.append(f"\nReview notebook: {gallery_link}")

                reason = "\n".join(reason_parts)
                json.dump(build_approval_output(reason), sys.stdout)
                sys.exit(0)

    # No approval needed — explicitly allow so hook decision overrides static lists
    json.dump(build_allow_output(), sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
