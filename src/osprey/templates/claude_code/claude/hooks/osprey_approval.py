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
   EXIT    channel_write?  ASK (approve every tool;
  (allow)     │            + notebook if execute)
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

Creates a pre-execution notebook artifact for code review whenever
`execute` requires approval (write mode, write patterns, or all_capabilities).
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import (
    get_hook_input,
    load_hook_config,
    load_osprey_config,
    log_hook,
)

# Fallback write patterns: used when osprey is not importable (e.g., standalone hook).
# Must stay in sync with get_framework_standard_patterns()["write"] (15 patterns).
# The parity test in test_approval_hook.py enforces this.
_FALLBACK_WRITE_PATTERNS = [
    # osprey.runtime unified API
    r"\bwrite_channel\s*\(",
    r"\bwrite_channels\s*\(",
    # EPICS (PyEPICS)
    r"\bcaput\s*\(",
    r"epics\.caput\(",
    r"\.put\s*\(",
    r"\.set_value\s*\(",
    r"PV\([^)]*\)\.put",
    r"epics\.PV\([^)]*\)\.put",
    # Tango (PyTango)
    r"DeviceProxy\([^)]*\)\.write_attribute\(",
    r"\.write_attribute\s*\(",
    r"\.write_attribute_asynch\s*\(",
    r"tango\.DeviceProxy\([^)]*\)\.write",
    # LabVIEW
    r"labview\.set_control\(",
    r"\.SetControlValue\(",
    # Direct connector access
    r"connector\.write_channel\(",
]

# Pattern detection: prefer framework module (regex-based, config-driven, 15 patterns)
# with graceful fallback to regex matching against _FALLBACK_WRITE_PATTERNS
try:
    from osprey.services.python_executor.analysis.pattern_detection import (
        detect_control_system_operations,
    )

    def has_write_patterns(code: str, config: dict | None = None) -> bool:
        """Check if code contains control system write patterns (framework detection)."""
        patterns = None
        pattern_mode = None
        if config:
            pat_config = config.get("control_system", {}).get("patterns")
            if pat_config:
                patterns = pat_config
                pattern_mode = pat_config.get("mode")
        return detect_control_system_operations(code, patterns=patterns, pattern_mode=pattern_mode)[
            "has_writes"
        ]

except ImportError:

    def has_write_patterns(code: str, config: dict | None = None) -> bool:  # type: ignore[misc]
        """Check if code contains control system write patterns (fallback)."""
        patterns = list(_FALLBACK_WRITE_PATTERNS)
        if config:
            pat_config = config.get("control_system", {}).get("patterns", {})
            custom = pat_config.get("write")
            mode = pat_config.get("mode", "extend")
            if custom:
                if mode == "override":
                    patterns = list(custom)
                else:
                    patterns.extend(p for p in custom if p not in patterns)
        return any(re.search(p, code) for p in patterns)


def build_approval_output(reason_detail: str) -> dict:
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


def _focus_artifact(gallery_base_url: str, artifact_id: str) -> None:
    """Fire-and-forget POST to bring an artifact into focus in the gallery.

    Non-fatal: silently swallows errors so focus failures never block approval.
    """
    try:
        import urllib.request

        data = json.dumps({"artifact_id": artifact_id}).encode()
        req = urllib.request.Request(
            f"{gallery_base_url}/api/focus",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        pass


def _create_pre_execution_notebook(code: str, exec_mode: str, config: dict) -> str | None:
    """Create a pre-execution notebook artifact for code review.

    Returns the gallery URL if successful, None otherwise.
    Failures are silently swallowed — notebook creation must never break approval.
    """
    try:
        import nbformat

        from osprey.stores.artifact_store import ArtifactStore

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
        entry = store.save_file(
            file_content=nb_bytes,
            filename="pre_execution_review.ipynb",
            artifact_type="notebook",
            title="Pre-Execution Review",
            description=f"Code pending approval (mode: {exec_mode or 'unspecified'})",
            mime_type="application/x-ipynb+json",
            tool_source="osprey_approval",
        )

        # Build gallery URL and bring the notebook into focus
        art_config = config.get("artifact_server", {})
        host = art_config.get("host", "127.0.0.1")
        port = art_config.get("port", 8086)
        base_url = f"http://{host}:{port}"

        # Fire-and-forget POST to switch gallery focus to this notebook
        _focus_artifact(base_url, entry.id)

        return f"{base_url}#focus"
    except Exception:
        return None


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")

    # Only inspect OSPREY tools (prefixes loaded from hook_config.json)
    _prefixes = load_hook_config().get("approval_prefixes", [])
    matched_prefix = None
    for prefix in _prefixes:
        if tool_name.startswith(prefix):
            matched_prefix = prefix
            break
    if matched_prefix is None:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    short_name = tool_name[len(matched_prefix) :]

    config = load_osprey_config(hook_input)
    approval_config = config.get("approval", {})
    mode = approval_config.get("global_mode", "selective")

    # Disabled — explicitly allow everything
    if mode == "disabled":
        log_hook("approval", hook_input, status="allow", detail="mode=disabled")
        json.dump(build_allow_output(), sys.stdout)
        sys.exit(0)

    # All-capabilities — approve every OSPREY tool
    if mode == "all_capabilities":
        reason_parts = [
            f"Tool: {short_name}",
            "Approval mode: all_capabilities",
            "All OSPREY tool calls require approval.",
        ]

        # Create a review notebook when the tool carries code
        if short_name == "execute":
            code = tool_input.get("code", "")
            exec_mode = tool_input.get("execution_mode", "")
            if code.strip():
                gallery_link = _create_pre_execution_notebook(code, exec_mode, config)
                if gallery_link:
                    reason_parts.append(f"\nReview notebook: {gallery_link}")

        reason = "\n".join(reason_parts)
        log_hook(
            "approval", hook_input, status="ask", detail=f"mode=all_capabilities tool={short_name}"
        )
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
            log_hook("approval", hook_input, status="ask", detail="channel_write")
            json.dump(build_approval_output(reason), sys.stdout)
            sys.exit(0)

        # execute: approve if mode=="write" or code has write patterns
        if short_name == "execute":
            exec_mode = tool_input.get("execution_mode", "")
            code = tool_input.get("code", "")

            writes_detected = has_write_patterns(code, config)
            needs_approval = exec_mode == "write" or writes_detected
            if needs_approval:
                reason_parts = [f"Python execution (mode: {exec_mode or 'unspecified'})"]
                if writes_detected:
                    reason_parts.append("Code contains control system write patterns.")

                # Create pre-execution notebook so reviewer can inspect the code
                if code.strip():
                    gallery_link = _create_pre_execution_notebook(code, exec_mode, config)
                    if gallery_link:
                        reason_parts.append(f"\nReview notebook: {gallery_link}")

                reason = "\n".join(reason_parts)
                log_hook("approval", hook_input, status="ask", detail="execute_write")
                json.dump(build_approval_output(reason), sys.stdout)
                sys.exit(0)

    # No approval needed — explicitly allow so hook decision overrides static lists
    log_hook("approval", hook_input, status="allow")
    json.dump(build_allow_output(), sys.stdout)
    sys.exit(0)


if __name__ == "__main__":
    main()
