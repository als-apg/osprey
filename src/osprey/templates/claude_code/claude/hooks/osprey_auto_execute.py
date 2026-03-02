#!/usr/bin/env python3
"""
---
name: Auto Execute Scripts
description: Automatically executes .py scripts written to osprey-workspace/scripts/ via the Write tool
summary: Auto-executes analysis scripts and returns results via additionalContext
event: PostToolUse
tools: Write
---

## Flow

```
stdin ──► Parse JSON
              │
              ▼
         tool_name == "Write"?  ──NO──► EXIT (silent)
              │
             YES
              │
              ▼
         file_path ends in .py?  ──NO──► EXIT (silent)
              │
             YES
              │
              ▼
         In osprey-workspace/  ──NO──► EXIT (silent)
         scripts/?
              │
             YES
              │
              ▼
         Code empty?  ──YES──► EXIT (silent)
              │
             NO
              │
              ▼
         Safety check passes?  ──NO──► additionalContext: "SKIPPED: safety"
              │
             YES
              │
              ▼
         Has write_channel()?  ──YES──► additionalContext: "SKIPPED: readonly"
              │
             NO
              │
              ▼
         execute_code(code, "readonly", description)
              │
              ▼
         Save notebook artifact
              │
              ▼
         additionalContext: execution results
```

## Details

PostToolUse hook for the Write tool. When Claude writes a ``.py`` file to
``osprey-workspace/scripts/``, this hook automatically executes it through
the existing Python executor pipeline and returns results via
``additionalContext``.

The execution is always ``readonly`` — for scripts that write to the control
system, Claude should use the ``execute`` MCP tool directly (which includes
human approval).

Always exits 0. Errors are reported via ``additionalContext``, never by
blocking the hook.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osprey_hook_log import get_hook_input, get_project_dir, log_hook


def _is_scripts_target(file_path: str, project_dir: str) -> bool:
    """Check if the file is a .py file directly in osprey-workspace/scripts/."""
    if not file_path.endswith(".py"):
        return False

    try:
        target = Path(file_path).expanduser().resolve()
    except (OSError, ValueError):
        return False

    scripts_dir = (Path(project_dir) / "osprey-workspace" / "scripts").resolve()

    try:
        target.relative_to(scripts_dir)
    except ValueError:
        return False

    return target.parent == scripts_dir


def _description_from_filename(file_path: str) -> str:
    """Derive a human-readable description from the script filename.

    Example: 'analyze_beam_data.py' -> 'analyze beam data'
    """
    name = Path(file_path).stem
    return name.replace("_", " ")


def _emit(additional_context: str) -> None:
    """Write PostToolUse additionalContext to stdout."""
    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": additional_context,
        }
    }
    json.dump(output, sys.stdout)


def _run_execution(code: str, description: str, project_dir: str) -> str:
    """Execute the script and return the additionalContext string.

    Uses the existing executor pipeline. Saves a notebook artifact
    with the code and results.
    """
    from osprey.mcp_server.python_executor.executor import execute_code

    result = asyncio.run(execute_code(code, "readonly", description))

    # Save notebook artifact
    artifact_ids = []
    try:
        from osprey.mcp_server.artifact_store import ArtifactStore
        from osprey.mcp_server.notebook_renderer import create_notebook_from_code

        notebook = create_notebook_from_code(
            code, description, stdout=result.stdout, stderr=result.stderr
        )

        import nbformat

        store = ArtifactStore()
        nb_content = nbformat.writes(notebook).encode("utf-8")
        entry = store.save_file(
            file_content=nb_content,
            filename=f"{description.replace(' ', '_')}.ipynb",
            artifact_type="notebook",
            title=description,
            description=f"Auto-executed script: {description}",
            mime_type="application/x-ipynb+json",
            tool_source="auto_execute",
        )
        artifact_ids.append(entry.id)
    except Exception:
        pass  # Artifact saving is best-effort

    # Save figure artifacts
    for fig_path in result.figures:
        try:
            store = ArtifactStore()
            entry = store.save_from_path(
                source_path=fig_path,
                title=f"{description} - {fig_path.name}",
                description=f"Figure from auto-executed script: {description}",
                tool_source="auto_execute",
            )
            artifact_ids.append(entry.id)
        except Exception:
            pass

    # Build additionalContext
    status = "SUCCESS" if result.success else "FAILED"
    parts = [f"SCRIPT AUTO-EXECUTED [{status}]"]

    if result.stdout:
        # Preview first 2000 chars of stdout
        preview = result.stdout[:2000]
        if len(result.stdout) > 2000:
            preview += f"\n... ({len(result.stdout)} chars total, truncated)"
        parts.append(f"Output:\n{preview}")

    if result.error_message:
        parts.append(f"Error: {result.error_message}")
    elif result.stderr and not result.success:
        parts.append(f"Stderr: {result.stderr[:1000]}")

    if result.execution_time_seconds is not None:
        parts.append(f"Execution time: {result.execution_time_seconds:.1f}s")

    if artifact_ids:
        parts.append(f"Artifacts saved: {', '.join(artifact_ids)}")

    return "\n\n".join(parts)


def main():
    hook_input = get_hook_input()
    if not hook_input:
        sys.exit(0)

    tool_name = hook_input.get("tool_name", "")
    if tool_name != "Write":
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path or not file_path.endswith(".py"):
        sys.exit(0)

    project_dir = get_project_dir(hook_input) or os.getcwd()

    if not _is_scripts_target(file_path, project_dir):
        sys.exit(0)

    code = tool_input.get("content", "")
    if not code or not code.strip():
        sys.exit(0)

    log_hook("auto-execute", hook_input, status="start", detail=f"file={file_path}")

    # Safety check
    try:
        from osprey.services.python_executor.analysis.safety_checks import (
            quick_safety_check,
        )

        passed, issues = quick_safety_check(code)
        if not passed:
            msg = "SCRIPT AUTO-EXECUTE SKIPPED: Safety check failed\n" + "\n".join(
                f"  - {i}" for i in issues
            )
            log_hook("auto-execute", hook_input, status="skipped", detail="safety")
            _emit(msg)
            sys.exit(0)
    except ImportError:
        pass  # Safety module not available — proceed anyway

    # Check for control system writes
    try:
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        ops = detect_control_system_operations(code)
        if ops.get("has_writes"):
            msg = (
                "SCRIPT AUTO-EXECUTE SKIPPED: Script contains control system write operations.\n"
                "Use the `execute` MCP tool directly for scripts that write to the control system "
                "(requires human approval)."
            )
            log_hook("auto-execute", hook_input, status="skipped", detail="writes")
            _emit(msg)
            sys.exit(0)
    except ImportError:
        pass  # Pattern detection not available — proceed anyway

    # Execute
    description = _description_from_filename(file_path)

    try:
        context = _run_execution(code, description, project_dir)
        log_hook("auto-execute", hook_input, status="done", detail=f"file={file_path}")
        _emit(context)
    except Exception as exc:
        msg = f"SCRIPT AUTO-EXECUTE FAILED: {type(exc).__name__}: {exc}"
        log_hook("auto-execute", hook_input, status="error", detail=str(exc)[:200])
        _emit(msg)

    sys.exit(0)


if __name__ == "__main__":
    main()
