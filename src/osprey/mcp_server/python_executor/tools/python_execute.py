"""MCP tool: execute — run user-provided Python code with safety checks.

PROMPT-PROVIDER: This tool's docstring is a static prompt visible to Claude Code.
  Future: source from FrameworkPromptProvider.get_python_prompt_builder()
  Facility-customizable: tool description, supported artifact types,
  execution mode descriptions, write pattern detection terminology
"""

import json
import logging

import nbformat

from osprey.mcp_server.common import make_error
from osprey.mcp_server.python_executor.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.execute")

# Maximum characters of stdout/stderr returned inline in the summary.
_STDOUT_PREVIEW_LIMIT = 500
_STDERR_PREVIEW_LIMIT = 500


@mcp.tool()
async def execute(
    code: str,
    description: str,
    execution_mode: str = "readonly",
    save_output: bool = True,
) -> str:
    """Execute Python code with process isolation, limits enforcement, and timeout.

    Code runs in a container or local subprocess (configured via config.yml).

    Safety layers applied before execution:
      1. ``quick_safety_check()`` — blocks exec/eval/__import__/subprocess
      2. ``detect_control_system_operations()`` — blocks writes in readonly mode
    Safety layers applied during execution:
      3. ``ExecutionWrapper`` monkeypatch — validates epics.caput() against limits DB
      4. Process isolation — code runs outside the MCP server process
      5. Execution timeout — kills execution after configured timeout

    A ``save_artifact(obj, title, description)`` helper is available in the
    subprocess for saving objects to the artifact gallery.

    Args:
        code: Python source code to execute.
        description: Human-readable description of what the code does.
        execution_mode: "readonly" (default) blocks detected write patterns;
                        "readwrite" allows them.
        save_output: If True, save the code and output to a workspace data file.

    Returns:
        JSON with a compact summary (truncated stdout/stderr) and a data file path.
    """
    if not code or not code.strip():
        return json.dumps(
            make_error(
                "validation_error",
                "No code provided.",
                ["Provide Python code to execute."],
            )
        )

    # --- Pre-execution safety checks (syntax, security, imports) ---
    try:
        from osprey.services.python_executor.analysis.safety_checks import quick_safety_check

        passed, safety_issues = quick_safety_check(code)
        if not passed:
            return json.dumps(
                make_error(
                    "safety_error",
                    "Code failed pre-execution safety checks.",
                    safety_issues,
                )
            )
    except ImportError:
        pass  # Framework analysis module unavailable — continue with pattern detection only

    # --- Pattern detection ---
    try:
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        patterns = detect_control_system_operations(code)
    except Exception:
        patterns = {"has_writes": False, "has_reads": False, "detected_patterns": {}}

    if patterns.get("has_writes") and execution_mode == "readonly":
        return json.dumps(
            make_error(
                "safety_error",
                "Control-system write patterns detected in readonly mode.",
                [
                    "Set execution_mode to 'readwrite' if writes are intentional.",
                    "Detected patterns: " + json.dumps(patterns.get("detected_patterns", {})),
                ],
            )
        )

    # --- Execute code via adapter (container / subprocess) ---
    artifact_ids: list[str] = []

    from osprey.mcp_server.python_executor.executor import execute_code

    exec_result = await execute_code(
        code=code,
        execution_mode=execution_mode,
        description=description,
    )

    stdout_text = exec_result.stdout
    stderr_text = exec_result.stderr

    # Auto-save figure artifacts collected by the adapter
    for fig_path in exec_result.figures:
        try:
            from osprey.mcp_server.artifact_store import get_artifact_store

            store = get_artifact_store()
            fig_entry = store.save_file(
                file_content=fig_path.read_bytes(),
                filename=fig_path.name,
                artifact_type="image",
                title=f"Figure: {fig_path.stem}",
                description=f"Figure from: {description}",
                mime_type=f"image/{fig_path.suffix.lstrip('.')}",
                tool_source="execute",
            )
            artifact_ids.append(fig_entry.id)
        except Exception:
            logger.debug("Figure artifact save failed for %s (non-fatal)", fig_path, exc_info=True)

    # Auto-save artifacts collected from subprocess save_artifact() calls
    for art in exec_result.artifacts:
        try:
            from osprey.mcp_server.artifact_store import get_artifact_store

            store = get_artifact_store()
            art_entry = store.save_file(
                file_content=art["path"].read_bytes(),
                filename=art["path"].name,
                artifact_type=art["artifact_type"],
                title=art["title"],
                description=art["description"],
                mime_type=art["mime_type"],
                tool_source="execute",
            )
            artifact_ids.append(art_entry.id)
        except Exception:
            logger.debug(
                "Subprocess artifact save failed for %s (non-fatal)",
                art.get("title", "unknown"),
                exc_info=True,
            )

    has_errors = not exec_result.success
    full_result = {
        "description": description,
        "execution_mode": execution_mode,
        "execution_method": exec_result.execution_method_used,
        "code": code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "has_errors": has_errors,
        "detected_patterns": patterns,
        "artifact_ids": artifact_ids,
    }

    # Auto-save a notebook artifact for every execution
    notebook_artifact_id = None
    try:
        from osprey.mcp_server.notebook_renderer import create_notebook_from_code

        nb = create_notebook_from_code(code, description, stdout_text, stderr_text)
        nb_bytes = nbformat.writes(nb).encode()

        from osprey.mcp_server.artifact_store import get_artifact_store

        store = get_artifact_store()
        nb_entry = store.save_file(
            file_content=nb_bytes,
            filename=f"{description[:40].replace(' ', '_')}.ipynb",
            artifact_type="notebook",
            title=f"Notebook: {description}",
            description=description,
            mime_type="application/x-ipynb+json",
            tool_source="execute",
        )
        notebook_artifact_id = nb_entry.id
        artifact_ids.append(notebook_artifact_id)
    except Exception:
        logger.debug("Notebook artifact creation failed (non-fatal)", exc_info=True)

    if not save_output:
        # Return inline (legacy-compatible path for callers that opt out)
        result = {
            "description": description,
            "execution_mode": execution_mode,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "has_errors": has_errors,
            "detected_patterns": patterns,
        }
        if artifact_ids:
            result["artifact_ids"] = artifact_ids
        return json.dumps(result, default=str)

    # Build compact summary inline
    summary = {
        "description": description,
        "status": "Failed" if has_errors else "Success",
        "output": stdout_text[:_STDOUT_PREVIEW_LIMIT],
        "error": stderr_text[:_STDERR_PREVIEW_LIMIT] if stderr_text else None,
        "has_errors": has_errors,
        "detected_patterns": patterns,
    }
    if artifact_ids:
        summary["artifact_ids"] = artifact_ids
        summary["artifact_count"] = len(artifact_ids)
    access_details = {
        "execution_mode": execution_mode,
        "code_lines": len(code.splitlines()),
        "stdout_length": len(stdout_text),
        "stderr_length": len(stderr_text),
    }

    # Save via ArtifactStore (unified)
    from osprey.mcp_server.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.save_data(
        tool="execute",
        data=full_result,
        title=description,
        description=description,
        summary=summary,
        access_details=access_details,
        category="code_output",
    )

    response = entry.to_tool_response()
    if artifact_ids:
        response["artifact_ids"] = artifact_ids
        try:
            from osprey.mcp_server.common import gallery_url

            response["gallery_url"] = gallery_url()
        except Exception:
            pass
    if notebook_artifact_id:
        response["notebook_artifact_id"] = notebook_artifact_id
    return json.dumps(response, default=str)
