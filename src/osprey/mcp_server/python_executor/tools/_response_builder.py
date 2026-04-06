"""Shared post-execution response builder for Python executor tools."""

import json
import logging

import nbformat

from osprey.mcp_server.python_executor.executor import ExecutionResult

logger = logging.getLogger("osprey.mcp_server.tools.execute")

# Maximum characters of stdout/stderr returned inline in the summary.
_STDOUT_PREVIEW_LIMIT = 500
_STDERR_PREVIEW_LIMIT = 500


async def build_execution_response(
    code: str,
    description: str,
    execution_mode: str,
    exec_result: ExecutionResult,
    patterns: dict,
    save_output: bool,
    tool_source: str = "execute",
) -> str:
    """Build a JSON response from an execution result.

    Handles figure/artifact saving, notebook creation, summary building,
    ArtifactStore persistence, and gallery URL injection.

    Args:
        code: Original source code (for notebook/metadata — not augmented).
        description: Human-readable description of what the code does.
        execution_mode: "readonly" or "readwrite".
        exec_result: Result from ``execute_code()``.
        patterns: Dict from ``detect_control_system_operations()``.
        save_output: If True, persist to ArtifactStore; otherwise return inline.
        tool_source: Tool identifier for stored metadata ("execute" or "execute_file").

    Returns:
        JSON string with execution summary and artifact references.
    """
    artifact_ids: list[str] = []

    stdout_text = exec_result.stdout
    stderr_text = exec_result.stderr

    # Auto-save figure artifacts collected by the adapter
    for fig_path in exec_result.figures:
        try:
            from osprey.stores.artifact_store import get_artifact_store

            store = get_artifact_store()
            fig_entry = store.save_file(
                file_content=fig_path.read_bytes(),
                filename=fig_path.name,
                artifact_type="image",
                title=f"Figure: {fig_path.stem}",
                description=f"Figure from: {description}",
                mime_type=f"image/{fig_path.suffix.lstrip('.')}",
                tool_source=tool_source,
            )
            artifact_ids.append(fig_entry.id)
        except Exception:
            logger.debug("Figure artifact save failed for %s (non-fatal)", fig_path, exc_info=True)

    # Auto-save artifacts collected from subprocess save_artifact() calls
    for art in exec_result.artifacts:
        try:
            from osprey.stores.artifact_store import get_artifact_store

            store = get_artifact_store()
            art_entry = store.save_file(
                file_content=art["path"].read_bytes(),
                filename=art["path"].name,
                artifact_type=art["artifact_type"],
                title=art["title"],
                description=art["description"],
                mime_type=art["mime_type"],
                tool_source=tool_source,
                category=art.get("category", ""),
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
        from osprey.stores.notebook_renderer import create_notebook_from_code

        nb = create_notebook_from_code(code, description, stdout_text, stderr_text)
        nb_bytes = nbformat.writes(nb).encode()

        from osprey.stores.artifact_store import get_artifact_store

        store = get_artifact_store()
        nb_entry = store.save_file(
            file_content=nb_bytes,
            filename=f"{description[:40].replace(' ', '_')}.ipynb",
            artifact_type="notebook",
            title=f"Notebook: {description}",
            description=description,
            mime_type="application/x-ipynb+json",
            tool_source=tool_source,
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
    from osprey.stores.artifact_store import get_artifact_store

    store = get_artifact_store()
    entry = store.save_data(
        tool=tool_source,
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
            from osprey.mcp_server.http import gallery_url

            response["gallery_url"] = gallery_url()
        except Exception:
            pass
    if notebook_artifact_id:
        response["notebook_artifact_id"] = notebook_artifact_id
    return json.dumps(response, default=str)
