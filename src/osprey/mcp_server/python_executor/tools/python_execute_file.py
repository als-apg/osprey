"""MCP tool: execute_file — run an existing Python file with safety checks."""

import logging
from pathlib import Path

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async
from osprey.mcp_server.python_executor.server import mcp
from osprey.mcp_server.python_executor.tools._execution_gates import (
    enforce_posture_clamp,
    require_known_execution_mode,
    run_gated_execution,
)

logger = logging.getLogger("osprey.mcp_server.tools.execute_file")


@mcp.tool()
async def execute_file(
    file_path: str,
    description: str,
    execution_mode: str = "readonly",
    script_args: list[str] | None = None,
    save_output: bool = True,
) -> str:
    """Execute an existing Python file with the same safety pipeline as ``execute``.

    The file is read, safety-checked, augmented with a script identity preamble
    (``sys.argv``, ``__file__``), and run in a subprocess, in the project's own
    Python environment when the project has one (otherwise OSPREY's own
    interpreter) — the same execution path used by the ``execute`` tool.

    Args:
        file_path: Path to a ``.py`` file.  Absolute paths are used as-is;
                   relative paths resolve against the project root.
        description: Human-readable description of what the script does.
        execution_mode: "readonly" (default) refuses every control-system
                        write; "readwrite" permits writes through
                        ``osprey.runtime.write_channel`` / ``write_channels``
                        after human approval, and still refuses raw client
                        puts. Any other value is rejected.
        script_args: Optional command-line arguments for the script
                     (populates ``sys.argv[1:]``).
        save_output: If True, save the code and output to a workspace data file.

    Returns:
        JSON with a compact summary (truncated stdout/stderr) and a data file path.
    """
    if not file_path or not file_path.strip():
        return make_error(
            "validation_error",
            "No file path provided.",
            ["Provide a path to a Python (.py) file."],
        )

    # Reject unrecognised modes before any gate: the write gates branch on
    # string equality and an unknown value would satisfy neither branch.
    require_known_execution_mode(execution_mode)

    # Session posture clamp, ahead of resolving and reading the file: a
    # sandboxed caller is told about the posture rather than about the path.
    # See :func:`run_gated_execution` for why the clamp precedes every gate.
    enforce_posture_clamp(execution_mode, tool="execute_file")

    # Resolve project root and file path
    from osprey.mcp_server.python_executor.executor import _resolve_project_root

    project_root = _resolve_project_root()
    target = Path(file_path)
    if not target.is_absolute():
        target = project_root / target
    resolved = target.resolve()

    # Containment check — file must be within project root
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        return make_error(
            "validation_error",
            f"File path is outside the project root: {file_path}",
            ["Provide a path within the project directory."],
        )

    # Validate file
    if not resolved.exists():
        return make_error(
            "file_not_found",
            f"File not found: {file_path}",
            ["Check the file path and try again."],
        )

    if resolved.suffix != ".py":
        return make_error(
            "validation_error",
            f"Not a Python file: {file_path}",
            ["Only .py files can be executed."],
        )

    # Read file contents
    try:
        code = resolved.read_text("utf-8")
    except UnicodeDecodeError:
        return make_error(
            "validation_error",
            f"File is not valid UTF-8 text: {file_path}",
            ["Ensure the file is a text-based Python script."],
        )
    except PermissionError:
        return make_error(
            "validation_error",
            f"Permission denied reading file: {file_path}",
            ["Check file permissions."],
        )

    if not code.strip():
        return make_error(
            "validation_error",
            f"File is empty: {file_path}",
            ["Provide a non-empty Python file."],
        )

    # Script identity preamble: sys.argv and __file__ point at the original file.
    argv_items = [str(resolved)]
    if script_args:
        argv_items.extend(script_args)
    preamble = f"import sys\nsys.argv = {argv_items!r}\n__file__ = {str(resolved)!r}\n"
    augmented_code = preamble + "\n" + code

    # Every gate, the launch and the run's reporting, in the order the
    # ``execute`` tool runs them — see :func:`run_gated_execution`. The gates
    # and the audit trail see the original file contents, not the preamble, so
    # the record shows what the operator would read; the subprocess gets the
    # augmented code. The clamp above runs again inside the helper as a no-op
    # re-check.
    exec_result, patterns = await run_gated_execution(
        tool="execute_file",
        code=augmented_code,
        record_code=code,
        description=description,
        execution_mode=execution_mode,
        project_root=project_root,
        notify=notify_agent_activity_async,
    )

    # Build response using original code (not augmented) for metadata/notebook
    from osprey.mcp_server.python_executor.tools._response_builder import build_execution_response

    return await build_execution_response(
        code=code,
        description=description,
        execution_mode=execution_mode,
        exec_result=exec_result,
        patterns=patterns,
        save_output=save_output,
        tool_source="execute_file",
    )
