"""MCP tool: execute_file — run an existing Python file with safety checks."""

import json
import logging
from pathlib import Path

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.python_executor.server import mcp

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
    (``sys.argv``, ``__file__``), and executed via the same container/subprocess
    adapter used by the ``execute`` tool.

    Args:
        file_path: Path to a ``.py`` file.  Absolute paths are used as-is;
                   relative paths resolve against the project root.
        description: Human-readable description of what the script does.
        execution_mode: "readonly" (default) blocks detected write patterns;
                        "readwrite" allows them.
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

    # Pre-execution safety checks on original file contents
    try:
        from osprey.services.python_executor.analysis.safety_checks import quick_safety_check

        passed, safety_issues = quick_safety_check(code)
        if not passed:
            return make_error(
                    "safety_error",
                    "File failed pre-execution safety checks.",
                    safety_issues,
                )
    except ImportError:
        logger.warning("Safety check module unavailable — executing without pre-checks")

    # Pattern detection (block writes in readonly mode)
    try:
        from osprey.services.python_executor.analysis.pattern_detection import (
            detect_control_system_operations,
        )

        patterns = detect_control_system_operations(code)
    except ImportError:
        logger.warning("Pattern detection module unavailable — skipping write detection")
        patterns = {"has_writes": False, "has_reads": False, "detected_patterns": {}}

    if patterns.get("has_writes") and execution_mode == "readonly":
        return make_error(
                "safety_error",
                "Control-system write patterns detected in readonly mode.",
                [
                    "Set execution_mode to 'readwrite' if writes are intentional.",
                    "Detected patterns: " + json.dumps(patterns.get("detected_patterns", {})),
                ],
            )

    # Build preamble: set sys.argv and __file__ to point at the original script
    argv_items = [str(resolved)]
    if script_args:
        argv_items.extend(script_args)
    preamble = f"import sys\nsys.argv = {argv_items!r}\n__file__ = {str(resolved)!r}\n"
    augmented_code = preamble + "\n" + code

    # Execute augmented code via adapter (container or subprocess)
    from osprey.mcp_server.python_executor.executor import execute_code

    exec_result = await execute_code(
        code=augmented_code,
        execution_mode=execution_mode,
        description=description,
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
