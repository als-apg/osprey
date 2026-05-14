"""MCP tool: execute — run user-provided Python code with safety checks."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.python_executor.server import mcp

logger = logging.getLogger("osprey.mcp_server.tools.execute")


@mcp.tool()
async def execute(
    code: str,
    description: str,
    execution_mode: str = "readonly",
    save_output: bool = True,
) -> str:
    """Execute Python code with process isolation, limits enforcement, and timeout.

    Code runs in a container or local subprocess (configured via config.yml).
    All packages installed in the deployment environment are available
    (e.g. numpy, pandas, scipy, at, matplotlib, plotly).

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
        return make_error(
                "validation_error",
                "No code provided.",
                ["Provide Python code to execute."],
            )

    # Pre-execution safety checks (syntax, security, imports)
    try:
        from osprey.services.python_executor.analysis.safety_checks import quick_safety_check

        passed, safety_issues = quick_safety_check(code)
        if not passed:
            return make_error(
                    "safety_error",
                    "Code failed pre-execution safety checks.",
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

    # Deployment-level kill switch (independent of pattern detection accuracy).
    # Fires whenever the caller asks for write mode, regardless of whether the
    # pattern detector recognises specific write syntax in `code`.
    if execution_mode == "readwrite":
        try:
            from osprey.services.python_executor.execution.control import (
                get_execution_control_config,
            )

            exec_control_config = get_execution_control_config()
        except ImportError:
            logger.warning(
                "Execution control config unavailable — skipping deployment-level writes check"
            )
            exec_control_config = None

        if exec_control_config is not None and exec_control_config.control_system_writes_enabled is False:
            return make_error(
                    "safety_error",
                    "Control-system writes are disabled in this deployment "
                    "(control_system.writes_enabled=false in project config).",
                    [
                        "Set control_system.writes_enabled=true in the project config to enable writes.",
                    ],
                )

    # Per-call execution-mode gate (uses pattern detection — readonly-mode safety).
    if patterns.get("has_writes") and execution_mode == "readonly":
        return make_error(
                "safety_error",
                "Control-system write patterns detected in readonly mode.",
                [
                    "Set execution_mode to 'readwrite' if writes are intentional.",
                    "Detected patterns: " + json.dumps(patterns.get("detected_patterns", {})),
                ],
            )

    # Execute code via adapter (container or subprocess)
    from osprey.mcp_server.python_executor.executor import execute_code

    exec_result = await execute_code(
        code=code,
        execution_mode=execution_mode,
        description=description,
    )

    from osprey.mcp_server.python_executor.tools._response_builder import build_execution_response

    return await build_execution_response(
        code=code,
        description=description,
        execution_mode=execution_mode,
        exec_result=exec_result,
        patterns=patterns,
        save_output=save_output,
        tool_source="execute",
    )
