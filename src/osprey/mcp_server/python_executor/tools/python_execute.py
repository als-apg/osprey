"""MCP tool: execute — run user-provided Python code with safety checks."""

import json
import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async
from osprey.mcp_server.python_executor.server import mcp
from osprey.mcp_server.python_executor.tools._execution_gates import (
    enforce_deployment_writes_gate,
    refuse_readonly_write,
    report_runtime_refusal,
    require_known_execution_mode,
)
from osprey.mcp_server.python_executor.tools._package_inventory import with_live_packages
from osprey.services.python_executor.refusal_audit import (
    LAYER_IMPORT_DENYLIST,
    LAYER_PATTERN_DETECTION,
)

logger = logging.getLogger("osprey.mcp_server.tools.execute")


@mcp.tool()
@with_live_packages
async def execute(
    code: str,
    description: str,
    execution_mode: str = "readonly",
    save_output: bool = True,
) -> str:
    """Execute Python code with process isolation, limits enforcement, and timeout.

    Code runs in a subprocess, in the project's own Python environment when the
    project has one (otherwise OSPREY's own interpreter).
    <<AVAILABLE_PACKAGES>>

    Safety layers applied before execution:
      1. ``quick_safety_check()`` — blocks exec/eval/__import__/subprocess
      2. ``check_readonly_imports()`` — readonly runs may not import
         control-system client libraries (epics, p4p, ...) at all
      3. ``detect_control_system_operations()`` — blocks detected write
         spellings in readonly mode
    Safety layers applied during execution:
      4. Readonly guard — in readonly runs every control-system write entry
         point refuses, as do the routes that reach one without a client
         import (starting a process, ``ctypes``); the connectors refuse
         ``write_channel``, and the EPICS connector stays on the read_only
         gateway (``OSPREY_EXECUTION_MODE``)
      5. ``ExecutionWrapper`` monkeypatch — validates epics.caput() against limits DB
      6. Process isolation — code runs outside the MCP server process
      7. Execution timeout — kills execution after configured timeout

    A refused write is reported to the operator and recorded in the
    deployment's audit log, at whichever layer catches it. A readonly script
    therefore cannot shell out or load a shared library at all, even for
    unrelated work — resubmit as readwrite, which requires human approval.

    A ``save_artifact(obj, title, description)`` helper is available in the
    subprocess for saving objects to the artifact gallery.

    Args:
        code: Python source code to execute.
        description: Human-readable description of what the code does.
        execution_mode: "readonly" (default) blocks detected write patterns;
                        "readwrite" allows them. Any other value is rejected.
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

    # Reject unrecognised modes before any gate: the write gates branch on
    # string equality and an unknown value would satisfy neither branch.
    require_known_execution_mode(execution_mode)

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

    # Readonly runs may not import control-system clients at all. This is the
    # pre-execution half of the readonly contract; the runtime half (the
    # wrapper's readonly guard and the connector refusal) catches what no
    # static check can.
    if execution_mode == "readonly":
        try:
            from osprey.services.python_executor.analysis.safety_checks import (
                check_readonly_imports,
            )

            import_issues = check_readonly_imports(code)
        except ImportError:
            import_issues = []
        if import_issues:
            await refuse_readonly_write(
                tool="execute",
                layer=LAYER_IMPORT_DENYLIST,
                trigger=import_issues,
                code=code,
                description=description,
                message="Control-system client libraries cannot be imported in readonly mode.",
                suggestions=[
                    *import_issues,
                    "Use read_channel() from osprey.runtime for reads.",
                    "Set execution_mode to 'readwrite' if writes are intentional.",
                ],
            )

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
    enforce_deployment_writes_gate(execution_mode)

    # Per-call execution-mode gate (uses pattern detection — readonly-mode safety).
    if patterns.get("has_writes") and execution_mode == "readonly":
        await refuse_readonly_write(
            tool="execute",
            layer=LAYER_PATTERN_DETECTION,
            trigger=patterns.get("detected_patterns", {}),
            code=code,
            description=description,
            message="Control-system write patterns detected in readonly mode.",
            suggestions=[
                "Set execution_mode to 'readwrite' if writes are intentional.",
                "Detected patterns: " + json.dumps(patterns.get("detected_patterns", {})),
            ],
        )

    # Execute code in the subprocess backend
    from osprey.mcp_server.python_executor.executor import execute_code

    exec_result = await execute_code(
        code=code,
        execution_mode=execution_mode,
        description=description,
    )

    # Report the write to the Web Terminal once the script has actually been
    # handed to the subprocess: writes it performed are already on the machine
    # and a mid-run error does not undo them, so a failed run still reports.
    #
    # `execution_time_seconds` is the launch discriminator — only the subprocess
    # path sets it, on both its completed and its timed-out return, while every
    # "never launched" outcome comes back from execute_code's setup handler with
    # it still None. Every pre-execution gate above returns before this point.
    #
    # The mode test is the complement of the readonly gate: with the mode set
    # closed by require_known_execution_mode, the two spellings are equivalent,
    # but the complement keeps this emit correct even if the set ever grows.
    # Emitting before build_execution_response keeps the report independent of
    # artifact/notebook persistence failures.
    if (
        patterns.get("has_writes")
        and execution_mode != "readonly"
        and exec_result.execution_time_seconds is not None
    ):
        await notify_agent_activity_async(
            "execute", "channel", detail="ran a script with control-system writes"
        )

    # A write the runtime guard refused mid-run reaches here only as a
    # traceback on stderr. Report it so the layer that catches the evasive
    # spellings alerts and audits like the pre-execution ones do.
    await report_runtime_refusal(
        tool="execute",
        stderr=exec_result.stderr,
        code=code,
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
