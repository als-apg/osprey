"""MCP tool: execute — run user-provided Python code with safety checks."""

import logging

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async
from osprey.mcp_server.python_executor.server import mcp
from osprey.mcp_server.python_executor.tools._execution_gates import run_gated_execution
from osprey.mcp_server.python_executor.tools._package_inventory import with_live_packages

logger = logging.getLogger("osprey.mcp_server.tools.execute")


@mcp.tool()
@with_live_packages
async def execute(
    code: str,
    description: str,
    execution_mode: str = "readonly",
    save_output: bool = True,
    approved_journal_sha256: str | None = None,
    approved_target: str | None = None,
) -> str:
    """Execute Python code with process isolation, limits enforcement, and timeout.

    Code runs in a subprocess, in the project's own Python environment when the
    project has one (otherwise OSPREY's own interpreter).
    <<AVAILABLE_PACKAGES>>

    Safety layers applied before execution:
      1. ``quick_safety_check()`` — blocks exec/eval/__import__/subprocess
      2. ``path_policy_issues()`` — in *every* mode, blocks literal writes into
         the render zone, the profile sources and the audit ledger, and code
         that names the sandbox guard's own internals
      3. ``check_readonly_imports()`` — readonly runs may not import a
         control-system client library at all; read through ``osprey.runtime``
      4. ``detect_control_system_operations()`` — blocks detected write
         spellings in readonly mode
    Safety layers applied during execution:
      5. Readonly guard — in readonly runs every control-system write entry
         point refuses, as do the routes that reach one without a client
         import (starting a process, ``ctypes``); the connectors refuse
         ``write_channel``, and the EPICS connector stays on the read_only
         gateway (``OSPREY_EXECUTION_MODE``)
      6. Raw-put block — in readwrite runs a direct client-library put
         (``epics.caput``, a caproto ``PV.write``, a Tango
         ``write_attribute`` …) is refused with ``RAW_CLIENT_WRITE``; writes go
         through ``osprey.runtime.write_channel`` / ``write_channels``, which
         carry limits checking and the audit record. PVAccess puts are the
         exception until the connector writes PVAccess: they pass the block
         and are limits-checked instead
      7. Process isolation — code runs outside the MCP server process
      8. Execution timeout — kills execution after configured timeout

    A refused write is reported to the operator and recorded in the
    deployment's audit log, at whichever layer catches it. A readonly script
    therefore cannot shell out or load a shared library at all, even for
    unrelated work — resubmit as readwrite, which requires human approval:
    the operator answers the terminal permission prompt, which links a
    pre-execution notebook of the code in the artifact gallery for review.

    A ``save_artifact(obj, title, description)`` helper is available in the
    subprocess for saving objects to the artifact gallery.

    Args:
        code: Python source code to execute.
        description: Human-readable description of what the code does.
        execution_mode: "readonly" (default) refuses every control-system
                        write; "readwrite" permits writes through
                        ``osprey.runtime.write_channel`` / ``write_channels``
                        after human approval, and still refuses raw client
                        puts. Any other value is rejected.
        save_output: If True, save the code and output to a workspace data file.
        approved_journal_sha256: Set by the approval hook; a value you pass is
                                 overwritten.
        approved_target: Set by the approval hook; a value you pass is
                         overwritten.

    Returns:
        JSON with a compact summary (truncated stdout/stderr) and a data file path.
    """
    if not code or not code.strip():
        return make_error(
            "validation_error",
            "No code provided.",
            ["Provide Python code to execute."],
        )

    # Every gate, the launch and the run's reporting, in the one order both
    # executor tools share — see :func:`run_gated_execution`. The notify is
    # passed in so the write-activity report goes through this module's seam.
    exec_result, patterns = await run_gated_execution(
        tool="execute",
        code=code,
        description=description,
        execution_mode=execution_mode,
        notify=notify_agent_activity_async,
        approved_journal_sha256=approved_journal_sha256,
        approved_target=approved_target,
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
