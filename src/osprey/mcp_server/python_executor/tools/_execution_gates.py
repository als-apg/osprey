"""Shared execution-mode gates for the ``execute`` and ``execute_file`` tools.

Both tools take an ``execution_mode`` string and guard control-system writes
with two independent checks: a per-call readonly gate (pattern detection) and a
deployment-level kill switch (``control_system.writes_enabled`` in the project
config). Each gate only recognises one canonical spelling, so any *other*
string used to fall through both — not "readonly", so write patterns were not
blocked; not "readwrite", so the kill switch never fired. Rejecting unknown
modes here closes that hole for every caller at once, and gives the kill
switch a single implementation instead of one copy per tool.
"""

from __future__ import annotations

import logging

from osprey.mcp_server.errors import make_error

logger = logging.getLogger("osprey.mcp_server.tools.execution_gates")

#: The closed set of recognised execution modes. Downstream gates may test
#: equality against either member only because this set is enforced first.
VALID_EXECUTION_MODES = frozenset({"readonly", "readwrite"})


def require_known_execution_mode(execution_mode: str) -> None:
    """Raise ``ToolError`` (validation_error) unless the mode is recognised.

    Must run before any write gate: the gates branch on string equality, and
    an unrecognised value would otherwise satisfy neither branch and execute
    with no write protection at all.
    """
    if execution_mode in VALID_EXECUTION_MODES:
        return
    make_error(
        "validation_error",
        f"Unknown execution_mode {execution_mode!r}.",
        ['Use "readonly" (default) to block control-system writes, or "readwrite" to allow them.'],
    )


def enforce_deployment_writes_gate(execution_mode: str) -> None:
    """Raise ``ToolError`` (safety_error) on readwrite runs in a no-writes deployment.

    Fires whenever the caller asks for write mode, regardless of whether the
    pattern detector recognises specific write syntax — the deployment-level
    kill switch must not depend on detection accuracy.
    """
    if execution_mode != "readwrite":
        return

    try:
        from osprey.services.python_executor.execution.control import (
            get_execution_control_config,
        )

        exec_control_config = get_execution_control_config()
    except ImportError:
        logger.warning(
            "Execution control config unavailable — skipping deployment-level writes check"
        )
        return

    if (
        exec_control_config is not None
        and exec_control_config.control_system_writes_enabled is False
    ):
        make_error(
            "safety_error",
            "Control-system writes are disabled in this deployment "
            "(control_system.writes_enabled=false in project config).",
            [
                "Set control_system.writes_enabled=true in the project config to enable writes.",
            ],
        )
