"""Shared execution-mode gates for the ``execute`` and ``execute_file`` tools.

Both tools take an ``execution_mode`` string and guard control-system writes
with two independent checks: a per-call readonly gate (pattern detection) and a
deployment-level kill switch (``control_system.writes_enabled`` in the project
config). Each gate only recognises one canonical spelling, so any *other*
string falls through both — not "readonly", so write patterns are not blocked;
not "readwrite", so the kill switch never fires. Rejecting unknown modes here
closes that hole for every caller at once, and gives the kill switch a single
implementation instead of one copy per tool.

This module also owns what happens *around* a refusal, which is three things
the tools must not each spell for themselves: the durable audit record, the
operator alert, and the error handed back to the agent. Keeping them together
is what makes "blocked" mean the same thing at every layer — a write stopped by
the import denylist before launch and one stopped by the runtime guard mid-run
produce the same audit record and the same alert, differing only in the
``layer`` field.
"""

from __future__ import annotations

import logging
from typing import Any, NoReturn

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async

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


async def record_and_alert_refusal(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None = None,
    execution_mode: str = "readonly",
) -> None:
    """Write the audit record and alert the operator for one refused write.

    Both halves of the issue's "the operator should see it, and it should be
    auditable" requirement, in the order that matters: the durable record is
    written first, so a Web Terminal that is not running (CLI-only mode, where
    the alert is a no-op) still leaves the refusal on disk.

    Never raises. The recorder swallows its own errors and
    ``notify_agent_activity_async`` is fire-and-forget by contract, so a
    refusal is never turned into a traceback by the act of reporting it.
    """
    try:
        from osprey.services.python_executor.refusal_audit import record_refusal

        record_refusal(
            layer=layer,
            trigger=trigger,
            source=code,
            description=description,
            tool=tool,
            execution_mode=execution_mode,
        )
    except Exception:
        # An import failure here must not mask the refusal itself.
        logger.warning("Could not record the refusal for audit", exc_info=True)

    await notify_agent_activity_async(
        tool,
        "channel",
        detail=f"BLOCKED a control-system write in {execution_mode} mode ({layer})",
    )


async def refuse_readonly_write(
    *,
    tool: str,
    layer: str,
    trigger: Any,
    code: str,
    description: str | None,
    message: str,
    suggestions: list[str],
) -> NoReturn:
    """Record, alert, then raise the ``safety_error`` the agent sees.

    The raise is last on purpose: :func:`make_error` raises rather than
    returning (it is the only path fastmcp turns into a clean error on the
    wire), so anything that has to happen for a refused write has to happen
    before it.
    """
    await record_and_alert_refusal(
        tool=tool,
        layer=layer,
        trigger=trigger,
        code=code,
        description=description,
    )
    make_error("safety_error", message, suggestions)


async def report_runtime_refusal(
    *,
    tool: str,
    stderr: str,
    code: str,
    description: str | None,
) -> bool:
    """Report a write the *runtime* guard refused, mid-run. Returns whether it did.

    The static layers refuse by raising, so they control their own reporting.
    The runtime ones cannot: they run inside the subprocess, and all that
    reaches here is a traceback on stderr. Without this, the layers that catch
    the evasive spellings — aliased imports, ``getattr``, ``importlib``, a
    shelled-out ``caput`` — would be the only ones that never alerted the
    operator or left an audit record.

    Matching is on
    :data:`~osprey.services.python_executor.execution.wrapper.READONLY_REFUSAL_MARKER`
    rather than on the guard's full message, so the connector reference
    monitor's own refusal — a write that took the *approved* ``write_channel``
    path in a readonly run — is reported too.

    The script's own result is left alone: its stderr already names the mode
    and the way forward, and converting it into a tool error here would
    discard whatever the run legitimately produced before the refusal.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER
    from osprey.services.python_executor.refusal_audit import LAYER_RUNTIME_GUARD

    if READONLY_REFUSAL_MARKER not in (stderr or ""):
        return False

    await record_and_alert_refusal(
        tool=tool,
        layer=LAYER_RUNTIME_GUARD,
        trigger=_refusal_lines(stderr),
        code=code,
        description=description,
    )
    return True


def _refusal_lines(stderr: str) -> list[str]:
    """The stderr lines naming the refusal, for the audit record's ``trigger``.

    The whole traceback would bury the fact in noise and the bare marker would
    drop the channel name the connector's message carries; the matching lines
    keep what an auditor actually reads.
    """
    from osprey.services.python_executor.execution.wrapper import READONLY_REFUSAL_MARKER

    return [line.strip() for line in stderr.splitlines() if READONLY_REFUSAL_MARKER in line]
