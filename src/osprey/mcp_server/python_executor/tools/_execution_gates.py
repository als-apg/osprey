"""Shared execution-mode gates for the ``execute`` and ``execute_file`` tools.

Both tools take an ``execution_mode`` string and guard control-system writes
with two independent checks: a per-call readonly gate (pattern detection) and a
deployment-level kill switch (the write posture of the control target this
session is on). Each gate only recognises one canonical spelling, so any *other*
string falls through both — not "readonly", so write patterns are not blocked;
not "readwrite", so the kill switch never fires. Rejecting unknown modes here
closes that hole for every caller at once, and gives the kill switch a single
implementation instead of one copy per tool. A third gate clamps a run to the
*session* posture inherited from the Web Terminal, which is about this session
rather than this deployment and so refuses in its own vocabulary.

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
import os
from pathlib import Path
from typing import Any, NoReturn

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.http import notify_agent_activity_async

logger = logging.getLogger("osprey.mcp_server.tools.execution_gates")

#: The closed set of recognised execution modes. Downstream gates may test
#: equality against either member only because this set is enforced first.
VALID_EXECUTION_MODES = frozenset({"readonly", "readwrite"})

#: Environment variable carrying the *session* posture into every child of a
#: Web Terminal session, MCP servers included. Read by value, never presence —
#: see :func:`enforce_posture_clamp`.
POSTURE_ENV_VAR = "OSPREY_EXECUTION_MODE"

#: The one value of :data:`POSTURE_ENV_VAR` that clamps a session to reads.
SANDBOX_POSTURE = "readonly"


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


def session_control_target() -> str | None:
    """The control target this session is on, or ``None`` when there is none to read.

    ``None`` is not a failure: it is the honest answer for a session that never
    selected a target, and the posture lookup reads it as the deployment
    baseline — the connector an unstamped run provably builds.

    Every way of not knowing lands there too, which is why the read lives here
    and not inside :func:`enforce_deployment_writes_gate`'s import guard.
    Failing to learn the target must *narrow* the write question to the
    baseline; joining a guard whose failure path is ``return`` would drop the
    write check altogether on a state directory that happened to be unreadable.
    """
    try:
        from osprey.mcp_server.python_executor.executor import _session_target_record

        record = _session_target_record()
    except Exception:
        logger.warning(
            "Session control target unavailable — the deployment writes gate "
            "answers for the baseline target",
            exc_info=True,
        )
        return None

    if record is None:
        return None
    target = record.get("target")
    return str(target) if target else None


def enforce_deployment_writes_gate(execution_mode: str, target: str | None) -> None:
    """Raise ``ToolError`` (safety_error) on readwrite runs the target's posture does not arm.

    Fires whenever the caller asks for write mode, regardless of whether the
    pattern detector recognises specific write syntax — the deployment-level
    kill switch must not depend on detection accuracy.

    Write posture is per control target, so the question is asked about the one
    this session is on: a deployment whose baseline is a live machine can have
    writes armed on its virtual accelerator and refused on the machine, and a
    single deployment-wide answer would be wrong for one of them. ``None``
    means no target was readable, which the posture lookup answers for the
    baseline rather than by skipping.

    Args:
        execution_mode: The run's mode; only ``"readwrite"`` is gated here.
        target: The session's control target, from
            :func:`session_control_target`, or ``None`` for the baseline.
    """
    if execution_mode != "readwrite":
        return

    try:
        from osprey.services.python_executor.execution.control import (
            get_execution_control_config,
        )

        exec_control_config = get_execution_control_config(target=target)
    except ImportError:
        logger.warning(
            "Execution control config unavailable — skipping deployment-level writes check"
        )
        return

    if (
        exec_control_config is not None
        and exec_control_config.control_system_writes_enabled is False
    ):
        key = exec_control_config.writes_enabled_key
        active_target = exec_control_config.active_target
        scope = f"control target '{active_target}'" if active_target else "this deployment"
        make_error(
            "safety_error",
            f"Control-system writes are disabled for {scope} ({key}=false in project config).",
            [
                f"Set {key}=true in the project config to enable writes for {scope}.",
            ],
            details={"active_target": active_target, "writes_enabled_key": key},
        )


def enforce_posture_clamp(execution_mode: str) -> None:
    """Raise ``ToolError`` (safety_error) on readwrite runs in a sandboxed session.

    A Web Terminal session switched to the sandbox posture spawns its child
    with ``OSPREY_EXECUTION_MODE=readonly``, and every MCP server launched
    under that session inherits it. This gate is what makes the executor obey
    that posture: without it, an agent could ask for ``readwrite`` and get it,
    because the deployment kill switch above only knows about the *deployment*
    and has nothing to say about one sandboxed session.

    The test is a **value** comparison, deliberately mirroring
    ``osprey_connectors``' ``is_readonly_run``: only the exact string
    ``"readonly"`` clamps. A presence check would sandbox every session whose
    environment carries the variable for any other reason — including the
    writes posture itself, which sets it to ``"readwrite"``.
    """
    if execution_mode != "readwrite":
        return
    if os.environ.get(POSTURE_ENV_VAR) != SANDBOX_POSTURE:
        return

    make_error(
        "safety_error",
        "This terminal session is in the sandbox posture, which refuses "
        "control-system writes regardless of what the run asks for.",
        [
            'Re-run with execution_mode="readonly" — reads are unaffected by the posture.',
            "To allow writes, switch the session to the writes posture from the "
            "terminal card; the deployment config is not the gate here.",
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
    execution_mode: str = "readonly",
) -> NoReturn:
    """Record, alert, then raise the ``safety_error`` the agent sees.

    The raise is last on purpose: :func:`make_error` raises rather than
    returning (it is the only path fastmcp turns into a clean error on the
    wire), so anything that has to happen for a refused write has to happen
    before it.

    ``execution_mode`` is recorded verbatim in the audit record — layers that
    refuse in every mode (the path policy) pass the run's real mode so a
    readwrite refusal is never logged as a readonly one.
    """
    await record_and_alert_refusal(
        tool=tool,
        layer=layer,
        trigger=trigger,
        code=code,
        description=description,
        execution_mode=execution_mode,
    )
    make_error("safety_error", message, suggestions)


async def enforce_path_policy(
    *,
    tool: str,
    code: str,
    description: str | None,
    execution_mode: str,
    project_root: Path | None = None,
) -> None:
    """Refuse *code* if it statically writes into the protected set.

    Deliberately *not* under the readonly branch. The render zone, the profile
    sources and the audit ledger are off limits to executed code in every mode,
    because the boundary here is the agent rewriting the configuration it is
    itself constrained by, which the write posture has nothing to say about.
    The roots are resolved parent-side and handed in; the walker never
    re-derives them.

    Both executor tools ask this question here rather than each spelling out
    the walk, the refusal and its suggestions, so the two cannot drift on what
    the protected set is or on what an agent is told when it hits one.

    Args:
        tool: Tool name as the agent knows it, recorded in the audit record.
        code: Source to walk.
        description: Caller's description, recorded with the refusal.
        execution_mode: The run's real mode. Recorded verbatim, and it selects
            the wording of one suggestion — both modes refuse, so the readwrite
            message must not read as "you are in readonly", or the agent would
            resubmit as readwrite and hit exactly the same refusal.
        project_root: Project root when the caller has already resolved one;
            ``None`` lets the resolvers derive it.

    Raises:
        Whatever :func:`make_error` raises for ``safety_error`` — the refusal
        the agent sees.
    """
    try:
        from osprey.mcp_server.python_executor.executor import (
            resolve_permitted_roots,
            resolve_protected_roots,
        )
        from osprey.services.python_executor.execution.path_policy import path_policy_issues

        path_issues = path_policy_issues(
            code,
            protected_roots=resolve_protected_roots(project_root),
            permitted_roots=resolve_permitted_roots(project_root),
        )
    except ImportError:
        logger.warning("Path policy module unavailable — skipping protected-path check")
        path_issues = []
    if not path_issues:
        return

    from osprey.services.python_executor.refusal_audit import LAYER_PATH_POLICY

    await refuse_readonly_write(
        tool=tool,
        layer=LAYER_PATH_POLICY,
        trigger=path_issues,
        code=code,
        description=description,
        execution_mode=execution_mode,
        message=(
            "This code writes into a location the deployment protects "
            "(the render zone, the profile sources, or the audit ledger). "
            "The path policy applies in every execution mode."
        ),
        suggestions=[
            *path_issues,
            "Write analysis output under the agent data zone instead.",
            (
                "Re-running as readwrite will not lift this: the protected set is "
                "independent of execution mode."
                if execution_mode == "readonly"
                else "The write posture permits control-system writes; it does not "
                "permit edits to OSPREY's own configuration."
            ),
            "Edit the profile sources yourself and re-run 'osprey build' to change them.",
        ],
    )


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
