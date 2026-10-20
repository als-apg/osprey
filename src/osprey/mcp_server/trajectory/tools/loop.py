"""MCP tools: trajectory_shot, trajectory_step, trajectory_apply.

A shot runs the facility's declared default chain and an apply writes correctors, so both are
put to a human by the approval hook, which binds the call to what the prompt
showed: the trajectory service's own target, and on an apply either the step's
digest or the put-back's values digest. Before either writes, the call is
checked against the hook's ask stamp and against the service as it stands now:
the stamp's target must be the service's target, and the approved digest must
be the one the call carries. A call that fails any check writes nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osprey.mcp_server.approval_stamp import (
    APPROVAL_MISSING,
    ApprovalRefused,
    approved_values_sha256,
    check_asked_call,
)
from osprey.mcp_server.errors import make_error
from osprey.mcp_server.trajectory import client
from osprey.mcp_server.trajectory.server import mcp


def _osprey_config() -> Mapping[str, Any]:
    from osprey_connectors.workspace import load_osprey_config

    return load_osprey_config()


def _check_approved(tool: str, approved: Mapping[str, str | None], target: str) -> bool:
    """Refuse unless the approval hook's stamp admits this call on *target*.

    The stamp is checked by :func:`~osprey.mcp_server.approval_stamp.check_asked_call`;
    a call it admits must also have been approved for *target*.

    Returns:
        Whether the call was put to a human, so its digest is checked too.
    """
    try:
        if not check_asked_call(_osprey_config(), tool, approved):
            return False
    except ApprovalRefused as refusal:
        make_error(refusal.code, str(refusal))
    approved_target = approved.get("approved_target")
    if approved_target != target:
        make_error(
            APPROVAL_MISSING,
            f"not run: this call was approved for target `{approved_target}`; the trajectory "
            f"service drives `{target}` ({APPROVAL_MISSING}). Call the tool again.",
        )
    return True


@mcp.tool()
async def trajectory_shot(
    n_shots: int = 1,
    average: str = "mean",
    approved_target: str | None = None,
) -> dict[str, Any]:
    """Take a trajectory shot and compute the correction step from it.

    Runs the facility's declared default chain (arm, inject, every command
    step ticked by default such as bunch cleaning, readout) once per
    injection, *n_shots* times, averages the first-turn BPM readings, and
    leaves a pending step (read it with ``trajectory_step``). The chain itself
    is the facility's; this call chooses only the shots and the averaging. The
    call waits for the shot to finish, however long the service's budgets
    allow.

    Args:
        n_shots: How many shots to take and average (at least 1).
        average: ``"mean"`` or ``"nanmean"``.
        approved_target: Set by the approval hook; a value you pass is
            overwritten.

    Returns:
        The shot result: x/y/sum per BPM, the lost BPMs, the shots included
        and failed, and the commands fired.
    """
    snapshot = await client.get_state()
    _check_approved(
        "trajectory_shot", {"approved_target": approved_target}, str(snapshot.get("target"))
    )
    return await client.write("/shot", {"n_shots": n_shots, "average": average}, snapshot)


@mcp.tool()
async def trajectory_step() -> dict[str, Any]:
    """Read the pending correction step.

    Returns:
        The step: its ``digest`` (pass it to ``trajectory_apply``), the
        corrector ``deltas``, the ``factor`` they were scaled by, the
        ``reason``/``status`` of the solve, the zeroed and unsolved correctors,
        and the shot and reference it was computed from. Refused with
        ``no_step`` when no step is pending.
    """
    return await client.get_json("/step")


@mcp.tool()
async def trajectory_apply(
    step_digest: str | None = None,
    model: str | None = None,
    values: list[dict[str, Any]] | None = None,
    approved_target: str | None = None,
    approved_step_digest: str | None = None,
    approved_values_sha256: str | None = None,
) -> dict[str, Any]:
    """Write the pending step, or put recorded values back.

    Pass exactly one of: ``step_digest`` (the digest ``trajectory_step``
    reported) to write that step, or ``model`` and ``values`` to put the
    ``before`` values a previous apply reported back.

    Args:
        step_digest: The digest of the step to write.
        model: The service's model, for a put-back.
        values: For a put-back, ``[{"name": ..., "before": ...}, ...]``.
        approved_target: Set by the approval hook; a value you pass is
            overwritten.
        approved_step_digest: Set by the approval hook; a value you pass is
            overwritten.
        approved_values_sha256: Set by the approval hook; a value you pass is
            overwritten.

    Returns:
        ``moved``: every name written, its value before, and the outcome.
    """
    put_back = model is not None or values is not None
    if put_back == (step_digest is not None) or (put_back and (not model or not values)):
        make_error(
            "bad_request",
            "pass exactly one of `step_digest`, or `model` and `values` for a put-back",
        )
    snapshot = await client.get_state()
    target = str(snapshot.get("target"))
    approved = {
        "approved_target": approved_target,
        "approved_step_digest": approved_step_digest,
        "approved_values_sha256": approved_values_sha256,
    }
    if _check_approved("trajectory_apply", approved, target):
        if put_back:
            _check_put_back(target, str(model), values or [], approved_values_sha256)
        else:
            _check_step(step_digest, approved_step_digest)
    payload: dict[str, Any] = (
        {"model": model, "values": values} if put_back else {"step_digest": step_digest}
    )
    return await client.write("/apply", payload, snapshot)


def _check_step(step_digest: str | None, approved_step_digest: str | None) -> None:
    """Refuse a step apply whose digest is not the one the prompt showed."""
    if approved_step_digest != step_digest:
        make_error(
            APPROVAL_MISSING,
            "not run: the step digest differs from the one the approval prompt showed "
            f"({APPROVAL_MISSING}). Read the step again and call the tool again.",
        )


def _check_put_back(
    target: str, model: str, values: list[dict[str, Any]], approved_sha: str | None
) -> None:
    """Refuse a put-back whose values are not the ones the prompt showed."""
    try:
        sha = approved_values_sha256(target, model, values)
    except (KeyError, ValueError, TypeError) as exc:
        make_error("bad_request", f"the put-back values cannot be digested: {exc}")
    if approved_sha != sha:
        make_error(
            APPROVAL_MISSING,
            "not run: the put-back differs from the one the approval prompt showed "
            f"({APPROVAL_MISSING}). Call the tool again.",
        )
