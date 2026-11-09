"""The application's tool server: runs beside the agent, reaches the backend by the panel door.

``plan_bump`` solves a bump and stores it as the caller's current plan; nothing is written.
``apply_bump`` is a write because the application's ``app.yml`` declares it under
``tools.approval``: it reaches the backend only with the plan id and setpoint digest the
approval hook bound from the approval route, and the backend applies only the current plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import httpx

from orbit_bump import APP_NAME
from osprey.apps.sdk import (
    ApprovalRefused,
    agent_panel_headers,
    check_approved_call,
    make_error,
    serve_tools,
    web_terminal_url,
)

if TYPE_CHECKING:
    from fastmcp import FastMCP

# The backend's routes, as ``orbit_bump.server`` serves them. They are repeated here so the
# tool server does not import the backend and its numerics; a test pins the two together.
STATE_ROUTE = "/api/state"
COMPUTE_ROUTE = "/api/compute"
APPLY_ROUTE = "/api/apply"

#: A panel call's bound, in seconds.
TIMEOUT_S = 30.0
#: Seconds the backend's apply may spend on its confirmed writes and the reads around them.
WRITE_BUDGET_S = 60.0
#: Seconds the backend's apply waits for the written readbacks to settle (its own bound).
SETTLE_TIMEOUT_S = 5.0
#: Seconds on top, for the re-read of the orbit and the trip through the panel door.
APPLY_MARGIN_S = 15.0
#: An apply call's bound; the panel's ``read_timeout_s`` in ``app.yml`` is not shorter.
APPLY_TIMEOUT_S = WRITE_BUDGET_S + SETTLE_TIMEOUT_S + APPLY_MARGIN_S

#: The row fields an interrupted apply reports, from the state's current plan.
_ROW_FIELDS = ("element", "new", "outcome", "readback", "settled", "reason")


class _NoAnswer(Exception):
    """A route timed out or answered a server error; ``detail`` says which."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


async def _request(
    method: str, route: str, *, timeout: float = TIMEOUT_S, **kwargs: Any
) -> dict[str, Any]:
    """Call one route through the panel door; raise ``_NoAnswer`` when it does not answer."""
    url = f"{web_terminal_url()}/panel/{APP_NAME}{route}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, headers=agent_panel_headers(), **kwargs)
    except httpx.HTTPError as exc:
        raise _NoAnswer(f"{route} could not be reached: {exc!r}") from exc
    if response.status_code >= 500:
        raise _NoAnswer(f"{route} answered {response.status_code}: {response.text}")
    if response.is_error:
        raise ApprovalRefused("backend_refused", f"{route} refused: {response.text}")
    answer: dict[str, Any] = response.json()
    return answer


async def _panel_call(
    method: str, route: str, *, timeout: float = TIMEOUT_S, **kwargs: Any
) -> dict[str, Any]:
    """Call one backend route through the panel door and answer its JSON object."""
    try:
        return await _request(method, route, timeout=timeout, **kwargs)
    except _NoAnswer as exc:
        make_error("unreachable", exc.detail)


async def _report_interrupted_apply(plan_id: str, detail: str) -> dict[str, Any]:
    """What the state says of a plan whose apply did not answer: its status and rows."""
    try:
        state = await _request("GET", STATE_ROUTE)
    except _NoAnswer as exc:
        make_error(
            "unreachable",
            f"{detail}; the state could not be read either ({exc.detail}), so what was "
            f"written of plan {plan_id} is unknown",
        )
    plan = state.get("plan") or {}
    if plan.get("id") != plan_id:
        current = plan.get("id") or "none"
        make_error(
            "unreachable",
            f"{detail}; plan {plan_id} is no longer the current plan (the current plan is "
            f"{current}), so what was written of it is unknown",
        )
    rows = [
        {field: row.get(field) for field in _ROW_FIELDS} for row in plan.get("correctors") or ()
    ]
    return {
        "plan_id": plan_id,
        "status": plan.get("status"),
        "apply_answer": detail,
        "correctors": rows,
    }


def register(server: FastMCP) -> None:
    """Register the application's tools on the server the SDK built for it."""

    @server.tool()
    async def plan_bump(
        model: str,
        plane: str,
        targets: dict[str, float],
        rule: str,
        correctors: list[str] | None = None,
        priority: Literal["closure", "targets"] = "closure",
        outside: Literal["hold", "golden"] = "hold",
    ) -> dict[str, Any]:
        """Solve a local orbit bump on the session's control target; nothing is written.

        The plan replaces the caller's current plan, and its id is what ``apply_bump`` takes.
        Correctors stay within their limits; ``missed`` lists the monitors the plan does not
        reach, ``target_miss_max`` the largest miss.

        Args:
            model: One of the offered ring models, as the panel's state names them.
            plane: ``x`` or ``y``.
            targets: Monitor element to its offset from golden, in the facility's unit.
            rule: How the correctors are picked: ``auto``, an MML signed-index rule such as
                ``mml:[-1 1]``, or ``manual``.
            correctors: The corrector elements to use, for the ``manual`` rule only.
            priority: ``closure`` keeps the orbit outside the bump; ``targets`` meets the
                targets first and lets the orbit outside leak.
            outside: ``hold`` keeps the monitors outside the bump where they are; ``golden``
                steers them to golden with every corrector of the plane (rule ignored).
        """
        body: dict[str, Any] = {
            "model": model,
            "plane": plane,
            "targets": targets,
            "rule": rule,
            "priority": priority,
            "outside": outside,
        }
        if correctors is not None:
            body["correctors"] = correctors
        return await _panel_call("POST", COMPUTE_ROUTE, json=body)

    @server.tool()
    async def apply_bump(
        plan_id: str,
        approved_target: str | None = None,
        approved_plan_id: str | None = None,
        approved_setpoints: str | None = None,
    ) -> dict[str, Any]:
        """Apply the caller's current bump plan to the session's control target.

        The operator approves the plan the panel holds; a plan id other than that one is
        refused without a write.

        Args:
            plan_id: The id ``plan_bump`` answered, or the panel's current plan id.
            approved_target: Filled in by the approval hook; leave unset.
            approved_plan_id: Filled in by the approval hook; leave unset.
            approved_setpoints: Filled in by the approval hook; leave unset.
        """
        bound = check_approved_call(
            "apply_bump",
            {
                "plan_id": plan_id,
                "approved_target": approved_target,
                "approved_plan_id": approved_plan_id,
                "approved_setpoints": approved_setpoints,
            },
        )
        fields = bound.state_fields
        if (
            not bound.bound
            or "approved_plan_id" not in fields
            or "approved_setpoints" not in fields
        ):
            raise ApprovalRefused(
                "approval_missing", "not run: the approval bound no plan and setpoints to apply."
            )
        approved = str(fields["approved_plan_id"])
        if plan_id != approved:
            raise ApprovalRefused(
                "plan_mismatch",
                f"not run: the approval is for plan {approved}, not plan {plan_id}.",
            )
        body = {"plan_id": approved, "setpoint_digest": str(fields["approved_setpoints"])}
        try:
            return await _request("POST", APPLY_ROUTE, timeout=APPLY_TIMEOUT_S, json=body)
        except _NoAnswer as exc:
            return await _report_interrupted_apply(approved, exc.detail)


if __name__ == "__main__":
    serve_tools(register)
