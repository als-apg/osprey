"""The application's tool server: runs beside the agent, reaches the backend by the panel door.

The tools give the agent what the panel gives the operator. ``bump_state`` reads what the
panel shows; ``plan_bump`` solves a bump and stores it as the caller's current plan;
``discard_bump`` drops that plan; ``recompute_orm`` rebuilds a model's response matrix.
None of them writes to the machine. ``apply_bump`` is a write because the application's
``app.yml`` declares it under ``tools.approval``: it reaches the backend only with the plan
id and setpoint digest the approval hook bound from the approval route, and the backend
applies only the current plan.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
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
APPROVAL_ROUTE = "/api/approval"
RECOMPUTE_ROUTE = "/api/recompute"
DISCARD_ROUTE = "/api/discard"
COMPUTE_ROUTE = "/api/compute"
APPLY_ROUTE = "/api/apply"

#: The bump shapes the panel offers, by the name its shape toggle sends.
SHAPES = ("flat", "peak", "cos")

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
    if not response.content:  # a 204, or the approval route with no ready plan
        return {}
    answer: dict[str, Any] = response.json() or {}
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


def shape_weights(count: int, shape: str) -> list[float]:
    """The weight of each of ``count`` BPMs in ring order, as the panel's shape toggle spreads it.

    ``flat`` is 1 everywhere; ``peak`` a triangle and ``cos`` a raised cosine, both 1 at the
    centre and 0 at the two ends. Under three BPMs every shape is flat.
    """
    if count < 3 or shape == "flat":
        return [1.0] * count
    weights = []
    for at in range(count):
        t = at / (count - 1)
        weights.append(
            1 - abs(2 * t - 1) if shape == "peak" else 0.5 - 0.5 * math.cos(2 * math.pi * t)
        )
    return weights


def _model_entry(state: Mapping[str, Any], model: str) -> Mapping[str, Any]:
    """The state's entry for ``model``; ``ValueError`` naming the offered ones."""
    models = state.get("models") or []
    for entry in models:
        if entry.get("name") == model:
            return entry
    offered = ", ".join(entry.get("name", "?") for entry in models) or "none"
    raise ValueError(f"model {model} is not offered; the offered models: {offered}")


def _members(entry: Mapping[str, Any], kind: str, plane: str) -> list[Mapping[str, Any]]:
    """The model's ``kind`` (``bpms`` or ``correctors``) of ``plane``, in ring order."""
    return [m for m in (entry.get("members") or {}).get(kind) or () if m.get("plane") == plane]


def shaped_targets(
    state: Mapping[str, Any], model: str, plane: str, bpms: list[str], amplitude: float, shape: str
) -> dict[str, float]:
    """The targets the panel's amplitude and shape make over ``bpms``, spread in ring order."""
    if shape not in SHAPES:
        raise ValueError(f"shape {shape} is not one of {', '.join(SHAPES)}")
    order = [m["element"] for m in _members(_model_entry(state, model), "bpms", plane)]
    unknown = [element for element in bpms if element not in order]
    if unknown:
        raise ValueError(f"not a BPM of plane {plane} of {model}: {', '.join(unknown)}")
    ordered = sorted(set(bpms), key=order.index)
    weights = shape_weights(len(ordered), shape)
    return {e: round(amplitude * w, 6) + 0.0 for e, w in zip(ordered, weights, strict=True)}


def draft_matches_plan(
    draft: Mapping[str, Any] | None, plan: Mapping[str, Any] | None
) -> bool | None:
    """Whether the panel's draft is what the plan was computed from; ``None`` without both."""
    if not draft or not plan:
        return None
    for key, default in (
        ("plane", None),
        ("model", None),
        ("rule", "auto"),
        ("priority", "closure"),
        ("others", "hold"),
        ("all_correctors", False),
        ("sigma_keep", None),
    ):
        if draft.get(key, default) != plan.get(key, default):
            return False
    targets, planned = draft.get("targets") or {}, plan.get("targets") or {}
    if set(targets) != set(planned) or any(abs(targets[e] - planned[e]) > 1e-6 for e in targets):
        return False
    if draft.get("rule") == "manual":
        rows = sorted(row.get("element") for row in plan.get("correctors") or ())
        return sorted(draft.get("correctors") or ()) == rows
    return True


#: Plan fields kept per BPM; :func:`digest` moves them into the BPM rows.
_PER_BPM = ("orbit_at_compute", "predicted", "miss", "residual_after")


def digest(
    state: Mapping[str, Any],
    approval: Mapping[str, Any] | None,
    *,
    model: str | None = None,
    plane: str | None = None,
    bpms: list[str] | None = None,
) -> dict[str, Any]:
    """What the panel shows, for one model and plane: per-BPM orbit, golden, target and
    prediction, the corrector ticks, the ORM, the posture, the plan, the operator's draft,
    and why Apply is held.
    """
    plan = state.get("plan") or None
    models = state.get("models") or []
    # Default to what the panel shows: the operator's draft, else the plan, else its start.
    draft = state.get("draft") or {}
    model = (
        model
        or draft.get("model")
        or (plan or {}).get("model")
        or (models[0]["name"] if models else None)
    )
    plane = plane or draft.get("plane") or (plan or {}).get("plane") or "x"
    if plane not in ("x", "y"):
        raise ValueError(f"plane {plane} is not x or y")
    answer: dict[str, Any] = {
        "models": [{"name": m.get("name"), "orm": m.get("orm")} for m in models],
        "model": model,
        "plane": plane,
    }
    posture = state.get("posture") or {}
    answer["writes"] = {
        "allowed": bool(posture.get("allowed")),
        "why_not": None
        if posture.get("allowed")
        else posture.get("phrase") or posture.get("reason"),
    }
    on_view = plan is not None and plan.get("model") == model and plan.get("plane") == plane
    if model is not None:
        entry = _model_entry(state, model)
        position = (state.get("orbit") or {}).get("position") or {}
        golden = (state.get("orbit") or {}).get("golden") or {}
        members = _members(entry, "bpms", plane)
        if bpms is not None:
            known = {m["element"] for m in members}
            unknown = [element for element in bpms if element not in known]
            if unknown:
                raise ValueError(f"not a BPM of plane {plane} of {model}: {', '.join(unknown)}")
            members = [m for m in members if m["element"] in set(bpms)]
        rows = []
        for m in members:
            orbit = position.get(m.get("position_addr"))
            gold = golden.get(m.get("golden_addr")) if m.get("golden_addr") else None
            row: dict[str, Any] = {
                "element": m["element"],
                "s_m": m.get("s_m"),
                "orbit": orbit,
                "golden": gold,
                "from_golden": None if orbit is None or gold is None else orbit - gold,
            }
            if on_view:
                row["target"] = (plan.get("targets") or {}).get(m["element"])
                for key in _PER_BPM:
                    row[key] = (plan.get(key) or {}).get(m["element"])
            rows.append(row)
        answer["unit"] = members[0].get("unit") if members else None
        answer["bpms"] = rows
        unsolved = set(((entry.get("orm") or {}).get("unsolved")) or ())
        answer["correctors"] = [
            {"element": m["element"], "s_m": m.get("s_m"), "solved": m["element"] not in unsolved}
            for m in _members(entry, "correctors", plane)
        ]
    answer["plan"] = None if plan is None else {k: v for k, v in plan.items() if k not in _PER_BPM}
    answer["draft"] = state.get("draft")
    answer["draft_matches_plan"] = draft_matches_plan(state.get("draft"), plan)
    held: list[str] = []
    if plan is None:
        held.append("no plan: compute one")
    elif plan.get("status") != "ready":
        reason = f": {plan['reason']}" if plan.get("reason") else ""
        held.append(f"plan {plan.get('id')} is {plan.get('status')}{reason}")
    elif not approval:
        held.append(f"plan {plan.get('id')} does not fit the approval card")
    if not posture.get("allowed"):
        held.append(answer["writes"]["why_not"] or "writes are refused")
    answer["apply"] = {
        "possible": not held,
        "held_by": held,
        "approval_lines": (approval or {}).get("display_lines"),
    }
    return answer


def register(server: FastMCP) -> None:
    """Register the application's tools on the server the SDK built for it."""

    @server.tool()
    async def bump_state(
        model: str | None = None, plane: str | None = None, bpms: list[str] | None = None
    ) -> dict[str, Any]:
        """Read what the orbit bump panel shows; nothing is written.

        Per BPM: orbit, golden, offset from golden and, when the current plan is on this
        model and plane, its target, prediction, miss and residual after apply. Also the
        models and their response-matrix status and age, the plane's correctors, the write
        posture, the current plan whole, and what holds Apply with the approval card's lines.
        ``draft`` is what the operator has set in the panel, computed or not, and
        ``draft_matches_plan`` whether the current plan was computed from it.

        Args:
            model: The model to show; default the panel's, else the plan's, else the first.
            plane: ``x`` or ``y``; default the panel's, else the plan's, else ``x``.
            bpms: Show only these BPM elements (the panel's zoom); default every BPM.
        """
        state = await _panel_call("GET", STATE_ROUTE)
        approval = await _panel_call("GET", APPROVAL_ROUTE)
        return digest(state, approval, model=model, plane=plane, bpms=bpms)

    @server.tool()
    async def plan_bump(
        model: str,
        plane: str,
        targets: dict[str, float] | None = None,
        rule: str = "auto",
        correctors: list[str] | None = None,
        priority: Literal["closure", "targets"] = "closure",
        others: Literal["free", "hold", "zero"] = "hold",
        all_correctors: bool = False,
        sigma_keep: int | None = None,
        bpms: list[str] | None = None,
        amplitude: float | None = None,
        shape: Literal["flat", "peak", "cos"] = "flat",
    ) -> dict[str, Any]:
        """Solve a local orbit bump on the session's control target; nothing is written.

        The plan replaces the caller's current plan, which the panel then shows, and its
        id is what ``apply_bump`` takes. Give ``targets``, or ``bpms`` with ``amplitude``
        and ``shape``, not both. Correctors stay within their limits; ``missed`` lists the
        monitors the plan does not reach, ``target_miss_max`` the largest miss.

        Args:
            model: One of the offered ring models, as ``bump_state`` names them.
            plane: ``x`` or ``y``.
            targets: Monitor element to its offset from golden, in the facility's unit.
            rule: How the correctors are picked: ``auto``, an MML signed-index rule
                (``mml:[-1 1]``, ``mml:[-2 -1 1 2]``, ``mml:[-3 -1 1 3]``), or ``manual``.
            correctors: The corrector elements to use, for the ``manual`` rule only.
            priority: ``closure`` holds the other BPMs first; ``targets`` meets the
                targets first and lets the other BPMs leak.
            others: Every BPM of the plane not in the bump, inside and outside the
                corrector span: ``free`` leaves them out of the solve, ``hold`` aims each
                at its orbit now, ``zero`` at its golden orbit.
            all_correctors: Use every solved corrector of the plane (rule ignored).
            sigma_keep: How many singular values of the solve to keep, at least 1 and
                clamped to the plan's ``sigma_total``; default keeps those above 1e-4 of
                the largest. Fewer gives smaller, smoother deltas and a looser fit; under
                ``targets`` closure directions go before target directions.
            bpms: The bump BPMs, for an amplitude spread in ring order.
            amplitude: The offset from golden at the bump's peak, in the facility's unit.
            shape: ``flat`` (every BPM at the amplitude), ``peak`` (a triangle) or ``cos``
                (a raised cosine); the ends of ``peak`` and ``cos`` are zero.
        """
        if (targets is None) == (bpms is None or amplitude is None):
            raise ValueError("give targets, or bpms with amplitude, not both")
        if targets is None:
            state = await _panel_call("GET", STATE_ROUTE)
            targets = shaped_targets(state, model, plane, bpms or [], amplitude or 0.0, shape)
        body: dict[str, Any] = {
            "model": model,
            "plane": plane,
            "targets": targets,
            "rule": rule,
            "priority": priority,
            "others": others,
            "all_correctors": all_correctors,
        }
        if sigma_keep is not None:
            body["sigma_keep"] = sigma_keep
        if correctors is not None:
            body["correctors"] = correctors
        return await _panel_call("POST", COMPUTE_ROUTE, json=body)

    @server.tool()
    async def discard_bump() -> dict[str, Any]:
        """Drop the caller's current plan, as the panel's Clear does; nothing is written.

        A plan being applied cannot be dropped.
        """
        await _panel_call("POST", DISCARD_ROUTE, json={})
        return {"discarded": True}

    @server.tool()
    async def recompute_orm(model: str) -> dict[str, Any]:
        """Rebuild a model's orbit response matrix from the model; nothing is measured or written.

        The build runs in the background; ``bump_state`` shows its status. A new matrix
        discards every plan on that model not yet applied.

        Args:
            model: One of the offered ring models.
        """
        return await _panel_call("POST", RECOMPUTE_ROUTE, json={"model": model})

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
