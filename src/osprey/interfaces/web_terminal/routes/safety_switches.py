"""The safety-switch routes: which safety features run, and the admin's off.

``GET /api/safety-switches`` lists this process's switchable features
(:mod:`osprey.interfaces.web_terminal.safety_switches`) for any operator, so a
person whose watchdog an admin turned off can see that it is off and who did it,
plus ``can_change`` for whether this terminal may move them.

``POST /api/safety-switches/{name}`` moves one, and only on the admin tier. The
admin tier is the render that may change the configuration the other tiers run
under, and on a running terminal that is ``web.config_panel.enabled``; the
handler calls the Config panel's own gate first, as the audit route does, so
there is one switch and one wording. Neither route is in ``PANEL_TIER_ROUTES``,
so the panel token an agent holds reaches neither: an agent can never turn off
what watches it.

Every move, refused or taken, files one audit line under its own subject, and
a move that changed something broadcasts a ``safety_switch`` frame so open tabs
re-read the list.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from osprey.interfaces.common_middleware import HTTP_MUTATION_POSTURE, HTTP_MUTATION_SURFACE
from osprey.interfaces.web_terminal.routes.config import _require_config_panel
from osprey.interfaces.web_terminal.safety_switches import switches_of
from osprey_connectors.identity import acting_identity

logger = logging.getLogger(__name__)

router = APIRouter()

#: Audit subject for an attempt to move a safety switch.
AUDIT_SUBJECT_SAFETY_SWITCH = "safety_switch_set"

#: The change notice; it carries the switch and its state, nothing else.
SAFETY_SWITCH_FRAME = "safety_switch"


class SwitchMove(BaseModel):
    """Body of ``POST /api/safety-switches/{name}``."""

    state: Literal["on", "off"]


def _record(decision: str, reason: str, detail: str) -> None:
    """File the move's audit line and claim the decision. Never raises."""
    try:
        from osprey.audit.dedup import record_and_mark
        from osprey.audit.envelope import POSTURE_SOURCE_APP

        record_and_mark(
            decision=decision,
            reason=reason,
            surface=HTTP_MUTATION_SURFACE,
            posture=HTTP_MUTATION_POSTURE,
            posture_source=POSTURE_SOURCE_APP,
            session=None,
            subject=AUDIT_SUBJECT_SAFETY_SWITCH,
            detail=detail,
        )
    except Exception:  # the audit trail degrades; the move does not
        logger.warning("Could not record a safety-switch move for audit", exc_info=True)


@router.get("/api/safety-switches")
async def get_switches(request: Request) -> dict[str, Any]:
    """This process's switchable safety features and whether this tier may move them."""
    switches = switches_of(request.app)
    return {
        "switches": switches.snapshot() if switches is not None else [],
        "can_change": bool(getattr(request.app.state, "config_panel_enabled", False)),
    }


@router.post("/api/safety-switches/{name}")
async def set_switch(name: str, body: SwitchMove, request: Request) -> dict[str, Any]:
    """Turn one safety feature off, or back to the profile's on. Admin tier only.

    403 off the admin tier, 404 for a feature this process does not run; 200
    ``{name, on, changed}`` otherwise. Async and inline throughout, so the audit
    marker is set on the task the audit middleware awaits.
    """
    from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED

    login = acting_identity()
    gesture = f"switch={name} state={body.state} by={login}"
    try:
        _require_config_panel(request)
    except HTTPException:
        _record(DECISION_REFUSED, "not_admin_tier", f"{gesture} outcome=not_admin_tier")
        raise
    switches = switches_of(request.app)
    if switches is None or name not in switches:
        _record(DECISION_REFUSED, "unknown_switch", f"{gesture} outcome=unknown_switch")
        raise HTTPException(status_code=404, detail="unknown_switch")
    on = body.state == "on"
    changed = switches.set(name, on=on, by=login)
    _record(DECISION_ALLOWED, "switch_set", f"{gesture} outcome=switch_set changed={changed}")
    if changed:
        logger.warning("Safety feature %r turned %s by %s", name, body.state, login)
        broadcaster = getattr(request.app.state, "broadcaster", None)
        if broadcaster is not None:
            try:
                broadcaster.broadcast({"type": SAFETY_SWITCH_FRAME, "name": name, "on": on})
            except Exception:  # a missed notice costs a re-read, not the move
                logger.warning("Could not push the safety-switch frame", exc_info=True)
    return {"name": name, "on": on, "changed": changed}
