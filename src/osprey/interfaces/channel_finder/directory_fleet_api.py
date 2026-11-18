"""channel directory fleet route: ``fleet``, the server fleet service's snapshot of the current target.

Mounted under ``/api/directory`` by :mod:`.directory_api`. The route answers the
fleet connection's :class:`~osprey.server_fleet_connection.FleetAnswer` for the
session's control target, in its own JSON shape. It asks the fleet service and
nothing else: no control-system read and no directory query. The target is the
control-context record's, else the deployment's baseline, as the directory
service resolves it. The route is absent (404) when the ``channel_directory_fleet``
predicate is false.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from osprey import server_fleet_connection
from osprey.services.channel_directory.resolve import predicates
from osprey_connectors import control_context
from osprey_connectors.types import baseline_target

router = APIRouter()


def _current_target(config: Any) -> str:
    """The session's control target: the record's, else the deployment's baseline."""
    record = control_context.read_record()
    if record is not None:
        return str(record.target)
    return str(baseline_target(config.get("control_system")))


@router.get("/fleet")
async def fleet(request: Request) -> dict[str, Any]:
    """The fleet answer of the current target: ``{outcome, target, reason, generated_at, snapshot}``."""
    state = request.app.state
    if not predicates(getattr(state, "channel_directory", None))["channel_directory_fleet"]:
        raise HTTPException(status_code=404, detail="Not Found")
    target = _current_target(state.config)
    answer = await server_fleet_connection.fetch_snapshot(state.config, target)
    return answer.to_json()
