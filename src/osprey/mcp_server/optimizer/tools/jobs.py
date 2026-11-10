"""MCP tools: read the optimizer's jobs, write drafts, and stop runs.

==========================  ===================================================
Tool                        Optimizer route
==========================  ===================================================
list_jobs                   GET  /jobs
get_job                     GET  /jobs/{job_id}?detail=summary|full
current_job                 GET  /jobs/current
get_job_config              GET  /jobs/{job_id}/config
create_draft                POST /optimizations
update_draft                PUT  /jobs/{job_id}/config
pause_job                   POST /jobs/{job_id}/pause
cancel_job                  POST /jobs/{job_id}/cancel
==========================  ===================================================
"""

import json
from typing import Any, Literal
from urllib.parse import quote

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import control_target, request

#: The response header carrying a stored config's revision, when the service sends one.
CONFIG_REVISION_HEADER = "X-Config-Revision"


def _job(job_id: str) -> str:
    return f"/jobs/{quote(job_id, safe='')}"


def _for_session_target(config: dict[str, Any]) -> dict[str, Any]:
    """*config* with the session's control target, refused unsent when it names another."""
    target = control_target()
    named = config.get("control_target")
    if named is not None and named != target:
        make_error(
            "validation_error",
            f"The config names control_target {named!r}, but this session's control "
            f"target is {target!r}: a draft is always for the session's control target.",
            [
                "Leave control_target out, or name the session's.",
                "To draft for another machine, the session's control target has to be "
                "switched first: ask the operator.",
            ],
            details={"control_target": named, "session_control_target": target},
        )
    return {**config, "control_target": target}


@mcp.tool()
async def list_jobs() -> str:
    """List every job that has not finished (created, submitted, running or paused), with its status. Finished jobs are read with get_job or list_results."""
    return json.dumps(await request("GET", "/jobs"))


@mcp.tool()
async def get_job(job_id: str, detail: Literal["summary", "full"] | None = None) -> str:
    """Read one job's status.

    ``full`` can be large (every evaluated point of the run): ask for ``summary`` first, and for ``full`` only when the evaluated data itself is needed.

    Args:
        job_id: Job id, from list_jobs or current_job.
        detail: For an optimization job: ``summary`` adds ``result_summary`` (best points, initial point, iteration count); ``full`` also adds the evaluated data, variable bounds and snapshots. Omitted: the basic status only.
    """
    params = {"detail": detail} if detail is not None else None
    return json.dumps(await request("GET", _job(job_id), params=params))


@mcp.tool()
async def current_job() -> str:
    """Read the optimizer's current-job pointer, the job a start launches: ``{"job_id": <id or null>}``."""
    return json.dumps(await request("GET", "/jobs/current"))


@mcp.tool()
async def get_job_config(job_id: str) -> str:
    """Read an optimization job's stored config (environment, target, variables and bounds, algorithm, objective).

    Returns ``{"config": ..., "revision": ...}``; ``revision`` is the stored config's revision, or null when the service does not report one.

    Args:
        job_id: Job id of an optimization job.
    """
    config, revision = await request("GET", f"{_job(job_id)}/config", header=CONFIG_REVISION_HEADER)
    return json.dumps({"config": config, "revision": revision})


@mcp.tool()
async def create_draft(config: dict[str, Any]) -> str:
    """Create a draft optimization job from a config. Nothing moves: a person starts it.

    Start from a preset where one fits (``list_presets``): name it in ``preset`` and set only what should differ. ``objective_name`` is always required, even with a preset and even when the environment declares a default objective. The fields that matter: ``environment_name``; ``preset``; ``variables`` (names to tune); ``variable_overrides`` (per variable: ``min``/``max``, or ``window_abs``/``window_fraction`` centred on the current setpoint, and ``step_size``); ``objective_name`` and ``objective_direction``; ``algorithm``; ``n_iterations``; ``n_initial_samples``; ``constraints`` (observable -> ``{"type", "value"}``); ``on_complete`` (``apply_best``, ``restore`` or ``hold``). The draft is for the session's control target: a config naming another is refused. The service resolves the bounds against that target's current setpoints and points its current job at the new draft.

    Returns the new job (its ``job_id``, status ``created``) and the ``control_target`` the draft is for. Read the stored draft back with get_job_config before telling the operator what they would be starting.

    Args:
        config: The optimization config, as a JSON object.
    """
    body = _for_session_target(config)
    job = await request("POST", "/optimizations", json_body=body)
    return json.dumps({**job, "control_target": body["control_target"]})


@mcp.tool()
async def update_draft(job_id: str, config: dict[str, Any]) -> str:
    """Replace a draft job's whole config. Refused once the job has started, and when the draft has expired.

    Read the current config with get_job_config, edit it, and send the whole of it back: a field left out returns to its default. The draft stays on the session's control target: ``control_target`` may be left out, and a config naming another is refused. The stored draft holds every window already resolved to a fixed ``min``/``max`` around the setpoint it was written against: to change a variable's range, edit that ``min``/``max``; to re-centre on the current setpoint, drop them and give ``window_abs`` or ``window_fraction`` instead. Both at once is refused.

    Args:
        job_id: Job id of a draft (status ``created``).
        config: The full optimization config, as a JSON object.
    """
    body = _for_session_target(config)
    return json.dumps(await request("PUT", f"{_job(job_id)}/config", json_body=body))


@mcp.tool()
async def pause_job(job_id: str) -> str:
    """Pause a running optimization job. The machine stays at the last point it evaluated; a person resumes it.

    Args:
        job_id: Job id of a running optimization job.
    """
    return json.dumps(await request("POST", f"{_job(job_id)}/pause"))


@mcp.tool()
async def cancel_job(job_id: str) -> str:
    """Cancel a job that has not finished, a draft or a run. A cancelled run leaves the machine at the last point it evaluated: nothing is restored.

    Args:
        job_id: Job id, from list_jobs or current_job.
    """
    return json.dumps(await request("POST", f"{_job(job_id)}/cancel"))
