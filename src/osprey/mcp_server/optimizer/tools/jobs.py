"""MCP tools: read the optimizer's jobs.

==========================  ===================================================
Tool                        Optimizer route
==========================  ===================================================
list_jobs                   GET  /jobs
get_job                     GET  /jobs/{job_id}?detail=summary|full
current_job                 GET  /jobs/current
get_job_config              GET  /jobs/{job_id}/config
==========================  ===================================================
"""

import json
from typing import Literal
from urllib.parse import quote

from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import get_json, request

#: The response header carrying a stored config's revision, when the service sends one.
CONFIG_REVISION_HEADER = "X-Config-Revision"


def _job(job_id: str) -> str:
    return f"/jobs/{quote(job_id, safe='')}"


@mcp.tool()
async def list_jobs() -> str:
    """List every job that has not finished (created, submitted, running or paused), with its status. Finished jobs are read with get_job or list_results."""
    return json.dumps(await get_json("/jobs"))


@mcp.tool()
async def get_job(job_id: str, detail: Literal["summary", "full"] | None = None) -> str:
    """Read one job's status.

    Args:
        job_id: Job id, from list_jobs or current_job.
        detail: For an optimization job: ``summary`` adds ``result_summary`` (best points, initial point, iteration count); ``full`` also adds the evaluated data, variable bounds and snapshots. Omitted: the basic status only.
    """
    params = {"detail": detail} if detail is not None else None
    return json.dumps(await get_json(_job(job_id), params=params))


@mcp.tool()
async def current_job() -> str:
    """Read the job the optimizer panel currently shows: ``{"job_id": <id or null>}``."""
    return json.dumps(await get_json("/jobs/current"))


@mcp.tool()
async def get_job_config(job_id: str) -> str:
    """Read an optimization job's stored config (environment, target, variables and bounds, algorithm, objective).

    Returns ``{"config": ..., "revision": ...}``; ``revision`` is the stored config's revision, or null when the service does not report one.

    Args:
        job_id: Job id of an optimization job.
    """
    resp = await request("GET", f"{_job(job_id)}/config")
    return json.dumps({"config": resp.json(), "revision": resp.headers.get(CONFIG_REVISION_HEADER)})
