"""MCP tools: read the optimizer's stored results and saved models.

==========================  ===================================================
Tool                        Optimizer route
==========================  ===================================================
list_results                GET  /results
get_result                  GET  /results/{env}/{timestamp}
list_models                 GET  /models
==========================  ===================================================
"""

import json
from urllib.parse import quote

from osprey.mcp_server.errors import make_error
from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import request


def _result_path(run_id: str) -> str:
    """``/results/<env>/<timestamp>`` for a run id of exactly those two segments."""
    segments = run_id.split("/")
    if len(segments) != 2 or any(s in ("", ".", "..") for s in segments):
        make_error(
            "validation_error",
            f"run_id must be '<environment>/<timestamp>', got {run_id!r}.",
            ["Pass a run_id exactly as list_results gives it."],
        )
    return "/results/" + "/".join(quote(s, safe="") for s in segments)


@mcp.tool()
async def list_results() -> str:
    """List the stored runs: each entry's ``run_id`` (``<environment>/<timestamp>``), environment, timestamp, objective, producing job and whether it saved a model."""
    return json.dumps(await request("GET", "/results"))


@mcp.tool()
async def get_result(run_id: str) -> str:
    """Read one stored run: its config, best point, initial point and every evaluated record.

    This can be large (every evaluated record of the run). Start from list_results, and from get_job with ``detail="summary"`` on the run's producing job for its best and initial points; read the full run only when the records themselves are needed.

    Args:
        run_id: ``<environment>/<timestamp>``, as list_results gives it.
    """
    return json.dumps(await request("GET", _result_path(run_id)))


@mcp.tool()
async def list_models() -> str:
    """List the saved surrogate model files: label, path, variables and objective of each."""
    return json.dumps(await request("GET", "/models"))
