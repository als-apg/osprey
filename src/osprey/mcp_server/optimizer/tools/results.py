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

from osprey.mcp_server.optimizer.server import mcp
from osprey.mcp_server.optimizer.server_context import get_json


@mcp.tool()
async def list_results() -> str:
    """List the stored runs: each entry's ``run_id`` (``<environment>/<timestamp>``), environment, timestamp, objective, producing job and whether it saved a model."""
    return json.dumps(await get_json("/results"))


@mcp.tool()
async def get_result(run_id: str) -> str:
    """Read one stored run: its config, best point, initial point and every evaluated record.

    Args:
        run_id: ``<environment>/<timestamp>``, as list_results gives it.
    """
    return json.dumps(await get_json(f"/results/{quote(run_id, safe='/')}"))


@mcp.tool()
async def list_models() -> str:
    """List the saved surrogate model files: label, path, variables and objective of each."""
    return json.dumps(await get_json("/models"))
