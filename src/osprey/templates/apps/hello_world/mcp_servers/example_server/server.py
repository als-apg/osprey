"""FastMCP server ``example_server`` — the hello-world preset's worked example.

Defines the server object and its single read-only tool, ``example_status``.
Launched over stdio by ``python -m example_server`` (see ``__main__``).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from fastmcp import FastMCP

mcp = FastMCP(
    "example_server",
    instructions=(
        "Worked example of a facility MCP server, shipped with the hello-world "
        "preset. example_status reports that the server is alive and which "
        "Python interpreter is running it. It is read-only and has no other "
        "tools; a real facility server replaces it."
    ),
)


@mcp.tool()
async def example_status() -> str:
    """Report that the example MCP server is alive and which Python runs it.

    This is the worked example that ships with the hello-world preset, so its
    job is to prove the wiring rather than to do facility work: a successful
    call means the profile's ``mcp_servers:`` entry launched this process and
    the two of you are speaking MCP. It is read-only and has no side effects,
    so call it freely whenever you want to confirm the server is reachable.

    Returns:
        A JSON object with ``server`` (the server name, always
        ``example_server``), ``utc`` (the current time as an ISO-8601 UTC
        timestamp), and ``python`` (the path of the interpreter running this
        server, useful when diagnosing which environment was launched).
    """
    return json.dumps(
        {
            "server": "example_server",
            "utc": datetime.now(UTC).isoformat(),
            "python": sys.executable,
        }
    )
