"""Demo Facility's own MCP server: read-only machine-status lookups.

Stands in for the one thing every facility has and no framework can ship — a
tool that answers questions from a local system nobody else can reach. It is
deliberately tiny: the point of the exemplar is the container and its place in
the pipeline, not what the tool does.

Read-only by construction. A facility tool that writes to the machine belongs
behind the control-system connector, which is the single interface every write
goes through.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

STATUS_FILE = Path(os.environ.get("FACILITY_STATUS_FILE", "/data/machine-status.json"))

mcp = FastMCP("demo-facility", host="0.0.0.0", port=int(os.environ.get("PORT", "10900")))


@mcp.tool()
def machine_status() -> str:
    """Report the control room's current machine state and operating mode."""
    if not STATUS_FILE.is_file():
        return f"No machine status available ({STATUS_FILE} is not present)."
    return json.dumps(json.loads(STATUS_FILE.read_text()), indent=2)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
