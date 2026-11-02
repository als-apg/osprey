"""Guards that the agent's bluesky MCP server starts without FastAPI.

The tool modules import the launch-token refusal codes, and the module that
defines them raises FastAPI's ``HTTPException`` only when it refuses a caller,
so loading the tools must leave ``fastapi`` out of ``sys.modules``.

This MUST run in a fresh subprocess: other tests in this suite import FastAPI
in-process, so an in-process check would answer for test order, not for what
the tools import. Importing ``osprey.mcp_server.bluesky.server`` alone proves
nothing either: it loads the tool modules only inside ``create_server()``, so
the child imports the same tool set that function loads.
"""

from __future__ import annotations

import json
import subprocess
import sys

#: The tool modules ``create_server()`` loads.
_TOOL_MODULES = ("authoring", "draft", "queue", "read_tools", "stop")


def test_importing_the_tools_does_not_import_fastapi() -> None:
    code = (
        "import json, sys\n"
        f"from osprey.mcp_server.bluesky.tools import {', '.join(_TOOL_MODULES)}\n"
        "print(json.dumps('fastapi' in sys.modules))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"importing the bluesky tool modules failed (child exit {result.returncode}):\n"
        f"{result.stderr}"
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded is False, (
        f"importing osprey.mcp_server.bluesky.tools ({', '.join(_TOOL_MODULES)}) loaded fastapi"
    )
