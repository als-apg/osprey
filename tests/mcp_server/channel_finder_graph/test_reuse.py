"""Guards on the two tools the graph channel finder does not own.

``read_cypher`` and ``get_schema`` belong to the main-agent graph server. The
channel-finder paradigm re-registers those exact callables rather than restating
them, so the read path, the query gate, the row cap and every error envelope
stay written once. The tests below are what makes that a fact rather than an
intention:

* **Identity, not equivalence.** A copied-and-edited fork would still pass a
  behavioural test while quietly drifting — a gate rule tightened on one side
  only, a suggestion list updated in one place. ``is`` fails the day a fork
  appears.
* **Registration on this server.** Importing the module must actually put the
  tool on the channel-finder instance; a re-registration that never runs is
  invisible until an agent asks for a tool that is not there.
* **No context as a side effect of import.** The tool-vocabulary walk imports
  every module in the package to read its registrations. If importing one of
  these reached the store, that walk — and every other importer — would need a
  live graph.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from osprey.mcp_server.channel_finder_graph import server as cf_graph_server
from osprey.mcp_server.channel_finder_graph.tools import get_schema as cf_get_schema
from osprey.mcp_server.channel_finder_graph.tools import read_cypher as cf_read_cypher
from osprey.mcp_server.graph.tools import get_schema as graph_get_schema
from osprey.mcp_server.graph.tools import read_cypher as graph_read_cypher


def _tool_names(mcp) -> set[str]:
    """Tool names registered on a FastMCP server, across FastMCP versions."""
    tools = mcp.get_tools() if hasattr(mcp, "get_tools") else mcp.list_tools()
    if asyncio.iscoroutine(tools):
        tools = asyncio.run(tools)
    if isinstance(tools, dict):
        return set(tools)
    return {t.name for t in tools}


@pytest.mark.unit
def test_read_cypher_is_the_graph_servers_callable():
    """Not a copy, not a wrapper — the same function object."""
    assert cf_read_cypher.read_cypher is graph_read_cypher.read_cypher


@pytest.mark.unit
def test_get_schema_is_the_graph_servers_callable():
    """Not a copy, not a wrapper — the same function object."""
    assert cf_get_schema.get_schema is graph_get_schema.get_schema


@pytest.mark.unit
def test_both_tools_land_on_the_channel_finder_server():
    """Importing the modules registers the tools on this paradigm's own instance."""
    registered = _tool_names(cf_graph_server.mcp)

    assert {"read_cypher", "get_schema"} <= registered


@pytest.mark.unit
def test_importing_the_tool_modules_does_not_initialise_the_graph_context():
    """Import is inert: the context singleton stays unset until ``create_server()``.

    Run in a fresh interpreter because the assertion is about what an import
    does on the way in, and by the time this test runs the modules are long
    since in ``sys.modules``.
    """
    src = Path(cf_graph_server.__file__).parents[3]
    env = {**os.environ, "PYTHONPATH": str(src) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    probe = (
        "import osprey.mcp_server.channel_finder_graph.tools.read_cypher\n"
        "import osprey.mcp_server.channel_finder_graph.tools.get_schema\n"
        "import osprey.mcp_server.graph.server_context as ctx\n"
        "assert ctx._context is None, 'importing the tool modules initialised the context'\n"
        "try:\n"
        "    ctx.get_server_context()\n"
        "except RuntimeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('a GraphContext existed after a bare import')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
