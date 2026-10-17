"""OSPREY Python Executor MCP Server.

FastMCP server exposing the execute tools, and the measurement tools when the
render records a pyAML view.

Usage:
    python -m osprey.mcp_server.python_executor
"""

import logging
from pathlib import Path

from fastmcp import FastMCP

logger = logging.getLogger("osprey.mcp_server.python_executor")

mcp = FastMCP(
    "python",
    instructions="Execute Python code in a sandboxed environment",
)


def create_server() -> FastMCP:
    """Initialize config and import tool modules, then return the server."""
    from osprey.mcp_server.startup import (
        initialize_workspace_singletons,
        prime_config_builder,
        startup_timer,
    )
    from osprey_connectors.workspace import resolve_agent_data_root

    prime_config_builder()

    # Session working root used by other tools at call time; the artifact
    # store itself is rooted at the shared data root inside
    # initialize_workspace_singletons().
    logger.info("Workspace root: %s", resolve_agent_data_root())
    initialize_workspace_singletons()

    # Import tool modules (each registers itself via @mcp.tool())
    with startup_timer("tool_imports"):
        from osprey.mcp_server.python_executor.tools import (  # noqa: F401
            python_execute,
            python_execute_file,
        )

    from osprey.utils.config_paths import resolve_config_dir

    register_optional_tools(mcp, resolve_config_dir())

    logger.info("Python Executor MCP server initialised with all tools registered")
    return mcp


def pyaml_view_present(render_root: Path | None) -> bool:
    """Whether the render's facts record a pyAML view.

    The same predicate the build renders the agent's permissions, skill and
    prompt section by: ``measurement_models`` of ``data/facility_facts.json``
    is not empty.

    Args:
        render_root: The directory holding the rendered ``config.yml``, or
            ``None`` when there is none.
    """
    if render_root is None:
        return False
    from osprey.facility.views.facts import read_facts

    return bool(read_facts(Path(render_root), "").get("measurement_models"))


def register_optional_tools(server: FastMCP, render_root: Path | None) -> None:
    """Register the tools a render's pyAML view brings, on ``server``.

    Args:
        server: The server to register on.
        render_root: The directory holding the rendered ``config.yml``.
    """
    if not pyaml_view_present(render_root):
        return
    from osprey.mcp_server.python_executor.tools import model_response_matrix, pyaml_measure

    pyaml_measure.register(server)
    model_response_matrix.register(server)
