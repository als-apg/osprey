"""Fixtures for the Graph channel finder MCP tests.

The paradigm reuses the main-agent graph server's context, so these tests share
its module-level singleton with every other in-process graph test. Resetting it
on the way *in* as well as out is what keeps them independent: the singleton has
no owner, and any test that initialises it leaves it populated for whoever lands
next on the same xdist worker.
"""

import pytest

from osprey.mcp_server.graph.server_context import reset_server_context


@pytest.fixture(autouse=True)
def _reset_graph_context():
    """Reset the graph server context and the config caches around every test."""
    import osprey.utils.config as _cfg
    from osprey.utils.workspace import reset_config_cache

    reset_server_context()
    reset_config_cache()
    _cfg._default_config = None
    _cfg._default_configurable = None
    _cfg._config_cache.clear()

    yield

    reset_server_context()
    reset_config_cache()
    _cfg._config_cache.clear()


def get_tool_fn(tool_or_fn):
    """Extract raw function from FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn
