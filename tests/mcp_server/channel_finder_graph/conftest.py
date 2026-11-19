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
    import osprey_connectors.config as _cfg
    from osprey_connectors.testing import clear_config_state

    reset_server_context()
    clear_config_state()
    _cfg._installed = None
    _cfg._last_lazy = None
    _cfg._loads.clear()

    yield

    reset_server_context()
    clear_config_state()
    _cfg._loads.clear()
