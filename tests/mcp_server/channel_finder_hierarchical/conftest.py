"""Fixtures for Hierarchical channel finder MCP tests."""

import pytest

from osprey.mcp_server.channel_finder_hierarchical.server_context import reset_cf_hier_context


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons and config caches before and after every test.

    Leak guarded: the hierarchical channel-finder context is a module global with
    no owner. Resetting it only on the way out is not enough — ``osprey.interfaces
    .channel_finder.app`` calls ``initialize_cf_hier_context()``, so an interfaces
    test that lands earlier on the same xdist worker leaves the context populated
    and ``test_server_context.py::test_registry_not_initialized`` finds it already
    there.
    """
    import osprey.utils.config as _cfg
    from osprey.utils.workspace import reset_config_cache

    reset_cf_hier_context()
    reset_config_cache()
    _cfg._default_config = None
    _cfg._default_configurable = None
    _cfg._config_cache.clear()

    yield

    reset_cf_hier_context()
    reset_config_cache()
    _cfg._config_cache.clear()


def get_tool_fn(tool_or_fn):
    """Extract raw function from FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn
