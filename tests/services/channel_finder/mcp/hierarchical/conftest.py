"""Fixtures for Hierarchical channel finder MCP tests."""

import pytest

from osprey.services.channel_finder.mcp.hierarchical.registry import reset_cf_hier_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons and config caches between tests."""
    from osprey.mcp_server.common import reset_config_cache
    import osprey.utils.config as _cfg

    reset_config_cache()
    _cfg._default_config = None
    _cfg._default_configurable = None
    _cfg._config_cache.clear()

    yield

    reset_cf_hier_registry()
    reset_config_cache()
    _cfg._config_cache.clear()


def get_tool_fn(tool_or_fn):
    """Extract raw function from FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn
