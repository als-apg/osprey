"""Fixtures for Direct Channel Finder MCP tests."""

import pytest

from osprey.mcp_server.direct_channel_finder.server_context import reset_dcf_context


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons and config caches between tests."""
    import osprey.utils.config as _cfg
    from osprey.utils.workspace import reset_config_cache

    reset_config_cache()
    _cfg._default_config = None
    _cfg._default_configurable = None
    _cfg._config_cache.clear()

    yield

    reset_dcf_context()
    reset_config_cache()
    _cfg._config_cache.clear()


def get_tool_fn(tool_or_fn):
    """Extract raw function from FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn
