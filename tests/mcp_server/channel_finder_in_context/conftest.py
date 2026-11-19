"""Fixtures for In-Context channel finder MCP tests."""

import pytest

from osprey.mcp_server.channel_finder_in_context.server_context import reset_cf_ic_context


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons and config caches between tests."""
    import osprey_connectors.config as _cfg
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    _cfg._installed = None
    _cfg._last_lazy = None
    _cfg._loads.clear()

    yield

    reset_cf_ic_context()
    reset_config_cache()
    _cfg._loads.clear()
