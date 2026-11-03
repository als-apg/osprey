"""Fixtures for Middle Layer channel finder MCP tests."""

import pytest

from osprey.mcp_server.channel_finder_middle_layer.server_context import reset_cf_ml_context


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons and config caches between tests."""
    import osprey_connectors.config as _cfg
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()
    _cfg._default_config = None
    _cfg._default_configurable = None
    _cfg._config_cache.clear()

    yield

    reset_cf_ml_context()
    reset_config_cache()
    _cfg._config_cache.clear()


@pytest.fixture
def cf_ml_context(tmp_path, monkeypatch):
    """Run from ``tmp_path`` with an empty ``config.yml`` and an initialized context."""
    from osprey.mcp_server.channel_finder_middle_layer.server_context import (
        initialize_cf_ml_context,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()
