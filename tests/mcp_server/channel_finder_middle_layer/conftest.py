"""Fixtures for Middle Layer channel finder MCP tests."""

import pytest

from osprey.mcp_server.channel_finder_middle_layer.server_context import reset_cf_ml_context


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset registry singletons and config caches between tests."""
    import osprey_connectors.config as _cfg
    from osprey_connectors.testing import clear_config_state

    clear_config_state()
    _cfg._installed = None
    _cfg._last_lazy = None
    _cfg._loads.clear()

    yield

    reset_cf_ml_context()
    clear_config_state()
    _cfg._loads.clear()


@pytest.fixture
def cf_ml_context(tmp_path, monkeypatch):
    """Run from ``tmp_path`` with an empty ``config.yml`` and an initialized context."""
    from osprey.mcp_server.channel_finder_middle_layer.server_context import (
        initialize_cf_ml_context,
    )

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text("{}")
    initialize_cf_ml_context()
