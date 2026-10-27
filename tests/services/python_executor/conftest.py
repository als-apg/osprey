"""Fixtures shared by the python executor suites."""

from __future__ import annotations

import pytest


@pytest.fixture
def reset_config_caches(monkeypatch):
    """Reset every config cache around a test that writes its own ``config.yml``.

    A test that reads its own ``config.yml`` (through ``monkeypatch.chdir``) must
    not inherit, or leave behind, another test's cached configuration.
    """
    from osprey_connectors.workspace import reset_config_cache

    reset_config_cache()

    import osprey_connectors.config as _cfg

    monkeypatch.setattr(_cfg, "_default_config", None)
    monkeypatch.setattr(_cfg, "_default_configurable", None)
    saved_cache = _cfg._config_cache.copy()
    _cfg._config_cache.clear()

    yield

    reset_config_cache()
    _cfg._config_cache.clear()
    _cfg._config_cache.update(saved_cache)
