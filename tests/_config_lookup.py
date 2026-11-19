"""Config resets for suites that patch or reload ``osprey_connectors.config``.

``reset_config_caches`` is a fixture; the conftests of the suites that use it
import it, so it is requested by name there.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def reset_config_caches(monkeypatch):
    """Reset every config cache around a test that writes its own ``config.yml``.

    Earlier modules install a process config (``install_config``, or
    ``loaded_or_empty``); a test that reads its own
    ``config.yml`` (through ``monkeypatch.chdir(tmp_path)``) must not inherit, or
    leave behind, another test's cached configuration.
    """
    from osprey_connectors.testing import clear_config_state

    clear_config_state()

    import osprey_connectors.config as _cfg

    monkeypatch.setattr(_cfg, "_installed", None)
    monkeypatch.setattr(_cfg, "_last_lazy", None)
    saved_cache = _cfg._loads.copy()
    _cfg._loads.clear()

    yield

    clear_config_state()
    _cfg._loads.clear()
    _cfg._loads.update(saved_cache)
