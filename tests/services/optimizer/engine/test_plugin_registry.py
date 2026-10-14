"""Tests for EnvironmentRegistry plugin discovery and loading."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from osprey.services.optimizer.services.plugin_registry import (
    EnvironmentRegistry,
    _discovered_environments,
)


def test_real_entry_point_discovery_loads_a_plugin() -> None:
    """Integration: real setuptools entry-point discovery + plugin load.

    Depends on the test environment having at least one registered plugin
    (via the `osprey.optimizer.environments` entry-point group) installed; the
    suite's conftest registers its fakes there.
    """
    registry = EnvironmentRegistry()

    registry.discover_environments()

    plugins = registry.list_environments()
    assert len(plugins) > 0

    plugin_name = plugins[0] if isinstance(plugins[0], str) else plugins[0].name
    env_class = registry.get_environment_class(plugin_name)
    assert env_class is not None

    env = registry.load_environment(plugin_name, {})
    assert env is not None


def test_reset_clears_state() -> None:
    """Verify EnvironmentRegistry.reset() drops the cached entry-point scan."""
    EnvironmentRegistry().discover_environments()
    assert _discovered_environments.cache_info().currsize == 1

    EnvironmentRegistry.reset()

    assert _discovered_environments.cache_info().currsize == 0


def test_load_environment_unknown_raises_keyerror() -> None:
    """Unknown environment name raises KeyError with available plugin list."""
    registry = EnvironmentRegistry()

    with pytest.raises(KeyError, match="not found"):
        registry.load_environment("nonexistent_plugin_xyz", {})


def test_get_environment_class_load_failure_propagates(caplog: pytest.LogCaptureFixture) -> None:
    """If EntryPoint.load() raises, the exception is logged and re-raised."""
    registry = EnvironmentRegistry()

    fake_ep = MagicMock()
    fake_ep.load.side_effect = ImportError("plugin module busted")

    with (
        patch(
            "osprey.services.optimizer.services.plugin_registry._discovered_environments",
            return_value={"wrecked_plugin": fake_ep},
        ),
        caplog.at_level(logging.ERROR, logger="osprey.services.optimizer.services.plugin_registry"),
        pytest.raises(ImportError, match="plugin module busted"),
    ):
        registry.get_environment_class("wrecked_plugin")
    assert "Failed to load environment 'wrecked_plugin'" in caplog.text


def test_get_environment_class_rejects_non_conforming_class() -> None:
    """An entry point resolving to a class missing protocol members raises the
    TypeError the docstring promises, naming the missing attributes — instead of
    passing the class through unchecked to fail deep inside a run."""
    registry = EnvironmentRegistry()

    class NotAnEnvironment:
        pass

    fake_ep = MagicMock()
    fake_ep.load.return_value = NotAnEnvironment

    with (
        patch(
            "osprey.services.optimizer.services.plugin_registry._discovered_environments",
            return_value={"bad_plugin": fake_ep},
        ),
        pytest.raises(TypeError, match=r"does not implement EnvironmentProtocol.*get_variables"),
    ):
        registry.get_environment_class("bad_plugin")
