"""Tests for EnvironmentRegistry plugin discovery and loading."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from osprey.services.optimizer.services.plugin_registry import (
    EnvironmentRegistry,
    _discovered_environments,
)
from osprey.services.optimizer.services.template_service import TemplateService, TemplateStatus


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


def test_load_environment_template_fallback() -> None:
    """When name is not a plugin, registry falls back to TemplateService."""
    registry = EnvironmentRegistry()

    mock_env = MagicMock()
    mock_env_cls = MagicMock(return_value=mock_env)

    template_config = {"override": "als_injector", "some_key": "value"}

    with (
        patch("osprey.services.optimizer.services.plugin_registry.TemplateService") as MockTS,
        patch.object(registry, "get_environment_class", return_value=mock_env_cls) as mock_get_cls,
    ):
        MockTS.return_value.load_template_config.return_value = template_config

        result = registry.load_environment("my_template", {})

    assert result is mock_env
    MockTS.return_value.load_template_config.assert_called_once_with("my_template")
    mock_get_cls.assert_called_once_with("als_injector")


def test_load_environment_invalid_template_error_propagates() -> None:
    """A template that EXISTS but is invalid surfaces its real error instead of
    falling through to plugin lookup and reporting 'not found'. Only
    FileNotFoundError (no such template) falls through."""
    registry = EnvironmentRegistry()

    with patch("osprey.services.optimizer.services.plugin_registry.TemplateService") as MockTS:
        MockTS.return_value.load_template_config.side_effect = ValueError(
            "template missing 'override'"
        )

        with pytest.raises(ValueError, match="template missing 'override'"):
            registry.load_environment("broken_template", {})


def test_list_environments_template_scan_failure_propagates() -> None:
    """A template scan error surfaces instead of quietly shrinking the list.

    Swallowing it presents to the operator as "those environments don't
    exist", which is indistinguishable from a real absence.
    """
    registry = EnvironmentRegistry()

    with patch("osprey.services.optimizer.services.plugin_registry.TemplateService") as MockTS:
        MockTS.return_value.scan_templates.side_effect = OSError("disk on fire")

        with pytest.raises(OSError, match="disk on fire"):
            registry.list_environments()


def test_list_environments_without_template_dir_returns_plugins_only(tmp_path: Path) -> None:
    """No template directory is a normal deployment (plugins-only), not a failure —
    the propagation above must not turn local dev / a template-less container into
    a hard error."""
    registry = EnvironmentRegistry()
    absent = tmp_path / "no_templates_here"

    with patch(
        "osprey.services.optimizer.services.plugin_registry.TemplateService",
        lambda: TemplateService(template_dir=absent),
    ):
        envs = registry.list_environments()

    assert envs
    assert all(e.source == "plugin" for e in envs)


def test_list_environments_skips_template_colliding_with_plugin(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A template whose name matches a registered plugin is skipped with a warning."""
    registry = EnvironmentRegistry()
    plugin_name = next(iter(_discovered_environments()))

    colliding = TemplateStatus(name=plugin_name, valid=True, content=MagicMock())

    with (
        patch("osprey.services.optimizer.services.plugin_registry.TemplateService") as MockTS,
        caplog.at_level(
            logging.WARNING, logger="osprey.services.optimizer.services.plugin_registry"
        ),
    ):
        MockTS.return_value.scan_templates.return_value = [colliding]
        envs = registry.list_environments()

    template_entries = [e for e in envs if e.source == "template"]
    assert template_entries == []
    assert "conflicts with existing plugin" in caplog.text


def test_list_environments_invalid_template_marked_invalid() -> None:
    """An invalid template surfaces as an EnvironmentInfo entry with valid=False and error set."""
    registry = EnvironmentRegistry()
    invalid = TemplateStatus(
        name="busted_template",
        valid=False,
        content=None,
        error="missing override",
    )

    with patch("osprey.services.optimizer.services.plugin_registry.TemplateService") as MockTS:
        MockTS.return_value.scan_templates.return_value = [invalid]
        envs = registry.list_environments()

    entry = next(e for e in envs if e.name == "busted_template")
    assert entry.valid is False
    assert entry.error == "missing override"
    assert entry.display_name == "busted_template (Invalid)"


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


@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [
        pytest.param(FileNotFoundError, None, id="missing_returns_none"),
        pytest.param(None, {"override": "plugin_x"}, id="found_returns_config"),
    ],
)
def test_get_template_defaults(
    side_effect: type[Exception] | None, expected: dict[str, str] | None
) -> None:
    """A missing template's FileNotFoundError is caught and yields None; a found
    template yields its config dict."""
    registry = EnvironmentRegistry()
    with patch("osprey.services.optimizer.services.plugin_registry.TemplateService") as mock_ts:
        mock_ts.return_value.load_template_config.side_effect = side_effect
        mock_ts.return_value.load_template_config.return_value = expected
        assert registry.get_template_defaults("tmpl") == expected
