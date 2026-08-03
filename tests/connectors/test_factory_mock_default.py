"""An unset ``control_system.type`` must resolve to mock, never to live EPICS.

Both readers of the key are pinned here because they are read independently:
:meth:`ConnectorFactory.create_control_system_connector` decides which connector
the runtime actually talks to, and
:func:`get_execution_control_config` reports the deployment's control system to
the Python executor. A config that omits the key is under-specified, not a
declaration that hardware is present, so both sites fall back to mock and say so
at WARNING level.
"""

from __future__ import annotations

import logging

import pytest

from osprey.connectors import types
from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries
from osprey.services.python_executor.execution.control import get_execution_control_config

FACTORY_LOGGER = "connector_factory"
EXECUTION_CONTROL_LOGGER = "execution_control"


class _LiveConnectorTripwire:
    """Registered as EPICS so a regressed fallback fails loudly instead of dialing CA."""

    def __init__(self) -> None:
        raise AssertionError(
            "control_system.type fallback selected the live EPICS connector; "
            "an unset key must resolve to mock"
        )


@pytest.fixture(autouse=True)
def registered_connectors():
    """Mock plus an EPICS tripwire, so 'not epics' is proven, not assumed."""
    with isolated_connector_registries(clear=True):
        ConnectorFactory.register_control_system(types.MOCK, MockConnector)
        ConnectorFactory.register_control_system(types.EPICS, _LiveConnectorTripwire)
        yield


class TestFactoryTypeFallback:
    @pytest.mark.asyncio
    async def test_empty_config_creates_mock_connector(self, caplog):
        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector({})

        assert isinstance(connector, MockConnector)
        assert "control_system.type" in caplog.text
        assert types.MOCK in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_blank_type_is_treated_as_unset(self, caplog):
        # A commented-out or emptied YAML value parses as None; it must take the
        # same fail-closed path as a missing key rather than raising later.
        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector({"type": None})

        assert isinstance(connector, MockConnector)
        assert "control_system.type" in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_config_none_loads_global_config_and_falls_back(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {},
        )

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector(None)

        assert isinstance(connector, MockConnector)
        assert "control_system.type" in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_explicit_type_is_honoured_without_warning(self, caplog):
        config = {"type": types.MOCK, "connector": {types.MOCK: {"response_delay_ms": 0}}}

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, MockConnector)
        assert "control_system.type is not set" not in caplog.text

        await connector.disconnect()


class TestExecutionControlTypeFallback:
    def test_missing_type_resolves_to_mock(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {"writes_enabled": False},
        )

        with caplog.at_level(logging.WARNING, logger=EXECUTION_CONTROL_LOGGER):
            cfg = get_execution_control_config()

        assert cfg.control_system_type == types.MOCK
        assert "control_system.type" in caplog.text
        assert types.MOCK in caplog.text

    def test_blank_type_resolves_to_mock(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {"type": None},
        )

        with caplog.at_level(logging.WARNING, logger=EXECUTION_CONTROL_LOGGER):
            cfg = get_execution_control_config()

        assert cfg.control_system_type == types.MOCK
        assert "control_system.type" in caplog.text

    def test_explicit_type_is_honoured_without_warning(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {"type": types.EPICS},
        )

        with caplog.at_level(logging.WARNING, logger=EXECUTION_CONTROL_LOGGER):
            cfg = get_execution_control_config()

        assert cfg.control_system_type == types.EPICS
        assert "control_system.type is not set" not in caplog.text
