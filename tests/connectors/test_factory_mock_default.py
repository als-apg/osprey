"""An unset ``control_system.type`` must resolve to the simulator in process, never to a network.

:meth:`ConnectorFactory.create_control_system_connector` decides which connector
the runtime actually talks to. A config that omits the key is under-specified,
not a declaration that hardware is present, so the factory falls back to the
simulator served in process and says so at WARNING level.

The executor's own reader of the key is gone together with its config class;
``TestFactoryTypeFallback`` is the one reader left and covers the same fallback.
"""

from __future__ import annotations

import logging

import pytest

from osprey_connectors import types
from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors_tests._harness import isolated_connector_registries
from tests.facility.served_tree import in_process_config, served_tree

FACTORY_LOGGER = "connector_factory"

# One phrase for both the "warns" and the "does not warn" checks, so rewording the
# warning breaks the positives instead of letting the negatives pass vacuously.
UNSET_WARNING = "control_system.type is not set"

# A commented-out or emptied YAML value parses as None; it must take the same
# fail-closed path as a missing key rather than raising later.
UNSET_TYPE_CONFIGS = [
    pytest.param({}, id="missing"),
    pytest.param({"type": None}, id="blank"),
]


class _NetworkConnectorTripwire:
    """Registered for every type that dials, so a regressed fallback fails loudly."""

    def __init__(self) -> None:
        raise AssertionError(
            "control_system.type fallback selected a connector that dials a network; "
            "an unset key must resolve to the simulator in process"
        )


@pytest.fixture(autouse=True)
def registered_connectors():
    """Tripwires for EPICS and the served simulator, so 'no network' is proven, not assumed."""
    with isolated_connector_registries(clear=True):
        ConnectorFactory.register_control_system(types.EPICS, _NetworkConnectorTripwire)
        ConnectorFactory.register_control_system(
            types.VIRTUAL_ACCELERATOR, _NetworkConnectorTripwire
        )
        yield


class TestFactoryTypeFallback:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("config", UNSET_TYPE_CONFIGS)
    async def test_unset_type_creates_the_in_process_connector(self, caplog, config, tmp_path):
        config = {
            **config,
            "connector": {types.VIRTUAL_ACCELERATOR: in_process_config(served_tree(tmp_path))},
        }

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, VAInProcessConnector)
        assert connector.transport == types.TRANSPORT_IN_PROCESS
        assert caplog.messages == [types.UNSET_TYPE_WARNING]

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_config_none_loads_global_config_and_falls_back(
        self, caplog, monkeypatch, tmp_path
    ):
        section = {
            "connector": {types.VIRTUAL_ACCELERATOR: in_process_config(served_tree(tmp_path))}
        }
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            lambda path, default=None, config_path=None: section,
        )

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector(None)

        assert isinstance(connector, VAInProcessConnector)
        assert UNSET_WARNING in caplog.text

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_explicit_type_is_honoured_without_warning(self, caplog, tmp_path):
        view = served_tree(tmp_path)
        config = {
            "type": types.VIRTUAL_ACCELERATOR,
            "connector": {types.VIRTUAL_ACCELERATOR: in_process_config(view, response_delay_ms=0)},
        }

        with caplog.at_level(logging.WARNING, logger=FACTORY_LOGGER):
            connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, VAInProcessConnector)
        assert UNSET_WARNING not in caplog.text

        await connector.disconnect()
