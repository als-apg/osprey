"""An unset ``control_system.type`` must resolve to the simulator in process, never to a network.

Both readers of the key are pinned here because they are read independently:
:meth:`ConnectorFactory.create_control_system_connector` decides which connector
the runtime actually talks to, and
:func:`get_execution_control_config` reports the deployment's control system to
the Python executor. A config that omits the key is under-specified, not a
declaration that hardware is present, so both sites fall back to the simulator
served in process and say so at WARNING level.
"""

from __future__ import annotations

import logging

import pytest

from osprey.connectors import types
from osprey.connectors.control_system.va_in_process_connector import VAInProcessConnector
from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries
from osprey.services.python_executor.execution.control import get_execution_control_config
from tests.facility.served_tree import in_process_config, served_tree

FACTORY_LOGGER = "connector_factory"
EXECUTION_CONTROL_LOGGER = "execution_control"

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
            "osprey.utils.config.get_config_value",
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


class TestExecutionControlTypeFallback:
    @pytest.mark.parametrize(
        "control_system",
        [
            # A sibling key keeps the section non-empty, so only ``type`` is missing.
            pytest.param({"writes_enabled": False}, id="missing"),
            pytest.param({"type": None}, id="blank"),
        ],
    )
    def test_unset_type_resolves_to_the_simulator(self, caplog, monkeypatch, control_system):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: control_system,
        )

        with caplog.at_level(logging.WARNING, logger=EXECUTION_CONTROL_LOGGER):
            cfg = get_execution_control_config()

        assert cfg.control_system_type == types.VIRTUAL_ACCELERATOR
        assert caplog.messages == [types.UNSET_TYPE_WARNING]

    def test_explicit_type_is_honoured_without_warning(self, caplog, monkeypatch):
        monkeypatch.setattr(
            "osprey.utils.config.get_config_value",
            lambda path, default=None, config_path=None: {"type": types.EPICS},
        )

        with caplog.at_level(logging.WARNING, logger=EXECUTION_CONTROL_LOGGER):
            cfg = get_execution_control_config()

        assert cfg.control_system_type == types.EPICS
        assert UNSET_WARNING not in caplog.text
