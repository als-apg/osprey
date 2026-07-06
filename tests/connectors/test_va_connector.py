"""Tests for the Virtual Accelerator control system connector.

Covers the config-selectable seam introduced by the ``virtual_accelerator``
connector type: factory resolution, registry resolution, and that the
``connector.virtual_accelerator`` config block (not ``connector.epics``) is
what gets passed to ``connect()``.
"""

import pytest

from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.connectors.control_system.va_connector import VirtualAcceleratorConnector
from osprey.connectors.factory import ConnectorFactory, register_builtin_connectors
from osprey.connectors.types import VIRTUAL_ACCELERATOR
from osprey.registry.base import ConnectorRegistration, RegistryConfig
from osprey.registry.initializers import initialize_connectors


@pytest.fixture(autouse=True)
def clean_connector_factory():
    """Isolate ConnectorFactory global state across tests."""
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._archiver_connectors.clear()
    yield
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._archiver_connectors.clear()


class TestVirtualAcceleratorConnectorClass:
    """Basic shape checks for the connector class itself."""

    def test_is_thin_subclass_of_epics_connector(self):
        assert issubclass(VirtualAcceleratorConnector, EPICSConnector)


class TestFactoryResolution:
    """The ConnectorFactory (used by register_builtin_connectors) resolves the type."""

    def test_register_builtin_connectors_registers_virtual_accelerator(self):
        register_builtin_connectors()

        assert VIRTUAL_ACCELERATOR in ConnectorFactory.list_control_systems()
        assert (
            ConnectorFactory._control_system_connectors[VIRTUAL_ACCELERATOR]
            is VirtualAcceleratorConnector
        )

    @pytest.mark.asyncio
    async def test_factory_creates_virtual_accelerator_connector(self):
        register_builtin_connectors()

        config = {"type": VIRTUAL_ACCELERATOR, "connector": {"virtual_accelerator": {}}}
        connector = await ConnectorFactory.create_control_system_connector(config)

        try:
            assert isinstance(connector, VirtualAcceleratorConnector)
            assert isinstance(connector, EPICSConnector)
            assert connector._connected is True
        finally:
            await connector.disconnect()


class TestRegistryResolution:
    """The registry initialization path (registry/builtins.py) resolves the type."""

    def test_initialize_connectors_registers_virtual_accelerator(self):
        config = RegistryConfig(
            connectors=[
                ConnectorRegistration(
                    name=VIRTUAL_ACCELERATOR,
                    connector_type="control_system",
                    module_path="osprey.connectors.control_system.va_connector",
                    class_name="VirtualAcceleratorConnector",
                    description="Virtual Accelerator connector for PyAT-backed soft-IOC simulations",
                ),
            ]
        )
        registries = {"connectors": {}}

        initialize_connectors(config=config, registries=registries, excluded_provider_names=[])

        assert registries["connectors"][VIRTUAL_ACCELERATOR] is VirtualAcceleratorConnector
        # Registry initialization also registers with ConnectorFactory as a side effect.
        assert (
            ConnectorFactory._control_system_connectors[VIRTUAL_ACCELERATOR]
            is VirtualAcceleratorConnector
        )

    def test_framework_registry_provider_includes_virtual_accelerator(self):
        from osprey.registry.builtins import FrameworkRegistryProvider

        config = FrameworkRegistryProvider().get_registry_config()
        va_registrations = [c for c in config.connectors if c.name == VIRTUAL_ACCELERATOR]

        assert len(va_registrations) == 1
        registration = va_registrations[0]
        assert registration.connector_type == "control_system"
        assert registration.class_name == "VirtualAcceleratorConnector"
        assert registration.module_path == "osprey.connectors.control_system.va_connector"


class TestConfigBlockRouting:
    """The config.connector.virtual_accelerator block (not .epics) reaches connect()."""

    @pytest.mark.asyncio
    async def test_virtual_accelerator_config_block_passed_to_connect(self, monkeypatch):
        register_builtin_connectors()

        captured: dict = {}

        async def fake_connect(self, config):
            captured["config"] = config
            self._connected = True

        monkeypatch.setattr(VirtualAcceleratorConnector, "connect", fake_connect)

        va_only_config = {"timeout": 1.23}
        epics_config = {"timeout": 9.99}
        config = {
            "type": VIRTUAL_ACCELERATOR,
            "connector": {
                "epics": epics_config,
                "virtual_accelerator": va_only_config,
            },
        }

        connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, VirtualAcceleratorConnector)
        assert captured["config"] == va_only_config
        assert captured["config"] != epics_config
