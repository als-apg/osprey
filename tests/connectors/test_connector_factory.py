"""Tests for connector factory."""

from unittest.mock import patch

import pytest

from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.connectors.factory import ConnectorFactory, isolated_connector_registries


@pytest.fixture(autouse=True)
def setup_test_connectors():
    """Register mock connectors as the only registrations each test sees.

    Snapshot/restore brackets the clear so registrations made elsewhere in the
    process survive this module's teardown.
    """
    with isolated_connector_registries(clear=True):
        # Register mock connectors (simulates what registry does)
        ConnectorFactory.register_control_system("mock", MockConnector)
        ConnectorFactory.register_archiver("mock_archiver", MockArchiverConnector)

        yield


class TestConnectorFactory:
    """Test ConnectorFactory functionality."""

    @pytest.mark.asyncio
    async def test_create_mock_control_system_connector(self):
        """Test creating a mock control system connector."""
        config = {
            "type": "mock",
            "connector": {"mock": {"response_delay_ms": 0, "noise_level": 0.01}},
        }

        connector = await ConnectorFactory.create_control_system_connector(config)

        assert isinstance(connector, ControlSystemConnector)
        assert isinstance(connector, MockConnector)
        assert connector._connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_create_mock_archiver_connector(self):
        """Test creating a mock archiver connector."""
        config = {
            "type": "mock_archiver",
            "mock_archiver": {"sample_rate_hz": 1.0, "noise_level": 0.01},
        }

        connector = await ConnectorFactory.create_archiver_connector(config)

        assert isinstance(connector, ArchiverConnector)
        assert isinstance(connector, MockArchiverConnector)
        assert connector._connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_create_with_invalid_type_raises_error(self):
        """Test that invalid connector type raises error."""
        config = {"type": "nonexistent_system", "connector": {}}

        with pytest.raises(ValueError, match="Unknown control system type"):
            await ConnectorFactory.create_control_system_connector(config)

    @pytest.mark.asyncio
    async def test_create_with_no_config_uses_defaults(self):
        """Test that factory works with no config provided."""
        # This should not raise an error
        # It will try to load from global config or use defaults
        try:
            connector = await ConnectorFactory.create_control_system_connector(None)
            assert connector is not None
            await connector.disconnect()
        except (ValueError, ImportError, ConnectionError) as e:
            # If config loading fails or dependencies are missing, that's OK for this test
            # We're just checking it gives a reasonable error
            error_msg = str(e).lower()
            assert (
                "unknown control system type" in error_msg
                or "config" in error_msg
                or "required" in error_msg
                or "install" in error_msg
            )

    @pytest.mark.asyncio
    async def test_factory_creates_independent_instances(self):
        """Test that factory creates independent connector instances."""
        config = {"type": "mock", "connector": {"mock": {"response_delay_ms": 0}}}

        connector1 = await ConnectorFactory.create_control_system_connector(config)
        connector2 = await ConnectorFactory.create_control_system_connector(config)

        # Should be different instances
        assert connector1 is not connector2

        # Disconnecting one should not affect the other
        await connector1.disconnect()
        assert connector1._connected is False
        assert connector2._connected is True

        await connector2.disconnect()

    def test_register_custom_connector(self):
        """Test registering a custom connector."""

        # Create a dummy connector class
        class CustomConnector(ControlSystemConnector):
            async def connect(self, config):
                pass

            async def disconnect(self):
                pass

            async def read_channel(self, channel_address, timeout=None):
                pass

            async def write_channel(
                self,
                channel_address,
                value,
                timeout=None,
                verification_level="callback",
                tolerance=None,
            ):
                pass

            async def read_multiple_channels(self, channel_addresses, timeout=None):
                pass

            async def subscribe(self, channel_address, callback):
                pass

            async def unsubscribe(self, subscription_id):
                pass

            async def get_metadata(self, channel_address):
                pass

            async def validate_channel(self, channel_address):
                pass

        # Register it
        ConnectorFactory.register_control_system("custom_test", CustomConnector)

        # Check it's in the list
        assert "custom_test" in ConnectorFactory.list_control_systems()

    @pytest.mark.asyncio
    async def test_switch_between_connectors(self):
        """Test switching between different connector types."""
        from unittest.mock import patch

        # Mock config access since test runs without config.yml
        with patch("osprey.utils.config.get_config_value") as mock_config_value:
            # Return True for writes_enabled, None for others
            def config_side_effect(key, default=None):
                if key == "control_system.writes_enabled":
                    return True
                return default

            mock_config_value.side_effect = config_side_effect

            # Create mock connector
            mock_config = {"type": "mock", "connector": {"mock": {"response_delay_ms": 0}}}
            mock_connector = await ConnectorFactory.create_control_system_connector(mock_config)
            assert isinstance(mock_connector, MockConnector)

            # Test it works
            result = await mock_connector.read_channel("TEST:PV")
            assert result.value is not None

            await mock_connector.disconnect()

            # This demonstrates how easy it is to switch connector types
            # Just change the config!


class TestBuiltinArchiverRegistration:
    """``register_builtin_connectors()`` heals a partially-populated registry.

    The function short-circuits when every name in ``_BUILTIN_ARCHIVERS`` is
    already registered. ``mongodb_archiver`` used to be appended outside that
    tuple, so a registry holding the other archivers satisfied the early return
    and the MongoDB entry was never added — leaving a project configured for
    ``mongodb_archiver`` unable to build its connector at all. These pin the
    tuple as the complete list.
    """

    def test_mongodb_archiver_is_registered_from_a_partial_registry(self):
        """A registry holding the other archivers still gains mongodb_archiver."""
        from osprey.connectors import types
        from osprey.connectors.archiver.doocs_archiver_connector import DOOCSArchiverConnector
        from osprey.connectors.archiver.epics_archiver_connector import EPICSArchiverConnector
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector
        from osprey.connectors.control_system.epics_connector import EPICSConnector
        from osprey.connectors.control_system.va_connector import VirtualAcceleratorConnector
        from osprey.connectors.factory import register_builtin_connectors

        # The exact state that used to short-circuit: every built-in present
        # except the MongoDB archiver. The autouse fixture already registered
        # the two mock names.
        ConnectorFactory.register_control_system(types.EPICS, EPICSConnector)
        ConnectorFactory.register_control_system(
            types.VIRTUAL_ACCELERATOR, VirtualAcceleratorConnector
        )
        ConnectorFactory.register_control_system(types.DOOCS, DOOCSConnector)
        ConnectorFactory.register_archiver(types.EPICS_ARCHIVER, EPICSArchiverConnector)
        ConnectorFactory.register_archiver(types.DOOCS_ARCHIVER, DOOCSArchiverConnector)
        assert types.MONGODB_ARCHIVER not in ConnectorFactory._archiver_connectors

        register_builtin_connectors()

        assert types.MONGODB_ARCHIVER in ConnectorFactory._archiver_connectors

    def test_registration_does_not_require_pymongo(self):
        """The MongoDB connector registers with pymongo absent from the process.

        It defers the pymongo import to ``connect()``, so registration must not
        depend on the driver being installed — the same rationale that lets the
        DOOCS connectors register without doocs4py.
        """
        import builtins

        from osprey.connectors import types
        from osprey.connectors.factory import register_builtin_connectors

        real_import = builtins.__import__

        def no_pymongo(name, *args, **kwargs):
            if name == "pymongo" or name.startswith("pymongo."):
                raise ImportError("No module named 'pymongo'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_pymongo):
            register_builtin_connectors()

        assert types.MONGODB_ARCHIVER in ConnectorFactory._archiver_connectors

    def test_factory_builtins_and_framework_registry_agree_on_archivers(self):
        """The factory's built-in archiver names match the framework registry's.

        Two independent lists name the shipped archivers; a name in one and not
        the other means a connector the registry advertises but the factory
        cannot construct, or the reverse.
        """
        from osprey.connectors.factory import _BUILTIN_ARCHIVERS
        from osprey.registry.builtins import FrameworkRegistryProvider

        registry_archivers = {
            reg.name
            for reg in FrameworkRegistryProvider().get_registry_config().connectors
            if reg.connector_type == "archiver"
        }
        assert set(_BUILTIN_ARCHIVERS) == registry_archivers
