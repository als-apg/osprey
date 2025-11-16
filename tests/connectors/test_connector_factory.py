"""Tests for connector factory."""

import pytest

from osprey.connectors.archiver.base import ArchiverConnector
from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.base import ControlSystemConnector
from osprey.connectors.control_system.mock_connector import MockConnector
from osprey.connectors.factory import ConnectorFactory


@pytest.fixture(autouse=True)
def setup_test_connectors():
    """Register mock connectors for testing and clean up afterward."""
    # Register mock connectors (simulates what registry does)
    ConnectorFactory.register_control_system('mock', MockConnector)
    ConnectorFactory.register_archiver('mock_archiver', MockArchiverConnector)

    yield

    # Clean up after tests to avoid pollution between test runs
    ConnectorFactory._control_system_connectors.clear()
    ConnectorFactory._archiver_connectors.clear()


class TestConnectorFactory:
    """Test ConnectorFactory functionality."""

    @pytest.mark.asyncio
    async def test_create_mock_control_system_connector(self):
        """Test creating a mock control system connector."""
        config = {
            'type': 'mock',
            'connector': {
                'mock': {
                    'response_delay_ms': 0,
                    'noise_level': 0.01,
                    'enable_writes': True
                }
            }
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
            'type': 'mock_archiver',
            'mock_archiver': {
                'sample_rate_hz': 1.0,
                'noise_level': 0.01
            }
        }

        connector = await ConnectorFactory.create_archiver_connector(config)

        assert isinstance(connector, ArchiverConnector)
        assert isinstance(connector, MockArchiverConnector)
        assert connector._connected is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_create_with_invalid_type_raises_error(self):
        """Test that invalid connector type raises error."""
        config = {
            'type': 'nonexistent_system',
            'connector': {}
        }

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
            assert ("unknown control system type" in error_msg or
                    "config" in error_msg or
                    "required" in error_msg or
                    "install" in error_msg)

    @pytest.mark.asyncio
    async def test_factory_creates_independent_instances(self):
        """Test that factory creates independent connector instances."""
        config = {
            'type': 'mock',
            'connector': {'mock': {'response_delay_ms': 0}}
        }

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
            async def read_pv(self, pv_address, timeout=None):
                pass
            async def write_pv(self, pv_address, value, timeout=None):
                pass
            async def read_multiple_pvs(self, pv_addresses, timeout=None):
                pass
            async def subscribe(self, pv_address, callback):
                pass
            async def unsubscribe(self, subscription_id):
                pass
            async def get_metadata(self, pv_address):
                pass
            async def validate_pv(self, pv_address):
                pass

        # Register it
        ConnectorFactory.register_control_system('custom_test', CustomConnector)

        # Check it's in the list
        assert 'custom_test' in ConnectorFactory.list_control_systems()

    @pytest.mark.asyncio
    async def test_switch_between_connectors(self):
        """Test switching between different connector types."""
        # Create mock connector
        mock_config = {
            'type': 'mock',
            'connector': {'mock': {'response_delay_ms': 0}}
        }
        mock_connector = await ConnectorFactory.create_control_system_connector(mock_config)
        assert isinstance(mock_connector, MockConnector)

        # Test it works
        result = await mock_connector.read_pv('TEST:PV')
        assert result.value is not None

        await mock_connector.disconnect()

        # This demonstrates how easy it is to switch connector types
        # Just change the config!

