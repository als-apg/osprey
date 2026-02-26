"""Unit tests for osprey.runtime module.

Tests the runtime utilities for control system operations in generated Python code.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.runtime import (
    _write_channel_async,
    cleanup_runtime,
    configure_from_context,
    read_channel,
    write_channel,
    write_channels,
)
from osprey.services.python_executor.exceptions import ChannelLimitsViolationError
from osprey.services.python_executor.execution.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)


class MockContext:
    """Mock context object for testing."""

    def __init__(self, data: dict):
        self._data = data


class MockConnector:
    """Mock control system connector for testing."""

    def __init__(self):
        self.write_calls = []
        self.read_calls = []
        self.disconnect_called = False

    async def write_channel(self, channel_address: str, value, **kwargs):
        """Mock write operation."""
        self.write_calls.append((channel_address, value, kwargs))
        # Return successful result
        result = MagicMock()
        result.success = True
        return result

    async def read_channel(self, channel_address: str, **kwargs):
        """Mock read operation."""
        self.read_calls.append((channel_address, kwargs))
        # Return mock PV value
        pv_value = MagicMock()
        pv_value.value = 42.0
        return pv_value

    async def disconnect(self):
        """Mock disconnect."""
        self.disconnect_called = True


@pytest.fixture
def mock_context_with_config():
    """Create mock context with valid config."""
    return MockContext(
        {"_execution_config": {"control_system": {"type": "mock", "connector": {"mock": {}}}}}
    )


@pytest.fixture
def mock_context_without_config():
    """Create mock context without config."""
    return MockContext({})


@pytest.fixture
def clear_runtime_state():
    """Clear runtime module state before each test."""
    import osprey.runtime as runtime

    runtime._runtime_connector = None
    runtime._runtime_config = None
    runtime._limits_validator = None
    yield
    # Cleanup after test
    runtime._runtime_connector = None
    runtime._runtime_config = None
    runtime._limits_validator = None


def test_configure_from_context_with_valid_config(mock_context_with_config, clear_runtime_state):
    """Test configure_from_context with valid context config."""
    import osprey.runtime as runtime

    configure_from_context(mock_context_with_config)

    # Should have loaded config from context
    assert runtime._runtime_config is not None
    assert runtime._runtime_config["type"] == "mock"


def test_configure_from_context_without_config(mock_context_without_config, clear_runtime_state):
    """Test configure_from_context falls back to global config."""
    import osprey.runtime as runtime

    # Mock get_config_value to return mock config
    with patch("osprey.utils.config.get_config_value") as mock_get_config:
        mock_get_config.return_value = {"type": "mock"}

        configure_from_context(mock_context_without_config)

        # Should have fallen back to global config
        assert runtime._runtime_config is not None
        assert runtime._runtime_config["type"] == "mock"
        mock_get_config.assert_called_once_with("control_system", {})


def test_configure_from_none_context(clear_runtime_state):
    """Test configure_from_context with None context falls back gracefully."""
    import osprey.runtime as runtime

    # Mock get_config_value to return mock config
    with patch("osprey.utils.config.get_config_value") as mock_get_config:
        mock_get_config.return_value = {"type": "mock"}

        configure_from_context(None)

        # Should have fallen back to global config
        assert runtime._runtime_config is not None
        mock_get_config.assert_called_once()


def test_configure_error_handling(clear_runtime_state):
    """Test error handling when both context and global config fail."""

    # Mock get_config_value to raise error
    with patch("osprey.utils.config.get_config_value") as mock_get_config:
        mock_get_config.side_effect = RuntimeError("Config not available")

        # Should raise clear error
        with pytest.raises(RuntimeError, match="Failed to configure runtime"):
            configure_from_context(None)


def test_write_channel_success(mock_context_with_config, clear_runtime_state):
    """Test write_channel with successful write."""

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connector
    mock_connector = MockConnector()

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        # Write channel (synchronous API)
        write_channel("TEST:PV", 42.0)

        # Verify write was called
        assert len(mock_connector.write_calls) == 1
        assert mock_connector.write_calls[0][0] == "TEST:PV"
        assert mock_connector.write_calls[0][1] == 42.0


def test_write_channel_failure(mock_context_with_config, clear_runtime_state):
    """Test write_channel with failed write."""

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connector that fails
    mock_connector = MockConnector()

    async def failing_write(channel_address, value, **kwargs):
        result = MagicMock()
        result.success = False
        result.error_message = "Write failed"
        return result

    mock_connector.write_channel = failing_write

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        # Write should raise RuntimeError (synchronous API)
        with pytest.raises(RuntimeError, match="Write failed"):
            write_channel("TEST:PV", 42.0)


def test_read_channel_success(mock_context_with_config, clear_runtime_state):
    """Test read_channel with successful read."""

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connector
    mock_connector = MockConnector()

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        # Read channel (synchronous API)
        value = read_channel("TEST:PV")

        # Verify read was called and value returned
        assert len(mock_connector.read_calls) == 1
        assert mock_connector.read_calls[0][0] == "TEST:PV"
        assert value == 42.0


def test_write_channels_bulk(mock_context_with_config, clear_runtime_state):
    """Test write_channels bulk operation."""

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connector
    mock_connector = MockConnector()

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        # Write multiple channels (synchronous API)
        write_channels({"PV1": 1.0, "PV2": 2.0, "PV3": 3.0})

        # Verify all writes were called
        assert len(mock_connector.write_calls) == 3
        assert mock_connector.write_calls[0][0] == "PV1"
        assert mock_connector.write_calls[1][0] == "PV2"
        assert mock_connector.write_calls[2][0] == "PV3"


@pytest.mark.asyncio
async def test_cleanup_runtime(mock_context_with_config, clear_runtime_state):
    """Test cleanup_runtime properly releases resources."""
    import osprey.runtime as runtime

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connector
    mock_connector = MockConnector()

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        # Create connector by writing (synchronous API)
        write_channel("TEST:PV", 42.0)

        # Verify connector exists
        assert runtime._runtime_connector is not None

        # Cleanup (async function - still needs await)
        await cleanup_runtime()

        # Verify cleanup was called
        assert mock_connector.disconnect_called
        assert runtime._runtime_connector is None


def test_connector_reuse(mock_context_with_config, clear_runtime_state):
    """Test that connector is created once and reused."""

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connector
    mock_connector = MockConnector()

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        # Write multiple times (synchronous API)
        write_channel("TEST:PV1", 1.0)
        write_channel("TEST:PV2", 2.0)
        read_channel("TEST:PV3")

        # Factory should only be called once
        mock_factory.assert_called_once()

        # All operations should use same connector
        assert len(mock_connector.write_calls) == 2
        assert len(mock_connector.read_calls) == 1


@pytest.mark.asyncio
async def test_connector_recreated_after_cleanup(mock_context_with_config, clear_runtime_state):
    """Test that connector is recreated after cleanup."""

    # Configure runtime
    configure_from_context(mock_context_with_config)

    # Create mock connectors
    mock_connector1 = MockConnector()
    mock_connector2 = MockConnector()

    # Mock the connector factory
    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.side_effect = [mock_connector1, mock_connector2]

        # First write (synchronous API)
        write_channel("TEST:PV", 1.0)
        assert len(mock_connector1.write_calls) == 1

        # Cleanup (async function - still needs await)
        await cleanup_runtime()

        # Second write should create new connector (synchronous API)
        write_channel("TEST:PV", 2.0)
        assert len(mock_connector2.write_calls) == 1

        # Factory should be called twice
        assert mock_factory.call_count == 2


class TestRuntimeLimitsValidation:
    """Tests that _limits_validator fires before the connector is called (I-2)."""

    @pytest.mark.asyncio
    async def test_limits_violation_raises_before_connector(
        self, mock_context_with_config, clear_runtime_state
    ):
        """When _limits_validator rejects a value, ChannelLimitsViolationError is raised
        and _get_connector() is never called."""
        import osprey.runtime as runtime

        configure_from_context(mock_context_with_config)

        # Set up a LimitsValidator with known limits
        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"allow_unlisted_pvs": False})
        runtime._limits_validator = validator

        # Mock _get_connector to track whether it's called
        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as mock_get_connector:
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await _write_channel_async("TEST:PV", 150.0)

            # Connector should never have been called
            mock_get_connector.assert_not_called()

            # Verify the error details
            assert exc_info.value.channel_address == "TEST:PV"
            assert exc_info.value.attempted_value == 150.0

    @pytest.mark.asyncio
    async def test_no_validator_calls_connector_normally(
        self, mock_context_with_config, clear_runtime_state
    ):
        """When _limits_validator is None, the connector is called normally."""
        import osprey.runtime as runtime

        configure_from_context(mock_context_with_config)

        # Ensure no validator is set
        assert runtime._limits_validator is None

        mock_connector = MockConnector()

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            await _write_channel_async("TEST:PV", 42.0)

            # Connector should have been called
            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][0] == "TEST:PV"
            assert mock_connector.write_calls[0][1] == 42.0

    @pytest.mark.asyncio
    async def test_valid_value_passes_through_to_connector(
        self, mock_context_with_config, clear_runtime_state
    ):
        """When _limits_validator approves the value, the connector write proceeds."""
        import osprey.runtime as runtime

        configure_from_context(mock_context_with_config)

        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"allow_unlisted_pvs": False})
        runtime._limits_validator = validator

        mock_connector = MockConnector()

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            # Value within limits — should pass through to connector
            await _write_channel_async("TEST:PV", 50.0)

            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][1] == 50.0
