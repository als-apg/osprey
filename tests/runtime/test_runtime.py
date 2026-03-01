"""Unit tests for osprey.runtime module.

Tests the runtime utilities for control system operations in generated Python code.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.runtime import (
    _write_channel_async,
    cleanup_runtime,
    read_channel,
    write_channel,
    write_channels,
)
from osprey.connectors.control_system.limits_validator import (
    ChannelLimitsConfig,
    LimitsValidator,
)
from osprey.errors import ChannelLimitsViolationError


class MockConnector:
    """Mock control system connector for testing."""

    def __init__(self):
        self.write_calls = []
        self.read_calls = []
        self.disconnect_called = False

    async def write_channel(self, channel_address: str, value, **kwargs):
        """Mock write operation."""
        self.write_calls.append((channel_address, value, kwargs))
        result = MagicMock()
        result.success = True
        return result

    async def read_channel(self, channel_address: str, **kwargs):
        """Mock read operation."""
        self.read_calls.append((channel_address, kwargs))
        pv_value = MagicMock()
        pv_value.value = 42.0
        return pv_value

    async def disconnect(self):
        """Mock disconnect."""
        self.disconnect_called = True


@pytest.fixture
def clear_runtime_state():
    """Clear runtime module state before each test."""
    import osprey.runtime as runtime

    runtime._runtime_connector = None
    runtime._limits_validator = None
    yield
    # Cleanup after test
    runtime._runtime_connector = None
    runtime._limits_validator = None


def test_write_channel_success(clear_runtime_state):
    """Test write_channel with successful write."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channel("TEST:PV", 42.0)

        assert len(mock_connector.write_calls) == 1
        assert mock_connector.write_calls[0][0] == "TEST:PV"
        assert mock_connector.write_calls[0][1] == 42.0


def test_write_channel_failure(clear_runtime_state):
    """Test write_channel with failed write."""
    mock_connector = MockConnector()

    async def failing_write(channel_address, value, **kwargs):
        result = MagicMock()
        result.success = False
        result.error_message = "Write failed"
        return result

    mock_connector.write_channel = failing_write

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        with pytest.raises(RuntimeError, match="Write failed"):
            write_channel("TEST:PV", 42.0)


def test_read_channel_success(clear_runtime_state):
    """Test read_channel with successful read."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        value = read_channel("TEST:PV")

        assert len(mock_connector.read_calls) == 1
        assert mock_connector.read_calls[0][0] == "TEST:PV"
        assert value == 42.0


def test_write_channels_bulk(clear_runtime_state):
    """Test write_channels bulk operation."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channels({"PV1": 1.0, "PV2": 2.0, "PV3": 3.0})

        assert len(mock_connector.write_calls) == 3
        assert mock_connector.write_calls[0][0] == "PV1"
        assert mock_connector.write_calls[1][0] == "PV2"
        assert mock_connector.write_calls[2][0] == "PV3"


@pytest.mark.asyncio
async def test_cleanup_runtime(clear_runtime_state):
    """Test cleanup_runtime properly releases resources."""
    import osprey.runtime as runtime

    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channel("TEST:PV", 42.0)

        assert runtime._runtime_connector is not None

        await cleanup_runtime()

        assert mock_connector.disconnect_called
        assert runtime._runtime_connector is None


def test_connector_reuse(clear_runtime_state):
    """Test that connector is created once and reused."""
    mock_connector = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.return_value = mock_connector

        write_channel("TEST:PV1", 1.0)
        write_channel("TEST:PV2", 2.0)
        read_channel("TEST:PV3")

        mock_factory.assert_called_once()

        assert len(mock_connector.write_calls) == 2
        assert len(mock_connector.read_calls) == 1


@pytest.mark.asyncio
async def test_connector_recreated_after_cleanup(clear_runtime_state):
    """Test that connector is recreated after cleanup."""
    mock_connector1 = MockConnector()
    mock_connector2 = MockConnector()

    with patch(
        "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
    ) as mock_factory:
        mock_factory.side_effect = [mock_connector1, mock_connector2]

        write_channel("TEST:PV", 1.0)
        assert len(mock_connector1.write_calls) == 1

        await cleanup_runtime()

        write_channel("TEST:PV", 2.0)
        assert len(mock_connector2.write_calls) == 1

        assert mock_factory.call_count == 2


class TestRuntimeLimitsValidation:
    """Tests that _limits_validator fires before the connector is called (I-2)."""

    @pytest.mark.asyncio
    async def test_limits_violation_raises_before_connector(self, clear_runtime_state):
        """When _limits_validator rejects a value, ChannelLimitsViolationError is raised
        and _get_connector() is never called."""
        import osprey.runtime as runtime

        test_db = {
            "TEST:PV": ChannelLimitsConfig(
                channel_address="TEST:PV", min_value=0.0, max_value=100.0, writable=True
            ),
        }
        validator = LimitsValidator(test_db, {"allow_unlisted_pvs": False})
        runtime._limits_validator = validator

        with patch("osprey.runtime._get_connector", new_callable=AsyncMock) as mock_get_connector:
            with pytest.raises(ChannelLimitsViolationError) as exc_info:
                await _write_channel_async("TEST:PV", 150.0)

            mock_get_connector.assert_not_called()
            assert exc_info.value.channel_address == "TEST:PV"
            assert exc_info.value.attempted_value == 150.0

    @pytest.mark.asyncio
    async def test_no_validator_calls_connector_normally(self, clear_runtime_state):
        """When _limits_validator is None, the connector is called normally."""
        import osprey.runtime as runtime

        assert runtime._limits_validator is None

        mock_connector = MockConnector()

        with patch(
            "osprey.connectors.factory.ConnectorFactory.create_control_system_connector"
        ) as mock_factory:
            mock_factory.return_value = mock_connector

            await _write_channel_async("TEST:PV", 42.0)

            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][0] == "TEST:PV"
            assert mock_connector.write_calls[0][1] == 42.0

    @pytest.mark.asyncio
    async def test_valid_value_passes_through_to_connector(self, clear_runtime_state):
        """When _limits_validator approves the value, the connector write proceeds."""
        import osprey.runtime as runtime

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

            await _write_channel_async("TEST:PV", 50.0)

            assert len(mock_connector.write_calls) == 1
            assert mock_connector.write_calls[0][1] == 50.0
