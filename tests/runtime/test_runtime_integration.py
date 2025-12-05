"""Integration tests for osprey.runtime module.

Tests runtime utilities with actual Mock connector (and EPICS if available).
"""

import pytest

from osprey.runtime import (
    cleanup_runtime,
    configure_from_context,
    read_channel,
    write_channel,
    write_channels,
)


# Custom test config with writes enabled
@pytest.fixture(autouse=True)
def setup_registry(tmp_path):
    """Initialize registry with test config for integration tests."""
    import os

    import yaml

    from osprey.registry import initialize_registry as init_reg
    from osprey.registry import reset_registry

    # Create test config with writes enabled and no noise
    config_file = tmp_path / "config.yml"
    config_data = {
        "control_system": {
            "type": "mock",
            "writes_enabled": True,  # Enable writes for integration tests
            "connector": {
                "mock": {
                    "noise_level": 0.0,  # Disable noise for predictable tests
                    "response_delay_ms": 1,  # Minimal delay for fast tests
                }
            },
        }
    }
    config_file.write_text(yaml.dump(config_data))

    # Create minimal registry
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(
        """
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    def get_registry_config(self):
        return extend_framework_registry()
"""
    )

    # Reset and initialize with test config
    reset_registry()
    os.environ["CONFIG_FILE"] = str(config_file)
    init_reg(auto_export=False, config_path=config_file)
    yield

    # Cleanup
    reset_registry()


class MockContext:
    """Mock context object for testing."""

    def __init__(self, data: dict):
        self._data = data


@pytest.fixture
def mock_control_system_context():
    """Create context with mock control system config."""
    return MockContext(
        {
            "_execution_config": {
                "control_system": {
                    "type": "mock",
                    "writes_enabled": True,  # Enable writes for testing
                    "connector": {
                        "mock": {
                            "noise_level": 0.0,  # No noise for predictable tests
                            "response_delay_ms": 1,  # Minimal delay
                        }
                    },
                }
            }
        }
    )


@pytest.fixture
async def clear_runtime_state():
    """Clear runtime module state before and after each test."""
    import osprey.runtime as runtime

    # Clear before test
    if runtime._runtime_connector is not None:
        try:
            await cleanup_runtime()
        except:
            pass
    runtime._runtime_connector = None
    runtime._runtime_config = None

    yield

    # Cleanup after test
    try:
        if runtime._runtime_connector is not None:
            await cleanup_runtime()
    except:
        pass
    runtime._runtime_connector = None
    runtime._runtime_config = None


def test_write_read_with_mock_connector(mock_control_system_context, clear_runtime_state):
    """Test write and read operations with Mock connector."""
    # Configure runtime with mock control system
    configure_from_context(mock_control_system_context)

    # Write to a test channel
    test_channel = "TEST:VOLTAGE"
    test_value = 123.45

    # Synchronous API
    write_channel(test_channel, test_value)

    # Read back the value (synchronous API)
    read_value = read_channel(test_channel)

    # Mock connector should return the written value
    assert read_value == test_value


def test_write_channels_bulk_with_mock(mock_control_system_context, clear_runtime_state):
    """Test bulk write operation with Mock connector."""
    # Configure runtime
    configure_from_context(mock_control_system_context)

    # Write multiple channels
    test_channels = {"MAGNET:H01": 5.0, "MAGNET:H02": 5.2, "MAGNET:H03": 4.8}

    # Synchronous API
    write_channels(test_channels)

    # Read back values (synchronous API)
    for channel, expected_value in test_channels.items():
        read_value = read_channel(channel)
        assert read_value == expected_value


@pytest.mark.asyncio
async def test_runtime_cleanup_and_reconnect(mock_control_system_context, clear_runtime_state):
    """Test that runtime can cleanup and reconnect."""
    # Configure runtime
    configure_from_context(mock_control_system_context)

    # First write (synchronous API)
    write_channel("TEST:PV1", 100.0)

    # Cleanup (async function - still needs await)
    await cleanup_runtime()

    # Should be able to write again (creates new connector, synchronous API)
    write_channel("TEST:PV2", 200.0)

    # Verify value was written (synchronous API)
    value = read_channel("TEST:PV2")
    assert value == 200.0


def test_context_snapshot_reproducibility(clear_runtime_state):
    """Test that context snapshot ensures reproducible configuration."""
    import osprey.runtime as runtime

    # Create context with mock config
    epics_context = MockContext(
        {
            "_execution_config": {
                "control_system": {
                    "type": "mock",  # Using mock for testing
                    "connector": {
                        "mock": {
                            "noise_level": 0.0,  # No noise for predictable tests
                            "response_delay_ms": 1,
                        }
                    },
                }
            }
        }
    )

    # Configure from snapshot
    configure_from_context(epics_context)

    # Verify config was loaded from snapshot
    assert runtime._runtime_config is not None
    assert runtime._runtime_config["type"] == "mock"

    # Write and read should work with snapshot config (synchronous API)
    write_channel("SNAPSHOT:TEST", 999.0)
    value = read_channel("SNAPSHOT:TEST")
    assert value == 999.0


def test_error_handling_invalid_channel(mock_control_system_context, clear_runtime_state):
    """Test error handling for invalid channel operations."""
    # Configure runtime
    configure_from_context(mock_control_system_context)

    # Mock connector typically accepts any channel, but we can test the flow
    # For a real connector, this would test actual error handling

    # This should work with mock connector (accepts any channel, synchronous API)
    write_channel("ANY:CHANNEL:NAME", 42.0)
    value = read_channel("ANY:CHANNEL:NAME")
    assert value == 42.0


def test_connector_reuse_across_operations(mock_control_system_context, clear_runtime_state):
    """Test that connector is reused efficiently across operations."""
    import osprey.runtime as runtime

    # Configure runtime
    configure_from_context(mock_control_system_context)

    # Perform multiple operations (synchronous API)
    write_channel("PV1", 1.0)
    write_channel("PV2", 2.0)
    read_channel("PV1")
    read_channel("PV2")

    # Connector should still be the same instance
    connector = runtime._runtime_connector
    assert connector is not None

    # More operations should reuse same connector (synchronous API)
    write_channel("PV3", 3.0)
    assert runtime._runtime_connector is connector


def test_kwargs_passthrough(mock_control_system_context, clear_runtime_state):
    """Test that additional kwargs are passed through to connector."""
    # Configure runtime
    configure_from_context(mock_control_system_context)

    # Write with additional kwargs (mock connector accepts them, synchronous API)
    write_channel("TEST:PV", 42.0, timeout=10.0)

    # Read with additional kwargs (synchronous API)
    value = read_channel("TEST:PV", timeout=5.0)
    assert value == 42.0


@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="EPICS connector requires actual EPICS environment")
async def test_with_epics_connector(clear_runtime_state):
    """Test with EPICS connector (requires EPICS environment).

    This test is skipped by default but can be enabled in EPICS-enabled
    test environments by removing the skipif decorator.
    """
    # Create context with EPICS config
    epics_context = MockContext(
        {
            "_execution_config": {
                "control_system": {"type": "epics", "connector": {"epics": {"timeout": 5.0}}}
            }
        }
    )

    # Configure runtime
    configure_from_context(epics_context)

    # This would test actual EPICS operations
    # await write_channel("ACTUAL:EPICS:PV", 100.0)
    # value = await read_channel("ACTUAL:EPICS:PV")
    # assert value == 100.0

    pass  # Placeholder for actual EPICS tests


def test_fallback_to_global_config(clear_runtime_state):
    """Test fallback to global config when context has no snapshot."""
    from unittest.mock import patch

    import osprey.runtime as runtime

    # Create context without execution config
    empty_context = MockContext({})

    # Mock get_config_value to return mock config with writes enabled and limits checking disabled
    def mock_config_side_effect(key, default=None):
        if key == "control_system":
            return {
                "type": "mock",
                "writes_enabled": True,  # Enable writes for this test
                "connector": {
                    "mock": {
                        "noise_level": 0.0,  # No noise for predictable tests
                        "response_delay_ms": 1,
                    }
                },
            }
        elif key == "control_system.writes_enabled":
            return True  # Enable writes for this test
        elif key == "control_system.limits_checking.enabled":
            return False  # Disable limits checking for this test
        return default

    with patch("osprey.utils.config.get_config_value", side_effect=mock_config_side_effect):
        # Configure should fall back to global config
        configure_from_context(empty_context)

        # Verify fallback was used
        assert runtime._runtime_config["type"] == "mock"

        # Operations should still work (synchronous API)
        write_channel("FALLBACK:TEST", 123.0)
        value = read_channel("FALLBACK:TEST")
        assert value == 123.0
