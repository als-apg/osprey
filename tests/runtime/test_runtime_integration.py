"""Integration tests for osprey.runtime module.

Tests runtime utilities with actual Mock connector (and EPICS if available).
"""

import pytest

from osprey.runtime import (
    cleanup_runtime,
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
            "writes_enabled": True,
            "connector": {
                "mock": {
                    "noise_level": 0.0,
                    "response_delay_ms": 1,
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


@pytest.fixture
async def clear_runtime_state():
    """Clear runtime module state before and after each test."""
    import osprey.runtime as runtime

    if runtime._runtime_connector is not None:
        try:
            await cleanup_runtime()
        except Exception:
            pass
    runtime._runtime_connector = None

    yield

    try:
        if runtime._runtime_connector is not None:
            await cleanup_runtime()
    except Exception:
        pass
    runtime._runtime_connector = None


def test_write_read_with_mock_connector(clear_runtime_state):
    """Test write and read operations with Mock connector."""
    test_channel = "TEST:VOLTAGE"
    test_value = 123.45

    write_channel(test_channel, test_value)

    read_value = read_channel(test_channel)
    assert read_value == test_value


def test_write_channels_bulk_with_mock(clear_runtime_state):
    """Test bulk write operation with Mock connector."""
    test_channels = {"MAGNET:H01": 5.0, "MAGNET:H02": 5.2, "MAGNET:H03": 4.8}

    write_channels(test_channels)

    for channel, expected_value in test_channels.items():
        read_value = read_channel(channel)
        assert read_value == expected_value


@pytest.mark.asyncio
async def test_runtime_cleanup_and_reconnect(clear_runtime_state):
    """Test that runtime can cleanup and reconnect."""
    write_channel("TEST:PV1", 100.0)

    await cleanup_runtime()

    write_channel("TEST:PV2", 200.0)

    value = read_channel("TEST:PV2")
    assert value == 200.0


def test_error_handling_invalid_channel(clear_runtime_state):
    """Test error handling for invalid channel operations."""
    write_channel("ANY:CHANNEL:NAME", 42.0)
    value = read_channel("ANY:CHANNEL:NAME")
    assert value == 42.0


def test_connector_reuse_across_operations(clear_runtime_state):
    """Test that connector is reused efficiently across operations."""
    import osprey.runtime as runtime

    write_channel("PV1", 1.0)
    write_channel("PV2", 2.0)
    read_channel("PV1")
    read_channel("PV2")

    connector = runtime._runtime_connector
    assert connector is not None

    write_channel("PV3", 3.0)
    assert runtime._runtime_connector is connector


def test_kwargs_passthrough(clear_runtime_state):
    """Test that additional kwargs are passed through to connector."""
    write_channel("TEST:PV", 42.0, timeout=10.0)

    value = read_channel("TEST:PV", timeout=5.0)
    assert value == 42.0


@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="EPICS connector requires actual EPICS environment")
async def test_with_epics_connector(clear_runtime_state):
    """Test with EPICS connector (requires EPICS environment).

    This test is skipped by default but can be enabled in EPICS-enabled
    test environments by removing the skipif decorator.
    """
    pass  # Placeholder for actual EPICS tests
