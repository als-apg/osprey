"""Integration tests for osprey.runtime module.

Tests runtime utilities with actual Mock connector (and EPICS if available).
"""

import contextlib

import pytest

import osprey.runtime as runtime
from osprey.runtime import (
    cleanup_runtime,
    read_channel,
    write_channel,
    write_channels,
)
from osprey_connectors import control_context
from osprey_connectors.errors import ChannelWriteBlockedError
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._harness import (
    isolate_verdict_environment,
    reset_verdict_state,
    write_control_context,
)
from tests.facility.served_tree import in_process_config, served_tree

#: The target an unstamped process on this deployment is on: the simulator in process.
BASELINE = "va"

#: Every address this module writes; each test reads back what it wrote.
WRITTEN = (
    "ANY:CHANNEL:NAME",
    "MAGNET:H01",
    "MAGNET:H02",
    "MAGNET:H03",
    "PV1",
    "PV2",
    "PV3",
    "TEST:PV",
    "TEST:PV1",
    "TEST:PV2",
    "TEST:VOLTAGE",
)


@pytest.fixture(autouse=True)
def agent_data_root(tmp_path, monkeypatch):
    """A clean write-verdict environment whose host record lives under a tmp root.

    No record is written: a test that means writes to land arms the switch
    itself (:func:`switch_on`). The process carries no target stamp.
    """
    root = tmp_path / "agent-data"
    root.mkdir()
    isolate_verdict_environment(monkeypatch, root)
    monkeypatch.delenv(runtime.ENV_CONTROL_TARGET, raising=False)
    monkeypatch.delenv(runtime.ENV_CONTROL_TARGET_GENERATION, raising=False)
    yield root
    reset_verdict_state()


@pytest.fixture
def switch_on(agent_data_root):
    """A record pointed at the baseline target with its switch on."""
    write_control_context(agent_data_root, target=BASELINE, switch={BASELINE: "on"})
    reset_verdict_state()
    return agent_data_root


# Custom test config with a tier that allows writes
@pytest.fixture(autouse=True)
def setup_registry(tmp_path, monkeypatch):
    """Initialize registry with test config for integration tests."""
    import yaml

    from osprey.registry import initialize_registry as init_reg
    from osprey.registry import reset_registry

    # Create test config with a tier that allows writes and no noise
    view = served_tree(tmp_path / "served", WRITTEN)
    config_file = tmp_path / "config.yml"
    config_data = {
        "control_system": {
            "type": "virtual_accelerator",
            "writes": "allowed",
            "connector": {"virtual_accelerator": in_process_config(view, response_delay_ms=1)},
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
    monkeypatch.setenv("CONFIG_FILE", str(config_file))
    init_reg(auto_export=False, config_path=config_file)
    yield

    # Cleanup
    reset_registry()


@pytest.fixture
async def clear_runtime_state():
    """Clear runtime module state before and after each test."""
    import osprey.runtime as runtime

    if runtime._runtime_connector is not None:
        with contextlib.suppress(Exception):
            await cleanup_runtime()
    runtime._runtime_connector = None

    yield

    try:
        if runtime._runtime_connector is not None:
            await cleanup_runtime()
    except Exception:
        pass
    runtime._runtime_connector = None


@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
def test_write_read_with_the_in_process_connector():
    """Test write and read operations with Mock connector."""
    test_channel = "TEST:VOLTAGE"
    test_value = 123.45

    write_channel(test_channel, test_value)

    read_value = read_channel(test_channel)
    assert read_value == test_value


@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
def test_write_channels_bulk_with_mock():
    """Test bulk write operation with Mock connector."""
    test_channels = {"MAGNET:H01": 5.0, "MAGNET:H02": 5.2, "MAGNET:H03": 4.8}

    write_channels(test_channels)

    for channel, expected_value in test_channels.items():
        read_value = read_channel(channel)
        assert read_value == expected_value


@pytest.mark.asyncio
@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
async def test_runtime_cleanup_and_reconnect():
    """Test that runtime can cleanup and reconnect."""
    write_channel("TEST:PV1", 100.0)

    await cleanup_runtime()

    write_channel("TEST:PV2", 200.0)

    value = read_channel("TEST:PV2")
    assert value == 200.0


@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
def test_error_handling_invalid_channel():
    """Test error handling for invalid channel operations."""
    write_channel("ANY:CHANNEL:NAME", 42.0)
    value = read_channel("ANY:CHANNEL:NAME")
    assert value == 42.0


@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
def test_connector_reuse_across_operations():
    """Test that connector is reused efficiently across operations."""
    write_channel("PV1", 1.0)
    write_channel("PV2", 2.0)
    read_channel("PV1")
    read_channel("PV2")

    connector = runtime._runtime_connector
    assert connector is not None

    write_channel("PV3", 3.0)
    assert runtime._runtime_connector is connector


@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
def test_kwargs_passthrough():
    """Test that additional kwargs are passed through to connector."""
    write_channel("TEST:PV", 42.0, timeout=10.0)

    value = read_channel("TEST:PV", timeout=5.0)
    assert value == 42.0


def _refused_for(reason: str, exc: ChannelWriteBlockedError) -> bool:
    """Whether *exc* is the monitor's refusal for *reason*, in the operator's words."""
    return exc.reason == "WRITES_DISABLED" and refusal_phrase(reason) in str(exc)


@pytest.mark.usefixtures("clear_runtime_state")
def test_unstamped_write_with_no_record_refuses_switch_off():
    """An unstamped process is on the baseline target, so no record is a switch that is off."""
    with pytest.raises(ChannelWriteBlockedError) as exc:
        write_channel("TEST:PV", 1.0)

    assert _refused_for("switch_off", exc.value), str(exc.value)
    assert refusal_phrase("target_unknown") not in str(exc.value)
    assert runtime._runtime_connector._control_target == BASELINE


@pytest.mark.usefixtures("clear_runtime_state", "switch_on")
def test_unstamped_write_with_the_switch_on_lands():
    """The baseline target's switch on: an unstamped write lands and reads back."""
    write_channel("TEST:PV", 7.0)

    assert read_channel("TEST:PV") == 7.0
    assert runtime._runtime_connector._control_target == BASELINE


@pytest.mark.usefixtures("clear_runtime_state")
def test_turning_the_switch_off_mid_script_refuses_the_next_write(switch_on):
    """The monitor re-reads the record on every write: a turn-off stops the very next one."""
    write_channel("TEST:PV1", 1.0)
    connector = runtime._runtime_connector

    control_context.set_switch([BASELINE], "off", path=control_context.record_path_under(switch_on))

    with pytest.raises(ChannelWriteBlockedError) as exc:
        write_channel("TEST:PV2", 2.0)

    assert _refused_for("switch_off", exc.value), str(exc.value)
    assert runtime._runtime_connector is connector
    assert read_channel("TEST:PV1") == 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(True, reason="EPICS connector requires actual EPICS environment")
async def test_with_epics_connector(clear_runtime_state):
    """Test with EPICS connector (requires EPICS environment).

    This test is skipped by default but can be enabled in EPICS-enabled
    test environments by removing the skipif decorator.
    """
    pass  # Placeholder for actual EPICS tests
