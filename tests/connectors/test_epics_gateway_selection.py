"""Gateway selection for the EPICS connector.

EPICS Channel Access uses one process-wide CA context, so the connector points
at a single gateway. A read-only gateway rejects writes, so a deployment that
arms writes for this connector's type must route through the write-capable
gateway. Selection is defense-in-depth: the write_access gateway is used only
when the type is armed *and* it is configured; otherwise the connector stays on
the read_only gateway, so a type left unarmed also has its writes rejected at
the network layer.

Posture is per connector type: ``control_system.connector.<type>.writes_enabled``
arms one type, and a type the deployment says nothing about keeps the
deployment-wide ``control_system.writes_enabled``. The type is the one the
factory stamped onto the connector; an unstamped connector has none, so the
deployment-wide key is the whole posture it can read.
"""

import os

import pytest

import osprey.connectors.control_system.epics_connector as epics_connector_module
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.connectors.control_system.va_connector import VirtualAcceleratorConnector

EPICS_VARS = [
    "EPICS_CA_ADDR_LIST",
    "EPICS_CA_SERVER_PORT",
    "EPICS_CA_NAME_SERVERS",
    "EPICS_CA_AUTO_ADDR_LIST",
]


@pytest.fixture
def clean_epics_env(monkeypatch):
    """Snapshot EPICS_* env vars so connect()'s direct os.environ writes are restored."""
    for var in EPICS_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


def _both_gateways():
    return {
        "read_only": {"address": "ro.example.com", "port": 5064},
        "write_access": {"address": "wr.example.com", "port": 5084},
    }


def _patch_control_system(monkeypatch, section: dict):
    """Serve one ``control_system:`` section to both posture reads.

    The connector reads the whole section when it knows its type and the
    deployment-wide dotted key when it does not.
    """

    def fake_get_config_value(key, default=None):
        if key == "control_system":
            return section
        if key == "control_system.writes_enabled":
            return section.get("writes_enabled", default)
        return default  # limits_checking.enabled etc. fall back to default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


def _patch_writes_enabled(monkeypatch, enabled: bool):
    _patch_control_system(monkeypatch, {"writes_enabled": enabled})


@pytest.mark.asyncio
async def test_write_access_gateway_used_when_writes_enabled(monkeypatch, clean_epics_env):
    """writes_enabled + write_access configured -> CA context points at write gateway."""
    _patch_writes_enabled(monkeypatch, True)

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5084"


@pytest.mark.asyncio
async def test_read_only_gateway_used_when_writes_disabled(monkeypatch, clean_epics_env):
    """writes disabled -> stay on read_only gateway even if write_access is configured."""
    _patch_writes_enabled(monkeypatch, False)

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"


@pytest.mark.asyncio
async def test_warns_when_writes_enabled_but_no_write_gateway(monkeypatch, clean_epics_env):
    """writes_enabled but only read_only configured -> use read_only and warn."""
    _patch_writes_enabled(monkeypatch, True)

    warnings: list[str] = []
    monkeypatch.setattr(
        epics_connector_module.logger,
        "warning",
        lambda msg, *a, **k: warnings.append(str(msg)),
    )

    connector = EPICSConnector()
    await connector.connect(
        {"gateways": {"read_only": {"address": "ro.example.com", "port": 5064}}}
    )

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert any("write" in w.lower() for w in warnings), warnings


@pytest.mark.asyncio
async def test_readonly_run_stays_on_read_only_gateway(monkeypatch, clean_epics_env):
    """A readonly sandbox run never routes through the write gateway.

    ``writes_enabled`` is the deployment posture; ``OSPREY_EXECUTION_MODE`` is
    the per-run claim the executor exports into the sandbox. A readonly run on a
    write-enabled deployment must still land on the read_only gateway so a raw
    ``caput`` issued after ``read_channel()`` is rejected at the network layer."""
    _patch_writes_enabled(monkeypatch, True)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"


@pytest.mark.asyncio
async def test_readwrite_run_uses_write_gateway(monkeypatch, clean_epics_env):
    _patch_writes_enabled(monkeypatch, True)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readwrite")

    connector = EPICSConnector()
    await connector.connect({"gateways": _both_gateways()})

    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"


# -- Per-type posture ------------------------------------------------------


# global false, the simulator armed, and a live block that says nothing about writes
_VA_ARMED_ONLY = {
    "writes_enabled": False,
    "connector": {
        "virtual_accelerator": {"writes_enabled": True},
        "epics": {"pva_channels": []},
    },
}

# the mirror: global true, with the live machine's block explicitly unarmed
_EPICS_DISARMED = {
    "writes_enabled": True,
    "connector": {"epics": {"writes_enabled": False}},
}


@pytest.mark.asyncio
async def test_unarmed_type_stays_on_read_only_gateway(monkeypatch, clean_epics_env):
    """A type with no posture of its own inherits the deployment-wide false."""
    # Arrange
    _patch_control_system(monkeypatch, _VA_ARMED_ONLY)
    connector = EPICSConnector()
    connector._connector_type = "epics"

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5064"


@pytest.mark.asyncio
async def test_armed_type_uses_write_gateway_under_global_false(monkeypatch, clean_epics_env):
    """The armed type routes through write_access even with the global key false."""
    # Arrange
    _patch_control_system(monkeypatch, _VA_ARMED_ONLY)
    connector = EPICSConnector()
    connector._connector_type = "virtual_accelerator"

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5084"


@pytest.mark.asyncio
async def test_type_block_false_overrides_global_true(monkeypatch, clean_epics_env):
    """A block that says false keeps that type off the write gateway."""
    # Arrange
    _patch_control_system(monkeypatch, _EPICS_DISARMED)
    connector = EPICSConnector()
    connector._connector_type = "epics"

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"


@pytest.mark.asyncio
async def test_type_without_block_inherits_global_true(monkeypatch, clean_epics_env):
    """A type the deployment says nothing about keeps the deployment-wide true."""
    # Arrange
    _patch_control_system(monkeypatch, _EPICS_DISARMED)
    connector = EPICSConnector()
    connector._connector_type = "virtual_accelerator"

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"


@pytest.mark.asyncio
async def test_unstamped_connector_reads_the_deployment_wide_key(monkeypatch, clean_epics_env):
    """No stamped type -> no per-type block to read, so the global key decides.

    ``_VA_ARMED_ONLY`` arms one type and leaves the deployment-wide key false;
    a connector nobody built through the factory has no type to key that on.
    """
    # Arrange
    _patch_control_system(monkeypatch, _VA_ARMED_ONLY)
    connector = EPICSConnector()

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert connector._connector_type is None
    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"


@pytest.mark.asyncio
async def test_readonly_run_overrides_an_armed_type(monkeypatch, clean_epics_env):
    """The per-run claim still wins over an armed type."""
    # Arrange
    _patch_control_system(monkeypatch, _VA_ARMED_ONLY)
    monkeypatch.setenv("OSPREY_EXECUTION_MODE", "readonly")
    connector = EPICSConnector()
    connector._connector_type = "virtual_accelerator"

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"


@pytest.mark.asyncio
async def test_va_connector_inherits_the_per_type_selection(monkeypatch, clean_epics_env):
    """The VA connector adds gateway-port filling, not a selection of its own."""
    # Arrange
    _patch_control_system(monkeypatch, _VA_ARMED_ONLY)
    connector = VirtualAcceleratorConnector()
    connector._connector_type = "virtual_accelerator"

    # Act
    await connector.connect({"gateways": _both_gateways()})

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "wr.example.com"
    assert os.environ["EPICS_CA_SERVER_PORT"] == "5084"


@pytest.mark.asyncio
async def test_no_write_gateway_warning_names_the_type_block(monkeypatch, clean_epics_env):
    """The warning points at the block the operator has to edit for this type."""
    # Arrange
    _patch_control_system(monkeypatch, _VA_ARMED_ONLY)
    warnings: list[str] = []
    monkeypatch.setattr(
        epics_connector_module.logger,
        "warning",
        lambda msg, *a, **k: warnings.append(str(msg)),
    )
    connector = EPICSConnector()
    connector._connector_type = "virtual_accelerator"

    # Act
    await connector.connect(
        {"gateways": {"read_only": {"address": "ro.example.com", "port": 5064}}}
    )

    # Assert
    assert os.environ["EPICS_CA_ADDR_LIST"] == "ro.example.com"
    assert any(
        "control_system.connector.virtual_accelerator.writes_enabled" in w for w in warnings
    ), warnings
