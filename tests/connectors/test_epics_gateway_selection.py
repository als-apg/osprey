"""Gateway selection for the EPICS connector.

EPICS Channel Access uses one process-wide CA context, so the connector points
at a single gateway. A read-only gateway rejects writes, so a write-enabled
deployment must route through the write-capable gateway. Selection is
defense-in-depth: the write_access gateway is used only when writes are enabled
*and* it is configured; otherwise the connector stays on the read_only gateway,
so a read-only deployment also has its writes rejected at the network layer.
"""

import os

import pytest

import osprey.connectors.control_system.epics_connector as epics_connector_module
from osprey.connectors.control_system.epics_connector import EPICSConnector

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


def _patch_writes_enabled(monkeypatch, enabled: bool):
    def fake_get_config_value(key, default=None):
        if key == "control_system.writes_enabled":
            return enabled
        return default  # limits_checking.enabled etc. fall back to default

    monkeypatch.setattr("osprey.utils.config.get_config_value", fake_get_config_value)


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
