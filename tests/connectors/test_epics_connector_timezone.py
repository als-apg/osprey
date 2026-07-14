"""Facility-timezone guard for the EPICS connector's read render path.

No general EPICS connector test exists, so a naive-revert of the read timestamp
(``datetime.fromtimestamp(ts)`` without a zone) would fail nothing. This pins
that ``read_channel`` emits a facility-tz-aware timestamp, using an injected
fake ``_epics`` so pyepics is not required.
"""

from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from osprey.connectors.control_system.epics_connector import EPICSConnector

TOKYO = ZoneInfo("Asia/Tokyo")  # UTC+9, no DST


def _fake_pv(epoch: float):
    pv = MagicMock()
    pv.connected = True
    pv.wait_for_connection.return_value = True
    pv.get.return_value = 1.23
    pv.timestamp = epoch
    pv.units = "A"
    pv.precision = 3
    pv.status = 0
    return pv


@pytest.mark.asyncio
async def test_read_channel_timestamp_is_facility_tz_aware(monkeypatch):
    monkeypatch.setattr(
        "osprey.connectors.control_system.epics_connector.get_facility_timezone",
        lambda: TOKYO,
    )
    connector = EPICSConnector()
    # Inject a fake epics module so no real CA / pyepics is needed.
    connector._epics = MagicMock()
    connector._epics.PV.return_value = _fake_pv(1_750_000_000.0)
    connector._epics_configured = True

    result = await connector.read_channel("SR:TEST:CHANNEL", timeout=1.0)

    assert result.timestamp.tzinfo is not None
    assert result.timestamp.utcoffset().total_seconds() == 9 * 3600  # facility, not UTC/box
    assert result.metadata.timestamp.utcoffset().total_seconds() == 9 * 3600
