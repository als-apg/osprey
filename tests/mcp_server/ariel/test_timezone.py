"""Facility-timezone guards for the ARIEL server's date parse + serialize contract.

The existing ARIEL tests run with the default (UTC) facility zone, so a regression
to a hardcoded UTC would stay green. These pin the contract under a non-UTC zone:
operator-supplied dates are interpreted facility-local, and emitted timestamps are
rendered in the facility zone with an explicit offset.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from osprey.mcp_server.ariel.server import parse_date_filters, serialize_entry

TOKYO = ZoneInfo("Asia/Tokyo")  # UTC+9, no DST → stable offset


@pytest.fixture()
def facility_tokyo(monkeypatch):
    monkeypatch.setattr("osprey.utils.config.get_facility_timezone", lambda: TOKYO)


def test_parse_date_filters_interprets_naive_input_as_facility_local(facility_tokyo):
    start, end = parse_date_filters("2026-06-01 09:00:00", "2026-06-01T17:00:00")
    assert start.utcoffset().total_seconds() == 9 * 3600
    assert end.utcoffset().total_seconds() == 9 * 3600


def test_parse_date_filters_respects_explicit_offset(facility_tokyo):
    # An input that already carries an offset must NOT be overridden.
    start, _ = parse_date_filters("2026-06-01T09:00:00+00:00", None)
    assert start.utcoffset().total_seconds() == 0


def test_serialize_entry_renders_timestamp_in_facility_zone(facility_tokyo):
    entry = {
        "entry_id": "e1",
        "timestamp": datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),  # midnight UTC
        "author": "op",
        "source_system": "elog",
        "raw_text": "hello",
    }
    out = serialize_entry(entry)
    # UTC midnight rendered in Tokyo (+09:00) → 09:00+09:00 on the same date.
    assert out["timestamp"].endswith("+09:00")
    assert "T09:00:00" in out["timestamp"]
