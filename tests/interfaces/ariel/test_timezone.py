"""Facility-timezone parity between the ARIEL web and MCP output paths.

The original bug was parallel-path drift: the web ``_entry_to_response`` emitted
the DB's raw UTC datetime while the MCP ``serialize_entry`` localized to the
facility zone, so the same entry showed different times in the UI vs. the agent.
Both now route through the shared ``to_facility_iso`` helper. This pins that they
render the SAME facility-local ISO string (with offset) for the same entry under
a non-UTC facility — the regression guard against the two paths diverging again.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from osprey.interfaces.ariel.api.routes import _entry_to_response
from osprey.mcp_server.ariel.server import serialize_entry

TOKYO = ZoneInfo("Asia/Tokyo")  # UTC+9, no DST → stable offset


@pytest.fixture()
def facility_tokyo(monkeypatch):
    # to_facility_iso resolves the zone via the config module's get_facility_timezone;
    # patch it there so both the web and MCP call sites see the same facility zone.
    monkeypatch.setattr("osprey.utils.config.get_facility_timezone", lambda: TOKYO)


def _entry() -> dict:
    midnight_utc = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
    return {
        "entry_id": "e1",
        "source_system": "elog",
        "timestamp": midnight_utc,
        "created_at": midnight_utc,
        "updated_at": midnight_utc,
        "author": "op",
        "raw_text": "hello",
        "summary": None,
        "keywords": [],
        "attachments": [],
        "metadata": {},
    }


def test_web_and_mcp_render_same_facility_local_timestamp(facility_tokyo):
    web = _entry_to_response(_entry())
    mcp = serialize_entry(_entry())

    # Midnight UTC → 09:00 Tokyo with an explicit +09:00 offset, identical on both.
    assert web.timestamp == "2026-06-01T09:00:00+09:00"
    assert web.timestamp == mcp["timestamp"]


def test_web_localizes_all_three_timestamp_fields(facility_tokyo):
    """timestamp AND created_at/updated_at must localize — leaving two of three in
    UTC would reintroduce an intra-object asymmetry on the same entry."""
    web = _entry_to_response(_entry())
    assert web.created_at.endswith("+09:00")
    assert web.updated_at.endswith("+09:00")


def test_web_handles_naive_or_missing_gracefully(facility_tokyo):
    """Naive datetimes degrade to str (no crash); the response stays a string."""
    entry = _entry()
    entry["timestamp"] = datetime(2026, 6, 1, 0, 0, 0)  # naive
    web = _entry_to_response(entry)
    assert isinstance(web.timestamp, str)
