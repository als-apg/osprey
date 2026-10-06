"""Tests for the agent-facing time rendering of the event dispatcher."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from osprey.dispatch.agent_time import agent_json, registry_instant

STAMP = datetime(2026, 1, 15, 20, 0, tzinfo=UTC)
TOKYO = "2026-01-16T05:00:00+09:00"


@pytest.mark.usefixtures("facility_zone_config")
def test_a_stamped_instant_renders_in_the_facility_zone():
    assert json.loads(agent_json({"timestamp": STAMP})) == {"timestamp": TOKYO}


@pytest.mark.usefixtures("facility_zone_config")
def test_text_that_reads_as_an_instant_is_shown_as_it_came():
    body = {"timestamp": "2026-01-15T20:00:00+00:00", "nested": {"at": "2026-01-15T20:00:00Z"}}
    assert json.loads(agent_json(body)) == body


@pytest.mark.usefixtures("facility_zone_config")
def test_a_registry_instant_renders_in_the_facility_zone():
    assert registry_instant(STAMP.isoformat()) == TOKYO
    assert registry_instant(None) is None


def test_the_zone_is_utc_without_a_config(tmp_path, monkeypatch):
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert json.loads(agent_json({"t": STAMP})) == {"t": "2026-01-15T20:00:00+00:00"}
