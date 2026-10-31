"""Which event names the live OTLP contract test holds the pinned CLI to.

``tests/e2e/test_telemetry_contract_live.py`` compares a live CLI run with the
recorded fixture by event name. Events the CLI emits from work beside the turn
(the retention sweep, the MCP servers connecting) appear in one run of the
pinned version and not the next, so neither side is held to them; any other
name that appears or disappears is a contract change. The comparison is
exercised here, in the unit lane, because the live test needs a model and runs
only under ``full-ci``.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.test_telemetry_contract_live import FIXTURE, contract_events, structure


@pytest.fixture
def recorded() -> dict[str, list[str]]:
    return structure(json.loads(FIXTURE.read_text())["requests"])["events"]


@pytest.mark.parametrize("name", ["mcp_server_connection", "retention_sweep"])
def test_a_housekeeping_event_in_the_live_run_only_is_no_drift(recorded, name):
    live = {**recorded, name: ["event.name", "session.id"]}

    _, kept_live, drift = contract_events(recorded, live)

    assert drift == {"missing": [], "extra": []}
    assert name not in kept_live


@pytest.mark.parametrize("name", ["mcp_server_connection", "retention_sweep"])
def test_a_housekeeping_event_in_the_fixture_only_is_no_drift(recorded, name):
    fixture = {**recorded, name: ["event.name", "session.id"]}

    kept_expected, _, drift = contract_events(fixture, recorded)

    assert drift == {"missing": [], "extra": []}
    assert name not in kept_expected


def test_a_new_event_name_is_drift(recorded):
    live = {**recorded, "brand_new_event": ["event.name", "session.id"]}

    _, _, drift = contract_events(recorded, live)

    assert drift == {"missing": [], "extra": ["brand_new_event"]}


def test_a_lost_event_name_is_drift(recorded):
    live = {name: keys for name, keys in recorded.items() if name != "tool_result"}

    _, _, drift = contract_events(recorded, live)

    assert drift == {"missing": ["tool_result"], "extra": []}
