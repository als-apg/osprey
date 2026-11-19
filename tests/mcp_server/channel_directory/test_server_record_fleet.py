"""Tests for ``server_record``'s two arms on the ``channel_directory_fleet`` predicate.

With the predicate true the IOC facts carry a ``fleet`` entry; the tool presents
it as the IOC's state, findings and read time, or the outcome as a statement,
and its description stops asking for a heartbeat read. With the predicate false
the description and the answer are byte-identical to what they were before.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.safety_file")

from fastmcp import FastMCP

from osprey.mcp_server.channel_directory import server
from osprey.mcp_server.channel_directory.tools import server_record
from osprey.server_fleet_connection import FleetEntry, FleetOutcome
from osprey.services.channel_directory.fleet.snapshot import Liveness, ServerFleetRow
from osprey.services.channel_directory.fleet.states import Finding, LivenessKind, State

NOW = 1_700_000_000.0

FACTS = {
    "server": "ioc-a",
    "outcome": "ok",
    "source": {"kind": "live", "target": "live", "service": "channelfinder"},
    "host": "host-1",
    "ports": [5064],
    "channel_count": 12,
    "inactive_count": 1,
    "curated_count": 3,
    "engineer": "J. Doe",
    "engineer_source": "directory",
    "health_channels": [{"name": "ioc-a:HEARTBEAT", "role": "heartbeat"}],
    "links": {"panel": "/panel/channel-finder/#/server/ioc-a"},
}


def _today(server: str) -> str:  # pragma: no cover - only its docstring is used
    """Look up one server in the control system's own directory.

    A server is the process serving channels, as the control system's
    directory names it. Answers the server's host, its ports, how many channels it
    serves (and how many are inactive, and how many the curated channel list
    declares, with ``curated_reason`` when that cannot be counted), its
    engineer and where that came from (``engineer_source``), and
    ``health_channels``: the server's health channels, each its ``name``
    (address) and ``role`` (``heartbeat``, ``uptime`` or ``engineer``). Names
    only, never values: to tell whether the server is alive,
    read its health channels with channel_read; a ``heartbeat`` channel that
    advances between two reads means alive. On the simulator the engineer is
    the address of a channel; read it with channel_read for the name.
    With an alarm feed configured, ``alarms`` lists the server's channels in alarm
    now (``outcome``, ``total``, ``rows`` of channel, severity, status, latched,
    acknowledged and time, and ``counts`` per severity over the whole server), or
    ``unavailable`` with the feed's ``reason``.
    ``links.panel`` is the panel page showing this server. ``not_listed`` means
    the directory lists no channel served by that server name.

    Args:
        server: The server's name, as the directory records it.

    Returns:
        JSON object: ``server``, ``outcome``, ``source``, ``host``, ``ports``,
        ``channel_count``, ``inactive_count``, ``curated_count``, ``engineer``,
        ``engineer_source``, ``health_channels``, ``links``, ``alarms`` with an alarm
        feed configured, and the reasons ``curated_reason`` and
        ``health_channels_reason`` where they apply.
    """
    return server


class StubFacts:
    """IOC facts that answer one fixed dict."""

    def __init__(self, facts: dict[str, Any]) -> None:
        self.facts = facts
        self.asked: list[str] = []

    async def server_facts(self, ioc: str) -> dict[str, Any]:
        self.asked.append(ioc)
        return self.facts


class NoDirectory:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"server_record touched the directory service ({name})")


def _install(facts: dict[str, Any], *, fleet: bool) -> StubFacts:
    stub = StubFacts(facts)
    server.set_tool_context(
        server.ToolContext(
            predicates={"channel_directory_fleet": fleet},
            service=NoDirectory(),  # type: ignore[arg-type]
            channel_facts=stub,  # type: ignore[arg-type]
        )
    )
    return stub


@pytest.fixture(autouse=True)
def _drop_context() -> Iterator[None]:
    yield
    server.set_tool_context(None)


def _row(**overrides: Any) -> ServerFleetRow:
    fields: dict[str, Any] = {
        "name": "ioc-a",
        "host": "host-1",
        "class_name": "default",
        "group": None,
        "state": State.STALLED,
        "since": NOW - 300,
        "reason": "its heartbeat has not advanced for 300 s",
        "findings": (Finding.DEGRADED,),
        "directory_status": "Active",
        "listed": True,
        "sources": (),
        "liveness": Liveness(
            channel="ioc-a:HEARTBEAT", kind=LivenessKind.COUNTER, value=7, read_at=NOW - 5
        ),
    }
    fields.update(overrides)
    return ServerFleetRow(**fields)


def _description(server_: FastMCP) -> str:
    (tool,) = asyncio.run(server_.list_tools())
    return tool.description


def _probe(*, fleet: bool) -> FastMCP:
    probe = FastMCP("server-record-probe")
    server_record.register_described(probe, fleet=fleet)
    return probe


# --- the predicate false ----------------------------------------------------------


def test_with_the_fleet_off_the_description_is_todays() -> None:
    today = FastMCP("server-record-today")
    today.tool(_today, name="server_record")

    assert _description(_probe(fleet=False)) == _description(today)
    assert "read its health channels with channel_read" in _description(_probe(fleet=False))


async def test_with_the_fleet_off_the_answer_is_the_facts_byte_for_byte() -> None:
    stub = _install(dict(FACTS), fleet=False)

    answer = await server_record.server_record(" ioc-a ")

    assert answer == json.dumps(FACTS)
    assert stub.asked == ["ioc-a"]


def test_the_server_registration_follows_the_installed_predicate() -> None:
    assert server_record._fleet_predicate() is False
    _install(FACTS, fleet=True)
    assert server_record._fleet_predicate() is True
    _install(FACTS, fleet=False)
    assert server_record._fleet_predicate() is False


# --- the predicate true -----------------------------------------------------------


def test_with_the_fleet_on_the_description_presents_fleet_and_asks_no_heartbeat_read() -> None:
    description = _description(_probe(fleet=True))

    assert description == server_record.FLEET_DESCRIPTION
    assert "``fleet``" in description
    for word in ("``state``", "``findings``", "``read_time``", "``statement``"):
        assert word in description
    assert "read its health channels" not in description
    assert "advances between two reads" not in description
    assert "``role`` (``heartbeat``" in description


async def test_the_registered_tool_keeps_title_and_read_only() -> None:
    for fleet in (False, True):
        (tool,) = await _probe(fleet=fleet).list_tools()
        assert tool.name == "server_record"
        assert tool.title == server_record.TITLE
        assert tool.annotations.readOnlyHint is True


async def test_an_ok_entry_is_the_servers_state_findings_and_read_time() -> None:
    entry = FleetEntry(
        outcome=FleetOutcome.OK, target="live", generated_at=NOW - 10, row=_row()
    ).to_json()
    _install({**FACTS, "fleet": entry}, fleet=True)

    answer = json.loads(await server_record.server_record("ioc-a"))

    assert {k: v for k, v in answer.items() if k != "fleet"} == FACTS
    assert answer["fleet"] == {
        "outcome": "ok",
        "target": "live",
        "generated_at": NOW - 10,
        "snapshot_time": "2023-11-14T22:13:10+00:00",
        "statement": None,
        "state": "stalled",
        "since": NOW - 300,
        "reason": "its heartbeat has not advanced for 300 s",
        "findings": ["degraded"],
        "read_at": NOW - 5,
        "read_time": "2023-11-14T22:13:15+00:00",
        "liveness_channel": "ioc-a:HEARTBEAT",
    }


@pytest.mark.parametrize(
    "outcome",
    [
        FleetOutcome.NO_SNAPSHOT,
        FleetOutcome.NOT_REFRESHED,
        FleetOutcome.CANNOT_LIST,
        FleetOutcome.UNAVAILABLE,
        FleetOutcome.STALE,
        FleetOutcome.NOT_IN_SNAPSHOT,
    ],
)
async def test_any_other_outcome_is_a_statement(outcome: FleetOutcome) -> None:
    generated = NOW - 1000 if outcome is FleetOutcome.STALE else None
    entry = FleetEntry(
        outcome=outcome, target="va", reason=f"because {outcome.value}", generated_at=generated
    ).to_json()
    _install({**FACTS, "fleet": entry}, fleet=True)

    fleet = json.loads(await server_record.server_record("ioc-a"))["fleet"]

    assert fleet["outcome"] == outcome.value
    assert fleet["target"] == "va"
    assert fleet["reason"] == f"because {outcome.value}"
    statement = fleet["statement"]
    assert "ioc-a" in statement and "va" in statement
    assert "not known" in statement
    assert "says nothing about whether it runs" in statement
    assert " down" not in statement.lower()
    assert "state" not in fleet and "findings" not in fleet
    if outcome is FleetOutcome.STALE:
        assert fleet["snapshot_time"] in statement


def test_an_unforeseen_outcome_still_reads_as_a_statement() -> None:
    fleet = server_record.present_fleet("ioc-z", {"outcome": "mystery", "target": "live"})

    assert fleet["statement"] and "mystery" in fleet["statement"]
    assert fleet["snapshot_time"] is None
