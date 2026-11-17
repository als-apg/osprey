"""Tests for ``ioc_record``'s two arms on the ``pv_directory_fleet`` predicate.

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

from osprey.ioc_fleet_connection import FleetEntry, FleetOutcome
from osprey.mcp_server.pv_directory import server
from osprey.mcp_server.pv_directory.tools import ioc_record
from osprey.services.pv_directory.fleet.snapshot import IocRow, Liveness
from osprey.services.pv_directory.fleet.states import Finding, LivenessKind, State

NOW = 1_700_000_000.0

FACTS = {
    "ioc": "ioc-a",
    "outcome": "ok",
    "source": {"kind": "live", "target": "live", "service": "channelfinder"},
    "host": "host-1",
    "ports": [5064],
    "pv_count": 12,
    "inactive_count": 1,
    "curated_count": 3,
    "engineer": "J. Doe",
    "engineer_source": "directory",
    "health_pvs": [{"name": "ioc-a:HEARTBEAT", "role": "heartbeat"}],
    "links": {"panel": "/panel/channel-finder/#/ioc/ioc-a"},
}


def _today(ioc: str) -> str:  # pragma: no cover - only its docstring is used
    """Look up one IOC in the control system's own directory.

    Answers the IOC's host, its ports, how many channels it serves (and how
    many are inactive, and how many the curated channel list declares, with
    ``curated_reason`` when that cannot be counted), its engineer and where
    that came from (``engineer_source``), and ``health_pvs``: the IOC's health
    channels, each its ``name`` (address) and ``role`` (``heartbeat``,
    ``uptime`` or ``engineer``). Names only, never values: to tell whether the
    IOC is alive, read its health channels with channel_read; a ``heartbeat``
    channel that advances between two reads means alive. On the simulator the
    engineer is the address of a channel; read it with channel_read for the name.
    With an alarm feed configured, ``alarms`` lists the IOC's channels in alarm
    now (``outcome``, ``total``, ``rows`` of pv, severity, status, latched,
    acknowledged and time, and ``counts`` per severity over the whole IOC), or
    ``unavailable`` with the feed's ``reason``.
    ``links.panel`` is the panel page showing this IOC. ``not_listed`` means
    the directory lists no channel served by that IOC name.

    Args:
        ioc: The IOC name, as the directory records it.

    Returns:
        JSON object: ``ioc``, ``outcome``, ``source``, ``host``, ``ports``,
        ``pv_count``, ``inactive_count``, ``curated_count``, ``engineer``,
        ``engineer_source``, ``health_pvs``, ``links``, ``alarms`` with an alarm
        feed configured, and the reasons ``curated_reason`` and
        ``health_pvs_reason`` where they apply.
    """
    return ioc


class StubFacts:
    """IOC facts that answer one fixed dict."""

    def __init__(self, facts: dict[str, Any]) -> None:
        self.facts = facts
        self.asked: list[str] = []

    async def ioc_facts(self, ioc: str) -> dict[str, Any]:
        self.asked.append(ioc)
        return self.facts


class NoDirectory:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"ioc_record touched the directory service ({name})")


def _install(facts: dict[str, Any], *, fleet: bool) -> StubFacts:
    stub = StubFacts(facts)
    server.set_tool_context(
        server.ToolContext(
            predicates={"pv_directory_fleet": fleet},
            service=NoDirectory(),  # type: ignore[arg-type]
            pv_facts=stub,  # type: ignore[arg-type]
        )
    )
    return stub


@pytest.fixture(autouse=True)
def _drop_context() -> Iterator[None]:
    yield
    server.set_tool_context(None)


def _row(**overrides: Any) -> IocRow:
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
            pv="ioc-a:HEARTBEAT", kind=LivenessKind.COUNTER, value=7, read_at=NOW - 5
        ),
    }
    fields.update(overrides)
    return IocRow(**fields)


def _description(server_: FastMCP) -> str:
    (tool,) = asyncio.run(server_.list_tools())
    return tool.description


def _probe(*, fleet: bool) -> FastMCP:
    probe = FastMCP("ioc-record-probe")
    ioc_record.register_described(probe, fleet=fleet)
    return probe


# --- the predicate false ----------------------------------------------------------


def test_with_the_fleet_off_the_description_is_todays() -> None:
    today = FastMCP("ioc-record-today")
    today.tool(_today, name="ioc_record")

    assert _description(_probe(fleet=False)) == _description(today)
    assert "read its health channels with channel_read" in _description(_probe(fleet=False))


async def test_with_the_fleet_off_the_answer_is_the_facts_byte_for_byte() -> None:
    stub = _install(dict(FACTS), fleet=False)

    answer = await ioc_record.ioc_record(" ioc-a ")

    assert answer == json.dumps(FACTS)
    assert stub.asked == ["ioc-a"]


def test_the_server_registration_follows_the_installed_predicate() -> None:
    assert ioc_record._fleet_predicate() is False
    _install(FACTS, fleet=True)
    assert ioc_record._fleet_predicate() is True
    _install(FACTS, fleet=False)
    assert ioc_record._fleet_predicate() is False


# --- the predicate true -----------------------------------------------------------


def test_with_the_fleet_on_the_description_presents_fleet_and_asks_no_heartbeat_read() -> None:
    description = _description(_probe(fleet=True))

    assert description == ioc_record.FLEET_DESCRIPTION
    assert "``fleet``" in description
    for word in ("``state``", "``findings``", "``read_time``", "``statement``"):
        assert word in description
    assert "read its health channels" not in description
    assert "advances between two reads" not in description
    assert "``role`` (``heartbeat``" in description


async def test_the_registered_tool_keeps_title_and_read_only() -> None:
    for fleet in (False, True):
        (tool,) = await _probe(fleet=fleet).list_tools()
        assert tool.name == "ioc_record"
        assert tool.title == ioc_record.TITLE
        assert tool.annotations.readOnlyHint is True


async def test_an_ok_entry_is_the_iocs_state_findings_and_read_time() -> None:
    entry = FleetEntry(
        outcome=FleetOutcome.OK, target="live", generated_at=NOW - 10, row=_row()
    ).to_json()
    _install({**FACTS, "fleet": entry}, fleet=True)

    answer = json.loads(await ioc_record.ioc_record("ioc-a"))

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
        "liveness_pv": "ioc-a:HEARTBEAT",
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

    fleet = json.loads(await ioc_record.ioc_record("ioc-a"))["fleet"]

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
    fleet = ioc_record.present_fleet("ioc-z", {"outcome": "mystery", "target": "live"})

    assert fleet["statement"] and "mystery" in fleet["statement"]
    assert fleet["snapshot_time"] is None
