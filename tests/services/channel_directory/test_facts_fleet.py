"""IOC facts carry the IOC's fleet entry whenever the fleet is on (D13).

The ``fleet`` field is :func:`~osprey.server_fleet_connection.entry_for`'s answer
for the control target: a row, or an outcome with its reason. It is present on
every ``server_facts`` answer while the ``channel_directory_fleet`` predicate holds, and
absent only when it does not, so a reader tells "feature off" from "no fresh
answer". The fleet service is the only thing asked: no extra directory query.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")
pytest.importorskip("osprey_connectors.safety_file")

from osprey import server_fleet_connection
from osprey.server_fleet_connection import SERVER_FLEET_URL_ENV_VAR, FleetAnswer, FleetOutcome
from osprey.services.channel_directory import service
from tests.services.channel_directory.test_facts import (  # noqa: F401  (fixtures used by name)
    FACILITY,
    LIVE_BUILD,
    VA_BUILD,
    FakeDirectory,
    _config,
    _facts,
    _Record,
    _record,
    _records,
    cold_roster,
    fakes,
    record,
)
from tests.test_server_fleet_connection import GENERATED_AT, _snapshot


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SERVER_FLEET_URL_ENV_VAR, raising=False)


class _Fleet:
    """A stubbed fleet service: the answer it gives and the targets it was asked for."""

    def __init__(self) -> None:
        self.answer: FleetAnswer | None = None
        self.asked: list[str] = []


@pytest.fixture
def fleet(monkeypatch: pytest.MonkeyPatch) -> _Fleet:
    stub = _Fleet()

    async def fetch(_cfg: Any, target: str, **_: Any) -> FleetAnswer:
        stub.asked.append(target)
        if stub.answer is None:
            return FleetAnswer(
                outcome=FleetOutcome.OK,
                target=target,
                generated_at=GENERATED_AT,
                snapshot=_snapshot(target),
            )
        return stub.answer

    monkeypatch.setattr(server_fleet_connection, "fetch_snapshot", fetch)
    return stub


def _fleet_on(section: dict[str, Any]) -> dict[str, Any]:
    return {**section, "fleet": {}}


def _querying(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """Every query the directory service is asked: its method and arguments."""
    calls: list[tuple[str, Any]] = []
    for method in ("count", "search", "describe"):
        real = getattr(service.DirectoryService, method)

        def recorded(real: Any = real, method: str = method) -> Any:
            async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
                calls.append((method, repr((args, sorted(kwargs.items())))))
                return await real(self, *args, **kwargs)

            return call

        monkeypatch.setattr(service.DirectoryService, method, recorded())
    return calls


# -- present whenever the predicate holds ----------------------------------------------


async def test_a_found_server_carries_its_fleet_row(tmp_path: Path, fleet: _Fleet) -> None:
    FakeDirectory.records = _records(_record("SR:A:X"))
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["outcome"] == "found"
    assert fleet.asked == ["live"]
    assert ioc["fleet"]["outcome"] == "ok"
    assert ioc["fleet"]["target"] == "live"
    assert ioc["fleet"]["generated_at"] == GENERATED_AT
    assert ioc["fleet"]["reason"] is None
    assert ioc["fleet"]["row"]["name"] == "ioc-a"
    assert ioc["fleet"]["row"]["state"] == "alive"


@pytest.mark.usefixtures("fleet")
async def test_the_fleet_entry_is_entry_for_to_json(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:A:X"))
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-a")

    answer = FleetAnswer(
        outcome=FleetOutcome.OK,
        target="live",
        generated_at=GENERATED_AT,
        snapshot=_snapshot("live"),
    )
    assert ioc["fleet"] == server_fleet_connection.entry_for(answer, "ioc-a").to_json()


@pytest.mark.usefixtures("fleet")
async def test_a_server_the_snapshot_lacks_is_not_in_snapshot(tmp_path: Path) -> None:
    FakeDirectory.records = _records(_record("SR:Z:X", ioc="ioc-z"))
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-z")

    assert ioc["fleet"]["outcome"] == "not_in_snapshot"
    assert ioc["fleet"]["row"] is None
    assert "ioc-z" in ioc["fleet"]["reason"]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (FleetOutcome.STALE, "the snapshot is 999 s old"),
        (FleetOutcome.NOT_REFRESHED, "target live is not refreshed"),
        (FleetOutcome.NO_SNAPSHOT, "no snapshot yet"),
        (FleetOutcome.UNAVAILABLE, "could not reach the server fleet service"),
    ],
)
async def test_a_non_ok_answer_is_carried_with_its_reason(
    tmp_path: Path, fleet: _Fleet, outcome: FleetOutcome, reason: str
) -> None:
    fleet.answer = FleetAnswer(outcome=outcome, target="live", reason=reason)
    FakeDirectory.records = _records(_record("SR:A:X"))
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["outcome"] == "found"
    assert ioc["fleet"]["outcome"] == outcome.value
    assert ioc["fleet"]["reason"] == reason
    assert ioc["fleet"]["row"] is None


async def test_a_stale_answer_carries_no_row(tmp_path: Path, fleet: _Fleet) -> None:
    fleet.answer = FleetAnswer(
        outcome=FleetOutcome.STALE,
        target="live",
        reason="stopped refreshing",
        generated_at=GENERATED_AT,
        snapshot=_snapshot("live"),
    )
    FakeDirectory.records = _records(_record("SR:A:X"))
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["fleet"]["outcome"] == "stale"
    assert ioc["fleet"]["generated_at"] == GENERATED_AT
    assert ioc["fleet"]["row"] is None


@pytest.mark.usefixtures("fleet")
async def test_a_server_the_directory_does_not_list_still_carries_fleet(tmp_path: Path) -> None:
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["outcome"] == "not_listed"
    assert ioc["fleet"]["outcome"] == "ok"
    assert ioc["fleet"]["row"]["name"] == "ioc-a"


async def test_an_unavailable_directory_still_carries_fleet(tmp_path: Path, fleet: _Fleet) -> None:
    FakeDirectory.fail = "connection refused"
    page = _facts(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY)), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["outcome"] == "unavailable"
    assert fleet.asked == ["live"]
    assert ioc["fleet"]["outcome"] == "ok"
    assert ioc["fleet"]["row"]["name"] == "ioc-a"


async def test_the_fleet_is_asked_for_the_control_target(
    tmp_path: Path,
    fleet: _Fleet,
    record: _Record,  # noqa: F811  (the fixture)
) -> None:
    record.move("va", 1)
    FakeDirectory.records = _records(_record("SR:BPM1:X", ioc="sim-SR"))
    page = _facts(_config(tmp_path, VA_BUILD, _fleet_on({})), tmp_path)

    ioc = await page.server_facts("sim-SR")

    assert ioc["source"]["target"] == "va"
    assert fleet.asked == ["va"]
    assert ioc["fleet"]["target"] == "va"
    assert ioc["fleet"]["outcome"] == "not_in_snapshot"


async def test_an_unreachable_service_reads_unavailable(tmp_path: Path) -> None:
    # The real fetch, dialing a port nothing listens on: a statement, never an error.
    FakeDirectory.records = _records(_record("SR:A:X"))
    config = _config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))
    config["services"] = {"server_fleet": {"port": 1}}
    page = _facts(config, tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["outcome"] == "found"
    assert ioc["fleet"]["outcome"] == "unavailable"
    assert ioc["fleet"]["reason"]
    assert ioc["fleet"]["row"] is None


# -- no added directory query ----------------------------------------------------------


async def test_the_fleet_adds_no_directory_query(
    tmp_path: Path, fleet: _Fleet, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeDirectory.records = _records(_record("SR:A:X"))
    calls = _querying(monkeypatch)
    answers = {}
    for on in (False, True):
        section = _fleet_on(FACILITY) if on else dict(FACILITY)
        page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)
        calls.clear()
        await page.server_facts("ioc-a")
        answers[on] = sorted(calls)

    assert fleet.asked == ["live"]
    assert answers[True] == answers[False]
    assert answers[True]


# -- absent only when the predicate is false -------------------------------------------


@pytest.mark.parametrize("records", [True, False])
async def test_fleet_off_carries_no_fleet_field(
    tmp_path: Path, fleet: _Fleet, records: bool
) -> None:
    if records:
        FakeDirectory.records = _records(_record("SR:A:X"))
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert "fleet" not in ioc
    assert fleet.asked == []


async def test_fleet_off_on_an_unavailable_directory_carries_no_fleet_field(
    tmp_path: Path, fleet: _Fleet
) -> None:
    FakeDirectory.fail = "connection refused"
    page = _facts(_config(tmp_path, LIVE_BUILD, FACILITY), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert ioc["outcome"] == "unavailable"
    assert "fleet" not in ioc
    assert fleet.asked == []


async def test_a_malformed_fleet_block_reads_as_off(tmp_path: Path, fleet: _Fleet) -> None:
    FakeDirectory.records = _records(_record("SR:A:X"))
    section = {**FACILITY, "fleet": {"no_such_key": 1}}
    page = _facts(_config(tmp_path, LIVE_BUILD, section), tmp_path)

    ioc = await page.server_facts("ioc-a")

    assert "fleet" not in ioc
    assert fleet.asked == []
