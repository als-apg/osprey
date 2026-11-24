"""The fleet route under ``/api/directory`` and the fleet service in the descriptor.

The route answers the fleet connection's answer for the control target,
unchanged in shape; it asks the fleet service and nothing else, and is absent
while the ``channel_directory_fleet`` predicate is false. The IOC page carries the
shared facts producer's ``fleet`` field through untouched (D13).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("osprey_connectors.service_answers")
pytest.importorskip("osprey_connectors.safety_file")

from osprey import server_fleet_connection
from osprey.server_fleet_connection import (
    SERVER_FLEET_URL_ENV_VAR,
    FleetAnswer,
    FleetOutcome,
    entry_for,
)
from osprey.services.channel_directory import service
from tests.interfaces.channel_finder import test_directory_api_pages as _pages
from tests.interfaces.channel_finder.test_directory_api_pages import (  # noqa: F401
    FACILITY,
    LIVE_BUILD,
    VA_BUILD,
    FakeDirectory,
    _config,
    _config_path,
    _get,
    _live_server_records,
    _routes,
    cold_roster,
    fakes,
)
from tests.test_server_fleet_connection import GENERATED_AT, _snapshot

start = _pages.start
record = _pages.record


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


def _querying(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every directory query and backend build the directory service is asked for."""
    calls: list[str] = []
    for method in ("count", "search", "describe", "backend"):
        real = getattr(service.DirectoryService, method)

        def recorded(real: Any = real, method: str = method) -> Any:
            async def call(self: Any, *args: Any, **kwargs: Any) -> Any:
                calls.append(method)
                return await real(self, *args, **kwargs)

            return call

        monkeypatch.setattr(service.DirectoryService, method, recorded())
    return calls


def _no_control_system_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test on any connector built for a control-system read."""
    from osprey_connectors.factory import ConnectorFactory

    async def refused(*_: Any, **__: Any) -> Any:
        raise AssertionError("the fleet route read the control system")

    monkeypatch.setattr(ConnectorFactory, "create_control_system_connector", refused)


# -- the route ----------------------------------------------------------------


def test_the_route_answers_the_fleet_connection_unchanged(
    start, tmp_path: Path, fleet: _Fleet
) -> None:
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        response = client.get("/api/directory/fleet")

    assert response.status_code == 200
    expected = FleetAnswer(
        outcome=FleetOutcome.OK,
        target="live",
        generated_at=GENERATED_AT,
        snapshot=_snapshot("live"),
    ).to_json()
    assert response.json() == expected
    assert fleet.asked == ["live"]


def test_the_route_asks_for_the_control_target(
    start, tmp_path: Path, fleet: _Fleet, record
) -> None:
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        record.move("va", 3)
        answer = client.get("/api/directory/fleet").json()
        record.value = None
        baseline = client.get("/api/directory/fleet").json()

    assert fleet.asked == ["va", "live"]
    assert answer["target"] == "va"
    assert baseline["target"] == "live"


def test_the_route_asks_the_baseline_of_a_simulator_deployment(
    start, tmp_path: Path, fleet: _Fleet
) -> None:
    with start(_config(tmp_path, VA_BUILD, _fleet_on(FACILITY))) as (_app, client):
        answer = client.get("/api/directory/fleet").json()

    assert fleet.asked == ["va"]
    assert answer["target"] == "va"


@pytest.mark.parametrize(
    "outcome",
    [
        FleetOutcome.NO_SNAPSHOT,
        FleetOutcome.NOT_REFRESHED,
        FleetOutcome.CANNOT_LIST,
        FleetOutcome.UNAVAILABLE,
    ],
)
def test_a_non_ok_answer_passes_through_with_its_reason(
    start, tmp_path: Path, fleet: _Fleet, outcome: FleetOutcome
) -> None:
    fleet.answer = FleetAnswer(outcome=outcome, target="live", reason=f"why {outcome.value}")
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        response = client.get("/api/directory/fleet")

    assert response.status_code == 200
    assert response.json() == {
        "outcome": outcome.value,
        "target": "live",
        "reason": f"why {outcome.value}",
        "generated_at": None,
        "snapshot": None,
    }


def test_a_stale_answer_keeps_its_snapshot(start, tmp_path: Path, fleet: _Fleet) -> None:
    fleet.answer = FleetAnswer(
        outcome=FleetOutcome.STALE,
        target="live",
        reason="the snapshot is older than its stale bound",
        generated_at=GENERATED_AT,
        snapshot=_snapshot("live"),
    )
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        answer = client.get("/api/directory/fleet").json()

    assert answer == fleet.answer.to_json()
    assert answer["snapshot"]["servers"]


def test_an_unreachable_service_is_an_unavailable_answer(start, tmp_path: Path) -> None:
    # The real connection, pointed at a port nothing listens on.
    section = _fleet_on(FACILITY)
    config = _config(tmp_path, LIVE_BUILD, section, services={"server_fleet": {"port": 1}})
    with start(config) as (_app, client):
        response = client.get("/api/directory/fleet")

    assert response.status_code == 200
    answer = response.json()
    assert answer["outcome"] == "unavailable"
    assert answer["target"] == "live"
    assert answer["reason"]
    assert answer["snapshot"] is None


def test_the_route_reads_no_control_system_and_queries_no_directory(
    start, tmp_path: Path, fleet: _Fleet, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeDirectory.records = _live_server_records()
    calls = _querying(monkeypatch)
    _no_control_system_read(monkeypatch)
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        response = client.get("/api/directory/fleet")

    assert response.status_code == 200
    assert fleet.asked == ["live"]
    assert calls == []


def test_the_route_is_absent_when_the_fleet_is_off(start, tmp_path: Path, fleet: _Fleet) -> None:
    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        response = client.get("/api/directory/fleet")

    assert response.status_code == 404
    assert fleet.asked == []


@pytest.mark.usefixtures("fleet")
def test_the_route_matches_the_shared_fixture(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        answer = client.get("/api/directory/fleet").json()

    shared = _routes()["fleet"]
    assert set(answer) == set(shared)
    assert set(answer["snapshot"]) == set(shared["snapshot"])
    assert set(answer["snapshot"]["counts"]) == set(shared["snapshot"]["counts"])
    assert set(answer["snapshot"]["servers"][0]) == set(shared["snapshot"]["servers"][0])


def test_the_shared_fixture_is_a_valid_answer() -> None:
    from osprey.services.channel_directory.fleet.snapshot import Snapshot

    shared = _routes()["fleet"]
    snapshot = Snapshot.from_json(shared["snapshot"])
    answer = FleetAnswer(
        outcome=FleetOutcome(shared["outcome"]),
        target=shared["target"],
        reason=shared["reason"],
        generated_at=shared["generated_at"],
        snapshot=snapshot,
    )
    assert answer.to_json() == shared


# -- the descriptor ------------------------------------------------------------


def test_the_descriptor_lists_the_fleet_when_it_is_on(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        services = client.get("/api/info").json()["directory"]["configured_services"]

    assert services == ["fleet"]


def test_the_descriptor_omits_the_fleet_when_it_is_off(start, tmp_path: Path) -> None:
    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        services = client.get("/api/info").json()["directory"]["configured_services"]

    assert "fleet" not in services


# -- the IOC page carries the shared producer's fleet field (D13) ---------------


@pytest.mark.usefixtures("fleet")
def test_the_server_page_carries_the_fleet_entry_through(start, tmp_path: Path) -> None:
    FakeDirectory.records = _live_server_records()
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY), "SR:BPM1")) as (_app, client):
        ioc = _get(client, "server", "ioc-a")

    expected = entry_for(
        FleetAnswer(
            outcome=FleetOutcome.OK,
            target="live",
            generated_at=GENERATED_AT,
            snapshot=_snapshot("live"),
        ),
        "ioc-a",
    ).to_json()
    assert ioc["fleet"] == expected
    assert ioc["fleet"]["row"]["name"] == "ioc-a"
    assert set(ioc) - {"alarms"} == (set(_routes()["ioc/{name}"]) - {"alarms"}) | {"fleet"}


def test_the_server_page_carries_a_non_ok_fleet_outcome(
    start, tmp_path: Path, fleet: _Fleet
) -> None:
    FakeDirectory.records = _live_server_records()
    fleet.answer = FleetAnswer.unavailable("live", "the server fleet service is down")
    with start(_config(tmp_path, LIVE_BUILD, _fleet_on(FACILITY))) as (_app, client):
        ioc = _get(client, "server", "ioc-a")

    assert ioc["fleet"] == entry_for(fleet.answer, "ioc-a").to_json()
    assert ioc["fleet"]["outcome"] == "unavailable"
    assert ioc["fleet"]["row"] is None


def test_the_server_page_has_no_fleet_field_when_the_fleet_is_off(
    start, tmp_path: Path, fleet: _Fleet
) -> None:
    FakeDirectory.records = _live_server_records()
    with start(_config(tmp_path, LIVE_BUILD, FACILITY)) as (_app, client):
        ioc = _get(client, "server", "ioc-a")

    assert "fleet" not in ioc
    assert fleet.asked == []
