"""The fleet connection (C6): which URL readers dial, and how every answer becomes an outcome."""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("osprey_connectors.safety_file")

from osprey.port_layout import SLOTS_BY_NAME, default_port
from osprey.server_fleet_connection import (
    DEFAULT_SERVER_FLEET_PORT,
    SERVER_FLEET_TIMEOUT_S,
    SERVER_FLEET_URL_ENV_VAR,
    FleetAnswer,
    FleetOutcome,
    entry_for,
    fetch_snapshot,
    fleet_url,
)
from osprey.services.channel_directory.fleet.snapshot import (
    Liveness,
    Outcome,
    ReadPathStatus,
    ServerFleetRow,
    Snapshot,
)
from osprey.services.channel_directory.fleet.states import LivenessKind, ReadPath, State

FLEET_ON = {"channel_directory": {"fleet": {}}}
GENERATED_AT = 1_700_000_000.0


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv(SERVER_FLEET_URL_ENV_VAR, raising=False)


def _row(name: str) -> ServerFleetRow:
    return ServerFleetRow(
        name=name,
        host="host-1",
        class_name="default",
        group=None,
        state=State.ALIVE,
        since=GENERATED_AT - 100,
        reason="counter moved",
        findings=(),
        directory_status="Active",
        listed=True,
        sources=(),
        liveness=Liveness(
            channel=f"{name}:HB", kind=LivenessKind.COUNTER, value=3, read_at=GENERATED_AT
        ),
    )


def _snapshot(target: str = "live", **overrides) -> Snapshot:
    fields = {
        "target": target,
        "generated_at": GENERATED_AT,
        "cycle": 4,
        "period_s": 30.0,
        "cycle_duration_s": 2.0,
        "read_path": ReadPathStatus(status=ReadPath.OK, channel="SR:PROBE"),
        "sources": (),
        "group_by": "host",
        "servers": [_row("ioc-a"), _row("ioc-b")],
    }
    fields.update(overrides)
    return Snapshot.ok(**fields)


def _answering(handler):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return httpx.MockTransport(record), seen


def _json(document) -> httpx.Response:
    return httpx.Response(200, json=document)


# --- fleet_url ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"channel_directory": None},
        {"channel_directory": {"type": "x"}},
        {"channel_directory": "bad"},
    ],
)
def test_no_url_when_the_fleet_is_off(config, monkeypatch):
    # Even an override does not turn the feature on: the predicate decides.
    monkeypatch.setenv(SERVER_FLEET_URL_ENV_VAR, "http://server-fleet:1")
    assert fleet_url(config) is None


def test_the_override_wins_and_loses_its_trailing_slash(monkeypatch):
    monkeypatch.setenv(SERVER_FLEET_URL_ENV_VAR, "http://server-fleet:9/")
    config = {**FLEET_ON, "services": {"server_fleet": {"port": 18184}}}
    assert fleet_url(config) == "http://server-fleet:9"


def test_an_empty_override_falls_back_to_the_configured_port(monkeypatch):
    monkeypatch.setenv(SERVER_FLEET_URL_ENV_VAR, "")
    config = {**FLEET_ON, "services": {"server_fleet": {"port": 18184}}}
    assert fleet_url(config) == "http://127.0.0.1:18184"


@pytest.mark.parametrize(
    ("config", "port"),
    [
        (FLEET_ON, DEFAULT_SERVER_FLEET_PORT),
        (
            {**FLEET_ON, "deployment": {"port_base": 20000}},
            default_port("server_fleet", base=20000),
        ),
        ({**FLEET_ON, "services": {"server_fleet": None}}, DEFAULT_SERVER_FLEET_PORT),
    ],
)
def test_no_port_is_the_layout_slot_at_this_configs_base(config, port):
    assert fleet_url(config) == f"http://127.0.0.1:{port}"


def test_the_slot_is_a_singleton_in_the_services_tier():
    slot = SLOTS_BY_NAME["server_fleet"]
    assert slot.tier == "services"
    assert slot.config_key == "services.server_fleet.port"
    assert not slot.per_index


# --- fetch_snapshot -----------------------------------------------------------------


async def test_an_ok_snapshot_is_fetched_by_target():
    transport, seen = _answering(lambda request: _json(_snapshot().to_json()))
    answer = await fetch_snapshot(FLEET_ON, "live", transport=transport, now=GENERATED_AT + 5)
    assert answer.outcome is FleetOutcome.OK
    assert answer.reason is None
    assert answer.snapshot == _snapshot()
    assert answer.generated_at == GENERATED_AT
    (request,) = seen
    assert request.method == "GET"
    assert request.url.path == "/v1/snapshot"
    assert request.url.params["target"] == "live"
    assert str(request.url).startswith(f"http://127.0.0.1:{DEFAULT_SERVER_FLEET_PORT}/")


async def test_the_override_url_is_dialled(monkeypatch):
    monkeypatch.setenv(SERVER_FLEET_URL_ENV_VAR, "http://server-fleet:7/")
    transport, seen = _answering(lambda request: _json(_snapshot().to_json()))
    await fetch_snapshot(FLEET_ON, "live", transport=transport, now=GENERATED_AT)
    assert str(seen[0].url).startswith("http://server-fleet:7/v1/snapshot")


@pytest.mark.parametrize(
    "outcome", [Outcome.NO_SNAPSHOT, Outcome.NOT_REFRESHED, Outcome.CANNOT_LIST]
)
async def test_the_services_own_outcomes_pass_through_with_their_reason(outcome):
    document = Snapshot.not_ok(outcome, target="live", reason="why not").to_json()
    transport, _ = _answering(lambda request: _json(document))
    answer = await fetch_snapshot(FLEET_ON, "live", transport=transport, now=GENERATED_AT)
    assert answer.outcome == FleetOutcome(outcome.value)
    assert answer.reason == "why not"
    assert answer.snapshot is None


@pytest.mark.parametrize(
    ("period_s", "cycle_duration_s", "age", "stale"),
    [
        (30.0, 2.0, 90.0, False),  # exactly 3 x period: not yet stale
        (30.0, 2.0, 90.5, True),
        (30.0, 50.0, 140.0, False),  # a slow cycle widens the bound to 3 x 50
        (30.0, 50.0, 150.5, True),
    ],
)
async def test_stale_is_older_than_three_times_the_larger_of_period_and_cycle(
    period_s, cycle_duration_s, age, stale
):
    document = _snapshot(period_s=period_s, cycle_duration_s=cycle_duration_s).to_json()
    transport, _ = _answering(lambda request: _json(document))
    answer = await fetch_snapshot(FLEET_ON, "live", transport=transport, now=GENERATED_AT + age)
    if stale:
        assert answer.outcome is FleetOutcome.STALE
        assert answer.reason and "refresh" in answer.reason
        # The view can still show the old snapshot, marked stale, with its time.
        assert answer.snapshot is not None
        assert answer.generated_at == GENERATED_AT
    else:
        assert answer.outcome is FleetOutcome.OK


def _raise(exc):
    def handler(_request):
        raise exc

    return handler


@pytest.mark.parametrize(
    ("handler", "fragment"),
    [
        (_raise(httpx.ConnectError("refused")), "could not reach"),
        (_raise(httpx.ConnectTimeout("slow connect")), "could not reach"),
        (_raise(httpx.ReadTimeout("slow")), f"within {SERVER_FLEET_TIMEOUT_S:g} s"),
        (_raise(httpx.RemoteProtocolError("garbled")), "RemoteProtocolError"),
        (lambda request: httpx.Response(503, text="down"), "HTTP 503"),
        (lambda request: httpx.Response(200, text="<html>"), "not JSON"),
        (lambda request: _json({"outcome": "ok", "target": "live"}), "not a snapshot"),
        (lambda request: _json(["not", "a", "snapshot"]), "not a snapshot"),
        (lambda request: _json(_snapshot(target="other").to_json()), "'other'"),
    ],
)
async def test_every_failure_is_unavailable_with_a_reason(handler, fragment):
    transport, _ = _answering(handler)
    answer = await fetch_snapshot(FLEET_ON, "live", transport=transport, now=GENERATED_AT)
    assert answer.outcome is FleetOutcome.UNAVAILABLE
    assert answer.target == "live"
    assert answer.snapshot is None
    assert fragment in answer.reason
    # A service answer is never worded as IOCs being down.
    assert "down" not in answer.reason.replace("HTTP 503", "")


async def test_no_url_is_unavailable_without_a_request():
    transport, seen = _answering(lambda request: _json(_snapshot().to_json()))
    answer = await fetch_snapshot({}, "live", transport=transport)
    assert answer.outcome is FleetOutcome.UNAVAILABLE
    assert "not configured" in answer.reason
    assert seen == []


def test_the_timeout_is_short():
    assert 0 < SERVER_FLEET_TIMEOUT_S <= 5


async def test_the_answer_serialises():
    transport, _ = _answering(lambda request: _json(_snapshot().to_json()))
    answer = await fetch_snapshot(FLEET_ON, "live", transport=transport, now=GENERATED_AT)
    document = json.loads(json.dumps(answer.to_json()))
    assert document["outcome"] == "ok"
    assert document["snapshot"] == _snapshot().to_json()
    unavailable = FleetAnswer.unavailable("live", "no answer")
    assert unavailable.to_json() == {
        "outcome": "unavailable",
        "target": "live",
        "reason": "no answer",
        "generated_at": None,
        "snapshot": None,
    }


# --- entry_for ----------------------------------------------------------------------


def _ok_answer() -> FleetAnswer:
    snapshot = _snapshot()
    return FleetAnswer(
        outcome=FleetOutcome.OK, target="live", generated_at=GENERATED_AT, snapshot=snapshot
    )


def test_an_ok_answer_gives_the_servers_row():
    entry = entry_for(_ok_answer(), "ioc-b")
    assert entry.outcome is FleetOutcome.OK
    assert entry.row == _snapshot().row("ioc-b")
    document = json.loads(json.dumps(entry.to_json()))
    assert document["outcome"] == "ok"
    assert document["row"] == _snapshot().to_json()["servers"][1]
    assert document["generated_at"] == GENERATED_AT


def test_a_server_missing_from_the_snapshot_is_not_in_snapshot():
    entry = entry_for(_ok_answer(), "ioc-z")
    assert entry.outcome is FleetOutcome.NOT_IN_SNAPSHOT
    assert entry.row is None
    assert "ioc-z" in entry.reason
    assert entry.to_json()["row"] is None


@pytest.mark.parametrize(
    "outcome",
    [
        FleetOutcome.STALE,
        FleetOutcome.UNAVAILABLE,
        FleetOutcome.NO_SNAPSHOT,
        FleetOutcome.NOT_REFRESHED,
        FleetOutcome.CANNOT_LIST,
    ],
)
def test_a_non_ok_answer_gives_its_outcome_never_a_row(outcome):
    snapshot = _snapshot() if outcome is FleetOutcome.STALE else None
    answer = FleetAnswer(
        outcome=outcome,
        target="live",
        reason="because",
        generated_at=GENERATED_AT if snapshot else None,
        snapshot=snapshot,
    )
    entry = entry_for(answer, "ioc-a")
    assert entry.outcome is outcome
    assert entry.reason == "because"
    assert entry.row is None


def test_an_answer_states_its_shape():
    with pytest.raises(ValueError):
        FleetAnswer(outcome=FleetOutcome.OK, target="live")
    with pytest.raises(ValueError):
        FleetAnswer(outcome=FleetOutcome.UNAVAILABLE, target="live")
    with pytest.raises(ValueError):
        FleetAnswer(outcome=FleetOutcome.NOT_IN_SNAPSHOT, target="live", reason="x")
