"""A queued plan's writes under a write authorization of its own, through a real RunEngine.

The connector is a real :class:`ControlSystemConnector` subclass, so its write
guard is the one every connector carries: it reads ``OSPREY_WRITE_AUTHORIZATION``
on every write and judges the write against the grant on disk, spending a
step. The switch record, the request, the approval, the grant and the ledger
are the real files of :mod:`osprey_connectors.write_authorization`, on a host
with no control-context tree. A thread stands in for the person at the chip:
it approves (or does not) whatever request the run files.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from bluesky import Msg, RunEngine
from bluesky import plan_stubs as bps
from bluesky.utils import FailedStatus
from pydantic import BaseModel, ConfigDict

from osprey.services.bluesky_bridge import qserver_startup, write_grant
from osprey.services.bluesky_bridge.devices.connector import ConnectorSettable
from osprey.services.bluesky_bridge.plan_fields import MovableChannel
from osprey_connectors import session_hold
from osprey_connectors import write_authorization as wa
from osprey_connectors.control_system._base import (
    ChannelWriteResult,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.posture_store import RESERVED_OWNER_KWARG
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from osprey_connectors_tests._harness._control_context_fixtures import FIXTURE_IDENTITY
from osprey_connectors_tests._write_fakes import config_reader

ME = FIXTURE_IDENTITY
TARGET = "va"
CHANNEL = "SR:QF:SETPOINT"
DEVICE = "quad"
SECTION = {"type": "epics", "writes": "allowed"}


class _Connector(ControlSystemConnector):
    """A machine of one channel: what was written, and the last value for a read."""

    def __init__(self) -> None:
        self._control_target = TARGET
        self.writes: list[float] = []
        self.value = 0.0

    async def connect(self, config: dict[str, Any]) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def read_channel(self, channel_address: str, timeout: float | None = None):  # noqa: ARG002 - interface signature
        return SimpleNamespace(value=self.value)

    async def read_multiple_channels(self, channel_addresses, timeout=None):  # noqa: ARG002 - interface signature
        return {address: SimpleNamespace(value=self.value) for address in channel_addresses}

    async def write_channel(self, channel_address, value, timeout=None, confirm=None):  # noqa: ARG002 - interface signature
        self.writes.append(value)
        self.value = value
        return ChannelWriteResult(
            channel_address=channel_address, value_written=value, outcome=WriteOutcome.CONFIRMED
        )

    async def subscribe(self, channel_address, callback, *, on_connection=None):
        raise NotImplementedError

    async def unsubscribe(self, subscription_id):
        raise NotImplementedError

    async def get_metadata(self, channel_address):
        raise NotImplementedError

    async def validate_channel(self, channel_address) -> bool:  # noqa: ARG002 - interface signature
        return True


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quad: MovableChannel = DEVICE
    points: list[float] = [1.0, 2.0, 3.0]


def _scan(devices: Any, params: Any) -> Iterator[Any]:
    """Step the quad through the points."""
    for point in params.points:
        yield from bps.mv(devices[params.quad], point)


def _adaptive(devices: Any, params: Any) -> Iterator[Any]:
    """Step through the points, then go further on what a read says: the survey cannot see that."""
    quad = devices[params.quad]
    yield from _scan(devices, params)
    reading = yield Msg("read", quad)
    if reading is not None:
        last = reading[quad.name]["value"]
        for extra in (last + 1.0, last + 2.0):
            yield from bps.mv(quad, extra)


def _about_the_working_point(devices: Any, params: Any) -> Iterator[Any]:
    """Kick the quad either side of where it sits, then put it back: an ORM sweep's shape."""
    quad = devices[params.quad]
    working_point = float((yield from bps.loc(quad))["setpoint"])
    for kick in (-1.0, 1.0):
        yield from bps.mv(quad, working_point + kick)
    yield from bps.mv(quad, working_point)


def _read_only(devices: Any, params: Any) -> Iterator[Any]:
    yield Msg("read", devices[params.quad])


def _plan_function(plan: Any, device: ConnectorSettable) -> Any:
    spec = SimpleNamespace(name="quad_scan", schema=_Params, plan=plan, description="")
    return qserver_startup._make_plan_function(spec, devices={DEVICE: device})


class _Person(threading.Thread):
    """The person at the chip: approves each request that appears, or none."""

    def __init__(self, ledger: Path, *, approve: bool = True) -> None:
        super().__init__(daemon=True)
        self.ledger = ledger
        self.approve = approve
        self.seen: list[wa.WriteRequest] = []
        self.done = threading.Event()

    def run(self) -> None:
        while not self.done.is_set():
            requests, _ = wa.list_requests(self.ledger / wa.REQUESTS_DIR_NAME)
            for request in requests:
                if request.id in {seen.id for seen in self.seen}:
                    continue
                self.seen.append(request)
                if self.approve:
                    wa.approve_request(request, surface="cli")
            time.sleep(0.02)


@pytest.fixture
def ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This login's host record with the switch at ``asked``, and the worker's ledger directory."""
    root = tmp_path / "root"
    isolate_verdict_environment(monkeypatch, root)
    for name in (wa.AUTHORIZATION_ENV_VAR, "OSPREY_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv(session_hold.HOLD_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr("osprey_connectors.config.get_config_value", config_reader(SECTION))
    write_control_context(root, target=TARGET, switch={TARGET: "asked"})
    directory = tmp_path / "ledger"
    directory.mkdir()
    monkeypatch.setenv(wa.LEDGER_DIR_ENV_VAR, str(directory))
    monkeypatch.setattr(write_grant, "POLL_S", 0.02)
    return directory


@pytest.fixture
def person(ledger: Path) -> Iterator[_Person]:
    person = _Person(ledger)
    person.start()
    yield person
    person.done.set()
    person.join(timeout=2)


def _run(plan: Any, connector: _Connector) -> None:
    device = ConnectorSettable(connector, CHANNEL, name=DEVICE)
    RunEngine(context_managers=[])(_plan_function(plan, device)(**{RESERVED_OWNER_KWARG: ME}))


def _grants() -> list[wa.Grant]:
    directory = wa.authorizations_dir(ME)
    assert directory is not None
    return [wa._parse_grant(path.read_text()) for path in sorted(directory.glob("*.json"))]


def test_a_plan_at_asked_writes_only_once_a_person_approves_what_the_survey_saw(
    person: _Person,
) -> None:
    connector = _Connector()

    _run(_scan, connector)

    assert connector.writes == [1.0, 2.0, 3.0]
    [request] = person.seen
    assert request.steps == 3
    assert request.channels == {CHANNEL: wa.ChannelRange(1.0, 3.0)}
    assert request.owner == ME
    [grant] = _grants()
    assert grant.id == request.id
    assert wa.named_authorization() is None


def test_a_write_the_survey_could_not_see_asks_for_a_replacement_grant(person: _Person) -> None:
    connector = _Connector()

    _run(_adaptive, connector)

    assert connector.writes == [1.0, 2.0, 3.0, 4.0, 5.0]
    first, second, third = person.seen
    # The survey's read is answered for real, before anything moved: 0.0, so it
    # sees the extra writes at 1.0 and 2.0. The run reads 3.0 and goes to 4.0
    # and 5.0, which is why a plan that reads is given headroom.
    assert first.channels == {CHANNEL: wa.ChannelRange(1.0, 3.0)}
    assert first.steps == 5 + write_grant.EXTRA_STEPS_FLOOR
    assert "5 write(s) seen + 10 headroom" in first.label
    # 4.0 is outside the first grant's range: the replacement widens it to 4.0
    # and carries the unspent steps plus headroom again.
    assert second.channels == {CHANNEL: wa.ChannelRange(1.0, 4.0)}
    assert second.steps == (first.steps - 3) + write_grant.EXTRA_STEPS_FLOOR
    assert f"extends {first.id[:8]}" in second.label
    assert "auth_range" in second.label
    # 5.0 is outside that one too: a range widens only to the value in hand,
    # so each new value past it is one more question to the person.
    assert third.channels == {CHANNEL: wa.ChannelRange(1.0, 5.0)}
    assert len(_grants()) == 3


def test_a_sweep_about_the_working_point_is_surveyed_at_the_values_it_writes(
    person: _Person,
) -> None:
    connector = _Connector()
    connector.value = 10.0

    _run(_about_the_working_point, connector)

    assert connector.writes == [9.0, 11.0, 10.0]
    # The survey's locate got the real working point, so one request covers
    # the whole sweep and nobody is asked again.
    [request] = person.seen
    assert request.channels == {CHANNEL: wa.ChannelRange(9.0, 11.0)}
    assert request.steps == 3 + write_grant.EXTRA_STEPS_FLOOR


def test_nobody_answering_ends_the_run_before_any_write(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(write_grant.WAIT_S_ENV, "0.2")
    person = _Person(ledger, approve=False)
    person.start()
    connector = _Connector()
    try:
        with pytest.raises(write_grant.WriteAuthorizationError, match="auth_timeout"):
            _run(_scan, connector)
    finally:
        person.done.set()
        person.join(timeout=2)

    assert connector.writes == []
    assert len(person.seen) == 1
    requests, _ = wa.list_requests(ledger / wa.REQUESTS_DIR_NAME)
    assert requests == ()  # withdrawn
    assert wa.named_authorization() is None


def test_at_allowed_the_run_issues_its_own_grant_and_asks_nobody(
    ledger: Path, tmp_path: Path, person: _Person
) -> None:
    write_control_context(tmp_path / "root", target=TARGET, switch={TARGET: "allowed"})
    connector = _Connector()

    _run(_scan, connector)

    assert connector.writes == [1.0, 2.0, 3.0]
    assert person.seen == []
    [grant] = [
        wa._parse_grant(path.read_text())
        for path in ledger.glob("*.json")
        if not path.name.endswith(".ledger.json")
    ]
    assert grant.approver.surface == wa.SELF_ISSUE_SURFACE
    assert grant.steps == 3


def test_at_off_nobody_is_asked_and_the_connector_refuses_as_before(
    tmp_path: Path, person: _Person
) -> None:
    write_control_context(tmp_path / "root", target=TARGET, switch={})
    connector = _Connector()

    with pytest.raises(FailedStatus, match="switch_off|off"):
        _run(_scan, connector)

    assert connector.writes == []
    assert person.seen == []


def test_a_plan_that_writes_nothing_asks_nobody(person: _Person) -> None:
    connector = _Connector()

    _run(_read_only, connector)

    assert person.seen == []
    assert connector.writes == []


def test_without_a_ledger_directory_the_flow_is_off(
    monkeypatch: pytest.MonkeyPatch, person: _Person
) -> None:
    monkeypatch.delenv(wa.LEDGER_DIR_ENV_VAR)
    connector = _Connector()

    _run(_scan, connector)

    # At asked with no grant the connector judges by the switch alone (#2337).
    assert connector.writes == [1.0, 2.0, 3.0]
    assert person.seen == []


# --- the bridge's view: what the queue summary says while a run waits --------


def test_the_bridge_reports_the_request_its_worker_waits_on(
    ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(wa.REQUESTS_DIR_ENV_VAR, str(ledger / wa.REQUESTS_DIR_NAME))
    monkeypatch.setenv(write_grant.WORKER_HOLDER_ENV, ME)
    assert write_grant.awaiting_approval() is None

    request = wa.write_request(
        TARGET,
        {CHANNEL: (1.0, 3.0)},
        steps=3,
        expires_at_ns=time.time_ns() + 600 * 10**9,
        lease_s=60.0,
        label="bluesky quad_scan",
        owner=ME,
    )

    assert write_grant.awaiting_approval() == {
        "id": request.id,
        "owner": ME,
        "target": TARGET,
        "steps": 3,
        "channels": [CHANNEL],
        "label": "bluesky quad_scan",
        "requested_at": write_grant.awaiting_approval()["requested_at"],
        "expires_at": write_grant.awaiting_approval()["expires_at"],
    }
    # Another holder's request (an optimizer run) is not this queue's.
    monkeypatch.setenv(write_grant.WORKER_HOLDER_ENV, "optimizer")
    assert write_grant.awaiting_approval() is None
    # Withdrawn: the summary goes back to nothing.
    monkeypatch.setenv(write_grant.WORKER_HOLDER_ENV, ME)
    wa.withdraw_request(request.id)
    assert write_grant.awaiting_approval() is None


def test_a_bridge_with_no_requests_directory_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(wa.REQUESTS_DIR_ENV_VAR, raising=False)
    monkeypatch.setenv(write_grant.WORKER_HOLDER_ENV, ME)
    assert write_grant.awaiting_approval() is None
